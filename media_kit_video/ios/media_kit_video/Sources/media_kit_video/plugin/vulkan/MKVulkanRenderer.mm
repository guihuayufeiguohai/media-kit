#import "MKVulkanRenderer.h"

#include <mpv/client.h>
#include <mpv/render_vk.h>
#include <vulkan/vulkan.h>

#import <Metal/Metal.h>

#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

constexpr int kRingSize = 3;
// Matches kCVPixelFormatType_32BGRA, what Flutter's Metal compositor expects.
constexpr VkFormat kFormat = VK_FORMAT_B8G8R8A8_UNORM;
constexpr VkImageUsageFlags kUsage =
    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

struct Slot {
  CVPixelBufferRef pb = nullptr;
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  int width = 0;
  int height = 0;
};

bool HasDeviceExtension(VkPhysicalDevice pd, const char *name) {
  uint32_t n = 0;
  vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, nullptr);
  std::vector<VkExtensionProperties> all(n);
  vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, all.data());
  for (const auto &e : all) {
    if (std::strcmp(e.extensionName, name) == 0) return true;
  }
  return false;
}

bool HasInstanceExtension(const char *name) {
  uint32_t n = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr);
  std::vector<VkExtensionProperties> all(n);
  vkEnumerateInstanceExtensionProperties(nullptr, &n, all.data());
  for (const auto &e : all) {
    if (std::strcmp(e.extensionName, name) == 0) return true;
  }
  return false;
}

void SetError(NSString **out, NSString *message) {
  if (out) *out = message;
}

}  // namespace

@implementation MKVulkanRenderer {
  mpv_handle *_mpv;
  mpv_render_context *_rc;

  VkInstance _instance;
  VkPhysicalDevice _pd;
  VkDevice _dev;
  VkQueue _queue;
  uint32_t _qf;
  // One timeline semaphore shared by all slots: every render signals value
  // _timelineValue, and we wait for it before handing the frame to Flutter.
  VkSemaphore _timeline;
  uint64_t _timelineValue;

  std::vector<Slot> _slots;
  int _writeSlot;
  int _readySlot;
  int _width;
  int _height;
  bool _deviceLost;
  NSString *_backend;

  std::mutex _mutex;
}

#pragma mark - Init

- (nullable instancetype)initWithMpvHandle:(void *)handle
                                     width:(int)width
                                    height:(int)height
                                     error:(NSString *_Nullable *_Nullable)error {
  self = [super init];
  if (!self) return nil;

  _mpv = (mpv_handle *)handle;
  _instance = VK_NULL_HANDLE;
  _dev = VK_NULL_HANDLE;
  _timeline = VK_NULL_HANDLE;
  _timelineValue = 0;
  _writeSlot = 0;
  _readySlot = -1;
  _width = width;
  _height = height;
  _deviceLost = false;

  // MoltenVK aborts the process (instead of failing) when there is no Metal
  // device, e.g. in a GPU-less VM.
  if (MTLCreateSystemDefaultDevice() == nil) {
    SetError(error, @"no Metal device");
    return nil;
  }

  if (![self createInstance:error]) return nil;
  if (![self pickDevice:error]) return nil;
  if (![self createDevice:error]) return nil;
  if (![self createRenderContext:error]) return nil;
  if (![self rebuildSlots:width height:height error:error]) return nil;

  return self;
}

- (BOOL)createInstance:(NSString **)error {
  VkApplicationInfo app = {};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.pApplicationName = "media_kit";
  app.apiVersion = VK_API_VERSION_1_3;

  std::vector<const char *> exts;
  // MoltenVK is a portability implementation.
  if (HasInstanceExtension(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
    exts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
  }

  VkInstanceCreateInfo ci = {};
  ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ci.pApplicationInfo = &app;
  ci.enabledExtensionCount = (uint32_t)exts.size();
  ci.ppEnabledExtensionNames = exts.data();
  if (!exts.empty()) {
    ci.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
  }

  VkResult r = vkCreateInstance(&ci, nullptr, &_instance);
  if (r != VK_SUCCESS) {
    SetError(error, [NSString stringWithFormat:@"vkCreateInstance failed (%d)", r]);
    return NO;
  }
  return YES;
}

- (BOOL)pickDevice:(NSString **)error {
  uint32_t n = 0;
  vkEnumeratePhysicalDevices(_instance, &n, nullptr);
  if (n == 0) {
    SetError(error, @"no Vulkan physical device");
    return NO;
  }
  std::vector<VkPhysicalDevice> all(n);
  vkEnumeratePhysicalDevices(_instance, &n, all.data());
  _pd = all[0];

  VkPhysicalDeviceProperties props = {};
  vkGetPhysicalDeviceProperties(_pd, &props);
  if (props.apiVersion < VK_API_VERSION_1_3) {
    SetError(error, @"Vulkan 1.3 is required");
    return NO;
  }

  uint32_t nqf = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(_pd, &nqf, nullptr);
  std::vector<VkQueueFamilyProperties> qf(nqf);
  vkGetPhysicalDeviceQueueFamilyProperties(_pd, &nqf, qf.data());
  for (uint32_t i = 0; i < nqf; i++) {
    if (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      _qf = i;
      break;
    }
  }
  if (_qf == UINT32_MAX) {
    SetError(error, @"no graphics queue family");
    return NO;
  }
  return YES;
}

- (BOOL)createDevice:(NSString **)error {
  // libplacebo needs timeline semaphores and benefits from synchronization2.
  VkPhysicalDeviceVulkan13Features f13 = {};
  f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
  f13.synchronization2 = VK_TRUE;
  VkPhysicalDeviceVulkan12Features f12 = {};
  f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  f12.pNext = &f13;
  f12.timelineSemaphore = VK_TRUE;
  f12.hostQueryReset = VK_TRUE;
  VkPhysicalDeviceFeatures2 feats = {};
  feats.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  feats.pNext = &f12;

  std::vector<const char *> exts;
  if (HasDeviceExtension(_pd, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME)) {
    exts.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
  }
  if (!HasDeviceExtension(_pd, VK_EXT_METAL_OBJECTS_EXTENSION_NAME)) {
    // Without it the IOSurface cannot back the VkImage, i.e. no zero-copy.
    SetError(error, @"VK_EXT_metal_objects is not available");
    return NO;
  }
  exts.push_back(VK_EXT_METAL_OBJECTS_EXTENSION_NAME);

  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci = {};
  qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  qci.queueFamilyIndex = _qf;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;

  VkDeviceCreateInfo dci = {};
  dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dci.pNext = &feats;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.enabledExtensionCount = (uint32_t)exts.size();
  dci.ppEnabledExtensionNames = exts.data();

  VkResult r = vkCreateDevice(_pd, &dci, nullptr, &_dev);
  if (r != VK_SUCCESS) {
    SetError(error, [NSString stringWithFormat:@"vkCreateDevice failed (%d)", r]);
    return NO;
  }
  vkGetDeviceQueue(_dev, _qf, 0, &_queue);

  VkSemaphoreTypeCreateInfo stc = {};
  stc.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  stc.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  stc.initialValue = 0;
  VkSemaphoreCreateInfo sci = {};
  sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  sci.pNext = &stc;
  r = vkCreateSemaphore(_dev, &sci, nullptr, &_timeline);
  if (r != VK_SUCCESS) {
    SetError(error, [NSString stringWithFormat:@"vkCreateSemaphore failed (%d)", r]);
    return NO;
  }
  return YES;
}

- (BOOL)createRenderContext:(NSString **)error {
  mpv_vulkan_init_params init = {};
  init.instance = _instance;
  init.get_instance_proc_addr = vkGetInstanceProcAddr;
  init.api_version = VK_API_VERSION_1_3;
  init.physical_device = _pd;
  init.device = _dev;
  // Must match what was enabled above; mpv forwards this list to FFmpeg.
  static const char *kExtensions[] = {
      VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME,
      VK_EXT_METAL_OBJECTS_EXTENSION_NAME,
  };
  init.extensions = kExtensions;
  init.num_extensions = 2;
  init.features = nullptr;
  init.queue_graphics = {_qf, 1};
  init.queue_compute = {_qf, 1};
  init.queue_transfer = {_qf, 1};

  // gpu-next is libplacebo's renderer and the one that handles Dolby Vision
  // IPT-PQ-C2 correctly; "gpu" is the legacy gl_video renderer as a fallback.
  const char *renderers[] = {"gpu-next", "gpu"};
  int err = 0;
  for (const char *renderer : renderers) {
    int advanced = 1;
    mpv_render_param params[] = {
        {MPV_RENDER_PARAM_API_TYPE,
         const_cast<char *>(MPV_RENDER_API_TYPE_VULKAN)},
        {MPV_RENDER_PARAM_VULKAN_INIT_PARAMS, &init},
        {MPV_RENDER_PARAM_VULKAN_RENDERER, const_cast<char *>(renderer)},
        {MPV_RENDER_PARAM_ADVANCED_CONTROL, &advanced},
        {MPV_RENDER_PARAM_INVALID, nullptr},
    };
    err = mpv_render_context_create(&_rc, _mpv, params);
    if (err >= 0) {
      const char *active = nullptr;
      int info = mpv_render_context_get_info(
          _rc, {MPV_RENDER_PARAM_VULKAN_RENDERER, &active});
      NSString *name = (info >= 0 && active) ? @(active) : @"unknown renderer";
      VkPhysicalDeviceProperties props = {};
      vkGetPhysicalDeviceProperties(_pd, &props);
      _backend = [NSString
          stringWithFormat:@"vulkan (%@) on %s", name, props.deviceName];
      return YES;
    }
    _rc = nullptr;
  }
  SetError(error, [NSString stringWithFormat:@"mpv_render_context_create failed: %s",
                                             mpv_error_string(err)]);
  return NO;
}

#pragma mark - Slots

- (BOOL)rebuildSlots:(int)width height:(int)height error:(NSString **)error {
  [self destroySlots];
  if (width <= 0 || height <= 0) return YES;

  _slots.resize(kRingSize);
  for (int i = 0; i < kRingSize; i++) {
    if (![self createSlot:&_slots[i] width:width height:height error:error]) {
      [self destroySlots];
      return NO;
    }
  }
  _width = width;
  _height = height;
  _writeSlot = 0;
  _readySlot = -1;
  return YES;
}

- (BOOL)createSlot:(Slot *)slot width:(int)width height:(int)height error:(NSString **)error {
  NSDictionary *attrs = @{
    (id)kCVPixelBufferIOSurfacePropertiesKey : @{},
    (id)kCVPixelBufferMetalCompatibilityKey : @YES,
  };
  CVPixelBufferRef pb = nullptr;
  if (CVPixelBufferCreate(kCFAllocatorDefault, (size_t)width, (size_t)height,
                          kCVPixelFormatType_32BGRA,
                          (__bridge CFDictionaryRef)attrs, &pb) != kCVReturnSuccess) {
    SetError(error, @"CVPixelBufferCreate failed");
    return NO;
  }
  IOSurfaceRef surface = CVPixelBufferGetIOSurface(pb);
  if (!surface) {
    CVPixelBufferRelease(pb);
    SetError(error, @"CVPixelBuffer is not IOSurface backed");
    return NO;
  }

  // MoltenVK backs this image's MTLTexture with the CVPixelBuffer's IOSurface.
  VkImportMetalIOSurfaceInfoEXT import = {};
  import.sType = VK_STRUCTURE_TYPE_IMPORT_METAL_IO_SURFACE_INFO_EXT;
  import.ioSurface = surface;

  VkImageCreateInfo ici = {};
  ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  ici.pNext = &import;
  ici.imageType = VK_IMAGE_TYPE_2D;
  ici.format = kFormat;
  ici.extent = {(uint32_t)width, (uint32_t)height, 1};
  ici.mipLevels = 1;
  ici.arrayLayers = 1;
  ici.samples = VK_SAMPLE_COUNT_1_BIT;
  ici.tiling = VK_IMAGE_TILING_OPTIMAL;
  ici.usage = kUsage;
  ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  VkResult r = vkCreateImage(_dev, &ici, nullptr, &slot->image);
  if (r != VK_SUCCESS) {
    CVPixelBufferRelease(pb);
    SetError(error, [NSString stringWithFormat:@"vkCreateImage failed (%d)", r]);
    return NO;
  }

  VkMemoryRequirements req = {};
  vkGetImageMemoryRequirements(_dev, slot->image, &req);
  VkPhysicalDeviceMemoryProperties mp = {};
  vkGetPhysicalDeviceMemoryProperties(_pd, &mp);
  uint32_t type = UINT32_MAX;
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
    if ((req.memoryTypeBits & (1u << i)) &&
        (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
      type = i;
      break;
    }
  }
  if (type == UINT32_MAX) {
    vkDestroyImage(_dev, slot->image, nullptr);
    slot->image = VK_NULL_HANDLE;
    CVPixelBufferRelease(pb);
    SetError(error, @"no device-local memory type for the imported image");
    return NO;
  }

  VkMemoryDedicatedAllocateInfo dedicated = {};
  dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
  dedicated.image = slot->image;
  VkMemoryAllocateInfo mai = {};
  mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  mai.pNext = &dedicated;
  mai.allocationSize = req.size;
  mai.memoryTypeIndex = type;
  r = vkAllocateMemory(_dev, &mai, nullptr, &slot->memory);
  if (r != VK_SUCCESS) {
    vkDestroyImage(_dev, slot->image, nullptr);
    slot->image = VK_NULL_HANDLE;
    CVPixelBufferRelease(pb);
    SetError(error, [NSString stringWithFormat:@"vkAllocateMemory failed (%d)", r]);
    return NO;
  }
  r = vkBindImageMemory(_dev, slot->image, slot->memory, 0);
  if (r != VK_SUCCESS) {
    vkFreeMemory(_dev, slot->memory, nullptr);
    vkDestroyImage(_dev, slot->image, nullptr);
    slot->image = VK_NULL_HANDLE;
    slot->memory = VK_NULL_HANDLE;
    CVPixelBufferRelease(pb);
    SetError(error, [NSString stringWithFormat:@"vkBindImageMemory failed (%d)", r]);
    return NO;
  }

  slot->pb = pb;
  slot->width = width;
  slot->height = height;
  return YES;
}

- (void)destroySlots {
  if (_dev) {
    for (auto &slot : _slots) {
      if (slot.image) vkDestroyImage(_dev, slot.image, nullptr);
      if (slot.memory) vkFreeMemory(_dev, slot.memory, nullptr);
      // Flutter may still hold references from copyPixelBuffer; the IOSurface
      // lives until the last one is released.
      if (slot.pb) CVPixelBufferRelease(slot.pb);
      slot = Slot{};
    }
  }
  _slots.clear();
  _readySlot = -1;
}

#pragma mark - Public

- (NSString *)backendDescription {
  return _backend ?: @"vulkan (unavailable)";
}

- (void)resizeWidth:(int)width height:(int)height {
  std::lock_guard<std::mutex> lock(_mutex);
  if (width <= 0 || height <= 0) return;
  if (width == _width && height == _height && !_slots.empty()) return;
  NSString *error = nil;
  if (![self rebuildSlots:width height:height error:&error]) {
    NSLog(@"TextureVK: resize %dx%d failed: %@", width, height, error);
  }
}

- (BOOL)render {
  std::lock_guard<std::mutex> lock(_mutex);
  if (!_rc || _slots.empty() || _deviceLost) return NO;

  uint64_t flags = mpv_render_context_update(_rc);
  if (!(flags & MPV_RENDER_UPDATE_FRAME)) return NO;

  int slotIndex = _writeSlot;
  Slot &slot = _slots[slotIndex];

  _timelineValue += 1;
  mpv_vulkan_image target = {};
  target.image = slot.image;
  target.format = kFormat;
  target.usage = kUsage;
  target.width = slot.width;
  target.height = slot.height;
  // Shared with Metal: ownership is transferred to/from the external queue
  // family, and the image stays in GENERAL between frames.
  target.layout = VK_IMAGE_LAYOUT_GENERAL;
  target.final_layout = VK_IMAGE_LAYOUT_GENERAL;
  target.queue_family = VK_QUEUE_FAMILY_EXTERNAL;
  target.signal_semaphore = _timeline;
  target.signal_value = _timelineValue;

  int flip = 0;
  mpv_render_param params[] = {
      {MPV_RENDER_PARAM_VULKAN_IMAGE, &target},
      {MPV_RENDER_PARAM_FLIP_Y, &flip},
      {MPV_RENDER_PARAM_INVALID, nullptr},
  };
  int err = mpv_render_context_render(_rc, params);
  if (err < 0) {
    NSLog(@"TextureVK: mpv_render_context_render failed: %s", mpv_error_string(err));
    mpv_render_context_report_swap(_rc);
    return NO;
  }

  // Flutter samples the CVPixelBuffer without any GPU synchronisation, so the
  // frame must be complete before we hand it over. Waiting here also bounds
  // the ring: a slot is never reused while mpv's GPU work on it is pending.
  uint64_t waitValue = _timelineValue;
  VkSemaphoreWaitInfo wi = {};
  wi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
  wi.semaphoreCount = 1;
  wi.pSemaphores = &_timeline;
  wi.pValues = &waitValue;
  VkResult r = vkWaitSemaphores(_dev, &wi, 1000ull * 1000ull * 1000ull);
  if (r != VK_SUCCESS) {
    if (r == VK_ERROR_DEVICE_LOST) _deviceLost = true;
    NSLog(@"TextureVK: vkWaitSemaphores returned %d", r);
    mpv_render_context_report_swap(_rc);
    return NO;
  }

  _readySlot = slotIndex;
  _writeSlot = (slotIndex + 1) % (int)_slots.size();
  mpv_render_context_report_swap(_rc);
  return YES;
}

- (nullable CVPixelBufferRef)copyPixelBuffer CF_RETURNS_RETAINED {
  std::lock_guard<std::mutex> lock(_mutex);
  if (_readySlot < 0 || _readySlot >= (int)_slots.size()) return nullptr;
  CVPixelBufferRef pb = _slots[_readySlot].pb;
  if (!pb) return nullptr;
  return CVPixelBufferRetain(pb);
}

- (void)shutdown {
  std::lock_guard<std::mutex> lock(_mutex);
  // Must precede device destruction: mpv owns objects on this VkDevice.
  if (_rc) {
    mpv_render_context_set_update_callback(_rc, nullptr, nullptr);
    mpv_render_context_free(_rc);
    _rc = nullptr;
  }
  [self destroySlots];
  if (_timeline) {
    vkDestroySemaphore(_dev, _timeline, nullptr);
    _timeline = VK_NULL_HANDLE;
  }
  if (_dev) {
    vkDestroyDevice(_dev, nullptr);
    _dev = VK_NULL_HANDLE;
  }
  if (_instance) {
    vkDestroyInstance(_instance, nullptr);
    _instance = VK_NULL_HANDLE;
  }
}

- (void)dealloc {
  [self shutdown];
}

@end
