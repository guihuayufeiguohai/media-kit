// ObjC++ bridge between mpv's Vulkan render API (MPV_RENDER_API_TYPE_VULKAN,
// the mpv_kit extension shipped in Mpv.framework) and Flutter's texture
// registry.
//
// Why ObjC++ and not Swift: creating the device needs a
// VkPhysicalDeviceFeatures2 chain and VkImportMetalIOSurfaceInfoEXT, and
// mpv_vulkan_init_params/mpv_vulkan_image are C structs with pNext chains.
// Neither is expressible in Swift without a bridging layer.
//
// Zero-copy path: every frame is an IOSurface-backed CVPixelBuffer that
// MoltenVK binds as the VkImage's storage (VK_EXT_metal_objects). Flutter's
// Metal compositor samples the very same CVPixelBuffer through
// CVMetalTextureCache, so no pixel data is ever copied.

#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface MKVulkanRenderer : NSObject

/// Returns nil when Vulkan is unavailable (no Metal device, MoltenVK init
/// failure, mpv built without the Vulkan render API, ...). `error` then
/// carries a human readable reason.
- (nullable instancetype)initWithMpvHandle:(void *)handle
                                     width:(int)width
                                    height:(int)height
                                     error:(NSString *_Nullable *_Nullable)error;

/// e.g. "vulkan (gpu-next) on Apple M1" or "vulkan (unknown renderer)".
@property(nonatomic, readonly, copy) NSString *backendDescription;

/// Invoked on mpv's update thread whenever a new frame is ready to be
/// rendered. Without this the render loop never starts: media_kit drives
/// everything (resize + render + textureFrameAvailable) from this callback,
/// exactly like TextureHW does through
/// mpv_render_context_set_update_callback.
@property(nonatomic, copy, nullable) void (^onUpdate)(void);

/// Recreates the image ring. Cheap when the size is unchanged.
- (void)resizeWidth:(int)width height:(int)height;

/// Renders one frame. Returns YES when a new frame is ready for Flutter.
- (BOOL)render;

/// Latest completed frame, +1 retained (Flutter consumes the reference).
/// NULL when no frame is ready yet.
- (nullable CVPixelBufferRef)copyPixelBuffer CF_RETURNS_RETAINED;

/// Frees the mpv render context. MUST run before the mpv handle is destroyed.
- (void)shutdown;

@end

NS_ASSUME_NONNULL_END
