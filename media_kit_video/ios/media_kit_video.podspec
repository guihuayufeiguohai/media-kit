#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint media_kit_video.podspec` to validate before publishing.
#

require_relative '../common/darwin/Podspec/media_kit_utils.rb'

Pod::Spec.new do |s|
  # Setup required files
  system("make -C ../common/darwin HEADERS_DESTDIR=\"$(pwd)/Headers\"")

  # Initialize `MediaKitUtils`
  mku = MediaKitUtils.new(MediaKitUtils::Platform::IOS)

  s.name             = 'media_kit_video'
  s.version          = '0.0.1'
  s.summary          = 'Native implementation for video playback in package:media_kit'
  s.description      = <<-DESC
  Native implementation for video playback in package:media_kit.
                       DESC
  s.homepage         = 'https://github.com/media-kit/media-kit.git'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Hitesh Kumar Saini' => 'saini123hitesh@gmail.com' }

  s.source           = { :path => '.' }
  # MoltenVK 1.4.2 (linked into Mpv.framework) declares
  # MinimumOSVersion 15.0; linking it against an older target produces a
  # deployment-target mismatch and fails to load on iOS 12-14.
  s.platform         = :ios, '15.0'
  s.swift_version    = '5.0'
  s.dependency         'Flutter'
  s.resource_bundles = {
    'media_kit_video_privacy' => ['media_kit_video/Sources/media_kit_video/PrivacyInfo.xcprivacy']
  }
  
  if mku.libs_found
    # Define paths to frameworks dir
    framework_search_paths_iphoneos        = sprintf('$(PROJECT_DIR)/../.symlinks/plugins/%s/ios/Frameworks/.symlinks/mpv/ios', mku.libs_package)
    framework_search_paths_iphonesimulator = sprintf('$(PROJECT_DIR)/../.symlinks/plugins/%s/ios/Frameworks/.symlinks/mpv/ios-simulator', mku.libs_package)

    # .h/.mm: the Vulkan renderer is ObjC++ (MKVulkanRenderer), because
    # mpv_vulkan_init_params and VkImportMetalIOSurfaceInfoEXT cannot be
    # expressed in Swift.
    s.source_files        = 'media_kit_video/Sources/media_kit_video/plugin/**/*.{swift,h,mm}', 'Headers/**/*.h'
    # render_vk.h does `#include <vulkan/vulkan.h>` with an explicit
    # subdirectory. CocoaPods FLATTENS public headers into
    # media_kit_video.framework/Headers (the generated umbrella reads
    # `#import "render_vk.h"`, `#import "client.h"`, ...), so the `vulkan/`
    # level disappears and that include can never resolve in any target that
    # builds the module -- which the host app does, via
    # GeneratedPluginRegistrant.m doing `@import media_kit_video`.
    #
    # Nothing outside this pod needs the Vulkan render API: it is used only by
    # MKVulkanRenderer.mm, which finds it through $(PODS_TARGET_SRCROOT)/Headers
    # where vulkan/ and vk_video/ still exist as real siblings. Keep it out of
    # the umbrella.
    s.private_header_files = 'Headers/mpv/render_vk.h'
    # Metal + CoreVideo + IOSurface back the zero-copy IOSurface path;
    # QuartzCore is needed for CAMetalLayer/CATransaction-free presentation.
    s.frameworks          = 'Metal', 'CoreVideo', 'IOSurface', 'QuartzCore'
    s.pod_target_xcconfig = {
      'DEFINES_MODULE'                               => 'YES',
      'CLANG_CXX_LANGUAGE_STANDARD'                  => 'c++17',
      'GCC_WARN_INHIBIT_ALL_WARNINGS'                => 'YES',
      # VK_USE_PLATFORM_METAL_EXT gates VkImportMetalIOSurfaceInfoEXT in
      # vulkan_metal.h.
      'GCC_PREPROCESSOR_DEFINITIONS'                 => '"$(inherited)" GL_SILENCE_DEPRECATION COREVIDEO_SILENCE_GL_DEPRECATION VK_USE_PLATFORM_METAL_EXT=1',
      'FRAMEWORK_SEARCH_PATHS[sdk=iphoneos*]'        => sprintf('"$(inherited)" "%s"', framework_search_paths_iphoneos),
      'FRAMEWORK_SEARCH_PATHS[sdk=iphonesimulator*]' => sprintf('"$(inherited)" "%s"', framework_search_paths_iphonesimulator),
      # render_vk.h does `#include <vulkan/vulkan.h>` with angle brackets, and
      # clang does NOT consult FRAMEWORK_SEARCH_PATHS for that form: it would
      # look for `vulkan.framework/Headers/vulkan.h`, which does not exist.
      # The bundle ships the headers under Mpv.framework/Headers/vulkan/, so
      # that directory has to be an -I path. The pod's own Headers dir is
      # listed as well so <mpv/render_vk.h> resolves without depending on
      # CocoaPods' public-header layout.
      'HEADER_SEARCH_PATHS[sdk=iphoneos*]'        => sprintf('"$(inherited)" "$(PODS_TARGET_SRCROOT)/Headers" "%s/Mpv.framework/Headers"', framework_search_paths_iphoneos),
      'HEADER_SEARCH_PATHS[sdk=iphonesimulator*]' => sprintf('"$(inherited)" "$(PODS_TARGET_SRCROOT)/Headers" "%s/Mpv.framework/Headers"', framework_search_paths_iphonesimulator),
      'OTHER_LDFLAGS'                                => '"$(inherited)" -framework Mpv',
      # Flutter.framework does not contain a i386 slice.
      'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',
    }
  else
    s.source_files        = 'media_kit_video/Sources/media_kit_video/stub/**/*.swift'
    s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES' }
  end
end
