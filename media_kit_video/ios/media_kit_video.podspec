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
      'OTHER_LDFLAGS'                                => '"$(inherited)" -framework Mpv',
      # Flutter.framework does not contain a i386 slice.
      'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',
    }
  else
    s.source_files        = 'media_kit_video/Sources/media_kit_video/stub/**/*.swift'
    s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES' }
  end
end
