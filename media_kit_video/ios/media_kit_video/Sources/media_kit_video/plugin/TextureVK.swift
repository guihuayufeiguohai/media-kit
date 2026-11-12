import CoreGraphics
import CoreVideo
import Foundation

#if canImport(Flutter)
  import Flutter
#elseif canImport(FlutterMacOS)
  import FlutterMacOS
#endif

#if SWIFT_PACKAGE
  import Mpv
#endif

// Vulkan-backed FlutterTexture.
//
// Drop-in replacement for TextureHW: same FlutterTexture +
// ResizableTextureProtocol surface, so VideoOutput does not care which one it
// holds. The difference is that mpv renders through MPV_RENDER_API_TYPE_VULKAN
// (gpu-next) into IOSurface-backed CVPixelBuffers instead of an OpenGL FBO,
// which is what makes Dolby Vision Profile 5 / 8 render with the correct
// colours.
//
// Returns nil from init when Vulkan is unusable, so VideoOutput can fall back
// to TextureHW/TextureSW.
public class TextureVK: NSObject, FlutterTexture, ResizableTextureProtocol {
  public typealias UpdateCallback = () -> Void

  private let renderer: MKVulkanRenderer
  private let updateCallback: UpdateCallback

  init?(
    handle: OpaquePointer,
    updateCallback: @escaping UpdateCallback
  ) {
    var error: NSString?
    guard
      let renderer = MKVulkanRenderer(
        mpvHandle: UnsafeMutableRawPointer(handle),
        width: 0,
        height: 0,
        error: &error
      )
    else {
      NSLog("TextureVK: unavailable: \(error ?? "unknown error")")
      return nil
    }

    self.renderer = renderer
    self.updateCallback = updateCallback

    super.init()

    // mpv calls this from its own thread. Hop to main like TextureHW does, so
    // VideoOutput.updateCallback() sees exactly the same threading as the
    // OpenGL path. `weak self` breaks the renderer -> block -> self cycle.
    renderer.onUpdate = { [weak self] in
      DispatchQueue.main.async {
        self?.updateCallback()
      }
    }

    NSLog("TextureVK: \(renderer.backendDescription)")
  }

  deinit {
    // Must run before the mpv handle is destroyed (mpv owns objects on our
    // VkDevice).
    renderer.shutdown()
  }

  public func copyPixelBuffer() -> Unmanaged<CVPixelBuffer>? {
    // -copyPixelBuffer returns +1; ARC releases that reference at the end of
    // this scope, so passRetained hands Flutter its own +1 as the protocol
    // requires.
    guard let pixelBuffer = renderer.copyPixelBuffer() else {
      return nil
    }
    return Unmanaged.passRetained(pixelBuffer)
  }

  public func resize(_ size: CGSize) {
    if size.width == 0 || size.height == 0 {
      return
    }
    NSLog("TextureVK: resize: \(size.width)x\(size.height)")
    renderer.resizeWidth(Int32(size.width), height: Int32(size.height))
  }

  public func render(_ size: CGSize) {
    // VideoOutput calls textureFrameAvailable right after this returns, so we
    // only have to produce the frame here.
    _ = renderer.render()
  }

  public var backendDescription: String {
    renderer.backendDescription
  }
}
