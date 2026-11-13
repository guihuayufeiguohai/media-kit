import CoreGraphics
import Foundation

#if canImport(Flutter)
  import Flutter
#elseif canImport(FlutterMacOS)
  import FlutterMacOS
#endif

// This class creates and manipulates the different types of FlutterTexture,
// handles resizing, rendering calls, and notify Flutter when a new frame is
// available to render.
//
// To improve the user experience, a worker is used to execute heavy tasks on a
// dedicated thread.
public class VideoOutput: NSObject {
  // Will be called on the main thread
  public typealias TextureUpdateCallback = (Int64, CGSize) -> Void

  private static let isSimulator: Bool = {
    let isSim: Bool
    #if targetEnvironment(simulator)
      isSim = true
    #else
      isSim = false
    #endif
    return isSim
  }()

  private let handle: OpaquePointer
  private let enableHardwareAcceleration: Bool
  private let registry: FlutterTextureRegistry
  private let textureUpdateCallback: TextureUpdateCallback
  private let worker: Worker = .init()
  private var width: Int64?
  private var height: Int64?
  private var texture: ResizableTextureProtocol!
  private var textureId: Int64 = -1
  private var currentSize: CGSize = CGSize.zero
  private var disposed: Bool = false

  init(
    handle: Int64,
    configuration: VideoOutputConfiguration,
    registry: FlutterTextureRegistry,
    textureUpdateCallback: @escaping TextureUpdateCallback
  ) {
    let handle = OpaquePointer(bitPattern: Int(handle))
    assert(handle != nil, "handle casting")

    self.handle = handle!
    width = configuration.width
    height = configuration.height
    enableHardwareAcceleration = configuration.enableHardwareAcceleration
    self.registry = registry
    self.textureUpdateCallback = textureUpdateCallback

    super.init()

    worker.enqueue {
      self._init()
    }
  }

  deinit {
    worker.cancel()

    disposed = true
    disposeTextureId()
  }

  public func setSize(width: Int64?, height: Int64?) {
    worker.enqueue {
      self.width = width
      self.height = height
    }
  }

  private func _init() {
    let enableHardwareAcceleration =
      VideoOutput.isSimulator ? false : enableHardwareAcceleration

    NSLog(
      "VideoOutput: enableHardwareAcceleration: \(enableHardwareAcceleration)"
    )

    if VideoOutput.isSimulator {
      NSLog(
        "VideoOutput: warning: hardware rendering is disabled in the iOS simulator, due to an incompatibility with OpenGL ES"
      )
    }

    if enableHardwareAcceleration {
      // Vulkan first: gpu-next is the only renderer that handles Dolby Vision
      // (IPT-PQ-C2) correctly. TextureVK returns nil when MoltenVK or mpv's
      // Vulkan render API is unavailable.
      //
      // The #if has to wrap the whole block: TextureVK only exists in the iOS
      // target, so merely naming the type outside the guard would break the
      // macOS build.
      var createdTexture = false
      #if os(iOS)
        if let vulkanTexture = TextureVK(
          handle: handle,
          // Use `weak self` to prevent memory leaks
          updateCallback: { [weak self]() in
            guard let that = self else {
              return
            }
            that.updateCallback()
          }
        ) {
          texture = SafeResizableTexture(vulkanTexture)
          createdTexture = true
        }

        if !createdTexture {
          // Do NOT fall back to TextureHW here. This libmpv is built with
          // -Dgl=disabled, so mpv_render_context_create(OPENGL) always fails;
          // TextureHW then keeps a nil render context and the first
          // mpv_render_context_render() dereferences it -- that is the crash
          // seen when opening the player. Software is the only renderer this
          // build still contains.
          NSLog("VideoOutput: Vulkan unavailable, falling back to software")
          texture = SafeResizableTexture(
            TextureSW(
              handle: handle,
              // Use `weak self` to prevent memory leaks
              updateCallback: { [weak self]() in
                guard let that = self else {
                  return
                }
                that.updateCallback()
              }
            )
          )
          createdTexture = true
        }
      #endif

      if !createdTexture {
        texture = SafeResizableTexture(
          TextureHW(
            handle: handle,
            // Use `weak self` to prevent memory leaks
            updateCallback: { [weak self]() in
              guard let that = self else {
                return
              }
              that.updateCallback()
            }
          )
        )
      }
    } else {
      texture = SafeResizableTexture(
        TextureSW(
          handle: handle,
          // Use `weak self` to prevent memory leaks
          updateCallback: { [weak self]() in
            guard let that = self else {
              return
            }
            that.updateCallback()
          }
        )
      )
    }

    DispatchQueue.main.sync { [weak self]() in
      guard let that = self else {
        return
      }
      that.registerTextureId()
    }
  }

  // Must be run on the main thread
  private func registerTextureId() {
    // Textures must be registered on the platform thread.
    textureId = registry.register(texture)
    // textureUpdateCallback must run on the main thread
    textureUpdateCallback(textureId, CGSize(width: 0, height: 0))
  }

  private func disposeTextureId() {
    let registry_ = self.registry
    let textureId_ = self.textureId
    textureId = -1
    DispatchQueue.main.async {
      // Textures must be unregistered on the platform thread
      registry_.unregisterTexture(textureId_)
    }
  }

  public func updateCallback() {
    worker.enqueue {
      self._updateCallback()
    }
  }

  private func _updateCallback() {
    let size = videoSize

    if size.width == 0 || size.height == 0 {
      return
    }

    if currentSize != size {
      currentSize = size

      texture.resize(size)
      DispatchQueue.main.sync { [weak self] in
        guard let that = self else { return }
        // textureUpdateCallback must run on the main thread
        that.textureUpdateCallback(that.textureId, size)
      }
    }

    if disposed {
      return
    }

    texture.render(size)
    // Async on purpose. This runs on Worker's single thread, so a synchronous
    // hop makes the render loop wait for whatever the main thread is doing
    // (Flutter's own compositing, layout, a platform channel...). At 24fps
    // that is usually invisible; the moment one main-thread frame runs long,
    // the render queue backs up behind it and playback stutters for seconds.
    // -textureFrameAvailable only marks the texture dirty, so ordering against
    // the next -render is not required -- Flutter samples whichever slot is
    // current when it gets around to compositing.
    DispatchQueue.main.async { [weak self] in
      guard let that = self else { return }
      // Textures must be marked as available from the main thread
      that.registry.textureFrameAvailable(that.textureId)
    }
  }

    private var videoSize: CGSize {
        // fixed size
        if width != nil && height != nil {
            return CGSize(
                width: Double(width!),
                height: Double(height!)
            )
        }
        
        let params = MPVHelpers.getVideoOutParams(handle)
        return CGSize(
            width: Double(width ?? (params.rotate == 0 || params.rotate == 180
                                    ? params.dw
                                    : params.dh)),
            height: Double(height ?? (params.rotate == 0 || params.rotate == 180
                                      ? params.dh
                                      : params.dw))
        )
  }
}
