import CoreGraphics
import Foundation

#if canImport(Flutter)
  import Flutter
#elseif canImport(FlutterMacOS)
  import FlutterMacOS
#endif

#if os(iOS)
  import UIKit
  /// The native view a frame sink can draw into. iOS only; the protocol still
  /// has to compile for macOS, where nothing implements it.
  typealias VideoLayerView = UIView
#else
  typealias VideoLayerView = AnyObject
#endif

/// Which sink a rendered frame is delivered to.
///
/// `texture` is the FlutterTexture path (today's behaviour). `metalLayer` is
/// the PlatformView + CAMetalLayer path, introduced in C2.
public enum VideoOutputMode: String {
  case texture
  case metalLayer
}

/// Where a rendered frame goes.
///
/// `VideoOutput` drives this from its `Worker` thread -- a plain `Thread` plus
/// a `DispatchSemaphore`, NOT a DispatchQueue -- so `present(_:)` must never
/// block and must never hop to the main thread synchronously.
///
/// `activate()` / `resized()` / `attach(_:)` / `shutdown()` are only ever
/// called on the main thread.
protocol VideoFramePresenter: AnyObject {
  var mode: VideoOutputMode { get }

  /// Once, right after the renderer exists. Registers the FlutterTexture
  /// (texture mode) or announces "layer mode, no texture id".
  func activate()

  /// After `texture.resize(size)`. Forwards the new video rect to Dart.
  func resized(_ size: CGSize)

  /// Immediately after `texture.render(size)`.
  func present(_ size: CGSize)

  /// `nil` means "no consumer attached".
  func attach(_ view: VideoLayerView?)

  /// On teardown.
  func shutdown()
}

/// The FlutterTexture sink: everything C1 does today, moved out of
/// `VideoOutput` unchanged.
final class TexturePresenter: VideoFramePresenter {
  let mode: VideoOutputMode = .texture

  private let texture: ResizableTextureProtocol
  private let registry: FlutterTextureRegistry
  private let onResize: VideoOutput.TextureUpdateCallback
  private var textureId: Int64 = -1

  init(
    texture: ResizableTextureProtocol,
    registry: FlutterTextureRegistry,
    onResize: @escaping VideoOutput.TextureUpdateCallback
  ) {
    self.texture = texture
    self.registry = registry
    self.onResize = onResize
  }

  func activate() {
    // Textures must be registered on the platform thread.
    textureId = registry.register(texture)
    // textureUpdateCallback must run on the main thread
    onResize(textureId, CGSize(width: 0, height: 0), .texture)
  }

  func resized(_ size: CGSize) {
    // `sync` matches the pre-C1 behaviour exactly: VideoOutput already blocked
    // here so the widget could lay out before the next present.
    DispatchQueue.main.sync {
      self.onResize(self.textureId, size, .texture)
    }
  }

  func present(_ size: CGSize) {
    // Async on purpose. This runs on Worker's single thread, so a synchronous
    // hop makes the render loop wait for whatever the main thread is doing
    // (Flutter's own compositing, layout, a platform channel...). At 24fps
    // that is usually invisible; the moment one main-thread frame runs long,
    // the render queue backs up behind it and playback stutters for seconds.
    // -textureFrameAvailable only marks the texture dirty, so ordering against
    // the next -render is not required -- Flutter samples whichever slot is
    // current when it gets around to compositing.
    //
    // `weak self` + reading `textureId` at execution time (rather than
    // snapshotting it) is deliberate: shutdown() sets it to -1, and a queued
    // block must not poke a texture id that was already unregistered.
    DispatchQueue.main.async { [weak self] in
      guard let self = self, self.textureId >= 0 else { return }
      // Textures must be marked as available from the main thread
      self.registry.textureFrameAvailable(self.textureId)
    }
  }

  func attach(_ view: VideoLayerView?) {
    // Nothing to attach: Flutter samples the texture directly.
  }

  func shutdown() {
    let registry_ = registry
    let id = textureId
    textureId = -1
    // Also fixes a pre-existing hazard: a VideoOutput torn down before _init
    // finished used to call unregisterTexture(-1).
    guard id >= 0 else { return }
    DispatchQueue.main.async {
      // Textures must be unregistered on the platform thread
      registry_.unregisterTexture(id)
    }
  }
}

// This class creates and manipulates the different types of FlutterTexture,
// handles resizing, rendering calls, and notify Flutter when a new frame is
// available to render.
//
// To improve the user experience, a worker is used to execute heavy tasks on a
// dedicated thread.
public class VideoOutput: NSObject {
  // Will be called on the main thread
  public typealias TextureUpdateCallback = (Int64, CGSize, VideoOutputMode) -> Void

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
  /// Which sink Dart asked for. Native is the final arbiter: iOS 15, a missing
  /// Vulkan renderer or an unsupported Metal device all fall back to texture.
  private let requestedMode: VideoOutputMode
  /// 16F intermediate instead of 8-bit BGRA. Layer sink only.
  private let requestedPixelFormat16F: Bool
  /// Present in HDR instead of tone mapping to SDR. Layer sink only, and it
  /// forces the 16F intermediate on: PQ cannot survive an 8-bit surface.
  private let requestedEDR: Bool
  private var width: Int64?
  private var height: Int64?
  private var texture: ResizableTextureProtocol!
  private var presenter: VideoFramePresenter!
  private var currentSize: CGSize = CGSize.zero
  private var disposed: Bool = false
  #if os(iOS)
    // A PlatformView can be created before _init() has finished building the
    // renderer, so the first attach is parked here and replayed afterwards.
    private var pendingView: VideoLayerView?
  #endif

  /// The sink actually in use. Native is the final arbiter: Dart may ask for
  /// `.metalLayer` and still get `.texture` (iOS 15, Vulkan unavailable, C1).
  public var mode: VideoOutputMode { presenter?.mode ?? .texture }

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
    requestedMode = configuration.outputMode
    requestedPixelFormat16F = configuration.pixelFormat16F
    requestedEDR = configuration.outputEDR
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
    presenter?.shutdown()
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

    #if os(iOS)
      // Decide the sink *before* building the renderer: the intermediate pixel
      // format is baked into the IOSurface ring, and only the layer sink can
      // take 16F (Flutter's external texture path rejects anything but BGRA8).
      var wantsLayer = false
      if #available(iOS 16.0, *), requestedMode == .metalLayer,
        MetalLayerPresenter.isSupported
      {
        wantsLayer = true
      }
      // EDR implies 16F: an 8-bit surface cannot carry PQ without banding so
      // severe that the HDR part stops mattering.
      let halfFloat = wantsLayer && (requestedPixelFormat16F || requestedEDR)

      // Kept out here so the sink selection below can tell "Vulkan is
      // available" apart from "we fell back to TextureSW".
      var vkTexture: TextureVK?
    #endif

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
          halfFloat: halfFloat,
          // Use `weak self` to prevent memory leaks
          updateCallback: { [weak self]() in
            guard let that = self else {
              return
            }
            that.updateCallback()
          }
        ) {
          vkTexture = vulkanTexture
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

    // ---- choose the sink ---------------------------------------------------
    var layerPresenter: VideoFramePresenter?
    #if os(iOS)
      if wantsLayer, vkTexture != nil {
        layerPresenter = MetalLayerPresenter(
          renderer: texture,
          pixelFormat16F: halfFloat,
          outputEDR: requestedEDR,
          onResize: textureUpdateCallback
        )
        if layerPresenter == nil {
          NSLog(
            "VideoOutput: MetalLayerPresenter unavailable, falling back to texture"
          )
        }
      } else if requestedMode == .metalLayer {
        // iOS 15, or no Vulkan renderer: the layer sink is not usable here.
        NSLog(
          "VideoOutput: metalLayer not available here, falling back to texture"
        )
      }
    #endif

    if let layerPresenter = layerPresenter {
      presenter = layerPresenter
      NSLog("VideoOutput: output=metalLayer")
    } else {
      presenter = TexturePresenter(
        texture: texture,
        registry: registry,
        onResize: textureUpdateCallback
      )
      NSLog("VideoOutput: output=texture")
    }

    DispatchQueue.main.sync { [weak self]() in
      guard let that = self else {
        return
      }
      that.presenter.activate()
      #if os(iOS)
        if let view = that.pendingView {
          that.pendingView = nil
          that.presenter.attach(view)
        }
      #endif
    }
  }

  #if os(iOS)
    /// Called on the main thread by the PlatformView when it enters a window.
    /// Internal: `VideoLayerView` is an internal typealias, and only
    /// `VideoOutputManager` (same module) calls this.
    func attachLayerView(_ view: VideoLayerView) {
      if let presenter = presenter {
        presenter.attach(view)
      } else {
        pendingView = view
      }
    }

    /// Called on the main thread when the PlatformView leaves its window.
    func detachLayerView(_ view: VideoLayerView) {
      if pendingView === view {
        pendingView = nil
      }
      presenter?.attach(nil)
    }
  #endif

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
      presenter.resized(size)
    }

    if disposed {
      return
    }

    texture.render(size)
    // The sink is the only thing that differs between texture and layer mode.
    // `render` above always runs, so mpv's "render() not being called" watchdog
    // can never be affected by whether a PlatformView happens to be attached.
    presenter.present(size)
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
