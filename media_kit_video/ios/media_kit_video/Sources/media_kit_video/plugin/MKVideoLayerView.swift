import Flutter
import Metal
import QuartzCore
import UIKit

#if os(iOS)

  /// Hosts the `CAMetalLayer` that will receive mpv's frames once C2 turns the
  /// layer sink on.
  ///
  /// C1 only wires the plumbing: the factory is registered, the view compiles
  /// and attaches/detaches correctly, but `VideoOutput` never selects
  /// `.metalLayer`, so no instance is ever created. That is deliberate -- it
  /// keeps "did the new file get compiled into the pod, and does registration
  /// crash?" separable from "does the picture look right?".
  final class MKVideoLayerViewFactory: NSObject, FlutterPlatformViewFactory {
    static let viewType = "media_kit_video/video_layer"

    private let manager: VideoOutputManager

    init(manager: VideoOutputManager) {
      self.manager = manager
      super.init()
    }

    func createArgsCodec() -> FlutterMessageCodec & NSObjectProtocol {
      return FlutterStandardMessageCodec.sharedInstance()
    }

    func create(
      withFrame frame: CGRect,
      viewIdentifier viewId: Int64,
      arguments args: Any?
    ) -> FlutterPlatformView {
      let dict = args as? [String: Any]
      // Only the mpv handle is passed. Sizes come through the existing
      // VideoOutput.Resize channel so there is a single source of truth.
      let handle = Int64(dict?["handle"] as? String ?? "") ?? -1
      return MKVideoLayerView(
        frame: frame,
        handle: handle,
        manager: manager
      )
    }
  }

  final class MKVideoLayerView: UIView, FlutterPlatformView {
    override class var layerClass: AnyClass { CAMetalLayer.self }
    var metalLayer: CAMetalLayer { layer as! CAMetalLayer }

    private let handle: Int64
    private weak var manager: VideoOutputManager?
    private var attached = false

    init(frame: CGRect, handle: Int64, manager: VideoOutputManager) {
      self.handle = handle
      self.manager = manager
      super.init(frame: frame)
      // Never swallow touches: the Flutter gesture detector above must win the
      // arena, otherwise tap-to-show-controls stops working over the video.
      isUserInteractionEnabled = false
      backgroundColor = .clear
      configureLayer()
    }

    required init?(coder: NSCoder) {
      // Never instantiated from a nib: Flutter creates it through the factory.
      fatalError("init(coder:) is not supported")
    }

    func view() -> UIView { self }

    // didMoveToWindow is the only reliable attach/detach signal: Flutter
    // re-parents platform views on every route push/pop and on Hero
    // transitions, and `addSubview` alone does not tell us about removal.
    override func didMoveToWindow() {
      super.didMoveToWindow()
      if window != nil, !attached {
        attached = true
        manager?.attach(handle: handle, view: self)
      } else if window == nil, attached {
        attached = false
        manager?.detach(handle: handle, view: self)
      }
    }

    override func layoutSubviews() {
      super.layoutSubviews()
      let scale = window?.screen.scale ?? UIScreen.main.scale
      metalLayer.contentsScale = scale
      // max(1, ...) matters: a zero-sized drawable makes nextDrawable() return
      // nil forever, which is indistinguishable from "the renderer is dead".
      let size = CGSize(
        width: max(1, bounds.width * scale),
        height: max(1, bounds.height * scale)
      )
      if metalLayer.drawableSize != size {
        metalLayer.drawableSize = size
      }
    }

    private func configureLayer() {
      metalLayer.device = MTLCreateSystemDefaultDevice()
      // SDR defaults only. MetalLayerPresenter overrides pixelFormat,
      // colorspace and wantsExtendedDynamicRangeContent from `attach` when it
      // knows whether EDR actually came up -- this view is created by Flutter's
      // platform view factory and only receives the mpv handle.
      // NOT `.bgra8Unorm_srgb`: the shader writes already-encoded values, and
      // an sRGB drawable would encode them a second time (washed out + wrong
      // contrast). Keep it linear-ish and let the colorspace do the mapping.
      metalLayer.pixelFormat = .bgra8Unorm
      // We never sample the drawable, only render into it.
      metalLayer.framebufferOnly = true
      metalLayer.maximumDrawableCount = 3
      // Return nil instead of blocking the caller for up to a second when the
      // display is asleep or the app is backgrounded.
      metalLayer.allowsNextDrawableTimeout = true
      metalLayer.presentsWithTransaction = false
      metalLayer.isOpaque = true
      metalLayer.colorspace = CGColorSpace(name: CGColorSpace.sRGB)
      if #available(iOS 16.0, *) {
        // Gates 1 and 2 are SDR only. EDR is a later round.
        metalLayer.wantsExtendedDynamicRangeContent = false
      }
    }
  }

#endif
