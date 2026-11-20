import CoreVideo
import Metal
import QuartzCore
import UIKit

#if os(iOS)

  /// Draws mpv's frames into a `CAMetalLayer`.
  ///
  /// Frame flow: mpv still renders into the IOSurface ring (MKVulkanRenderer is
  /// untouched) -> we take the ready slot's CVPixelBuffer -> wrap it as an
  /// MTLTexture through CVMetalTextureCache -> one full-screen triangle into the
  /// layer's drawable.
  ///
  /// Everything here runs on VideoOutput's Worker thread (a plain Thread plus a
  /// DispatchSemaphore, NOT a DispatchQueue) except `attach` and `shutdown`,
  /// which run on the main thread.
  final class MetalLayerPresenter: NSObject, VideoFramePresenter {
    let mode: VideoOutputMode = .metalLayer

    private let renderer: ResizableTextureProtocol
    private let onResize: VideoOutput.TextureUpdateCallback
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let cache: CVMetalTextureCache
    private let sampler: MTLSamplerState
    private let pipeline: MTLRenderPipelineState
    /// Must match what MKVulkanRenderer produces: 32BGRA or 64RGBAHalf.
    private let sourceFormat: MTLPixelFormat
    /// PQ (`ITU-R BT.2100`). Present only when EDR was requested *and*
    /// CoreGraphics knows the space -- permission to go HDR, not a decision.
    private let pqColorspace: CGColorSpace?
    private let srgbColorspace: CGColorSpace?
    /// True when EDR was requested *and* CoreGraphics has a PQ colorspace.
    private let edrCapable: Bool
    /// One pipeline per drawable format. The pipeline is chosen from the
    /// drawable we just got (`drawable.texture.pixelFormat`), never from shared
    /// state, so a format change on the main thread cannot race a draw.
    private let pipeline8: MTLRenderPipelineState
    private let pipeline16F: MTLRenderPipelineState
    /// mpv handle, used to ask which transfer function the frames in the ring
    /// are actually in. See `syncLayerToMPV`.
    private let mpvHandle: OpaquePointer?
    /// Last transfer mpv reported. nil until it answers.
    private var mpvHDR: Bool? = nil
    /// What the layer is configured for right now. Diagnostics only; the draw
    /// path reads the drawable instead.
    private var appliedHDR: Bool? = nil
    /// Frames since the last reconfig; sets the property poll rate. Written by
    /// `resized` on Worker and bumped by `draw` on `presentQueue` -- a lost
    /// update only costs a slower re-tag, so it stays unsynchronised.
    private var framesSinceResize = 0

    private let stateLock = NSLock()
    private weak var view: UIView?
    private var suspended = false
    private var closed = false

    // The ring has 3 slots, so at most 2 frames may be in flight. With 3, mpv
    // can wrap around and overwrite the slot Metal is still sampling -- that is
    // the tearing you see at 4K. inflight <= ring - 1 keeps the write slot out
    // of the in-flight set.
    private let inflight = DispatchSemaphore(value: 2)

    // Drawing happens here, never on Worker: CAMetalLayer can withhold a
    // drawable for a long time and the render loop must not wait for it.
    private let presentQueue = DispatchQueue(
      label: "media_kit_video.metal.present",
      qos: .userInteractive
    )

    // Diagnostics only; deliberately unsynchronised. Losing a count here is
    // preferable to taking a lock in the per-frame path.
    private var presented = 0
    private var skippedNil = 0
    private var skippedBusy = 0
    private var skippedNoView = 0
    private var maxDrawMs: Double = 0
    private var firstPresentLogged = false
    private var lastStats = CACurrentMediaTime()

    init?(
      renderer: ResizableTextureProtocol,
      pixelFormat16F: Bool,
      outputEDR: Bool = false,
      mpvHandle: OpaquePointer? = nil,
      onResize: @escaping VideoOutput.TextureUpdateCallback
    ) {
      guard let device = MTLCreateSystemDefaultDevice(),
        let queue = device.makeCommandQueue(),
        let library = MetalLayerPresenter.shaderLibrary(device),
        let vertexFn = library.makeFunction(name: "mk_vertex"),
        let fragmentFn = library.makeFunction(name: "mk_fragment")
      else {
        NSLog("MetalLayer: unavailable (no Metal device or shader)")
        return nil
      }

      var cache: CVMetalTextureCache?
      guard
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &cache)
          == kCVReturnSuccess,
        let cache = cache
      else {
        NSLog("MetalLayer: CVMetalTextureCacheCreate failed")
        return nil
      }

      let samplerDescriptor = MTLSamplerDescriptor()
      samplerDescriptor.minFilter = .linear
      samplerDescriptor.magFilter = .linear
      samplerDescriptor.sAddressMode = .clampToEdge
      samplerDescriptor.tAddressMode = .clampToEdge
      guard let sampler = device.makeSamplerState(descriptor: samplerDescriptor)
      else {
        NSLog("MetalLayer: makeSamplerState failed")
        return nil
      }

      // EDR is only worth it if CoreGraphics can hand us a PQ colorspace.
      var pqColorspace: CGColorSpace? = nil
      if outputEDR {
        pqColorspace = CGColorSpace(name: CGColorSpace.itur_2100_PQ)
        if pqColorspace == nil {
          NSLog("MetalLayer: PQ colorspace unavailable, staying SDR")
        }
      }
      let edrCapable = pqColorspace != nil

      // Both pipelines up front: the drawable format follows the source and can
      // flip mid-playback (SDR clip -> HDR clip without a teardown), so
      // building one on demand would stall a frame inside the render loop.
      //
      // NOT `_srgb`: the shader writes already-encoded values, and an sRGB
      // attachment would encode them a second time.
      func makePipeline(_ format: MTLPixelFormat) -> MTLRenderPipelineState? {
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vertexFn
        descriptor.fragmentFunction = fragmentFn
        descriptor.colorAttachments[0].pixelFormat = format
        return try? device.makeRenderPipelineState(descriptor: descriptor)
      }
      guard let pipeline8 = makePipeline(.bgra8Unorm),
        let pipeline16F = makePipeline(.rgba16Float)
      else {
        NSLog("MetalLayer: makeRenderPipelineState failed")
        return nil
      }

      self.renderer = renderer
      self.onResize = onResize
      self.device = device
      self.queue = queue
      self.cache = cache
      self.sampler = sampler
      self.pipeline8 = pipeline8
      self.pipeline16F = pipeline16F
      self.sourceFormat = pixelFormat16F ? .rgba16Float : .bgra8Unorm
      self.pqColorspace = pqColorspace
      self.srgbColorspace = CGColorSpace(name: CGColorSpace.sRGB)
      self.edrCapable = edrCapable
      self.mpvHandle = mpvHandle

      super.init()

      NotificationCenter.default.addObserver(
        self,
        selector: #selector(onBackground),
        name: UIApplication.didEnterBackgroundNotification,
        object: nil
      )
      NotificationCenter.default.addObserver(
        self,
        selector: #selector(onForeground),
        name: UIApplication.willEnterForegroundNotification,
        object: nil
      )

      NSLog(
        "MetalLayer: init device=\(device.name) src=\(pixelFormat16F ? "rgba16f" : "bgra8") edrCapable=\(edrCapable ? 1 : 0) drawable=pending"
      )
    }

    // MARK: - VideoFramePresenter

    func activate() {
      // There is no FlutterTexture on this path. -1 means "no texture id"; the
      // mode is what switches the widget over to UiKitView. Without this call
      // Dart's create() waits forever and the player page never finishes
      // loading.
      onResize(-1, .zero, .metalLayer)
    }

    func resized(_ size: CGSize) {
      // New content may mean a different transfer function. Restart the fast
      // poll so `syncLayerToMPV` re-tags the layer within a frame or two.
      framesSinceResize = 0
      DispatchQueue.main.sync {
        self.onResize(-1, size, .metalLayer)
      }
    }

    func attach(_ view: VideoLayerView?) {
      stateLock.lock()
      self.view = view
      stateLock.unlock()
      guard let view = view else {
        NSLog("MetalLayer: detached")
        return
      }
      // The view is a fresh layer every time (route push/pop, Hero), so this
      // has to configure unconditionally rather than trust `appliedHDR`.
      applyDrawable(
        hdr: edrCapable && (mpvHDR ?? false),
        reason: "attach"
      )
      NSLog(
        "MetalLayer: attached \(view.bounds.size) drawable=\(appliedHDR == true ? "rgba16f/pq" : "bgra8/srgb") edrCapable=\(edrCapable ? 1 : 0)"
      )
    }

    /// Points the layer at the transfer function mpv is really producing.
    ///
    /// The drawable format is part of the tag, not just the colorspace: a
    /// 16F drawable tagged sRGB is *not* a pass-through on iOS -- CoreAnimation
    /// reads float content as linear and re-encodes it for the display, which
    /// lifts every midtone. That is the "washed out, no white veil" SDR picture
    /// from C4.2. The 8-bit drawable below is bit-for-bit the C3 path, which
    /// was verified against the baseline.
    ///
    /// Main thread only.
    private func applyDrawable(hdr: Bool, reason: String) {
      stateLock.lock()
      let view = self.view
      stateLock.unlock()

      appliedHDR = hdr
      guard let layer = view?.layer as? CAMetalLayer else { return }

      let format: MTLPixelFormat = hdr ? .rgba16Float : .bgra8Unorm
      // Assigning `pixelFormat` discards the drawables already handed out, so
      // only touch it when it actually differs.
      if layer.pixelFormat != format {
        layer.pixelFormat = format
      }
      layer.colorspace = hdr ? pqColorspace : srgbColorspace
      if #available(iOS 16.0, *) {
        layer.wantsExtendedDynamicRangeContent = hdr
      }
      NSLog(
        "MetalLayer: drawable \(hdr ? "rgba16f/pq edr=1" : "bgra8/srgb edr=0") (\(reason))"
      )
    }

    /// Asks mpv which transfer function the frames in the ring are in.
    ///
    /// `video-target-params` would be the right source (`vo_get_target_params`),
    /// but this mpv build (0.41.0) does not export it: the property reads back
    /// NULL, which C4.2 treated as "not HDR" for *every* clip -- SDR and DV P5
    /// alike -- so the layer was never tagged PQ again.
    ///
    /// `video-out-params` is available and, in this pipeline, accurate: the VO
    /// does not convert (SDR keeps `bt.1886` even with `target-trc=pq` set, DV
    /// arrives as `pq`), so it is exactly what lands in the IOSurface. If the
    /// VO ever starts converting, this reading and the picture will disagree --
    /// the log line below is what tells them apart.
    ///
    /// Runs on Worker, before `nextDrawable()`. Fast poll right after a
    /// reconfig (mpv may still report the previous clip), then one read every
    /// 30 frames so a same-size content switch still re-tags the layer.
    private func syncLayerToMPV() {
      guard edrCapable, let handle = mpvHandle else { return }
      framesSinceResize += 1
      if framesSinceResize >= 90 && framesSinceResize % 30 != 0 { return }
      guard
        let gamma = MPVHelpers.getStringProperty(
          handle,
          "video-out-params/gamma"
        )
      else { return }
      let hdr = (gamma == "pq" || gamma == "hlg")
      if mpvHDR == hdr { return }
      mpvHDR = hdr
      DispatchQueue.main.async { [weak self] in
        self?.applyDrawable(hdr: hdr, reason: "video-out-params/gamma=\(gamma)")
      }
    }

    func present(_ size: CGSize) {
      stateLock.lock()
      let dead = closed || suspended
      let target = view
      stateLock.unlock()

      guard !dead, let target = target, target.bounds.width >= 1 else {
        skippedNoView += 1
        return
      }

      guard let pixelBuffer = renderer.copyPixelBuffer()?.takeRetainedValue()
      else {
        return
      }

      // Back-pressure *before* queueing, so at most `inflight` frames are
      // pending and the ring slot behind `pixelBuffer` cannot be recycled while
      // we still hold it.
      guard inflight.wait(timeout: .now()) == .success else {
        skippedBusy += 1
        return
      }

      // Hand the frame off instead of drawing here.
      //
      // Everything from nextDrawable() onwards can block for hundreds of
      // milliseconds: the display link is idle while the app is paused, the
      // screen is off, or the GPU is behind, and CAMetalLayer will not hand out
      // a drawable until one comes back. Blocking Worker delays the next
      // mpv_render_context_render(), which trips mpv's watchdog -- it then logs
      // "render() not being called or stuck", tears the video chain down
      // (EOF -> new Decoder format -> playback restart) and the user waits
      // 1-2.7s for the next frame. That is the "stutter after pause" symptom.
      let semaphore = inflight
      presentQueue.async { [weak self, weak target] in
        guard let self = self, let target = target else {
          semaphore.signal()
          return
        }
        self.draw(pixelBuffer: pixelBuffer, target: target)
      }
    }

    /// Runs on `presentQueue`, never on Worker.
    private func draw(pixelBuffer: CVPixelBuffer, target: UIView) {
      let started = CACurrentMediaTime()
      defer {
        let ms = (CACurrentMediaTime() - started) * 1000
        if ms > maxDrawMs { maxDrawMs = ms }
        if ms > 50 {
          NSLog(
            "MetalLayer: draw took \(Int(ms))ms (nextDrawable or GPU backpressure)"
          )
        }
      }

      guard !closed, let layer = target.layer as? CAMetalLayer else {
        inflight.signal()
        return
      }

      // Cheap after the first frames (a couple of property reads, then one per
      // 30 frames). Must happen before nextDrawable so the layer is tagged
      // correctly for the frame we are about to present.
      syncLayerToMPV()

      guard let drawable = layer.nextDrawable() else {
        inflight.signal()
        skippedNil += 1
        return
      }

      // Picked from the drawable, not from `appliedHDR`: the format can change
      // on the main thread between nextDrawable() and here.
      let pipeline: MTLRenderPipelineState
      switch drawable.texture.pixelFormat {
      case .rgba16Float: pipeline = pipeline16F
      case .bgra8Unorm: pipeline = pipeline8
      default:
        inflight.signal()
        NSLog(
          "MetalLayer: unexpected drawable format \(drawable.texture.pixelFormat.rawValue)"
        )
        return
      }

      let width = CVPixelBufferGetWidth(pixelBuffer)
      let height = CVPixelBufferGetHeight(pixelBuffer)
      var cvTexture: CVMetalTexture?
      let status = CVMetalTextureCacheCreateTextureFromImage(
        kCFAllocatorDefault, cache, pixelBuffer, nil, sourceFormat,
        width, height, 0, &cvTexture)
      guard status == kCVReturnSuccess, let cvTexture = cvTexture,
        let source = CVMetalTextureGetTexture(cvTexture)
      else {
        inflight.signal()
        NSLog(
          "MetalLayer: CVMetalTextureCacheCreateTextureFromImage failed \(status)"
        )
        return
      }

      guard let commandBuffer = queue.makeCommandBuffer() else {
        inflight.signal()
        return
      }

      let pass = MTLRenderPassDescriptor()
      pass.colorAttachments[0].texture = drawable.texture
      // Full-screen overwrite: loading the previous contents is pure waste.
      pass.colorAttachments[0].loadAction = .dontCare
      pass.colorAttachments[0].storeAction = .store

      guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass)
      else {
        inflight.signal()
        return
      }
      encoder.setRenderPipelineState(pipeline)
      encoder.setFragmentTexture(source, index: 0)
      encoder.setFragmentSamplerState(sampler, index: 0)
      encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
      encoder.endEncoding()
      commandBuffer.present(drawable)

      // cvTexture has to outlive the GPU work: the MTLTexture returned by
      // CVMetalTextureGetTexture is backed by storage owned by the
      // CVMetalTexture. Capturing it here is what actually extends its
      // lifetime -- `_ = cvTexture` alone only silences a warning and ARC
      // releases it as soon as present() returns.
      commandBuffer.addCompletedHandler { [weak self, cvTexture] buffer in
        _ = cvTexture
        if buffer.status == .error {
          NSLog(
            "MetalLayer: command buffer error \(String(describing: buffer.error))"
          )
        }
        self?.inflight.signal()
        self?.tick()
      }
      commandBuffer.commit()
    }

    func shutdown() {
      stateLock.lock()
      closed = true
      view = nil
      stateLock.unlock()
      NotificationCenter.default.removeObserver(self)
      CVMetalTextureCacheFlush(cache, 0)
    }

    // MARK: - private

    @objc private func onBackground() {
      stateLock.lock()
      suspended = true
      stateLock.unlock()
    }

    @objc private func onForeground() {
      stateLock.lock()
      suspended = false
      stateLock.unlock()
    }

    private func tick() {
      presented += 1
      if !firstPresentLogged {
        firstPresentLogged = true
        NSLog(
          "MetalLayer: first present src=\(sourceFormat == .rgba16Float ? "rgba16f" : "bgra8") dst=\(appliedHDR == true ? "rgba16f/pq" : "bgra8/srgb")"
        )
      }
      let now = CACurrentMediaTime()
      if now - lastStats >= 5.0 {
        NSLog(
          "MetalLayer: stats presented=\(presented) nil=\(skippedNil) busy=\(skippedBusy) noview=\(skippedNoView) drawMax=\(Int(maxDrawMs))ms"
        )
        lastStats = now
      }
    }

    /// Cheap precondition check so `VideoOutput` can pick the intermediate pixel
    /// format *before* building the renderer.
    ///
    /// A 16F ring is only safe when the layer sink can actually be created:
    /// Flutter's external texture path rejects anything but BGRA8, so a failed
    /// presenter would otherwise leave the texture fallback holding a buffer the
    /// engine refuses.
    static var isSupported: Bool {
      guard let device = MTLCreateSystemDefaultDevice() else { return false }
      return shaderLibrary(device) != nil
    }

    /// Compiled once per process. Doing this lazily inside present() would stall
    /// the first frame for a few hundred milliseconds -- long enough to trip
    /// mpv's "render() not being called" watchdog.
    private static var cachedLibrary: MTLLibrary?

    private static func shaderLibrary(_ device: MTLDevice) -> MTLLibrary? {
      if let cachedLibrary = cachedLibrary { return cachedLibrary }
      cachedLibrary = try? device.makeLibrary(source: shader, options: nil)
      return cachedLibrary
    }

    private static let shader = """
      #include <metal_stdlib>
      using namespace metal;

      struct VOut {
        float4 position [[position]];
        float2 uv;
      };

      vertex VOut mk_vertex(uint vid [[vertex_id]]) {
        // Full-screen triangle. NDC y=+1 is the top of the screen and uv (0,0)
        // is the top-left of the source, matching mpv's
        // MPV_RENDER_PARAM_FLIP_Y=0 output -- no flip needed.
        const float2 pos[3] = { float2(-1.0, -3.0), float2(-1.0, 1.0), float2(3.0, 1.0) };
        const float2 tex[3] = { float2( 0.0,  2.0), float2( 0.0, 0.0), float2(2.0, 0.0) };
        VOut o;
        o.position = float4(pos[vid], 0.0, 1.0);
        o.uv = tex[vid];
        return o;
      }

      fragment float4 mk_fragment(VOut in [[stage_in]],
                                  texture2d<float> src [[texture(0)]],
                                  sampler s [[sampler(0)]]) {
        return src.sample(s, in.uv);
      }
      """
  }

#endif
