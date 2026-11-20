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
    /// What the drawable is set to. `.rgba16Float` only when EDR is actually
    /// active -- see `init`.
    private let drawableFormat: MTLPixelFormat
    /// PQ (`ITU-R BT.2100`) when EDR is active, nil otherwise.
    private let drawableColorspace: CGColorSpace?
    /// True when EDR was requested *and* CoreGraphics has a PQ colorspace.
    /// This is only permission to go HDR, not a decision: `edrActive` is what
    /// the layer is actually doing right now.
    private let edrCapable: Bool
    /// Whether the layer is currently tagged PQ + EDR.
    private var edrActive = false
    /// mpv handle, used to ask what transfer function the VO is really
    /// producing. See `resolveTargetTransfer`.
    private let mpvHandle: OpaquePointer?
    /// nil until mpv reports a target transfer.
    private var targetHDR: Bool? = nil

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
      // Without it the drawable would be tagged sRGB, the PQ code values would
      // be read as sRGB and the picture would come out dark and desaturated --
      // worse than plain SDR, not better.
      var pqColorspace: CGColorSpace? = nil
      if outputEDR {
        pqColorspace = CGColorSpace(name: CGColorSpace.itur_2100_PQ)
        if pqColorspace == nil {
          NSLog("MetalLayer: PQ colorspace unavailable, staying SDR")
        }
      }
      let edrCapable = pqColorspace != nil
      // Float for the whole life of the layer. mpv can move from an SDR file to
      // an HDR one without tearing the layer down, and swapping pixelFormat
      // mid-playback discards drawables and needs a new pipeline.
      let drawableFormat: MTLPixelFormat = edrCapable ? .rgba16Float : .bgra8Unorm

      let descriptor = MTLRenderPipelineDescriptor()
      descriptor.vertexFunction = vertexFn
      descriptor.fragmentFunction = fragmentFn
      // Must match the layer's pixel format, which `attach` sets from
      // `drawableFormat`. NOT `.bgra8Unorm_srgb`: the shader writes
      // already-encoded values, and an sRGB attachment would encode them a
      // second time.
      descriptor.colorAttachments[0].pixelFormat = drawableFormat
      guard let pipeline = try? device.makeRenderPipelineState(descriptor: descriptor)
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
      self.pipeline = pipeline
      self.sourceFormat = pixelFormat16F ? .rgba16Float : .bgra8Unorm
      self.drawableFormat = drawableFormat
      self.drawableColorspace = pqColorspace
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
        "MetalLayer: init device=\(device.name) src=\(pixelFormat16F ? "rgba16f" : "bgra8") dst=\(drawableFormat == .rgba16Float ? "rgba16f" : "bgra8") edrCapable=\(edrCapable ? 1 : 0)"
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
      // New content may mean a different transfer function (SDR file -> HDR
      // file). Drop the cached answer so the next frame re-reads it.
      targetHDR = nil
      DispatchQueue.main.sync {
        self.onResize(-1, size, .metalLayer)
      }
    }

    func attach(_ view: VideoLayerView?) {
      stateLock.lock()
      self.view = view
      stateLock.unlock()
      if let view = view {
        configureLayer(of: view)
        applyTargetTransfer()
        NSLog(
          "MetalLayer: attached \(view.bounds.size) dst=\(drawableFormat == .rgba16Float ? "rgba16f" : "bgra8") edrCapable=\(edrCapable ? 1 : 0) edr=\(edrActive ? 1 : 0)"
        )
      } else {
        NSLog("MetalLayer: detached")
      }
    }

    /// Called on the main thread, before the first drawable is requested.
    ///
    /// The drawable format and colorspace live on the layer, not here, and
    /// `MKVideoLayerView` cannot know them: it is created by Flutter's platform
    /// view factory, which only receives the mpv handle. The presenter is the
    /// only object that knows both the requested configuration and whether EDR
    /// actually came up, so it takes over.
    private func configureLayer(of view: UIView) {
      guard let layer = view.layer as? CAMetalLayer else { return }
      // Guarded: assigning `pixelFormat` discards existing drawables, and a
      // re-attach (route push/pop, Hero) must not throw away a frame.
      if layer.pixelFormat != drawableFormat {
        layer.pixelFormat = drawableFormat
      }
    }

    /// Tag the layer with whatever mpv is *actually* producing.
    ///
    /// Guessing PQ here is what broke SDR playback: mpv passes an SDR file
    /// through as `bt.1886`, and gamma-2.4 code values read through the PQ
    /// EOTF come out massively lifted -- a white veil over the whole picture.
    /// `video-target-params` is the VO's own answer (`vo_get_target_params`),
    /// so it is correct whether or not mpv decided to convert.
    private func resolveTargetTransfer() {
      guard edrCapable, targetHDR == nil, let handle = mpvHandle else { return }
      guard
        let gamma = MPVHelpers.getStringProperty(
          handle,
          "video-target-params/gamma"
        )
      else { return }
      let hdr = (gamma == "pq" || gamma == "hlg")
      targetHDR = hdr
      DispatchQueue.main.async { [weak self] in
        guard let self = self else { return }
        self.applyTargetTransfer()
        NSLog("MetalLayer: target gamma=\(gamma) -> \(hdr ? "pq/edr" : "sdr")")
      }
    }

    /// Main thread only.
    private func applyTargetTransfer() {
      stateLock.lock()
      let view = self.view
      stateLock.unlock()
      guard let layer = view?.layer as? CAMetalLayer else { return }

      let hdr = edrCapable && (targetHDR ?? false)
      edrActive = hdr
      if hdr, let colorspace = drawableColorspace {
        layer.colorspace = colorspace
      } else {
        layer.colorspace = CGColorSpace(name: CGColorSpace.sRGB)
      }
      if #available(iOS 16.0, *) {
        layer.wantsExtendedDynamicRangeContent = hdr
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

      // Cheap after the first frame (early-returns once targetHDR is known).
      // Must happen before nextDrawable so the layer is tagged correctly for
      // the frame we are about to present.
      resolveTargetTransfer()

      guard let drawable = layer.nextDrawable() else {
        inflight.signal()
        skippedNil += 1
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
          "MetalLayer: first present src=\(sourceFormat == .rgba16Float ? "rgba16f" : "bgra8") dst=\(drawableFormat == .rgba16Float ? "rgba16f" : "bgra8") edr=\(edrActive ? 1 : 0)"
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
