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
    /// Last transfer mpv reported (`video-out-params/gamma`). nil until it
    /// answers, which is different from "answered SDR".
    private var mpvGamma: String? = nil
    /// What the layer is configured for right now. Diagnostics only; the draw
    /// path reads the drawable instead.
    private var appliedHDR: Bool? = nil
    /// Frames since the last reconfig; sets the property poll rate. Both
    /// writers (`resized` and `present`) run on Worker.
    private var framesSinceResize = 0
    /// One pixel-value probe per clip. See `probeFrameRange`.
    private var probePending = true
    /// Second probe once playback has settled. The first frame of a clip is
    /// often black or a fade-in, which says nothing about the mapping.
    private var steadyProbePending = true
    /// EDR headroom bookkeeping for the silent-drop fallback.
    private var headroomEverHigh = false
    private var headroomLowSamples = 0
    private var headroomLost = false

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

    // Every `mpv_set_property` call lands here: one writer at a time, on a
    // thread that is neither main (can cycle with `resized`'s main.sync) nor
    // Worker (the render thread, which the play loop may be waiting for).
    private let targetQueue = DispatchQueue(
      label: "media_kit_video.metal.target",
      qos: .userInitiated
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
      probePending = true
      steadyProbePending = true
      headroomLost = false
      headroomEverHigh = false
      headroomLowSamples = 0
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
      // Tag for what mpv is writing *right now*, exactly like `syncLayerToMPV`
      // does per frame. At attach that is normally `auto`/SDR, because the
      // clip's target has not been applied yet -- and that is correct: those
      // first frames really are SDR-mapped, so showing them as SDR is honest.
      //
      // Tagging them PQ instead is what produced the over-exposed first frame,
      // and refusing to present them is what threw 0.25s of picture away.
      let source = mpvHandle.flatMap {
        MPVHelpers.getStringProperty($0, "video-params/gamma")
      }
      let tag = MetalLayerPresenter.transferTag(
        mpvHandle.flatMap { MPVHelpers.getStringProperty($0, "target-trc") }
      )
      // The view is a fresh layer every time (route push/pop, Hero), so this
      // has to configure unconditionally rather than trust `appliedHDR`.
      applyDrawable(gamma: tag, reason: "attach")
      // Pre-arm mpv so the picture becomes HDR as early as it can.
      if let source = source {
        applyMPVTarget(hdr: hdrAllowed && MetalLayerPresenter.isHDR(source))
      }
      // An attached layer with `isOpaque = true` and no drawable yet composites
      // as *white*, not black. Until the first real frame arrives (200-500ms on
      // HDR) that window is the "white veil on the first frame" users report.
      paintBlack(layer: view.layer as? CAMetalLayer)
      NSLog(
        "MetalLayer: attached \(view.bounds.size) tag=\(tag ?? "sdr") edrCapable=\(edrCapable ? 1 : 0) source=\(source ?? "-")"
      )
    }

    /// Points the layer at the transfer mpv is writing **right now**.
    ///
    /// `gamma` here is the effective transfer (`target-trc`), not the source's.
    /// Tagging from the source means the tag runs ahead of the content for as
    /// long as a target change takes to land -- and the frame rendered inside
    /// that window is shown with the wrong interpretation. That is exactly the
    /// grey DV P5 first frame: rendered at `target=auto`, presented tagged PQ.
    ///
    /// Tag = target, always. The source only decides what the target *should*
    /// be, and that is `applyMPVTarget`'s job.
    ///
    /// The drawable format is part of the tag, not just the colorspace: a 16F
    /// drawable tagged sRGB is *not* a pass-through on iOS -- CoreAnimation reads
    /// float content as linear and re-encodes it, lifting every midtone (C4.2).
    /// The 8-bit drawable below is bit-for-bit the C3 path.
    ///
    /// Main thread only.
    private func applyDrawable(gamma: String?, reason: String) {
      stateLock.lock()
      let view = self.view
      mpvGamma = gamma
      stateLock.unlock()

      let hdr = edrCapable && MetalLayerPresenter.isHDR(gamma)
      appliedHDR = hdr
      guard let layer = view?.layer as? CAMetalLayer else { return }

      let format: MTLPixelFormat = hdr ? .rgba16Float : .bgra8Unorm
      // One transaction for all three: `pixelFormat`, the EDR flag and the
      // colorspace describe a single display state, and the render server has
      // to see them together. Assigning them one at a time leaves a window in
      // which the layer is "16F but still SDR" -- and that is the state that
      // sticks (washed-out DV that a rotation only fixes for the duration of
      // the re-layout).
      CATransaction.begin()
      CATransaction.setDisableActions(true)
      // Assigning `pixelFormat` discards the drawables already handed out, so
      // only touch it when it actually differs.
      if layer.pixelFormat != format {
        layer.pixelFormat = format
      }
      // EDR *before* the colorspace: a layer that is still flagged as SDR can
      // have an HDR colorspace clamped.
      if #available(iOS 16.0, *) {
        if layer.wantsExtendedDynamicRangeContent != hdr {
          layer.wantsExtendedDynamicRangeContent = hdr
        }
      }
      if hdr {
        // HLG is not PQ -- same 16F + EDR treatment, different transfer curve.
        layer.colorspace =
          gamma == "hlg"
          ? (CGColorSpace(name: CGColorSpace.itur_2100_HLG) ?? pqColorspace)
          : pqColorspace
      } else {
        layer.colorspace = srgbColorspace
      }
      CATransaction.commit()
      // Read back what the layer actually holds. The washed-out DV report was
      // "composited as SDR while tagged PQ", which is invisible from the code
      // path but obvious here -- a clamped colorspace or a dropped EDR flag.
      let cs = layer.colorspace.map { String(describing: $0.name) } ?? "nil"
      var edr = ""
      if #available(iOS 16.0, *) {
        edr =
          " wantsEDR=\(layer.wantsExtendedDynamicRangeContent ? 1 : 0)"
          + " headroom=\(String(format: "%.2f", UIScreen.main.currentEDRHeadroom))"
          + "/\(String(format: "%.2f", UIScreen.main.potentialEDRHeadroom))"
      }
      NSLog(
        "MetalLayer: drawable \(hdr ? "rgba16f/\(gamma ?? "?") edr=1" : "bgra8/srgb edr=0") pf=\(layer.pixelFormat.rawValue) cs=\(cs)\(edr) (\(reason))"
      )
    }

    private static func isHDR(_ gamma: String?) -> Bool {
      gamma == "pq" || gamma == "hlg"
    }

    /// Whether HDR presentation is worth attempting at all.
    ///
    /// Low Power Mode pins `UIScreen.currentEDRHeadroom` to 1.0 -- confirmed on
    /// device: same build, same clip, washed out with LPM on and correct with it
    /// off. A PQ layer at headroom 1.0 has nowhere to put its highlights, so iOS
    /// squeezes the whole range and the picture reads flat/grey to the eye while
    /// a screenshot still looks right (the capture path uses the colorspace,
    /// not the panel headroom).
    ///
    /// In that state mpv's own tone mapping to SDR (`target-trc=auto`) produces
    /// a normal-looking SDR picture, which is strictly better than a washed-out
    /// PQ one. Decided once per clip, before the layer owns a drawable, so no
    /// SDR<->EDR transition can happen mid-playback; toggling LPM takes effect
    /// on the next clip.
    private var hdrAllowed: Bool {
      edrCapable
        && !ProcessInfo.processInfo.isLowPowerModeEnabled
        && !headroomLost
    }

    /// Fills the layer with black, once, right after attach.
    ///
    /// Main thread. `allowsNextDrawableTimeout` keeps this from blocking when
    /// the view is not on screen yet (it returns nil and we simply skip).
    private func paintBlack(layer: CAMetalLayer?) {
      guard let layer = layer else { return }
      if layer.drawableSize.width < 1 || layer.drawableSize.height < 1 {
        let scale = layer.contentsScale > 0 ? layer.contentsScale : 3
        guard layer.bounds.width > 1, layer.bounds.height > 1 else { return }
        layer.drawableSize = CGSize(
          width: layer.bounds.width * scale,
          height: layer.bounds.height * scale
        )
      }
      guard let drawable = layer.nextDrawable(),
        let commandBuffer = queue.makeCommandBuffer()
      else { return }

      let pass = MTLRenderPassDescriptor()
      pass.colorAttachments[0].texture = drawable.texture
      pass.colorAttachments[0].loadAction = .clear
      pass.colorAttachments[0].storeAction = .store
      // (0,0,0,1) is 0 nits in PQ just as it is in sRGB: black either way.
      pass.colorAttachments[0].clearColor = MTLClearColor(
        red: 0, green: 0, blue: 0, alpha: 1
      )
      guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass)
      else { return }
      encoder.endEncoding()
      commandBuffer.present(drawable)
      commandBuffer.commit()
      NSLog(
        "MetalLayer: painted black \(Int(drawable.texture.width))x\(Int(drawable.texture.height))"
      )
    }

    /// Maps an mpv choice to a transfer we can tag, or nil for SDR.
    private static func transferTag(_ trc: String?) -> String? {
      isHDR(trc) ? trc : nil
    }

    /// Samples a sparse grid of the frame mpv just produced and logs its range.
    /// One shot per clip; runs on `presentQueue`.
    ///
    /// This is the only honest answer to "what is in the ring": every mpv
    /// property describes the frame *before* the GPU chain, and that chain is
    /// free to convert to the target. A bright SDR frame reads `max≈1.000` when
    /// the ring is coded (bt.1886 / sRGB / rescaled linear), and `max≈0.51`
    /// when it is PQ -- PQ places SDR white at 100 nits, code 0.508.
    private func probeFrameRange(_ pixelBuffer: CVPixelBuffer, label: String) {
      // The pod targets iOS 9; `Float16` needs 14. The layer sink itself is
      // iOS 16+, so this only ever returns early on builds that never run it.
      guard #available(iOS 14.0, *) else { return }
      guard
        CVPixelBufferGetPixelFormatType(pixelBuffer)
          == kCVPixelFormatType_64RGBAHalf,
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly) == kCVReturnSuccess
      else {
        NSLog("MetalLayer: probe skipped (not 64RGBAHalf or not CPU-readable)")
        return
      }
      defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
      guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return }

      let width = CVPixelBufferGetWidth(pixelBuffer)
      let height = CVPixelBufferGetHeight(pixelBuffer)
      let rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
      // Every 32nd pixel: ~1500 samples at 1080p, enough for a max/mean and
      // cheap enough to stay off the frame budget.
      let step = 32
      var maximum: Float = 0
      var sum: Float = 0
      var count = 0
      var y = 0
      while y < height {
        let row = base.advanced(by: y * rowBytes)
          .assumingMemoryBound(to: UInt16.self)
        var x = 0
        while x < width {
          let i = x * 4
          let value = max(
            Float(Float16(bitPattern: row[i])),
            max(
              Float(Float16(bitPattern: row[i + 1])),
              Float(Float16(bitPattern: row[i + 2]))
            )
          )
          if value > maximum { maximum = value }
          sum += value
          count += 1
          x += step
        }
        y += step
      }
      let mean = count > 0 ? sum / Float(count) : 0
      NSLog(
        "MetalLayer: probe \(label) max=\(String(format: "%.3f", maximum)) mean=\(String(format: "%.3f", mean)) samples=\(count) tagged=\(appliedHDR == true ? "pq" : "srgb")"
      )
    }

    /// Keeps mpv's target in step with the clip rather than with the config.
    ///
    /// `target-*` are global options, and the VO builds its chain with whatever
    /// they say at that moment -- before the first frame exists. So the decision
    /// has to be made per clip, from the source transfer:
    ///
    /// | source `gamma` | target |
    /// |---|---|
    /// | `pq` / `hlg` | `pq` / `bt.2020` / `1000` (== C4, verified) |
    /// | anything else | `auto` / `auto` / `auto` (== C3, verified) |
    ///
    /// Measured on device: an SDR clip reads `probe max≈0.57` when the target is
    /// PQ (the GPU converts in the shader, `video-out-params` cannot see it) and
    /// `1.000` when it is auto. So the target is what decides the ring content --
    /// which is why the layer tag is taken from `target-trc` and not from this.
    ///
    /// Serialised on `targetQueue`, never main: `mpv_set_property` waits for the
    /// core's play loop, which can itself be waiting for the render thread. From
    /// main that cycles with `resized()`'s `main.sync`; from Worker it is the
    /// C4.1 deadlock.
    private func applyMPVTarget(hdr: Bool) {
      guard edrCapable, let handle = mpvHandle else { return }
      let trc = hdr ? "pq" : "auto"
      targetQueue.async { [weak self] in
        guard let self = self, !self.isClosed else { return }
        // mpv reports the current choice back as a string ("pq"/"auto"/...).
        guard MPVHelpers.getStringProperty(handle, "target-trc") != trc else {
          return
        }
        var ok = MPVHelpers.setStringProperty(handle, "target-trc", trc)
        ok =
          MPVHelpers.setStringProperty(
            handle, "target-prim", hdr ? "bt.2020" : "auto") && ok
        ok =
          MPVHelpers.setStringProperty(
            handle, "target-peak", hdr ? "1000" : "auto") && ok
        NSLog(
          "MetalLayer: mpv target -> trc=\(trc) prim=\(hdr ? "bt.2020" : "auto") peak=\(hdr ? "1000" : "auto")\(ok ? "" : " FAILED")"
        )
      }
    }

    private var isClosed: Bool {
      stateLock.lock()
      defer { stateLock.unlock() }
      return closed
    }

    /// Keeps mpv's target in step with the clip, and tags the layer for what
    /// mpv is writing **right now**.
    ///
    /// mpv builds its chain -- and renders the first frame -- before any
    /// property write can land: the play loop is blocked waiting for exactly
    /// that frame, so `mpv_set_property` cannot get in ahead of it. For an HDR
    /// clip that means the first frames are SDR-mapped. There are only three
    /// things to do with them, and only one loses nothing:
    ///
    /// | handling | result |
    /// |---|---|
    /// | tag them PQ (source-based) | over-exposed garbage |
    /// | do not present (mask) | ~0.25s of picture never shown |
    /// | tag them SDR (what this does) | they look like correct SDR, then mpv's target lands and the picture becomes HDR |
    ///
    /// So the tag follows `target-trc`, never the source. The clip still decides
    /// what the target *should* be -- that is `applyMPVTarget`'s job -- but the
    /// layer only ever claims the transfer that is actually in the ring.
    ///
    /// Runs on Worker, from `present`. Fast poll right after a reconfig (mpv
    /// may still report the previous clip), then one read every 30 frames so a
    /// same-size content switch still re-tags the layer.
    private func syncLayerToMPV() {
      guard edrCapable, let handle = mpvHandle else { return }
      framesSinceResize += 1
      if framesSinceResize >= 90 && framesSinceResize % 30 != 0 { return }

      guard
        let source = MPVHelpers.getStringProperty(
          handle,
          "video-out-params/gamma"
        )
      else { return }
      // Same rule as `attach`: no EDR headroom -> no PQ, let mpv tone map to SDR.
      let want = hdrAllowed ? MetalLayerPresenter.transferTag(source) : nil
      let now = MetalLayerPresenter.transferTag(
        MPVHelpers.getStringProperty(handle, "target-trc")
      )

      // 1) Tag for the transfer that is in the ring right now. Nothing is ever
      //    dropped and no frame is ever shown with the wrong interpretation.
      stateLock.lock()
      let retag = (mpvGamma != now)
      if retag { mpvGamma = now }
      stateLock.unlock()
      if retag {
        DispatchQueue.main.async { [weak self] in
          self?.applyDrawable(gamma: now, reason: "target-trc=\(now ?? "auto")")
        }
      }

      // 2) Drive mpv to agree with the clip.
      if want != now {
        applyMPVTarget(hdr: want != nil)
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

      // Cheap after the first frames (a couple of property reads, then one per
      // 30 frames). Kept on Worker, next to `resized()`, so every mpv client
      // API call this class makes stays on one thread and the frame counter
      // needs no synchronisation.
      syncLayerToMPV()

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

      // No mpv calls here: the layer was already tagged by `present` on Worker,
      // and everything below runs on `presentQueue`.
      guard let drawable = layer.nextDrawable() else {
        inflight.signal()
        skippedNil += 1
        return
      }

      // Numbers beat reasoning here: `video-out-params` is the vf chain's
      // output, NOT what the GPU wrote into the ring, so it cannot tell us
      // whether mpv converted. A bright frame has max=1.0 if the ring is coded
      // (bt.1886/sRGB/linear-scaled), and ≈0.51 if it is PQ (SDR white = 100
      // nits = PQ 0.508).
      if probePending {
        probePending = false
        probeFrameRange(pixelBuffer, label: "first")
      } else if steadyProbePending, framesSinceResize >= 30 {
        steadyProbePending = false
        probeFrameRange(pixelBuffer, label: "steady")
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
        // Sample the display's EDR headroom while playing, not just at attach.
        // A PQ layer with headroom 1.00 has nowhere to put its highlights and
        // reads washed out to the eye while a screenshot still looks right --
        // the "same build, fine yesterday, flat today" report. UIKit wants the
        // main thread, and one block per 5s is free.
        DispatchQueue.main.async { [weak self] in
          guard let self = self, !self.isClosed else { return }
          guard #available(iOS 16.0, *) else { return }
          let current = UIScreen.main.currentEDRHeadroom
          let potential = UIScreen.main.potentialEDRHeadroom
          let lpm = ProcessInfo.processInfo.isLowPowerModeEnabled
          NSLog(
            "MetalLayer: edr headroom current=\(String(format: "%.2f", current)) potential=\(String(format: "%.2f", potential)) lpm=\(lpm ? 1 : 0)"
          )
          // iOS also drops EDR on its own (thermal / power budget) with Low
          // Power Mode off; toggling it on and off restores it. Track it and,
          // after it has clearly gone away, fall back to mpv's SDR tone map for
          // the rest of the clip -- a correct-looking SDR picture beats a flat
          // PQ one.
          //
          // Hysteresis matters: at clip start the headroom ramps 1.00 -> 8.00
          // over ~10s, so only act once it has been high and then stays at 1.00
          // for two consecutive samples. One-way per clip: switching back to EDR
          // mid-playback is the transition that has bitten us before.
          if current > 1.5 {
            self.headroomEverHigh = true
            self.headroomLowSamples = 0
          } else if self.headroomEverHigh, self.appliedHDR == true,
            !self.headroomLost
          {
            self.headroomLowSamples += 1
            if self.headroomLowSamples >= 2 {
              self.headroomLost = true
              NSLog(
                "MetalLayer: EDR headroom gone for 10s (current=\(String(format: "%.2f", current)) lpm=\(lpm ? 1 : 0)) -- falling back to SDR tone mapping for this clip"
              )
            } else {
              NSLog(
                "MetalLayer: EDR headroom lost (current=\(String(format: "%.2f", current)) lpm=\(lpm ? 1 : 0)) -- waiting one more sample"
              )
            }
          }
        }
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
