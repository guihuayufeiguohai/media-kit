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
    /// PQ. Built only when EDR was requested *and* CoreGraphics has it.
    private let hdrColorspace: CGColorSpace?
    /// Linear Display-P3. Kept for A/B against `.pq`; nothing selects it now
    /// that a headroom collapse goes straight to `.sdr` (see `degradeToSDR`).
    private let linearColorspace: CGColorSpace?
    private let srgbColorspace: CGColorSpace?
    /// Which container mpv writes into and the layer is tagged with.
    ///
    /// Measured on device, one clip at a time -- each combination has a distinct
    /// failure, and no single container serves both:
    ///
    /// | container | SDR clip | HDR clip | headroom 1.0 |
    /// |---|---|---|---|
    /// | `.sdr` (bgra8/sRGB, EDR off) | correct (== C3, verified pixel-exact) | tone mapped, no highlights | fine |
    /// | `.pq` (16F/PQ, EDR on) | **too bright** (SDR white lands above SDR) | correct (== C4.13, approved) | **blown out to white** |
    /// | `.linearEDR` (16F/linear-P3) | correct-ish | **too dim** | dim, colours intact |
    ///
    /// So the container follows the clip, and Low Power Mode forces `.sdr`
    /// (mpv tone maps for us). It is chosen at `attach`, from `video-params`,
    /// before the layer owns a drawable -- the only place a change is safe:
    /// flipping an already-composited layer leaves the old identity behind
    /// (C4.8).
    private enum Container: String {
      case sdr
      case pq
      case linearEDR
    }

    /// Chosen at `resized` (source known) and applied at `attach`. Guarded by
    /// `stateLock`; the present queue reads it through `container`.
    private var desiredContainer: Container = .sdr
    private var container: Container = .sdr
    /// The `target-trc` the current container asked for. Read by
    /// `frameMustWait` so it does not have to re-read `video-params` per frame.
    private var wantedTrc = "auto"
    /// Set once mpv reports the target the container expects. Until then the
    /// ring still holds frames rendered for the old target and must not be
    /// shown (bounded by `holdDeadline`).
    private var targetLanded = true
    private var framesToSkip = 0
    private var holdDeadline: CFTimeInterval = 0
    private var skippedHold = 0
    /// One-way per clip: once the display loses its headroom we stop claiming
    /// EDR for the rest of it.
    private var headroomEverHigh = false
    private var headroomLowSamples = 0
    private var degraded = false

    /// True when EDR was requested *and* CoreGraphics has a PQ colorspace.
    private let edrCapable: Bool
    /// One pipeline per drawable format. The pipeline is chosen from the
    /// drawable we just got (`drawable.texture.pixelFormat`), never from shared
    /// state, so a format change on the main thread cannot race a draw.
    private let pipeline8: MTLRenderPipelineState
    private let pipeline16F: MTLRenderPipelineState
    /// mpv handle, used once at attach to put mpv into linear output.
    private let mpvHandle: OpaquePointer?
    /// Whether the layer ended up presenting EDR content. Diagnostics only.
    private var layerIsEDR = false
    /// Bumped when a clip starts (new texture) or a fresh view is attached to
    /// one. The present queue keeps its own probe state and resets it when this
    /// changes, so nothing has to be shared across threads.
    private var clipGeneration = 0
    /// When the current clip's sink came up. Printed next to every probe sample
    /// so "how long is the screen black" is a number, not a feeling.
    private var clipStartedAt: CFTimeInterval = 0

    // Probe series state. `presentQueue` only.
    private var probedGeneration = -1
    private var framesThisClip = 0
    private var probeAtFrame = 1
    private var probeStep = 1
    private var probeSamples = 0

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
    /// Headroom is sampled every second while an EDR container is in use, but
    /// logged every five so the log stays readable. Present queue only.
    private var lastHeadroomCheck = CACurrentMediaTime()
    private var lastHeadroomLog = CACurrentMediaTime()

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

      // Both EDR spaces up front; which one gets used is decided per clip at
      // attach. See `Container` for the measurements behind that.
      var hdrColorspace: CGColorSpace? = nil
      var linearColorspace: CGColorSpace? = nil
      if outputEDR {
        hdrColorspace = CGColorSpace(name: CGColorSpace.itur_2100_PQ)
        linearColorspace = CGColorSpace(
          name: CGColorSpace.extendedLinearDisplayP3
        )
        if hdrColorspace == nil || linearColorspace == nil {
          NSLog("MetalLayer: EDR colorspace unavailable, HDR clips stay SDR")
        }
      }
      let edrCapable = hdrColorspace != nil && linearColorspace != nil

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
      self.hdrColorspace = hdrColorspace
      self.linearColorspace = linearColorspace
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
      // `inactive` 也要停：拉控制中心、下拉通知、截图、来电这些都会短暂
      // resign active，而 CAMetalLayer 在这段时间里拿不到 drawable。之前只挡了
      // `didEnterBackground`，所以这些场景会漏 —— 表现出来就是"截屏/下拉时卡一下"。
      NotificationCenter.default.addObserver(
        self,
        selector: #selector(onBackground),
        name: UIApplication.willResignActiveNotification,
        object: nil
      )
      NotificationCenter.default.addObserver(
        self,
        selector: #selector(onForeground),
        name: UIApplication.didBecomeActiveNotification,
        object: nil
      )

      // Neutral default before any media exists. `resized` moves it to the
      // clip's container as soon as the source transfer is known -- still
      // ~200ms before the first frame is presented.
      applyVideoTarget(for: .sdr)
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
      // New clip: restart the probe series and pick the container.
      //
      // `video-params` is filled in as soon as the VO has a decoded frame, and
      // this callback fires right after that -- typically 200-400ms before the
      // first present. Deciding here (and applying at `attach`, which follows
      // within a few ms) is the only window where changing the container is
      // safe, because the layer does not own a drawable yet.
      let source = currentSourceGamma()
      let isHDRSource = MetalLayerPresenter.isHDR(source)
      let container = containerFor(source)

      stateLock.lock()
      clipGeneration += 1
      clipStartedAt = CACurrentMediaTime()
      // A same-size content switch can call this after the layer exists; the
      // container must not change then (it would be a mid-life flip), so keep
      // the one already in use and say so.
      let canApply = (view == nil) || (container == self.container)
        || !firstPresentLogged
      let needsLateConfigure = canApply && (view != nil)
        && (container != self.container)
      if canApply { desiredContainer = container }
      framesToSkip = 0
      holdDeadline = CACurrentMediaTime() + 1.5
      let applied = desiredContainer
      stateLock.unlock()

      // Already on the right target (SDR after SDR, HDR after HDR)? Then there
      // is nothing to wait for. After the first clip of a session it usually is
      // not: mpv keeps the previous target until this write lands, and until it
      // does the ring holds frames for the old one.
      let want = MetalLayerPresenter.trc(for: applied, sourceIsHDR: isHDRSource)
      let current = mpvHandle.flatMap {
        MPVHelpers.getStringProperty($0, "target-trc")
      }
      stateLock.lock()
      targetLanded = (current == want)
      stateLock.unlock()

      if canApply {
        applyVideoTarget(for: applied, sourceIsHDR: isHDRSource)
      }
      if needsLateConfigure {
        // `attach` ran before `video-params` was filled in (no size change to
        // trigger this callback), so the layer was born as SDR. Correct it while
        // nothing has been presented yet -- after the first present a change
        // would leave the layer with a stale identity (C4.8).
        DispatchQueue.main.async { [weak self] in
          guard let self = self, !self.isClosed else { return }
          self.stateLock.lock()
          self.container = container
          self.stateLock.unlock()
          self.configureLayer(container: container)
          NSLog("MetalLayer: late container fix -> \(container.rawValue)")
        }
      }
      if !canApply {
        NSLog(
          "MetalLayer: container change to \(container.rawValue) ignored (layer already composited as \(self.container.rawValue))"
        )
      }
      NSLog(
        "MetalLayer: clip source=\(source ?? "-") -> container=\(applied.rawValue)"
      )

      DispatchQueue.main.sync {
        self.onResize(-1, size, .metalLayer)
      }
    }

    func attach(_ view: VideoLayerView?) {
      stateLock.lock()
      self.view = view
      if view != nil {
        clipGeneration += 1
        clipStartedAt = CACurrentMediaTime()
      }
      stateLock.unlock()
      guard let view = view else {
        NSLog("MetalLayer: detached")
        return
      }
      // Decide here rather than trusting `resized`: a clip that does not change
      // the texture size never calls it, and this is the last moment where the
      // container can be set safely -- before the layer owns a drawable.
      // Flipping it afterwards is what left the DV picture washed out in C4.8.
      let source = currentSourceGamma()
      let isHDRSource = MetalLayerPresenter.isHDR(source)
      let container = containerFor(source)
      stateLock.lock()
      self.container = container
      self.desiredContainer = container
      stateLock.unlock()
      // Idempotent: it early-returns when mpv already agrees. Called
      // unconditionally because the container can stay `.sdr` while the target
      // still has to change (`auto` for an SDR clip, `bt.1886` for an HDR one
      // that lost its EDR).
      applyVideoTarget(for: container, sourceIsHDR: isHDRSource)
      configureLayer(container: container)
      // An attached layer with `isOpaque = true` and no drawable yet composites
      // as *white*, not black. Until the first real frame arrives that window is
      // the "white veil on the first frame" users report.
      paintBlack(layer: view.layer as? CAMetalLayer)
      NSLog(
        "MetalLayer: attached \(view.bounds.size) container=\(container.rawValue) source=\(source ?? "-") edrCapable=\(edrCapable ? 1 : 0)"
      )
    }

    /// Configures the layer for the container `resized` picked.
    ///
    /// `pixelFormat`, the EDR flag and the colorspace are one display state; the
    /// three are committed together because a layer seen as "16F but still SDR"
    /// keeps that identity until something forces a re-evaluation.
    ///
    /// The transaction matters: `pixelFormat`, the EDR flag and the colorspace
    /// describe one display state and must be seen together. Set one at a time,
    /// the layer can end up "16F but still SDR", which the compositor keeps
    /// until something forces it to re-evaluate.
    ///
    /// Main thread only.
    private func configureLayer(container: Container) {
      stateLock.lock()
      let view = self.view
      stateLock.unlock()
      guard let layer = view?.layer as? CAMetalLayer else { return }

      let wantsEDR = (container != .sdr)
      layerIsEDR = wantsEDR
      let format: MTLPixelFormat = wantsEDR ? .rgba16Float : .bgra8Unorm
      let colorspace: CGColorSpace?
      switch container {
      case .sdr: colorspace = srgbColorspace
      case .pq: colorspace = hdrColorspace
      case .linearEDR: colorspace = linearColorspace
      }

      CATransaction.begin()
      CATransaction.setDisableActions(true)
      if layer.pixelFormat != format {
        layer.pixelFormat = format
      }
      if #available(iOS 16.0, *) {
        if layer.wantsExtendedDynamicRangeContent != wantsEDR {
          layer.wantsExtendedDynamicRangeContent = wantsEDR
        }
      }
      layer.colorspace = colorspace
      CATransaction.commit()

      let cs = layer.colorspace.map { String(describing: $0.name) } ?? "nil"
      var edr = ""
      if #available(iOS 16.0, *) {
        edr =
          " wantsEDR=\(layer.wantsExtendedDynamicRangeContent ? 1 : 0)"
          + " headroom=\(String(format: "%.2f", UIScreen.main.currentEDRHeadroom))"
          + "/\(String(format: "%.2f", UIScreen.main.potentialEDRHeadroom))"
      }
      NSLog(
        "MetalLayer: layer pf=\(layer.pixelFormat.rawValue) cs=\(cs)\(edr)"
      )
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

    private var isClosed: Bool {
      stateLock.lock()
      defer { stateLock.unlock() }
      return closed
    }

    /// Samples a sparse grid of the frame mpv just produced and logs its range.
    /// Runs on `presentQueue`, for the first eight frames of each clip.
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
        "MetalLayer: probe \(label) max=\(String(format: "%.3f", maximum)) mean=\(String(format: "%.3f", mean)) samples=\(count) layerEDR=\(layerIsEDR ? 1 : 0)"
      )
    }

    /// `video-params/gamma` -- the *source* transfer, which is what the container
    /// choice is based on. Available as soon as the VO has a decoded frame.
    private func currentSourceGamma() -> String? {
      mpvHandle.flatMap {
        MPVHelpers.getStringProperty($0, "video-params/gamma")
      }
    }

    /// Which container a clip with this source transfer needs.
    ///
    /// Low Power Mode takes the EDR headroom away; claiming EDR then paints a
    /// blown-out white picture, so those clips go to mpv's SDR tone mapper.
    static func isHDR(_ source: String?) -> Bool {
      source == "pq" || source == "hlg"
        || source?.hasPrefix("dolbyvision") == true
    }

    private func containerFor(_ source: String?) -> Container {
      guard MetalLayerPresenter.isHDR(source), edrCapable else { return .sdr }
      return ProcessInfo.processInfo.isLowPowerModeEnabled ? .sdr : .pq
    }

    /// The `target-trc` a container needs. Single source of truth for both the
    /// write and the "has it landed" check.
    ///
    /// The `.sdr` case depends on the *source*: `auto` means "keep the source
    /// transfer", which is a no-op for SDR but leaves PQ untouched for an HDR
    /// clip -- and PQ code values read as sRGB are exactly the "white veil" this
    /// path exists to avoid. An HDR clip in the SDR container therefore asks for
    /// `bt.1886` explicitly, which makes mpv do the tone map.
    private static func trc(for container: Container, sourceIsHDR: Bool) -> String {
      switch container {
      case .sdr: return sourceIsHDR ? "bt.1886" : "auto"
      case .pq: return "pq"
      case .linearEDR: return "linear"
      }
    }

    private func sourceIsHDR() -> Bool {
      MetalLayerPresenter.isHDR(currentSourceGamma())
    }

    /// Puts mpv into the given container's output.
    ///
    /// | container | `target-trc` | `target-prim` | why |
    /// |---|---|---|---|
    /// | `.sdr` | `auto` | `auto` | mpv tone maps to bt.1886, layer is plain sRGB (== C3) |
    /// | `.pq` | `pq` | `bt.2020` | HDR video, which is the representation iOS maps best |
    /// | `.linearEDR` | `linear` | `display-p3` | headroom-collapse fallback |
    ///
    /// `target-peak` is `100 x potentialEDRHeadroom` in both cases -- deliberately
    /// **not** a fixed 1000:
    ///
    /// - it is a *ceiling*: content at or below it is passed through untouched,
    ///   anything above is rolled off by mpv. Asking 1000 when the panel does 800
    ///   makes the compositor clip what mpv left alone; asking 1000 when the panel
    ///   can do 1600 throws highlights away;
    /// - measured proof that it is only a ceiling: this DV P5 title reports
    ///   `sig-peak = 4.929` (493 nits) and the probe reads 3.93 at the peak frame.
    ///   If 1.0 were the target peak, 493 nits could never exceed 0.616.
    ///
    /// So for a 493-nit title, 800 and 1000 are the same pixels -- `target-peak`
    /// cannot be the reason the picture changed brightness between builds.
    ///
    /// Once RPU handling lands, `target-peak` is also the scale the L2 trims are
    /// computed against, which is why it must track the real panel.
    ///
    /// Serialised on `targetQueue`, never main: `mpv_set_property` waits for the
    /// core's play loop, which can itself be waiting for the render thread. From
    /// main that cycles with `resized()`'s `main.sync`; from Worker it is the
    /// C4.1 deadlock.
    private func applyVideoTarget(
      for container: Container,
      sourceIsHDR: Bool = false
    ) {
      guard let handle = mpvHandle else { return }
      let trc = MetalLayerPresenter.trc(for: container, sourceIsHDR: sourceIsHDR)
      let current = MPVHelpers.getStringProperty(handle, "target-trc")
      stateLock.lock()
      wantedTrc = trc
      if current != trc {
        // Whatever is in the ring now was rendered for the old target.
        targetLanded = false
        framesToSkip = 0
        holdDeadline = CACurrentMediaTime() + 1.5
      }
      stateLock.unlock()
      let prim: String
      switch container {
      case .sdr: prim = sourceIsHDR ? "bt.709" : "auto"
      case .pq: prim = "bt.2020"
      case .linearEDR: prim = "display-p3"
      }
      let peak = (container == .sdr) ? "auto" : MetalLayerPresenter.linearPeak()
      targetQueue.async { [weak self] in
        guard let self = self, !self.isClosed else { return }
        if MPVHelpers.getStringProperty(handle, "target-trc") == trc,
          MPVHelpers.getStringProperty(handle, "target-prim") == prim
        {
          return
        }
        let ok = MPVHelpers.setStringProperty(handle, "target-trc", trc)
        let okPrim = MPVHelpers.setStringProperty(handle, "target-prim", prim)
        let okPeak = MPVHelpers.setStringProperty(handle, "target-peak", peak)
        NSLog(
          "MetalLayer: mpv target -> trc=\(trc) prim=\(prim) peak=\(peak)\(ok && okPrim && okPeak ? "" : " FAILED")"
        )
      }
    }

    /// The panel's peak in nits, derived from the EDR headroom: headroom is by
    /// definition "how many times diffuse white the display can go", so the peak
    /// is `100 x headroom` on the assumption that diffuse white sits at 100 nits.
    /// This only sets mpv's roll-off ceiling, not the display mapping.
    ///
    /// Read once, at attach. `currentEDRHeadroom` is still 1.00 there because the
    /// system ramps it once it sees EDR content, so `potential` is the honest
    /// number; the running value is logged every 5s in `tick` for comparison.
    private static func linearPeak() -> String {
      var headroom: CGFloat = 1.0
      if #available(iOS 16.0, *) {
        headroom = UIScreen.main.potentialEDRHeadroom
      }
      let peak = max(200.0, min(10000.0, 100.0 * Double(headroom)))
      return String(Int(peak))
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

      if frameMustWait() {
        skippedHold += 1
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

      // No mpv calls here: the layer was already tagged by `present` on Worker,
      // and everything below runs on `presentQueue`.
      guard let drawable = layer.nextDrawable() else {
        inflight.signal()
        skippedNil += 1
        return
      }

      // Probe *series* rather than a single sample.
      //
      // The question this has to answer is "how long is the picture black at the
      // start", and one sample at frame 1 cannot tell a black *clip* from a
      // black *frame* -- which is exactly the ambiguity that made the last round
      // of reports unanswerable. Samples land at frames 1, 2, 4, 8, 16, 32, 64
      // and stop: eight CVPixelBuffer locks per clip, so it stays off the frame
      // budget.
      stateLock.lock()
      let generation = clipGeneration
      let startedAt = clipStartedAt
      stateLock.unlock()
      if generation != probedGeneration {
        probedGeneration = generation
        framesThisClip = 0
        probeAtFrame = 1
        probeStep = 1
        probeSamples = 0
      }
      framesThisClip += 1
      if probeSamples < 8, framesThisClip >= probeAtFrame {
        probeSamples += 1
        probeFrameRange(
          pixelBuffer,
          label:
            "frame \(framesThisClip) +\(Int((CACurrentMediaTime() - startedAt) * 1000))ms"
        )
        probeAtFrame += probeStep
        if probeStep < 32 { probeStep *= 2 }
      }

      // Picked from the drawable, not from shared state: the format can change
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

    /// `didEnterBackground` 与 `willResignActive` 都走这里。
    ///
    /// 两者都要停上屏：后台时 CAMetalLayer 不会给 drawable，而 inactive
    /// （控制中心 / 通知中心 / 截图 / 来电横幅）期间合成器也会暂停这一层，
    /// `nextDrawable` 卡住会连带把渲染线程拖住，mpv 于是报
    /// "render() not being called or stuck"，恢复后要等一次重建。
    @objc private func onBackground() {
      stateLock.lock()
      let changed = !suspended
      suspended = true
      stateLock.unlock()
      if changed { NSLog("MetalLayer: presenting suspended") }
    }

    @objc private func onForeground() {
      stateLock.lock()
      let changed = suspended
      suspended = false
      stateLock.unlock()
      if changed { NSLog("MetalLayer: presenting resumed") }
    }

    private func tick() {
      presented += 1
      if !firstPresentLogged {
        firstPresentLogged = true
        NSLog(
          "MetalLayer: first present src=\(sourceFormat == .rgba16Float ? "rgba16f" : "bgra8") dst=\(layerIsEDR ? "rgba16f/\(container.rawValue)" : "bgra8/srgb")"
        )
      }
      let now = CACurrentMediaTime()
      if now - lastStats >= 5.0 {
        NSLog(
          "MetalLayer: stats presented=\(presented) nil=\(skippedNil) busy=\(skippedBusy) noview=\(skippedNoView) hold=\(skippedHold) drawMax=\(Int(maxDrawMs))ms"
        )
        lastStats = now
      }

      // Headroom decides whether an EDR layer is safe at all: Low Power Mode can
      // take it away in one second, and every frame until we react is painted
      // white. Check every second while an EDR container is in use; log every
      // five so the file stays readable.
      stateLock.lock()
      let usesEDR = (container == .pq)
      stateLock.unlock()
      guard usesEDR, now - lastHeadroomCheck >= 1.0 else { return }
      lastHeadroomCheck = now
      let shouldLog = (now - lastHeadroomLog >= 5.0)
      if shouldLog { lastHeadroomLog = now }

      DispatchQueue.main.async { [weak self] in
        guard let self = self, !self.isClosed else { return }
        guard #available(iOS 16.0, *) else { return }
        let current = UIScreen.main.currentEDRHeadroom
        let lpm = ProcessInfo.processInfo.isLowPowerModeEnabled
        if shouldLog {
          NSLog(
            "MetalLayer: edr headroom current=\(String(format: "%.2f", current)) potential=\(String(format: "%.2f", UIScreen.main.potentialEDRHeadroom)) lpm=\(lpm ? 1 : 0)"
          )
        }
        guard self.container == .pq, !self.degraded else { return }

        // Low Power Mode is a deliberate user action: react on the first sample.
        if lpm {
          self.degradeToSDR()
          return
        }
        // A thermal/power dip can recover; it needs to persist.
        if current > 1.5 {
          self.headroomEverHigh = true
          self.headroomLowSamples = 0
          return
        }
        guard self.headroomEverHigh else { return }
        self.headroomLowSamples += 1
        if self.headroomLowSamples < 3 {
          NSLog(
            "MetalLayer: EDR headroom at \(String(format: "%.2f", current)) -- waiting for it to settle"
          )
          return
        }
        self.degradeToSDR()
      }
    }

    /// Gives up the EDR claim when the display has no headroom left.
    ///
    /// A PQ layer on a headroom-1.0 display paints the picture white -- that is
    /// what Low Power Mode does to DV mid-playback, and mpv cannot compensate:
    /// the damage happens in the compositor, after our pixels. The only exit is
    /// the SDR container, which is the path SDR clips already use: bgra8 + sRGB
    /// + EDR off, with mpv told to output `bt.1886`.
    ///
    /// `bt.1886` matters: `auto` means "keep the source", which for a PQ clip
    /// leaves PQ in the ring -- and PQ read as sRGB is the white veil itself.
    ///
    /// One-way per clip. Coming back to EDR mid-playback is the transition that
    /// leaves a layer stuck with its old identity (C4.8).
    ///
    /// Main thread (it touches the layer).
    private func degradeToSDR() {
      let isHDRSource = MetalLayerPresenter.isHDR(currentSourceGamma())
      stateLock.lock()
      guard !degraded, container == .pq else {
        stateLock.unlock()
        return
      }
      degraded = true
      container = .sdr
      desiredContainer = .sdr
      targetLanded = false
      framesToSkip = 0
      holdDeadline = CACurrentMediaTime() + 1.5
      stateLock.unlock()

      NSLog(
        "MetalLayer: EDR headroom gone -- dropping to the SDR container for this clip"
      )
      configureLayer(container: .sdr)
      applyVideoTarget(for: .sdr, sourceIsHDR: isHDRSource)
    }

    /// Runs on Worker, before a frame is handed to the present queue. True when
    /// this frame must not be shown.
    ///
    /// After a container change the ring still holds frames rendered for the old
    /// target, and showing those with the new tag is the "blown out first frame"
    /// (C4.12) or the white veil (SDR container, PQ content). The wait ends as
    /// soon as mpv reports the expected target -- typically one or two frames --
    /// and is bounded so a rejected write cannot leave a black screen.
    private func frameMustWait() -> Bool {
      stateLock.lock()
      let kind = container
      let landed = targetLanded
      let skip = framesToSkip
      let deadline = holdDeadline
      stateLock.unlock()

      if skip > 0 {
        stateLock.lock()
        framesToSkip = skip - 1
        stateLock.unlock()
        return true
      }
      if landed { return false }

      guard let handle = mpvHandle else { return false }
      stateLock.lock()
      let want = wantedTrc
      stateLock.unlock()
      if MPVHelpers.getStringProperty(handle, "target-trc") == want {
        stateLock.lock()
        targetLanded = true
        framesToSkip = 2
        stateLock.unlock()
        NSLog(
          "MetalLayer: target \(want) landed -- waiting two frames for the ring to catch up"
        )
        return true
      }
      if CACurrentMediaTime() > deadline {
        stateLock.lock()
        targetLanded = true
        stateLock.unlock()
        NSLog("MetalLayer: target write never landed -- showing the frame anyway")
        return false
      }
      return true
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
