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
    /// Linear Display-P3. Present only when EDR was requested *and*
    /// CoreGraphics knows the space -- permission to go HDR, not a decision.
    private let hdrColorspace: CGColorSpace?
    private let srgbColorspace: CGColorSpace?
    /// Which colour container mpv writes into and the layer is tagged with.
    ///
    /// Both are constant for the life of the sink, so neither brings back the
    /// per-clip target race. They differ only in how iOS displays the result:
    ///
    /// | container | mpv target | iOS treatment | measured |
    /// |---|---|---|---|
    /// | `.pq` | `pq` / `bt.2020` | standard HDR video, absolute nits, system tone mapping | midtones match the earlier (approved) build |
    /// | `.linearDisplayP3` | `linear` / `display-p3` | EDR layer, 1.0 = SDR white anchored at `peak / headroom` | visibly darker on device for the same clip |
    ///
    /// C4.14 chose the linear one for tidiness ("one container for SDR and HDR")
    /// and it cost brightness: `probe max` showed the same absolute values
    /// (DV P5 peaks at 3.93 = 393 nits, source `sig-peak` 493), yet the picture
    /// read darker than the PQ build on the same frame. The values were never
    /// wrong -- only the display mapping. PQ is what iOS is tuned for, so PQ is
    /// the default again; the linear branch is kept for A/B in one rebuild.
    private enum Container: String {
      case pq
      case linearDisplayP3
    }

    private static let container: Container = .pq

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

      // One container for every clip -- see `Container` for why it is PQ again.
      //
      // Constant per sink means the layer is configured once, at attach, and
      // never re-tagged: that removes the per-clip target change, and with it
      // the first-frame race that no amount of dropping or masking could fix.
      var hdrColorspace: CGColorSpace? = nil
      if outputEDR {
        switch MetalLayerPresenter.container {
        case .pq:
          hdrColorspace = CGColorSpace(name: CGColorSpace.itur_2100_PQ)
        case .linearDisplayP3:
          hdrColorspace = CGColorSpace(
            name: CGColorSpace.extendedLinearDisplayP3
          )
        }
        if hdrColorspace == nil {
          NSLog("MetalLayer: \(MetalLayerPresenter.container) unavailable, staying SDR")
        }
      }
      let edrCapable = hdrColorspace != nil

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

      // Before any media is loaded, so mpv builds every clip's chain with the
      // final target from the very first frame. This is what actually removes
      // the first-frame race: there is nothing left to change later.
      applyVideoTarget()
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
      // New clip: restart the probe series. Nothing else depends on the clip any
      // more -- the layer configuration and mpv's target are both constant.
      stateLock.lock()
      clipGeneration += 1
      clipStartedAt = CACurrentMediaTime()
      stateLock.unlock()
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
      // Constant configuration. Everything below is the same for a 4K Dolby
      // Vision clip and a 720p SDR one, which is the whole point: there is no
      // per-clip decision left to get wrong, and no re-tag mid-playback.
      configureLayer()
      applyVideoTarget()
      // An attached layer with `isOpaque = true` and no drawable yet composites
      // as *white*, not black. Until the first real frame arrives that window is
      // the "white veil on the first frame" users report.
      paintBlack(layer: view.layer as? CAMetalLayer)
      NSLog(
        "MetalLayer: attached \(view.bounds.size) edr=\(layerIsEDR ? 1 : 0) edrCapable=\(edrCapable ? 1 : 0)"
      )
    }

    /// Configures the layer for the chosen EDR container.
    ///
    /// Same three values for every clip: `rgba16Float` + the container's colour
    /// space + EDR. mpv has already converted the clip into that container by the
    /// time frames reach the ring, so there is nothing per-clip to decide and
    /// nothing to change mid-playback.
    ///
    /// The transaction matters: `pixelFormat`, the EDR flag and the colorspace
    /// describe one display state and must be seen together. Set one at a time,
    /// the layer can end up "16F but still SDR", which the compositor keeps
    /// until something forces it to re-evaluate.
    ///
    /// Main thread only.
    private func configureLayer() {
      stateLock.lock()
      let view = self.view
      stateLock.unlock()
      guard let layer = view?.layer as? CAMetalLayer else { return }

      let wantsEDR = hdrColorspace != nil
      layerIsEDR = wantsEDR
      let format: MTLPixelFormat = wantsEDR ? .rgba16Float : .bgra8Unorm

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
      layer.colorspace = wantsEDR ? hdrColorspace : srgbColorspace
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

    /// Puts mpv into the chosen container's output, once, before any clip exists.
    ///
    /// | container | `target-trc` | `target-prim` |
    /// |---|---|---|
    /// | `.pq` | `pq` | `bt.2020` |
    /// | `.linearDisplayP3` | `linear` | `display-p3` |
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
    private func applyVideoTarget() {
      guard let handle = mpvHandle else { return }
      let trc: String
      let prim: String
      switch (edrCapable, MetalLayerPresenter.container) {
      case (false, _):
        trc = "auto"
        prim = "auto"
      case (true, .pq):
        trc = "pq"
        prim = "bt.2020"
      case (true, .linearDisplayP3):
        trc = "linear"
        prim = "display-p3"
      }
      let peak = edrCapable ? MetalLayerPresenter.linearPeak() : "auto"
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
          "MetalLayer: first present src=\(sourceFormat == .rgba16Float ? "rgba16f" : "bgra8") dst=\(layerIsEDR ? "rgba16f/linear-p3" : "bgra8/srgb")"
        )
      }
      let now = CACurrentMediaTime()
      if now - lastStats >= 5.0 {
        NSLog(
          "MetalLayer: stats presented=\(presented) nil=\(skippedNil) busy=\(skippedBusy) noview=\(skippedNoView) drawMax=\(Int(maxDrawMs))ms"
        )
        lastStats = now
        // Sample the display's EDR headroom while playing, not just at attach.
        //
        // In the linear pipeline this is informational: values up to 1.0 are
        // diffuse white and are unaffected by headroom, so a drop to 1.00 only
        // clips highlights instead of flattening the whole picture (which is
        // what the old PQ path did). No fallback is needed any more.
        DispatchQueue.main.async { [weak self] in
          guard let self = self, !self.isClosed else { return }
          guard #available(iOS 16.0, *) else { return }
          NSLog(
            "MetalLayer: edr headroom current=\(String(format: "%.2f", UIScreen.main.currentEDRHeadroom)) potential=\(String(format: "%.2f", UIScreen.main.potentialEDRHeadroom)) lpm=\(ProcessInfo.processInfo.isLowPowerModeEnabled ? 1 : 0)"
          )
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
