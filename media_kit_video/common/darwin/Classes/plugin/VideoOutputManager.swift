#if canImport(Flutter)
  import Flutter
#elseif canImport(FlutterMacOS)
  import FlutterMacOS
#endif

#if os(iOS)
  import UIKit
#endif

public class VideoOutputManager: NSObject {
  private let registry: FlutterTextureRegistry
  private var videoOutputs = [Int64: VideoOutput]()

  init(registry: FlutterTextureRegistry) {
    self.registry = registry
  }

  public func create(
    handle: Int64,
    configuration: VideoOutputConfiguration,
    textureUpdateCallback: @escaping VideoOutput.TextureUpdateCallback
  ) {
    let videoOutput = VideoOutput(
      handle: handle,
      configuration: configuration,
      registry: self.registry,
      textureUpdateCallback: textureUpdateCallback
    )

    self.videoOutputs[handle] = videoOutput
  }

  public func setSize(
    handle: Int64,
    width: Int64?,
    height: Int64?
  ) {
    let videoOutput = self.videoOutputs[handle]
    if videoOutput == nil {
      return
    }

    videoOutput!.setSize(
      width: width,
      height: height
    )
  }

  public func destroy(
    handle: Int64
  ) {
    let videoOutput = self.videoOutputs[handle]
    if videoOutput == nil {
      return
    }

    self.videoOutputs[handle] = nil
  }

  /// Look up the output that owns `handle`.
  ///
  /// The PlatformView factory needs this: it receives only the mpv handle from
  /// Dart and has to find the already-created `VideoOutput` without a second
  /// Create call (which would build a second renderer on the same mpv handle).
  public func output(handle: Int64) -> VideoOutput? {
    return self.videoOutputs[handle]
  }

  #if os(iOS)
    public func attach(handle: Int64, view: UIView) {
      self.videoOutputs[handle]?.attachLayerView(view)
    }

    public func detach(handle: Int64, view: UIView) {
      self.videoOutputs[handle]?.detachLayerView(view)
    }

    /// The sink actually in use. `.texture` when the handle is unknown.
    public func mode(handle: Int64) -> VideoOutputMode {
      return self.videoOutputs[handle]?.mode ?? .texture
    }
  #endif
}
