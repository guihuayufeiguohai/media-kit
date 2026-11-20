public class VideoOutputConfiguration {
  public let width: Int64?
  public let height: Int64?
  public let enableHardwareAcceleration: Bool
  /// Which sink the frames should go to. Honoured on iOS only; C1 always
  /// falls back to `.texture` inside `VideoOutput._init`.
  public let outputMode: VideoOutputMode
  /// 16F intermediate (kCVPixelFormatType_64RGBAHalf) instead of 8-bit BGRA.
  /// Only meaningful together with `.metalLayer` -- Flutter's external texture
  /// path rejects anything but BGRA8.
  public let pixelFormat16F: Bool

  init(
    width: Int64?,
    height: Int64?,
    enableHardwareAcceleration: Bool,
    outputMode: VideoOutputMode = .texture,
    pixelFormat16F: Bool = false
  ) {
    self.width = width
    self.height = height
    self.enableHardwareAcceleration = enableHardwareAcceleration
    self.outputMode = outputMode
    self.pixelFormat16F = pixelFormat16F
  }

  public static func fromDict(_ dict: [String: Any])
    -> VideoOutputConfiguration
  {
    // These used to be `as!`. A Dart build that omitted one key (or a new
    // key added on the Dart side first) crashed the app on the platform
    // thread, which is impossible to catch -- parse defensively instead.
    let widthStr = dict["width"] as? String
    let heightStr = dict["height"] as? String
    let enableHardwareAcceleration =
      dict["enableHardwareAcceleration"] as? Bool ?? true
    let outputMode =
      (dict["outputMode"] as? String).flatMap(VideoOutputMode.init(rawValue:))
      ?? .texture
    let pixelFormat16F = dict["pixelFormat16F"] as? Bool ?? false

    let width: Int64? = widthStr.flatMap { Int64($0) }
    let height: Int64? = heightStr.flatMap { Int64($0) }

    return VideoOutputConfiguration(
      width: width,
      height: height,
      enableHardwareAcceleration: enableHardwareAcceleration,
      outputMode: outputMode,
      pixelFormat16F: pixelFormat16F
    )
  }
}
