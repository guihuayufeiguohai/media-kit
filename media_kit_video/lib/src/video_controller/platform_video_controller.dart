/// This file is a part of media_kit (https://github.com/media-kit/media-kit).
///
/// Copyright © 2021 & onwards, Hitesh Kumar Saini <saini123hitesh@gmail.com>.
/// All rights reserved.
/// Use of this source code is governed by MIT license that can be found in the LICENSE file.
import 'dart:async';
import 'package:flutter/widgets.dart';

import 'package:media_kit/media_kit.dart';

import 'package:media_kit_video/src/video_controller/video_controller.dart';

/// Where rendered frames are delivered.
enum VideoOutputMode {
  /// FlutterTexture, registered through `FlutterTextureRegistry`.
  ///
  /// The only sink that works everywhere, and the only one the Flutter engine
  /// accepts for external textures today (32BGRA / 420v / 420f).
  texture,

  /// PlatformView + `CAMetalLayer`. iOS 16+ only.
  ///
  /// Native is the final arbiter: iOS 15, a missing Vulkan renderer, or a build
  /// where the layer sink is not wired yet all fall back to [texture]. The mode
  /// actually in use is reported through [PlatformVideoController.mode].
  metalLayer,
}

/// Wire format for [VideoOutputMode] over the method channel.
extension VideoOutputModeWire on VideoOutputMode {
  String get wire =>
      this == VideoOutputMode.metalLayer ? 'metalLayer' : 'texture';

  static VideoOutputMode parse(String? value) =>
      value == 'metalLayer' ? VideoOutputMode.metalLayer : VideoOutputMode.texture;
}

/// {@template platform_video_controller}
///
/// PlatformVideoController
/// -----------------------
///
/// This class provides the interface for platform specific [VideoController] implementations.
/// The platform specific implementations are expected to implement the methods accordingly.
///
/// The subclasses are then used in composition with the [VideoController] class, based on the platform the application is running on.
///
/// {@endtemplate}
abstract class PlatformVideoController {
  /// The [Player] instance associated with this instance.
  final Player player;

  /// User defined configuration for [VideoController].
  final VideoControllerConfiguration configuration;

  /// Texture ID of the video output, registered with Flutter engine by the native implementation.
  final ValueNotifier<int?> id = ValueNotifier<int?>(null);

  /// [Rect] of the video output, received from the native implementation.
  final ValueNotifier<Rect?> rect = ValueNotifier<Rect?>(null);

  /// The sink actually in use, as reported by the native implementation.
  ///
  /// Native is the final arbiter, so this can differ from
  /// [VideoControllerConfiguration.outputMode]: asking for
  /// [VideoOutputMode.metalLayer] on iOS 15 (or on a build where the layer sink
  /// is not wired yet) yields [VideoOutputMode.texture] here. The widget must
  /// branch on this value, never on the configuration.
  final ValueNotifier<VideoOutputMode> mode =
      ValueNotifier<VideoOutputMode>(VideoOutputMode.texture);

  /// The mpv handle this output belongs to.
  ///
  /// Passed to the PlatformView as `creationParams` so the view can find its
  /// [VideoOutput] in the native manager without creating a second renderer.
  int? handle;

  /// {@macro platform_video_controller}
  PlatformVideoController(
    this.player,
    this.configuration,
  );

  /// Sets the required size of the video output.
  /// This may yield substantial performance improvements if a small [width] & [height] is specified.
  ///
  /// Remember:
  /// * “Premature optimization is the root of all evil”
  /// * “With great power comes great responsibility”
  Future<void> setSize({
    int? width,
    int? height,
  });

  /// A [Future] that completes when the first video frame has been rendered.
  Future<void> get waitUntilFirstFrameRendered =>
      waitUntilFirstFrameRenderedCompleter.future;

  /// [Completer] used to signal the decoding & rendering of the first video frame.
  /// Use [waitUntilFirstFrameRendered] to wait for the first frame to be rendered.
  @protected
  final waitUntilFirstFrameRenderedCompleter = Completer<void>();

  void dispose() {
    id.dispose();
    rect.dispose();
    mode.dispose();
  }
}

/// {@template video_controller_configuration}
///
/// VideoControllerConfiguration
/// ----------------------------
/// Configurable options for customizing the [VideoController] behavior.
///
/// {@endtemplate}
class VideoControllerConfiguration {
  /// Sets the [`--vo`](https://mpv.io/manual/stable/#options-vo) property on native backend.
  ///
  /// Default: Platform specific.
  /// * Windows, GNU/Linux, macOS & iOS: `libmpv`
  /// * Android: `gpu`
  final String? vo;

  /// Sets the [`--hwdec`](https://mpv.io/manual/stable/#options-hwdec) property on native backend.
  ///
  /// Default: Platform specific.
  /// * Windows, GNU/Linux, macOS & iOS : `auto`
  /// * Android: `auto-safe`
  final String? hwdec;

  /// The scale for the video output.
  /// This may be used for performance reasons. Specifying this option will cause [width] & [height] to be ignored.
  ///
  /// Default: `1.0`
  final double scale;

  /// The fixed width for the video output.
  /// This may be used for performance reasons.
  ///
  /// Default: `null`
  final int? width;

  /// The fixed height for the video output.
  /// This may be used for performance reasons.
  ///
  /// Default: `null`
  final int? height;

  /// Whether to enable hardware acceleration.
  ///
  /// DO NOT DISABLE THIS OPTION MEANINGLESSLY.
  /// THE BATTERY WILL DRAIN, THE DEVICE MAY HEAT UP & CPU USAGE WILL BE HIGH.
  ///
  /// Default: `true`
  final bool enableHardwareAcceleration;

  /// Whether to attach `android.view.Surface` after video parameters are known.
  ///
  /// Default:
  /// * [vo] == gpu : `true`
  /// * [vo] != gpu : `false`
  final bool? androidAttachSurfaceAfterVideoParameters;

  /// Which sink the frames should go to.
  ///
  /// iOS only; ignored on every other platform. Native is the final arbiter --
  /// see [PlatformVideoController.mode] for the value actually in use.
  ///
  /// Default: [VideoOutputMode.texture]
  final VideoOutputMode outputMode;

  /// Use a 16-bit float intermediate (`kCVPixelFormatType_64RGBAHalf`) instead
  /// of 8-bit BGRA.
  ///
  /// iOS + [outputMode] == [VideoOutputMode.metalLayer] only. Must stay false
  /// otherwise: Flutter's external texture path rejects anything but BGRA8 with
  /// "Unsupported pixel format". Native ignores it when the layer sink is not
  /// selected.
  ///
  /// Default: `false`
  final bool pixelFormat16F;

  /// Present in HDR instead of tone mapping down to SDR.
  ///
  /// iOS + [outputMode] == [VideoOutputMode.metalLayer] only, and it implies
  /// [pixelFormat16F]: an 8-bit drawable has no room for a PQ signal. Native
  /// additionally requires a PQ capable `CAMetalLayer` colorspace, and falls
  /// back to SDR if that is unavailable.
  ///
  /// When honoured, mpv stops tone mapping to `bt.709/srgb` and writes PQ
  /// (`bt.2020/pq`) straight into the IOSurface, so highlights reach the
  /// display's real peak instead of being compressed into 135 nits.
  ///
  /// Default: `false`
  final bool outputEDR;

  /// {@macro video_controller_configuration}
  const VideoControllerConfiguration({
    this.vo,
    this.hwdec,
    this.width,
    this.height,
    this.scale = 1.0,
    this.enableHardwareAcceleration = true,
    this.androidAttachSurfaceAfterVideoParameters,
    this.outputMode = VideoOutputMode.texture,
    this.pixelFormat16F = false,
    this.outputEDR = false,
  });

  /// Returns a copy of this class with the given fields replaced by the new values.
  VideoControllerConfiguration copyWith({
    String? vo,
    String? hwdec,
    double? scale,
    int? width,
    int? height,
    bool? enableHardwareAcceleration,
    bool? androidAttachSurfaceAfterVideoParameters,
    VideoOutputMode? outputMode,
    bool? pixelFormat16F,
    bool? outputEDR,
  }) =>
      VideoControllerConfiguration(
        vo: vo ?? this.vo,
        hwdec: hwdec ?? this.hwdec,
        scale: scale ?? this.scale,
        width: width ?? this.width,
        height: height ?? this.height,
        enableHardwareAcceleration:
            enableHardwareAcceleration ?? this.enableHardwareAcceleration,
        androidAttachSurfaceAfterVideoParameters:
            androidAttachSurfaceAfterVideoParameters ??
                this.androidAttachSurfaceAfterVideoParameters,
        outputMode: outputMode ?? this.outputMode,
        pixelFormat16F: pixelFormat16F ?? this.pixelFormat16F,
        outputEDR: outputEDR ?? this.outputEDR,
      );
}
