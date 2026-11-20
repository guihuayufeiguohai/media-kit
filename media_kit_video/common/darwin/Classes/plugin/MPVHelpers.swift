import Darwin
import Foundation

#if SWIFT_PACKAGE
  import Mpv
#endif

public enum MPVHelpers {
  public static func checkError(_ status: CInt) {
    if status < 0 {
      NSLog("MPVHelpers: error: \(String(cString: mpv_error_string(status)))")
      exit(1)
    }
  }

  /// Reads a string property, returning nil if mpv does not have one yet.
  ///
  /// Used by the layer sink to find out what transfer function the VO is
  /// really producing before tagging the CAMetalLayer.
  public static func getStringProperty(
    _ handle: OpaquePointer,
    _ name: String
  ) -> String? {
    guard let raw = mpv_get_property_string(handle, name) else { return nil }
    defer { mpv_free(raw) }
    let value = String(cString: raw)
    return value.isEmpty ? nil : value
  }

  public static func getVideoOutParams(
    _ handle: OpaquePointer
  ) -> MPVVideoOutParams {
    var node = mpv_node()
    defer {
      mpv_free_node_contents(&node)
    }

    mpv_get_property(handle, "video-out-params", MPV_FORMAT_NODE, &node)

    if node.format != MPV_FORMAT_NODE_MAP {
      return MPVVideoOutParams.empty
    }

    let map: mpv_node_list = node.u.list!.pointee
    if map.num == 0 {
      return MPVVideoOutParams.empty
    }

    return MPVVideoOutParams.fromMPVNodeList(map)
  }
}
