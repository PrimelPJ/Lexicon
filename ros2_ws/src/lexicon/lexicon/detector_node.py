"""Lifecycle-managed perception node.

Why a lifecycle (managed) node: the VLM is large and slow to load. You do not
want it resident when the robot is idle, and you want a well-defined moment to
load and release it. The lifecycle contract gives you that:

  unconfigured --configure--> inactive --activate--> active
       ^                          |                     |
       +-------- cleanup ---------+----- deactivate ----+

  on_configure : read params, set up subscriptions and the Ground service
  on_activate  : load VLM weights, enable marker publishing
  on_deactivate: unload VLM weights (free memory)
  on_cleanup   : tear everything down

The node subscribes to time-synchronized RGB, depth, and CameraInfo, caches the
latest synced frame, and answers a Ground service: given text phrases, it runs
open-vocabulary detection, back-projects the best match to 3D using depth and
intrinsics, transforms it into the map frame via tf2, and returns a PoseStamped.

Run:
  ros2 run lexicon detector_node --ros-args -p device:=cpu
  ros2 lifecycle set /lexicon_detector configure
  ros2 lifecycle set /lexicon_detector activate
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, LifecycleState
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_sensor_data

import message_filters
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray

import tf2_ros
from tf2_ros import TransformException

from lexicon_interfaces.srv import Ground
from lexicon.open_vocab_detector import OpenVocabDetector
from lexicon.grounding import (
    CameraIntrinsics, pixel_to_camera_point, transform_point, median_depth,
)
from lexicon.markers import target_markers


def transform_to_matrix(t) -> np.ndarray:
    """geometry_msgs/TransformStamped -> 4x4 homogeneous matrix."""
    q = t.transform.rotation
    tr = t.transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    # rotation matrix from quaternion
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = [tr.x, tr.y, tr.z]
    return M


class DetectorNode(LifecycleNode):
    def __init__(self):
        super().__init__("lexicon_detector")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("model_name", "google/owlvit-base-patch32")
        self.declare_parameter("score_thresh", 0.1)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("rgb_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_rect_raw")
        self.declare_parameter("info_topic", "/camera/color/camera_info")

        self._detector = None
        self._latest = None       # (rgb np.ndarray, depth np.ndarray, CameraInfo, header)
        self._tf_buffer = None
        self._marker_pub = None
        self._cb_group = ReentrantCallbackGroup()

    # ---- lifecycle transitions ---------------------------------------------

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("configuring")
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        rgb = message_filters.Subscriber(self, Image, self.get_parameter("rgb_topic").value,
                                         qos_profile=qos_profile_sensor_data)
        depth = message_filters.Subscriber(self, Image, self.get_parameter("depth_topic").value,
                                           qos_profile=qos_profile_sensor_data)
        info = message_filters.Subscriber(self, CameraInfo, self.get_parameter("info_topic").value,
                                          qos_profile=qos_profile_sensor_data)
        # Approximate time sync: the three streams share timestamps but not
        # exact ones, so allow a small slop window.
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [rgb, depth, info], queue_size=5, slop=0.1)
        self._sync.registerCallback(self._on_frame)

        self._marker_pub = self.create_lifecycle_publisher(MarkerArray, "/lexicon/markers", 5)
        self._srv = self.create_service(Ground, "ground", self._on_ground,
                                        callback_group=self._cb_group)

        self._detector = OpenVocabDetector(
            model_name=self.get_parameter("model_name").value,
            device=self.get_parameter("device").value,
            score_thresh=float(self.get_parameter("score_thresh").value),
        )
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("activating: loading VLM weights")
        self._detector.load()
        return super().on_activate(state)

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("deactivating: releasing VLM weights")
        self._detector.unload()
        return super().on_deactivate(state)

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self._latest = None
        self.destroy_service(self._srv)
        return TransitionCallbackReturn.SUCCESS

    # ---- data ----------------------------------------------------------------

    def _on_frame(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo) -> None:
        rgb = self._image_to_rgb(rgb_msg)
        depth = self._depth_to_metres(depth_msg)
        if rgb is not None and depth is not None:
            self._latest = (rgb, depth, info_msg, rgb_msg.header)

    def _on_ground(self, request, response):
        if not self._detector or not self._detector.loaded:
            response.success = False
            response.message = "detector not active"
            return response
        if self._latest is None:
            response.success = False
            response.message = "no camera frame received yet"
            return response

        rgb, depth, info, header = self._latest
        intr = CameraIntrinsics.from_k(info.k)

        dets, latency = self._detector.detect(rgb, list(request.phrases))
        if not dets:
            response.success = False
            response.message = f"no match for {list(request.phrases)}"
            return response

        # Look up camera -> map once for this frame.
        try:
            tf = self._tf_buffer.lookup_transform(
                self.get_parameter("map_frame").value,
                header.frame_id, rclpy.time.Time())
        except TransformException as e:
            response.success = False
            response.message = f"tf lookup failed: {e}"
            return response
        cam_to_map = transform_to_matrix(tf)

        poses, phrases, scores = [], [], []
        candidates = []
        for d in dets:
            u, v = d.center
            z = self._patch_depth(depth, d.box_xyxy)
            if z is None:
                continue
            p_cam = pixel_to_camera_point(u, v, z, intr)
            p_map = transform_point(p_cam, cam_to_map)
            candidates.append((d, p_map, z))

        self.get_logger().debug(
            f"Grounding: {len(candidates)}/{len(dets)} detections had valid depth")

        if not candidates:
            response.success = False
            response.message = "matched in image but no valid depth on target"
            return response

        # nearest -> smallest depth; otherwise highest score (already sorted).
        if request.nearest:
            candidates.sort(key=lambda c: c[2])

        for d, p_map, z in candidates:
            ps = PoseStamped()
            ps.header.frame_id = self.get_parameter("map_frame").value
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.pose.position.x = float(p_map[0])
            ps.pose.position.y = float(p_map[1])
            ps.pose.position.z = float(p_map[2])
            ps.pose.orientation.w = 1.0
            poses.append(ps)
            phrases.append(d.phrase)
            scores.append(d.score)

        best = candidates[0]
        self._marker_pub.publish(target_markers(
            self.get_parameter("map_frame").value, self.get_clock().now().to_msg(),
            best[1], best[0].phrase, best[0].score))

        response.success = True
        response.poses = poses
        response.phrases = phrases
        response.scores = scores
        response.message = f"grounded {len(poses)} candidate(s) in {latency:.0f} ms"
        return response

    # ---- conversions ---------------------------------------------------------

    def _patch_depth(self, depth: np.ndarray, box) -> float | None:
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = depth.shape[:2]
        # central 50 percent of the box, clamped to the image
        bw, bh = x2 - x1, y2 - y1
        cx1 = max(0, int(x1 + bw * 0.25)); cx2 = min(w, int(x2 - bw * 0.25))
        cy1 = max(0, int(y1 + bh * 0.25)); cy2 = min(h, int(y2 - bh * 0.25))
        if cx2 <= cx1 or cy2 <= cy1:
            return None
        return median_depth(depth[cy1:cy2, cx1:cx2])

    @staticmethod
    def _image_to_rgb(msg: Image):
        if msg.encoding not in ("rgb8", "bgr8"):
            return None
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        if msg.encoding == "bgr8":
            arr = arr[:, :, ::-1]
        return np.ascontiguousarray(arr)

    @staticmethod
    def _depth_to_metres(msg: Image):
        if msg.encoding == "16UC1":
            d = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            return d.astype(np.float32) / 1000.0  # mm -> m
        if msg.encoding == "32FC1":
            return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
        return None


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
