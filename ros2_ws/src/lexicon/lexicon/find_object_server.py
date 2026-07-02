"""FindObject action server: the orchestrator.

Flow for one goal ("go to the red chair"):
  1. Parse the instruction into open-vocab phrases (command_parser).
  2. Call the detector node's Ground service to get 3D candidate poses.
  3. Pick a candidate, compute an approach pose that stops short and faces it.
  4. Send a NavigateToPose goal to Nav2 and relay its progress as feedback.
  5. Return success with the final pose, or a useful failure message.

Why an action, not a service: navigation takes seconds to minutes, needs live
feedback, and must be cancelable. That is exactly what ROS2 actions are for.

Concurrency: this node is a service client and an action client while also
being an action server, so every callback runs in a ReentrantCallbackGroup
under a MultiThreadedExecutor. Blocking on a future inside a single-threaded
executor would deadlock; this is the standard way to compose async ROS2 calls.
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.action import ActionServer, ActionClient, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from tf2_ros import Buffer, TransformListener, TransformException

from lexicon_interfaces.action import FindObject
from lexicon_interfaces.srv import Ground
from lexicon.command_parser import parse_instruction, wants_nearest
from lexicon.grounding import approach_pose, quaternion_from_yaw


class FindObjectServer(Node):
    def __init__(self):
        super().__init__("lexicon_find_object")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("default_standoff", 0.6)

        self._cb = ReentrantCallbackGroup()
        self._ground = self.create_client(Ground, "ground", callback_group=self._cb)
        self._nav = ActionClient(self, NavigateToPose, "navigate_to_pose",
                                 callback_group=self._cb)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._server = ActionServer(
            self, FindObject, "find_object",
            execute_callback=self._execute,
            goal_callback=lambda g: GoalResponse.ACCEPT,
            cancel_callback=lambda c: CancelResponse.ACCEPT,
            callback_group=self._cb,
        )
        self.get_logger().info("find_object action server ready")

    async def _execute(self, goal_handle):
        instruction = goal_handle.request.instruction
        standoff = goal_handle.request.standoff or float(
            self.get_parameter("default_standoff").value)
        self.get_logger().info(f"instruction: {instruction!r}")

        result = FindObject.Result()

        # 1. parse
        phrases = parse_instruction(instruction)
        if not phrases:
            goal_handle.abort()
            result.success = False
            result.message = "could not extract a target from the instruction"
            return result
        self._feedback(goal_handle, f"looking for: {', '.join(phrases)}", -1.0)

        # 2. ground via the detector service
        if not self._ground.wait_for_service(timeout_sec=5.0):
            goal_handle.abort()
            result.success = False
            result.message = "detector Ground service unavailable (is it active?)"
            return result

        req = Ground.Request()
        req.phrases = phrases
        req.nearest = wants_nearest(instruction)
        ground_resp = await self._ground.call_async(req)

        if not ground_resp.success or not ground_resp.poses:
            goal_handle.abort()
            result.success = False
            result.message = ground_resp.message or "target not found"
            return result

        target = ground_resp.poses[0]
        self._feedback(
            goal_handle,
            f"found {ground_resp.phrases[0]} ({ground_resp.scores[0]:.2f}), navigating", -1.0)

        # 3. approach pose (stop short, face target)
        nav_goal = self._make_approach_goal(target, standoff)
        if nav_goal is None:
            goal_handle.abort()
            result.success = False
            result.message = "could not determine robot pose for approach"
            return result

        # 4. drive there via Nav2, relaying feedback
        if not self._nav.wait_for_server(timeout_sec=5.0):
            goal_handle.abort()
            result.success = False
            result.message = "Nav2 navigate_to_pose action server unavailable"
            return result

        nav_req = NavigateToPose.Goal()
        nav_req.pose = nav_goal
        send_future = await self._nav.send_goal_async(
            nav_req, feedback_callback=lambda fb: self._relay_nav_feedback(goal_handle, fb))
        if not send_future.accepted:
            goal_handle.abort()
            result.success = False
            result.message = "Nav2 rejected the goal"
            return result

        nav_result = await send_future.get_result_async()

        # 5. done
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result.success = False
            result.message = "canceled"
            return result

        goal_handle.succeed()
        result.success = True
        result.final_pose = nav_goal
        result.message = f"arrived near {ground_resp.phrases[0]}"
        return result

    # ---- helpers -------------------------------------------------------------

    def _make_approach_goal(self, target: PoseStamped, standoff: float):
        try:
            tf = self._tf_buffer.lookup_transform(
                self.get_parameter("map_frame").value,
                self.get_parameter("base_frame").value, rclpy.time.Time())
        except TransformException:
            return None
        robot_xy = np.array([tf.transform.translation.x, tf.transform.translation.y])
        target_xy = np.array([target.pose.position.x, target.pose.position.y])
        goal_xy, yaw = approach_pose(target_xy, robot_xy, standoff)

        goal = PoseStamped()
        goal.header.frame_id = self.get_parameter("map_frame").value
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = float(goal_xy[0])
        goal.pose.position.y = float(goal_xy[1])
        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        goal.pose.orientation.x = qx
        goal.pose.orientation.y = qy
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw
        return goal

    def _relay_nav_feedback(self, goal_handle, fb):
        remaining = float(fb.feedback.distance_remaining)
        self._feedback(goal_handle, "navigating", remaining)

    def _feedback(self, goal_handle, status: str, distance_remaining: float):
        fb = FindObject.Feedback()
        fb.status = status
        fb.distance_remaining = distance_remaining
        goal_handle.publish_feedback(fb)


def main(args=None):
    rclpy.init(args=args)
    node = FindObjectServer()
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
