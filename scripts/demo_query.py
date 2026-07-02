"""Send a FindObject goal to the Lexicon action server from the command line.

Usage (with the stack running):
  ros2 run lexicon ... (bringup)   # in one terminal
  python demo_query.py "go to the red chair"

This is a thin ActionClient. It prints live feedback and the final result, which
is what you would show while narrating a demo.
"""

import sys
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from lexicon_interfaces.action import FindObject


class QueryClient(Node):
    def __init__(self):
        super().__init__("lexicon_demo_query")
        self._client = ActionClient(self, FindObject, "find_object")

    def send(self, instruction: str):
        self._client.wait_for_server()
        goal = FindObject.Goal()
        goal.instruction = instruction
        goal.standoff = 0.6
        fut = self._client.send_goal_async(goal, feedback_callback=self._on_feedback)
        rclpy.spin_until_future_complete(self, fut)
        handle = fut.result()
        if not handle.accepted:
            self.get_logger().error("goal rejected")
            return
        result_fut = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_fut)
        res = result_fut.result().result
        self.get_logger().info(f"success={res.success} message={res.message!r}")

    def _on_feedback(self, fb):
        f = fb.feedback
        d = f"{f.distance_remaining:.2f}m" if f.distance_remaining >= 0 else "-"
        self.get_logger().info(f"[{f.status}] remaining={d}")


def main():
    instruction = " ".join(sys.argv[1:]) or "go to the nearest chair"
    rclpy.init()
    node = QueryClient()
    try:
        node.send(instruction)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
