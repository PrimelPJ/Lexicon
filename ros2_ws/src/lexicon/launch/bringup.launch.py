"""Bring up the Lexicon stack: the lifecycle detector (auto-configured and
activated) and the FindObject action server.

For a full sim, launch this alongside your robot + Nav2, for example:
  ros2 launch nav2_bringup tb3_simulation_launch.py
  ros2 launch lexicon bringup.launch.py

The detector is a lifecycle node, so we drive it to 'active' automatically here
using an event handler. In production you might leave that to a mission manager.
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, EmitEvent
from launch.substitutions import LaunchConfiguration
from launch.events import matches_action
from launch_ros.actions import LifecycleNode, Node
from launch_ros.events.lifecycle import ChangeState
from launch_ros.event_handlers import OnStateTransition
import lifecycle_msgs.msg


def generate_launch_description():
    params = LaunchConfiguration("params_file")
    declare_params = DeclareLaunchArgument(
        "params_file",
        default_value=os.path.join(
            os.path.dirname(__file__), "..", "config", "params.yaml"),
        description="path to the params YAML",
    )

    detector = LifecycleNode(
        package="lexicon", executable="detector_node",
        name="lexicon_detector", namespace="", output="screen",
        parameters=[params],
    )

    # Auto-configure on startup.
    configure = EmitEvent(event=ChangeState(
        lifecycle_node_matcher=matches_action(detector),
        transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
    ))
    # When it reaches 'inactive', auto-activate.
    activate = RegisterEventHandler(OnStateTransition(
        target_lifecycle_node=detector, goal_state="inactive",
        entities=[EmitEvent(event=ChangeState(
            lifecycle_node_matcher=matches_action(detector),
            transition_id=lifecycle_msgs.msg.Transition.TRANSITION_ACTIVATE,
        ))],
    ))

    action_server = Node(
        package="lexicon", executable="find_object_server",
        name="lexicon_find_object", output="screen", parameters=[params],
    )

    return LaunchDescription([declare_params, detector, configure, activate, action_server])
