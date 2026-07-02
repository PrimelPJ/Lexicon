"""RViz visualization helpers. Turn grounded targets into a MarkerArray so an
operator can see what the robot understood the command to mean, in 3D, in the
map frame. Good visualization is underrated: it is how you debug a perception
stack and how you demo it convincingly.
"""

from __future__ import annotations

from typing import List

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Duration


def target_markers(frame_id: str, stamp, target_xyz, phrase: str,
                   score: float, marker_id: int = 0) -> MarkerArray:
    """A sphere at the target plus a text label above it."""
    arr = MarkerArray()

    sphere = Marker()
    sphere.header.frame_id = frame_id
    sphere.header.stamp = stamp
    sphere.ns = "lexicon_target"
    sphere.id = marker_id
    sphere.type = Marker.SPHERE
    sphere.action = Marker.ADD
    sphere.pose.position = Point(x=float(target_xyz[0]), y=float(target_xyz[1]),
                                 z=float(target_xyz[2]))
    sphere.pose.orientation.w = 1.0
    sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.25
    sphere.color = ColorRGBA(r=0.1, g=0.8, b=0.3, a=0.9)
    sphere.lifetime = Duration(sec=10)
    arr.markers.append(sphere)

    text = Marker()
    text.header.frame_id = frame_id
    text.header.stamp = stamp
    text.ns = "lexicon_label"
    text.id = marker_id + 1000
    text.type = Marker.TEXT_VIEW_FACING
    text.action = Marker.ADD
    text.pose.position = Point(x=float(target_xyz[0]), y=float(target_xyz[1]),
                               z=float(target_xyz[2]) + 0.35)
    text.pose.orientation.w = 1.0
    text.scale.z = 0.2
    text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
    text.text = f"{phrase} ({score:.2f})"
    text.lifetime = Duration(sec=10)
    arr.markers.append(text)

    return arr
