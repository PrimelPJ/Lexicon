"""Geometry for turning a 2D detection into a 3D goal in the map frame.

Kept dependency-free (numpy only) and free of any ROS imports so it is unit
testable on its own. The node layer feeds it depth, camera intrinsics, and the
camera-to-map transform it reads from tf2.

Pipeline:
  pixel (u, v) + depth  --intrinsics-->  point in camera frame
  point in camera frame --tf (4x4)-->    point in map frame
  point in map frame    --standoff-->    a PoseStamped the robot can navigate to
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @staticmethod
    def from_k(k) -> "CameraIntrinsics":
        """K is the row-major 3x3 from sensor_msgs/CameraInfo.k."""
        return CameraIntrinsics(fx=k[0], fy=k[4], cx=k[2], cy=k[5])


def pixel_to_camera_point(u: float, v: float, depth_m: float,
                          intr: CameraIntrinsics) -> np.ndarray:
    """Back-project a pixel to a 3D point in the optical camera frame.

    Uses the ROS optical frame convention: z forward (out of the lens), x right,
    y down. Returns [x, y, z] in metres. Depth must already be in metres.
    """
    if depth_m <= 0.0 or math.isnan(depth_m):
        raise ValueError("invalid depth")
    x = (u - intr.cx) * depth_m / intr.fx
    y = (v - intr.cy) * depth_m / intr.fy
    z = depth_m
    return np.array([x, y, z], dtype=np.float64)


def transform_point(point_cam: np.ndarray, cam_to_target: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform (camera frame -> target frame)."""
    p_h = np.array([point_cam[0], point_cam[1], point_cam[2], 1.0])
    out = cam_to_target @ p_h
    return out[:3]


def quaternion_from_yaw(yaw: float):
    """Return (x, y, z, w) for a rotation about z. For a 2D nav goal heading."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def approach_pose(target_xy: np.ndarray, robot_xy: np.ndarray, standoff_m: float = 0.6):
    """Compute a navigation goal that stops `standoff_m` short of the target and
    faces it, so the robot approaches rather than driving into the object.

    Returns (position_xy, yaw). Works in the map frame (z ignored for a 2D goal).
    """
    delta = target_xy - robot_xy
    dist = float(np.linalg.norm(delta))
    if dist < 1e-6:
        return target_xy, 0.0
    direction = delta / dist
    goal_dist = max(0.0, dist - standoff_m)
    goal_xy = robot_xy + direction * goal_dist
    yaw = math.atan2(direction[1], direction[0])
    return goal_xy, yaw


def median_depth(depth_patch: np.ndarray) -> Optional[float]:
    """Robust depth for a detection: median of valid pixels in the box interior.

    A single-pixel depth is noisy and often lands on an edge or a hole. Taking
    the median over the central patch of the bounding box rejects outliers.
    """
    valid = depth_patch[(depth_patch > 0) & np.isfinite(depth_patch)]
    if valid.size == 0:
        return None
    return float(np.median(valid))
