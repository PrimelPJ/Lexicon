"""Unit tests for the grounding geometry. Pure numpy, no ROS needed:
    pytest test/test_grounding.py
"""
import math
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lexicon.grounding import (
    CameraIntrinsics, pixel_to_camera_point, transform_point,
    approach_pose, median_depth, quaternion_from_yaw,
)


def test_pixel_at_principal_point_is_straight_ahead():
    intr = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240)
    p = pixel_to_camera_point(320, 240, 2.0, intr)
    assert np.allclose(p, [0.0, 0.0, 2.0])


def test_pixel_offset_projects_correctly():
    intr = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240)
    p = pixel_to_camera_point(420, 240, 2.0, intr)  # 100px right
    assert math.isclose(p[0], (100 * 2.0) / 500)
    assert math.isclose(p[2], 2.0)


def test_from_k_parses_row_major():
    k = [500, 0, 320, 0, 500, 240, 0, 0, 1]
    intr = CameraIntrinsics.from_k(k)
    assert (intr.fx, intr.fy, intr.cx, intr.cy) == (500, 500, 320, 240)


def test_identity_transform_is_noop():
    p = np.array([1.0, 2.0, 3.0])
    assert np.allclose(transform_point(p, np.eye(4)), p)


def test_translation_transform():
    T = np.eye(4)
    T[:3, 3] = [10, 0, 0]
    assert np.allclose(transform_point(np.array([1.0, 0, 0]), T), [11, 0, 0])


def test_approach_stops_short_and_faces_target():
    goal, yaw = approach_pose(np.array([5.0, 0.0]), np.array([0.0, 0.0]), standoff_m=1.0)
    assert math.isclose(goal[0], 4.0)          # 1m short of the 5m target
    assert math.isclose(yaw, 0.0)              # facing +x


def test_approach_faces_diagonal():
    goal, yaw = approach_pose(np.array([1.0, 1.0]), np.array([0.0, 0.0]), standoff_m=0.0)
    assert math.isclose(yaw, math.pi / 4, abs_tol=1e-6)


def test_median_depth_rejects_zeros_and_nans():
    patch = np.array([[0.0, 2.0, np.nan], [2.1, 0.0, 1.9]])
    assert math.isclose(median_depth(patch), 2.0)


def test_median_depth_all_invalid_returns_none():
    patch = np.zeros((3, 3))
    assert median_depth(patch) is None


def test_quaternion_yaw_zero_is_identity():
    x, y, z, w = quaternion_from_yaw(0.0)
    assert (x, y, z, w) == (0.0, 0.0, 0.0, 1.0)
