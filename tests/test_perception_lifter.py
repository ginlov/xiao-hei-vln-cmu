"""Tests for ``xiao_hei_vln.perception.lifter``."""

from __future__ import annotations

import math

import numpy as np
import pytest

from xiao_hei_vln.messages.common import Quaternion, Vector3
from xiao_hei_vln.perception.geometry import (
    EQUIRECT_H,
    EQUIRECT_W,
    project_camera_points_to_equirect,
    sensor_to_camera_transform,
)
from xiao_hei_vln.perception.lifter import (
    DEFAULT_MIN_INLIERS,
    LiftResult,
    PointLifter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity_pose() -> tuple[Vector3, Quaternion]:
    """Robot at the map origin, no rotation. Means map frame ≡ sensor frame."""
    return (
        Vector3(x=0.0, y=0.0, z=0.0),
        Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )


def _full_mask() -> np.ndarray:
    """Equirect mask that covers every pixel (no FOV filtering)."""
    return np.ones((EQUIRECT_H, EQUIRECT_W), dtype=bool)


def _empty_mask() -> np.ndarray:
    return np.zeros((EQUIRECT_H, EQUIRECT_W), dtype=bool)


def _scatter_around(center: Vector3, n: int, spread: float = 0.05) -> np.ndarray:
    """Generate ``n`` LiDAR points clustered around ``center`` in map coords."""
    rng = np.random.default_rng(0)
    deltas = rng.normal(0, spread, size=(n, 3))
    pts = np.column_stack([
        deltas[:, 0] + center.x,
        deltas[:, 1] + center.y,
        deltas[:, 2] + center.z,
    ])
    # Append a dummy intensity column to exercise the >=4 column path.
    intensity = np.zeros((n, 1))
    return np.concatenate([pts, intensity], axis=1)


def _project_map_point_to_equirect_pixel(
    map_point: Vector3, pose_position: Vector3, pose_orientation: Quaternion,
) -> tuple[int, int]:
    """Mirror the lifter's projection for a single point — used in tests
    to construct masks that *actually cover* the cluster's projection."""
    from xiao_hei_vln.perception.lifter import _rotation_from_quaternion

    R_ms = _rotation_from_quaternion(pose_orientation)
    t = np.array([pose_position.x, pose_position.y, pose_position.z])
    p_map = np.array([[map_point.x, map_point.y, map_point.z]])
    p_sensor = (p_map - t) @ R_ms
    R_sc, t_sc = sensor_to_camera_transform()
    p_cam = p_sensor @ R_sc.T + t_sc
    u, v, valid = project_camera_points_to_equirect(p_cam)
    assert valid[0], "test setup error: point falls outside equirect FOV"
    return int(round(u[0])) % EQUIRECT_W, int(round(v[0]))


# ---------------------------------------------------------------------------
# 1. Inlier count + median XYZ
# ---------------------------------------------------------------------------


class TestLiftMedian:
    def test_empty_scan_returns_none(self) -> None:
        lifter = PointLifter()
        scan = np.zeros((0, 4), dtype=np.float64)
        pose_p, pose_q = _identity_pose()
        result = lifter.lift(_full_mask(), scan, pose_p, pose_q)
        assert isinstance(result, LiftResult)
        assert result.position is None
        assert result.n_inliers == 0

    def test_below_min_inliers_returns_none(self) -> None:
        lifter = PointLifter(min_inliers=20)
        scan = _scatter_around(Vector3(x=2.0, y=0.0, z=0.0), n=5)
        pose_p, pose_q = _identity_pose()
        result = lifter.lift(_full_mask(), scan, pose_p, pose_q)
        assert result.position is None
        # All 5 points project inside the full mask, but n < min_inliers.
        assert result.n_inliers == 5

    def test_above_min_inliers_returns_median(self) -> None:
        lifter = PointLifter(min_inliers=DEFAULT_MIN_INLIERS)
        center = Vector3(x=3.0, y=-1.0, z=0.5)
        scan = _scatter_around(center, n=100, spread=0.1)
        pose_p, pose_q = _identity_pose()
        result = lifter.lift(_full_mask(), scan, pose_p, pose_q)
        assert result.position is not None
        # Median of a Gaussian cluster ≈ its center.
        assert result.position.x == pytest.approx(center.x, abs=0.05)
        assert result.position.y == pytest.approx(center.y, abs=0.05)
        assert result.position.z == pytest.approx(center.z, abs=0.05)
        assert result.n_inliers == 100

    def test_empty_mask_returns_none(self) -> None:
        lifter = PointLifter()
        scan = _scatter_around(Vector3(x=2.0, y=0.0, z=0.0), n=100)
        pose_p, pose_q = _identity_pose()
        result = lifter.lift(_empty_mask(), scan, pose_p, pose_q)
        assert result.position is None
        # Inliers measured after FOV/mask filtering, so this is 0.
        assert result.n_inliers == 0


# ---------------------------------------------------------------------------
# 2. Mask filtering uses the right projection
# ---------------------------------------------------------------------------


class TestMaskFiltering:
    def test_pixel_disc_mask_isolates_the_right_cluster(self) -> None:
        """Two clusters at different bearings; mask covers only one →
        we should only see that one's median."""
        lifter = PointLifter(min_inliers=DEFAULT_MIN_INLIERS)
        pose_p, pose_q = _identity_pose()

        forward = Vector3(x=3.0, y=0.0, z=0.0)
        rightward = Vector3(x=0.0, y=-3.0, z=0.0)

        scan = np.concatenate([
            _scatter_around(forward, n=100, spread=0.05),
            _scatter_around(rightward, n=100, spread=0.05),
        ], axis=0)

        # Mask: 40-px disc around where the *forward* cluster projects.
        u_fwd, v_fwd = _project_map_point_to_equirect_pixel(
            forward, pose_p, pose_q,
        )
        mask = np.zeros((EQUIRECT_H, EQUIRECT_W), dtype=bool)
        yy, xx = np.ogrid[:EQUIRECT_H, :EQUIRECT_W]
        # Account for horizontal wrap.
        dx = np.minimum(np.abs(xx - u_fwd), EQUIRECT_W - np.abs(xx - u_fwd))
        dy = yy - v_fwd
        mask[(dx * dx + dy * dy) < 40 * 40] = True

        result = lifter.lift(mask, scan, pose_p, pose_q)
        assert result.position is not None
        # The forward cluster should dominate the median.
        assert result.position.x == pytest.approx(forward.x, abs=0.5)
        assert result.position.y == pytest.approx(forward.y, abs=0.5)


# ---------------------------------------------------------------------------
# 3. Pose transform — moving the robot moves the inferred direction
# ---------------------------------------------------------------------------


class TestPoseTransform:
    def test_rotated_pose_changes_projection_pixel(self) -> None:
        """Sanity: when the robot yaws 90°, a point that was at the
        front projects to a different equirect column."""
        forward = Vector3(x=3.0, y=0.0, z=0.0)

        # Pose 1: identity → forward projects somewhere central.
        u1, _ = _project_map_point_to_equirect_pixel(
            forward,
            Vector3(x=0.0, y=0.0, z=0.0),
            Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )

        # Pose 2: yaw +90° around +z → the same forward point now lies
        # at the robot's right; equirect column should shift by ≈ W/4.
        yaw = math.pi / 2
        u2, _ = _project_map_point_to_equirect_pixel(
            forward,
            Vector3(x=0.0, y=0.0, z=0.0),
            Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2)),
        )
        # Pixel shift modulo wrap; should be ≈ W/4 = 480. The slack
        # absorbs the 0.1 m camera-above-LiDAR parallax which biases
        # the projection by ~1.9° at 3 m range (~10 px on equirect).
        diff = abs(u1 - u2)
        diff = min(diff, EQUIRECT_W - diff)
        assert abs(diff - EQUIRECT_W // 4) <= 16

    def test_translated_pose_inferred_position_stays_in_map_frame(self) -> None:
        """When the robot is at (5, 0) instead of origin, lifting a
        cluster centred at (10, 0) in the map frame should still
        return ~(10, 0). The lifter must un-translate the LiDAR scan
        before projecting."""
        lifter = PointLifter(min_inliers=DEFAULT_MIN_INLIERS)
        center = Vector3(x=10.0, y=0.0, z=0.0)
        scan = _scatter_around(center, n=100, spread=0.05)
        pose_p = Vector3(x=5.0, y=0.0, z=0.0)
        pose_q = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        result = lifter.lift(_full_mask(), scan, pose_p, pose_q)
        assert result.position is not None
        assert result.position.x == pytest.approx(10.0, abs=0.05)
        assert result.position.y == pytest.approx(0.0, abs=0.05)


# ---------------------------------------------------------------------------
# 4. Defensive input validation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 5. inlier_points payload (for ObjectMap fusion)
# ---------------------------------------------------------------------------


class TestInlierPoints:
    def test_inlier_points_match_position_and_count(self) -> None:
        lifter = PointLifter(min_inliers=DEFAULT_MIN_INLIERS)
        center = Vector3(x=3.0, y=-1.0, z=0.5)
        scan = _scatter_around(center, n=100, spread=0.1)
        pose_p, pose_q = _identity_pose()
        result = lifter.lift(_full_mask(), scan, pose_p, pose_q)
        assert result.inlier_points is not None
        assert result.inlier_points.shape == (result.n_inliers, 3)
        # The committed position is exactly the median of the returned points.
        med = np.median(result.inlier_points, axis=0)
        assert result.position.x == pytest.approx(float(med[0]))
        assert result.position.y == pytest.approx(float(med[1]))
        assert result.position.z == pytest.approx(float(med[2]))

    def test_inlier_points_none_when_no_position(self) -> None:
        lifter = PointLifter(min_inliers=20)
        scan = _scatter_around(Vector3(x=2.0, y=0.0, z=0.0), n=5)
        pose_p, pose_q = _identity_pose()
        result = lifter.lift(_full_mask(), scan, pose_p, pose_q)
        assert result.position is None
        assert result.inlier_points is None


# ---------------------------------------------------------------------------
# 6. z-buffer occlusion gate
# ---------------------------------------------------------------------------


class TestZBuffer:
    def _two_depth_scan(self):
        """A near cluster and a far cluster along the SAME +x bearing, so
        they project onto the same equirect pixels at different ranges."""
        near = _scatter_around(Vector3(x=2.0, y=0.0, z=0.0), n=100, spread=0.03)
        far = _scatter_around(Vector3(x=6.0, y=0.0, z=0.0), n=100, spread=0.03)
        return np.concatenate([near, far], axis=0)

    def test_disabled_keeps_both_surfaces(self) -> None:
        lifter = PointLifter(min_inliers=DEFAULT_MIN_INLIERS, enable_zbuffer=False)
        pose_p, pose_q = _identity_pose()
        result = lifter.lift(_full_mask(), self._two_depth_scan(), pose_p, pose_q)
        assert result.position is not None
        # All 200 points project inside the full mask; both clusters
        # contribute → median sits between 2 and 6.
        assert result.n_inliers == 200
        assert result.position.x > 3.5

    def test_enabled_drops_the_occluded_far_surface(self) -> None:
        lifter = PointLifter(min_inliers=DEFAULT_MIN_INLIERS, enable_zbuffer=True)
        pose_p, pose_q = _identity_pose()
        result = lifter.lift(_full_mask(), self._two_depth_scan(), pose_p, pose_q)
        assert result.position is not None
        # The near surface occludes much of the far cluster along shared
        # bearings → fewer inliers and the median snaps to the near cluster.
        assert result.n_inliers < 200
        assert result.position.x == pytest.approx(2.0, abs=0.3)


class TestValidation:
    def test_wrong_mask_shape_raises(self) -> None:
        lifter = PointLifter()
        pose_p, pose_q = _identity_pose()
        with pytest.raises(ValueError, match="mask must be"):
            lifter.lift(
                np.zeros((100, 100), dtype=bool),
                np.zeros((5, 4)), pose_p, pose_q,
            )

    def test_wrong_scan_shape_raises(self) -> None:
        lifter = PointLifter()
        pose_p, pose_q = _identity_pose()
        with pytest.raises(ValueError, match="scan_points_map must be"):
            lifter.lift(_full_mask(), np.zeros((5, 2)), pose_p, pose_q)
