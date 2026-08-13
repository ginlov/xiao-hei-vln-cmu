"""Tests for ``xiao_hei_vln.perception.scan_accumulator``."""

from __future__ import annotations

import math

import numpy as np
import pytest

from xiao_hei_vln.messages.common import Quaternion, Vector3
from xiao_hei_vln.perception.scan_accumulator import (
    ScanAccumulator,
    voxel_downsample,
)


def _pos(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Vector3:
    return Vector3(x=x, y=y, z=z)


def _quat_yaw(yaw: float) -> Quaternion:
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))


def _grid(n_side: int, step: float = 0.2, origin=(0.0, 0.0, 0.0)) -> np.ndarray:
    """A regular grid of points, each in its own ``> voxel`` cell."""
    ox, oy, oz = origin
    coords = np.arange(n_side) * step           # e.g. [0, 0.2, 0.4, ...]
    pts = np.array([[ox + i, oy + j, oz]
                    for i in coords for j in coords], dtype=np.float64)
    return pts


# ── buffer policy ─────────────────────────────────────────────────────────────

def test_first_update_stores_keyframe_and_returns_points():
    acc = ScanAccumulator()
    out = acc.update(_grid(5), _pos(0, 0), _quat_yaw(0.0))
    assert acc.n_keyframes == 1
    assert out.shape[0] > 0 and out.shape[1] == 3


def test_every_update_stores_a_keyframe():
    """There is no motion gate: a stationary tick still commits a keyframe.

    This is exactly what the old move/rotation gate denied, so it is asserted
    directly rather than left implicit.
    """
    acc = ScanAccumulator()
    acc.update(_grid(5), _pos(0, 0), _quat_yaw(0.0))
    acc.update(_grid(5), _pos(0, 0), _quat_yaw(0.0))          # identical pose
    acc.update(_grid(5), _pos(0.01, 0.0), _quat_yaw(0.0))     # 1 cm
    assert acc.n_keyframes == 3


def test_pose_arguments_do_not_affect_the_buffer():
    """Pose is accepted but unused — two runs differing only in pose agree."""
    moved, still = ScanAccumulator(), ScanAccumulator()
    for i in range(4):
        a = moved.update(_grid(3), _pos(i * 5.0, 0.0), _quat_yaw(i * 1.0))
        b = still.update(_grid(3), _pos(0, 0), _quat_yaw(0.0))
    assert moved.n_keyframes == still.n_keyframes
    assert np.array_equal(np.sort(a, axis=0), np.sort(b, axis=0))


def test_max_keyframes_caps_the_buffer():
    acc = ScanAccumulator(max_keyframes=3)
    for i in range(5):
        acc.update(_grid(3), _pos(i * 1.0, 0.0), _quat_yaw(0.0))
    assert acc.n_keyframes == 3


def test_standing_still_evicts_accumulated_coverage():
    """The cost of a tick-keyed window, pinned down so it cannot drift silently.

    The buffer holds the last ``max_keyframes`` *ticks*, so a robot that stops
    refills it with copies of one sweep and the earlier coverage is gone. At
    the 2 Hz live tick, ``max_keyframes=10`` means 5 s of standing still.
    """
    acc = ScanAccumulator(max_keyframes=3, voxel_m=0.05)
    far = _grid(3, origin=(50.0, 0.0, 0.0))
    near = _grid(3, origin=(0.0, 0.0, 0.0))
    acc.update(far, _pos(50.0, 0.0), _quat_yaw(0.0))
    out = acc.update(near, _pos(0, 0), _quat_yaw(0.0))
    assert (out[:, 0] > 40.0).any()                  # far sweep still present
    for _ in range(3):                               # parked, buffer turns over
        out = acc.update(near, _pos(0, 0), _quat_yaw(0.0))
    assert not (out[:, 0] > 40.0).any()              # coverage dropped


def test_reset_clears_buffer():
    acc = ScanAccumulator()
    acc.update(_grid(5), _pos(0, 0), _quat_yaw(0.0))
    acc.reset()
    assert acc.n_keyframes == 0


# ── densification + voxel downsample ──────────────────────────────────────────

def test_accumulation_densifies_disjoint_regions():
    acc = ScanAccumulator(voxel_m=0.05)
    a = _grid(5, origin=(0.0, 0.0, 0.0))       # 25 pts near origin
    b = _grid(5, origin=(10.0, 0.0, 0.0))      # 25 pts far away (disjoint)
    acc.update(a, _pos(0, 0), _quat_yaw(0.0))
    out = acc.update(b, _pos(1.0, 0.0), _quat_yaw(0.0))
    # two disjoint keyframes → the merged cloud carries both.
    assert out.shape[0] == 50


def test_overlapping_sweeps_are_deduped_by_voxel():
    acc = ScanAccumulator(voxel_m=0.05)
    g = _grid(5, step=0.2)                      # points 0.2 m apart → own voxels
    acc.update(g, _pos(0, 0), _quat_yaw(0.0))
    # identical cloud from a moved pose → same voxels → no growth.
    out = acc.update(g.copy(), _pos(1.0, 0.0), _quat_yaw(0.0))
    assert acc.n_keyframes == 2
    assert out.shape[0] == g.shape[0]          # 50 raw → 25 after dedup


def test_voxel_downsample_passthrough_and_empty():
    pts = _grid(4)
    assert voxel_downsample(pts, 0.0).shape == pts.shape       # disabled
    assert voxel_downsample(np.empty((0, 3)), 0.05).shape == (0, 3)


def test_voxel_downsample_collapses_dense_cluster():
    # 1000 points inside a single 5 cm voxel collapse to one.
    rng = np.random.default_rng(0)
    cluster = rng.uniform(0.0, 0.04, size=(1000, 3))
    out = voxel_downsample(cluster, 0.05)
    assert out.shape[0] == 1


# ── integration: accumulation rescues a sparse small object ───────────────────

def _sparse_object(center, n, seed):
    """``n`` LiDAR returns clustered tightly on one small object."""
    rng = np.random.default_rng(seed)
    d = rng.normal(0.0, 0.03, size=(n, 3))
    return np.column_stack([d[:, 0] + center[0], d[:, 1] + center[1],
                            d[:, 2] + center[2]])


def test_accumulation_rescues_a_sparse_small_object():
    """A single sweep leaves a small object below ``min_inliers``; a few
    accumulated keyframes push it over — the exact failure the change targets."""
    from xiao_hei_vln.messages.common import Quaternion
    from xiao_hei_vln.perception.geometry import EQUIRECT_H, EQUIRECT_W
    from xiao_hei_vln.perception.lifter import PointLifter

    lifter = PointLifter(min_inliers=10)
    acc = ScanAccumulator(voxel_m=0.0)  # keep every return
    full_mask = np.ones((EQUIRECT_H, EQUIRECT_W), dtype=bool)
    q = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    center = (3.0, 0.0, 0.0)

    # Frame 0: only 4 returns on the object → below the gate.
    dense0 = acc.update(_sparse_object(center, 4, seed=0), _pos(0, 0), q)
    assert lifter.lift(full_mask, dense0, _pos(0, 0), q).position is None

    # Three more keyframes (robot edging forward) each add ~4 returns.
    last_pose = _pos(0, 0)
    for i in range(1, 4):
        last_pose = _pos(i * 0.3, 0.0)
        dense = acc.update(_sparse_object(center, 4, seed=i), last_pose, q)

    res = lifter.lift(full_mask, dense, last_pose, q)
    assert res.position is not None            # now supported
    assert res.n_inliers >= 10
    assert res.position.x == pytest.approx(3.0, abs=0.1)
