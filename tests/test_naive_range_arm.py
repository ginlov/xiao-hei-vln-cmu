"""The naive-range ablation: metres from the model instead of from the lidar.

The arm exists for one measurement (Sec. V-F of the paper): hold the bearing,
the standoff, the converter prediction and everything downstream identical, and
change only where the range comes from. These tests pin the two properties that
make it a valid ablation --- that the bearing really is untouched, and that the
lift's refusals really do go with the lift --- because both are easy to break
by a later edit that looks harmless.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from vlm_approach import MIN_ADVANCE_M, next_waypoint  # noqa: E402


def pose_at(x=0.0, y=0.0, z=0.75):
    """Identity orientation, so map frame and camera frame share a heading."""
    return {"position": [x, y, z], "orientation": [0.0, 0.0, 0.0, 1.0]}


def box_center_of(face_idx=0, half=30):
    """A box in the middle of a face, big enough to have angular size."""
    c = 320
    return [c - half, c - half, c + half, c + half]


def empty_scan():
    """A scan with no returns anywhere: the lidar arm can never commit on it."""
    return np.zeros((0, 3), float)


def wall_scan(distance=3.0, n=4000):
    """A dense frontal wall at `distance`, so the lidar arm has a real lift."""
    rng = np.random.default_rng(0)
    y = rng.uniform(-0.6, 0.6, n)
    z = rng.uniform(0.2, 1.4, n)
    return np.column_stack([np.full(n, distance), y, z])


def test_naive_arm_commits_on_the_models_metres():
    wp = next_waypoint(box_center_of(), 0, wall_scan(3.0), pose_at(),
                       standoff=0.9, model_range=8.0)
    assert wp.committed
    assert wp.range_m == pytest.approx(8.0)
    # It drove to the model's range minus the standoff, not the lidar's.
    assert float(np.linalg.norm(wp.xy)) == pytest.approx(8.0 - 0.9, abs=1e-6)
    assert "naive arm" in wp.reason


def test_the_two_arms_differ_only_in_range_never_in_bearing():
    """The ablation is worthless if the arms also point somewhere different."""
    scan, pose, box = wall_scan(3.0), pose_at(), box_center_of()
    lidar = next_waypoint(box, 0, scan, pose, standoff=0.9)
    naive = next_waypoint(box, 0, scan, pose, standoff=0.9, model_range=8.0)
    assert lidar.committed and naive.committed

    def unit(v):
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-9 else v

    # Same ray, different distance along it.
    assert unit(np.asarray(naive.xy)) == pytest.approx(unit(np.asarray(lidar.xy)),
                                                       abs=1e-9)
    assert float(np.linalg.norm(naive.xy)) > float(np.linalg.norm(lidar.xy))


def test_naive_arm_cannot_abstain_where_the_lidar_arm_does():
    """No returns at all: the shipped arm steps and re-observes, the naive arm
    commits anyway. That asymmetry IS the effect being measured --- a system
    taking metres from a model has no second sensor to disagree with it."""
    scan, pose, box = empty_scan(), pose_at(), box_center_of()
    lidar = next_waypoint(box, 0, scan, pose, standoff=0.9)
    naive = next_waypoint(box, 0, scan, pose, standoff=0.9, model_range=6.0)
    assert not lidar.committed and lidar.range_m is None
    assert naive.committed and naive.range_m == pytest.approx(6.0)


def test_a_blind_bearing_does_not_stop_the_naive_arm():
    """The elevation-floor refusal is a property of the scanner. Steeply down,
    the shipped arm refuses to commit; the naive arm has nothing to refuse
    with, which is the honest consequence of removing the lift."""
    # Box low in the face -> a bearing below the scanner's floor.
    low_box = [560, 290, 620, 350]
    scan, pose = wall_scan(3.0), pose_at()
    lidar = next_waypoint(low_box, 2, scan, pose, standoff=0.9)
    naive = next_waypoint(low_box, 2, scan, pose, standoff=0.9, model_range=5.0)
    assert not lidar.committed
    assert naive.committed and naive.range_m == pytest.approx(5.0)


def test_default_is_unchanged():
    """model_range=None must reproduce the shipped behaviour exactly."""
    scan, pose, box = wall_scan(3.0), pose_at(), box_center_of()
    a = next_waypoint(box, 0, scan, pose, standoff=0.9)
    b = next_waypoint(box, 0, scan, pose, standoff=0.9, model_range=None)
    assert a.committed == b.committed
    assert a.range_m == b.range_m
    assert np.asarray(a.xy) == pytest.approx(np.asarray(b.xy))


def test_naive_arm_respects_the_minimum_advance():
    """A model range inside the standoff must not produce a backwards waypoint."""
    wp = next_waypoint(box_center_of(), 0, wall_scan(3.0), pose_at(),
                       standoff=2.0, model_range=0.3)
    assert wp.committed
    assert float(np.linalg.norm(wp.xy)) == pytest.approx(MIN_ADVANCE_M, abs=1e-6)
