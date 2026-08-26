"""The converter-modelling ablation: `free` clamps to floor, `raw` does not.

Arms C and D remove our *prediction* of `waypointConverter`, never the
converter itself -- it runs on the robot and re-snaps whatever we publish. What
these tests pin is the difference between the two skip modes, because that
difference is the whole steel-manning argument: a naive team writes a
free-space check and does not write a settle model, so `free` is the honest
baseline and `raw` is only there to price the check.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from waypoint_converter_model import ConverterModel  # noqa: E402


def floor_strip(length=8.0, half_width=2.0, step=0.1):
    """Traversable floor from x=0 out to `length`, intensity 0 (no obstacle)."""
    xs = np.arange(0.0, length + step, step)
    ys = np.arange(-half_width, half_width + step, step)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    return np.column_stack([pts[:, 0], pts[:, 1],
                            np.zeros(len(pts)), np.zeros(len(pts))])


def wall_at(x, half_width=2.0, step=0.1, height=1.0):
    """A band of high-intensity points: the terrain map's idea of an obstacle."""
    ys = np.arange(-half_width, half_width + step, step)
    xs = np.arange(x, x + 0.4, step)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    return np.column_stack([pts[:, 0], pts[:, 1],
                            np.full(len(pts), height),
                            np.full(len(pts), height)])


def clamp_free(cm, origin, aim):
    """The `free` mode's arithmetic, isolated from the loop that runs it."""
    goal = np.asarray(aim, float)
    u = goal - origin
    d = float(np.linalg.norm(u))
    if d <= 1e-6:
        return goal
    u = u / d
    free = float(cm.reach_along(origin, u))
    return origin + u * max(free, 0.0) if free < d else goal


def test_free_mode_clamps_an_aim_that_is_past_a_wall():
    terrain = np.vstack([floor_strip(length=3.0), wall_at(3.0)])
    cm = ConverterModel(terrain)
    origin = np.array([0.2, 0.0])
    aim = np.array([8.0, 0.0])          # well past the wall
    goal = clamp_free(cm, origin, aim)
    assert goal[0] < aim[0], "an aim beyond the floor must be pulled back"
    assert goal[0] <= 3.2, "it must not be published past the wall"


def test_free_mode_leaves_a_reachable_aim_alone():
    cm = ConverterModel(floor_strip(length=8.0))
    origin = np.array([0.2, 0.0])
    aim = np.array([4.0, 0.0])
    goal = clamp_free(cm, origin, aim)
    assert goal == pytest.approx(aim, abs=0.35), \
        "floor all the way: the free-space check has nothing to do"


def test_raw_mode_would_publish_the_unreachable_aim():
    """`raw` is the weak baseline, and this is exactly why we do not default to
    it: the aim is published unchanged even with a wall in between."""
    terrain = np.vstack([floor_strip(length=3.0), wall_at(3.0)])
    cm = ConverterModel(terrain)
    origin = np.array([0.2, 0.0])
    aim = np.array([8.0, 0.0])
    raw_goal = np.asarray(aim, float)            # what "raw" publishes
    free_goal = clamp_free(cm, origin, aim)
    assert raw_goal[0] > free_goal[0]
    assert float(np.linalg.norm(raw_goal - free_goal)) > 1.0


def test_the_skip_is_not_the_converters_settle_model():
    """The ablation is only meaningful if `reach_along` and `best_waypoint_toward`
    genuinely differ -- one is a free-space query, the other predicts where the
    platform re-minimises to. On the same frame they must disagree."""
    cm = ConverterModel(floor_strip(length=8.0))
    origin = np.array([0.2, 0.0])
    aim = np.array([6.0, 0.0])
    free_goal = clamp_free(cm, origin, aim)
    best = cm.best_waypoint_toward(aim, origin)
    assert best is not None
    modelled_goal, settles, _ = best
    # The settle model returns a point it has reasoned about; the free-space
    # clamp returns a point on the ray. They are not the same computation.
    assert float(np.linalg.norm(np.asarray(modelled_goal) - free_goal)) > 1e-6
    assert settles is not None


def test_modes_are_the_three_the_flag_allows():
    import approach_loop
    assert approach_loop.SKIP_CONVERTER in ("off", "free", "raw")
    # Default must stay 'off': every existing run and every shipped drive
    # depends on the converter being modelled.
    import os
    if "XIAO_HEI_SKIP_CONVERTER" not in os.environ:
        assert approach_loop.SKIP_CONVERTER == "off"


def clamp_free_fixed(cm, origin, aim, floor_m=0.3):
    """The shipped `free` arithmetic after the degenerate-clamp fix."""
    goal = np.asarray(aim, float)
    u = goal - origin
    d = float(np.linalg.norm(u))
    if d <= 1e-6:
        return goal
    u = u / d
    free = float(cm.reach_along(origin, u))
    if floor_m <= free < d:
        return origin + u * free
    return goal


def test_an_empty_terrain_frame_must_not_pin_the_vehicle_in_place():
    """The bug that made the first arm-C sweep unusable.

    On the frame right after a spawn the terrain map held three points, so
    `reach_along` returned 0.0 for every direction and the old clamp published
    `origin + u * 0` -- the vehicle's own position. 22% of steps and 15 of 26
    runs died that way, and the arm measured our bug rather than the ablation.
    An uninformative free-space check has to leave the aim alone.
    """
    cm = ConverterModel(np.zeros((3, 4), float))   # a nearly empty frame
    origin = np.array([0.0, 0.0])
    aim = np.array([3.0, 0.0])
    assert float(cm.reach_along(origin, np.array([1.0, 0.0]))) < 0.3
    goal = clamp_free_fixed(cm, origin, aim)
    assert goal == pytest.approx(aim), "an empty map must not clamp the aim"
    assert float(np.linalg.norm(goal - origin)) > 0.3, "must still move"


def test_a_real_wall_still_clamps_after_the_fix():
    """The fix must not disable the check where the map can actually see."""
    terrain = np.vstack([floor_strip(length=3.0), wall_at(3.0)])
    cm = ConverterModel(terrain)
    origin = np.array([0.2, 0.0])
    goal = clamp_free_fixed(cm, origin, np.array([8.0, 0.0]))
    assert goal[0] < 8.0 and goal[0] <= 3.2
