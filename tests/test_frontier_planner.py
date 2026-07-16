"""Unit tests for `xiao_hei_vln.exploration.frontier`."""

from __future__ import annotations

import math

import numpy as np
import pytest

from xiao_hei_vln.exploration.frontier import FrontierPlanner, ScoringWeights
from xiao_hei_vln.messages import Header, OdomPose, Quaternion, Stamp, Vector3
from xiao_hei_vln.perception.global_map import FREE, OCCUPIED, UNKNOWN, GlobalMap


def _stamp() -> Stamp:
    return Stamp.from_seconds(0.0)


def _pose(x: float, y: float) -> OdomPose:
    return OdomPose(
        header=Header(stamp=_stamp(), frame_id="map"),
        position=Vector3(x=x, y=y, z=0.75),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )


def _bare_grid_with_top_frontier() -> GlobalMap:
    """A GlobalMap whose lower half is FREE and upper half is UNKNOWN.

    The boundary row is a wide frontier cluster.
    """
    gm = GlobalMap(half_extent_m=3.0, resolution_m=0.5)
    # initialise without any points so origin is anchored to (0, 0) at the centre
    gm.update(np.zeros((0,), dtype=np.float32), _pose(0.0, 0.0))
    gm.grid[6:, :] = FREE
    return gm


def _mark_pose_free(gm: GlobalMap, pose: OdomPose) -> None:
    """Ensure the robot's cell is FREE — matches production update() behaviour."""
    i, j = gm.world_to_idx(pose.position.x, pose.position.y)
    H, W = gm.shape
    if 0 <= i < H and 0 <= j < W:
        gm.grid[i, j] = FREE


class TestEmpty:
    def test_returns_none_when_no_clusters(self) -> None:
        gm = GlobalMap()
        gm.update(np.zeros((0,), dtype=np.float32), _pose(0.0, 0.0))
        # All UNKNOWN except the robot cell → no FREE cell with an UNKNOWN
        # neighbour because the robot cell is surrounded by UNKNOWN but is
        # itself only one cell, which IS a frontier. So we need a stricter
        # setup: actively fill everything as OCCUPIED so no frontier exists.
        gm.grid[:] = OCCUPIED
        planner = FrontierPlanner(gm)
        assert planner.select(_pose(0.0, 0.0)) is None


class TestSelection:
    def test_picks_larger_when_clusters_are_symmetric(self) -> None:
        """Two clusters mirrored around the robot → size term decides."""
        gm = GlobalMap(half_extent_m=5.0, resolution_m=0.5)
        gm.update(np.zeros((0,), dtype=np.float32), _pose(0.0, 0.0))
        gm.grid[:] = UNKNOWN
        # Build a single FREE corridor that contains both clusters and the
        # robot, so BFS reachability connects them all (BFS-based reachability
        # otherwise correctly treats disconnected components as unreachable).
        gm.grid[8:11, 4:19] = FREE
        planner = FrontierPlanner(gm, min_cluster_size=2)
        pose = _pose(0.0, -0.75)
        _mark_pose_free(gm, pose)
        chosen = planner.select(pose)
        assert chosen is not None
        # Bigger cluster's representative cell is on the right (x > 0).
        assert chosen.centroid_xy[0] > 0.0

    def test_history_dampens_revisits(self) -> None:
        gm = _bare_grid_with_top_frontier()
        planner = FrontierPlanner(
            gm,
            weights=ScoringWeights(size=0.0, distance=0.0, loop=10.0, loop_decay_m=2.0),
            history_length=3,
        )
        pose = _pose(0.0, 0.0)
        _mark_pose_free(gm, pose)

        # Force-feed the history with the candidate centroid to drive its score
        # way below zero. Then a second call must still produce *something*
        # (only one frontier exists in this grid) — but the loop penalty must
        # show up in the score breakdown.
        scored_before = planner.score_all(pose)
        assert len(scored_before) >= 1
        baseline_score = scored_before[0].score
        # baseline has loop_term == 0
        assert scored_before[0].loop_term == pytest.approx(0.0)

        # Trigger a selection to push the centroid into history.
        first_pick = planner.select(pose)
        assert first_pick is not None

        # Now score_all again: the same cluster should have a *worse* score.
        scored_after = planner.score_all(pose)
        assert len(scored_after) >= 1
        repeat_score = scored_after[0].score
        assert repeat_score < baseline_score
        assert scored_after[0].loop_term < 0  # negative penalty term


class TestScoringFormula:
    def test_distance_term_negative(self) -> None:
        gm = _bare_grid_with_top_frontier()
        planner = FrontierPlanner(gm)
        # Robot must stand on a FREE cell connected to the frontier for the
        # BFS reachability check to admit it. The frontier is along y ≈ 0.25
        # and the FREE region extends down to y ≈ 2.75, so a robot anywhere
        # inside the FREE region will see a non-zero positive distance.
        pose = _pose(0.0, 2.0)
        scored = planner.score_all(pose)
        assert scored
        assert scored[0].distance_term < 0
        assert scored[0].distance_m > 0

    def test_size_term_grows_with_size(self) -> None:
        gm = _bare_grid_with_top_frontier()
        planner = FrontierPlanner(gm)
        pose = _pose(0.0, 0.0)
        _mark_pose_free(gm, pose)
        scored = planner.score_all(pose)
        assert scored
        # log1p(size) is positive and increases with size.
        bigger = max(scored, key=lambda s: s.size_term)
        assert bigger.size_term > 0
        assert math.isclose(
            bigger.size_term, math.log1p(bigger.cluster.size), rel_tol=1e-6,
        )


class TestReset:
    def test_reset_clears_history(self) -> None:
        gm = _bare_grid_with_top_frontier()
        planner = FrontierPlanner(gm)
        pose = _pose(0.0, 0.0)
        _mark_pose_free(gm, pose)
        planner.select(pose)
        assert len(planner.history) == 1
        planner.reset()
        assert planner.history == ()


class TestReachability:
    def test_wall_between_robot_and_frontier_makes_unreachable(self) -> None:
        gm = GlobalMap(half_extent_m=3.0, resolution_m=0.5)
        gm.update(np.zeros((0,), dtype=np.float32), _pose(0.0, 0.0))
        gm.grid[:] = UNKNOWN
        # Build a frontier cluster on the right edge of a FREE strip
        gm.grid[5:7, 8:12] = FREE  # tiny FREE block top right
        # Robot in a separate FREE block on the left
        gm.grid[5:7, 0:3] = FREE
        # Add an OCCUPIED wall between them
        gm.grid[:, 6:7] = OCCUPIED

        planner = FrontierPlanner(gm, min_cluster_size=1)
        pose = _pose(-2.0, -0.25)  # inside the left FREE block

        # The right-side frontier is unreachable due to the wall — planner
        # should not return it.
        chosen = planner.select(pose)
        # Either chooses something on the same side, or returns None — but
        # crucially it does not return the right-side frontier (x > 1).
        if chosen is not None:
            assert chosen.centroid_xy[0] < 1.0
