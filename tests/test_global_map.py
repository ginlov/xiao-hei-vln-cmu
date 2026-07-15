"""Unit tests for `xiao_hei_vln.perception.global_map`.

All synthetic — no ROS, no fixtures from disk. The point is that the
math is correct on hand-built inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

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


def _points(xyz_cost: list[tuple[float, float, float, float]]) -> np.ndarray:
    return np.array(xyz_cost, dtype=np.float32)


class TestEmpty:
    def test_fresh_grid_is_all_unknown(self) -> None:
        gm = GlobalMap()
        assert gm.is_initialised is False
        assert np.all(gm.grid == UNKNOWN)
        # World queries before first update should be safe to inspect via
        # ``is_initialised`` but raise if mis-used.
        assert gm.is_traversable(0.0, 0.0) is False

    def test_find_frontier_clusters_returns_empty_before_init(self) -> None:
        gm = GlobalMap()
        assert gm.find_frontier_clusters() == []


class TestUpdate:
    def test_update_marks_free_and_occupied(self) -> None:
        gm = GlobalMap(half_extent_m=5.0, resolution_m=0.5, obstacle_cost_threshold=0.5)
        # Pose at origin → grid anchor at (-5, -5).
        pose = _pose(0.0, 0.0)
        # Two points: one low-cost (FREE) at (1, 0), one high-cost (OCCUPIED) at (2, 1).
        pts = _points([(1.0, 0.0, 0.0, 0.1), (2.0, 1.0, 0.0, 0.9)])
        gm.update(pts, pose)

        i_free, j_free = gm.world_to_idx(1.0, 0.0)
        i_occ, j_occ = gm.world_to_idx(2.0, 1.0)
        assert gm.grid[i_free, j_free] == FREE
        assert gm.grid[i_occ, j_occ] == OCCUPIED
        # Robot cell is always FREE even if no point covers it.
        ri, rj = gm.world_to_idx(0.0, 0.0)
        assert gm.grid[ri, rj] == FREE

    def test_origin_anchored_on_first_update(self) -> None:
        """A robot spawned at far-away world coords still gets a centred grid."""
        gm = GlobalMap(half_extent_m=10.0, resolution_m=0.5)
        far_pose = _pose(1000.0, -2500.0)
        gm.update(_points([]), far_pose)
        assert gm.is_initialised
        ox, oy = gm.origin_xy
        # Origin should be 10 m down-left of the pose.
        assert ox == pytest.approx(990.0)
        assert oy == pytest.approx(-2510.0)
        # Robot cell should be marked FREE inside the grid.
        ri, rj = gm.world_to_idx(1000.0, -2500.0)
        assert gm.grid[ri, rj] == FREE

    def test_update_drops_out_of_bounds_points(self) -> None:
        gm = GlobalMap(half_extent_m=2.0, resolution_m=0.5)
        pose = _pose(0.0, 0.0)
        # One in-bounds, one way outside the 2m half-extent.
        pts = _points([(0.5, 0.5, 0.0, 0.0), (100.0, 100.0, 0.0, 0.0)])
        gm.update(pts, pose)
        # In-bounds cell is FREE.
        i, j = gm.world_to_idx(0.5, 0.5)
        assert gm.grid[i, j] == FREE
        # The grid was preallocated to (2*2/0.5)^2 = 64 cells; only a handful are FREE.
        n_free = int((gm.grid == FREE).sum())
        # Robot cell + sample point = at least 1 free cell; some cells could
        # coincide if grid resolution lands the same index → assert "small".
        assert 1 <= n_free <= 4

    def test_repeated_updates_accumulate(self) -> None:
        gm = GlobalMap(half_extent_m=3.0, resolution_m=0.5)
        gm.update(_points([(0.5, 0.0, 0.0, 0.0)]), _pose(0.0, 0.0))
        free_before = int((gm.grid == FREE).sum())
        gm.update(_points([(1.5, 1.5, 0.0, 0.0)]), _pose(0.0, 0.0))
        free_after = int((gm.grid == FREE).sum())
        assert free_after > free_before


class TestFrontierExtraction:
    def _build_grid_with_open_top(self) -> GlobalMap:
        """
        Hand-build a grid where the bottom half is FREE, the top half is
        UNKNOWN, and the boundary row contains FREE cells adjacent to
        UNKNOWN — i.e. a clean frontier row.
        """
        gm = GlobalMap(half_extent_m=3.0, resolution_m=0.5)
        gm.update(_points([]), _pose(0.0, 0.0))  # initialise
        # Force a known geometry by writing directly into the grid:
        # bottom 6 rows FREE, top 6 rows stay UNKNOWN. The 6th row from top
        # of the FREE block is a frontier (its top neighbour is UNKNOWN).
        gm.grid[6:, :] = FREE
        return gm

    def test_finds_frontier_along_known_edge(self) -> None:
        gm = self._build_grid_with_open_top()
        clusters = gm.find_frontier_clusters(min_cluster_size=2)
        assert len(clusters) >= 1
        # The biggest cluster should span the full width of the boundary row.
        biggest = max(clusters, key=lambda c: c.size)
        assert biggest.size >= 6  # at least the row width
        # Its centroid y should fall on the boundary row.
        _cx, cy = biggest.centroid_xy
        # Row 6 (zero-indexed from top) has world y centred at:
        #     oy + (6 + 0.5) * 0.5 = -3 + 3.25 = 0.25
        assert cy == pytest.approx(0.25, abs=0.5)

    def test_no_frontiers_when_fully_enclosed(self) -> None:
        """A FREE box surrounded by OCCUPIED → no frontier (no UNKNOWN neighbours)."""
        gm = GlobalMap(half_extent_m=3.0, resolution_m=0.5)
        gm.update(_points([]), _pose(0.0, 0.0))
        gm.grid[:] = OCCUPIED
        gm.grid[4:8, 4:8] = FREE
        assert gm.find_frontier_clusters() == []

    def test_min_cluster_size_filters_small_clusters(self) -> None:
        gm = self._build_grid_with_open_top()
        big = gm.find_frontier_clusters(min_cluster_size=2)
        impossibly_large = gm.find_frontier_clusters(min_cluster_size=10_000)
        assert big and not impossibly_large


class TestReset:
    def test_reset_clears_grid_and_origin(self) -> None:
        gm = GlobalMap(half_extent_m=3.0, resolution_m=0.5)
        gm.update(_points([(0.0, 0.0, 0.0, 0.0)]), _pose(0.0, 0.0))
        assert gm.is_initialised
        gm.reset()
        assert not gm.is_initialised
        assert np.all(gm.grid == UNKNOWN)


class TestLineCheck:
    def test_line_through_free_returns_false(self) -> None:
        gm = GlobalMap(half_extent_m=3.0, resolution_m=0.5)
        gm.update(_points([]), _pose(0.0, 0.0))
        gm.grid[:] = FREE
        # Any straight line stays inside FREE → not blocked.
        assert gm.line_passes_through(-2.0, -2.0, 2.0, 2.0) is False

    def test_line_through_occupied_returns_true(self) -> None:
        gm = GlobalMap(half_extent_m=3.0, resolution_m=0.5)
        gm.update(_points([]), _pose(0.0, 0.0))
        gm.grid[:] = FREE
        # Drop an occupied wall along a row that the (-2, -2) → (2, 2) line crosses.
        gm.grid[6, :] = OCCUPIED
        assert gm.line_passes_through(-2.0, -2.0, 2.0, 2.0) is True
