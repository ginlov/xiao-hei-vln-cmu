"""Unit tests for NextBestViewExplorer scoring helpers."""

from __future__ import annotations

from xiao_hei_vln.exploration._grid import OccupancyGrid
from xiao_hei_vln.exploration._nbv import NextBestViewExplorer


def test_unknown_region_size_counts_local_unknown() -> None:
    grid = OccupancyGrid(resolution=1.0)
    # One free cell at origin; neighbors are unknown.
    grid._free.add((0, 0))
    assert grid.unknown_region_size([(0, 0)], radius_cells=1) == 4
    assert grid.unknown_region_size([(0, 0)], radius_cells=2) > 4


def test_reachable_path_costs_bfs() -> None:
    grid = OccupancyGrid(resolution=1.0)
    grid._free.update({(0, 0), (1, 0), (2, 0)})
    costs = grid.reachable_path_costs(0.1, 0.1)
    assert costs[(0, 0)] == 0.0
    assert costs[(1, 0)] == 1.0
    assert costs[(2, 0)] == 2.0


def test_nbv_picks_toward_unknown() -> None:
    explorer = NextBestViewExplorer(
        max_waypoints=5,
        grid_resolution=1.0,
        waypoint_reach_dist=0.5,
        n_samples=20,
        seed=0,
    )
    # Corridor of free cells; unknown opens only to the right of x=2.
    explorer._grid._free.update({(0, 0), (1, 0), (2, 0)})
    wp = explorer._pick(0.0, 0.0)
    assert wp is not None
    # Prefer the frontier at the open end (higher unknown gain).
    assert wp.x >= 1.0
