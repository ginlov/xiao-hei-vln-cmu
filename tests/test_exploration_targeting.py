"""Regression tests for the target-selection failures seen in exploration_logs/.

Three distinct bugs, each reproduced from a real log:

* `loft` (frontier): the same centroid was re-issued 18 times in a row because
  suppressing the cells *under* a rejected centroid leaves the rest of the
  cluster averaging to the same point.
* `home_building_2` (nbv): the robot stopped moving entirely — visited
  waypoints were stamped OCCUPIED, which walled off the reachable-cost BFS.
* both: a single barren selection tick ended the sweep as `no_frontiers`.
"""

from __future__ import annotations

import numpy as np

from xiao_hei_vln.exploration._frontier import FrontierExplorer
from xiao_hei_vln.exploration._grid import OccupancyGrid
from xiao_hei_vln.exploration._nbv import NextBestViewExplorer
from xiao_hei_vln.messages.outputs import Waypoint


def _corridor(grid: OccupancyGrid, n: int = 12) -> None:
    """A 1-cell-wide east-west corridor of FREE cells from (0,0)."""
    grid._free.update({(i, 0) for i in range(n)})


# ----------------------------------------------------------------------
# OccupancyGrid: suppression must not change traversability


def test_mark_no_target_keeps_cells_traversable() -> None:
    grid = OccupancyGrid(resolution=1.0)
    _corridor(grid)
    grid.mark_no_target(5.5, 0.5, radius_cells=2)

    costs = grid.reachable_path_costs(0.5, 0.5)
    # The far end is still reachable *through* the suppressed disc.
    assert (11, 0) in costs
    assert grid.is_targetable((5, 0)) is False
    assert grid.is_targetable((11, 0)) is True


def test_mark_occupied_still_blocks_traversal() -> None:
    """mark_occupied keeps its old meaning — it is for real obstacles."""
    grid = OccupancyGrid(resolution=1.0)
    _corridor(grid)
    grid.mark_occupied(5.5, 0.5, radius_cells=1)

    costs = grid.reachable_path_costs(0.5, 0.5)
    assert (11, 0) not in costs


# ----------------------------------------------------------------------
# NBV: visiting a waypoint must not disconnect the map behind it


def test_nbv_visited_waypoint_does_not_wall_off_the_corridor() -> None:
    explorer = NextBestViewExplorer(grid_resolution=1.0, waypoint_reach_dist=0.5)
    _corridor(explorer._grid)

    explorer._current_target = Waypoint(x=5.5, y=0.5, heading=0.0)
    explorer.advance()

    costs = explorer._grid.reachable_path_costs(0.5, 0.5)
    assert (11, 0) in costs, "a visited waypoint must not become an obstacle"
    # It is still off-limits as a *goal*.
    assert explorer._grid.is_targetable((5, 0)) is False


def test_nbv_skip_suppresses_a_disc_not_a_single_cell() -> None:
    explorer = NextBestViewExplorer(
        grid_resolution=1.0, waypoint_reach_dist=0.5, skip_mark_radius_cells=2
    )
    _corridor(explorer._grid)
    explorer._current_target = Waypoint(x=5.5, y=0.5, heading=0.0)
    explorer.force_skip()

    # The neighbour 1 cell away used to be picked next and failed identically.
    assert (5, 0) in explorer._blacklist
    assert (6, 0) in explorer._blacklist
    assert (4, 0) in explorer._blacklist


def test_nbv_never_reproposes_a_skipped_target() -> None:
    explorer = NextBestViewExplorer(
        grid_resolution=1.0, waypoint_reach_dist=0.5, max_consecutive_skips=99, seed=0
    )
    _corridor(explorer._grid, n=20)

    seen: list[tuple[float, float]] = []
    for _ in range(8):
        wp = explorer._pick(0.5, 0.5)
        if wp is None:
            break
        assert (wp.x, wp.y) not in seen, "re-issued an already-skipped target"
        seen.append((wp.x, wp.y))
        explorer._current_target = wp
        explorer.force_skip()
    assert len(seen) >= 3


# ----------------------------------------------------------------------
# Frontier: the loft thrash loop


def test_frontier_does_not_reissue_the_same_centroid() -> None:
    explorer = FrontierExplorer(
        grid_resolution=1.0,
        waypoint_reach_dist=0.5,
        min_frontier_size=1,
        max_waypoint_dist=50.0,
        max_consecutive_skips=99,
        reject_radius=1.5,
    )
    # Two separated frontier blobs so there is always an alternative.
    explorer._grid._free.update({(6, 0), (6, 1), (7, 0), (7, 1)})
    explorer._grid._free.update({(-6, 0), (-6, 1), (-7, 0), (-7, 1)})

    seen: list[tuple[float, float]] = []
    for _ in range(4):
        wp = explorer._select_frontier(0.5, 0.5)
        if wp is None:
            break
        key = (round(wp.x, 2), round(wp.y, 2))
        assert key not in seen, f"frontier re-issued {key} after skipping it"
        seen.append(key)
        explorer._current_target = wp
        explorer.force_skip()
    assert len(seen) >= 2


def test_frontier_skip_leaves_the_map_traversable() -> None:
    explorer = FrontierExplorer(grid_resolution=1.0, max_consecutive_skips=99)
    _corridor(explorer._grid)
    explorer._current_target = Waypoint(x=5.5, y=0.5, heading=0.0)
    explorer.force_skip()

    assert (5, 0) in explorer._grid.free_cells
    assert (11, 0) in explorer._grid.reachable_path_costs(0.5, 0.5)


# ----------------------------------------------------------------------
# Termination: neither a barren tick nor a skip run should end a live sweep


def _snapshot(x: float, y: float, t: float):
    """Minimal VLMInput stand-in — update() only touches these four fields."""

    class _P:
        position = type("V", (), {"x": x, "y": y, "z": 0.0})()

    class _T:
        @staticmethod
        def to_seconds() -> float:
            return t

    class _S:
        terrain_ext = None
        pose = _P()
        tick_time = _T()

    return _S()


def test_frontier_survives_a_single_barren_tick() -> None:
    explorer = FrontierExplorer(grid_resolution=1.0, empty_ticks_before_done=5)
    explorer._grid._free.update({(0, 0), (1, 0)})
    explorer._grid.mark_no_target(0.5, 0.5, radius_cells=4)  # nothing targetable

    for i in range(4):
        explorer.update(_snapshot(0.5, 0.5, float(i)))
        assert not explorer.is_complete(), "one empty tick must not end the sweep"
    explorer.update(_snapshot(0.5, 0.5, 5.0))
    assert explorer.is_complete()


def test_skip_run_resets_while_the_map_is_still_growing() -> None:
    explorer = FrontierExplorer(
        grid_resolution=1.0, max_consecutive_skips=2, skip_reset_free_cells=10
    )
    explorer._grid._free.update({(i, 0) for i in range(50)})

    explorer._current_target = Waypoint(x=1.5, y=0.5, heading=0.0)
    explorer.force_skip()
    explorer._current_target = Waypoint(x=2.5, y=0.5, heading=0.0)
    explorer.force_skip()
    assert not explorer.is_complete(), "still discovering free space — not stalled"
    assert explorer._consecutive_skip_count == 0

    # Now the map stops growing; the same run of skips does terminate.
    explorer._current_target = Waypoint(x=3.5, y=0.5, heading=0.0)
    explorer.force_skip()
    explorer._current_target = Waypoint(x=4.5, y=0.5, heading=0.0)
    explorer.force_skip()
    assert explorer.is_complete()


def test_nbv_skip_run_resets_while_the_map_is_still_growing() -> None:
    explorer = NextBestViewExplorer(
        grid_resolution=1.0, max_consecutive_skips=2, skip_reset_free_cells=10
    )
    explorer._grid._free.update({(i, 0) for i in range(50)})

    for x in (1.5, 2.5):
        explorer._current_target = Waypoint(x=x, y=0.5, heading=0.0)
        explorer.force_skip()
    assert not explorer.is_complete()

    for x in (3.5, 4.5):
        explorer._current_target = Waypoint(x=x, y=0.5, heading=0.0)
        explorer.force_skip()
    assert explorer.is_complete()


def test_frontier_falls_back_to_undersized_clusters() -> None:
    """A 2-cell frontier is still unseen space — better than declaring done."""
    explorer = FrontierExplorer(
        grid_resolution=1.0, min_frontier_size=5, max_waypoint_dist=50.0
    )
    explorer._grid._free.update({(6, 0), (6, 1)})
    assert explorer._select_frontier(0.5, 0.5) is not None


def test_nbv_scores_every_frontier_before_sampling_free_space() -> None:
    """Frontier cells carry the information gain; they must not be left to luck."""
    explorer = NextBestViewExplorer(
        grid_resolution=1.0, waypoint_reach_dist=0.5, n_samples=4, seed=1
    )
    # Walled corridor, open only at the far end: the interior cells have no
    # UNKNOWN neighbour, so (29, 0) is the sole frontier among 30 free cells.
    explorer._grid._free.update({(i, 0) for i in range(30)})
    explorer._grid._occupied.update({(i, 1) for i in range(-1, 30)})
    explorer._grid._occupied.update({(i, -1) for i in range(-1, 30)})
    explorer._grid._occupied.add((-1, 0))

    wp = explorer._pick(0.5, 0.5)
    assert wp is not None
    # With only 4 samples, uniform draws over 30 cells would rarely land on the
    # open end; frontier-first scoring makes it deterministic.
    assert wp.x == 29.5


def test_unknown_gain_still_prefers_the_open_end() -> None:
    grid = OccupancyGrid(resolution=1.0)
    _corridor(grid, n=3)
    assert grid.unknown_region_size([(2, 0)], radius_cells=4) > 0
    assert np.isfinite(grid.unknown_region_size([(0, 0)], radius_cells=4))


# ----------------------------------------------------------------------
# Log diagnostics — the fields the overnight sweep is analysed from


def test_grid_stats_separates_reachable_from_free() -> None:
    """The counter that would have caught the nbv collapse on sight."""
    grid = OccupancyGrid(resolution=1.0)
    _corridor(grid, n=12)

    healthy = grid.stats(0.5, 0.5)
    assert healthy["free"] == 12
    assert healthy["reachable"] == 12
    assert healthy["free_m2"] == 12.0

    # A real obstacle mid-corridor: free stays put, reachable halves.
    grid.mark_occupied(6.5, 0.5, radius_cells=0)
    walled = grid.stats(0.5, 0.5)
    assert walled["free"] == 11
    assert walled["reachable"] == 6

    # Target suppression must NOT show up as a reachability loss.
    grid2 = OccupancyGrid(resolution=1.0)
    _corridor(grid2, n=12)
    grid2.mark_no_target(6.5, 0.5, radius_cells=1)
    suppressed = grid2.stats(0.5, 0.5)
    assert suppressed["reachable"] == 12
    assert suppressed["no_target"] > 0
    assert suppressed["frontier_open"] < suppressed["frontier"]


def test_grid_stats_without_a_pose_omits_reachable() -> None:
    stats = OccupancyGrid(resolution=1.0).stats()
    assert "reachable" not in stats
    assert stats["free"] == 0


def test_frontier_selection_reason_names_the_failure() -> None:
    explorer = FrontierExplorer(grid_resolution=1.0, min_frontier_size=1,
                                max_waypoint_dist=50.0, reject_radius=1.5)
    explorer._grid._free.update({(6, 0), (6, 1)})
    assert explorer._select_frontier(0.5, 0.5) is not None
    assert explorer.selection_diagnostics()["why"] == "ok"

    # Suppressed frontier cells and absent ones are different diagnoses.
    explorer._grid.mark_no_target(6.5, 0.5, radius_cells=3)
    assert explorer._select_frontier(0.5, 0.5) is None
    assert explorer.selection_diagnostics()["why"] == "all_suppressed"

    empty = FrontierExplorer(grid_resolution=1.0)
    assert empty._select_frontier(0.0, 0.0) is None
    assert empty.selection_diagnostics()["why"] == "no_frontier_cells"


def test_frontier_diagnostics_count_rejected_clusters() -> None:
    explorer = FrontierExplorer(grid_resolution=1.0, min_frontier_size=1,
                                max_waypoint_dist=50.0, max_consecutive_skips=99,
                                reject_radius=1.5)
    explorer._grid._free.update({(6, 0), (6, 1)})
    wp = explorer._select_frontier(0.5, 0.5)
    explorer._current_target = wp
    explorer.force_skip()

    explorer._grid._no_target.clear()  # isolate the rejected-centroid path
    assert explorer._select_frontier(0.5, 0.5) is None
    diag = explorer.selection_diagnostics()
    assert diag["why"] == "all_rejected"
    assert diag["rejected"] >= 1


def test_nbv_selection_reason_flags_an_unreachable_map() -> None:
    """free cells but none reachable — the state that killed every big scene."""
    explorer = NextBestViewExplorer(grid_resolution=1.0)
    explorer._grid._free.update({(50, 50), (51, 50)})  # far from the robot
    assert explorer._pick(0.5, 0.5) is None
    assert explorer.selection_diagnostics()["why"] == "unreachable"

    empty = NextBestViewExplorer(grid_resolution=1.0)
    assert empty._pick(0.0, 0.0) is None
    assert empty.selection_diagnostics()["why"] == "no_free_cells"


def test_nbv_diagnostics_report_candidate_counts() -> None:
    explorer = NextBestViewExplorer(grid_resolution=1.0, waypoint_reach_dist=0.5)
    _corridor(explorer._grid, n=12)
    assert explorer._pick(0.5, 0.5) is not None
    diag = explorer.selection_diagnostics()
    assert diag["why"] == "ok"
    assert diag["reachable"] == 12
    assert diag["frontiers"] > 0


def test_skip_hatch_resets_are_counted() -> None:
    explorer = FrontierExplorer(grid_resolution=1.0, max_consecutive_skips=2,
                                skip_reset_free_cells=10)
    explorer._grid._free.update({(i, 0) for i in range(50)})
    assert explorer.skip_hatch_resets == 0
    for x in (1.5, 2.5):
        explorer._current_target = Waypoint(x=x, y=0.5, heading=0.0)
        explorer.force_skip()
    assert explorer.skip_hatch_resets == 1
