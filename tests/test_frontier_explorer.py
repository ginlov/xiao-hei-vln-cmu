"""Unit tests for FrontierExplorer reachability, path-cost scoring, and skip blacklist."""

from __future__ import annotations

from xiao_hei_vln.exploration._frontier import FrontierExplorer
from xiao_hei_vln.exploration._grid import OccupancyGrid
from xiao_hei_vln.messages.outputs import Waypoint


def _fill_free_rect(grid: OccupancyGrid, x0: int, y0: int, x1: int, y1: int) -> None:
    for ix in range(x0, x1 + 1):
        for iy in range(y0, y1 + 1):
            grid._free.add((ix, iy))


def test_reachable_free_cells_stops_at_wall() -> None:
    grid = OccupancyGrid(resolution=0.2)
    # Left room and right room separated by an occupied wall at ix=5.
    _fill_free_rect(grid, 0, 0, 4, 4)
    _fill_free_rect(grid, 6, 0, 10, 4)
    for iy in range(0, 5):
        grid._occupied.add((5, iy))

    left = grid.reachable_free_cells(0.1, 0.1)  # cell (0, 0)
    assert (0, 0) in left
    assert (4, 2) in left
    assert (6, 2) not in left
    assert (10, 0) not in left


def test_path_costs_increase_with_hops() -> None:
    grid = OccupancyGrid(resolution=0.2)
    _fill_free_rect(grid, 0, 0, 5, 0)  # a 1-cell-high corridor
    costs = grid.reachable_path_costs(0.1, 0.1)
    assert costs[(0, 0)] == 0.0
    assert abs(costs[(5, 0)] - 1.0) < 1e-9  # 5 hops * 0.2 m


def test_select_frontier_ignores_unreachable_cluster() -> None:
    explorer = FrontierExplorer(
        max_waypoints=10,
        min_frontier_size=3,
        max_waypoint_dist=10.0,
        waypoint_reach_dist=0.3,
    )
    grid = explorer.get_grid()

    # Robot room on the left; a large frontier-looking free strip on the right
    # behind a wall. Right strip is free and bordered by UNKNOWN → frontier,
    # but unreachable.
    _fill_free_rect(grid, 0, 0, 4, 4)
    _fill_free_rect(grid, 8, 0, 12, 4)
    for iy in range(0, 5):
        grid._occupied.add((5, iy))
        grid._occupied.add((6, iy))
        grid._occupied.add((7, iy))

    # Left room fully enclosed by occupied → no UNKNOWN neighbour.
    for ix in range(0, 5):
        grid._occupied.add((ix, 5))
        grid._occupied.add((ix, -1))
    for iy in range(0, 5):
        grid._occupied.add((-1, iy))

    wp = explorer._select_frontier(rx=0.5, ry=0.5)
    assert wp is None, "unreachable right-room frontier must not be selected"


def test_select_frontier_picks_reachable_over_farther_unreachable() -> None:
    explorer = FrontierExplorer(
        max_waypoints=10,
        min_frontier_size=3,
        max_waypoint_dist=10.0,
        waypoint_reach_dist=0.3,
    )
    grid = explorer.get_grid()

    # Connected free corridor from robot at (0,0) up to a small frontier at y=3
    # (UNKNOWN above iy=3). Separately, a huge unreachable free block far away.
    _fill_free_rect(grid, 0, 0, 2, 3)
    for iy in range(0, 4):
        grid._occupied.add((-1, iy))
        grid._occupied.add((3, iy))
    for ix in range(0, 3):
        grid._occupied.add((ix, -1))

    _fill_free_rect(grid, 20, 0, 30, 10)
    for iy in range(0, 11):
        grid._occupied.add((15, iy))

    wp = explorer._select_frontier(rx=0.3, ry=0.3)
    assert wp is not None
    assert wp.x < 2.0
    assert wp.y < 2.0


def test_select_frontier_prefers_shorter_path_when_sizes_equal() -> None:
    """Equal-size reachable frontiers: lower path cost wins."""
    explorer = FrontierExplorer(
        max_waypoints=10,
        min_frontier_size=3,
        max_waypoint_dist=20.0,
        waypoint_reach_dist=0.05,
    )
    grid = explorer.get_grid()

    # Floor connecting robot to two equal spurs.
    _fill_free_rect(grid, 0, 0, 10, 0)
    # Near spur (3 free cells) at ix=2.
    _fill_free_rect(grid, 2, 1, 2, 3)
    # Far spur (3 free cells) at ix=9.
    _fill_free_rect(grid, 9, 1, 9, 3)

    for iy in range(1, 4):
        grid._occupied.add((1, iy))
        grid._occupied.add((3, iy))
        grid._occupied.add((8, iy))
        grid._occupied.add((10, iy))
    for ix in range(0, 11):
        grid._occupied.add((ix, -1))

    wp = explorer._select_frontier(rx=0.1, ry=0.1)
    assert wp is not None
    # Near spur is around x=0.5; far spur around x=1.9.
    assert wp.x < 1.2, f"expected nearer spur, got {wp}"


def test_pick_goal_cell_prefers_clearance_from_occupied() -> None:
    explorer = FrontierExplorer(
        max_waypoints=10,
        min_frontier_size=3,
        max_waypoint_dist=10.0,
        waypoint_reach_dist=0.05,
    )
    grid = explorer.get_grid()
    # Free strip; occupied wall along left. Frontier cells at top.
    _fill_free_rect(grid, 0, 0, 4, 3)
    for iy in range(0, 4):
        grid._occupied.add((-1, iy))
    for ix in range(0, 5):
        grid._occupied.add((ix, -1))
    # Leave y=4 unknown so iy=3 is frontier.

    path_costs = grid.reachable_path_costs(0.5, 0.5)
    cluster = [(ix, 3) for ix in range(0, 5)]
    cell = explorer._pick_goal_cell(cluster, path_costs)
    assert cell is not None
    # Should prefer cells farther from the occupied column at ix=-1 → higher ix.
    assert cell[0] >= 2, f"expected clearance-preferring cell, got {cell}"


def _corridor_map(explorer: FrontierExplorer) -> None:
    grid = explorer.get_grid()
    _fill_free_rect(grid, 0, 0, 2, 3)
    for iy in range(0, 4):
        grid._occupied.add((-1, iy))
        grid._occupied.add((3, iy))
    for ix in range(0, 3):
        grid._occupied.add((ix, -1))


def test_force_skip_soft_bans_without_wiping_free_cells() -> None:
    explorer = FrontierExplorer(
        max_waypoints=10,
        min_frontier_size=3,
        max_waypoint_dist=10.0,
        waypoint_reach_dist=0.3,
        soft_clear_dist_m=1.0,
        soft_cooldown_s=30.0,
        hard_ban_fail_count=3,
        hard_ban_pose_sep_m=2.0,
    )
    _corridor_map(explorer)
    grid = explorer.get_grid()

    wp = explorer._select_frontier(rx=0.3, ry=0.3)
    assert wp is not None
    cells_before = list(explorer._current_target_cells)
    assert cells_before
    explorer._current_target = wp
    explorer._last_rx, explorer._last_ry, explorer._last_now = 0.3, 0.3, 1.0
    explorer.force_skip()

    # Soft ban must not permanently erase FREE cells.
    for cell in cells_before:
        assert cell in grid.free_cells
    assert explorer._current_target is None

    # Same pose: region stays soft-banned → not re-selected.
    explorer._refresh_soft_bans(0.3, 0.3, 2.0)
    assert explorer._select_frontier(rx=0.3, ry=0.3) is None


def test_soft_ban_lifts_after_robot_moves() -> None:
    explorer = FrontierExplorer(
        max_waypoints=10,
        min_frontier_size=3,
        max_waypoint_dist=10.0,
        waypoint_reach_dist=0.05,
        soft_clear_dist_m=1.0,
        soft_cooldown_s=30.0,
        hard_ban_fail_count=3,
        hard_ban_pose_sep_m=2.0,
    )
    _corridor_map(explorer)

    wp = explorer._select_frontier(rx=0.3, ry=0.3)
    assert wp is not None
    explorer._current_target = wp
    explorer._last_rx, explorer._last_ry, explorer._last_now = 0.3, 0.3, 1.0
    explorer.force_skip()

    # Move > soft_clear_dist → soft ban lifts; frontier can be tried again.
    explorer._refresh_soft_bans(0.3, 1.5, 2.0)
    wp2 = explorer._select_frontier(rx=0.3, ry=1.5)
    assert wp2 is not None


def test_hard_ban_after_three_separated_fail_poses() -> None:
    explorer = FrontierExplorer(
        max_waypoints=10,
        min_frontier_size=3,
        max_waypoint_dist=20.0,
        waypoint_reach_dist=0.05,
        soft_clear_dist_m=0.5,
        soft_cooldown_s=30.0,
        hard_ban_fail_count=3,
        hard_ban_pose_sep_m=2.0,
    )
    grid = explorer.get_grid()
    # Large connected floor so three fail poses ≥2 m apart still see the same
    # top-edge frontier.
    _fill_free_rect(grid, 0, 0, 20, 15)
    for iy in range(0, 16):
        grid._occupied.add((-1, iy))
        grid._occupied.add((21, iy))
    for ix in range(0, 21):
        grid._occupied.add((ix, -1))
    # iy=16 left UNKNOWN → iy=15 is the frontier strip.

    fail_poses = [(0.5, 0.5), (0.5, 2.7), (2.7, 0.5)]
    cells_snapshot: list[tuple[int, int]] = []
    for i, (rx, ry) in enumerate(fail_poses):
        explorer._refresh_soft_bans(rx, ry, float(i + 1))
        wp = explorer._select_frontier(rx=rx, ry=ry)
        assert wp is not None, f"expected frontier before fail {i + 1}"
        if not cells_snapshot:
            cells_snapshot = list(explorer._current_target_cells)
        explorer._current_target = wp
        explorer._last_rx, explorer._last_ry, explorer._last_now = rx, ry, float(i + 1)
        explorer.force_skip()

    assert len(cells_snapshot) > 0
    for cell in cells_snapshot:
        assert cell not in grid.free_cells, "third separated fail should hard-ban cells"

    explorer._refresh_soft_bans(1.5, 1.5, 10.0)
    assert explorer._select_frontier(rx=1.5, ry=1.5) is None


def test_unknown_region_size_larger_for_open_chamber() -> None:
    grid = OccupancyGrid(resolution=0.2)
    # Tiny sealed cavity: free (1,0) touches ONLY unknown (1,1); all other
    # neighbours occupied so flood cannot escape into the outdoor UNKNOWN.
    grid._free.add((1, 0))
    grid._occupied.update({
        (0, 0), (2, 0), (1, -1),
        (0, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
    })
    tiny = [(1, 0)]

    # Doorway into open UNKNOWN above (large gain within radius).
    grid._free.add((10, 0))
    grid._occupied.update({(9, 0), (11, 0), (10, -1)})
    open_front = [(10, 0)]

    tiny_gain = grid.unknown_region_size(tiny, radius_cells=10)
    open_gain = grid.unknown_region_size(open_front, radius_cells=10)
    assert tiny_gain == 1, tiny_gain
    assert open_gain > tiny_gain
    assert open_gain >= 20


def test_select_frontier_prefers_larger_unknown_gain() -> None:
    explorer = FrontierExplorer(
        max_waypoints=10,
        min_frontier_size=1,
        max_waypoint_dist=20.0,
        waypoint_reach_dist=0.05,
    )
    grid = explorer.get_grid()
    # Floor with sealed bottom/sides so only the two spurs are frontiers.
    _fill_free_rect(grid, 0, 0, 12, 0)
    for ix in range(0, 13):
        grid._occupied.add((ix, -1))
        if ix not in (2, 10):
            grid._occupied.add((ix, 1))
    grid._occupied.add((-1, 0))
    grid._occupied.add((13, 0))

    # Near spur: 2-cell frontier into a 1-cell sealed UNKNOWN cavity.
    grid._free.update({(2, 1), (2, 2)})
    grid._occupied.update({
        (1, 1), (3, 1), (1, 2), (3, 2),
        (1, 3), (3, 3), (1, 4), (2, 4), (3, 4),
    })
    # Leave (2, 3) UNKNOWN — the only cavity cell.

    # Far spur: 2-cell frontier into open UNKNOWN.
    grid._free.update({(10, 1), (10, 2)})
    grid._occupied.update({(9, 1), (11, 1), (9, 2), (11, 2)})
    # (10, 3) and beyond stay UNKNOWN → large gain.

    near_gain = grid.unknown_region_size([(2, 1), (2, 2)], radius_cells=15)
    far_gain = grid.unknown_region_size([(10, 1), (10, 2)], radius_cells=15)
    assert far_gain > near_gain, (near_gain, far_gain)

    wp = explorer._select_frontier(rx=0.1, ry=0.1)
    assert wp is not None
    assert wp.x > 1.5, f"expected high-UNKNOWN doorway, got {wp}"


def test_soft_ban_rescue_before_no_frontiers() -> None:
    """When only soft-banned frontiers remain, clear bans once and replan."""
    from xiao_hei_vln.messages import Header, OdomPose, Quaternion, Stamp, Vector3, VLMInput

    explorer = FrontierExplorer(
        max_waypoints=10,
        min_frontier_size=3,
        max_waypoint_dist=10.0,
        waypoint_reach_dist=0.05,
        soft_clear_dist_m=10.0,  # don't auto-clear by motion
        soft_cooldown_s=1e9,
        max_soft_rescues=1,
        hard_ban_fail_count=99,
    )
    _corridor_map(explorer)

    wp = explorer._select_frontier(rx=0.3, ry=0.3)
    assert wp is not None
    explorer._current_target = wp
    explorer._last_rx, explorer._last_ry, explorer._last_now = 0.3, 0.3, 1.0
    explorer.force_skip()
    assert explorer._has_active_soft_bans()
    assert explorer._select_frontier(rx=0.3, ry=0.3) is None

    snap = VLMInput(
        tick_id=1,
        tick_time=Stamp.from_seconds(2.0),
        pose=OdomPose(
            header=Header(stamp=Stamp.from_seconds(2.0), frame_id="map"),
            position=Vector3(x=0.3, y=0.3, z=0.0),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    )
    out = explorer.update(snap)
    assert out is not None, "soft-ban rescue should replan instead of finishing"
    assert explorer._soft_rescue_count == 1
    assert not explorer.is_complete()
