"""Frontier-based exploration strategy.

Each call to `update()` consumes one VLMInput snapshot and returns the
current target Waypoint (or None if exploration is complete / not ready).

Algorithm per tick
------------------
1. Ingest terrain_ext into OccupancyGrid to expand the known map.
2. If the robot is within `waypoint_reach_dist` of the current target,
   mark that waypoint as visited and clear the target.
3. If the robot has not reached the target within `stuck_timeout_s` seconds,
   soft-ban the frontier from this pose (retry later) and select another.
   Hard-ban only after failures from ``hard_ban_fail_count`` poses that are
   each ≥ ``hard_ban_pose_sep_m`` apart.
4. If the budget is exhausted, mark done and return None.
5. If no target is set, find frontier cells, cluster them, keep only
   clusters reachable from the robot via FREE-space BFS (and not soft-banned),
   score each by (cluster_size + unknown_region_size) / (1 + path_cost) so
   doorways into large unseen space beat local fringe.
6. Before declaring no_frontiers, clear active soft bans once and replan so a
   skipped doorway can be retried from the current pose.
7. Return the current target waypoint.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from xiao_hei_vln.exploration._grid import OccupancyGrid
from xiao_hei_vln.messages.inputs import VLMInput
from xiao_hei_vln.messages.outputs import Waypoint


def _cluster_frontier(cells: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Group adjacent frontier cells into connected clusters (4-connectivity)."""
    cell_set = set(cells)
    visited: set[tuple[int, int]] = set()
    clusters: list[list[tuple[int, int]]] = []

    for seed in cells:
        if seed in visited:
            continue
        cluster: list[tuple[int, int]] = []
        queue: deque[tuple[int, int]] = deque([seed])
        while queue:
            cell = queue.popleft()
            if cell in visited:
                continue
            visited.add(cell)
            cluster.append(cell)
            ix, iy = cell
            for nx, ny in ((ix + 1, iy), (ix - 1, iy), (ix, iy + 1), (ix, iy - 1)):
                if (nx, ny) in cell_set and (nx, ny) not in visited:
                    queue.append((nx, ny))
        clusters.append(cluster)

    return clusters


@dataclass
class _RegionFailState:
    """Per-frontier-region skip memory (soft ban + distinct fail poses)."""

    cells: set[tuple[int, int]] = field(default_factory=set)
    fail_poses: list[tuple[float, float]] = field(default_factory=list)
    soft_pose: tuple[float, float] | None = None
    soft_time: float | None = None
    soft_active: bool = False
    hard_banned: bool = False


class FrontierExplorer:
    """Reactive frontier-based exploration strategy.

    Stops when `max_waypoints` have been visited (budget exhausted) or when
    no frontier clusters remain (full coverage).
    """

    def __init__(
        self,
        max_waypoints: int = 30,
        grid_resolution: float = 0.2,
        waypoint_reach_dist: float = 0.3,
        min_frontier_size: int = 5,
        cost_threshold: float = 0.5,
        stuck_timeout_s: float = 30.0,
        max_waypoint_dist: float = 1.5,
        max_consecutive_skips: int = 3,
        soft_clear_dist_m: float = 1.0,
        soft_cooldown_s: float = 30.0,
        hard_ban_fail_count: int = 3,
        hard_ban_pose_sep_m: float = 2.0,
        max_soft_rescues: int = 1,
        unknown_gain_weight: float = 1.0,
    ) -> None:
        self._max_waypoints = max_waypoints
        self._reach_dist = waypoint_reach_dist
        self._min_frontier_size = min_frontier_size
        self._cost_threshold = cost_threshold
        self._stuck_timeout_s = stuck_timeout_s
        self._max_waypoint_dist = max_waypoint_dist
        self._max_consecutive_skips = max_consecutive_skips
        self._soft_clear_dist_m = soft_clear_dist_m
        self._soft_cooldown_s = soft_cooldown_s
        self._hard_ban_fail_count = hard_ban_fail_count
        self._hard_ban_pose_sep_m = hard_ban_pose_sep_m
        self._max_soft_rescues = max_soft_rescues
        self._unknown_gain_weight = unknown_gain_weight

        self._grid = OccupancyGrid(grid_resolution)
        self._current_target: Waypoint | None = None
        self._current_target_cells: list[tuple[int, int]] = []
        self._current_region_key: tuple[int, int] | None = None
        self._visited: list[Waypoint] = []
        self._done = False

        # Stuck detection state — reset whenever a new target is assigned.
        self._target_set_time: float | None = None
        self.skipped_count: int = 0
        self._consecutive_skip_count: int = 0

        # Last known robot pose/time (for force_skip from the nav tick).
        self._last_rx: float = 0.0
        self._last_ry: float = 0.0
        self._last_now: float = 0.0

        # region_key → fail / soft-ban state. Soft bans lift after the robot
        # moves or cooldown; hard ban only after enough separated fail poses.
        self._region_fails: dict[tuple[int, int], _RegionFailState] = {}
        # How many times we have cleared soft bans to avoid a premature
        # no_frontiers finish. Resets after a successful visit.
        self._soft_rescue_count: int = 0

    # ------------------------------------------------------------------
    # Strategy interface

    def update(self, snapshot: VLMInput) -> Waypoint | None:
        """Consume one tick snapshot; return the waypoint to publish or None."""
        if self._done:
            return None

        if snapshot.terrain_ext is not None:
            self._grid.update(snapshot.terrain_ext, self._cost_threshold)

        pose = snapshot.pose
        if pose is None:
            return self._current_target

        rx, ry = pose.position.x, pose.position.y
        now = snapshot.tick_time.to_seconds()
        self._last_rx, self._last_ry, self._last_now = rx, ry, now
        self._refresh_soft_bans(rx, ry, now)

        # Advance when robot reaches the current target via odometry.
        if self._current_target is not None and self._within_reach(rx, ry, self._current_target):
            if self._current_region_key is not None:
                self._region_fails.pop(self._current_region_key, None)
            self._visited.append(self._current_target)
            self._current_target = None
            self._current_target_cells = []
            self._current_region_key = None
            self._target_set_time = None
            self._consecutive_skip_count = 0
            self._soft_rescue_count = 0

        # Stuck detection: skip if timeout exceeded without nav-stack advance.
        if self._current_target is not None:
            elapsed = now - self._target_set_time  # type: ignore[operator]
            if elapsed > self._stuck_timeout_s:
                self._skip_current_target(rx, ry, now)
                if self._done:
                    return None

        # Budget check
        if len(self._visited) >= self._max_waypoints:
            self._done = True
            return None

        # Select a new frontier target when needed.
        if self._current_target is None:
            self._current_target = self._select_frontier(rx, ry)
            if self._current_target is None:
                if not self._grid.free_cells:
                    return None  # waiting for terrain data
                # Robot not seated in FREE yet (or no path costs) — keep waiting
                # instead of declaring the map fully explored.
                if not self._grid.reachable_path_costs(rx, ry):
                    return None
                # Soft bans may be hiding the only doorway into unseen space.
                # Clear them once (per visit wave) and replan before finishing.
                if (
                    self._has_active_soft_bans()
                    and self._soft_rescue_count < self._max_soft_rescues
                ):
                    self._soft_rescue_count += 1
                    self._clear_soft_bans()
                    self._current_target = self._select_frontier(rx, ry)
                if self._current_target is None:
                    self._done = True
                    return None
            self._target_set_time = now

        return self._current_target

    def force_skip(self) -> None:
        """Skip the current target immediately — call when nav stack has demonstrably settled
        above the advance threshold with no improvement, or never engaged (inf)."""
        if self._current_target is None:
            return
        self._skip_current_target(self._last_rx, self._last_ry, self._last_now)

    def advance(self) -> None:
        """Mark the current target as visited — call when nav stack signals arrival."""
        if self._current_target is not None:
            # Suppress this frontier so it isn't re-selected next tick.
            if self._current_target_cells:
                self._grid.mark_cells(self._current_target_cells, radius_cells=1)
            else:
                self._grid.mark_occupied(
                    self._current_target.x, self._current_target.y, radius_cells=1
                )
            if self._current_region_key is not None:
                self._region_fails.pop(self._current_region_key, None)
            self._visited.append(self._current_target)
            self._current_target = None
            self._current_target_cells = []
            self._current_region_key = None
            self._target_set_time = None
            self._consecutive_skip_count = 0
            self._soft_rescue_count = 0

    def is_complete(self) -> bool:
        return self._done

    def reset(self) -> None:
        self._grid = OccupancyGrid(self._grid.resolution)
        self._current_target = None
        self._current_target_cells = []
        self._current_region_key = None
        self._visited = []
        self._done = False
        self._target_set_time = None
        self.skipped_count = 0
        self._consecutive_skip_count = 0
        self._region_fails.clear()
        self._soft_rescue_count = 0
        self._last_rx = self._last_ry = self._last_now = 0.0

    # ------------------------------------------------------------------
    # Accessors for visualisation / reporting

    def get_visited_waypoints(self) -> list[Waypoint]:
        """All completed waypoints plus the in-progress target (if any)."""
        wps = list(self._visited)
        if self._current_target is not None:
            wps.append(self._current_target)
        return wps

    def get_grid(self) -> OccupancyGrid:
        return self._grid

    def get_viz_layers(self) -> dict:
        """Snapshot for RViz: free / frontier / soft-ban / hard-ban / path / goal.

        World XY lists (metres). Soft-ban = temporarily skipped; hard-ban =
        permanently rejected after multi-pose fails.
        """
        grid = self._grid
        free_xy = [grid.to_world(ix, iy) for ix, iy in grid.free_cells]
        frontier_xy = [grid.to_world(ix, iy) for ix, iy in grid.frontier_cells()]

        soft_cells: set[tuple[int, int]] = set()
        hard_cells: set[tuple[int, int]] = set()
        for state in self._region_fails.values():
            if state.hard_banned:
                hard_cells |= state.cells
            elif state.soft_active:
                soft_cells |= state.cells

        visited_xy = [(wp.x, wp.y) for wp in self._visited]
        current = None
        if self._current_target is not None:
            current = (self._current_target.x, self._current_target.y)

        return {
            "free": free_xy,
            "frontier": frontier_xy,
            "soft_ban": [grid.to_world(ix, iy) for ix, iy in soft_cells],
            "hard_ban": [grid.to_world(ix, iy) for ix, iy in hard_cells],
            "visited": visited_xy,
            "current": current,
            "resolution": grid.resolution,
        }

    # ------------------------------------------------------------------
    # Internals

    def _skip_current_target(self, rx: float, ry: float, now: float) -> None:
        """Soft-ban the active frontier from this pose; hard-ban only after enough separated fails."""
        if self._current_target is None:
            return
        self.skipped_count += 1
        self._consecutive_skip_count += 1

        cells = set(self._current_target_cells)
        region_key = self._current_region_key or self._region_key_from_waypoint(
            self._current_target
        )
        state = self._region_fails.setdefault(region_key, _RegionFailState())
        if cells:
            state.cells |= cells
        else:
            cx, cy = self._grid.world_to_grid(self._current_target.x, self._current_target.y)
            state.cells.add((cx, cy))

        # Record this fail pose only if it is far enough from prior fails.
        pose = (rx, ry)
        if self._is_new_fail_pose(state.fail_poses, pose):
            state.fail_poses.append(pose)

        if len(state.fail_poses) >= self._hard_ban_fail_count:
            # Confirmed unreachable from several distant poses → permanent ban.
            self._grid.mark_cells(list(state.cells), radius_cells=0)
            state.hard_banned = True
            state.soft_active = False
            state.soft_pose = None
            state.soft_time = None
        else:
            # Temporary: don't wipe FREE cells; just avoid re-picking from here/now.
            state.soft_active = True
            state.soft_pose = pose
            state.soft_time = now

        self._current_target = None
        self._current_target_cells = []
        self._current_region_key = None
        self._target_set_time = None
        if self._consecutive_skip_count >= self._max_consecutive_skips:
            self._done = True

    def _select_frontier(self, rx: float, ry: float) -> Waypoint | None:
        raw_cells = self._grid.frontier_cells()
        if not raw_cells:
            return None

        clusters = _cluster_frontier(raw_cells)
        clusters = [c for c in clusters if len(c) >= self._min_frontier_size]
        if not clusters:
            return None

        # Classic Yamauchi gate + path costs: only consider frontiers in the
        # robot's connected FREE component, scored by grid path length.
        path_costs = self._grid.reachable_path_costs(rx, ry)
        if not path_costs:
            return None
        clusters = [
            c for c in clusters if any(cell in path_costs for cell in c)
        ]
        if not clusters:
            return None

        # Drop soft-banned / hard-banned regions (hard-banned cells are also
        # already removed from FREE, but region keys catch remapped clusters).
        clusters = [c for c in clusters if not self._is_cluster_banned(c)]
        if not clusters:
            return None

        best_wp: Waypoint | None = None
        best_cells: list[tuple[int, int]] = []
        best_region: tuple[int, int] | None = None
        best_score = -1.0
        nearest_wp: Waypoint | None = None
        nearest_cells: list[tuple[int, int]] = []
        nearest_region: tuple[int, int] | None = None
        nearest_path = float("inf")

        for cluster in clusters:
            goal_cell = self._pick_goal_cell(cluster, path_costs)
            if goal_cell is None:
                continue
            cx, cy = self._grid.to_world(*goal_cell)
            region_key = self._region_key_from_cells(cluster)

            # Path cost to the chosen goal cell.
            path_cost = path_costs.get(goal_cell, min(
                path_costs[cell] for cell in cluster if cell in path_costs
            ))
            if path_cost < 1e-6:
                path_cost = 1e-6

            euclid = math.hypot(cx - rx, cy - ry)
            wp = Waypoint(x=cx, y=cy, heading=math.atan2(cy - ry, cx - rx))

            # Skip targets the robot is already at — assigning them causes
            # immediate false-visits without the robot moving anywhere.
            if euclid <= self._reach_dist:
                continue

            # Track nearest VALID cluster by path cost as fallback.
            if path_cost < nearest_path:
                nearest_path = path_cost
                nearest_wp = wp
                nearest_cells = list(cluster)
                nearest_region = region_key

            # Only score clusters within the preferred path-distance cap.
            if path_cost > self._max_waypoint_dist:
                continue

            # Prefer frontiers that open large UNKNOWN regions (doorways into
            # the other half of a room) over local fringe of equal size.
            unknown_gain = self._grid.unknown_region_size(cluster)
            score = (len(cluster) + self._unknown_gain_weight * unknown_gain) / (
                1.0 + path_cost
            )
            if score > best_score:
                best_score = score
                best_wp = wp
                best_cells = list(cluster)
                best_region = region_key

        # Fall back to the nearest *reachable* cluster when all exceeded
        # max_waypoint_dist.
        if best_wp is None:
            best_wp = nearest_wp
            best_cells = nearest_cells
            best_region = nearest_region

        self._current_target_cells = best_cells
        self._current_region_key = best_region
        return best_wp

    def _pick_goal_cell(
        self,
        cluster: list[tuple[int, int]],
        path_costs: dict[tuple[int, int], float],
    ) -> tuple[int, int] | None:
        """Choose a navigable cell in ``cluster`` instead of the raw centroid.

        Prefers FREE cells with clearance from OCCUPIED neighbours (walls /
        furniture). Falls back to the path-nearest free cell if nothing has
        clearance — still better than a centroid that can sit inside a wall.
        """
        reachable = [c for c in cluster if c in path_costs]
        if not reachable:
            return None

        clear = [c for c in reachable if self._grid.is_clear_for_goal(c[0], c[1], clearance_cells=1)]
        pool = clear if clear else reachable

        # Among candidates: maximize clearance, then minimize path cost.
        def _key(cell: tuple[int, int]) -> tuple[int, float]:
            return (self._grid.clearance_score(cell[0], cell[1]), -path_costs[cell])

        return max(pool, key=_key)

    def _within_reach(self, rx: float, ry: float, wp: Waypoint) -> bool:
        return math.hypot(rx - wp.x, ry - wp.y) < self._reach_dist

    # ------------------------------------------------------------------
    # Soft / hard ban helpers

    def _region_key_from_cells(self, cells: list[tuple[int, int]]) -> tuple[int, int]:
        cx = sum(c[0] for c in cells) / len(cells)
        cy = sum(c[1] for c in cells) / len(cells)
        wx, wy = self._grid.to_world(int(round(cx)), int(round(cy)))
        # 1 m bins — stable across small cluster-centroid jitter.
        return (int(round(wx)), int(round(wy)))

    def _region_key_from_waypoint(self, wp: Waypoint) -> tuple[int, int]:
        return (int(round(wp.x)), int(round(wp.y)))

    def _is_new_fail_pose(
        self,
        existing: list[tuple[float, float]],
        pose: tuple[float, float],
    ) -> bool:
        if not existing:
            return True
        return all(
            math.hypot(pose[0] - px, pose[1] - py) >= self._hard_ban_pose_sep_m
            for px, py in existing
        )

    def _refresh_soft_bans(self, rx: float, ry: float, now: float) -> None:
        for state in self._region_fails.values():
            if not state.soft_active or state.soft_pose is None or state.soft_time is None:
                continue
            moved = math.hypot(rx - state.soft_pose[0], ry - state.soft_pose[1])
            if moved >= self._soft_clear_dist_m or (now - state.soft_time) >= self._soft_cooldown_s:
                state.soft_active = False

    def _has_active_soft_bans(self) -> bool:
        return any(s.soft_active and not s.hard_banned for s in self._region_fails.values())

    def _clear_soft_bans(self) -> None:
        for state in self._region_fails.values():
            if state.hard_banned:
                continue
            state.soft_active = False
            state.soft_pose = None
            state.soft_time = None

    def _is_cluster_banned(self, cluster: list[tuple[int, int]]) -> bool:
        key = self._region_key_from_cells(cluster)
        state = self._region_fails.get(key)
        if state is not None:
            if state.hard_banned or state.soft_active:
                return True
        # Also match by cell overlap in case the centroid bin drifted.
        cluster_set = set(cluster)
        for state in self._region_fails.values():
            if not (state.hard_banned or state.soft_active):
                continue
            if state.cells & cluster_set:
                return True
        return False
