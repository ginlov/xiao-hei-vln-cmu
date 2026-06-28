"""Frontier-based exploration strategy.

Each call to `update()` consumes one VLMInput snapshot and returns the
current target Waypoint (or None if exploration is complete / not ready).

Algorithm per tick
------------------
1. Ingest terrain_ext into OccupancyGrid to expand the known map.
2. If the robot is within `waypoint_reach_dist` of the current target,
   mark that waypoint as visited and clear the target.
3. If the robot has not reached the target within `stuck_timeout_s` seconds,
   skip the waypoint and select a new frontier.
4. If the budget is exhausted, mark done and return None.
5. If no target is set, find frontier cells, cluster them, score each
   cluster by size / (1 + distance) and pick the best centroid.
6. Return the current target waypoint.
"""

from __future__ import annotations

import math
from collections import deque

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
    ) -> None:
        self._max_waypoints = max_waypoints
        self._reach_dist = waypoint_reach_dist
        self._min_frontier_size = min_frontier_size
        self._cost_threshold = cost_threshold
        self._stuck_timeout_s = stuck_timeout_s
        self._max_waypoint_dist = max_waypoint_dist
        self._max_consecutive_skips = max_consecutive_skips

        self._grid = OccupancyGrid(grid_resolution)
        self._current_target: Waypoint | None = None
        self._visited: list[Waypoint] = []
        self._done = False

        # Stuck detection state — reset whenever a new target is assigned.
        self._target_set_time: float | None = None
        self.skipped_count: int = 0
        self._consecutive_skip_count: int = 0

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

        # Advance when robot reaches the current target via odometry.
        if self._current_target is not None and self._within_reach(rx, ry, self._current_target):
            self._visited.append(self._current_target)
            self._current_target = None
            self._target_set_time = None
            self._consecutive_skip_count = 0

        # Stuck detection: skip if timeout exceeded without nav-stack advance.
        if self._current_target is not None:
            elapsed = now - self._target_set_time  # type: ignore[operator]
            if elapsed > self._stuck_timeout_s:
                self.skipped_count += 1
                self._consecutive_skip_count += 1
                # Mark a larger area so we don't re-select the same unreachable frontier.
                self._grid.mark_occupied(self._current_target.x, self._current_target.y, radius_cells=3)
                self._current_target = None
                self._target_set_time = None
                if self._consecutive_skip_count >= self._max_consecutive_skips:
                    self._done = True
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
                self._done = True
                return None
            self._target_set_time = now

        return self._current_target

    def force_skip(self) -> None:
        """Skip the current target immediately — call when nav stack has demonstrably settled
        above the advance threshold with no improvement."""
        if self._current_target is None:
            return
        self.skipped_count += 1
        self._consecutive_skip_count += 1
        self._grid.mark_occupied(self._current_target.x, self._current_target.y, radius_cells=3)
        self._current_target = None
        self._target_set_time = None
        if self._consecutive_skip_count >= self._max_consecutive_skips:
            self._done = True

    def advance(self) -> None:
        """Mark the current target as visited — call when nav stack signals arrival."""
        if self._current_target is not None:
            # Suppress this frontier so it isn't re-selected next tick.
            self._grid.mark_occupied(self._current_target.x, self._current_target.y, radius_cells=1)
            self._visited.append(self._current_target)
            self._current_target = None
            self._target_set_time = None
            self._consecutive_skip_count = 0

    def is_complete(self) -> bool:
        return self._done

    def reset(self) -> None:
        self._grid = OccupancyGrid(self._grid.resolution)
        self._current_target = None
        self._visited = []
        self._done = False
        self._target_set_time = None
        self.skipped_count = 0
        self._consecutive_skip_count = 0

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

    # ------------------------------------------------------------------
    # Internals

    def _select_frontier(self, rx: float, ry: float) -> Waypoint | None:
        raw_cells = self._grid.frontier_cells()
        if not raw_cells:
            return None

        clusters = _cluster_frontier(raw_cells)
        clusters = [c for c in clusters if len(c) >= self._min_frontier_size]
        if not clusters:
            return None

        best_wp: Waypoint | None = None
        best_score = -1.0
        nearest_wp: Waypoint | None = None
        nearest_dist = float("inf")

        for cluster in clusters:
            xs = [self._grid.to_world(ix, iy)[0] for ix, iy in cluster]
            ys = [self._grid.to_world(ix, iy)[1] for ix, iy in cluster]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)

            dist = math.hypot(cx - rx, cy - ry)
            if dist < 1e-6:
                dist = 1e-6

            wp = Waypoint(x=cx, y=cy, heading=math.atan2(cy - ry, cx - rx))

            # Skip targets the robot is already at — assigning them causes
            # immediate false-visits without the robot moving anywhere.
            if dist <= self._reach_dist:
                continue

            # Track nearest VALID (outside reach_dist) cluster as fallback.
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_wp = wp

            # Only score clusters within the preferred distance cap.
            if dist > self._max_waypoint_dist:
                continue

            score = len(cluster) / (1.0 + dist)
            if score > best_score:
                best_score = score
                best_wp = wp

        # Fall back to the nearest cluster when all exceeded max_waypoint_dist.
        if best_wp is None:
            best_wp = nearest_wp

        return best_wp

    def _within_reach(self, rx: float, ry: float, wp: Waypoint) -> bool:
        return math.hypot(rx - wp.x, ry - wp.y) < self._reach_dist
