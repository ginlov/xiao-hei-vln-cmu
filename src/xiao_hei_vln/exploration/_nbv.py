"""Next-Best-View exploration on the online belief occupancy map.

Uses only ``terrain_ext`` + pose (no GT floor plan). Candidates are sampled
from reachable FREE cells (frontier-biased + wall-proximity-biased) and
scored as:

    score = (unknown_gain + wall_weight * wall_touch) / (1 + path_cost)

Heading faces the nearest occupied cell when possible so the camera looks
at walls/furniture (where referring-expression targets live), not only
along the travel direction.

Visited waypoints are blacklisted with a small radius to reduce local thrashing.
"""

from __future__ import annotations

import math
import random

from xiao_hei_vln.exploration._grid import OccupancyGrid
from xiao_hei_vln.messages.inputs import VLMInput
from xiao_hei_vln.messages.outputs import Waypoint


class NextBestViewExplorer:
    """Sample candidate FREE poses; pick max (unknown+wall) gain / path cost."""

    def __init__(
        self,
        *,
        max_waypoints: int = 100,
        grid_resolution: float = 0.2,
        waypoint_reach_dist: float = 0.45,
        cost_threshold: float = 0.5,
        stuck_timeout_s: float = 12.0,
        max_consecutive_skips: int = 25,
        n_samples: int = 40,
        visit_mark_radius_cells: int = 3,
        # Mild wall bias: look-at-wall heading does most of the work;
        # heavy wall_weight pulled goals into unnavigable pockets.
        wall_weight: float = 0.45,
        wall_radius_cells: int = 8,
        seed: int = 0,
    ) -> None:
        self._max_waypoints = max_waypoints
        self._reach_dist = waypoint_reach_dist
        self._cost_threshold = cost_threshold
        self._stuck_timeout_s = stuck_timeout_s
        self._max_consecutive_skips = max_consecutive_skips
        self._n_samples = n_samples
        self._visit_mark_radius_cells = visit_mark_radius_cells
        self._wall_weight = wall_weight
        self._wall_radius_cells = wall_radius_cells
        self._rng = random.Random(seed)

        self._grid = OccupancyGrid(grid_resolution)
        self._current_target: Waypoint | None = None
        self._visited: list[Waypoint] = []
        self._done = False
        self._target_set_time: float | None = None
        self.skipped_count = 0
        self._consecutive_skip_count = 0
        self._blacklist: set[tuple[int, int]] = set()

    def update(self, snapshot: VLMInput) -> Waypoint | None:
        if self._done:
            return None
        if snapshot.terrain_ext is not None:
            self._grid.update(snapshot.terrain_ext, self._cost_threshold)
        pose = snapshot.pose
        if pose is None:
            return self._current_target
        rx, ry = pose.position.x, pose.position.y
        now = snapshot.tick_time.to_seconds()

        if self._current_target is not None and self._within(rx, ry, self._current_target):
            self._visited.append(self._current_target)
            self._mark(self._current_target)
            self._current_target = None
            self._target_set_time = None
            self._consecutive_skip_count = 0

        if self._current_target is not None and self._target_set_time is not None:
            if now - self._target_set_time > self._stuck_timeout_s:
                self._skip(self._current_target)
                if self._done:
                    return None

        if len(self._visited) >= self._max_waypoints:
            self._done = True
            return None

        if self._current_target is None:
            self._current_target = self._pick(rx, ry)
            if self._current_target is None:
                if not self._grid.free_cells or not self._grid.reachable_path_costs(rx, ry):
                    return None
                self._done = True
                return None
            self._target_set_time = now
        return self._current_target

    def force_skip(self) -> None:
        if self._current_target is not None:
            self._skip(self._current_target)

    def advance(self) -> None:
        if self._current_target is not None:
            self._visited.append(self._current_target)
            self._mark(self._current_target)
            self._current_target = None
            self._target_set_time = None
            self._consecutive_skip_count = 0

    def force_complete(self) -> None:
        """End exploration immediately (e.g. wall-clock hard stop)."""
        self._done = True
        self._current_target = None
        self._target_set_time = None

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
        self._blacklist.clear()

    def get_visited_waypoints(self) -> list[Waypoint]:
        wps = list(self._visited)
        if self._current_target is not None:
            wps.append(self._current_target)
        return wps

    def get_grid(self) -> OccupancyGrid:
        return self._grid

    def _pick(self, rx: float, ry: float) -> Waypoint | None:
        costs = self._grid.reachable_path_costs(rx, ry)
        if not costs:
            return None
        candidates = [c for c in costs if c not in self._blacklist and costs[c] > self._reach_dist]
        if not candidates:
            candidates = [c for c in costs if c not in self._blacklist]
        if not candidates:
            return None

        frontiers = [
            c for c in self._grid.frontier_cells()
            if c in costs and c not in self._blacklist
        ]
        # Prefer open free cells that can *see* walls/furniture from a
        # comfortable standoff (not hugging occupied cells).
        near_wall = [
            c for c in candidates
            if self._grid.wall_view_score(c, inner=4, outer=self._wall_radius_cells) >= 3
            and self._grid.has_nav_clearance(c, radius_cells=3)
        ]
        # Frontiers first — coverage beats wall-hugging.
        pool = frontiers * 5 + near_wall * 1 + candidates
        samples = [self._rng.choice(pool) for _ in range(min(self._n_samples, max(len(pool), 1) * 2))]
        samples.extend(frontiers[:20])
        samples.extend(near_wall[:3])

        best_cell = None
        best_score = -1.0
        for cell in samples:
            if cell not in costs or cell in self._blacklist:
                continue
            if costs[cell] <= self._reach_dist:
                continue
            # Require 3-cell clearance (~0.6 m) so the nav stack can settle
            # within the WP-reached threshold near obstacles.
            if not self._grid.has_nav_clearance(cell, radius_cells=3):
                continue
            gain = self._grid.unknown_region_size([cell], radius_cells=12)
            wall = self._grid.wall_view_score(
                cell, inner=4, outer=self._wall_radius_cells,
            )
            path = max(costs[cell], 1e-3)
            score = (gain + self._wall_weight * wall) / (1.0 + path)
            if score > best_score:
                best_score = score
                best_cell = cell
        if best_cell is None:
            # Fallback: any cleared candidate, else nearest reachable.
            for rad in (2, 1, 0):
                cleared = [
                    c for c in candidates
                    if rad == 0 or self._grid.has_nav_clearance(c, radius_cells=rad)
                ]
                if cleared:
                    best_cell = min(cleared, key=lambda c: costs[c])
                    break
            if best_cell is None:
                return None
        return self._wp(best_cell, rx, ry)

    def _wp(self, cell: tuple[int, int], rx: float, ry: float) -> Waypoint:
        cx, cy = self._grid.to_world(*cell)
        # Face the nearest wall/obstacle so RGB+LiDAR see vertical surfaces.
        look = self._grid.nearest_occupied_world(cx, cy)
        if look is not None:
            lx, ly = look
            if math.hypot(lx - cx, ly - cy) > 0.15:
                heading = math.atan2(ly - cy, lx - cx)
            else:
                heading = math.atan2(cy - ry, cx - rx)
        else:
            heading = math.atan2(cy - ry, cx - rx)
        return Waypoint(x=cx, y=cy, heading=heading)

    def _skip(self, wp: Waypoint) -> None:
        self.skipped_count += 1
        self._consecutive_skip_count += 1
        self._blacklist.add(self._grid.world_to_grid(wp.x, wp.y))
        self._current_target = None
        self._target_set_time = None
        if self._consecutive_skip_count >= self._max_consecutive_skips:
            self._done = True

    def _mark(self, wp: Waypoint) -> None:
        self._grid.mark_occupied(wp.x, wp.y, radius_cells=self._visit_mark_radius_cells)

    def _within(self, rx: float, ry: float, wp: Waypoint) -> bool:
        return math.hypot(rx - wp.x, ry - wp.y) < self._reach_dist
