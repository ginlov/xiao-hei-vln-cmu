"""Lawnmower / boustrophedon coverage over the known FREE map."""

from __future__ import annotations

import math

from xiao_hei_vln.exploration._grid import OccupancyGrid
from xiao_hei_vln.messages.inputs import VLMInput
from xiao_hei_vln.messages.outputs import Waypoint


class LawnmowerExplorer:
    """Serpentine sweep of known FREE cells; expands as the map grows."""

    def __init__(
        self,
        *,
        max_waypoints: int = 100,
        grid_resolution: float = 0.2,
        waypoint_reach_dist: float = 0.45,
        cost_threshold: float = 0.5,
        stuck_timeout_s: float = 12.0,
        row_stride_cells: int = 3,
        max_consecutive_skips: int = 25,
    ) -> None:
        self._max_waypoints = max_waypoints
        self._reach_dist = waypoint_reach_dist
        self._cost_threshold = cost_threshold
        self._stuck_timeout_s = stuck_timeout_s
        self._row_stride = max(1, row_stride_cells)
        self._max_consecutive_skips = max_consecutive_skips

        self._grid = OccupancyGrid(grid_resolution)
        self._current_target: Waypoint | None = None
        self._visited: list[Waypoint] = []
        self._done = False
        self._target_set_time: float | None = None
        self.skipped_count = 0
        self._consecutive_skip_count = 0
        self._covered: set[tuple[int, int]] = set()
        self._lane_y: int | None = None
        self._lane_dir = 1

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
        self._covered.add(self._grid.world_to_grid(rx, ry))

        if self._current_target is not None and math.hypot(
            rx - self._current_target.x, ry - self._current_target.y
        ) < self._reach_dist:
            self._visited.append(self._current_target)
            self._covered.add(
                self._grid.world_to_grid(self._current_target.x, self._current_target.y)
            )
            self._current_target = None
            self._target_set_time = None
            self._consecutive_skip_count = 0

        if self._current_target is not None and self._target_set_time is not None:
            if now - self._target_set_time > self._stuck_timeout_s:
                self.skipped_count += 1
                self._consecutive_skip_count += 1
                self._covered.add(
                    self._grid.world_to_grid(self._current_target.x, self._current_target.y)
                )
                self._current_target = None
                self._target_set_time = None
                if self._consecutive_skip_count >= self._max_consecutive_skips:
                    self._done = True
                    return None

        if len(self._visited) >= self._max_waypoints:
            self._done = True
            return None

        if self._current_target is None:
            self._current_target = self._next_lawn_wp(rx, ry)
            if self._current_target is None:
                if not self._grid.free_cells:
                    return None
                self._done = True
                return None
            self._target_set_time = now
        return self._current_target

    def force_skip(self) -> None:
        if self._current_target is None:
            return
        self.skipped_count += 1
        self._consecutive_skip_count += 1
        self._covered.add(
            self._grid.world_to_grid(self._current_target.x, self._current_target.y)
        )
        self._current_target = None
        self._target_set_time = None
        if self._consecutive_skip_count >= self._max_consecutive_skips:
            self._done = True

    def advance(self) -> None:
        if self._current_target is not None:
            self._visited.append(self._current_target)
            self._covered.add(
                self._grid.world_to_grid(self._current_target.x, self._current_target.y)
            )
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
        self._covered.clear()
        self._lane_y = None
        self._lane_dir = 1

    def get_visited_waypoints(self) -> list[Waypoint]:
        wps = list(self._visited)
        if self._current_target is not None:
            wps.append(self._current_target)
        return wps

    def get_grid(self) -> OccupancyGrid:
        return self._grid

    def _next_lawn_wp(self, rx: float, ry: float) -> Waypoint | None:
        free = self._grid.free_cells
        if not free:
            return None
        costs = self._grid.reachable_path_costs(rx, ry)
        if not costs:
            return None

        ys = sorted({c[1] for c in free})
        if self._lane_y is None:
            self._lane_y = min(ys, key=lambda y: abs(y - self._grid.world_to_grid(rx, ry)[1]))

        # Advance lane until we find an uncovered reachable cell.
        for _ in range(len(ys) + 2):
            assert self._lane_y is not None
            row = [
                c for c in free
                if c[1] == self._lane_y and c not in self._covered and c in costs
            ]
            if row:
                row.sort(key=lambda c: c[0], reverse=(self._lane_dir < 0))
                cell = row[0]
                cx, cy = self._grid.to_world(*cell)
                return Waypoint(x=cx, y=cy, heading=math.atan2(cy - ry, cx - rx))

            # Next lane.
            lane_candidates = [y for y in ys if (y - self._lane_y) * self._lane_dir > 0]
            if not lane_candidates:
                self._lane_dir *= -1
                lane_candidates = [y for y in ys if (y - self._lane_y) * self._lane_dir > 0]
            if not lane_candidates:
                # Jump to any uncovered reachable free cell.
                leftover = [c for c in free if c not in self._covered and c in costs]
                if not leftover:
                    return None
                cell = min(leftover, key=lambda c: costs[c])
                cx, cy = self._grid.to_world(*cell)
                return Waypoint(x=cx, y=cy, heading=math.atan2(cy - ry, cx - rx))
            self._lane_y = min(
                lane_candidates,
                key=lambda y: abs(y - self._lane_y),  # type: ignore[arg-type]
            )
            # Skip rows by stride.
            ordered = sorted(ys) if self._lane_dir > 0 else sorted(ys, reverse=True)
            try:
                idx = ordered.index(self._lane_y)
                idx = min(idx + self._row_stride - 1, len(ordered) - 1)
                self._lane_y = ordered[idx]
            except ValueError:
                pass
        return None
