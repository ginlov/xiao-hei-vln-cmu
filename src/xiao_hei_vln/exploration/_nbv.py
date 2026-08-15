"""Next-Best-View exploration on the online belief occupancy map.

Uses only ``terrain_ext`` + pose (no GT floor plan). Candidates are sampled
from reachable FREE cells (frontier-biased) and scored as:

    score = unknown_gain / (1 + path_cost)

Visited waypoints are blacklisted with a small radius to reduce local thrashing.
"""

from __future__ import annotations

import math
import random

from xiao_hei_vln.exploration._grid import OccupancyGrid
from xiao_hei_vln.messages.inputs import VLMInput
from xiao_hei_vln.messages.outputs import Waypoint


class NextBestViewExplorer:
    """Sample candidate FREE poses; pick max unknown-gain / path cost."""

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
        skip_mark_radius_cells: int = 4,
        skip_reset_free_cells: int = 60,
        seed: int = 0,
    ) -> None:
        self._max_waypoints = max_waypoints
        self._reach_dist = waypoint_reach_dist
        self._cost_threshold = cost_threshold
        self._stuck_timeout_s = stuck_timeout_s
        self._max_consecutive_skips = max_consecutive_skips
        self._n_samples = n_samples
        self._visit_mark_radius_cells = visit_mark_radius_cells
        self._skip_mark_radius_cells = skip_mark_radius_cells
        self._skip_reset_free_cells = skip_reset_free_cells
        self._rng = random.Random(seed)
        self._free_at_last_progress = 0
        # Diagnostics for the exploration log — see selection_diagnostics().
        self._select_reason = "not_run"
        self._select_diag: dict[str, object] = {}
        self.skip_hatch_resets = 0

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
            self._reach(self._current_target)

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
            self._reach(self._current_target)

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
        self._free_at_last_progress = 0
        self._select_reason = "not_run"
        self._select_diag = {}
        self.skip_hatch_resets = 0

    def get_visited_waypoints(self) -> list[Waypoint]:
        wps = list(self._visited)
        if self._current_target is not None:
            wps.append(self._current_target)
        return wps

    def get_grid(self) -> OccupancyGrid:
        return self._grid

    def _pick(self, rx: float, ry: float) -> Waypoint | None:
        costs = self._grid.reachable_path_costs(rx, ry)
        self._select_diag = {
            "reachable": len(costs),
            "candidates": 0,
            "frontiers": 0,
            "blacklist": len(self._blacklist),
            "gain": 0,
        }
        if not costs:
            # The failure that killed nbv on every big scene: free cells exist
            # but none of them is reachable from where the robot stands.
            self._select_reason = (
                "unreachable" if self._grid.free_cells else "no_free_cells"
            )
            return None
        candidates = [c for c in costs if c not in self._blacklist and costs[c] > self._reach_dist]
        if not candidates:
            candidates = [c for c in costs if c not in self._blacklist]
        self._select_diag["candidates"] = len(candidates)
        if not candidates:
            self._select_reason = "all_blacklisted"
            return None

        frontiers = [
            c for c in self._grid.frontier_cells()
            if c in costs and c not in self._blacklist and costs[c] > self._reach_dist
        ]
        # Frontier cells are the only ones with real information gain, so score
        # every one of them (nearest-first, capped) instead of hoping a handful
        # of uniform draws lands on them. Random draws over free space only fill
        # in when there is no frontier left to aim at.
        frontiers.sort(key=lambda c: costs[c])
        self._select_diag["frontiers"] = len(frontiers)
        samples = list(frontiers[: self._n_samples])
        if len(samples) < self._n_samples:
            pool = frontiers + candidates if frontiers else candidates
            samples.extend(
                self._rng.choice(pool) for _ in range(self._n_samples - len(samples))
            )

        best_cell = None
        best_score = -1.0
        for cell in samples:
            if cell not in costs or cell in self._blacklist:
                continue
            if costs[cell] <= self._reach_dist:
                continue
            gain = self._grid.unknown_region_size([cell], radius_cells=12)
            path = max(costs[cell], 1e-3)
            score = gain / (1.0 + path)
            if score > best_score:
                best_score = score
                best_cell = cell
        if best_cell is None:
            # Nothing scored — every sample was underfoot or zero-gain. Falling
            # back to the nearest candidate keeps the robot moving, but a log
            # full of "nearest_fallback" means the gain term is doing no work.
            best_cell = min(candidates, key=lambda c: costs[c])
            self._select_reason = "nearest_fallback"
        else:
            self._select_reason = "ok"
            self._select_diag["gain"] = round(best_score, 2)
        return self._wp(best_cell, rx, ry)

    def selection_diagnostics(self) -> dict[str, object]:
        """Why the last `_pick` call returned what it did."""
        return {"why": self._select_reason, **self._select_diag}

    def _wp(self, cell: tuple[int, int], rx: float, ry: float) -> Waypoint:
        cx, cy = self._grid.to_world(*cell)
        return Waypoint(x=cx, y=cy, heading=math.atan2(cy - ry, cx - rx))

    def _reach(self, wp: Waypoint) -> None:
        self._visited.append(wp)
        self._mark(wp)
        self._current_target = None
        self._target_set_time = None
        self._consecutive_skip_count = 0
        self._free_at_last_progress = len(self._grid.free_cells)

    def _skip(self, wp: Waypoint) -> None:
        self.skipped_count += 1
        self._consecutive_skip_count += 1
        # Blacklisting the single cell under the target let the very next pick
        # land 20 cm away and fail identically. Suppress the whole disc.
        self._blacklist.update(self._disc(wp, self._skip_mark_radius_cells))
        self._grid.mark_no_target(wp.x, wp.y, radius_cells=self._skip_mark_radius_cells)
        self._current_target = None
        self._target_set_time = None
        if self._consecutive_skip_count < self._max_consecutive_skips:
            return
        # Still discovering new free space → not stalled, keep exploring.
        free_now = len(self._grid.free_cells)
        if free_now - self._free_at_last_progress >= self._skip_reset_free_cells:
            self._free_at_last_progress = free_now
            self._consecutive_skip_count = 0
            self.skip_hatch_resets += 1
        else:
            self._done = True

    def _mark(self, wp: Waypoint) -> None:
        # Target suppression only — NOT mark_occupied. Stamping visited
        # waypoints as obstacles carved fake walls into the belief map; a
        # waypoint in a doorway disconnected everything behind it from the
        # reachable-cost BFS, which is why the robot stopped moving at all.
        self._grid.mark_no_target(wp.x, wp.y, radius_cells=self._visit_mark_radius_cells)
        self._blacklist.update(self._disc(wp, self._visit_mark_radius_cells))

    def _disc(self, wp: Waypoint, radius_cells: int) -> set[tuple[int, int]]:
        cx, cy = self._grid.world_to_grid(wp.x, wp.y)
        r2 = radius_cells * radius_cells
        return {
            (cx + dx, cy + dy)
            for dx in range(-radius_cells, radius_cells + 1)
            for dy in range(-radius_cells, radius_cells + 1)
            if dx * dx + dy * dy <= r2
        }

    def _within(self, rx: float, ry: float, wp: Waypoint) -> bool:
        return math.hypot(rx - wp.x, ry - wp.y) < self._reach_dist
