"""VLM-backed waypoint proposer, as an ExplorationStrategy.

Drop-in replacement for :class:`FrontierExplorer` / :class:`NextBestViewExplorer`:
same ``update()/is_complete()/reset()`` protocol, same ``advance()``/
``force_skip()`` hooks and ``_current_target``/``_visited``/``skipped_count``
surface the app tick-loop drives. Select it with
``XIAO_HEI_EXPLORATION_STRATEGY=nav_vlm``.

Design (see docs/tasks TASK 38)
-------------------------------
Trigger policy — the model is asked for a waypoint EXACTLY on the events that
end a waypoint's life, never on a timer:

  * cold start   — no target yet, at episode start;
  * reached      — the app calls :meth:`advance` (nav stack settled on target);
  * cannot reach — the app calls :meth:`force_skip` (planner blocked short), OR
                   the internal stuck watchdog fires.

That is ~one model call per waypoint outcome. The call runs on a background
thread so the 2 Hz control loop never blocks on the multi-second round trip;
while it is in flight ``update()`` returns ``None`` and the robot holds.

Two safeguards fix what the geometric explorers lacked:
  * every proposal is SNAPPED to the nearest grid-reachable free cell (BFS
    from the robot), so the model cannot send the robot into a wall or an
    unreachable pocket — the NBV "wedge" failure mode;
  * a cannot-reach outcome feeds the failure reason back into the next
    prompt, so the model re-plans around the dead zone instead of re-proposing
    it.
"""

from __future__ import annotations

import logging
import math
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

from xiao_hei_vln.exploration._grid import OccupancyGrid
from xiao_hei_vln.messages.inputs import VLMInput
from xiao_hei_vln.messages.outputs import Waypoint
from xiao_hei_vln.nav_vlm.prompts import build_user_text
from xiao_hei_vln.nav_vlm.render import (
    panorama_to_jpeg,
    render_grid_png,
    trajectory_summary,
)

if TYPE_CHECKING:  # pragma: no cover
    from xiao_hei_vln.nav_vlm.config import NavVLMConfig
    from xiao_hei_vln.nav_vlm.engine import NavVLMEngineProtocol, WaypointProposal

log = logging.getLogger(__name__)

_TRAJ_SAMPLE_M = 0.2   # only record a trajectory point after moving this far
_TRAJ_MAX = 200        # cap trajectory history sent to the model


class NavVLMExplorer:
    """Ask a VLM for the next waypoint on each reach / skip event."""

    def __init__(
        self,
        engine: NavVLMEngineProtocol,
        *,
        config: NavVLMConfig | None = None,
        max_waypoints: int = 30,
        grid_resolution: float = 0.2,
        waypoint_reach_dist: float = 0.3,
        cost_threshold: float = 0.5,
        max_hop_m: float = 3.0,
        stuck_timeout_s: float = 25.0,
        max_consecutive_skips: int = 6,
        max_propose_failures: int = 4,
        image_long_edge: int = 1024,
        occupancy_dpi: int = 100,
        async_calls: bool = True,
    ) -> None:
        self._engine = engine
        self._max_waypoints = max_waypoints
        self._reach_dist = waypoint_reach_dist
        self._cost_threshold = cost_threshold
        self._max_hop_m = max_hop_m
        self._stuck_timeout_s = stuck_timeout_s
        self._max_consecutive_skips = max_consecutive_skips
        self._max_propose_failures = max_propose_failures
        # Config overrides the plain image knobs when provided.
        self._image_long_edge = config.image_long_edge if config else image_long_edge
        self._occupancy_dpi = config.occupancy_dpi if config else occupancy_dpi
        self._async = async_calls

        self._grid = OccupancyGrid(grid_resolution)

        # --- app-facing surface (mirrors FrontierExplorer) ---------------
        self._current_target: Waypoint | None = None
        self._visited: list[Waypoint] = []
        self.skipped_count: int = 0
        self._consecutive_skip_count: int = 0
        self._done = False

        # --- proposer state ----------------------------------------------
        self._pending: Future[WaypointProposal] | None = None
        self._target_set_time: float | None = None
        self._last_failure: str | None = None
        self._failed_xy: tuple[float, float] | None = None
        # Rolling history of recently-rejected waypoints, so the model gets the
        # whole blocked frontier (all red stars + a list), not just the last
        # point — that is what lets it decide the direct approach is blocked and
        # route AROUND. Cleared on a real reach (the situation changed).
        self._failed_history: list[tuple[float, float]] = []
        self._propose_failures = 0
        self._last_rationale: str = ""   # model's reasoning for the current target

        # --- latest observation ------------------------------------------
        self._snapshot: VLMInput | None = None
        self._robot_xy: tuple[float, float] | None = None
        self._robot_yaw: float | None = None
        self._trajectory: list[tuple[float, float]] = []

        self._executor = ThreadPoolExecutor(max_workers=1) if async_calls else None

    # ------------------------------------------------------------------
    # Strategy interface

    def update(self, snapshot: VLMInput) -> Waypoint | None:
        if self._done:
            return None
        self._ingest_observation(snapshot)
        return self._step(snapshot.tick_time.to_seconds())

    def _ingest_observation(self, snapshot: VLMInput) -> None:
        """Fold this frame into the grid, pose, and trajectory state.

        Split from :meth:`_step` so a question-directed subclass can ingest
        every tick (keeping the map fresh) while still holding before a
        question arrives.
        """
        if snapshot.terrain_ext is not None:
            self._grid.update(snapshot.terrain_ext, self._cost_threshold)
        self._snapshot = snapshot
        if snapshot.pose is not None:
            self._ingest_pose(snapshot.pose)

    def _step(self, now: float) -> Waypoint | None:
        """Run the trigger state machine for one tick (observation already in)."""
        # 1. Resolve an in-flight proposal (async path).
        if self._pending is not None:
            if self._pending.done():
                self._resolve_pending(now)
            else:
                return self._current_target  # None while thinking → robot holds

        # 2. Stuck watchdog: an unreached target that overstays is a cannot-reach.
        if (
            self._current_target is not None
            and self._target_set_time is not None
            and now - self._target_set_time > self._stuck_timeout_s
        ):
            self._register_skip(f"not reached within {self._stuck_timeout_s:.0f}s (watchdog)")

        # 3. Budget.
        if len(self._visited) >= self._max_waypoints:
            self._done = True
            return None

        # 4. No target and nothing in flight → ask for the next waypoint.
        if not self._done and self._current_target is None and self._pending is None:
            self._trigger(now)

        return self._current_target

    def is_complete(self) -> bool:
        return self._done

    def reset(self) -> None:
        if self._pending is not None:
            self._pending.cancel()
            self._pending = None
        self._grid = OccupancyGrid(self._grid.resolution)
        self._current_target = None
        self._visited = []
        self.skipped_count = 0
        self._consecutive_skip_count = 0
        self._done = False
        self._target_set_time = None
        self._last_failure = None
        self._failed_xy = None
        self._failed_history = []
        self._propose_failures = 0
        self._snapshot = None
        self._robot_xy = None
        self._robot_yaw = None
        self._trajectory = []

    # ------------------------------------------------------------------
    # App-driven lifecycle hooks

    def advance(self) -> None:
        """Reached: mark current target visited and request the next one."""
        if self._current_target is None:
            return
        t = self._current_target
        self._grid.mark_occupied(t.x, t.y, radius_cells=1)  # don't re-pick the same spot
        self._visited.append(t)
        self._current_target = None
        self._target_set_time = None
        self._consecutive_skip_count = 0
        self._last_failure = None
        self._failed_xy = None
        self._failed_history.clear()  # a real reach = new vantage; old blocks stale

    def force_skip(self) -> None:
        """Cannot reach: the local planner settled short of the target."""
        self._register_skip("blocked — the local planner settled short of it")

    # ------------------------------------------------------------------
    # Accessors for visualisation / reporting

    def get_visited_waypoints(self) -> list[Waypoint]:
        wps = list(self._visited)
        if self._current_target is not None:
            wps.append(self._current_target)
        return wps

    def get_grid(self) -> OccupancyGrid:
        return self._grid

    # ------------------------------------------------------------------
    # Internals

    def _register_skip(self, reason: str) -> None:
        if self._current_target is None:
            return
        t = self._current_target
        self.skipped_count += 1
        self._consecutive_skip_count += 1
        # Blacklist a small area so snapping steers clear of the dead zone.
        self._grid.mark_occupied(t.x, t.y, radius_cells=3)
        self._failed_xy = (t.x, t.y)
        self._failed_history.append((t.x, t.y))
        if len(self._failed_history) > 8:
            del self._failed_history[0]
        self._last_failure = f"target ({t.x:.2f},{t.y:.2f}) was {reason}."
        self._current_target = None
        self._target_set_time = None
        if self._consecutive_skip_count >= self._max_consecutive_skips:
            self._done = True

    def _trigger(self, now: float) -> None:
        """Build the multimodal request and dispatch the proposer call."""
        occ_png = self._render_occupancy()
        pano = None
        if self._snapshot is not None and self._snapshot.image is not None:
            try:
                pano = panorama_to_jpeg(self._snapshot.image, long_edge=self._image_long_edge)
            except Exception as exc:  # noqa: BLE001 — a bad frame must not kill nav
                log.warning("nav_vlm: panorama encode failed: %s", exc)
        user_text = self._build_user_text()

        if self._async and self._executor is not None:
            self._pending = self._executor.submit(
                self._engine.propose,
                user_text=user_text,
                panorama_jpg=pano,
                occupancy_png=occ_png,
            )
            return
        # Synchronous path (tests): call inline and apply immediately.
        try:
            proposal = self._engine.propose(
                user_text=user_text, panorama_jpg=pano, occupancy_png=occ_png,
            )
        except Exception as exc:  # noqa: BLE001
            self._on_propose_error(exc)
            return
        self._apply_proposal(proposal, now)

    def _render_occupancy(self) -> bytes:
        """Render the occupancy PNG for the model. Overridden by the Task-1
        navigator to also draw the detected objects (so the model can route
        toward a target it can see on the map)."""
        return render_grid_png(
            self._grid,
            robot_xy=self._robot_xy,
            failed_xy=self._failed_xy,
            trajectory_xy=self._trajectory,
            dpi=self._occupancy_dpi,
        )

    def _build_user_text(self) -> str:
        """The per-call user text. Overridden by the Task-1 navigator to inject
        the question + scene-graph objects."""
        return build_user_text(
            robot_xy=self._robot_xy,
            robot_yaw=self._robot_yaw,
            failure_reason=self._last_failure,
            visited=len(self._visited),
            trajectory_summary=trajectory_summary(self._trajectory),
        )

    def _resolve_pending(self, now: float) -> None:
        fut = self._pending
        self._pending = None
        if fut is None:
            return
        try:
            proposal = fut.result()
        except Exception as exc:  # noqa: BLE001
            self._on_propose_error(exc)
            return
        self._apply_proposal(proposal, now)

    def _on_propose_error(self, exc: BaseException) -> None:
        log.warning("nav_vlm: proposer call failed: %s", exc)
        self._propose_failures += 1
        if self._propose_failures >= self._max_propose_failures:
            log.error("nav_vlm: giving up after %d failed calls", self._propose_failures)
            self._done = True

    def _apply_proposal(self, proposal: WaypointProposal, now: float) -> None:
        if proposal.done:
            log.info("nav_vlm: model signalled exploration complete (%s)", proposal.rationale)
            self._done = True
            return
        if proposal.x is None or proposal.y is None:
            self._on_propose_error(ValueError("proposal missing x/y"))
            return

        wp = self._snap_to_reachable(proposal.x, proposal.y, proposal.heading)
        if wp is None:
            # No reachable free path to the pick — treat as a soft failure so
            # the next prompt gets the context, but don't blacklist (it may be
            # reachable once more terrain is observed).
            self._last_failure = (
                f"proposed ({proposal.x:.2f},{proposal.y:.2f}) had no reachable "
                "free path from the robot."
            )
            self._on_propose_error(ValueError("proposal not reachable"))
            return

        self._propose_failures = 0
        self._current_target = wp
        self._target_set_time = now
        self._last_failure = None
        self._failed_xy = None
        self._last_rationale = proposal.rationale
        log.info(
            "nav_vlm: target (%.2f,%.2f) [%s]", wp.x, wp.y, proposal.rationale,
        )

    def _snap_to_reachable(
        self, x: float, y: float, heading: float | None,
    ) -> Waypoint | None:
        """Snap the model's pick to the nearest grid-reachable free cell.

        Guarantees the published waypoint sits on a free cell the robot can
        actually path to (BFS over free space from the current pose), within
        ``max_hop_m`` of the robot so the local planner can finish it in one go.
        Returns ``None`` if nothing reachable exists yet.
        """
        if self._robot_xy is None:
            return None
        rx, ry = self._robot_xy

        costs = self._grid.reachable_path_costs(rx, ry)
        if not costs:
            # Cold start: no belief map yet. Accept the raw pick, clamped to a
            # modest hop so it's plausibly reachable.
            return self._clamped_raw(x, y, heading, rx, ry)

        best_xy: tuple[float, float] | None = None
        best_dist = float("inf")
        for cell, path_cost in costs.items():
            if path_cost <= 1e-6:
                continue  # the robot's own cell — would false-reach instantly
            if path_cost > self._max_hop_m:
                continue  # keep the hop finishable in one planner cycle
            wx, wy = self._grid.to_world(*cell)
            d = math.hypot(wx - x, wy - y)
            if d < best_dist:
                best_dist = d
                best_xy = (wx, wy)

        if best_xy is None:
            # Everything reachable is beyond the hop horizon — step toward the
            # pick using the farthest cell we can reach.
            far_cell = max(costs.items(), key=lambda kv: kv[1])[0]
            if costs[far_cell] <= 1e-6:
                return None
            best_xy = self._grid.to_world(*far_cell)

        hx, hy = best_xy
        h = heading if heading is not None else math.atan2(hy - ry, hx - rx)
        return Waypoint(x=hx, y=hy, heading=h)

    def _clamped_raw(
        self, x: float, y: float, heading: float | None, rx: float, ry: float,
    ) -> Waypoint:
        dx, dy = x - rx, y - ry
        d = math.hypot(dx, dy)
        if d > self._max_hop_m and d > 1e-6:
            s = self._max_hop_m / d
            x, y = rx + dx * s, ry + dy * s
        h = heading if heading is not None else math.atan2(y - ry, x - rx)
        return Waypoint(x=x, y=y, heading=h)

    def _ingest_pose(self, pose) -> None:
        p = pose.position
        self._robot_xy = (p.x, p.y)
        self._robot_yaw = _yaw_from_quat(pose.orientation)
        if not self._trajectory or math.hypot(
            p.x - self._trajectory[-1][0], p.y - self._trajectory[-1][1]
        ) >= _TRAJ_SAMPLE_M:
            self._trajectory.append((p.x, p.y))
            if len(self._trajectory) > _TRAJ_MAX:
                del self._trajectory[0]


def _yaw_from_quat(q) -> float:
    """Yaw (rad) about +Z from a geometry quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)
