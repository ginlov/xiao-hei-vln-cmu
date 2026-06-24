"""``PerceptionResponder`` — Phase A frontier-exploration responder.

API mirrors :class:`xiao_hei_vln.dummy.responder.DummyResponder` so the
main loop in ``app/main.py`` can swap it in with a one-line change:

* ``respond(snapshot) -> VLMOutput | None`` — called each tick.
* ``is_done() -> bool`` — True once the responder considers the question
  resolved. Phase A is never "done" within a question because exploration
  has no terminal condition until Phase 2 plugs in detection-based goal
  commitment; we simply keep emitting waypoints (or "stand still") and
  let the main loop reset us when the next question arrives.
* ``reset()`` — called by the main loop on question change; wipes the
  global map and frontier history.
* ``close()`` — release any held resources (no-op here).
"""

from __future__ import annotations

import math

from xiao_hei_vln.exploration.frontier import FrontierPlanner, ScoringWeights
from xiao_hei_vln.messages.inputs import VLMInput
from xiao_hei_vln.messages.outputs import VLMOutput, Waypoint, WaypointPathResponse
from xiao_hei_vln.perception.global_map import GlobalMap


class PerceptionResponder:
    """Stateful per-question exploration loop.

    Holds the cross-tick :class:`GlobalMap` + :class:`FrontierPlanner` and
    turns each :class:`VLMInput` snapshot into either:

    * a ``WaypointPathResponse`` toward the best-scored frontier, or
    * a "stand still" waypoint at the current pose once no frontier
      remains (the cue for the future Phase B / TSP coverage stage), or
    * ``None`` when the snapshot is too incomplete to act on.
    """

    def __init__(
        self,
        global_map: GlobalMap | None = None,
        weights: ScoringWeights | None = None,
        history_length: int = 5,
        min_cluster_size: int = 4,
    ) -> None:
        self._map = global_map if global_map is not None else GlobalMap()
        self._planner = FrontierPlanner(
            self._map,
            weights=weights,
            history_length=history_length,
            min_cluster_size=min_cluster_size,
        )
        # Phase A: there is no internal "done" signal — only the main loop's
        # question-change reset stops us. Kept here for API symmetry with the
        # dummy/qwen responders.
        self._done = False

    # ------------------------------------------------------------------ main API

    def respond(self, snapshot: VLMInput) -> VLMOutput | None:
        if snapshot.question is None:
            return None
        if snapshot.pose is None or snapshot.terrain_ext is None:
            # Cold-start: cache hasn't received both terrain_ext and pose yet.
            # Stay quiet so the publisher doesn't emit garbage.
            return None

        # Stitch the latest 20m local map into the persistent grid.
        self._map.update(snapshot.terrain_ext.points, snapshot.pose)

        target = self._planner.select(snapshot.pose)
        if target is None:
            # Phase A is exhausted (no reachable frontiers left). For v0 we
            # emit a "stand still" waypoint at the current pose so the system
            # has *some* output; Phase 3 (TSP coverage) will replace this
            # branch with the next sweep waypoint.
            return self._stand_still(snapshot)

        return self._waypoint_toward(snapshot, target.centroid_xy)

    def is_done(self) -> bool:
        return self._done

    def reset(self) -> None:
        """Wipe everything that's persistent across ticks.

        Called by ``app/main.py`` whenever the question text changes.
        """
        self._map.reset()
        self._planner.reset()
        self._done = False

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _stand_still(snapshot: VLMInput) -> WaypointPathResponse:
        assert snapshot.pose is not None
        px = snapshot.pose.position.x
        py = snapshot.pose.position.y
        # Use the robot's current yaw if available; for the Vector3/Quaternion
        # primitives we have on hand a 0 heading is fine — the downstream
        # waypoint follower interprets it as "no heading constraint".
        return WaypointPathResponse(
            waypoints=[Waypoint(x=px, y=py, heading=0.0)],
            rationale="phase_a_exhausted: no reachable frontiers remaining",
        )

    @staticmethod
    def _waypoint_toward(
        snapshot: VLMInput,
        target_xy: tuple[float, float],
    ) -> WaypointPathResponse:
        assert snapshot.pose is not None
        dx = target_xy[0] - snapshot.pose.position.x
        dy = target_xy[1] - snapshot.pose.position.y
        heading = math.atan2(dy, dx)
        return WaypointPathResponse(
            waypoints=[Waypoint(x=target_xy[0], y=target_xy[1], heading=heading)],
            rationale="frontier_target",
        )
