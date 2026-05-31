"""1:1 Python port of `dummy_vlm/src/dummyVLM.cpp`.

`DummyResponder.respond()` consumes a `VLMInput` snapshot and returns a
`VLMOutput` (or `None` if there's nothing to publish this tick). The
three behaviours mirror the C++ reference:

- *Numerical* — emit a random `int` in `[1, 10]`, one shot.
- *Object reference* — emit the loaded object marker and a waypoint to
  its center, one shot.
- *Instruction following* — emit the current waypoint each tick, advance
  when the vehicle pose (from `/state_estimation`) is within
  `waypoint_reach_dist` of it.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from xiao_hei_vln.dummy.fixtures import read_object_list, read_waypoints_ply
from xiao_hei_vln.messages.inputs import VLMInput
from xiao_hei_vln.messages.outputs import (
    NumericalResponse,
    ObjectReferenceResponse,
    VLMOutput,
    Waypoint,
    WaypointPathResponse,
)
from xiao_hei_vln.messages.question import QuestionType

DEFAULT_DATA_DIR = Path(__file__).parent / "data"
DEFAULT_WAYPOINT_REACH_DIST = 1.0


class DummyResponder:
    def __init__(
        self,
        object_fixture: ObjectReferenceResponse | None = None,
        waypoint_fixture: list[Waypoint] | None = None,
        waypoint_reach_dist: float = DEFAULT_WAYPOINT_REACH_DIST,
        rng: random.Random | None = None,
    ) -> None:
        if object_fixture is None:
            object_fixture = read_object_list(DEFAULT_DATA_DIR / "object_list.txt")
        if waypoint_fixture is None:
            waypoint_fixture = read_waypoints_ply(DEFAULT_DATA_DIR / "waypoints.ply")
        if not waypoint_fixture:
            raise ValueError("waypoint_fixture must contain at least one waypoint")

        self._object = object_fixture
        self._waypoints = waypoint_fixture
        self._reach_dist = float(waypoint_reach_dist)
        self._rng = rng if rng is not None else random.Random()

        self._wp_idx = 0
        self._done = False

    # --- main entry --------------------------------------------------------------

    def respond(self, snapshot: VLMInput) -> VLMOutput | None:
        """Return the message to publish this tick, or None to stay quiet."""
        if snapshot.question is None or self._done:
            return None

        qtype = snapshot.question.type
        if qtype is QuestionType.NUMERICAL:
            self._done = True
            return NumericalResponse(value=self._rng.randint(1, 10))

        if qtype is QuestionType.OBJECT_REFERENCE:
            self._done = True
            return self._object

        # Instruction following: emit the current waypoint, advance on proximity.
        current = self._waypoints[self._wp_idx]
        if snapshot.pose is not None and self._within_reach(snapshot, current):
            if self._wp_idx + 1 >= len(self._waypoints):
                self._done = True
            else:
                self._wp_idx += 1
                current = self._waypoints[self._wp_idx]
        return WaypointPathResponse(waypoints=[current])

    # --- lifecycle ---------------------------------------------------------------

    def is_done(self) -> bool:
        return self._done

    def reset(self) -> None:
        """Re-arm for the next question (after the app clears the cache slot)."""
        self._wp_idx = 0
        self._done = False

    def close(self) -> None:
        pass

    # --- internals ---------------------------------------------------------------

    def _within_reach(self, snapshot: VLMInput, wp: Waypoint) -> bool:
        assert snapshot.pose is not None  # caller checks
        dx = snapshot.pose.position.x - wp.x
        dy = snapshot.pose.position.y - wp.y
        return math.hypot(dx, dy) < self._reach_dist
