"""``GeminiResponder`` — exploration-then-answer pipeline backed by Gemini.

Wraps the existing frontier-based :class:`PerceptionResponder` so the
agent keeps exploring the room with the proven geometric loop (PR #12);
on a configurable trigger it serialises the cross-tick
:class:`SceneRepresentation` plus current sensors and asks Gemini for the
final answer (Task 1) or the waypoint plan (Task 2).

Question-type routing:

* **Numerical / object-reference (Task 1)** — keep exploring via
  ``PerceptionResponder`` until either the frontier list is exhausted or
  ``config.max_explore_ticks`` is hit; then call Gemini once, parse its
  ``VLMOutput``, commit, and mark ``is_done = True``.
* **Instruction-following (Task 2)** — call Gemini on the first tick to
  produce a waypoint sequence; on subsequent ticks emit one waypoint at
  a time until the sequence is exhausted.

API mirrors :class:`xiao_hei_vln.qwen.responder.QwenResponder`
(``respond / is_done / reset / close``) so it slots into
``app/main.py`` via ``XIAO_HEI_RESPONDER=gemini``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from xiao_hei_vln.gemini.config import GeminiConfig
from xiao_hei_vln.gemini.engine import GeminiEngineProtocol
from xiao_hei_vln.gemini.prompts import build_system_prompt, build_user_message
from xiao_hei_vln.gemini.scene_rep import build_bundle, serialize_for_gemini
from xiao_hei_vln.messages import (
    QuestionType,
    VLMInput,
    VLMOutput,
    Waypoint,
    WaypointPathResponse,
)
from xiao_hei_vln.perception.global_map import GlobalMap
from xiao_hei_vln.perception_responder.responder import PerceptionResponder
from xiao_hei_vln.scene import SceneRepresentation

if TYPE_CHECKING:  # pragma: no cover
    from xiao_hei_vln.logger import VLMLogger

log = logging.getLogger(__name__)


class GeminiResponder:
    """Per-question stateful responder that wraps explore + Gemini."""

    def __init__(
        self,
        engine: GeminiEngineProtocol,
        config: GeminiConfig,
        *,
        perception: PerceptionResponder | None = None,
        global_map: GlobalMap | None = None,
        logger: VLMLogger | None = None,
    ) -> None:
        self._engine = engine
        self._config = config
        # Reuse a single GlobalMap between PerceptionResponder and our
        # scene-bundle renderer so they always agree on what's been seen.
        self._map = global_map if global_map is not None else GlobalMap()
        self._perception = perception or PerceptionResponder(global_map=self._map)
        self._scene = SceneRepresentation()
        self._logger = logger

        # Cross-tick state.
        self._tick_count = 0
        self._trajectory_xy: list[tuple[float, float]] = []
        self._committed_answer: VLMOutput | None = None
        self._planned_waypoints: list[Waypoint] = []
        self._planned_wp_idx = 0
        self._last_perception_rationale: str | None = None
        self._done = False

    # ------------------------------------------------------------------ main

    def respond(self, snapshot: VLMInput) -> VLMOutput | None:
        if snapshot.question is None or self._done:
            return None

        self._tick_count += 1

        # Update cross-tick state from this snapshot — both the scene
        # graph and our local trajectory ring.
        self._scene.update(snapshot)
        if snapshot.pose is not None:
            p = snapshot.pose.position
            self._trajectory_xy.append((p.x, p.y))
            # Cap the trajectory so prompts stay bounded.
            if len(self._trajectory_xy) > 200:
                self._trajectory_xy = self._trajectory_xy[-200:]

        if snapshot.question.type is QuestionType.INSTRUCTION_FOLLOWING:
            return self._respond_task2(snapshot)
        return self._respond_task1(snapshot)

    def is_done(self) -> bool:
        return self._done

    def reset(self) -> None:
        """Wipe everything that's persistent across ticks."""
        self._perception.reset()
        self._map.reset()
        self._scene = SceneRepresentation()
        self._tick_count = 0
        self._trajectory_xy = []
        self._committed_answer = None
        self._planned_waypoints = []
        self._planned_wp_idx = 0
        self._last_perception_rationale = None
        self._done = False

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------ Task 1

    def _respond_task1(self, snapshot: VLMInput) -> VLMOutput | None:
        """Numerical / object-reference: explore → commit Gemini answer."""
        if self._committed_answer is not None:
            return self._committed_answer

        if not self._should_commit(snapshot):
            # Delegate one waypoint to the frontier explorer.
            explore_out = self._perception.respond(snapshot)
            if isinstance(explore_out, WaypointPathResponse):
                self._last_perception_rationale = explore_out.rationale
            return explore_out

        # Trigger fired — call Gemini once, commit, done.
        self._committed_answer = self._call_gemini(
            snapshot,
            exploration_summary=self._exploration_summary(),
        )
        self._done = True
        return self._committed_answer

    def _should_commit(self, snapshot: VLMInput) -> bool:
        """Has the explore phase produced enough coverage to commit?"""
        # Soft cap: keep exploring until the perception responder declares
        # exhaustion. We watch its rationale (set by `_stand_still`).
        if self._last_perception_rationale and self._last_perception_rationale.startswith(
            "phase_a_exhausted",
        ):
            return True
        # Hard caps: ``max_explore_ticks=N`` lets the perception responder
        # run for N ticks; commit fires on tick N+1.
        if self._tick_count > self._config.max_explore_ticks:
            return True
        if self._tick_count > self._config.max_ticks_per_question:
            return True
        return False

    # ------------------------------------------------------------------ Task 2

    def _respond_task2(self, snapshot: VLMInput) -> VLMOutput | None:
        """Instruction-following: plan once via Gemini, then step waypoints."""
        if not self._planned_waypoints:
            out = self._call_gemini(snapshot, exploration_summary=None)
            if isinstance(out, WaypointPathResponse):
                self._planned_waypoints = list(out.waypoints)
            else:
                # Gemini returned a non-waypoint shape for an
                # instruction-following question. Log it, abandon the
                # plan, and let the system see no output this tick.
                log.warning(
                    "Gemini returned %s for INSTRUCTION_FOLLOWING; ignoring",
                    type(out).__name__,
                )
                self._done = True
                return None

        if not self._planned_waypoints:
            self._done = True
            return None

        # Emit the current waypoint. Advance only when the robot is close
        # enough — same advance trigger DummyResponder uses.
        idx = min(self._planned_wp_idx, len(self._planned_waypoints) - 1)
        current = self._planned_waypoints[idx]
        if self._reached(snapshot, current):
            if self._planned_wp_idx + 1 >= len(self._planned_waypoints):
                self._done = True
            else:
                self._planned_wp_idx += 1
                current = self._planned_waypoints[self._planned_wp_idx]
        return WaypointPathResponse(
            waypoints=[current],
            rationale=f"gemini_route_wp_{self._planned_wp_idx + 1}_of_{len(self._planned_waypoints)}",
        )

    # ------------------------------------------------------------------ Gemini call

    def _call_gemini(
        self,
        snapshot: VLMInput,
        *,
        exploration_summary: str | None,
    ) -> VLMOutput:
        bundle = build_bundle(
            snapshot=snapshot,
            scene=self._scene,
            global_map=self._map,
            trajectory_xy=list(self._trajectory_xy),
            panorama_long_edge=self._config.image_long_edge,
        )
        scene_text, images = serialize_for_gemini(bundle)
        system = build_system_prompt(snapshot.question.type)
        user_text = build_user_message(
            snapshot,
            snapshot.question,
            trajectory_xy=self._trajectory_xy,
            exploration_summary=exploration_summary,
        )
        full_user_text = f"{user_text}\n\n{scene_text}"
        log.info(
            "Calling Gemini for %s question (tick=%d, %d images, %d viewpoints)",
            snapshot.question.type.value,
            self._tick_count,
            len(images),
            len(self._scene.viewpoints),
        )
        return self._engine.infer_multimodal(
            system=system,
            user_text=full_user_text,
            images=images,
        )

    # ------------------------------------------------------------------ helpers

    def _exploration_summary(self) -> str:
        n_vps = len(self._scene.viewpoints)
        n_objs = len(self._scene.objects)
        return (
            f"explored over {self._tick_count} ticks; "
            f"{n_vps} viewpoint(s); "
            f"{n_objs} object(s) tracked in scene graph; "
            f"last perception rationale: {self._last_perception_rationale or '(n/a)'}"
        )

    @staticmethod
    def _reached(
        snapshot: VLMInput,
        wp: Waypoint,
        *,
        reach_dist: float = 0.5,
    ) -> bool:
        if snapshot.pose is None:
            return False
        dx = snapshot.pose.position.x - wp.x
        dy = snapshot.pose.position.y - wp.y
        return (dx * dx + dy * dy) ** 0.5 < reach_dist
