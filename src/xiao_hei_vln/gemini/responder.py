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
import time
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

        # First-tick hook: open a per-question subdir in the logger so
        # `log_tick` has somewhere to write.
        if self._tick_count == 1 and self._logger is not None:
            self._logger.new_question(snapshot.question.text)

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
        if self._logger is not None:
            self._logger.close()

    # ------------------------------------------------------------------ Task 1

    def _respond_task1(self, snapshot: VLMInput) -> VLMOutput | None:
        """Numerical / object-reference: explore → commit Gemini answer."""
        if self._committed_answer is not None:
            return self._committed_answer

        if not self._should_commit(snapshot):
            # Delegate one waypoint to the frontier explorer. Log the
            # tick with empty prompts — useful for offline analysis of
            # how the explore phase progressed even though Gemini wasn't
            # called.
            explore_out = self._perception.respond(snapshot)
            if isinstance(explore_out, WaypointPathResponse):
                self._last_perception_rationale = explore_out.rationale
            self._log_tick(
                snapshot,
                system_prompt="",
                user_text=f"<explore-phase tick={self._tick_count}>",
                output=explore_out,
                inference_ms=0.0,
            )
            return explore_out

        # Trigger fired — call Gemini once, commit, done. On failure
        # (network blip, malformed JSON) bail out cleanly: re-emit a
        # last-ditch perception waypoint and let the next tick re-fire
        # the trigger.
        result = self._call_gemini(
            snapshot,
            exploration_summary=self._exploration_summary(),
        )
        if result is None:
            return self._perception.respond(snapshot)
        self._committed_answer = result
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
            if out is None:
                # Gemini failed mid-plan; bail out so the next tick can retry.
                return None
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
    ) -> VLMOutput | None:
        """Build the bundle, call Gemini, log the tick.

        Returns ``None`` if anything in the build → call → parse chain
        raises (network blip, malformed JSON, etc.). The caller decides
        whether to retry on the next tick or fall back to exploration.
        """
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

        output: VLMOutput | None = None
        t0 = time.perf_counter()
        try:
            output = self._engine.infer_multimodal(
                system=system,
                user_text=full_user_text,
                images=images,
            )
        except Exception:
            log.exception("Gemini inference failed; skipping commit this tick")
        finally:
            inference_ms = (time.perf_counter() - t0) * 1000.0
            self._log_tick(
                snapshot,
                system_prompt=system,
                user_text=full_user_text,
                output=output,
                inference_ms=inference_ms,
            )
        return output

    def _log_tick(
        self,
        snapshot: VLMInput,
        *,
        system_prompt: str,
        user_text: str,
        output: VLMOutput | None,
        inference_ms: float,
    ) -> None:
        if self._logger is None:
            return
        try:
            self._logger.log_tick(
                snapshot,
                system_prompt,
                user_text,
                output,
                inference_ms,
                [self._last_perception_rationale or ""],
            )
        except Exception:
            log.exception("VLMLogger.log_tick failed; continuing")

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
