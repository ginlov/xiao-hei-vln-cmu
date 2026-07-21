"""``SceneGeminiResponder`` — perception scene graph + Gemini answering.

This is the team's submission responder. It unifies the three pieces the
challenge needs into one coherent pipeline, driven by the **shared**
app-level frontier explorer (``app/main.py``):

1. **Frontier exploration** — the app tick loop runs
   :class:`xiao_hei_vln.exploration.FrontierExplorer` and, on every
   exploration tick, calls :meth:`ingest` (never :meth:`respond`). The
   sweep runs to its own completion; a question arriving mid-sweep is
   deferred (the Task-12 contract).
2. **Scene graph building** — :meth:`ingest` delegates to a real
   :class:`xiao_hei_vln.perception.PerceptionResponder`, which runs the
   perception sidecar's detect → lift → ``add_object`` cycle into the
   **shared** :class:`SceneRepresentation`. So by the time exploration
   finishes, the scene graph holds real objects (labels, 3D positions,
   colours, and — with ObjectMap fusion — 3D boxes).
3. **Gemini answering** — once exploration is complete the app loop
   starts calling :meth:`respond`. For Task 1 (numerical /
   object-reference) we serialise the populated scene graph plus a
   panorama JPEG and an occupancy/trajectory PNG and ask Gemini once,
   then commit. For Task 2 (instruction-following) we ask Gemini for a
   waypoint plan and step through it.

This retires the old ``GeminiResponder``, which explored *internally*
(duplicating the app explorer) and fed Gemini an object-less scene. Here
exploration happens exactly once, and Gemini reasons over a grounded
object graph.

The responder protocol (``respond / ingest / is_done / reset / close``)
matches :class:`xiao_hei_vln.perception.PerceptionResponder`, so it slots
into ``app/main.py`` via ``XIAO_HEI_RESPONDER=scene_gemini``.
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
from xiao_hei_vln.scene import SceneRepresentation

if TYPE_CHECKING:  # pragma: no cover
    from xiao_hei_vln.logger import VLMLogger
    from xiao_hei_vln.perception.responder import PerceptionResponder

log = logging.getLogger(__name__)

# XY distance (m) at which a planned Task-2 waypoint counts as reached —
# same trigger DummyResponder uses.
_REACH_DIST_M = 0.5
# Cap the pose history sent to Gemini so the prompt stays bounded.
_MAX_TRAJECTORY = 200


class SceneGeminiResponder:
    """Per-question responder: build the scene while exploring, answer with Gemini.

    ``scene`` is the **shared** :class:`SceneRepresentation` owned by
    ``app/main.py`` — the same instance the app loop updates every tick and
    that ``perception`` writes objects into. We serialise *that* graph to
    Gemini, so the app-level ``scene.update`` (viewpoints/bounds) and the
    perception object layer are always in agreement.
    """

    def __init__(
        self,
        engine: GeminiEngineProtocol,
        config: GeminiConfig,
        scene: SceneRepresentation,
        *,
        perception: PerceptionResponder,
        global_map: GlobalMap | None = None,
        logger: VLMLogger | None = None,
    ) -> None:
        self._engine = engine
        self._config = config
        self._scene = scene
        self._perception = perception
        # Occupancy grid rendered into the Gemini bundle. Maintained here
        # from terrain_ext because the app-level FrontierExplorer keeps its
        # own private OccupancyGrid we don't have a handle on.
        self._map = global_map if global_map is not None else GlobalMap()
        self._logger = logger

        # Cross-tick state.
        self._tick_count = 0
        self._trajectory_xy: list[tuple[float, float]] = []
        self._committed_answer: VLMOutput | None = None
        self._planned_waypoints: list[Waypoint] = []
        self._planned_wp_idx = 0
        self._done = False

    # ------------------------------------------------------------------ exploration

    def ingest(self, snapshot: VLMInput) -> None:
        """Build the scene from this frame *without* answering.

        Called by the app tick loop on every exploration tick. Delegates to
        the perception responder's own ``ingest`` (detect → lift →
        add_object into the shared scene) and keeps the occupancy map +
        trajectory ring fresh for the eventual Gemini bundle. Emits no
        answer, so a question that arrives mid-sweep stays deferred until
        exploration completes.
        """
        self._update_map(snapshot)
        self._accumulate_trajectory(snapshot)
        self._perception.ingest(snapshot)

    # ------------------------------------------------------------------ answering

    def respond(self, snapshot: VLMInput) -> VLMOutput | None:
        """Answer the active question.

        Only reached once the app-level explorer has completed, so the scene
        graph is already fully built. Task 1 asks Gemini once and commits;
        Task 2 plans a route once and steps it.
        """
        if snapshot.question is None or self._done:
            return None

        self._tick_count += 1
        if self._tick_count == 1 and self._logger is not None:
            self._logger.new_question(snapshot.question.text)

        # Keep the map + trajectory current through the answer window too.
        self._update_map(snapshot)
        self._accumulate_trajectory(snapshot)

        if snapshot.question.type is QuestionType.INSTRUCTION_FOLLOWING:
            return self._respond_task2(snapshot)
        return self._respond_task1(snapshot)

    def is_done(self) -> bool:
        return self._done

    def reset(self) -> None:
        """Wipe per-question state. Called by the app loop on question change.

        The shared scene graph is intentionally **not** cleared here — the
        physical scene is the same for every question in a session, so the
        map built during exploration is reused. (``app/main.py`` owns the
        scene lifetime.)
        """
        self._perception.reset()
        self._tick_count = 0
        self._trajectory_xy = []
        self._committed_answer = None
        self._planned_waypoints = []
        self._planned_wp_idx = 0
        self._done = False

    def close(self) -> None:
        self._perception.close()
        if self._logger is not None:
            self._logger.close()

    # ------------------------------------------------------------------ Task 1

    def _respond_task1(self, snapshot: VLMInput) -> VLMOutput | None:
        """Numerical / object-reference: ask Gemini once, commit."""
        if self._committed_answer is not None:
            return self._committed_answer

        result = self._call_gemini(
            snapshot,
            exploration_summary=self._exploration_summary(),
        )
        if result is None:
            # Transient failure (network blip, malformed JSON). Hold
            # position and let the next tick retry the Gemini call.
            return self._hold(snapshot)
        self._committed_answer = result
        self._done = True
        return self._committed_answer

    # ------------------------------------------------------------------ Task 2

    def _respond_task2(self, snapshot: VLMInput) -> VLMOutput | None:
        """Instruction-following: plan once via Gemini, then step waypoints."""
        if not self._planned_waypoints:
            out = self._call_gemini(snapshot, exploration_summary=None)
            if out is None:
                # Gemini failed mid-plan; retry on the next tick.
                return None
            if isinstance(out, WaypointPathResponse):
                self._planned_waypoints = list(out.waypoints)
            else:
                log.warning(
                    "Gemini returned %s for INSTRUCTION_FOLLOWING; ignoring",
                    type(out).__name__,
                )
                self._done = True
                return None

        if not self._planned_waypoints:
            self._done = True
            return None

        idx = min(self._planned_wp_idx, len(self._planned_waypoints) - 1)
        current = self._planned_waypoints[idx]
        if self._reached(snapshot, current):
            if self._planned_wp_idx + 1 >= len(self._planned_waypoints):
                self._done = True
            else:
                self._planned_wp_idx += 1
                current = self._planned_waypoints[self._planned_wp_idx]
        n = len(self._planned_waypoints)
        return WaypointPathResponse(
            waypoints=[current],
            rationale=f"gemini_route_wp_{self._planned_wp_idx + 1}_of_{n}",
        )

    # ------------------------------------------------------------------ Gemini call

    def _call_gemini(
        self,
        snapshot: VLMInput,
        *,
        exploration_summary: str | None,
    ) -> VLMOutput | None:
        """Build the multimodal bundle, call Gemini, log the tick.

        Returns ``None`` if anything in the build → call → parse chain
        raises; the caller decides whether to retry or hold.
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
            "Calling Gemini for %s question (tick=%d, %d images, %d objects, %d viewpoints)",
            snapshot.question.type.value,
            self._tick_count,
            len(images),
            len(self._scene.objects),
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
                [f"scene_objects={len(self._scene.objects)}"],
            )
        except Exception:
            log.exception("VLMLogger.log_tick failed; continuing")

    # ------------------------------------------------------------------ helpers

    def _update_map(self, snapshot: VLMInput) -> None:
        if snapshot.terrain_ext is not None and snapshot.pose is not None:
            self._map.update(snapshot.terrain_ext.points, snapshot.pose)

    def _accumulate_trajectory(self, snapshot: VLMInput) -> None:
        if snapshot.pose is None:
            return
        p = snapshot.pose.position
        self._trajectory_xy.append((p.x, p.y))
        if len(self._trajectory_xy) > _MAX_TRAJECTORY:
            self._trajectory_xy = self._trajectory_xy[-_MAX_TRAJECTORY:]

    def _exploration_summary(self) -> str:
        return (
            f"explored over {self._tick_count} answer-tick(s); "
            f"{len(self._scene.viewpoints)} viewpoint(s); "
            f"{len(self._scene.objects)} object(s) tracked in scene graph"
        )

    @staticmethod
    def _hold(snapshot: VLMInput) -> WaypointPathResponse | None:
        """Stand-still waypoint at the current pose (Gemini-failure fallback)."""
        if snapshot.pose is None:
            return None
        p = snapshot.pose.position
        return WaypointPathResponse(
            waypoints=[Waypoint(x=p.x, y=p.y, heading=0.0)],
            rationale="scene_gemini: holding position (Gemini call failed, retrying)",
        )

    @staticmethod
    def _reached(
        snapshot: VLMInput,
        wp: Waypoint,
        *,
        reach_dist: float = _REACH_DIST_M,
    ) -> bool:
        if snapshot.pose is None:
            return False
        dx = snapshot.pose.position.x - wp.x
        dy = snapshot.pose.position.y - wp.y
        return (dx * dx + dy * dy) ** 0.5 < reach_dist
