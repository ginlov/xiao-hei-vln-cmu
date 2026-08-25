"""``SceneClaudeResponder`` — perception scene graph + Claude object-reference answering.

The Claude counterpart to :class:`SceneGeminiResponder`, specialised for the
question-directed Task-1 flow:

1. **Scene building (2 Hz).** :meth:`ingest` delegates to a real
   :class:`~xiao_hei_vln.perception.PerceptionResponder`, running the sidecar's
   detect → lift → fuse cycle into the shared
   :class:`~xiao_hei_vln.scene.SceneRepresentation` on every tick — including
   while the robot is driving toward the target object.
2. **Answering at arrival.** The app tick loop only calls :meth:`respond` once
   the (question-directed) explorer has completed — i.e. the model declared the
   robot arrived at the target object. For an OBJECT_REFERENCE question we dump
   the full scene graph to Claude, which picks the ``object_id`` the referring
   expression denotes, and return the corresponding
   :class:`ObjectReferenceResponse`.

Only OBJECT_REFERENCE ("the first type of question") is handled by Claude here;
NUMERICAL / INSTRUCTION_FOLLOWING questions fall back to the wrapped perception
responder so the mode degrades gracefully rather than refusing to answer.

The responder protocol (``respond / ingest / is_done / reset / close``) matches
:class:`PerceptionResponder`, so it slots into ``app/main.py`` via
``XIAO_HEI_RESPONDER=scene_claude``.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from xiao_hei_vln.messages import (
    ObjectReferenceResponse,
    QuestionType,
    Vector3,
    VLMInput,
    VLMOutput,
    Waypoint,
    WaypointPathResponse,
)
from xiao_hei_vln.nav_vlm.task1_prompts import (
    ANSWER_OBJECT_REFERENCE_TOOL,
    ANSWER_SYSTEM_PROMPT,
    build_answer_user_text,
)
from xiao_hei_vln.perception.global_map import GlobalMap

if TYPE_CHECKING:  # pragma: no cover
    from xiao_hei_vln.nav_vlm.config import NavVLMConfig
    from xiao_hei_vln.nav_vlm.engine import ToolCallerProtocol
    from xiao_hei_vln.perception.responder import PerceptionResponder
    from xiao_hei_vln.scene import SceneRepresentation
    from xiao_hei_vln.scene.representation import ObjectObservation

log = logging.getLogger(__name__)

_MAX_TRAJECTORY = 200
# How many answer ticks to keep retrying a failed / empty Claude call before
# committing a best-effort label fallback so the run always produces an answer.
_MAX_ANSWER_ATTEMPTS = 3


class SceneClaudeResponder:
    """Answer an OBJECT_REFERENCE question with Claude, from the built scene graph."""

    def __init__(
        self,
        engine: ToolCallerProtocol,
        config: NavVLMConfig,
        scene: SceneRepresentation,
        *,
        perception: PerceptionResponder,
        global_map: GlobalMap | None = None,
        logger=None,
    ) -> None:
        self._engine = engine
        self._config = config
        self._scene = scene
        self._perception = perception
        self._map = global_map if global_map is not None else GlobalMap()
        self._logger = logger

        self._tick_count = 0
        self._answer_attempts = 0
        self._trajectory_xy: list[tuple[float, float]] = []
        self._committed: VLMOutput | None = None
        self._done = False
        # True once respond() has routed a non-OBJECT_REFERENCE question to the
        # wrapped perception responder, so is_done() mirrors that responder.
        self._delegating = False

    # ------------------------------------------------------------------ scene building

    def ingest(self, snapshot: VLMInput) -> None:
        """Build the scene from this frame without answering (2 Hz)."""
        self._update_map(snapshot)
        self._accumulate_trajectory(snapshot)
        self._perception.ingest(snapshot)

    # ------------------------------------------------------------------ answering

    def respond(self, snapshot: VLMInput) -> VLMOutput | None:
        if snapshot.question is None or self._done:
            return None

        self._tick_count += 1
        if self._tick_count == 1 and self._logger is not None:
            try:
                self._logger.new_question(snapshot.question.text)
            except Exception:  # noqa: BLE001
                log.exception("VLMLogger.new_question failed; continuing")

        self._update_map(snapshot)
        self._accumulate_trajectory(snapshot)

        if snapshot.question.type is not QuestionType.OBJECT_REFERENCE:
            # Out of scope for the Claude answerer — hand to perception.
            self._delegating = True
            return self._perception.respond(snapshot)

        return self._answer_object_reference(snapshot)

    def is_done(self) -> bool:
        if self._delegating:
            return self._perception.is_done()
        return self._done

    def reset(self) -> None:
        self._perception.reset()
        self._tick_count = 0
        self._answer_attempts = 0
        self._trajectory_xy = []
        self._committed = None
        self._done = False
        self._delegating = False

    def close(self) -> None:
        self._perception.close()
        if self._logger is not None:
            self._logger.close()

    # ------------------------------------------------------------------ object reference

    def _answer_object_reference(self, snapshot: VLMInput) -> VLMOutput | None:
        if self._committed is not None:
            return self._committed

        tool_input = self._call_claude(snapshot)
        chosen: ObjectObservation | None = None
        rationale = ""
        if tool_input is not None:
            oid = _as_int(tool_input.get("object_id"))
            rationale = str(tool_input.get("rationale", ""))
            if oid is not None and oid >= 0:
                chosen = next(
                    (o for o in self._scene.objects if o.object_id == oid), None
                )

        if chosen is None:
            # Claude failed, timed out, or named an id not in the graph. Retry a
            # few ticks (the scene may still be settling), then fall back to a
            # label match so the run always emits an answer.
            self._answer_attempts += 1
            if self._answer_attempts < _MAX_ANSWER_ATTEMPTS:
                return self._hold(snapshot)
            chosen = self._label_fallback(snapshot)
            rationale = "scene_claude: label/proximity fallback (no valid model pick)"
            if chosen is None:
                log.warning("scene_claude: no object to answer with; giving up")
                self._done = True
                return self._hold(snapshot)

        answer = ObjectReferenceResponse(
            label=chosen.label,
            object_id=chosen.object_id,
            center=chosen.position,
            size=_size_from_bbox(chosen),
            heading=0.0,
            rationale=rationale or "scene_claude",
        )
        self._committed = answer
        self._done = True
        log.info(
            "scene_claude: answered object_reference with #%d '%s' (%s)",
            chosen.object_id, chosen.label, rationale,
        )
        return answer

    def _call_claude(self, snapshot: VLMInput) -> dict | None:
        """Build the bundle, call Claude's answer tool; None on any failure."""
        scene_dict = self._scene.to_dict()
        user_text = build_answer_user_text(
            question=snapshot.question.text, scene_dict=scene_dict,
        )
        images = self._build_images(snapshot)

        t0 = time.perf_counter()
        tool_input: dict | None = None
        try:
            tool_input = self._engine.call_tool(
                system=ANSWER_SYSTEM_PROMPT,
                tool=ANSWER_OBJECT_REFERENCE_TOOL,
                user_text=user_text,
                images=images,
            )
        except Exception:
            log.exception("scene_claude: Claude answer call failed this tick")
        finally:
            inference_ms = (time.perf_counter() - t0) * 1000.0
            self._log_tick(snapshot, user_text, tool_input, inference_ms)
        return tool_input

    def _build_images(self, snapshot: VLMInput) -> list[tuple[bytes, str]]:
        """Occupancy map + latest panorama, best-effort (never fail the answer)."""
        images: list[tuple[bytes, str]] = []
        try:
            # Reuse the backend-agnostic serialization from the gemini package
            # (no google SDK is imported by these helpers).
            from xiao_hei_vln.gemini.scene_rep import build_bundle

            bundle = build_bundle(
                snapshot=snapshot,
                scene=self._scene,
                global_map=self._map,
                trajectory_xy=list(self._trajectory_xy),
                panorama_long_edge=self._config.image_long_edge,
                occupancy_dpi=self._config.occupancy_dpi,
            )
            if bundle.panorama_jpg_bytes is not None:
                images.append((bundle.panorama_jpg_bytes, "image/jpeg"))
            images.append((bundle.occupancy_png_bytes, "image/png"))
        except Exception as exc:  # noqa: BLE001 — images are optional context
            log.warning("scene_claude: image bundle failed (%s); text-only answer", exc)
        return images

    def _label_fallback(self, snapshot: VLMInput) -> ObjectObservation | None:
        """Closest object whose label appears in the question text."""
        text = snapshot.question.text.lower()
        candidates = [
            o for o in self._scene.objects if _label_in_text(o.label, text)
        ]
        if not candidates:
            return None
        origin = snapshot.pose.position if snapshot.pose is not None else None
        if origin is not None:
            candidates.sort(
                key=lambda o: (o.position.x - origin.x) ** 2 + (o.position.y - origin.y) ** 2
            )
        return candidates[0]

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

    def _log_tick(self, snapshot, user_text, tool_input, inference_ms) -> None:
        if self._logger is None:
            return
        try:
            self._logger.log_tick(
                snapshot,
                ANSWER_SYSTEM_PROMPT,
                user_text,
                None,
                inference_ms,
                [f"scene_objects={len(self._scene.objects)}", f"tool_input={tool_input}"],
            )
        except Exception:
            log.exception("VLMLogger.log_tick failed; continuing")

    @staticmethod
    def _hold(snapshot: VLMInput) -> WaypointPathResponse | None:
        """Stand-still waypoint (retry / give-up fallback)."""
        if snapshot.pose is None:
            return None
        p = snapshot.pose.position
        return WaypointPathResponse(
            waypoints=[Waypoint(x=p.x, y=p.y, heading=0.0)],
            rationale="scene_claude: holding (answer pending)",
        )


# ---------------------------------------------------------------------------
# module helpers


def _as_int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _size_from_bbox(o: ObjectObservation) -> Vector3:
    """AABB extent from the object's bbox_min/bbox_max (zero if unavailable)."""
    mn, mx = o.bbox_min, o.bbox_max
    if mn is None or mx is None:
        return Vector3(x=0.0, y=0.0, z=0.0)
    return Vector3(x=abs(mx.x - mn.x), y=abs(mx.y - mn.y), z=abs(mx.z - mn.z))


def _label_in_text(label: str, text_lower: str) -> bool:
    lab = label.lower().replace("_", " ")
    return lab in text_lower or (lab + "s") in text_lower
