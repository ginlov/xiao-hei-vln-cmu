"""Marker-based Gemini responder for all three challenge question types."""

from __future__ import annotations

import logging
from typing import Any

from xiao_hei_vln.eval_sampler.object_list import ObjectEntry
from xiao_hei_vln.gemini.engine import GeminiEngineProtocol
from xiao_hei_vln.gemini.object_provider import ObjectEntryProvider, object_entries_to_markers
from xiao_hei_vln.gemini.prompts import build_system_prompt, build_user_message
from xiao_hei_vln.messages import (
    NumericalResponse,
    ObjectReferenceResponse,
    QuestionType,
    VLMInput,
    VLMOutput,
    Waypoint,
    WaypointPathResponse,
    parse_vlm_output,
)

log = logging.getLogger(__name__)


class GeminiResponder:
    """Consumes a VLMInput plus file-backed object entries and emits VLMOutput."""

    def __init__(
        self,
        engine: GeminiEngineProtocol,
        object_provider: ObjectEntryProvider,
        *,
        max_ticks_per_question: int = 30,
    ) -> None:
        self._engine = engine
        self._object_provider = object_provider
        self._max_ticks_per_question = max_ticks_per_question
        self._done = False
        self._tick_count = 0
        self._evidence: list[str] = []

    def respond(self, snapshot: VLMInput) -> VLMOutput | None:
        if snapshot.question is None or self._done:
            return None

        if self._tick_count >= self._max_ticks_per_question:
            return self._timeout_response(snapshot.question.type)

        self._tick_count += 1
        objects = self._object_provider.get()
        markers = object_entries_to_markers(objects)
        system_prompt = build_system_prompt(snapshot.question.type)
        user_text = build_user_message(snapshot, snapshot.question, markers, self._evidence)

        try:
            raw = self._engine.infer(system_prompt, user_text, snapshot.image)
            output = self._coerce_output(snapshot.question.type, raw, objects)
        except Exception:
            log.exception("Gemini inference failed; skipping tick")
            return None

        if output is None:
            self._record_raw_evidence(raw)
            return None

        self._record_output_evidence(output)
        self._update_done_flag(snapshot.question.type, output)
        return output

    def is_done(self) -> bool:
        return self._done

    def reset(self) -> None:
        self._done = False
        self._tick_count = 0
        self._evidence = []

    def close(self) -> None:
        pass

    def _coerce_output(
        self,
        qtype: QuestionType,
        raw: dict[str, Any],
        objects: dict[int, ObjectEntry],
    ) -> VLMOutput | None:
        if "kind" in raw:
            return parse_vlm_output(raw)
        if qtype is QuestionType.NUMERICAL:
            return _coerce_numerical(raw)
        if qtype is QuestionType.OBJECT_REFERENCE:
            return _coerce_object_reference(raw, objects)
        return _coerce_waypoints(raw)

    def _update_done_flag(self, qtype: QuestionType, output: VLMOutput) -> None:
        if isinstance(output, (NumericalResponse, ObjectReferenceResponse)):
            self._done = True
            return
        if qtype is QuestionType.INSTRUCTION_FOLLOWING:
            self._done = True

    def _record_output_evidence(self, output: VLMOutput) -> None:
        rationale = output.rationale.strip() if output.rationale else ""
        if isinstance(output, NumericalResponse):
            entry = f"answered NUMERICAL={output.value}"
        elif isinstance(output, ObjectReferenceResponse):
            c = output.center
            entry = (
                f"selected OBJECT id={output.object_id} label={output.label!r} "
                f"center=({c.x:.2f},{c.y:.2f},{c.z:.2f})"
            )
        else:
            wp = output.waypoints[0]
            entry = f"navigated toward waypoint ({wp.x:.2f},{wp.y:.2f})"
        if rationale:
            entry = f"{entry} :: {rationale}"
        self._evidence.append(entry)

    def _record_raw_evidence(self, raw: dict[str, Any]) -> None:
        reasoning = raw.get("reasoning") or raw.get("rationale")
        if isinstance(reasoning, str) and reasoning.strip():
            self._evidence.append(reasoning.strip())

    def _timeout_response(self, qtype: QuestionType) -> VLMOutput | None:
        self._done = True
        if qtype is QuestionType.NUMERICAL:
            return NumericalResponse(
                value=0,
                rationale=f"timeout after {self._max_ticks_per_question} ticks",
            )
        return None


def _coerce_numerical(raw: dict[str, Any]) -> NumericalResponse | WaypointPathResponse | None:
    if "waypoints" in raw:
        return _coerce_waypoints(raw)

    value = raw.get("answer", raw.get("value"))
    if isinstance(value, bool) or not isinstance(value, int):
        log.warning("Gemini numerical response missing integer answer: %r", raw)
        return None
    return NumericalResponse(value=value, rationale=_reasoning(raw))


def _coerce_object_reference(
    raw: dict[str, Any],
    objects: dict[int, ObjectEntry],
) -> ObjectReferenceResponse | WaypointPathResponse | None:
    if "waypoints" in raw:
        return _coerce_waypoints(raw)

    selected = raw.get("selected_marker_id", raw.get("object_id"))
    if selected is None:
        log.warning("Gemini did not select an object marker: %r", raw)
        return None
    if isinstance(selected, bool) or not isinstance(selected, int):
        log.warning("Gemini selected marker id is not an int: %r", raw)
        return None

    obj = objects.get(selected)
    if obj is None:
        log.warning("Gemini selected unknown marker id=%s", selected)
        return None

    return ObjectReferenceResponse(
        label=obj.label,
        object_id=obj.object_id,
        center=obj.center,
        size=obj.size,
        heading=obj.heading,
        rationale=_reasoning(raw),
    )


def _coerce_waypoints(raw: dict[str, Any]) -> WaypointPathResponse | None:
    raw_waypoints = raw.get("waypoints")
    if not isinstance(raw_waypoints, list) or not raw_waypoints:
        log.warning("Gemini waypoint response missing non-empty waypoints: %r", raw)
        return None

    waypoints: list[Waypoint] = []
    for item in raw_waypoints:
        if not isinstance(item, dict):
            log.warning("Gemini waypoint item is not an object: %r", item)
            return None
        x = item.get("x")
        y = item.get("y")
        heading = item.get("heading", item.get("theta", 0.0))
        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, int | float)
            or not isinstance(y, int | float)
        ):
            log.warning("Gemini waypoint missing numeric x/y: %r", item)
            return None
        if isinstance(heading, bool) or not isinstance(heading, int | float):
            log.warning("Gemini waypoint heading is not numeric: %r", item)
            return None
        waypoints.append(Waypoint(x=float(x), y=float(y), heading=float(heading)))

    return WaypointPathResponse(waypoints=waypoints, rationale=_reasoning(raw))


def _reasoning(raw: dict[str, Any]) -> str | None:
    value = raw.get("reasoning", raw.get("rationale"))
    return value.strip() if isinstance(value, str) and value.strip() else None
