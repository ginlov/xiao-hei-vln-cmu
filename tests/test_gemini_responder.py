"""Tests for the file-backed Gemini responder without calling Gemini."""

from __future__ import annotations

from typing import Any

from xiao_hei_vln.gemini import GeminiResponder, ObjectEntryProvider, object_entries_to_markers
from xiao_hei_vln.messages import (
    ChallengeQuestion,
    Header,
    NumericalResponse,
    ObjectReferenceResponse,
    OdomPose,
    Quaternion,
    Stamp,
    Vector3,
    VLMInput,
    WaypointPathResponse,
)
from xiao_hei_vln.messages.sensors import ImageFrame


def _stamp(t: float = 0.0) -> Stamp:
    return Stamp.from_seconds(t)


def _pose(x: float = 0.0, y: float = 0.0) -> OdomPose:
    return OdomPose(
        header=Header(stamp=_stamp(), frame_id="map"),
        position=Vector3(x=x, y=y, z=0.75),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )


def _input(question_text: str) -> VLMInput:
    return VLMInput(
        tick_id=0,
        tick_time=_stamp(),
        question=ChallengeQuestion.from_text(question_text, _stamp()),
        pose=_pose(),
    )


def _provider(tmp_path) -> ObjectEntryProvider:
    path = tmp_path / "object_list.txt"
    path.write_text(
        "\n".join(
            [
                '72 1.00 2.00 3.00 0.50 0.60 0.70 0.00 "book"',
                '5 10.00 20.00 0.50 1.00 1.00 0.80 1.57 "table"',
            ],
        ),
    )
    return ObjectEntryProvider(path)


class ScriptedGeminiEngine:
    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        self._outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    def infer(
        self,
        system: str,
        user_text: str,
        image: ImageFrame | None,
    ) -> dict[str, Any]:
        self.calls.append({"system": system, "user_text": user_text, "image": image})
        if not self._outputs:
            raise AssertionError("ScriptedGeminiEngine ran out of outputs")
        return self._outputs.pop(0)


def test_object_provider_serializes_text_file(tmp_path) -> None:
    provider = _provider(tmp_path)
    markers = object_entries_to_markers(provider.get())

    assert markers[0]["object_id"] == 5
    assert markers[0]["label"] == "table"
    assert markers[0]["center"] == {"x": 10.0, "y": 20.0, "z": 0.5}
    assert markers[0]["color"] is None


def test_numerical_answer_from_gemini_json(tmp_path) -> None:
    engine = ScriptedGeminiEngine([{"answer": 2, "reasoning": "two books"}])
    responder = GeminiResponder(engine, _provider(tmp_path))

    out = responder.respond(_input("How many books are there?"))

    assert isinstance(out, NumericalResponse)
    assert out.value == 2
    assert out.rationale == "two books"
    assert responder.is_done() is True
    assert '"label": "book"' in engine.calls[0]["user_text"]


def test_object_reference_selected_id_maps_to_bbox(tmp_path) -> None:
    engine = ScriptedGeminiEngine(
        [{"selected_marker_id": 72, "reasoning": "book satisfies the relation"}],
    )
    responder = GeminiResponder(engine, _provider(tmp_path))

    out = responder.respond(_input("Find the book above the table."))

    assert isinstance(out, ObjectReferenceResponse)
    assert out.object_id == 72
    assert out.label == "book"
    assert out.center.x == 1.0
    assert out.size.z == 0.7
    assert out.heading == 0.0
    assert responder.is_done() is True


def test_instruction_following_waypoints_are_terminal(tmp_path) -> None:
    engine = ScriptedGeminiEngine(
        [
            {
                "waypoints": [{"x": 3.0, "y": 4.0, "heading": 1.57}],
                "reasoning": "go there",
            },
        ],
    )
    responder = GeminiResponder(engine, _provider(tmp_path))

    out = responder.respond(_input("Go near the table."))

    assert isinstance(out, WaypointPathResponse)
    assert out.waypoints[0].x == 3.0
    assert out.waypoints[0].heading == 1.57
    assert responder.is_done() is True


def test_object_provider_reloads_when_file_changes(tmp_path) -> None:
    path = tmp_path / "object_list.txt"
    path.write_text('1 0 0 0 1 1 1 0 "chair"')
    provider = ObjectEntryProvider(path)
    assert provider.get()[1].label == "chair"

    path.write_text('2 0 0 0 1 1 1 0 "lamp"')
    assert provider.get()[2].label == "lamp"
