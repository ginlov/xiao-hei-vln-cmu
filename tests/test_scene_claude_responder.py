"""Unit tests for SceneClaudeResponder — object-reference answering via Claude.

A fake engine returns scripted ``call_tool`` results and a fake perception
responder stands in for the sidecar, so no key, network, or GPU is touched.
The image bundle (matplotlib) is stubbed out per-instance.
"""

from __future__ import annotations

import pytest

from xiao_hei_vln.messages import (
    Header,
    NumericalResponse,
    ObjectReferenceResponse,
    OdomPose,
    Quaternion,
    Stamp,
    Vector3,
)
from xiao_hei_vln.messages.inputs import VLMInput
from xiao_hei_vln.messages.question import ChallengeQuestion
from xiao_hei_vln.nav_vlm.config import NavVLMConfig
from xiao_hei_vln.scene.representation import ObjectObservation
from xiao_hei_vln.scene_claude.responder import SceneClaudeResponder


class FakeEngine:
    def __init__(self, result=None, *, raises=False) -> None:
        self._result = result
        self._raises = raises
        self.calls = 0

    def call_tool(self, *, system, tool, user_text, images):
        self.calls += 1
        if self._raises:
            raise RuntimeError("boom")
        return self._result


class FakePerception:
    def __init__(self) -> None:
        self.ingested = 0
        self.responded = 0
        self._done = False

    def ingest(self, snapshot) -> None:
        self.ingested += 1

    def respond(self, snapshot):
        self.responded += 1
        self._done = True
        return NumericalResponse(value=3)

    def is_done(self) -> bool:
        return self._done

    def reset(self) -> None: ...
    def close(self) -> None: ...


class FakeScene:
    def __init__(self, objects: list[ObjectObservation]) -> None:
        self._objects = objects

    @property
    def objects(self) -> list[ObjectObservation]:
        return self._objects

    def to_dict(self) -> dict:
        return {
            "objects": [
                {"object_id": o.object_id, "label": o.label,
                 "position": {"x": o.position.x, "y": o.position.y, "z": o.position.z}}
                for o in self._objects
            ]
        }


def _obj(oid: int, label: str, x: float, y: float) -> ObjectObservation:
    return ObjectObservation(
        label=label,
        position=Vector3(x=x, y=y, z=0.0),
        bbox_min=Vector3(x=x - 0.5, y=y - 0.5, z=0.0),
        bbox_max=Vector3(x=x + 0.5, y=y + 0.5, z=1.0),
        object_id=oid,
    )


def _snapshot(text: str = "the red chair") -> VLMInput:
    stamp = Stamp.from_seconds(1.0)
    pose = OdomPose(
        header=Header(stamp=stamp, frame_id="map"),
        position=Vector3(x=0.0, y=0.0, z=0.0),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    return VLMInput(
        tick_id=1, tick_time=stamp, pose=pose,
        question=ChallengeQuestion.from_text(text, stamp),
    )


def _make(engine, scene) -> SceneClaudeResponder:
    r = SceneClaudeResponder(
        engine, NavVLMConfig(api_key="sk-test"), scene, perception=FakePerception(),
    )
    r._build_images = lambda snapshot: []  # skip matplotlib
    return r


def test_claude_pick_becomes_object_reference_answer() -> None:
    scene = FakeScene([_obj(5, "red chair", 3.0, 1.0), _obj(6, "blue chair", -2.0, 0.0)])
    eng = FakeEngine({"object_id": 5, "label": "red chair", "rationale": "the red one"})
    r = _make(eng, scene)

    out = r.respond(_snapshot("the red chair"))

    assert isinstance(out, ObjectReferenceResponse)
    assert out.object_id == 5 and out.label == "red chair"
    assert (out.center.x, out.center.y) == (3.0, 1.0)
    assert out.size.x == pytest.approx(1.0)  # bbox extent
    assert r.is_done()


def test_ingest_builds_scene_without_answering() -> None:
    scene = FakeScene([])
    r = _make(FakeEngine(None), scene)
    r.ingest(_snapshot())
    r.ingest(_snapshot())
    assert r._perception.ingested == 2
    assert not r.is_done()


def test_invalid_id_retries_then_label_fallback() -> None:
    scene = FakeScene([_obj(9, "red chair", 4.0, 0.0)])
    eng = FakeEngine({"object_id": -1, "rationale": "none"})
    r = _make(eng, scene)

    # First attempts hold (retry while the scene may still settle) …
    out1 = r.respond(_snapshot("the red chair"))
    assert not isinstance(out1, ObjectReferenceResponse)
    assert not r.is_done()
    r.respond(_snapshot("the red chair"))
    # … then a label/proximity fallback commits an answer.
    out3 = r.respond(_snapshot("the red chair"))
    assert isinstance(out3, ObjectReferenceResponse)
    assert out3.object_id == 9
    assert r.is_done()


def test_engine_failure_is_tolerated() -> None:
    scene = FakeScene([_obj(2, "red chair", 1.0, 1.0)])
    eng = FakeEngine(None, raises=True)
    r = _make(eng, scene)
    # Never raises out; eventually falls back to the label match.
    out = None
    for _ in range(3):
        out = r.respond(_snapshot("the red chair"))
    assert isinstance(out, ObjectReferenceResponse)
    assert out.object_id == 2


def test_non_object_reference_delegates_to_perception() -> None:
    scene = FakeScene([])
    r = _make(FakeEngine(None), scene)

    out = r.respond(_snapshot("how many chairs are there"))

    assert isinstance(out, NumericalResponse)
    assert r._perception.responded == 1
    assert r.is_done()  # mirrors the delegated responder
