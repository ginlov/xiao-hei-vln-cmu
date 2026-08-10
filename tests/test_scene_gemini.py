"""Tests for :class:`xiao_hei_vln.scene_gemini.SceneGeminiResponder`.

The responder is driven by the *app-level* frontier explorer, so its
contract is:

  - :meth:`ingest` (called every exploration tick) builds the scene via a
    perception responder and emits **no** answer, even with a question
    active.
  - :meth:`respond` (called only after exploration completes) asks Gemini
    once for Task 1 and commits, or plans + steps a route for Task 2.

A ``FakeGeminiEngine`` is injected — no SDK or API key needed. A
``FakePerception`` stands in for the sidecar-backed perception responder
and (optionally) drops objects into the shared scene, so we can prove the
populated graph is what reaches Gemini.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from xiao_hei_vln.gemini.config import GeminiConfig
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
    Waypoint,
    WaypointPathResponse,
)
from xiao_hei_vln.messages.sensors import TerrainMap
from xiao_hei_vln.scene import SceneRepresentation
from xiao_hei_vln.scene_gemini import SceneGeminiResponder


def _stamp(t: float = 0.0) -> Stamp:
    return Stamp.from_seconds(t)


def _pose(x: float, y: float) -> OdomPose:
    return OdomPose(
        header=Header(stamp=_stamp(), frame_id="map"),
        position=Vector3(x=x, y=y, z=0.75),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )


def _terrain() -> TerrainMap:
    return TerrainMap(
        header=Header(stamp=_stamp(), frame_id="map"),
        points=np.zeros((0, 4), dtype=np.float32),
        range="ext_20m",
    )


def _snapshot(
    *,
    tick_id: int,
    question_text: str | None,
    pose: OdomPose | None = None,
) -> VLMInput:
    return VLMInput(
        tick_id=tick_id,
        tick_time=_stamp(tick_id * 0.5),
        question=(
            ChallengeQuestion.from_text(question_text, _stamp())
            if question_text is not None
            else None
        ),
        pose=pose,
        terrain_ext=_terrain(),
    )


@dataclass
class FakeGeminiEngine:
    """Returns a canned response; records the calls it received."""

    response: object | None = None
    calls: list[dict] = field(default_factory=list)
    warmed_up: bool = False

    def infer_multimodal(self, *, system: str, user_text: str, images: list[bytes]) -> object:
        self.calls.append({"system": system, "user_text": user_text, "images": list(images)})
        return self.response

    def warmup(self) -> None:
        self.warmed_up = True


class FakePerception:
    """Stand-in for the perception responder used via ``ingest``.

    When ``add_label`` is set, each ``ingest`` drops one object of that
    label into the shared scene — so a test can prove the populated scene
    graph reaches Gemini.
    """

    def __init__(self, scene: SceneRepresentation | None = None, *, add_label: str | None = None) -> None:
        self._scene = scene
        self._add_label = add_label
        self.ingest_calls = 0
        self.reset_calls = 0
        self.closed = False

    def ingest(self, snapshot: VLMInput) -> None:
        self.ingest_calls += 1
        if self._add_label and self._scene is not None:
            # One fused node per ingest, mirroring the real responder: it
            # re-syncs the whole object layer from the ObjectMap each tick,
            # so every node seen so far must be re-sent.
            self._scene.sync_from_object_map([
                {
                    "node_id": i,
                    "label": self._add_label,
                    "score": 1.0,
                    "center_3d": [float(i) * 3.0, 0.0, 0.5],
                    "bbox_aabb": {"min": [float(i) * 3.0 - 0.1, -0.1, 0.4],
                                  "max": [float(i) * 3.0 + 0.1, 0.1, 0.6]},
                    "color_rgb": None,
                    "color_name": None,
                }
                for i in range(1, self.ingest_calls + 1)
            ])

    def reset(self) -> None:
        self.reset_calls += 1

    def close(self) -> None:
        self.closed = True


def _config(**overrides) -> GeminiConfig:
    base = dict(api_key="test")
    base.update(overrides)
    return GeminiConfig(**base)


def _build(engine, *, scene=None, perception=None, logger=None):
    scene = scene if scene is not None else SceneRepresentation()
    perception = perception if perception is not None else FakePerception(scene)
    responder = SceneGeminiResponder(
        engine, _config(), scene, perception=perception, logger=logger,
    )
    return responder, scene, perception


# ===========================================================================
# Exploration: ingest builds the scene, emits no answer
# ===========================================================================


class TestIngest:
    def test_ingest_builds_scene_without_answering(self) -> None:
        engine = FakeGeminiEngine(response=NumericalResponse(value=1, rationale="x"))
        scene = SceneRepresentation()
        perception = FakePerception(scene, add_label="cup")
        responder, _, _ = _build(engine, scene=scene, perception=perception)

        for t in range(1, 4):
            responder.ingest(
                _snapshot(tick_id=t, question_text="How many cups", pose=_pose(t * 0.1, 0)),
            )

        assert perception.ingest_calls == 3
        assert not engine.calls              # ingest never calls Gemini
        assert not responder.is_done()
        assert len(scene.objects) == 3       # perception populated the shared scene


# ===========================================================================
# Task 1 — numerical
# ===========================================================================


class TestTask1Numerical:
    def test_first_respond_calls_gemini_and_commits(self) -> None:
        engine = FakeGeminiEngine(response=NumericalResponse(value=4, rationale="counted 4"))
        scene = SceneRepresentation()
        perception = FakePerception(scene, add_label="cup")
        responder, _, _ = _build(engine, scene=scene, perception=perception)

        # Exploration built the scene.
        for t in range(1, 3):
            responder.ingest(_snapshot(tick_id=t, question_text="How many cups", pose=_pose(0, 0)))

        # First respond → one Gemini call, commit, done.
        out = responder.respond(_snapshot(tick_id=3, question_text="How many cups", pose=_pose(0, 0)))
        assert isinstance(out, NumericalResponse)
        assert out.value == 4
        assert len(engine.calls) == 1
        assert responder.is_done()
        assert '"kind": "numerical"' in engine.calls[0]["system"]
        # The populated object graph reached Gemini.
        assert "cup" in engine.calls[0]["user_text"]

    def test_done_responder_returns_none(self) -> None:
        engine = FakeGeminiEngine(response=NumericalResponse(value=2, rationale="ok"))
        responder, _, _ = _build(engine)

        out1 = responder.respond(_snapshot(tick_id=1, question_text="How many", pose=_pose(0, 0)))
        assert isinstance(out1, NumericalResponse)
        assert responder.is_done()
        out2 = responder.respond(_snapshot(tick_id=2, question_text="How many", pose=_pose(0, 0)))
        assert out2 is None
        assert len(engine.calls) == 1        # no second Gemini call

    def test_no_question_returns_none(self) -> None:
        engine = FakeGeminiEngine()
        responder, _, _ = _build(engine)
        out = responder.respond(_snapshot(tick_id=0, question_text=None, pose=_pose(0, 0)))
        assert out is None
        assert not engine.calls


# ===========================================================================
# Task 1 — object reference
# ===========================================================================


class TestTask1ObjectReference:
    def test_routes_to_object_reference_prompt(self) -> None:
        engine = FakeGeminiEngine(
            response=ObjectReferenceResponse(
                label="sofa",
                object_id=0,
                center=Vector3(x=2.0, y=1.0, z=0.5),
                size=Vector3(x=2.0, y=0.9, z=0.8),
            ),
        )
        responder, _, _ = _build(engine)
        out = responder.respond(_snapshot(tick_id=1, question_text="Find the sofa", pose=_pose(0, 0)))
        assert isinstance(out, ObjectReferenceResponse)
        assert out.label == "sofa"
        assert '"kind": "object_reference"' in engine.calls[0]["system"]


# ===========================================================================
# Task 2 — instruction following
# ===========================================================================


class TestTask2InstructionFollowing:
    def test_plans_once_then_steps_through_waypoints(self) -> None:
        plan = WaypointPathResponse(
            waypoints=[
                Waypoint(x=1.0, y=0.0, heading=0.0),
                Waypoint(x=2.0, y=0.0, heading=0.0),
                Waypoint(x=3.0, y=0.0, heading=0.0),
            ],
            rationale="three-step",
        )
        engine = FakeGeminiEngine(response=plan)
        responder, _, _ = _build(engine)

        out1 = responder.respond(_snapshot(tick_id=1, question_text="Go forward", pose=_pose(0, 0)))
        assert isinstance(out1, WaypointPathResponse)
        assert out1.waypoints[0].x == pytest.approx(1.0)
        assert len(engine.calls) == 1
        assert '"kind": "waypoint_path"' in engine.calls[0]["system"]

        # Still far from wp0 → keep emitting wp0, no second Gemini call.
        out2 = responder.respond(_snapshot(tick_id=2, question_text="Go forward", pose=_pose(0.2, 0)))
        assert out2.waypoints[0].x == pytest.approx(1.0)
        assert len(engine.calls) == 1

        # Reached wp0 → advance to wp1.
        out3 = responder.respond(_snapshot(tick_id=3, question_text="Go forward", pose=_pose(1.0, 0)))
        assert out3.waypoints[0].x == pytest.approx(2.0)
        assert not responder.is_done()

        # Reach wp1 → wp2.
        out4 = responder.respond(_snapshot(tick_id=4, question_text="Go forward", pose=_pose(2.0, 0)))
        assert out4.waypoints[0].x == pytest.approx(3.0)
        assert not responder.is_done()

        # Reach final wp2 → done.
        out5 = responder.respond(_snapshot(tick_id=5, question_text="Go forward", pose=_pose(3.0, 0)))
        assert out5.waypoints[0].x == pytest.approx(3.0)
        assert responder.is_done()

    def test_gemini_returning_non_waypoint_marks_done(self) -> None:
        engine = FakeGeminiEngine(response=NumericalResponse(value=0, rationale="oops"))
        responder, _, _ = _build(engine)
        out = responder.respond(_snapshot(tick_id=1, question_text="Go to the kitchen", pose=_pose(0, 0)))
        assert out is None
        assert responder.is_done()


# ===========================================================================
# Failure handling + lifecycle
# ===========================================================================


class TestEngineFailureFallback:
    def test_task1_gemini_raises_then_holds_and_retries(self) -> None:
        class FlakyEngine:
            def __init__(self) -> None:
                self.calls = 0

            def infer_multimodal(self, **_kwargs):
                self.calls += 1
                raise RuntimeError("network blip")

            def warmup(self) -> None:
                pass

        engine = FlakyEngine()
        responder, _, _ = _build(engine)

        out = responder.respond(_snapshot(tick_id=1, question_text="How many cups", pose=_pose(0, 0)))
        assert engine.calls == 1
        assert not responder.is_done()
        # Falls back to a stand-still hold; next tick will retry.
        assert isinstance(out, WaypointPathResponse)
        assert out.waypoints[0].x == pytest.approx(0.0)

        responder.respond(_snapshot(tick_id=2, question_text="How many cups", pose=_pose(0, 0)))
        assert engine.calls == 2             # retried


@dataclass
class FakeLogger:
    new_questions: list[str] = field(default_factory=list)
    ticks: list[dict] = field(default_factory=list)
    closed: bool = False

    def new_question(self, text: str) -> None:
        self.new_questions.append(text)

    def log_tick(self, snapshot, system_prompt, user_text, output, inference_ms, evidence) -> None:
        self.ticks.append({
            "tick_id": snapshot.tick_id,
            "system_prompt": system_prompt,
            "output": output,
        })

    def close(self) -> None:
        self.closed = True


class TestLoggerIntegration:
    def test_logger_receives_new_question_log_tick_close(self) -> None:
        engine = FakeGeminiEngine(response=NumericalResponse(value=4, rationale="ok"))
        logger = FakeLogger()
        responder, _, perception = _build(engine, logger=logger)

        responder.respond(_snapshot(tick_id=1, question_text="How many", pose=_pose(0, 0)))
        assert logger.new_questions == ["How many"]
        assert len(logger.ticks) == 1
        assert '"kind": "numerical"' in logger.ticks[0]["system_prompt"]
        assert logger.ticks[0]["output"] is not None

        responder.close()
        assert logger.closed is True
        assert perception.closed is True


class TestLifecycle:
    def test_reset_clears_state_and_perception(self) -> None:
        engine = FakeGeminiEngine(response=NumericalResponse(value=1, rationale="ok"))
        responder, _, perception = _build(engine)

        responder.respond(_snapshot(tick_id=1, question_text="How many", pose=_pose(0, 0)))
        assert responder.is_done()

        responder.reset()
        assert not responder.is_done()
        assert perception.reset_calls == 1

        engine.response = NumericalResponse(value=7, rationale="after reset")
        out = responder.respond(_snapshot(tick_id=1, question_text="How many", pose=_pose(0, 0)))
        assert isinstance(out, NumericalResponse)
        assert out.value == 7
