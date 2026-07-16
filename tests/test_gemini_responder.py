"""Tests for :class:`xiao_hei_vln.gemini.responder.GeminiResponder`.

Use a ``FakeGeminiEngine`` injected at construction — no real SDK or
API key needed. The wrapped ``PerceptionResponder`` is real but uses a
synthetic ``GlobalMap``, so we can drive Phase A frontier exhaustion
deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from xiao_hei_vln.gemini.config import GeminiConfig
from xiao_hei_vln.gemini.responder import GeminiResponder
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
    question_text: str,
    pose: OdomPose | None = None,
    with_terrain: bool = True,
) -> VLMInput:
    return VLMInput(
        tick_id=tick_id,
        tick_time=_stamp(tick_id * 0.5),
        question=ChallengeQuestion.from_text(question_text, _stamp()),
        pose=pose,
        terrain_ext=_terrain() if with_terrain else None,
    )


@dataclass
class FakeGeminiEngine:
    """Returns canned responses; records the calls it received."""

    response: object | None = None
    calls: list[dict] = field(default_factory=list)
    warmed_up: bool = False

    def infer_multimodal(
        self,
        *,
        system: str,
        user_text: str,
        images: list[bytes],
    ) -> object:
        self.calls.append(
            {"system": system, "user_text": user_text, "images": list(images)},
        )
        return self.response

    def warmup(self) -> None:
        self.warmed_up = True


class FakeExploringPerception:
    """A perception responder that always returns a non-exhausted waypoint.

    Lets us drive the GeminiResponder's `max_explore_ticks` trigger in
    isolation, without `phase_a_exhausted` ending exploration early.
    The real PerceptionResponder is unit-tested separately.
    """

    def __init__(self) -> None:
        self.respond_calls = 0
        self.reset_calls = 0

    def respond(self, snapshot: VLMInput):
        self.respond_calls += 1
        if snapshot.pose is None:
            return None
        return WaypointPathResponse(
            waypoints=[Waypoint(x=snapshot.pose.position.x + 1.0,
                                y=snapshot.pose.position.y, heading=0.0)],
            rationale="frontier_target",
        )

    def is_done(self) -> bool:
        return False

    def reset(self) -> None:
        self.reset_calls += 1

    def close(self) -> None:
        pass


def _config(**overrides) -> GeminiConfig:
    base = dict(api_key="test", max_explore_ticks=3, max_ticks_per_question=10)
    base.update(overrides)
    return GeminiConfig(**base)


# ===========================================================================
# Task 1 — numerical
# ===========================================================================


class TestTask1Numerical:
    def test_explores_until_max_explore_ticks_then_commits(self) -> None:
        engine = FakeGeminiEngine(
            response=NumericalResponse(value=4, rationale="counted 4"),
        )
        responder = GeminiResponder(
            engine, _config(max_explore_ticks=3),
            perception=FakeExploringPerception(),
        )

        # Tick 1, 2, 3 — explore (no Gemini yet)
        for t in range(1, 4):
            out = responder.respond(
                _snapshot(tick_id=t, question_text="How many cups", pose=_pose(t * 0.1, 0)),
            )
            # Frontier responder may return WaypointPathResponse or None.
            # Critical: no Gemini call yet, not done.
            assert not engine.calls
            assert not responder.is_done()

        # Tick 4 — past max_explore_ticks, Gemini fires + commits.
        out = responder.respond(
            _snapshot(tick_id=4, question_text="How many cups", pose=_pose(0.4, 0)),
        )
        assert isinstance(out, NumericalResponse)
        assert out.value == 4
        assert len(engine.calls) == 1
        assert responder.is_done()
        # System prompt is the numerical one.
        assert '"kind": "numerical"' in engine.calls[0]["system"]

    def test_subsequent_ticks_return_committed_answer(self) -> None:
        engine = FakeGeminiEngine(
            response=NumericalResponse(value=2, rationale="ok"),
        )
        responder = GeminiResponder(
            engine, _config(max_explore_ticks=1),
            perception=FakeExploringPerception(),
        )

        # Tick 1 explores; tick 2 triggers (past max_explore_ticks); tick 3 should
        # just re-emit the committed answer.
        responder.respond(_snapshot(tick_id=1, question_text="How many", pose=_pose(0, 0)))
        out2 = responder.respond(_snapshot(tick_id=2, question_text="How many", pose=_pose(0, 0)))
        assert isinstance(out2, NumericalResponse)
        # After is_done is True, respond() returns None per the dummy/qwen contract.
        out3 = responder.respond(_snapshot(tick_id=3, question_text="How many", pose=_pose(0, 0)))
        assert out3 is None

    def test_no_question_returns_none(self) -> None:
        engine = FakeGeminiEngine()
        responder = GeminiResponder(engine, _config())
        out = responder.respond(
            VLMInput(tick_id=0, tick_time=_stamp(), question=None, pose=_pose(0, 0)),
        )
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
        responder = GeminiResponder(
            engine, _config(max_explore_ticks=1),
            perception=FakeExploringPerception(),
        )
        # Tick 1 explores; tick 2 trips trigger and commits.
        responder.respond(
            _snapshot(tick_id=1, question_text="Find the sofa", pose=_pose(0, 0)),
        )
        out = responder.respond(
            _snapshot(tick_id=2, question_text="Find the sofa", pose=_pose(0.5, 0)),
        )
        assert isinstance(out, ObjectReferenceResponse)
        assert out.label == "sofa"
        # System prompt is the object-reference one.
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
        responder = GeminiResponder(engine, _config())

        # Tick 1 — plan + emit first waypoint
        out1 = responder.respond(
            _snapshot(tick_id=1, question_text="Go forward", pose=_pose(0, 0)),
        )
        assert isinstance(out1, WaypointPathResponse)
        assert out1.waypoints[0].x == pytest.approx(1.0)
        assert len(engine.calls) == 1
        assert '"kind": "waypoint_path"' in engine.calls[0]["system"]

        # Tick 2 — robot still far from wp0 → keep emitting wp0
        out2 = responder.respond(
            _snapshot(tick_id=2, question_text="Go forward", pose=_pose(0.2, 0)),
        )
        assert out2.waypoints[0].x == pytest.approx(1.0)
        assert len(engine.calls) == 1  # no second Gemini call

        # Tick 3 — robot reached wp0 → advance to wp1
        out3 = responder.respond(
            _snapshot(tick_id=3, question_text="Go forward", pose=_pose(1.0, 0)),
        )
        assert out3.waypoints[0].x == pytest.approx(2.0)
        assert not responder.is_done()

        # Tick 4 — reach wp1 → advance to wp2
        out4 = responder.respond(
            _snapshot(tick_id=4, question_text="Go forward", pose=_pose(2.0, 0)),
        )
        assert out4.waypoints[0].x == pytest.approx(3.0)
        assert not responder.is_done()

        # Tick 5 — reach final wp2 → done
        out5 = responder.respond(
            _snapshot(tick_id=5, question_text="Go forward", pose=_pose(3.0, 0)),
        )
        assert out5.waypoints[0].x == pytest.approx(3.0)
        assert responder.is_done()

    def test_gemini_returning_non_waypoint_for_task2_marks_done(self) -> None:
        engine = FakeGeminiEngine(
            response=NumericalResponse(value=0, rationale="oops"),
        )
        responder = GeminiResponder(engine, _config())
        out = responder.respond(
            _snapshot(tick_id=1, question_text="Go to the kitchen", pose=_pose(0, 0)),
        )
        assert out is None
        assert responder.is_done()


# ===========================================================================
# Lifecycle
# ===========================================================================


@dataclass
class FakeLogger:
    """Captures the VLMLogger calls GeminiResponder makes."""

    new_questions: list[str] = field(default_factory=list)
    ticks: list[dict] = field(default_factory=list)
    closed: bool = False

    def new_question(self, text: str) -> None:
        self.new_questions.append(text)

    def log_tick(
        self,
        snapshot,
        system_prompt: str,
        user_text: str,
        output,
        inference_ms: float,
        evidence,
    ) -> None:
        self.ticks.append({
            "tick_id": snapshot.tick_id,
            "system_prompt": system_prompt,
            "user_text": user_text,
            "output": output,
            "inference_ms": inference_ms,
        })

    def close(self) -> None:
        self.closed = True


class TestLoggerIntegration:
    def test_logger_receives_new_question_log_tick_close(self) -> None:
        engine = FakeGeminiEngine(
            response=NumericalResponse(value=4, rationale="ok"),
        )
        logger = FakeLogger()
        responder = GeminiResponder(
            engine,
            _config(max_explore_ticks=2),
            perception=FakeExploringPerception(),
            logger=logger,
        )

        # Tick 1, 2 — explore. Tick 3 — past max_explore_ticks, Gemini fires.
        for t in range(1, 4):
            responder.respond(_snapshot(tick_id=t, question_text="How many", pose=_pose(t * 0.1, 0)))

        assert logger.new_questions == ["How many"]
        # One log_tick per tick — including the two explore ticks.
        assert len(logger.ticks) == 3
        # First two log_tick calls came from explore-phase (empty system).
        assert logger.ticks[0]["system_prompt"] == ""
        assert logger.ticks[1]["system_prompt"] == ""
        # Third was the real Gemini call.
        assert '"kind": "numerical"' in logger.ticks[2]["system_prompt"]
        assert logger.ticks[2]["output"] is not None

        responder.close()
        assert logger.closed is True


class TestEngineFailureFallback:
    def test_task1_gemini_raises_then_responder_keeps_exploring(self) -> None:
        class FlakyEngine:
            def __init__(self) -> None:
                self.calls = 0

            def infer_multimodal(self, **_kwargs):
                self.calls += 1
                raise RuntimeError("network blip")

            def warmup(self) -> None:
                pass

        engine = FlakyEngine()
        responder = GeminiResponder(
            engine,
            _config(max_explore_ticks=1),
            perception=FakeExploringPerception(),
        )

        # Tick 1 explores; tick 2 trips the trigger but engine raises —
        # responder should fall back to a perception waypoint and not commit.
        responder.respond(_snapshot(tick_id=1, question_text="How many cups", pose=_pose(0, 0)))
        out = responder.respond(_snapshot(tick_id=2, question_text="How many cups", pose=_pose(0.1, 0)))
        assert engine.calls == 1
        assert not responder.is_done()
        # Falls back to a perception waypoint.
        assert isinstance(out, WaypointPathResponse)


class TestLifecycle:
    def test_reset_clears_state(self) -> None:
        engine = FakeGeminiEngine(
            response=NumericalResponse(value=1, rationale="ok"),
        )
        responder = GeminiResponder(
            engine, _config(max_explore_ticks=1),
            perception=FakeExploringPerception(),
        )

        # Drive to committed state on Q1.
        responder.respond(_snapshot(tick_id=1, question_text="How many", pose=_pose(0, 0)))
        responder.respond(_snapshot(tick_id=2, question_text="How many", pose=_pose(0.1, 0)))
        assert responder.is_done()

        responder.reset()
        assert not responder.is_done()
        # New question — should respond again without falling through.
        engine.response = NumericalResponse(value=7, rationale="after reset")
        responder.respond(_snapshot(tick_id=1, question_text="How many", pose=_pose(0, 0)))
        out = responder.respond(_snapshot(tick_id=2, question_text="How many", pose=_pose(0.1, 0)))
        assert isinstance(out, NumericalResponse)
        assert out.value == 7
