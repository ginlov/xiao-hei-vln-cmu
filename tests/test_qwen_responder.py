"""Tests for `QwenResponder` driven by a scripted in-memory engine.

vLLM / PIL / CUDA are NOT imported here — the responder talks to an
`EngineProtocol`-shaped fake, so the loop semantics are exercisable on
any platform.
"""

from __future__ import annotations

import logging

import pytest

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
    VLMOutput,
    Waypoint,
    WaypointPathResponse,
)
from xiao_hei_vln.messages.sensors import ImageFrame
from xiao_hei_vln.qwen import QwenConfig, QwenResponder

# --- fixtures --------------------------------------------------------------


def _stamp(t: float = 0.0) -> Stamp:
    return Stamp.from_seconds(t)


def _pose(x: float, y: float) -> OdomPose:
    return OdomPose(
        header=Header(stamp=_stamp(), frame_id="map"),
        position=Vector3(x=x, y=y, z=0.75),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )


def _input(question_text: str | None, pose: OdomPose | None = None) -> VLMInput:
    q = (
        ChallengeQuestion.from_text(question_text, _stamp())
        if question_text is not None
        else None
    )
    return VLMInput(tick_id=0, tick_time=_stamp(), question=q, pose=pose)


class ScriptedEngine:
    """Returns scripted outputs in order; records every call for assertions."""

    def __init__(self, outputs: list[VLMOutput]) -> None:
        self._outputs = list(outputs)
        self.calls: list[dict] = []

    def infer(
        self,
        system: str,
        user_text: str,
        image: ImageFrame | None,
    ) -> VLMOutput:
        self.calls.append({"system": system, "user_text": user_text, "image": image})
        if not self._outputs:
            raise AssertionError("ScriptedEngine ran out of outputs")
        return self._outputs.pop(0)


class FailingEngine:
    def __init__(self) -> None:
        self.call_count = 0

    def infer(self, system: str, user_text: str, image: ImageFrame | None) -> VLMOutput:
        self.call_count += 1
        raise RuntimeError("engine boom")


# --- terminal outputs ------------------------------------------------------


class TestTerminalNumerical:
    def test_terminal_numerical_sets_done_and_returns_value(self) -> None:
        engine = ScriptedEngine([NumericalResponse(value=4)])
        r = QwenResponder(engine)
        out = r.respond(_input("How many cups"))
        assert isinstance(out, NumericalResponse)
        assert out.value == 4
        assert r.is_done() is True

    def test_no_response_when_done(self) -> None:
        engine = ScriptedEngine([NumericalResponse(value=2)])
        r = QwenResponder(engine)
        assert isinstance(r.respond(_input("How many books")), NumericalResponse)
        # Second tick: still has same question (sticky) but we're done.
        assert r.respond(_input("How many books")) is None


class TestTerminalObjectReference:
    def test_object_reference_sets_done(self) -> None:
        marker = ObjectReferenceResponse(
            label="sofa",
            object_id=0,
            center=Vector3(x=3.37, y=-2.09, z=0.50),
            size=Vector3(x=2.86, y=1.2, z=1.02),
        )
        engine = ScriptedEngine([marker])
        r = QwenResponder(engine)
        out = r.respond(_input("Find the red cup"))
        assert isinstance(out, ObjectReferenceResponse)
        assert out.label == "sofa"
        assert r.is_done() is True


# --- non-terminal: waypoint navigation -------------------------------------


class TestNumericalWithNavigation:
    def test_waypoint_response_keeps_loop_alive(self) -> None:
        engine = ScriptedEngine(
            [
                WaypointPathResponse(waypoints=[Waypoint(x=1.0, y=2.0)]),
                WaypointPathResponse(waypoints=[Waypoint(x=3.0, y=4.0)]),
                NumericalResponse(value=7),
            ],
        )
        r = QwenResponder(engine)
        q = _input("How many books on the shelf", pose=_pose(0.0, 0.0))

        first = r.respond(q)
        assert isinstance(first, WaypointPathResponse)
        assert r.is_done() is False

        second = r.respond(q)
        assert isinstance(second, WaypointPathResponse)
        assert r.is_done() is False

        third = r.respond(q)
        assert isinstance(third, NumericalResponse)
        assert third.value == 7
        assert r.is_done() is True

    def test_evidence_log_grows_across_ticks(self) -> None:
        engine = ScriptedEngine(
            [
                WaypointPathResponse(waypoints=[Waypoint(x=1.0, y=2.0)]),
                WaypointPathResponse(waypoints=[Waypoint(x=3.0, y=4.0)]),
                NumericalResponse(value=3),
            ],
        )
        r = QwenResponder(engine)
        q = _input("How many cups", pose=_pose(0.0, 0.0))

        r.respond(q)
        # Second tick prompt should reference the first navigation.
        r.respond(q)
        assert "navigated toward waypoint (1.00,2.00)" in engine.calls[1]["user_text"]
        # Third tick should reference both prior navigations.
        r.respond(q)
        assert "navigated toward waypoint (1.00,2.00)" in engine.calls[2]["user_text"]
        assert "navigated toward waypoint (3.00,4.00)" in engine.calls[2]["user_text"]


class TestInstructionFollowing:
    def test_waypoint_path_is_terminal_for_instruction_following(self) -> None:
        engine = ScriptedEngine(
            [WaypointPathResponse(waypoints=[Waypoint(x=7.5, y=-1.0)])],
        )
        r = QwenResponder(engine)
        out = r.respond(_input("Take the path near the window", pose=_pose(0.0, 0.0)))
        assert isinstance(out, WaypointPathResponse)
        # Distinct from the numerical case: instruction-following is done after
        # one path emission.
        assert r.is_done() is True


# --- lifecycle / safety --------------------------------------------------


class TestQuiescent:
    def test_no_question_no_call(self) -> None:
        engine = ScriptedEngine([])
        r = QwenResponder(engine)
        assert r.respond(_input(None)) is None
        assert engine.calls == []
        assert r.is_done() is False


class TestEngineFailure:
    def test_engine_exception_skips_tick(self, caplog: pytest.LogCaptureFixture) -> None:
        engine = FailingEngine()
        r = QwenResponder(engine)
        with caplog.at_level(logging.ERROR):
            out = r.respond(_input("How many cups"))
        assert out is None
        # Not marked done — next tick gets another chance.
        assert r.is_done() is False
        assert engine.call_count == 1
        assert any("inference failed" in rec.message for rec in caplog.records)


class TestTimeout:
    def test_numerical_timeout_emits_zero(self, caplog: pytest.LogCaptureFixture) -> None:
        # Always keep navigating, never converge → trip the cap.
        config = QwenConfig(max_ticks_per_question=3)
        engine = ScriptedEngine(
            [WaypointPathResponse(waypoints=[Waypoint(x=float(i), y=0.0)]) for i in range(3)],
        )
        r = QwenResponder(engine, config)
        q = _input("How many cups", pose=_pose(0.0, 0.0))

        for _ in range(3):
            assert isinstance(r.respond(q), WaypointPathResponse)
            assert r.is_done() is False

        with caplog.at_level(logging.WARNING):
            timeout_out = r.respond(q)
        assert isinstance(timeout_out, NumericalResponse)
        assert timeout_out.value == 0
        assert r.is_done() is True
        assert any("timed out" in rec.message for rec in caplog.records)

    def test_non_numerical_timeout_returns_none(self) -> None:
        config = QwenConfig(max_ticks_per_question=2)
        engine = ScriptedEngine(
            [
                WaypointPathResponse(waypoints=[Waypoint(x=1.0, y=0.0)]),
                WaypointPathResponse(waypoints=[Waypoint(x=2.0, y=0.0)]),
            ],
        )
        # Object-reference question loop that never commits to a marker.
        r = QwenResponder(engine, config)
        q = _input("Find the red cup", pose=_pose(0.0, 0.0))
        assert isinstance(r.respond(q), WaypointPathResponse)
        assert isinstance(r.respond(q), WaypointPathResponse)
        assert r.respond(q) is None
        assert r.is_done() is True


class TestReset:
    def test_reset_clears_state(self) -> None:
        engine = ScriptedEngine(
            [
                WaypointPathResponse(waypoints=[Waypoint(x=1.0, y=2.0)]),
                NumericalResponse(value=5),
                NumericalResponse(value=9),
            ],
        )
        r = QwenResponder(engine)
        q = _input("How many cups", pose=_pose(0.0, 0.0))

        r.respond(q)
        r.respond(q)
        assert r.is_done() is True

        r.reset()
        assert r.is_done() is False
        # Evidence should be cleared: the numerical prompt should flag this as
        # the FIRST tick again.
        r.respond(q)
        assert "FIRST tick on this question" in engine.calls[2]["user_text"]
