"""Tests for the Python port of dummyVLM.cpp."""

from __future__ import annotations

import random

import pytest

from xiao_hei_vln.dummy import DummyResponder
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


def _make_responder(**kwargs) -> DummyResponder:
    return DummyResponder(
        object_fixture=ObjectReferenceResponse(
            label="sofa",
            object_id=0,
            center=Vector3(x=3.37, y=-2.09, z=0.50),
            size=Vector3(x=2.86, y=1.2, z=1.02),
            heading=0.0,
        ),
        waypoint_fixture=[Waypoint(x=7.5, y=-1.0), Waypoint(x=5.0, y=-4.0)],
        **kwargs,
    )


class TestQuiescent:
    def test_no_question_no_response(self) -> None:
        r = _make_responder()
        assert r.respond(_input(None)) is None
        assert r.is_done() is False

    def test_no_response_when_done(self) -> None:
        r = _make_responder(rng=random.Random(0))
        first = r.respond(_input("How many cups"))
        assert isinstance(first, NumericalResponse)
        assert r.is_done() is True
        assert r.respond(_input("How many cups")) is None


class TestNumerical:
    def test_value_in_1_to_10(self) -> None:
        rng = random.Random(42)
        r = _make_responder(rng=rng)
        out = r.respond(_input("How many books are on the sofa"))
        assert isinstance(out, NumericalResponse)
        assert 1 <= out.value <= 10
        assert r.is_done() is True


class TestObjectReference:
    def test_returns_loaded_object(self) -> None:
        r = _make_responder()
        out = r.respond(_input("Find the red cup"))
        assert isinstance(out, ObjectReferenceResponse)
        assert out.label == "sofa"
        assert out.object_id == 0
        assert r.is_done() is True


class TestInstructionFollowing:
    def test_emits_first_waypoint_when_far(self) -> None:
        r = _make_responder()
        out = r.respond(_input("Take the path", pose=_pose(0.0, 0.0)))
        assert isinstance(out, WaypointPathResponse)
        assert len(out.waypoints) == 1
        assert out.waypoints[0].x == pytest.approx(7.5)
        assert r.is_done() is False

    def test_advances_when_within_reach(self) -> None:
        r = _make_responder()
        # Tick 1: far from waypoint 0 → emit waypoint 0, no advance.
        first = r.respond(_input("Go", pose=_pose(0.0, 0.0)))
        assert isinstance(first, WaypointPathResponse)
        assert first.waypoints[0].x == pytest.approx(7.5)

        # Tick 2: vehicle now near waypoint 0 → advance and emit waypoint 1.
        second = r.respond(_input("Go", pose=_pose(7.5, -1.0)))
        assert isinstance(second, WaypointPathResponse)
        assert second.waypoints[0].x == pytest.approx(5.0)
        assert r.is_done() is False

    def test_marks_done_after_final_waypoint(self) -> None:
        r = _make_responder()
        # Drive close enough to both waypoints in succession.
        r.respond(_input("Go", pose=_pose(7.5, -1.0)))  # advance to wp 1
        out = r.respond(_input("Go", pose=_pose(5.0, -4.0)))  # at last wp
        assert isinstance(out, WaypointPathResponse)
        assert r.is_done() is True

    def test_no_pose_means_no_advance(self) -> None:
        r = _make_responder()
        first = r.respond(_input("Go", pose=None))
        assert isinstance(first, WaypointPathResponse)
        assert first.waypoints[0].x == pytest.approx(7.5)
        second = r.respond(_input("Go", pose=None))
        assert second.waypoints[0].x == pytest.approx(7.5)  # still on wp 0


class TestReset:
    def test_reset_clears_done_and_index(self) -> None:
        r = _make_responder()
        r.respond(_input("Take the path", pose=_pose(7.5, -1.0)))
        r.respond(_input("Take the path", pose=_pose(5.0, -4.0)))
        assert r.is_done() is True
        r.reset()
        assert r.is_done() is False
        out = r.respond(_input("Take the path", pose=_pose(0.0, 0.0)))
        assert isinstance(out, WaypointPathResponse)
        assert out.waypoints[0].x == pytest.approx(7.5)
