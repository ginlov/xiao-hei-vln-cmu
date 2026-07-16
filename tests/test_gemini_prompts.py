"""Tests for ``xiao_hei_vln.gemini.prompts``."""

from __future__ import annotations

import numpy as np
import pytest

from xiao_hei_vln.gemini.prompts import (
    SYSTEM_PROMPT_NUMERICAL,
    SYSTEM_PROMPT_OBJECT_REFERENCE,
    SYSTEM_PROMPT_ROUTE,
    build_system_prompt,
    build_user_message,
)
from xiao_hei_vln.messages import (
    ChallengeQuestion,
    Header,
    OdomPose,
    Quaternion,
    QuestionType,
    Stamp,
    Vector3,
    VLMInput,
)
from xiao_hei_vln.messages.sensors import TerrainMap


def _stamp() -> Stamp:
    return Stamp.from_seconds(0.0)


def _pose(x: float, y: float) -> OdomPose:
    return OdomPose(
        header=Header(stamp=_stamp(), frame_id="map"),
        position=Vector3(x=x, y=y, z=0.75),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )


def _input(question_text: str, pose: OdomPose | None = None) -> VLMInput:
    return VLMInput(
        tick_id=0,
        tick_time=_stamp(),
        question=ChallengeQuestion.from_text(question_text, _stamp()),
        pose=pose,
    )


class TestSystemPrompts:
    def test_numerical_prompt_specifies_json_shape(self) -> None:
        assert '"kind": "numerical"' in SYSTEM_PROMPT_NUMERICAL
        assert '"value"' in SYSTEM_PROMPT_NUMERICAL
        assert "non-negative integer" in SYSTEM_PROMPT_NUMERICAL

    def test_object_reference_prompt_specifies_json_shape(self) -> None:
        assert '"kind": "object_reference"' in SYSTEM_PROMPT_OBJECT_REFERENCE
        assert '"label"' in SYSTEM_PROMPT_OBJECT_REFERENCE
        assert '"center"' in SYSTEM_PROMPT_OBJECT_REFERENCE
        assert '"size"' in SYSTEM_PROMPT_OBJECT_REFERENCE
        assert '"heading"' in SYSTEM_PROMPT_OBJECT_REFERENCE

    def test_route_prompt_specifies_json_shape(self) -> None:
        assert '"kind": "waypoint_path"' in SYSTEM_PROMPT_ROUTE
        assert '"waypoints"' in SYSTEM_PROMPT_ROUTE
        assert "at least one" in SYSTEM_PROMPT_ROUTE

    def test_build_system_prompt_by_type(self) -> None:
        assert build_system_prompt(QuestionType.NUMERICAL) == SYSTEM_PROMPT_NUMERICAL
        assert build_system_prompt(QuestionType.OBJECT_REFERENCE) == SYSTEM_PROMPT_OBJECT_REFERENCE
        assert (
            build_system_prompt(QuestionType.INSTRUCTION_FOLLOWING) == SYSTEM_PROMPT_ROUTE
        )


class TestUserMessage:
    def test_includes_question_text_and_type(self) -> None:
        snap = _input("How many sofas in the room?", pose=_pose(1.0, 2.0))
        msg = build_user_message(snap, snap.question, trajectory_xy=[(0.0, 0.0)])
        assert "How many sofas" in msg
        assert "type=numerical" in msg

    def test_pose_summary(self) -> None:
        snap = _input("Find the lamp", pose=_pose(1.23, 4.56))
        msg = build_user_message(snap, snap.question, trajectory_xy=None)
        assert "x=1.23" in msg
        assert "y=4.56" in msg

    def test_no_pose(self) -> None:
        snap = _input("How many", pose=None)
        msg = build_user_message(snap, snap.question, trajectory_xy=None)
        assert "Pose: unknown" in msg

    def test_trajectory_truncates_when_long(self) -> None:
        traj = [(float(i), float(i)) for i in range(20)]
        snap = _input("How many", pose=_pose(0, 0))
        msg = build_user_message(snap, snap.question, trajectory_xy=traj)
        # First two points, ellipsis, last four points
        assert "(0.0,0.0)" in msg
        assert "(19.0,19.0)" in msg
        assert "omitted" in msg

    def test_terrain_summary_shown(self) -> None:
        snap = VLMInput(
            tick_id=0,
            tick_time=_stamp(),
            question=ChallengeQuestion.from_text("How many cups", _stamp()),
            pose=_pose(0, 0),
            terrain_ext=TerrainMap(
                header=Header(stamp=_stamp(), frame_id="map"),
                points=np.zeros((42, 4), dtype=np.float32),
                range="ext_20m",
            ),
        )
        msg = build_user_message(snap, snap.question)
        assert "terrain_ext: 42 pts" in msg

    def test_response_hint_per_type(self) -> None:
        snap_n = _input("How many sofas")
        snap_o = _input("Find the lamp")
        snap_i = _input("Go to the kitchen")
        msg_n = build_user_message(snap_n, snap_n.question)
        msg_o = build_user_message(snap_o, snap_o.question)
        msg_i = build_user_message(snap_i, snap_i.question)
        assert "NUMERICAL" in msg_n
        assert "OBJECT-REFERENCE" in msg_o
        assert "INSTRUCTION-FOLLOWING" in msg_i

    def test_exploration_summary_appended(self) -> None:
        snap = _input("How many", pose=_pose(0, 0))
        msg = build_user_message(
            snap,
            snap.question,
            trajectory_xy=[(0.0, 0.0)],
            exploration_summary="Phase A frontier exhausted",
        )
        assert "Phase A frontier exhausted" in msg
