"""Tests for the Phase-2 baseline prompt builders.

Pure-python; no vLLM/PIL needed.
"""

from __future__ import annotations

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
from xiao_hei_vln.qwen.prompts import build_system_prompt, build_user_message


def _stamp() -> Stamp:
    return Stamp.from_seconds(0.0)


def _pose(x: float, y: float) -> OdomPose:
    return OdomPose(
        header=Header(stamp=_stamp(), frame_id="map"),
        position=Vector3(x=x, y=y, z=0.75),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )


def _input(text: str, pose: OdomPose | None = None) -> tuple[VLMInput, ChallengeQuestion]:
    q = ChallengeQuestion.from_text(text, _stamp())
    return VLMInput(tick_id=0, tick_time=_stamp(), question=q, pose=pose), q


def test_system_prompt_documents_all_three_response_shapes() -> None:
    sys = build_system_prompt()
    assert '"kind": "numerical"' in sys
    assert '"kind": "object_reference"' in sys
    assert '"kind": "waypoint_path"' in sys


def test_user_message_includes_question_text_and_type() -> None:
    snapshot, q = _input("How many cups", pose=_pose(1.0, 2.0))
    msg = build_user_message(snapshot, q, evidence_log=[])
    assert "How many cups" in msg
    assert "type=numerical" in msg
    assert q.type is QuestionType.NUMERICAL


def test_user_message_renders_pose_summary() -> None:
    snapshot, q = _input("How many cups", pose=_pose(1.23, -4.56))
    msg = build_user_message(snapshot, q, evidence_log=[])
    assert "Pose: x=1.23 y=-4.56" in msg


def test_user_message_handles_missing_pose() -> None:
    snapshot, q = _input("How many cups", pose=None)
    msg = build_user_message(snapshot, q, evidence_log=[])
    assert "Pose: unknown" in msg


def test_user_message_lists_evidence_when_present() -> None:
    snapshot, q = _input("How many cups", pose=_pose(0.0, 0.0))
    msg = build_user_message(
        snapshot,
        q,
        evidence_log=["navigated toward waypoint (1.00,2.00) [1 in path]"],
    )
    assert "Evidence so far" in msg
    assert "1. navigated toward waypoint (1.00,2.00)" in msg


def test_user_message_signals_first_tick_when_no_evidence() -> None:
    snapshot, q = _input("How many cups", pose=_pose(0.0, 0.0))
    msg = build_user_message(snapshot, q, evidence_log=[])
    assert "first tick on this question" in msg


def test_user_message_hint_is_question_type_specific() -> None:
    snapshot, q_num = _input("How many cups")
    snapshot_obj, q_obj = _input("Find the red cup")
    snapshot_instr, q_instr = _input("Take the path near the window")

    assert "NUMERICAL question" in build_user_message(snapshot, q_num, [])
    assert "OBJECT REFERENCE question" in build_user_message(snapshot_obj, q_obj, [])
    assert "INSTRUCTION FOLLOWING question" in build_user_message(
        snapshot_instr, q_instr, [],
    )
