"""Tests for ``VLMLogger`` — session init, per-question dirs, JSONL, JPEG."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from xiao_hei_vln.messages import (
    ChallengeQuestion,
    Header,
    NumericalResponse,
    OdomPose,
    Quaternion,
    Stamp,
    Vector3,
    VLMInput,
    Waypoint,
    WaypointPathResponse,
)
from xiao_hei_vln.messages.sensors import ImageFrame
from xiao_hei_vln.qwen.config import QwenConfig
from xiao_hei_vln.qwen.logger import VLMLogger


def _stamp(t: float = 0.0) -> Stamp:
    return Stamp.from_seconds(t)


def _pose() -> OdomPose:
    return OdomPose(
        header=Header(stamp=_stamp(), frame_id="map"),
        position=Vector3(x=1.0, y=2.0, z=0.5),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )


def _image_frame(width: int = 4, height: int = 2) -> ImageFrame:
    data = np.zeros((height, width, 3), dtype=np.uint8)
    data[:, :, 0] = 255
    return ImageFrame(
        header=Header(stamp=_stamp(), frame_id="camera"),
        width=width,
        height=height,
        step=width * 3,
        data=data.tobytes(),
    )


def _snapshot(
    tick_id: int = 0,
    *,
    with_image: bool = False,
    question_text: str = "How many chairs",
) -> VLMInput:
    return VLMInput(
        tick_id=tick_id,
        tick_time=_stamp(tick_id * 0.5),
        question=ChallengeQuestion.from_text(question_text, _stamp()),
        pose=_pose(),
        image=_image_frame() if with_image else None,
    )


@pytest.fixture()
def logger(tmp_path: Path) -> VLMLogger:
    cfg = QwenConfig()
    lg = VLMLogger(tmp_path, config=cfg, responder_name="qwen", tick_hz=2.0)
    yield lg
    lg.close()


# --- session-level tests ---


def test_session_json_written(logger: VLMLogger) -> None:
    session_json = logger.session_dir / "session.json"
    assert session_json.exists()
    data = json.loads(session_json.read_text())
    assert data["responder"] == "qwen"
    assert data["tick_hz"] == 2.0
    assert "config" in data
    assert data["config"]["model"] == "/models/Qwen3.5-4B"
    assert "start_time" in data


# --- per-question tests ---


def test_new_question_creates_subdir(logger: VLMLogger) -> None:
    logger.new_question("How many chairs")
    dirs = sorted(d.name for d in logger.session_dir.iterdir() if d.is_dir())
    assert "q_001_how_many_chairs" in dirs


def test_tick_appended_to_question_jsonl(logger: VLMLogger) -> None:
    logger.new_question("How many chairs")
    snap = _snapshot(tick_id=1)
    output = NumericalResponse(value=3, rationale="counted 3")
    logger.log_tick(snap, "system", "user text", output, 42.5, ["prev"])

    q_dir = logger.session_dir / "q_001_how_many_chairs"
    lines = (q_dir / "ticks.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["tick_id"] == 1
    assert record["question_text"] == "How many chairs"
    assert record["question_type"] == "numerical"
    assert record["output"]["kind"] == "numerical"
    assert record["output"]["value"] == 3
    assert record["inference_ms"] == 42.5
    assert record["evidence"] == ["prev"]
    assert record["image_path"] is None


def test_image_saved_in_question_dir(logger: VLMLogger) -> None:
    logger.new_question("How many cups")
    snap = _snapshot(tick_id=5, with_image=True, question_text="How many cups")
    output = NumericalResponse(value=1)
    logger.log_tick(snap, "sys", "usr", output, 10.0, [])

    img_path = (
        logger.session_dir / "q_001_how_many_cups" / "images" / "tick_000005.jpg"
    )
    assert img_path.exists()
    assert img_path.stat().st_size > 0

    q_dir = logger.session_dir / "q_001_how_many_cups"
    record = json.loads(
        (q_dir / "ticks.jsonl").read_text().strip(),
    )
    assert record["image_path"] == "images/tick_000005.jpg"


def test_no_image_when_none(logger: VLMLogger) -> None:
    logger.new_question("How many chairs")
    snap = _snapshot(tick_id=3, with_image=False)
    output = NumericalResponse(value=0)
    logger.log_tick(snap, "sys", "usr", output, 5.0, [])

    q_dir = logger.session_dir / "q_001_how_many_chairs"
    images = list((q_dir / "images").iterdir())
    assert images == []

    record = json.loads(
        (q_dir / "ticks.jsonl").read_text().strip(),
    )
    assert record["image_path"] is None


def test_multiple_ticks_in_one_question(logger: VLMLogger) -> None:
    logger.new_question("How many chairs")
    for i in range(3):
        snap = _snapshot(tick_id=i)
        output = WaypointPathResponse(
            waypoints=[Waypoint(x=float(i), y=0.0)],
            rationale=f"exploring tick {i}",
        )
        logger.log_tick(snap, "sys", "usr", output, float(i * 10), [])

    q_dir = logger.session_dir / "q_001_how_many_chairs"
    lines = (q_dir / "ticks.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3
    ids = [json.loads(line)["tick_id"] for line in lines]
    assert ids == [0, 1, 2]


def test_multiple_questions_separate_dirs(logger: VLMLogger) -> None:
    logger.new_question("How many chairs")
    logger.log_tick(
        _snapshot(tick_id=0), "s", "u", NumericalResponse(value=2), 10.0, [],
    )

    logger.new_question("Find the red cup")
    logger.log_tick(
        _snapshot(tick_id=1, question_text="Find the red cup"),
        "s", "u", NumericalResponse(value=0), 5.0, [],
    )

    q1 = logger.session_dir / "q_001_how_many_chairs"
    q2 = logger.session_dir / "q_002_find_the_red_cup"
    assert q1.exists()
    assert q2.exists()
    assert len((q1 / "ticks.jsonl").read_text().strip().splitlines()) == 1
    assert len((q2 / "ticks.jsonl").read_text().strip().splitlines()) == 1


def test_output_none_logged(logger: VLMLogger) -> None:
    logger.new_question("How many chairs")
    snap = _snapshot(tick_id=7)
    logger.log_tick(snap, "sys", "usr", None, 0.0, [])

    q_dir = logger.session_dir / "q_001_how_many_chairs"
    record = json.loads(
        (q_dir / "ticks.jsonl").read_text().strip(),
    )
    assert record["output"] is None
    assert record["tick_id"] == 7


def test_pose_serialized(logger: VLMLogger) -> None:
    logger.new_question("How many chairs")
    logger.log_tick(
        _snapshot(tick_id=0), "s", "u", NumericalResponse(value=0), 1.0, [],
    )

    q_dir = logger.session_dir / "q_001_how_many_chairs"
    record = json.loads(
        (q_dir / "ticks.jsonl").read_text().strip(),
    )
    assert record["pose"]["position"] == {"x": 1.0, "y": 2.0, "z": 0.5}
    assert record["pose"]["orientation"] == {
        "x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0,
    }


def test_no_pose_serialized_as_null(logger: VLMLogger) -> None:
    logger.new_question("How many cups")
    snap = VLMInput(
        tick_id=0,
        tick_time=_stamp(),
        question=ChallengeQuestion.from_text("How many cups", _stamp()),
        pose=None,
    )
    logger.log_tick(snap, "s", "u", NumericalResponse(value=0), 1.0, [])

    q_dir = logger.session_dir / "q_001_how_many_cups"
    record = json.loads(
        (q_dir / "ticks.jsonl").read_text().strip(),
    )
    assert record["pose"] is None


def test_log_tick_without_new_question_is_noop(logger: VLMLogger) -> None:
    """Ticks before any new_question() call are silently dropped."""
    snap = _snapshot(tick_id=0)
    logger.log_tick(snap, "s", "u", NumericalResponse(value=0), 1.0, [])
    dirs = [d for d in logger.session_dir.iterdir() if d.is_dir()]
    assert dirs == []


def test_close_is_idempotent(logger: VLMLogger) -> None:
    logger.new_question("How many chairs")
    logger.close()
    logger.close()


def test_slugify_special_characters(logger: VLMLogger) -> None:
    logger.new_question("How many red chairs & tables?!")
    dirs = sorted(d.name for d in logger.session_dir.iterdir() if d.is_dir())
    assert "q_001_how_many_red_chairs_tables" in dirs
