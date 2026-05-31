"""Tests for ``VLMLogger`` — session init, JSONL appending, and JPEG saving."""

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
    data[:, :, 0] = 255  # blue channel (BGR)
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


def _session_dirs(tmp_path: Path) -> list[Path]:
    return sorted(tmp_path.glob("session_*"))


# --- tests ---


def test_session_json_written(logger: VLMLogger) -> None:
    session_json = logger.session_dir / "session.json"
    assert session_json.exists()
    data = json.loads(session_json.read_text())
    assert data["responder"] == "qwen"
    assert data["tick_hz"] == 2.0
    assert "config" in data
    assert data["config"]["model"] == "/models/Qwen3.5-4B"
    assert "start_time" in data


def test_tick_appended_to_jsonl(logger: VLMLogger) -> None:
    snap = _snapshot(tick_id=1)
    output = NumericalResponse(value=3, rationale="counted 3")
    logger.log_tick(snap, "system", "user text", output, 42.5, ["prev evidence"])

    jsonl = logger.session_dir / "ticks.jsonl"
    lines = jsonl.read_text().strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["tick_id"] == 1
    assert record["question_text"] == "How many chairs"
    assert record["question_type"] == "numerical"
    assert record["system_prompt"] == "system"
    assert record["user_text"] == "user text"
    assert record["output"]["kind"] == "numerical"
    assert record["output"]["value"] == 3
    assert record["inference_ms"] == 42.5
    assert record["evidence"] == ["prev evidence"]
    assert record["image_path"] is None


def test_image_saved_as_jpeg(logger: VLMLogger) -> None:
    snap = _snapshot(tick_id=5, with_image=True)
    output = NumericalResponse(value=1)
    logger.log_tick(snap, "sys", "usr", output, 10.0, [])

    img_path = logger.session_dir / "images" / "tick_000005.jpg"
    assert img_path.exists()
    assert img_path.stat().st_size > 0

    record = json.loads(
        (logger.session_dir / "ticks.jsonl").read_text().strip(),
    )
    assert record["image_path"] == "images/tick_000005.jpg"


def test_no_image_when_none(logger: VLMLogger) -> None:
    snap = _snapshot(tick_id=3, with_image=False)
    output = NumericalResponse(value=0)
    logger.log_tick(snap, "sys", "usr", output, 5.0, [])

    images = list((logger.session_dir / "images").iterdir())
    assert images == []

    record = json.loads(
        (logger.session_dir / "ticks.jsonl").read_text().strip(),
    )
    assert record["image_path"] is None


def test_multiple_ticks_append(logger: VLMLogger) -> None:
    for i in range(3):
        snap = _snapshot(tick_id=i)
        output = WaypointPathResponse(
            waypoints=[Waypoint(x=float(i), y=0.0)],
            rationale=f"exploring tick {i}",
        )
        logger.log_tick(snap, "sys", "usr", output, float(i * 10), [])

    jsonl = logger.session_dir / "ticks.jsonl"
    lines = jsonl.read_text().strip().splitlines()
    assert len(lines) == 3
    ids = [json.loads(line)["tick_id"] for line in lines]
    assert ids == [0, 1, 2]


def test_output_none_logged(logger: VLMLogger) -> None:
    snap = _snapshot(tick_id=7)
    logger.log_tick(snap, "sys", "usr", None, 0.0, [])

    record = json.loads(
        (logger.session_dir / "ticks.jsonl").read_text().strip(),
    )
    assert record["output"] is None
    assert record["tick_id"] == 7


def test_pose_serialized(logger: VLMLogger) -> None:
    snap = _snapshot(tick_id=0)
    logger.log_tick(snap, "sys", "usr", NumericalResponse(value=0), 1.0, [])

    record = json.loads(
        (logger.session_dir / "ticks.jsonl").read_text().strip(),
    )
    assert record["pose"]["position"] == {"x": 1.0, "y": 2.0, "z": 0.5}
    assert record["pose"]["orientation"] == {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}


def test_no_pose_serialized_as_null(logger: VLMLogger) -> None:
    snap = VLMInput(
        tick_id=0,
        tick_time=_stamp(),
        question=ChallengeQuestion.from_text("How many cups", _stamp()),
        pose=None,
    )
    logger.log_tick(snap, "sys", "usr", NumericalResponse(value=0), 1.0, [])

    record = json.loads(
        (logger.session_dir / "ticks.jsonl").read_text().strip(),
    )
    assert record["pose"] is None


def test_close_is_idempotent(logger: VLMLogger) -> None:
    logger.close()
    logger.close()
