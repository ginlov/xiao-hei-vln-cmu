"""Tests for point-cloud saving in ``VLMLogger``."""

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
)
from xiao_hei_vln.messages.sensors import LidarScan, TerrainMap
from xiao_hei_vln.logger import VLMLogger
from xiao_hei_vln.qwen.config import QwenConfig


def _stamp(t: float = 0.0) -> Stamp:
    return Stamp.from_seconds(t)


def _pose() -> OdomPose:
    return OdomPose(
        header=Header(stamp=_stamp(), frame_id="map"),
        position=Vector3(x=1.0, y=2.0, z=0.5),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )


def _lidar(n: int = 100, source: str = "registered") -> LidarScan:
    pts = np.random.default_rng(42).random((n, 4)).astype(np.float32)
    return LidarScan(
        header=Header(stamp=_stamp(), frame_id="map"),
        points=pts,
        source=source,
    )


def _terrain(n: int = 50, range_: str = "local_5m") -> TerrainMap:
    pts = np.random.default_rng(7).random((n, 4)).astype(np.float32)
    return TerrainMap(
        header=Header(stamp=_stamp(), frame_id="map"),
        points=pts,
        range=range_,
    )


@pytest.fixture()
def logger(tmp_path: Path) -> VLMLogger:
    from dataclasses import asdict

    lg = VLMLogger(
        tmp_path, config=asdict(QwenConfig()), responder_name="qwen", tick_hz=2.0,
    )
    yield lg
    lg.close()


def test_pointcloud_saved_as_npy(logger: VLMLogger) -> None:
    logger.new_question("How many chairs")
    scan = _lidar(100, source="registered")
    snap = VLMInput(
        tick_id=3,
        tick_time=_stamp(1.5),
        question=ChallengeQuestion.from_text("How many chairs", _stamp()),
        pose=_pose(),
        registered_scan=scan,
    )
    logger.log_tick(snap, "s", "u", NumericalResponse(value=0), 1.0, [])

    q_dir = logger.session_dir / "q_001_how_many_chairs"
    npy_path = q_dir / "pointclouds" / "tick_000003_registered.npy"
    assert npy_path.exists()

    loaded = np.load(npy_path)
    np.testing.assert_array_equal(loaded, scan.points)


def test_multiple_pointclouds_saved(logger: VLMLogger) -> None:
    logger.new_question("How many chairs")
    snap = VLMInput(
        tick_id=1,
        tick_time=_stamp(0.5),
        question=ChallengeQuestion.from_text("How many chairs", _stamp()),
        pose=_pose(),
        registered_scan=_lidar(80, "registered"),
        sensor_scan=_lidar(60, "sensor"),
        terrain_local=_terrain(50, "local_5m"),
        terrain_ext=_terrain(30, "ext_20m"),
    )
    logger.log_tick(snap, "s", "u", NumericalResponse(value=0), 1.0, [])

    q_dir = logger.session_dir / "q_001_how_many_chairs"
    pc_dir = q_dir / "pointclouds"
    assert (pc_dir / "tick_000001_registered.npy").exists()
    assert (pc_dir / "tick_000001_sensor.npy").exists()
    assert (pc_dir / "tick_000001_terrain_local.npy").exists()
    assert (pc_dir / "tick_000001_terrain_ext.npy").exists()


def test_no_pointcloud_when_none(logger: VLMLogger) -> None:
    logger.new_question("How many chairs")
    snap = VLMInput(
        tick_id=0,
        tick_time=_stamp(),
        question=ChallengeQuestion.from_text("How many chairs", _stamp()),
        pose=_pose(),
    )
    logger.log_tick(snap, "s", "u", NumericalResponse(value=0), 1.0, [])

    q_dir = logger.session_dir / "q_001_how_many_chairs"
    pc_files = list((q_dir / "pointclouds").iterdir())
    assert pc_files == []


def test_pointclouds_field_in_jsonl(logger: VLMLogger) -> None:
    logger.new_question("How many chairs")
    snap = VLMInput(
        tick_id=5,
        tick_time=_stamp(2.5),
        question=ChallengeQuestion.from_text("How many chairs", _stamp()),
        pose=_pose(),
        registered_scan=_lidar(40, "registered"),
        terrain_local=_terrain(20, "local_5m"),
    )
    logger.log_tick(snap, "s", "u", NumericalResponse(value=2), 1.0, [])

    q_dir = logger.session_dir / "q_001_how_many_chairs"
    record = json.loads(
        (q_dir / "ticks.jsonl").read_text().strip(),
    )
    pcs = record["pointclouds"]
    assert pcs["registered"] == "pointclouds/tick_000005_registered.npy"
    assert pcs["terrain_local"] == "pointclouds/tick_000005_terrain_local.npy"
    assert "sensor" not in pcs
    assert "terrain_ext" not in pcs


def test_empty_pointclouds_when_no_sensors(logger: VLMLogger) -> None:
    logger.new_question("How many chairs")
    snap = VLMInput(
        tick_id=0,
        tick_time=_stamp(),
        question=ChallengeQuestion.from_text("How many chairs", _stamp()),
        pose=_pose(),
    )
    logger.log_tick(snap, "s", "u", NumericalResponse(value=0), 1.0, [])

    q_dir = logger.session_dir / "q_001_how_many_chairs"
    record = json.loads(
        (q_dir / "ticks.jsonl").read_text().strip(),
    )
    assert record["pointclouds"] == {}
