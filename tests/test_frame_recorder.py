"""Tests for ``FrameRecorder`` — what lands on disk, and what gets skipped."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from xiao_hei_vln.messages import (
    Header,
    OdomPose,
    Quaternion,
    Stamp,
    Vector3,
    VLMInput,
)
from xiao_hei_vln.messages.sensors import ImageFrame, LidarScan
from xiao_hei_vln.perception.recorder import FrameRecorder, from_env

_has_pillow = importlib.util.find_spec("PIL") is not None
_needs_pillow = pytest.mark.skipif(not _has_pillow, reason="pillow not installed")


def _pose(t: float = 1.0, *, with_velocity: bool = True) -> OdomPose:
    return OdomPose(
        header=Header(stamp=Stamp.from_seconds(t), frame_id="map"),
        position=Vector3(x=1.0, y=2.0, z=0.5),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        linear_velocity=Vector3(x=0.4, y=0.0, z=0.0) if with_velocity else None,
        angular_velocity=Vector3(x=0.0, y=0.0, z=0.2) if with_velocity else None,
    )


def _image(t: float = 0.75, width: int = 4, height: int = 2) -> ImageFrame:
    data = np.zeros((height, width, 3), dtype=np.uint8)
    return ImageFrame(
        header=Header(stamp=Stamp.from_seconds(t), frame_id="camera"),
        width=width,
        height=height,
        step=width * 3,
        data=data.tobytes(),
    )


def _scan(t: float = 0.9, n: int = 5) -> LidarScan:
    return LidarScan(
        header=Header(stamp=Stamp.from_seconds(t), frame_id="map"),
        points=np.arange(n * 4, dtype=np.float32).reshape(n, 4),
        source="registered",
    )


def _snapshot(tick_id: int = 0, **overrides) -> VLMInput:
    fields = {
        "image": _image(),
        "registered_scan": _scan(),
        "pose": _pose(),
    }
    fields.update(overrides)
    return VLMInput(
        tick_id=tick_id,
        tick_time=Stamp.from_seconds(tick_id * 0.5),
        **fields,
    )


def _rows(out_dir: Path) -> list[dict]:
    text = (out_dir / "frames.jsonl").read_text()
    return [json.loads(ln) for ln in text.splitlines() if ln.strip()]


@_needs_pillow
def test_records_image_scan_and_all_three_stamps(tmp_path: Path) -> None:
    rec = FrameRecorder(tmp_path, meta={"scene": "livingroom_3"})
    assert rec.record(_snapshot(4)) is True
    rec.close()

    assert (tmp_path / "tick_000004.jpg").exists()
    saved = np.load(tmp_path / "tick_000004_registered.npy")
    assert saved.shape == (5, 4)

    (row,) = _rows(tmp_path)
    assert row["tick_id"] == 4
    # The three stamps are the whole point: they are what makes the
    # image-vs-pose skew measurable offline.
    assert row["image_stamp"] == pytest.approx(0.75)
    assert row["pose_stamp"] == pytest.approx(1.0)
    assert row["scan_stamp"] == pytest.approx(0.9)
    assert row["linear_velocity"] == pytest.approx([0.4, 0.0, 0.0])
    assert row["angular_velocity"] == pytest.approx([0.0, 0.0, 0.2])
    assert row["image"] == "tick_000004.jpg"
    assert row["scan"] == "tick_000004_registered.npy"
    assert json.loads((tmp_path / "session.json").read_text())["scene"] == "livingroom_3"


@_needs_pillow
def test_missing_input_is_skipped_and_does_not_consume_stride(tmp_path: Path) -> None:
    rec = FrameRecorder(tmp_path, stride=2)
    # Cold-start ticks (no image / no pose / no scan) are not replayable, so
    # they must not be written *and* must not advance the stride counter —
    # otherwise a burst of empty ticks silently shifts which frames we keep.
    assert rec.record(_snapshot(0, image=None)) is False
    assert rec.record(_snapshot(1, pose=None)) is False
    assert rec.record(_snapshot(2, registered_scan=None)) is False
    assert rec.record(_snapshot(3)) is True      # 1st recordable → kept
    assert rec.record(_snapshot(4)) is False     # 2nd recordable → strided out
    assert rec.record(_snapshot(5)) is True      # 3rd recordable → kept
    rec.close()

    assert [r["tick_id"] for r in _rows(tmp_path)] == [3, 5]
    assert rec.n_written == 2


@_needs_pillow
def test_missing_velocities_serialize_as_null(tmp_path: Path) -> None:
    rec = FrameRecorder(tmp_path)
    rec.record(_snapshot(0, pose=_pose(with_velocity=False)))
    rec.close()

    (row,) = _rows(tmp_path)
    assert row["linear_velocity"] is None
    assert row["angular_velocity"] is None


def test_from_env_off_by_default(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("XIAO_HEI_FRAME_RECORD_DIR", raising=False)
    assert from_env() is None


@_needs_pillow
def test_from_env_reads_dir_and_stride(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XIAO_HEI_FRAME_RECORD_DIR", str(tmp_path / "frames"))
    monkeypatch.setenv("XIAO_HEI_FRAME_RECORD_STRIDE", "3")
    rec = from_env(meta={"scene": "x"})
    assert rec is not None
    assert rec.out_dir == tmp_path / "frames"
    assert rec._stride == 3
    rec.close()


@_needs_pillow
def test_write_failure_does_not_raise(tmp_path: Path, monkeypatch) -> None:
    # Recording is best-effort: a full disk must not take the ROS node down
    # mid-run, it must just cost us that frame.
    rec = FrameRecorder(tmp_path)

    def _boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(FrameRecorder, "_save_image", staticmethod(_boom))
    assert rec.record(_snapshot(7)) is False
    assert rec.n_written == 0
    rec.close()
