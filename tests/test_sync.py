"""Tests for the latest-cache + tick-snapshot pattern."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from xiao_hei_vln.messages import (
    ChallengeQuestion,
    Header,
    ImageFrame,
    LidarScan,
    OdomPose,
    Quaternion,
    Stamp,
    TerrainMap,
    Vector3,
)
from xiao_hei_vln.sync import LatestCache


def _header(frame: str, t: float = 0.0) -> Header:
    return Header(stamp=Stamp.from_seconds(t), frame_id=frame)


def _image(t: float) -> ImageFrame:
    return ImageFrame(
        header=_header("camera", t),
        width=2,
        height=1,
        encoding="bgr8",
        step=6,
        data=b"\x00" * 6,
    )


def _scan(source: str, t: float) -> LidarScan:
    frame = "map" if source == "registered" else "sensor_at_scan"
    return LidarScan(
        header=_header(frame, t),
        points=np.zeros((4, 4), dtype=np.float32),
        source=source,  # type: ignore[arg-type]
    )


def _terrain(rng: str, t: float) -> TerrainMap:
    return TerrainMap(
        header=_header("map", t),
        points=np.zeros((2, 4), dtype=np.float32),
        range=rng,  # type: ignore[arg-type]
    )


def _pose(t: float) -> OdomPose:
    return OdomPose(
        header=_header("map", t),
        position=Vector3(x=0.0, y=0.0, z=0.75),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )


class TestLatestCacheBasics:
    def test_cold_start_snapshot_is_empty(self) -> None:
        cache = LatestCache()
        snap = cache.snapshot(0, Stamp.from_seconds(0.0))
        assert snap.tick_id == 0
        assert snap.image is None
        assert snap.pose is None
        assert snap.question is None

    def test_put_then_snapshot_returns_latest(self) -> None:
        cache = LatestCache()
        cache.put_image(_image(1.0))
        cache.put_image(_image(2.0))
        cache.put_pose(_pose(2.5))
        snap = cache.snapshot(7, Stamp.from_seconds(3.0))
        assert snap.image is not None
        assert snap.image.header.stamp.to_seconds() == pytest.approx(2.0)
        assert snap.pose is not None
        assert snap.tick_id == 7

    def test_source_mismatch_raises(self) -> None:
        cache = LatestCache()
        with pytest.raises(ValueError):
            cache.put_registered_scan(_scan("sensor", 0.0))
        with pytest.raises(ValueError):
            cache.put_sensor_scan(_scan("registered", 0.0))
        with pytest.raises(ValueError):
            cache.put_terrain_local(_terrain("ext_20m", 0.0))
        with pytest.raises(ValueError):
            cache.put_terrain_ext(_terrain("local_5m", 0.0))

    def test_question_is_sticky_until_cleared(self) -> None:
        cache = LatestCache()
        cache.put_question(
            ChallengeQuestion.from_text("How many cups", Stamp.from_seconds(0.0))
        )
        first = cache.snapshot(0, Stamp.from_seconds(0.5))
        second = cache.snapshot(1, Stamp.from_seconds(1.0))
        assert first.question is not None
        assert second.question == first.question  # frozen models compare by value
        cache.clear_question()
        third = cache.snapshot(2, Stamp.from_seconds(1.5))
        assert third.question is None


class TestLatestCacheConcurrency:
    def test_concurrent_writers_and_snapshots(self) -> None:
        cache = LatestCache()
        stop = threading.Event()
        errors: list[BaseException] = []

        def writer(start_t: float) -> None:
            try:
                t = start_t
                while not stop.is_set():
                    cache.put_image(_image(t))
                    cache.put_registered_scan(_scan("registered", t))
                    cache.put_pose(_pose(t))
                    t += 0.01
            except BaseException as exc:  # pragma: no cover - safety net
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i * 1.0,)) for i in range(4)]
        for th in threads:
            th.start()

        try:
            for i in range(200):
                snap = cache.snapshot(i, Stamp.from_seconds(float(i)))
                # No torn reads: every populated slot must be a fully-constructed model.
                if snap.image is not None:
                    assert snap.image.width == 2
                if snap.registered_scan is not None:
                    assert snap.registered_scan.source == "registered"
                if snap.pose is not None:
                    assert snap.pose.position.z == pytest.approx(0.75)
        finally:
            stop.set()
            for th in threads:
                th.join(timeout=2.0)

        assert not errors, errors
