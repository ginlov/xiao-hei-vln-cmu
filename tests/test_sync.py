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


class TestTimestampMatching:
    """The camera lags, so snapshot() must pair pose+scan to the IMAGE stamp,
    not hand back the freshest pose (TASK 27)."""

    def test_pose_is_matched_to_the_image_stamp_not_the_latest(self) -> None:
        cache = LatestCache()
        cache.put_image(_image(1.0))            # camera is 0.4 s behind
        for t in (0.8, 1.0, 1.2, 1.4):          # pose stream runs ahead
            cache.put_pose(_pose(t))
        snap = cache.snapshot(0, Stamp.from_seconds(1.4))
        # Nearest to the image (1.0), not the newest (1.4).
        assert snap.pose.header.stamp.to_seconds() == pytest.approx(1.0)

    def test_registered_scan_is_matched_too(self) -> None:
        cache = LatestCache()
        cache.put_image(_image(2.0))
        for t in (1.9, 2.05, 2.5):
            cache.put_registered_scan(_scan("registered", t))
        snap = cache.snapshot(0, Stamp.from_seconds(2.5))
        assert snap.registered_scan.header.stamp.to_seconds() == pytest.approx(2.05)

    def test_no_image_falls_back_to_latest(self) -> None:
        # Without an anchor there is nothing to match to; latest is the only
        # honest choice (and the old behaviour).
        cache = LatestCache()
        cache.put_pose(_pose(1.0))
        cache.put_pose(_pose(2.0))
        snap = cache.snapshot(0, Stamp.from_seconds(3.0))
        assert snap.pose.header.stamp.to_seconds() == pytest.approx(2.0)

    def test_gap_wider_than_window_falls_back_to_latest(self) -> None:
        # A pose 5 s from the image is not "the image's pose" — a stream
        # dropped out. Pairing them would be worse than using latest.
        cache = LatestCache()
        cache.put_image(_image(10.0))
        cache.put_pose(_pose(2.0))
        cache.put_pose(_pose(3.0))
        snap = cache.snapshot(0, Stamp.from_seconds(10.0))
        assert snap.pose.header.stamp.to_seconds() == pytest.approx(3.0)

    def test_within_window_matches_even_when_latest_is_far(self) -> None:
        cache = LatestCache()
        cache.put_image(_image(5.0))
        cache.put_pose(_pose(4.7))              # 0.3 s away — inside the window
        cache.put_pose(_pose(9.0))              # newest, but way off
        snap = cache.snapshot(0, Stamp.from_seconds(9.0))
        assert snap.pose.header.stamp.to_seconds() == pytest.approx(4.7)

    def test_history_bound_does_not_break_matching(self) -> None:
        # More poses than the ring holds; the matched one must still be there
        # as long as it is recent relative to the image.
        cache = LatestCache()
        for t in range(0, 400):
            cache.put_pose(_pose(float(t) * 0.1))
        cache.put_image(_image(39.5))
        snap = cache.snapshot(0, Stamp.from_seconds(39.9))
        assert snap.pose.header.stamp.to_seconds() == pytest.approx(39.5, abs=0.06)


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
