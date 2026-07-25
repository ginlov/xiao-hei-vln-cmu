"""Tests for time-syncing the camera frame to the pose that produced it.

The camera runs at ~4 Hz while ``/state_estimation`` runs at ~200 Hz, so the
newest pose is never the pose the image was taken from. ``LatestCache`` keeps a
short pose history and hands the snapshot an ``image_pose`` interpolated at the
image's own stamp; these tests pin that behaviour down.
"""

from __future__ import annotations

import math

import pytest

from xiao_hei_vln.messages import (
    Header,
    ImageFrame,
    OdomPose,
    Quaternion,
    Stamp,
    Vector3,
)
from xiao_hei_vln.sync import LatestCache


def _image(t: float) -> ImageFrame:
    return ImageFrame(
        header=Header(stamp=Stamp.from_seconds(t), frame_id="camera"),
        width=2,
        height=1,
        encoding="bgr8",
        step=6,
        data=b"\x00" * 6,
    )


def _pose(t: float, x: float = 0.0, *, yaw: float = 0.0, vx: float = 0.0) -> OdomPose:
    return OdomPose(
        header=Header(stamp=Stamp.from_seconds(t), frame_id="map"),
        position=Vector3(x=x, y=0.0, z=0.75),
        orientation=Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2)),
        linear_velocity=Vector3(x=vx, y=0.0, z=0.0),
        angular_velocity=Vector3(x=0.0, y=0.0, z=0.0),
    )


def _yaw_of(q: Quaternion) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class TestPoseAt:
    def test_interpolates_between_bracketing_samples(self) -> None:
        cache = LatestCache()
        cache.put_pose(_pose(10.0, x=0.0))
        cache.put_pose(_pose(11.0, x=2.0))

        mid = cache.pose_at(10.25)
        assert mid is not None
        assert mid.position.x == pytest.approx(0.5)

    def test_clamps_instead_of_extrapolating(self) -> None:
        cache = LatestCache()
        cache.put_pose(_pose(10.0, x=1.0))
        cache.put_pose(_pose(11.0, x=2.0))

        # Older than anything we hold, and newer than anything we hold: return
        # the nearest end rather than inventing motion beyond the history.
        assert cache.pose_at(5.0).position.x == pytest.approx(1.0)
        assert cache.pose_at(99.0).position.x == pytest.approx(2.0)

    def test_empty_history_returns_none(self) -> None:
        assert LatestCache().pose_at(1.0) is None

    def test_single_sample_history(self) -> None:
        cache = LatestCache()
        cache.put_pose(_pose(10.0, x=3.0))
        assert cache.pose_at(9.0).position.x == pytest.approx(3.0)

    def test_orientation_takes_the_short_way_round(self) -> None:
        # Two quaternions for headings either side of +/-pi. A naive blend
        # would swing the long way and put the robot facing backwards, which
        # would rotate every lifted point about the sensor.
        cache = LatestCache()
        cache.put_pose(_pose(10.0, yaw=math.pi - 0.1))
        cache.put_pose(_pose(11.0, yaw=-math.pi + 0.1))

        mid = cache.pose_at(10.5)
        yaw = abs(_yaw_of(mid.orientation))
        assert yaw > math.pi - 0.2  # near +/-pi, i.e. it did NOT swing to 0

    def test_history_is_bounded(self) -> None:
        cache = LatestCache(pose_history=4)
        for i in range(20):
            cache.put_pose(_pose(float(i), x=float(i)))
        # The oldest surviving sample is what an over-old query clamps to.
        assert cache.pose_at(0.0).position.x == pytest.approx(16.0)


class TestSnapshotImagePose:
    def test_image_pose_is_rewound_to_the_image_stamp(self) -> None:
        cache = LatestCache()
        # Robot drives from x=0 to x=1.2 over 1.2 s (1 m/s), 200 Hz odometry.
        for i in range(241):
            t = 10.0 + i * 0.005
            cache.put_pose(_pose(t, x=i * 0.005))
        # The image is 1.2 s old — the lag we actually measured on the sim.
        cache.put_image(_image(10.0))

        snap = cache.snapshot(0, Stamp.from_seconds(11.2))
        assert snap.pose is not None and snap.image_pose is not None
        # The newest pose is a full 1.2 m away from where the frame was taken.
        assert snap.pose.position.x == pytest.approx(1.2)
        assert snap.image_pose.position.x == pytest.approx(0.0, abs=1e-6)

    def test_image_pose_is_none_without_an_image(self) -> None:
        cache = LatestCache()
        cache.put_pose(_pose(10.0))
        assert cache.snapshot(0, Stamp.from_seconds(10.0)).image_pose is None

    def test_image_pose_is_none_without_pose_history(self) -> None:
        cache = LatestCache()
        cache.put_image(_image(10.0))
        assert cache.snapshot(0, Stamp.from_seconds(10.0)).image_pose is None

    def test_pose_slot_still_reports_the_newest(self) -> None:
        # Exploration asks "where am I now" — that must not start returning a
        # rewound pose just because the lift needs one.
        cache = LatestCache()
        cache.put_pose(_pose(10.0, x=0.0))
        cache.put_pose(_pose(10.5, x=5.0))
        cache.put_image(_image(10.0))

        snap = cache.snapshot(0, Stamp.from_seconds(10.5))
        assert snap.pose.position.x == pytest.approx(5.0)
        assert snap.image_pose.position.x == pytest.approx(0.0)
