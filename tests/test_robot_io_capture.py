"""`Capture` must pair the pose with the image, not with subscription time.

`scripts/robot_io.py` runs inside `iros2026_system`, the only place with ROS on
its path, so it cannot be imported here as it stands. The ROS surface it uses is
six names and none of their behaviour matters to this question, so they are
stubbed and the real module is imported on top — this exercises the shipped
callbacks, not a copy of them.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _stub_ros() -> dict:
    """Minimal stand-ins for the ROS names `robot_io` imports."""
    saved = {k: sys.modules.get(k) for k in
             ("rclpy", "rclpy.node", "rclpy.qos", "geometry_msgs",
              "geometry_msgs.msg", "nav_msgs", "nav_msgs.msg", "sensor_msgs",
              "sensor_msgs.msg", "std_msgs", "std_msgs.msg", "robot_io")}

    class Node:
        def __init__(self, name: str) -> None:
            self._name = name

        def create_subscription(self, *a, **k) -> None:
            return None

        def destroy_node(self) -> None:
            return None

    def mod(name: str, **attrs) -> types.ModuleType:
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    mod("rclpy", spin_once=lambda *a, **k: None, init=lambda *a, **k: None,
        shutdown=lambda *a, **k: None)
    mod("rclpy.node", Node=Node)
    mod("rclpy.qos", qos_profile_sensor_data=object())
    mod("geometry_msgs"); mod("geometry_msgs.msg", Pose2D=type("Pose2D", (), {}))
    mod("nav_msgs"); mod("nav_msgs.msg", Odometry=type("Odometry", (), {}))
    mod("sensor_msgs")
    mod("sensor_msgs.msg", CompressedImage=type("CompressedImage", (), {}),
        PointCloud2=type("PointCloud2", (), {}))
    mod("std_msgs"); mod("std_msgs.msg", Float32=type("Float32", (), {}))
    return saved


@pytest.fixture()
def capture(tmp_path, monkeypatch):
    saved = _stub_ros()
    sys.path.insert(0, str(SCRIPTS))
    try:
        sys.modules.pop("robot_io", None)
        rio = importlib.import_module("robot_io")
        # Keep the callbacks off /tmp, which parallel runs share.
        monkeypatch.setattr(rio, "IMG", str(tmp_path / "img.jpg"))
        monkeypatch.setattr(rio, "POSE", str(tmp_path / "pose.json"))
        yield rio
    finally:
        sys.path.remove(str(SCRIPTS))
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _img(payload=b"jpeg"):
    return types.SimpleNamespace(data=payload)


def _odom(x, y, z=0.75):
    p = types.SimpleNamespace(x=x, y=y, z=z)
    q = types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
    return types.SimpleNamespace(
        pose=types.SimpleNamespace(pose=types.SimpleNamespace(
            position=p, orientation=q)))


class TestThePoseComesAfterTheImage:
    """Pose publishes at 100-200 Hz and the camera at ~9.3 Hz, so first-wins on
    each topic pairs the frame with a pose up to a 107 ms camera period older
    than it. PR #28 fixed the same defect on the perception side, where it was
    worth up to 17° of azimuth while the vehicle turned.
    """

    def test_a_pose_from_before_the_frame_is_discarded(self, capture):
        c = capture.Capture()
        c._on_pose(_odom(1.0, 1.0))            # arrives first, as it always does
        assert c.pose is not None
        c._on_img(_img())
        assert c.pose is None, "the pre-frame pose must not survive the frame"

    def test_the_next_pose_is_kept(self, capture):
        c = capture.Capture()
        c._on_pose(_odom(1.0, 1.0))
        c._on_img(_img())
        c._on_pose(_odom(2.0, 2.0))
        assert c.pose["position"][:2] == [2.0, 2.0]

    def test_a_pose_after_the_frame_is_not_thrown_away_twice(self, capture):
        """Only the frame clears it; later poses must not keep resetting it, or
        `done()` never becomes true and every capture times out."""
        c = capture.Capture()
        c._on_img(_img())
        c._on_pose(_odom(2.0, 2.0))
        c._on_pose(_odom(3.0, 3.0))
        assert c.pose["position"][:2] == [2.0, 2.0]

    def test_capture_is_not_done_until_the_pose_is_re_read(self, capture):
        c = capture.Capture()
        c.scan = c.terrain = True
        c._on_pose(_odom(1.0, 1.0))
        c._on_img(_img())
        assert not c.done(), "the stale pose must not satisfy `done`"
        c._on_pose(_odom(1.01, 1.0))
        assert c.done()

    def test_the_image_is_still_taken_once(self, capture):
        """Clearing the pose must not make the frame re-writable."""
        c = capture.Capture()
        c._on_img(_img(b"first"))
        c._on_img(_img(b"second"))
        assert Path(capture.IMG).read_bytes() == b"first"

    def test_the_scan_is_deliberately_left_alone(self, capture):
        """`/registered_scan` arrives already in the map frame, so a stale one
        is stale geometry rather than misregistered geometry — and
        `noDecayDis` gives terrain a 1.75 m memory regardless."""
        c = capture.Capture()
        c.scan = c.terrain = True
        c._on_img(_img())
        assert c.scan is True and c.terrain is True
