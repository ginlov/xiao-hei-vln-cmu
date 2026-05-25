"""ROS 2 → `LatestCache` glue.

`rclpy` and `sensor_msgs` are imported lazily so the rest of the
package stays importable without a ROS install. This module is only
meaningful inside a ROS-enabled environment (the challenge docker
container or a workstation with ROS Jazzy installed).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from xiao_hei_vln.messages.common import Header, Quaternion, Stamp, Vector3
from xiao_hei_vln.messages.question import ChallengeQuestion
from xiao_hei_vln.messages.sensors import ImageFrame, LidarScan, OdomPose, TerrainMap
from xiao_hei_vln.sync.latest_cache import LatestCache

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rclpy.node import Node


def _stamp_from_ros(rs) -> Stamp:
    return Stamp(sec=int(rs.sec), nanosec=int(rs.nanosec))


def _header_from_ros(rh) -> Header:
    return Header(stamp=_stamp_from_ros(rh.stamp), frame_id=str(rh.frame_id))


def _pointcloud2_to_xyzi(msg) -> np.ndarray:
    """Convert a PointCloud2 with at least x/y/z (and optional intensity) to an (N, 4) ndarray.

    Topics observed in the simulator use FLOAT32 fields. `/terrain_map`
    carries 16 bytes of padding after intensity (point_step=32); we
    read each named field by offset rather than slicing the raw buffer
    contiguously, which keeps the adapter robust to padding/field
    re-ordering.
    """
    n = int(msg.width) * int(msg.height)
    step = int(msg.point_step)
    raw = bytes(msg.data)
    buf = np.frombuffer(raw, dtype=np.uint8).reshape(n, step)

    out = np.zeros((n, 4), dtype=np.float32)
    field_offsets = {f.name: int(f.offset) for f in msg.fields}
    for idx, name in enumerate(("x", "y", "z", "intensity")):
        offset = field_offsets.get(name)
        if offset is None:
            continue
        out[:, idx] = np.frombuffer(
            buf[:, offset : offset + 4].tobytes(), dtype=np.float32
        )
    return out


def bind_subscribers(node: Node, cache: LatestCache) -> dict[str, object]:
    """Create QoS-matched subscribers on `node` that write into `cache`.

    Returns a mapping {channel_name: subscription} so the caller can
    hold references and later destroy them.
    """
    # Local imports — see module docstring.
    from nav_msgs.msg import Odometry
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image, PointCloud2
    from std_msgs.msg import String

    qos = QoSProfile(
        depth=5,
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
    )

    def on_image(msg) -> None:
        cache.put_image(
            ImageFrame(
                header=_header_from_ros(msg.header),
                width=int(msg.width),
                height=int(msg.height),
                encoding=str(msg.encoding),
                step=int(msg.step),
                data=bytes(msg.data),
            )
        )

    def on_registered(msg) -> None:
        cache.put_registered_scan(
            LidarScan(
                header=_header_from_ros(msg.header),
                points=_pointcloud2_to_xyzi(msg),
                source="registered",
            )
        )

    def on_sensor(msg) -> None:
        cache.put_sensor_scan(
            LidarScan(
                header=_header_from_ros(msg.header),
                points=_pointcloud2_to_xyzi(msg),
                source="sensor",
            )
        )

    def on_terrain(msg) -> None:
        cache.put_terrain_local(
            TerrainMap(
                header=_header_from_ros(msg.header),
                points=_pointcloud2_to_xyzi(msg),
                range="local_5m",
            )
        )

    def on_terrain_ext(msg) -> None:
        cache.put_terrain_ext(
            TerrainMap(
                header=_header_from_ros(msg.header),
                points=_pointcloud2_to_xyzi(msg),
                range="ext_20m",
            )
        )

    def on_pose(msg) -> None:
        twist_lin = msg.twist.twist.linear
        twist_ang = msg.twist.twist.angular
        cache.put_pose(
            OdomPose(
                header=_header_from_ros(msg.header),
                child_frame_id=str(msg.child_frame_id),
                position=Vector3(
                    x=float(msg.pose.pose.position.x),
                    y=float(msg.pose.pose.position.y),
                    z=float(msg.pose.pose.position.z),
                ),
                orientation=Quaternion(
                    x=float(msg.pose.pose.orientation.x),
                    y=float(msg.pose.pose.orientation.y),
                    z=float(msg.pose.pose.orientation.z),
                    w=float(msg.pose.pose.orientation.w),
                ),
                linear_velocity=Vector3(
                    x=float(twist_lin.x), y=float(twist_lin.y), z=float(twist_lin.z)
                ),
                angular_velocity=Vector3(
                    x=float(twist_ang.x), y=float(twist_ang.y), z=float(twist_ang.z)
                ),
            )
        )

    def on_question(msg) -> None:
        now = node.get_clock().now().to_msg()
        cache.put_question(
            ChallengeQuestion.from_text(str(msg.data), _stamp_from_ros(now))
        )

    return {
        "image": node.create_subscription(Image, "/camera/image", on_image, qos),
        "registered_scan": node.create_subscription(
            PointCloud2, "/registered_scan", on_registered, qos
        ),
        "sensor_scan": node.create_subscription(
            PointCloud2, "/sensor_scan", on_sensor, qos
        ),
        "terrain_local": node.create_subscription(
            PointCloud2, "/terrain_map", on_terrain, qos
        ),
        "terrain_ext": node.create_subscription(
            PointCloud2, "/terrain_map_ext", on_terrain_ext, qos
        ),
        "pose": node.create_subscription(Odometry, "/state_estimation", on_pose, qos),
        "question": node.create_subscription(
            String, "/challenge_question", on_question, qos
        ),
    }
