"""`VLMOutput` → ROS 2 publishers.

Routes each response variant to the topic the challenge evaluation
node expects:

- `NumericalResponse`       → `/numerical_response`            (Int32)
- `ObjectReferenceResponse` → `/selected_object_marker`        (Marker)
- `WaypointPathResponse`    → `/way_point_with_heading`        (Pose2D, one at a time)

`rclpy` and the message packages are imported lazily so the core
contract remains testable without a ROS install.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from xiao_hei_vln.messages.outputs import (
    NumericalResponse,
    ObjectReferenceResponse,
    VLMOutput,
    Waypoint,
    WaypointPathResponse,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rclpy.node import Node


class VLMOutputPublisher:
    """Bundle of publishers; call `.publish(output)` with any `VLMOutput`."""

    def __init__(self, node: Node) -> None:
        from geometry_msgs.msg import Pose2D
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Int32
        from visualization_msgs.msg import Marker

        qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )

        self._node = node
        self._pose2d_cls = Pose2D
        self._marker_cls = Marker
        self._int32_cls = Int32

        self._numerical_pub = node.create_publisher(Int32, "/numerical_response", qos)
        self._marker_pub = node.create_publisher(Marker, "/selected_object_marker", qos)
        self._waypoint_pub = node.create_publisher(Pose2D, "/way_point_with_heading", qos)

    def publish(self, output: VLMOutput) -> None:
        if isinstance(output, NumericalResponse):
            self._publish_numerical(output)
        elif isinstance(output, ObjectReferenceResponse):
            self._publish_object_reference(output)
        elif isinstance(output, WaypointPathResponse):
            self._publish_waypoint(output.waypoints[0])
        else:  # pragma: no cover - exhaustiveness check
            raise TypeError(f"Unsupported VLMOutput type: {type(output)!r}")

    def publish_waypoint(self, waypoint: Waypoint) -> None:
        """Publish a single waypoint — used to advance through a path on `/way_point_reached`."""
        self._publish_waypoint(waypoint)

    # --- internals ---------------------------------------------------------------

    def _publish_numerical(self, msg: NumericalResponse) -> None:
        ros_msg = self._int32_cls()
        ros_msg.data = int(msg.value)
        self._numerical_pub.publish(ros_msg)

    def _publish_object_reference(self, msg: ObjectReferenceResponse) -> None:
        ros_msg = self._marker_cls()
        ros_msg.header.frame_id = "map"
        ros_msg.header.stamp = self._node.get_clock().now().to_msg()
        ros_msg.ns = msg.label
        ros_msg.id = int(msg.object_id)
        ros_msg.action = self._marker_cls.ADD
        ros_msg.type = self._marker_cls.CUBE
        ros_msg.pose.position.x = float(msg.center.x)
        ros_msg.pose.position.y = float(msg.center.y)
        ros_msg.pose.position.z = float(msg.center.z)
        half = msg.heading / 2.0
        ros_msg.pose.orientation.x = 0.0
        ros_msg.pose.orientation.y = 0.0
        ros_msg.pose.orientation.z = math.sin(half)
        ros_msg.pose.orientation.w = math.cos(half)
        ros_msg.scale.x = float(msg.size.x)
        ros_msg.scale.y = float(msg.size.y)
        ros_msg.scale.z = float(msg.size.z)
        ros_msg.color.a = 0.5
        ros_msg.color.r = 0.0
        ros_msg.color.g = 0.0
        ros_msg.color.b = 1.0
        self._marker_pub.publish(ros_msg)

    def _publish_waypoint(self, wp: Waypoint) -> None:
        ros_msg = self._pose2d_cls()
        ros_msg.x = float(wp.x)
        ros_msg.y = float(wp.y)
        ros_msg.theta = float(wp.heading)
        self._waypoint_pub.publish(ros_msg)
