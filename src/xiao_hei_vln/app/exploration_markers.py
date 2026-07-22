"""Live RViz markers for frontier exploration state.

Publishes ``visualization_msgs/MarkerArray`` on ``/exploration/markers`` with
per-layer namespaces (toggle independently in RViz → MarkerArray → Namespaces):

* ``free``      — explored FREE cells (grey points)
* ``frontier``  — current FREE↔UNKNOWN frontier (cyan)
* ``soft_ban``  — temporarily skipped frontiers (yellow)
* ``hard_ban``  — permanently banned after multi-pose fails (red)
* ``visited``   — completed waypoints (green spheres)
* ``current``   — active goal (magenta sphere)
* ``path``      — polyline through visited + current
* ``legend``    — floating text key

ROS types are imported lazily so unit tests need no rclpy.
"""

from __future__ import annotations

from typing import Any


# Cap dense point clouds so RViz stays responsive in large rooms.
_MAX_FREE_POINTS = 2500
_MAX_LAYER_POINTS = 1500
_Z = 0.15  # float markers slightly above the floor


class ExplorationPublisher:
    """Builds + publishes exploration MarkerArray snapshots."""

    def __init__(
        self,
        node: Any,
        *,
        topic: str = "/exploration/markers",
        frame_id: str = "map",
    ) -> None:
        from visualization_msgs.msg import MarkerArray

        self._node = node
        self._frame_id = frame_id
        self._pub = node.create_publisher(MarkerArray, topic, 10)
        self._prev: set[tuple[str, int]] = set()

    def publish(self, layers: dict) -> None:
        from visualization_msgs.msg import MarkerArray

        stamp = self._node.get_clock().now().to_msg()
        arr = MarkerArray()
        cur: set[tuple[str, int]] = set()

        res = float(layers.get("resolution") or 0.2)
        point_scale = max(res * 0.9, 0.08)

        free = self._downsample(layers.get("free") or [], _MAX_FREE_POINTS)
        arr.markers.append(
            self._points("free", 0, free, stamp, point_scale, (0.75, 0.75, 0.75, 0.35))
        )
        cur.add(("free", 0))

        frontier = self._downsample(layers.get("frontier") or [], _MAX_LAYER_POINTS)
        arr.markers.append(
            self._points("frontier", 0, frontier, stamp, point_scale * 1.2, (0.1, 0.85, 1.0, 0.95))
        )
        cur.add(("frontier", 0))

        soft = self._downsample(layers.get("soft_ban") or [], _MAX_LAYER_POINTS)
        arr.markers.append(
            self._points("soft_ban", 0, soft, stamp, point_scale * 1.3, (1.0, 0.85, 0.1, 0.95))
        )
        cur.add(("soft_ban", 0))

        hard = self._downsample(layers.get("hard_ban") or [], _MAX_LAYER_POINTS)
        arr.markers.append(
            self._points("hard_ban", 0, hard, stamp, point_scale * 1.3, (1.0, 0.15, 0.1, 0.95))
        )
        cur.add(("hard_ban", 0))

        visited = list(layers.get("visited") or [])
        arr.markers.append(
            self._spheres("visited", 0, visited, stamp, 0.18, (0.2, 0.9, 0.3, 0.95))
        )
        cur.add(("visited", 0))

        current = layers.get("current")
        cur_pts = [current] if current is not None else []
        arr.markers.append(
            self._spheres("current", 0, cur_pts, stamp, 0.28, (0.95, 0.2, 0.95, 1.0))
        )
        cur.add(("current", 0))

        path_pts = list(visited)
        if current is not None:
            path_pts.append(current)
        arr.markers.append(self._line("path", 0, path_pts, stamp, 0.06, (0.2, 0.45, 1.0, 0.9)))
        cur.add(("path", 0))

        n_free = len(layers.get("free") or [])
        n_front = len(layers.get("frontier") or [])
        n_soft = len(layers.get("soft_ban") or [])
        n_hard = len(layers.get("hard_ban") or [])
        legend = (
            "EXPLORATION\n"
            f"grey=free ({n_free})\n"
            f"cyan=frontier ({n_front})\n"
            f"yellow=soft-ban ({n_soft})\n"
            f"red=hard-ban ({n_hard})\n"
            f"green=visited  magenta=goal"
        )
        arr.markers.append(self._legend(legend, stamp))
        cur.add(("legend", 0))

        for ns, oid in self._prev - cur:
            arr.markers.append(self._delete(ns, oid, stamp))
        self._prev = cur
        self._pub.publish(arr)

    # ------------------------------------------------------------------

    @staticmethod
    def _downsample(
        pts: list[tuple[float, float]],
        limit: int,
    ) -> list[tuple[float, float]]:
        if len(pts) <= limit or limit <= 0:
            return pts
        step = max(1, len(pts) // limit)
        return pts[::step][:limit]

    def _header(self, stamp):
        from std_msgs.msg import Header

        h = Header()
        h.frame_id = self._frame_id
        h.stamp = stamp
        return h

    def _points(self, ns, oid, pts, stamp, scale, rgba):
        from geometry_msgs.msg import Point
        from visualization_msgs.msg import Marker

        m = Marker()
        m.header = self._header(stamp)
        m.ns = ns
        m.id = oid
        m.action = Marker.ADD
        m.type = Marker.POINTS
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = scale
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        m.points = [Point(x=float(x), y=float(y), z=_Z) for x, y in pts]
        return m

    def _spheres(self, ns, oid, pts, stamp, scale, rgba):
        from geometry_msgs.msg import Point
        from visualization_msgs.msg import Marker

        m = Marker()
        m.header = self._header(stamp)
        m.ns = ns
        m.id = oid
        m.action = Marker.ADD
        m.type = Marker.SPHERE_LIST
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = scale
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        m.points = [Point(x=float(x), y=float(y), z=_Z + 0.05) for x, y in pts]
        return m

    def _line(self, ns, oid, pts, stamp, width, rgba):
        from geometry_msgs.msg import Point
        from visualization_msgs.msg import Marker

        m = Marker()
        m.header = self._header(stamp)
        m.ns = ns
        m.id = oid
        m.action = Marker.ADD
        m.type = Marker.LINE_STRIP
        m.pose.orientation.w = 1.0
        m.scale.x = width
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        m.points = [Point(x=float(x), y=float(y), z=_Z + 0.08) for x, y in pts]
        return m

    def _legend(self, text: str, stamp):
        from visualization_msgs.msg import Marker

        m = Marker()
        m.header = self._header(stamp)
        m.ns = "legend"
        m.id = 0
        m.action = Marker.ADD
        m.type = Marker.TEXT_VIEW_FACING
        m.pose.position.x = 0.0
        m.pose.position.y = 0.0
        m.pose.position.z = 2.2
        m.pose.orientation.w = 1.0
        m.scale.z = 0.22
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 1.0, 1.0, 0.95
        m.text = text
        return m

    def _delete(self, ns, oid, stamp):
        from visualization_msgs.msg import Marker

        m = Marker()
        m.header = self._header(stamp)
        m.ns = ns
        m.id = oid
        m.action = Marker.DELETE
        return m
