"""Publish the perception scene graph as rviz markers (+ optional scoreboard).

The online perception responder keeps its object map in-process (it answers
challenge questions, it does not draw anything). This helper turns the
current scene graph into a ``visualization_msgs/MarkerArray`` on
``/perception/objects`` so the fused 3D boxes + labels can be watched live
in rviz while driving — same view the offline prototype gave, but backed by
the online pipeline (multi-frame accumulation + occlusion + fusion).

Each object's box/label markers live in a **per-class namespace**
(``box/<label>``), so rviz's MarkerArray "Namespaces" list gives one
checkbox per class — tick classes on/off directly, no extra topic.

:class:`Scoreboard` (dev only, needs ground truth) scores the live scene
graph against the scene's ``object_list.txt`` each cycle and renders the
metrics as a fixed text marker.

ROS message types are imported lazily so importing this module off-robot
(e.g. under the test suite) never requires an rclpy install.
"""

from __future__ import annotations

import colorsys
import logging
import zlib
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Where the floating scoreboard text sits in the map frame (metres).
SCOREBOARD_POS = (0.0, 0.0, 2.5)


def _color_for_label(label: str) -> tuple[float, float, float]:
    """A vivid, stable, per-class colour in 0-1 RGB.

    The masked-pixel median colour is muddy (walls/floor sample to near-grey
    and read as indistinct boxes), so we colour by class instead: a stable
    hash of the label → hue, at high saturation/value. Same label always maps
    to the same colour within and across runs (crc32, not the salted
    ``hash()``), so a class keeps its colour as the map grows.
    """
    hue = (zlib.crc32(label.encode("utf-8")) % 360) / 360.0
    return colorsys.hsv_to_rgb(hue, 0.65, 0.95)


class Scoreboard:
    """Live perception score vs the scene's authoritative GT (dev only).

    Loads ``object_list.txt`` once; :meth:`text` scores the current scene
    graph and returns a short multi-line summary for a text marker. Ground
    truth is NOT available at test time — this is a development aid.
    """

    def __init__(self, object_list_path: str | Path) -> None:
        from xiao_hei_vln.eval_sampler.object_list import parse_object_list
        from xiao_hei_vln.perception.eval import object_entries_to_eval

        lines = Path(object_list_path).read_text().splitlines()
        self._gt = object_entries_to_eval(parse_object_list(lines))
        self._n_gt = len(self._gt)

    def text(self, scene_dict: dict) -> str:
        from xiao_hei_vln.perception.eval import evaluate, scene_objects_to_eval

        preds = scene_objects_to_eval(scene_dict.get("objects", []))
        rep, _ = evaluate(self._gt, preds, [0.5, 1.0, 2.0], [0.25])
        m = rep["mAP"]
        op = rep["operating_point"]["dist@1.0m"]
        cerr = op.get("mean_center_err_m")
        cerr_s = f"{cerr:.2f}m" if cerr is not None else "-"
        return (
            "PERCEPTION vs GT (dev)\n"
            f"objects {rep['n_pred']} / {self._n_gt} GT\n"
            f"mAP  d0.5 {m['dist@0.5m']:.2f}  d1.0 {m['dist@1.0m']:.2f}  "
            f"d2.0 {m['dist@2.0m']:.2f}\n"
            f"count MAE {rep['counting_MAE']:.2f}   center {cerr_s}"
        )


class ScenePublisher:
    """Builds + publishes MarkerArray snapshots of the scene graph."""

    def __init__(
        self,
        node: Any,
        *,
        topic: str = "/perception/objects",
        frame_id: str = "map",
    ) -> None:
        from visualization_msgs.msg import MarkerArray

        self._node = node
        self._frame_id = frame_id
        self._pub = node.create_publisher(MarkerArray, topic, 10)
        # (namespace, id) of every marker published last cycle, so vanished
        # objects (pruned/merged) get an explicit DELETE in the right ns.
        self._prev: set[tuple[str, int]] = set()

    def publish(self, scene_dict: dict, scoreboard_text: str | None = None) -> None:
        from visualization_msgs.msg import MarkerArray

        stamp = self._node.get_clock().now().to_msg()
        arr = MarkerArray()
        cur: set[tuple[str, int]] = set()

        for obj in scene_dict.get("objects", []):
            oid = int(obj.get("object_id", 0))
            label = obj.get("label", "?")
            box_ns, label_ns = f"box/{label}", f"label/{label}"
            arr.markers.append(self._box_marker(obj, box_ns, oid, stamp))
            arr.markers.append(self._label_marker(obj, label_ns, oid, stamp))
            cur.add((box_ns, oid))
            cur.add((label_ns, oid))

        if scoreboard_text is not None:
            arr.markers.append(self._scoreboard_marker(scoreboard_text, stamp))
            cur.add(("scoreboard", 0))

        for ns, oid in self._prev - cur:
            arr.markers.append(self._delete_marker(ns, oid, stamp))
        self._prev = cur

        self._pub.publish(arr)

    # ------------------------------------------------------------------

    def _header(self, stamp):
        from std_msgs.msg import Header

        h = Header()
        h.frame_id = self._frame_id
        h.stamp = stamp
        return h

    def _box_marker(self, obj: dict, ns: str, oid: int, stamp):
        from visualization_msgs.msg import Marker

        m = Marker()
        m.header = self._header(stamp)
        m.ns = ns
        m.id = oid
        m.action = Marker.ADD

        bmin = obj.get("bbox_min")
        bmax = obj.get("bbox_max")
        pos = obj.get("position") or [0.0, 0.0, 0.0]
        if bmin is not None and bmax is not None:
            cx, cy, cz = (
                (bmin[0] + bmax[0]) / 2.0,
                (bmin[1] + bmax[1]) / 2.0,
                (bmin[2] + bmax[2]) / 2.0,
            )
            sx = max(abs(bmax[0] - bmin[0]), 0.05)
            sy = max(abs(bmax[1] - bmin[1]), 0.05)
            sz = max(abs(bmax[2] - bmin[2]), 0.05)
        else:
            cx, cy, cz = pos[0], pos[1], pos[2]
            sx = sy = sz = 0.2

        m.type = Marker.CUBE
        m.pose.position.x, m.pose.position.y, m.pose.position.z = cx, cy, cz
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = sx, sy, sz

        r, g, b = _color_for_label(obj.get("label", "?"))
        m.color.r, m.color.g, m.color.b = r, g, b
        # Low alpha so a big room-spanning box stays see-through — the robot
        # and the small-object boxes inside it remain visible.
        m.color.a = 0.3
        return m

    def _label_marker(self, obj: dict, ns: str, oid: int, stamp):
        from visualization_msgs.msg import Marker

        m = Marker()
        m.header = self._header(stamp)
        m.ns = ns
        m.id = oid
        m.action = Marker.ADD
        m.type = Marker.TEXT_VIEW_FACING

        bmax = obj.get("bbox_max")
        pos = obj.get("position") or [0.0, 0.0, 0.0]
        top_z = bmax[2] if bmax is not None else pos[2]
        m.pose.position.x = pos[0]
        m.pose.position.y = pos[1]
        m.pose.position.z = top_z + 0.15
        m.pose.orientation.w = 1.0
        m.scale.z = 0.2
        m.color.r = m.color.g = m.color.b = 1.0
        m.color.a = 0.9
        n_obs = obj.get("n_obs")
        base = obj.get("label", "?")
        m.text = base if n_obs is None else f"{base} ({n_obs})"
        return m

    def _scoreboard_marker(self, text: str, stamp):
        from visualization_msgs.msg import Marker

        m = Marker()
        m.header = self._header(stamp)
        m.ns = "scoreboard"
        m.id = 0
        m.action = Marker.ADD
        m.type = Marker.TEXT_VIEW_FACING
        m.pose.position.x, m.pose.position.y, m.pose.position.z = SCOREBOARD_POS
        m.pose.orientation.w = 1.0
        m.scale.z = 0.28
        m.color.r, m.color.g, m.color.b = 1.0, 0.95, 0.3
        m.color.a = 1.0
        m.text = text
        return m

    def _delete_marker(self, ns: str, oid: int, stamp):
        from visualization_msgs.msg import Marker

        m = Marker()
        m.header = self._header(stamp)
        m.ns = ns
        m.id = oid
        m.action = Marker.DELETE
        return m
