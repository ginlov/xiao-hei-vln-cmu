"""Tests for SceneRepresentation — see TASK 10 test plan."""

from __future__ import annotations

import math

import numpy as np
import pytest

from xiao_hei_vln.messages import (
    Header,
    ImageFrame,
    OdomPose,
    Quaternion,
    Stamp,
    Vector3,
    VLMInput,
)
from xiao_hei_vln.messages.sensors import LidarScan
from xiao_hei_vln.scene import SceneRepresentation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stamp() -> Stamp:
    return Stamp(sec=0, nanosec=0)


def _header(frame: str = "map") -> Header:
    return Header(stamp=_stamp(), frame_id=frame)


def _pose(x: float = 0.0, y: float = 0.0, z: float = 0.0, yaw: float = 0.0) -> OdomPose:
    hz = math.sin(yaw / 2)
    hw = math.cos(yaw / 2)
    return OdomPose(
        header=_header(),
        position=Vector3(x=x, y=y, z=z),
        orientation=Quaternion(x=0.0, y=0.0, z=hz, w=hw),
    )


def _image() -> ImageFrame:
    return ImageFrame(
        header=_header("camera"),
        width=2,
        height=1,
        encoding="bgr8",
        step=6,
        data=b"\x00" * 6,
    )


def _scan(points: list[tuple[float, float, float, float]]) -> LidarScan:
    arr = np.array(points, dtype=np.float32).reshape(-1, 4)
    return LidarScan(header=_header(), points=arr, source="registered")


def _snapshot(
    tick_id: int = 0,
    pose: OdomPose | None = None,
    image: ImageFrame | None = None,
    scan: LidarScan | None = None,
) -> VLMInput:
    return VLMInput(
        tick_id=tick_id,
        tick_time=_stamp(),
        pose=pose,
        image=image,
        registered_scan=scan,
    )


def _node(
    node_id: int,
    label: str,
    x: float,
    y: float,
    score: float = 1.0,
    half: float = 0.25,
) -> dict:
    """One entry in the shape `ObjectMap.export()` emits.

    Objects only reach the graph through `sync_from_object_map`, whose
    contract is this dict — so the scene-rep tests state it literally rather
    than driving a real ObjectMap. The fusion itself is test_object_map.py's
    subject; here we test what the scene does with the result.
    """
    return {
        "node_id": node_id,
        "label": label,
        "score": score,
        "center_3d": [x, y, 0.0],
        "bbox_aabb": {
            "min": [x - half, y - half, -half],
            "max": [x + half, y + half, half],
        },
        "color_rgb": None,
        "color_name": None,
    }


# ---------------------------------------------------------------------------
# 1. ViewpointNode creation
# ---------------------------------------------------------------------------


class TestViewpointCreation:
    def test_first_update_creates_one_viewpoint(self) -> None:
        rep = SceneRepresentation()
        rep.update(_snapshot(tick_id=1, pose=_pose(0, 0)))
        assert len(rep.viewpoints) == 1
        assert rep.viewpoints[0].tick_id == 1

    def test_close_second_update_no_new_viewpoint(self) -> None:
        rep = SceneRepresentation(viewpoint_radius=2.0)
        rep.update(_snapshot(pose=_pose(0, 0)))
        rep.update(_snapshot(pose=_pose(1, 0)))  # 1 m — under threshold
        assert len(rep.viewpoints) == 1

    def test_far_second_update_adds_viewpoint(self) -> None:
        rep = SceneRepresentation(viewpoint_radius=2.0)
        rep.update(_snapshot(pose=_pose(0, 0)))
        rep.update(_snapshot(tick_id=5, pose=_pose(3, 0)))  # 3 m — over threshold
        assert len(rep.viewpoints) == 2

    def test_stored_position_and_yaw(self) -> None:
        rep = SceneRepresentation()
        rep.update(_snapshot(pose=_pose(1.5, 2.5, 0.75, yaw=math.pi / 2)))
        vp = rep.viewpoints[0]
        assert vp.position.x == pytest.approx(1.5)
        assert vp.position.y == pytest.approx(2.5)
        assert vp.yaw == pytest.approx(math.pi / 2, abs=1e-5)

    def test_no_pose_no_viewpoint(self) -> None:
        rep = SceneRepresentation()
        rep.update(_snapshot(pose=None))
        assert len(rep.viewpoints) == 0


# ---------------------------------------------------------------------------
# 2. RoomNode best image
# ---------------------------------------------------------------------------


class TestRoomBestImage:
    def test_initialises_with_no_image(self) -> None:
        rep = SceneRepresentation()
        assert rep.room.best_image is None
        assert rep.room.best_image_tick_id is None

    def test_updated_when_new_viewpoint_created(self) -> None:
        rep = SceneRepresentation()
        img = _image()
        rep.update(_snapshot(tick_id=3, pose=_pose(0, 0), image=img))
        assert rep.room.best_image is img
        assert rep.room.best_image_tick_id == 3
        assert rep.room.best_image_position == Vector3(x=0, y=0, z=0)

    def test_not_updated_when_no_new_viewpoint(self) -> None:
        rep = SceneRepresentation(viewpoint_radius=2.0)
        img1 = _image()
        rep.update(_snapshot(tick_id=1, pose=_pose(0, 0), image=img1))
        img2 = _image()
        rep.update(_snapshot(tick_id=2, pose=_pose(0.5, 0), image=img2))  # too close
        assert rep.room.best_image is img1
        assert rep.room.best_image_tick_id == 1

    def test_updated_at_each_new_viewpoint(self) -> None:
        rep = SceneRepresentation(viewpoint_radius=2.0)
        rep.update(_snapshot(tick_id=1, pose=_pose(0, 0), image=_image()))
        img2 = _image()
        rep.update(_snapshot(tick_id=5, pose=_pose(3, 0), image=img2))
        assert rep.room.best_image is img2
        assert rep.room.best_image_tick_id == 5
        assert rep.room.best_image_position.x == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 3. Object layer replacement
#
# Same-label merging is no longer the scene rep's job — detections are fused
# in ObjectMap (see test_object_map.py) and the whole layer is replaced from
# its export. What is tested here is the replacement contract itself.
# ---------------------------------------------------------------------------


class TestObjectLayerSync:
    def test_sync_into_empty_scene_creates_nodes(self) -> None:
        rep = SceneRepresentation()
        rep.sync_from_object_map([_node(0, "chair", 1.0, 0.0)])
        assert len(rep.objects) == 1
        assert rep.objects[0].label == "chair"

    def test_sync_replaces_rather_than_appends(self) -> None:
        # The same node re-synced must not duplicate: the layer is rebuilt,
        # not accumulated.
        rep = SceneRepresentation()
        rep.sync_from_object_map([_node(0, "chair", 1.0, 0.0)])
        rep.sync_from_object_map([_node(0, "chair", 1.2, 0.0)])
        assert len(rep.objects) == 1
        assert rep.objects[0].position.x == pytest.approx(1.2)

    def test_node_dropped_by_fusion_leaves_the_scene(self) -> None:
        # export() runs NMS/prune, so a node can vanish between ticks. The
        # scene must follow it rather than keep a stale object alive.
        rep = SceneRepresentation()
        rep.sync_from_object_map([_node(0, "chair", 1.0, 0.0),
                                  _node(1, "table", 5.0, 0.0)])
        assert len(rep.objects) == 2
        rep.sync_from_object_map([_node(0, "chair", 1.0, 0.0)])
        assert [o.label for o in rep.objects] == ["chair"]

    def test_boxes_are_always_populated(self) -> None:
        rep = SceneRepresentation()
        rep.sync_from_object_map([_node(0, "chair", 1.0, 0.0, half=0.5)])
        o = rep.objects[0]
        assert o.bbox_min is not None and o.bbox_max is not None
        assert o.bbox_max.x - o.bbox_min.x == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3b. Stable object_id assignment
# ---------------------------------------------------------------------------


class TestObjectIds:
    def test_new_objects_get_distinct_monotonic_ids(self) -> None:
        rep = SceneRepresentation()
        rep.sync_from_object_map([
            _node(0, "chair", 0.0, 0.0),
            _node(1, "table", 5.0, 0.0),
            _node(2, "lamp", 10.0, 0.0),
        ])
        ids = [o.object_id for o in rep.objects]
        assert ids == sorted(set(ids))  # strictly monotonic, no dupes
        assert all(i > 0 for i in ids)  # 0 reserved as "unassigned" sentinel

    def test_node_id_keeps_its_object_id_across_ticks(self) -> None:
        rep = SceneRepresentation()
        rep.sync_from_object_map([_node(0, "chair", 1.0, 0.0)])
        first_id = rep.objects[0].object_id
        # The node accumulated points and its centre moved; same node_id.
        rep.sync_from_object_map([_node(0, "chair", 1.2, 0.0)])
        assert rep.objects[0].object_id == first_id

    def test_id_counter_does_not_reuse_after_a_node_reappears(self) -> None:
        rep = SceneRepresentation()
        rep.sync_from_object_map([_node(0, "chair", 1.0, 0.0)])
        rep.sync_from_object_map([])                       # suppressed this tick
        rep.sync_from_object_map([_node(0, "chair", 1.0, 0.0),
                                  _node(1, "table", 5.0, 0.0)])
        chair = next(o for o in rep.objects if o.label == "chair")
        table = next(o for o in rep.objects if o.label == "table")
        assert chair.object_id == 1     # node_id 0 kept its original id
        assert table.object_id == 2     # counter advanced exactly once

    def test_first_tick_id_survives_a_resync(self) -> None:
        rep = SceneRepresentation()
        rep._tick_id = 4
        rep.sync_from_object_map([_node(0, "chair", 1.0, 0.0)])
        rep._tick_id = 9
        rep.sync_from_object_map([_node(0, "chair", 1.0, 0.0)])
        assert rep.objects[0].first_tick_id == 4
        assert rep.objects[0].last_tick_id == 9


# ---------------------------------------------------------------------------
# 3c. SpatialRelation.target_object_id is populated
# ---------------------------------------------------------------------------


class TestYawExtraction:
    def test_identity_gives_zero_yaw(self) -> None:
        rep = SceneRepresentation()
        rep.update(_snapshot(pose=_pose(0, 0, 0, yaw=0.0)))
        assert rep.viewpoints[0].yaw == pytest.approx(0.0, abs=1e-6)

    def test_90_degree_yaw(self) -> None:
        rep = SceneRepresentation()
        rep.update(_snapshot(pose=_pose(0, 0, 0, yaw=math.pi / 2)))
        assert rep.viewpoints[0].yaw == pytest.approx(math.pi / 2, abs=1e-5)

    def test_negative_90_degree_yaw(self) -> None:
        rep = SceneRepresentation()
        rep.update(_snapshot(pose=_pose(0, 0, 0, yaw=-math.pi / 2)))
        assert rep.viewpoints[0].yaw == pytest.approx(-math.pi / 2, abs=1e-5)


# ---------------------------------------------------------------------------
# 6. scene_bounds from registered scan
# ---------------------------------------------------------------------------


class TestSceneBounds:
    def test_no_scan_bounds_none(self) -> None:
        rep = SceneRepresentation()
        rep.update(_snapshot())
        assert rep.room.scene_bounds is None

    def test_bounds_computed_from_scan(self) -> None:
        rep = SceneRepresentation()
        rep.update(_snapshot(scan=_scan([(-1, -2, 0, 0), (3, 4, 1, 0)])))
        mn, mx = rep.room.scene_bounds
        assert mn.x == pytest.approx(-1.0)
        assert mn.y == pytest.approx(-2.0)
        assert mx.x == pytest.approx(3.0)
        assert mx.y == pytest.approx(4.0)

    def test_bounds_updated_on_each_tick(self) -> None:
        rep = SceneRepresentation()
        rep.update(_snapshot(scan=_scan([(0, 0, 0, 0), (1, 1, 0, 0)])))
        rep.update(_snapshot(scan=_scan([(-5, -5, 0, 0), (5, 5, 0, 0)])))
        mn, mx = rep.room.scene_bounds
        assert mn.x == pytest.approx(-5.0)
        assert mx.x == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 7. to_dict serialisation
# ---------------------------------------------------------------------------


class TestEdgeStorage:
    # Room → Viewpoint
    def test_room_viewpoint_ids_empty_on_init(self) -> None:
        rep = SceneRepresentation()
        assert rep.room.viewpoint_tick_ids == []

    def test_room_viewpoint_ids_appended_on_new_viewpoint(self) -> None:
        rep = SceneRepresentation(viewpoint_radius=2.0)
        rep.update(_snapshot(tick_id=3, pose=_pose(0, 0)))
        rep.update(_snapshot(tick_id=7, pose=_pose(5, 0)))
        assert rep.room.viewpoint_tick_ids == [3, 7]

    def test_room_viewpoint_ids_not_appended_when_too_close(self) -> None:
        rep = SceneRepresentation(viewpoint_radius=2.0)
        rep.update(_snapshot(tick_id=1, pose=_pose(0, 0)))
        rep.update(_snapshot(tick_id=2, pose=_pose(0.5, 0)))  # too close
        assert rep.room.viewpoint_tick_ids == [1]

    # Viewpoint → Object (observing_viewpoint_ids)
    #
    # These edges are stamped by sync_from_object_map, which is the only path
    # objects take into the graph. Everything below drives that method.
    def test_observing_viewpoint_appended_on_creation(self) -> None:
        # Need a viewpoint to exist first; otherwise no edge is recorded.
        rep = SceneRepresentation()
        rep.update(_snapshot(tick_id=5, pose=_pose(0, 0)))
        rep.sync_from_object_map([_node(0, "chair", 1.0, 0.0)])
        assert rep.objects[0].observing_viewpoint_ids == [5]

    def test_observing_viewpoint_empty_when_no_viewpoint_yet(self) -> None:
        # Object reported before the first viewpoint exists → no edge.
        # ``first_tick_id`` still tracks the raw observation tick.
        rep = SceneRepresentation()
        rep._tick_id = 5
        rep.sync_from_object_map([_node(0, "chair", 1.0, 0.0)])
        assert rep.objects[0].observing_viewpoint_ids == []
        assert rep.objects[0].first_tick_id == 5

    def test_observing_viewpoint_accumulates_across_viewpoints(self) -> None:
        # The same fused node re-synced from a second viewpoint picks up that
        # viewpoint id without losing the first.
        rep = SceneRepresentation(viewpoint_radius=2.0)
        rep.update(_snapshot(tick_id=3, pose=_pose(0, 0)))
        rep.sync_from_object_map([_node(0, "chair", 1.0, 0.0)])
        rep.update(_snapshot(tick_id=7, pose=_pose(5, 0)))  # far → new vp
        rep.sync_from_object_map([_node(0, "chair", 1.2, 0.0)])
        assert rep.objects[0].observing_viewpoint_ids == [3, 7]

    def test_observing_viewpoint_dedupes_within_same_viewpoint(self) -> None:
        # Every tick re-syncs the whole layer, so without the dedupe guard a
        # stationary robot would append the same viewpoint id forever.
        rep = SceneRepresentation(viewpoint_radius=2.0)
        rep.update(_snapshot(tick_id=3, pose=_pose(0, 0)))
        rep.sync_from_object_map([_node(0, "chair", 1.0, 0.0)])
        rep.update(_snapshot(tick_id=4, pose=_pose(0.2, 0)))  # same vp
        rep.sync_from_object_map([_node(0, "chair", 1.05, 0.0)])
        rep.update(_snapshot(tick_id=5, pose=_pose(0.4, 0)))  # same vp
        rep.sync_from_object_map([_node(0, "chair", 1.1, 0.0)])
        assert rep.objects[0].observing_viewpoint_ids == [3]

    def test_observing_viewpoint_edges_are_dropped_with_the_node(self) -> None:
        # A node suppressed by fusion takes its edges with it; a later node
        # reusing that node_id starts from the ids it is seen from now.
        rep = SceneRepresentation(viewpoint_radius=2.0)
        rep.update(_snapshot(tick_id=3, pose=_pose(0, 0)))
        rep.sync_from_object_map([_node(0, "chair", 1.0, 0.0)])
        rep.sync_from_object_map([])                        # suppressed
        rep.update(_snapshot(tick_id=7, pose=_pose(5, 0)))  # new vp
        rep.sync_from_object_map([_node(0, "chair", 1.0, 0.0)])
        assert rep.objects[0].observing_viewpoint_ids == [7]


class TestToDict:
    def test_empty_scene_serialises(self) -> None:
        import json

        rep = SceneRepresentation()
        d = rep.to_dict()
        assert json.dumps(d)  # must be JSON-serialisable
        assert d["viewpoints"] == []
        assert d["objects"] == []
        assert d["room"]["scene_bounds"] is None

    def test_populated_scene_round_trips(self) -> None:
        import json

        rep = SceneRepresentation()
        rep.update(_snapshot(tick_id=3, pose=_pose(1.0, 2.0)))
        rep.update(_snapshot(
            tick_id=7, pose=_pose(5.0, 0.0),
            scan=_scan([(-1, -1, 0, 0), (5, 5, 0, 0)]),
        ))
        rep.sync_from_object_map([
            _node(0, "chair", 1.5, 2.0),
            _node(1, "table", 4.5, 0.0),
        ])

        d = rep.to_dict()
        json.dumps(d)  # JSON-clean

        assert len(d["viewpoints"]) == 2
        assert d["viewpoints"][0]["tick_id"] == 3
        assert d["viewpoints"][0]["position"] == [1.0, 2.0, 0.0]
        assert d["room"]["scene_bounds"] == [[-1.0, -1.0, 0.0], [5.0, 5.0, 0.0]]
        assert d["room"]["viewpoint_tick_ids"] == [3, 7]

        labels = {o["label"] for o in d["objects"]}
        assert labels == {"chair", "table"}
        chair = next(o for o in d["objects"] if o["label"] == "chair")
        table = next(o for o in d["objects"] if o["label"] == "table")
        assert "object_id" in chair and chair["object_id"] > 0
        assert "object_id" in table and table["object_id"] > 0
