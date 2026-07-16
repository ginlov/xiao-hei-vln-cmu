"""Tests for xiao_hei_vln.perception.object_map and the scene sync path."""

from __future__ import annotations

import numpy as np

from xiao_hei_vln.perception.object_map import ObjectMap
from xiao_hei_vln.scene import SceneRepresentation


def _cube(center, half=0.25, n=30, seed=0):
    """n points uniformly inside an axis-aligned cube around center."""
    rng = np.random.default_rng(seed)
    c = np.asarray(center, float)
    return c + rng.uniform(-half, half, size=(n, 3))


# ── fusion behaviour ──────────────────────────────────────────────────────────

def test_same_label_union_grows_the_box():
    om = ObjectMap()
    om.add("chair", 0.8, _cube([0.0, 0.0, 0.0], half=0.25, seed=1))
    # second view, center 0.25 away (< MERGE_DIST) → merges, box unions.
    om.add("chair", 0.9, _cube([0.25, 0.0, 0.0], half=0.25, seed=2))
    nodes = om.to_list()
    assert len(nodes) == 1                       # fused, not duplicated
    node = nodes[0]
    assert node["n_obs"] == 2
    assert node["score"] == 0.9                  # follows the stronger observation
    # union spans roughly [-0.25, 0.5] in x → clearly wider than one cube (0.5).
    assert node["bbox_aabb"]["size"][0] > 0.6


def test_different_labels_do_not_merge():
    om = ObjectMap()
    pts = _cube([0.0, 0.0, 0.0])
    om.add("chair", 0.8, pts)
    om.add("table", 0.8, _cube([0.1, 0.0, 0.0]))
    assert len(om.to_list()) == 2


def test_cross_label_nms_drops_the_weaker_overlap():
    om = ObjectMap()
    pts = _cube([0.0, 0.0, 0.0], half=0.3, n=30, seed=3)
    om.add("cabinet", 0.9, pts)          # same box, conflicting labels
    om.add("shelf", 0.5, pts.copy())
    exported = om.export(min_pts=15)     # export runs finalize() (NMS) + prune()
    assert len(exported) == 1
    assert exported[0]["label"] == "cabinet"   # higher score survives


def test_wall_sheet_phantom_is_pruned():
    om = ObjectMap()
    rng = np.random.default_rng(4)
    # A 3 m (x) × 2 m (z) vertical plane, ~planar in y (thickness < 0.12),
    # normal ≈ +y (horizontal) → a bare-wall "sheet", not a real picture.
    x = rng.uniform(-1.5, 1.5, size=40)
    z = rng.uniform(0.0, 2.0, size=40)
    y = rng.uniform(-0.02, 0.02, size=40)
    sheet = np.column_stack([x, y, z])
    om.add("picture", 0.9, sheet)
    assert len(om.to_list()) == 1                 # present before prune
    assert om.export(min_pts=15) == []            # rejected as a wall sheet


def test_node_ids_are_assigned_and_stable():
    om = ObjectMap()
    om.add("chair", 0.8, _cube([0.0, 0.0, 0.0]))
    om.add("table", 0.8, _cube([5.0, 0.0, 0.0]))
    ids = [n["node_id"] for n in om.to_list()]
    assert ids == [0, 1]
    # merging into an existing node keeps its id.
    om.add("chair", 0.9, _cube([0.1, 0.0, 0.0]))
    chair = next(n for n in om.to_list() if n["label"] == "chair")
    assert chair["node_id"] == 0 and chair["n_obs"] == 2


# ── scene sync ────────────────────────────────────────────────────────────────

def test_sync_populates_boxes_and_colors():
    om = ObjectMap()
    om.add("chair", 0.8, _cube([1.0, 2.0, 0.0]),
           color_rgb=(120, 80, 40), color_name="brown")
    scene = SceneRepresentation()
    scene.sync_from_object_map(om.export(min_pts=15))
    objs = scene.objects
    assert len(objs) == 1
    o = objs[0]
    assert o.label == "chair"
    assert o.bbox_min is not None and o.bbox_max is not None   # real 3D box
    assert o.color_rgb == (120, 80, 40) and o.color_name == "brown"
    assert o.object_id >= 1
    # center within the cube.
    assert abs(o.position.x - 1.0) < 0.3 and abs(o.position.y - 2.0) < 0.3


def test_sync_keeps_object_id_stable_across_ticks():
    om = ObjectMap()
    om.add("chair", 0.8, _cube([0.0, 0.0, 0.0]))
    scene = SceneRepresentation()
    scene.sync_from_object_map(om.export(min_pts=15))
    oid = scene.objects[0].object_id
    # another observation of the same physical chair, then re-sync.
    om.add("chair", 0.85, _cube([0.15, 0.0, 0.0]))
    scene.sync_from_object_map(om.export(min_pts=15))
    assert len(scene.objects) == 1
    assert scene.objects[0].object_id == oid          # identity preserved
