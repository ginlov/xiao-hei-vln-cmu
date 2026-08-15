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


def _slab(center, half_xyz, n=60, seed=0):
    """n points inside an axis-aligned box with per-axis half-extents.

    A thin z half-extent makes a "flat" box (carpet/rug), which the merge gate
    scores with 2D footprint IoU instead of the volumetric 3D IoU."""
    rng = np.random.default_rng(seed)
    c = np.asarray(center, float)
    h = np.asarray(half_xyz, float)
    return c + rng.uniform(-h, h, size=(n, 3))


# ── fusion behaviour ──────────────────────────────────────────────────────────

def test_second_view_of_one_object_does_not_grow_the_box():
    """Two views of one chair must describe a chair, not a chair-and-a-half.

    Each observation lands its own centre a little off, in whatever direction
    the robot happened to be standing. Pooling the point clouds adds that
    scatter to the object — measured against ground truth it made boxes ~3.5x
    too big by volume — so the node takes the median of the per-observation
    boxes instead of their union.
    """
    om = ObjectMap()
    om.add("chair", 0.8, _cube([0.0, 0.0, 0.0], half=0.25, seed=1))
    # second view, center 0.25 away → boxes overlap (gap 0) → merges.
    om.add("chair", 0.9, _cube([0.25, 0.0, 0.0], half=0.25, seed=2))
    nodes = om.to_list()
    assert len(nodes) == 1                       # fused, not duplicated
    node = nodes[0]
    assert node["n_obs"] == 2
    assert node["score"] == 0.9                  # follows the stronger observation
    # Each view is a 0.5 m cube, so the fused box stays about one cube wide —
    # the union would have spanned [-0.25, 0.5], i.e. 0.75 m.
    assert node["bbox_aabb"]["size"][0] < 0.6
    # ...and the box sits between the two views rather than spanning both.
    mid_x = (node["bbox_aabb"]["min"][0] + node["bbox_aabb"]["max"][0]) / 2
    assert 0.0 < mid_x < 0.25


def test_iou_2d_ignores_z_where_iou_3d_collapses():
    """A carpet box is ~0 m thick, so volumetric IoU is 0 even for two fully
    overlapping footprints; the 2D fallback recovers the real overlap."""
    from xiao_hei_vln.perception.object_map import iou_2d, iou_3d
    amin, amax = np.array([0.0, 0.0, 0.04]), np.array([2.0, 1.0, 0.04])
    bmin, bmax = np.array([1.0, 0.0, 0.04]), np.array([3.0, 1.0, 0.04])
    assert iou_3d(amin, amax, bmin, bmax) == 0.0          # zero thickness -> 0
    assert iou_2d(amin, amax, bmin, bmax) > 0.3           # footprints overlap


def test_flat_fragments_touching_merge_at_add():
    """The carpet case: two flat fragments whose footprints touch (surface gap
    ~0) fuse on the way in, where the old fixed centre-distance gate left them
    apart because their offset centroids sat >0.4 m apart."""
    om = ObjectMap()
    om.add("carpet", 0.8, _slab([-0.5, 0.0, 0.04], [0.5, 0.7, 0.01], seed=1))
    om.add("carpet", 0.7, _slab([0.5, 0.0, 0.04], [0.5, 0.7, 0.01], seed=2))
    assert len(om.nodes) == 1                             # merged, not two nodes


def test_size_scaled_gap_merges_small_fragment_into_large_flat_node():
    """A small fragment 0.1 m off a large carpet merges — the tolerance scales
    with the LARGER footprint, so a big object can absorb a nearby sliver."""
    om = ObjectMap()
    om.add("carpet", 0.9, _slab([0.0, 0.0, 0.04], [1.0, 0.7, 0.01], seed=1))
    om.add("carpet", 0.6, _slab([1.2, 0.0, 0.04], [0.1, 0.1, 0.01], seed=2))
    assert len(om.nodes) == 1


def test_size_scaled_gap_keeps_small_same_label_neighbours_apart():
    """Two small same-label objects with a real gap (two pillows side by side)
    stay separate — the tolerance is a fraction of their small footprint."""
    om = ObjectMap()
    om.add("pillow", 0.9, _slab([0.0, 0.0, 0.1], [0.15, 0.15, 0.15], seed=1))
    om.add("pillow", 0.8, _slab([0.5, 0.0, 0.1], [0.15, 0.15, 0.15], seed=2))
    assert len(om.nodes) == 2


def test_flat_node_box_takes_max_extent_not_mean():
    """A flat node sized under many redundant narrow views must keep the reach
    of the one wide view (max per-view extent), not shrink to their mean.

    Simulates the dwell case: one full-width look at the carpet, then five thin
    slices from a parked pose. The mean of the per-view widths would collapse
    the box; the flat estimator takes the max, recovering the object."""
    om = ObjectMap()
    om.add("carpet", 0.9, _slab([1.0, 0.5, 0.04], [1.0, 0.5, 0.01], n=200, seed=0))
    for s in range(5):                                   # redundant thin slices
        om.add("carpet", 0.8, _slab([1.0, 0.5, 0.04], [0.2, 0.5, 0.01], n=200, seed=s + 1))
    assert len(om.nodes) == 1
    dx = om.nodes[0].cmax[0] - om.nodes[0].cmin[0]
    assert dx > 1.5                                      # kept the wide view's ~2.0 m reach


def test_volumetric_node_still_uses_mean_extent():
    """The flat estimator must not touch volumetric objects: a tall (non-flat)
    node keeps the shrink-to-mean behaviour that cancels pooling inflation."""
    om = ObjectMap()
    om.add("chair", 0.8, _cube([0.0, 0.0, 0.0], half=0.25, seed=1))
    om.add("chair", 0.9, _cube([0.25, 0.0, 0.0], half=0.25, seed=2))
    dx = om.nodes[0].cmax[0] - om.nodes[0].cmin[0]
    assert dx < 0.6                                      # mean, not the 0.75 m union/max


def test_flat_node_centre_is_union_midpoint_not_weighted_centroid():
    """A flat node's centre must sit on the span it covers, not drift toward the
    part the robot dwelled on.

    Simulates partial reach: one wide view sees the whole carpet (x in [0, 2],
    true centre x=1.0), then five heavy dwell views see only the right half
    (centroid ~x=1.5). A point-weighted centroid would pull the centre to ~1.4;
    the union midpoint of the per-view boxes stays on the true centre."""
    om = ObjectMap()
    om.add("carpet", 0.9, _slab([1.0, 0.5, 0.04], [1.0, 0.5, 0.01], n=200, seed=0))
    for s in range(5):                                   # heavy dwell on the right half
        om.add("carpet", 0.8, _slab([1.5, 0.5, 0.04], [0.4, 0.5, 0.01], n=400, seed=s + 1))
    assert len(om.nodes) == 1
    cx = om.nodes[0].center[0]
    assert abs(cx - 1.0) < 0.2                           # on the span midpoint, not the ~1.4 dwell centroid


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


def test_add_returns_the_node_id_it_fused_into():
    """The id is what ties a 2D detection to its 3D box in the debug dumps.

    A merge must report the *existing* node, not a new one, or the overlay
    would print a number that appears nowhere in the scene.
    """
    om = ObjectMap()
    first = om.add("chair", 0.8, _cube([0.0, 0.0, 0.0]))
    other = om.add("table", 0.9, _cube([5.0, 0.0, 0.0]))
    merged = om.add("chair", 0.85, _cube([0.1, 0.0, 0.0]))

    assert first is not None and other != first
    assert merged == first                       # merged, so the id is reused
    assert {n["node_id"] for n in om.to_list()} == {first, other}


def test_add_returns_none_for_an_empty_cloud():
    # Nothing was recorded, so there is no node to point at.
    assert ObjectMap().add("chair", 0.8, np.empty((0, 3))) is None


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


# ---------------------------------------------------------------------------
# Same-label gap/distance suppression (NMS_DIST / NMS_GAP)
# ---------------------------------------------------------------------------
#
# These pass `merge_gap_frac=0.0` to isolate the finalize-time rule: with the
# gap gate off, only exactly-overlapping boxes merge at add() (these pairs have
# a positive surface gap, so they don't). The finalize rule only earns its keep
# on nodes whose surfaces DRIFT together after creation, which cannot be
# constructed in a couple of calls.


def _pair(label_a, label_b, sep, **kw):
    om = ObjectMap(merge_gap_frac=0.0, **kw)       # no merging at add() time
    om.add(label_a, 0.9, _cube([0.0, 0.0, 0.0], half=0.08, n=60))
    om.add(label_b, 0.7, _cube([sep, 0.0, 0.0], half=0.08, n=40, seed=1))
    return om


def test_same_label_fragments_collapse_when_iou_is_zero():
    """Two tight boxes of one object, adjacent but not overlapping — IoU is
    exactly 0, so only the gap/distance path can catch them. Cubes are 0.16 m
    wide and 0.20 m apart, leaving a 0.04 m surface gap (under NMS_GAP)."""
    om = _pair("sofa", "sofa", 0.20)
    assert len(om.nodes) == 2                      # add() left them separate
    out = om.export(min_pts=5)
    assert len(out) == 1 and out[0]["label"] == "sofa"


def test_cross_label_pair_is_left_alone():
    """Same geometry, different labels: deferred to B3, must NOT be suppressed
    (a pillow genuinely touching a sofa is two objects, not one)."""
    out = _pair("sofa", "pillow", 0.20).export(min_pts=5)
    assert sorted(o["label"] for o in out) == ["pillow", "sofa"]


def test_distant_same_label_nodes_are_kept():
    """Beyond NMS_DIST the two really are different objects."""
    assert len(_pair("chair", "chair", 2.0).export(min_pts=5)) == 2


def test_gap_guard_blocks_centres_that_are_close_but_surfaces_that_are_not():
    """Centres 0.30 m apart is inside NMS_DIST, but the 0.16 m cubes leave a
    0.14 m surface gap — beyond NMS_GAP, so these stay two objects. This is
    the guard that stops the distance term collapsing genuine neighbours."""
    om = _pair("chair", "chair", 0.30)
    assert len(om.export(min_pts=5)) == 2


def test_export_does_not_mutate_the_live_map():
    """export() runs suppression on a throwaway view sharing the same _Node
    objects; the live map must keep both, and repeat calls must agree."""
    om = _pair("lamp", "lamp", 0.20)
    first = om.export(min_pts=5)
    assert len(om.nodes) == 2                      # live map untouched
    assert om.export(min_pts=5) == first           # idempotent
