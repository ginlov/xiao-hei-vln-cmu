"""Tests for xiao_hei_vln.perception.eval — the offline scene-graph scorer."""

from __future__ import annotations

import pytest

from xiao_hei_vln.eval_sampler.object_list import ObjectEntry
from xiao_hei_vln.messages.common import Vector3
from xiao_hei_vln.perception import eval as pe


def _obj(label, c, score=1.0, box=None):
    o = {"label": label, "score": score, "center_3d": list(c)}
    if box is None:
        o["bbox_aabb"] = {"min": list(c), "max": list(c), "size": [0, 0, 0]}
    else:
        lo, hi = box
        o["bbox_aabb"] = {"min": list(lo), "max": list(hi),
                          "size": [hi[i] - lo[i] for i in range(3)]}
    return o


# ── metrics ───────────────────────────────────────────────────────────────────

def test_perfect_match_gives_full_scores():
    gt = [_obj("chair", (0, 0, 0)), _obj("chair", (5, 0, 0)), _obj("table", (0, 5, 0))]
    pred = [_obj("chair", (0, 0, 0)), _obj("chair", (5, 0, 0)), _obj("table", (0, 5, 0))]
    report, primary = pe.evaluate(gt, pred, [0.5, 1.0, 2.0], [0.25])
    assert report["n_gt"] == 3 and report["n_pred"] == 3
    op = report["operating_point"][f"dist@{primary}m"]
    assert op["tp"] == 3 and op["fp"] == 0 and op["fn"] == 0
    assert op["precision"] == 1.0 and op["recall"] == 1.0 and op["f1"] == 1.0
    assert report["mAP"]["dist@1.0m"] == 1.0
    assert report["counting_MAE"] == 0.0 and report["counting_exact_frac"] == 1.0


def test_miss_and_false_positive():
    gt = [_obj("chair", (0, 0, 0)), _obj("chair", (5, 0, 0)), _obj("table", (0, 5, 0))]
    # chair@0 matched; chair@5 offset 0.3m (within 1m); table missing; sofa is a FP.
    pred = [_obj("chair", (0, 0, 0)), _obj("chair", (5.3, 0, 0)), _obj("sofa", (9, 9, 0))]
    report, primary = pe.evaluate(gt, pred, [0.5, 1.0, 2.0], [0.25])
    op = report["operating_point"][f"dist@{primary}m"]
    assert op["tp"] == 2          # both chairs
    assert op["fp"] == 1          # sofa
    assert op["fn"] == 1          # table
    assert op["precision"] == round(2 / 3, 4)
    assert op["recall"] == round(2 / 3, 4)
    # counting: table under-counted by 1, sofa over-counted by 1 -> MAE over 3 GT classes
    assert report["counting"]["table"]["err"] == -1
    assert report["counting"]["sofa"]["err"] == +1
    assert report["counting"]["chair"]["err"] == 0


def test_distance_threshold_tightening_drops_match():
    gt = [_obj("chair", (0, 0, 0))]
    pred = [_obj("chair", (0.7, 0, 0))]   # 0.7 m away
    report, _ = pe.evaluate(gt, pred, [0.5, 1.0], [0.25])
    assert report["operating_point"]["dist@0.5m"]["tp"] == 0   # too far at 0.5
    assert report["operating_point"]["dist@1.0m"]["tp"] == 1   # ok at 1.0


def test_label_confusion_reported():
    gt = [_obj("couch", (0, 0, 0))]
    pred = [_obj("sofa", (0.1, 0, 0))]
    report, _ = pe.evaluate(gt, pred, [1.0], [0.25])
    assert report["confusion"] == [("sofa", "couch", 0.1)]


def test_iou_map_uses_boxes():
    box = ([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
    gt = [_obj("chair", (0, 0, 0), box=box)]
    pred = [_obj("chair", (0, 0, 0), box=box)]     # identical box -> IoU 1
    report, _ = pe.evaluate(gt, pred, [1.0], [0.25])
    assert report["mAP"]["iou@0.25"] == 1.0


# ── adapters ────────────────────────────────────────────────────────────────

def test_scene_objects_adapter_with_and_without_box():
    scene_objs = [
        {"label": "chair", "position": [1.0, 2.0, 0.0], "confidence": 0.8,
         "bbox_min": [0.5, 1.5, -0.5], "bbox_max": [1.5, 2.5, 0.5]},
        {"label": "lamp", "position": [3.0, 0.0, 1.0], "confidence": 0.4,
         "bbox_min": None, "bbox_max": None},
        {"label": "ghost", "position": None, "confidence": 0.1},  # dropped
    ]
    out = pe.scene_objects_to_eval(scene_objs)
    assert len(out) == 2                                   # ghost with no position dropped
    chair = next(o for o in out if o["label"] == "chair")
    assert chair["score"] == 0.8 and chair["center_3d"] == [1.0, 2.0, 0.0]
    assert chair["bbox_aabb"]["size"] == [1.0, 1.0, 1.0]
    lamp = next(o for o in out if o["label"] == "lamp")
    assert lamp["bbox_aabb"]["min"] == lamp["bbox_aabb"]["max"] == [3.0, 0.0, 1.0]


def test_object_entries_adapter_builds_aabb_from_size():
    entries = {
        7: ObjectEntry(object_id=7, label="table",
                       center=Vector3(x=2.0, y=0.0, z=0.5),
                       size=Vector3(x=1.0, y=2.0, z=0.8), heading=0.0),
    }
    out = pe.object_entries_to_eval(entries)
    assert len(out) == 1
    o = out[0]
    assert o["label"] == "table" and o["score"] == 1.0
    assert o["center_3d"] == [2.0, 0.0, 0.5]
    assert o["bbox_aabb"]["min"] == pytest.approx([1.5, -1.0, 0.1])
    assert o["bbox_aabb"]["max"] == pytest.approx([2.5, 1.0, 0.9])


def test_adapters_round_trip_through_evaluate():
    # authoritative GT (one chair) vs scene-graph pred (same chair, no box).
    gt = pe.object_entries_to_eval([
        ObjectEntry(object_id=1, label="chair", center=Vector3(x=0, y=0, z=0),
                    size=Vector3(x=0.6, y=0.6, z=1.0), heading=0.0)])
    pred = pe.scene_objects_to_eval([
        {"label": "chair", "position": [0.1, 0.0, 0.0], "confidence": 0.9,
         "bbox_min": None, "bbox_max": None}])
    report, primary = pe.evaluate(gt, pred, [0.5, 1.0], [0.25])
    assert report["operating_point"][f"dist@{primary}m"]["tp"] == 1
