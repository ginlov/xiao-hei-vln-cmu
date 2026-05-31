"""Tests for object reference evaluation metrics."""

import math

import pytest

from xiao_hei_vln.evaluator.metrics.object_reference import (
    _center_distance,
    _challenge_score,
    _iou_3d_aabb,
    compute_object_reference_metrics,
)
from xiao_hei_vln.messages.common import Vector3
from xiao_hei_vln.messages.outputs import ObjectReferenceResponse


def _box(cx, cy, cz, sx, sy, sz) -> ObjectReferenceResponse:
    return ObjectReferenceResponse(
        label="obj",
        object_id=0,
        center=Vector3(x=cx, y=cy, z=cz),
        size=Vector3(x=sx, y=sy, z=sz),
    )


# --- IoU -------------------------------------------------------------------


def test_iou_perfect_overlap():
    box = _box(0, 0, 0, 2, 2, 2)
    assert _iou_3d_aabb(box.center, box.size, box.center, box.size) == pytest.approx(1.0)


def test_iou_no_overlap():
    a = _box(0, 0, 0, 1, 1, 1)
    b = _box(10, 10, 10, 1, 1, 1)
    assert _iou_3d_aabb(a.center, a.size, b.center, b.size) == pytest.approx(0.0)


def test_iou_partial_overlap():
    # Two 2×2×2 cubes shifted 1 unit in x — overlap is a 1×2×2=4 slab
    a = _box(0, 0, 0, 2, 2, 2)   # x: [-1, 1]
    b = _box(1, 0, 0, 2, 2, 2)   # x: [0, 2]
    iou = _iou_3d_aabb(a.center, a.size, b.center, b.size)
    # intersection = 1*2*2 = 4; union = 8+8-4 = 12
    assert iou == pytest.approx(4 / 12)


def test_iou_zero_size_box():
    a = _box(0, 0, 0, 0, 0, 0)
    b = _box(0, 0, 0, 1, 1, 1)
    assert _iou_3d_aabb(a.center, a.size, b.center, b.size) == pytest.approx(0.0)


# --- Center distance -------------------------------------------------------


def test_center_distance_zero():
    v = Vector3(x=1.0, y=2.0, z=3.0)
    assert _center_distance(v, v) == pytest.approx(0.0)


def test_center_distance_known():
    a = Vector3(x=0.0, y=0.0, z=0.0)
    b = Vector3(x=3.0, y=4.0, z=0.0)
    assert _center_distance(a, b) == pytest.approx(5.0)


# --- Challenge score -------------------------------------------------------


def test_challenge_score_thresholds():
    assert _challenge_score(1.0) == 2
    assert _challenge_score(0.5) == 2
    assert _challenge_score(0.49) == 1
    assert _challenge_score(0.25) == 1
    assert _challenge_score(0.24) == 0
    assert _challenge_score(0.0) == 0


# --- compute_object_reference_metrics --------------------------------------


def test_empty():
    m = compute_object_reference_metrics([])
    assert m.n == 0
    assert m.mean_iou == 0.0


def test_perfect_predictions():
    box = _box(1, 2, 3, 1, 1, 1)
    pairs = [(box, box), (box, box)]
    m = compute_object_reference_metrics(pairs)
    assert m.n == 2
    assert m.mean_iou == pytest.approx(1.0)
    assert m.sr_at_iou_50 == pytest.approx(1.0)
    assert m.sr_at_iou_25 == pytest.approx(1.0)
    assert m.mean_center_distance == pytest.approx(0.0)
    assert m.mean_challenge_score == pytest.approx(2.0)
    assert m.mean_challenge_score_norm == pytest.approx(1.0)


def test_no_overlap_predictions():
    gt = _box(0, 0, 0, 1, 1, 1)
    pred = _box(100, 100, 100, 1, 1, 1)
    m = compute_object_reference_metrics([(gt, pred)])
    assert m.mean_iou == pytest.approx(0.0)
    assert m.sr_at_iou_50 == pytest.approx(0.0)
    assert m.sr_at_iou_25 == pytest.approx(0.0)
    assert m.mean_challenge_score == pytest.approx(0.0)
    assert m.mean_challenge_score_norm == pytest.approx(0.0)
    expected_dist = math.sqrt(3 * 100**2)
    assert m.mean_center_distance == pytest.approx(expected_dist)


def test_mixed_predictions():
    perfect = _box(0, 0, 0, 2, 2, 2)
    miss = _box(100, 0, 0, 2, 2, 2)
    # pair 1: perfect (IoU=1.0 → score 2); pair 2: miss (IoU=0 → score 0)
    m = compute_object_reference_metrics([(perfect, perfect), (perfect, miss)])
    assert m.n == 2
    assert m.mean_iou == pytest.approx(0.5)
    assert m.sr_at_iou_50 == pytest.approx(0.5)
    assert m.mean_challenge_score == pytest.approx(1.0)
    assert m.mean_challenge_score_norm == pytest.approx(0.5)


def test_as_dict_keys():
    box = _box(0, 0, 0, 1, 1, 1)
    m = compute_object_reference_metrics([(box, box)])
    d = m.as_dict()
    assert set(d.keys()) == {
        "n", "mean_iou", "sr_at_iou_50", "sr_at_iou_25",
        "mean_center_distance", "mean_challenge_score", "mean_challenge_score_norm",
    }
