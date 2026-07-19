"""Tests for xiao_hei_vln.detected_dataset (perception → LLM-eval dataset)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xiao_hei_vln.detected_dataset import (
    build_dataset,
    extract_scene_graph_from_session,
    load_qa_for_scene,
    scene_graph_to_object_list,
    splice_detected,
    truth_records,
)
from xiao_hei_vln.eval_sampler.object_list import parse_object_list


def _online_scene() -> dict:
    """A SceneRepresentation.to_dict()-style graph with two boxed objects."""
    return {
        "objects": [
            {
                "object_id": 5, "label": "sofa",
                "position": [1.0, 2.0, 0.5],
                "bbox_min": [0.6, 1.7, 0.1], "bbox_max": [1.4, 2.3, 0.9],
            },
            {
                "object_id": 9, "label": "vase",
                "position": [3.0, -1.0, 0.4],
                "bbox_min": [2.9, -1.1, 0.3], "bbox_max": [3.1, -0.9, 0.5],
            },
        ]
    }


# ── converter ────────────────────────────────────────────────────────────────

def test_converter_roundtrips_through_parse_object_list():
    lines = scene_graph_to_object_list(_online_scene())
    parsed = parse_object_list(lines)
    # ids are freshly assigned 0..N-1, NOT the source object_ids (5, 9).
    assert set(parsed.keys()) == {0, 1}
    e0 = parsed[0]
    assert e0.label == "sofa"
    assert (e0.center.x, e0.center.y, e0.center.z) == pytest.approx((1.0, 2.0, 0.5))
    # size = bbox_max - bbox_min
    assert (e0.size.x, e0.size.y, e0.size.z) == pytest.approx((0.8, 0.6, 0.8))
    assert e0.heading == pytest.approx(0.0)
    assert parsed[1].label == "vase"


def test_converter_point_object_has_zero_size():
    scene = {"objects": [{"label": "cup", "position": [1.0, 1.0, 1.0],
                          "bbox_min": None, "bbox_max": None}]}
    parsed = parse_object_list(scene_graph_to_object_list(scene))
    assert (parsed[0].size.x, parsed[0].size.y, parsed[0].size.z) == pytest.approx((0.0, 0.0, 0.0))
    assert parsed[0].label == "cup"


def test_converter_accepts_offline_scene_objects_v1_format():
    scene = {"objects": [
        {"label": "book", "center_3d": [5.0, 0.1, 1.0],
         "bbox_aabb": {"size": [0.24, 0.47, 0.27]}},
    ]}
    parsed = parse_object_list(scene_graph_to_object_list(scene))
    assert (parsed[0].center.x, parsed[0].center.z) == pytest.approx((5.0, 1.0))
    assert (parsed[0].size.x, parsed[0].size.y) == pytest.approx((0.24, 0.47))


def test_converter_label_with_quote_is_sanitized():
    scene = {"objects": [{"label": 'a "quoted" thing', "position": [0, 0, 0]}]}
    # must not break the single-quoted-label parse — one clean label survives.
    parsed = parse_object_list(scene_graph_to_object_list(scene))
    assert len(parsed) == 1
    assert '"' not in parsed[0].label


# ── splice ───────────────────────────────────────────────────────────────────

def test_splice_adds_detected_keeps_everything_else():
    records = [{"scene": "x", "type": "numerical", "question": "how many?",
                "object_list": ["0 0 0 0 1 1 1 0 \"gt\""], "answer": 3}]
    det = ["0 1 1 1 0.5 0.5 0.5 0.0 \"sofa\""]
    out = splice_detected(records, det)
    assert out[0]["detected_object_list"] == det
    # GT + question + answer preserved.
    assert out[0]["object_list"] == records[0]["object_list"]
    assert out[0]["answer"] == 3 and out[0]["question"] == "how many?"


# ── qa loading ───────────────────────────────────────────────────────────────

def test_load_qa_for_scene_filters(tmp_path: Path):
    p = tmp_path / "qa.jsonl"
    p.write_text(
        json.dumps({"scene": "arabic_room", "question": "a"}) + "\n"
        + json.dumps({"scene": "loft", "question": "b"}) + "\n"
        + json.dumps({"scene": "arabic_room", "question": "c"}) + "\n"
    )
    got = load_qa_for_scene([p], "arabic_room")
    assert [r["question"] for r in got] == ["a", "c"]


# ── truth ────────────────────────────────────────────────────────────────────

def test_truth_records_numerical_and_ref_box():
    records = [
        {"scene": "x", "type": "numerical", "question": "n?", "answer": 2,
         "object_list": []},
        {"scene": "x", "type": "object_reference", "question": "find it",
         "answer": {"object_id": 1, "label": "lamp"},
         "object_list": [
             "0 0 0 0 1 1 1 0 \"floor\"",
             "1 2.0 3.0 1.5 0.2 0.2 0.4 0 \"lamp\"",
         ]},
    ]
    truth = truth_records(records)
    assert truth[0] == {"scene": "x", "type": "numerical", "question": "n?", "answer": 2}
    # ref: GT target box resolved from object_list by object_id.
    assert truth[1]["target_center"] == pytest.approx([2.0, 3.0, 1.5])
    assert truth[1]["target_size"] == pytest.approx([0.2, 0.2, 0.4])


# ── session extraction ───────────────────────────────────────────────────────

def test_extract_scene_graph_from_session(tmp_path: Path):
    qdir = tmp_path / "q_001_test"
    qdir.mkdir(parents=True)
    (qdir / "ticks.jsonl").write_text(
        json.dumps({"tick_id": 1, "scene": {"objects": [{"label": "old"}]}}) + "\n"
        + json.dumps({"tick_id": 2, "scene": {"objects": [{"label": "new"}]}}) + "\n"
    )
    scene = extract_scene_graph_from_session(tmp_path)
    assert scene["objects"][0]["label"] == "new"   # last tick wins (fullest)


def test_extract_scene_graph_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        extract_scene_graph_from_session(tmp_path)


# ── end-to-end ───────────────────────────────────────────────────────────────

def test_build_dataset_end_to_end(tmp_path: Path):
    qa = tmp_path / "num.jsonl"
    qa.write_text(
        json.dumps({"scene": "arabic_room", "type": "numerical",
                    "question": "how many sofas?", "object_list": [], "answer": 1}) + "\n"
        + json.dumps({"scene": "other", "type": "numerical",
                      "question": "x", "object_list": [], "answer": 0}) + "\n"
    )
    spliced, truth, detected = build_dataset(_online_scene(), "arabic_room", [qa])
    assert len(spliced) == 1 and len(truth) == 1        # only arabic_room matched
    assert len(detected) == 2                            # our two objects
    # the LLM input carries our detections and is parseable.
    parsed = parse_object_list(spliced[0]["detected_object_list"])
    assert {e.label for e in parsed.values()} == {"sofa", "vase"}
    assert truth[0]["answer"] == 1
