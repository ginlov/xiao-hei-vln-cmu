"""Tests for eval_sampler: object_list parser, gt_converter, offline combiner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xiao_hei_vln.eval_sampler.object_list import parse_object_list
from xiao_hei_vln.eval_sampler.gt_converter import gt_from_entry
from xiao_hei_vln.eval_sampler.__main__ import run
from xiao_hei_vln.messages.outputs import NumericalResponse, ObjectReferenceResponse


# ── shared fixtures ───────────────────────────────────────────────────────────

OBJECT_LINES = [
    '72 1.00 2.00 3.00 0.50 0.60 0.70 0.00 "book"',
    '5 10.00 20.00 0.50 1.00 1.00 0.80 1.57 "table"',
]

ENTRY_OBJ_REF = {
    "type": "object_reference",
    "question": "Find the book above the table.",
    "answer": {"object_id": 72, "label": "book"},
    "object_list": OBJECT_LINES,
}

ENTRY_NUMERICAL = {
    "type": "numerical",
    "question": "How many tables are there?",
    "answer": 3,
    "object_list": OBJECT_LINES,
}


# ── object_list parser ────────────────────────────────────────────────────────


def test_parse_object_list_basic():
    lookup = parse_object_list(OBJECT_LINES)
    assert 72 in lookup
    assert 5 in lookup


def test_parse_object_list_values():
    lookup = parse_object_list(OBJECT_LINES)
    book = lookup[72]
    assert book.label == "book"
    assert book.center.x == pytest.approx(1.0)
    assert book.size.y == pytest.approx(0.60)
    assert book.heading == pytest.approx(0.0)


def test_parse_object_list_multiword_label():
    lines = ['10 0.0 0.0 0.0 1.0 1.0 1.0 0.0 "potted plant"']
    lookup = parse_object_list(lines)
    assert lookup[10].label == "potted plant"


def test_parse_object_list_empty():
    assert parse_object_list([]) == {}


def test_parse_object_list_malformed_skipped():
    lines = ["not a valid line", '72 1.0 2.0 3.0 0.5 0.6 0.7 0.0 "book"']
    lookup = parse_object_list(lines)
    assert len(lookup) == 1


# ── gt_converter ──────────────────────────────────────────────────────────────


def test_gt_numerical():
    gt = gt_from_entry(ENTRY_NUMERICAL)
    assert isinstance(gt, NumericalResponse)
    assert gt.value == 3


def test_gt_object_reference():
    gt = gt_from_entry(ENTRY_OBJ_REF)
    assert isinstance(gt, ObjectReferenceResponse)
    assert gt.object_id == 72
    assert gt.label == "book"
    assert gt.center.x == pytest.approx(1.0)
    assert gt.size.z == pytest.approx(0.70)


def test_gt_unknown_type_returns_none():
    assert gt_from_entry({"type": "instruction_following", "question": "Go left."}) is None


def test_gt_obj_ref_missing_object_id_returns_none():
    bad = {**ENTRY_OBJ_REF, "answer": {"label": "book"}}
    assert gt_from_entry(bad) is None


def test_gt_obj_ref_id_not_in_object_list_returns_none():
    bad = {**ENTRY_OBJ_REF, "answer": {"object_id": 999, "label": "ghost"}}
    assert gt_from_entry(bad) is None


# ── offline combiner (run()) ──────────────────────────────────────────────────


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_run_matches_by_question(tmp_path):
    gt_file = tmp_path / "gt.jsonl"
    pred_file = tmp_path / "pred.jsonl"
    out_file = tmp_path / "out.jsonl"

    _write_jsonl(gt_file, [ENTRY_NUMERICAL, ENTRY_OBJ_REF])
    _write_jsonl(pred_file, [
        {"question": "How many tables are there?",
         "prediction": {"kind": "numerical", "value": 5}},
        {"question": "Find the book above the table.",
         "prediction": {"kind": "object_reference", "label": "book",
                        "object_id": 72,
                        "center": {"x": 1.0, "y": 2.0, "z": 3.0},
                        "size": {"x": 0.5, "y": 0.6, "z": 0.7},
                        "heading": 0.0}},
    ])

    run(gt_path=gt_file, pred_path=pred_file, out_path=out_file, limit=None)

    lines = [json.loads(l) for l in out_file.read_text().strip().splitlines()]
    assert len(lines) == 2

    num_line = next(l for l in lines if l["question"] == "How many tables are there?")
    assert num_line["ground_truth"]["kind"] == "numerical"
    assert num_line["ground_truth"]["value"] == 3
    assert num_line["prediction"]["value"] == 5

    obj_line = next(l for l in lines if "Find" in l["question"])
    assert obj_line["ground_truth"]["kind"] == "object_reference"
    assert obj_line["prediction"]["object_id"] == 72


def test_run_unmatched_prediction_skipped(tmp_path):
    gt_file = tmp_path / "gt.jsonl"
    pred_file = tmp_path / "pred.jsonl"
    out_file = tmp_path / "out.jsonl"

    _write_jsonl(gt_file, [ENTRY_NUMERICAL])
    _write_jsonl(pred_file, [
        {"question": "A completely different question?",
         "prediction": {"kind": "numerical", "value": 1}},
    ])

    run(gt_path=gt_file, pred_path=pred_file, out_path=out_file, limit=None)
    assert out_file.read_text().strip() == ""


def test_run_instruction_following_gt_skipped(tmp_path):
    gt_file = tmp_path / "gt.jsonl"
    pred_file = tmp_path / "pred.jsonl"
    out_file = tmp_path / "out.jsonl"

    instr_entry = {"type": "instruction_following", "question": "Go to the door.",
                   "answer": None, "object_list": []}
    _write_jsonl(gt_file, [instr_entry, ENTRY_NUMERICAL])
    _write_jsonl(pred_file, [
        {"question": "How many tables are there?",
         "prediction": {"kind": "numerical", "value": 3}},
    ])

    run(gt_path=gt_file, pred_path=pred_file, out_path=out_file, limit=None)
    lines = [json.loads(l) for l in out_file.read_text().strip().splitlines()]
    assert len(lines) == 1
    assert lines[0]["ground_truth"]["kind"] == "numerical"


def test_run_limit(tmp_path):
    gt_file = tmp_path / "gt.jsonl"
    pred_file = tmp_path / "pred.jsonl"
    out_file = tmp_path / "out.jsonl"

    _write_jsonl(gt_file, [ENTRY_NUMERICAL, ENTRY_OBJ_REF])
    _write_jsonl(pred_file, [
        {"question": "How many tables are there?",
         "prediction": {"kind": "numerical", "value": 3}},
        {"question": "Find the book above the table.",
         "prediction": {"kind": "object_reference", "label": "book",
                        "object_id": 72,
                        "center": {"x": 1.0, "y": 2.0, "z": 3.0},
                        "size": {"x": 0.5, "y": 0.6, "z": 0.7},
                        "heading": 0.0}},
    ])

    run(gt_path=gt_file, pred_path=pred_file, out_path=out_file, limit=1)
    lines = [json.loads(l) for l in out_file.read_text().strip().splitlines()]
    # Only ENTRY_NUMERICAL loaded into GT (limit=1), so only one match possible.
    assert len(lines) == 1
    assert lines[0]["ground_truth"]["kind"] == "numerical"


# ── optional colour token (2nd quoted field) ──────────────────────────────────

def test_parse_object_list_reads_optional_color():
    lines = [
        '3 1.0 2.0 3.0 0.5 0.6 0.7 0.0 "book" "red"',   # with colour
        '4 0.0 0.0 0.0 1.0 1.0 1.0 0.0 "table"',          # none → backward compat
    ]
    got = parse_object_list(lines)
    assert got[3].label == "book" and got[3].color == "red"
    assert got[4].label == "table" and got[4].color is None
    # numeric fields still parse correctly alongside the colour token.
    assert (got[3].center.x, got[3].size.z, got[3].heading) == pytest.approx((1.0, 0.7, 0.0))


def test_parse_object_list_empty_color_token_is_none():
    (got,) = parse_object_list(['1 0 0 0 1 1 1 0 "chair" ""']).values()
    assert got.label == "chair" and got.color is None
