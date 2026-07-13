"""Tests for the offline Gemini eval HTML report.

Matplotlib renders headless (Agg), so plot generation runs for real; no
network or API key needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xiao_hei_vln.gemini import eval_report as er

# A tiny two-object scene: id cx cy cz lx ly lz heading "label"
SOFA = '5 1.00 2.00 0.40 2.00 0.90 0.80 0.00 "sofa"'
LAMP = '9 -1.00 0.00 0.60 0.30 0.30 1.20 0.00 "lamp"'
OBJECT_LIST = [SOFA, LAMP]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _gt_ref(question: str, object_id: int, label: str) -> dict:
    return {
        "type": "object_reference",
        "question": question,
        "answer": {"object_id": object_id, "label": label},
        "object_list": OBJECT_LIST,
    }


def _pred_ref(question: str, *, center, size, label: str, rationale: str = "because") -> dict:
    return {
        "question": question,
        "prediction": {
            "kind": "object_reference",
            "label": label,
            "object_id": 0,
            "center": {"x": center[0], "y": center[1], "z": center[2]},
            "size": {"x": size[0], "y": size[1], "z": size[2]},
            "heading": 0.0,
            "rationale": rationale,
        },
    }


def test_build_report_scores_a_hit(tmp_path: Path) -> None:
    gt = tmp_path / "gt.jsonl"
    pred = tmp_path / "pred.jsonl"
    _write_jsonl(gt, [_gt_ref("Find the sofa.", 5, "sofa")])
    # Predict the sofa's exact box → IoU 1.0, score 2.
    _write_jsonl(pred, [_pred_ref("Find the sofa.", center=(1, 2, 0.4), size=(2, 0.9, 0.8), label="sofa")])

    report = er.build_report(gt, pred)
    assert len(report.ref) == 1 and report.unmatched == 0
    r = report.ref[0]
    assert r.iou == pytest.approx(1.0) and r.score == 2
    assert r.center_dist == 0.0
    assert len(r.objects) == 2  # scene reconstructed for the plot


def test_build_report_scores_a_miss(tmp_path: Path) -> None:
    gt = tmp_path / "gt.jsonl"
    pred = tmp_path / "pred.jsonl"
    _write_jsonl(gt, [_gt_ref("Find the lamp.", 9, "lamp")])
    # Predict the sofa's box for a lamp question → no overlap → score 0.
    _write_jsonl(pred, [_pred_ref("Find the lamp.", center=(1, 2, 0.4), size=(2, 0.9, 0.8), label="sofa")])

    r = er.build_report(gt, pred).ref[0]
    assert r.iou == 0.0 and r.score == 0
    assert r.center_dist > 0


def test_unmatched_prediction_counted(tmp_path: Path) -> None:
    gt = tmp_path / "gt.jsonl"
    pred = tmp_path / "pred.jsonl"
    _write_jsonl(gt, [_gt_ref("Find the sofa.", 5, "sofa")])
    _write_jsonl(pred, [_pred_ref("A question with no GT.", center=(0, 0, 0), size=(1, 1, 1), label="x")])

    report = er.build_report(gt, pred)
    assert report.ref == [] and report.unmatched == 1


def test_render_topdown_png_is_a_png(tmp_path: Path) -> None:
    gt = tmp_path / "gt.jsonl"
    pred = tmp_path / "pred.jsonl"
    _write_jsonl(gt, [_gt_ref("Find the sofa.", 5, "sofa")])
    _write_jsonl(pred, [_pred_ref("Find the sofa.", center=(1, 2, 0.4), size=(2, 0.9, 0.8), label="sofa")])
    r = er.build_report(gt, pred).ref[0]

    png = er.render_topdown_png(r)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_generate_html_is_self_contained(tmp_path: Path) -> None:
    gt = tmp_path / "gt.jsonl"
    pred = tmp_path / "pred.jsonl"
    _write_jsonl(
        gt,
        [_gt_ref("Find the sofa.", 5, "sofa"), _gt_ref("Find the lamp.", 9, "lamp")],
    )
    _write_jsonl(
        pred,
        [
            _pred_ref("Find the sofa.", center=(1, 2, 0.4), size=(2, 0.9, 0.8), label="sofa", rationale="the big couch"),
            _pred_ref("Find the lamp.", center=(-1, 0, 0.6), size=(0.3, 0.3, 1.2), label="lamp"),
        ],
    )
    html = er.generate_html(er.build_report(gt, pred))

    assert html.startswith("<!doctype html>")
    assert "Find the sofa." in html
    assert "the big couch" in html            # rationale surfaced
    assert "data:image/png;base64," in html   # plot embedded, no external asset
    assert "http://" not in html and "https://" not in html  # self-contained
    assert "mean IoU" in html


def test_numerical_predictions_reported(tmp_path: Path) -> None:
    gt = tmp_path / "gt.jsonl"
    pred = tmp_path / "pred.jsonl"
    _write_jsonl(gt, [{"type": "numerical", "question": "How many sofas?", "answer": 1}])
    _write_jsonl(
        pred,
        [{"question": "How many sofas?", "prediction": {"kind": "numerical", "value": 1, "rationale": "one"}}],
    )
    report = er.build_report(gt, pred)
    assert len(report.num) == 1 and report.num[0].correct is True
    html = er.generate_html(report)
    assert "accuracy" in html and "How many sofas?" in html
