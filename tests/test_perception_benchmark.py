"""Tests for the multi-scene benchmark — mostly that a bad scene stays a row.

A benchmark that aborts on the first unreadable corpus is worse than useless
overnight, so the contract is that every scene produces a row either way.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xiao_hei_vln.perception import benchmark
from xiao_hei_vln.perception.benchmark import SceneResult, format_table, run


def _corpus(root: Path, name: str, *, detections: bool = True) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "frames.jsonl").write_text("")
    if detections:
        (d / "detections.jsonl").write_text("")
    return d


def test_missing_corpus_is_a_row_not_an_exception(tmp_path: Path) -> None:
    results = run(["ghost"], frames_root=tmp_path, gt_root=tmp_path)
    assert len(results) == 1
    assert "no corpus" in results[0].error


def test_corpus_without_detections_names_the_fix(tmp_path: Path) -> None:
    _corpus(tmp_path, "loft_run1", detections=False)
    results = run(["loft"], frames_root=tmp_path, gt_root=tmp_path)
    assert "replay detect" in results[0].error


def test_unreadable_gt_is_a_row(tmp_path: Path) -> None:
    _corpus(tmp_path, "loft_run1")
    results = run(["loft"], frames_root=tmp_path, gt_root=tmp_path)
    assert results[0].error.startswith("GT unreadable")


def test_one_broken_scene_does_not_hide_the_others(tmp_path: Path, monkeypatch) -> None:
    for name in ("a_run1", "b_run1"):
        _corpus(tmp_path, name)

    def fake_run_scene(scene: str, **_kw) -> SceneResult:
        if scene == "a":
            return SceneResult(scene="a", error="boom")
        return SceneResult(scene="b", n_gt=10, n_pred=9, recall=0.5)

    monkeypatch.setattr(benchmark, "run_scene", fake_run_scene)
    results = run(["a", "b"], frames_root=tmp_path, gt_root=tmp_path)
    assert [r.scene for r in results] == ["a", "b"]
    assert results[0].error and not results[1].error


def test_vocab_coverage_normalises_labels() -> None:
    gt = [{"label": "Coffee_Table"}, {"label": "unicorn"}]
    assert benchmark._vocab_coverage(gt, {"coffee table"}) == pytest.approx(0.5)


def test_table_has_a_mean_row_over_scored_scenes_only() -> None:
    scored = [
        SceneResult(scene="a", keyframes=10, n_gt=10, n_pred=8, recall=0.4, precision=0.5),
        SceneResult(scene="b", keyframes=20, n_gt=20, n_pred=16, recall=0.6, precision=0.7),
        SceneResult(scene="c", error="no corpus"),
    ]
    table = format_table(scored)
    lines = table.splitlines()
    assert lines[-1].startswith("MEAN")
    # 0.4 and 0.6 average to 0.5; the failed scene must not drag it to 0.333.
    assert " 0.500" in lines[-1]
    assert any("no corpus" in ln for ln in lines)


def test_summary_json_round_trips(tmp_path: Path) -> None:
    from dataclasses import asdict

    r = SceneResult(scene="a", n_gt=3, recall=0.5, report={"mAP": {"dist@1.0m": 0.5}})
    out = tmp_path / "benchmark.json"
    out.write_text(json.dumps([asdict(r)], indent=1))
    assert json.loads(out.read_text())[0]["scene"] == "a"
