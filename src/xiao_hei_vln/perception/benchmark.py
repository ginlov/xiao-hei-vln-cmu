"""Score the perception stack across every recorded scene at once.

A single scene tells you very little: it is one room's furniture, one tour, and
whatever the vocabulary happens to cover there. Tuning against one and checking
against a second — which is how the current numbers were produced — leaves no
way to see which gains generalise. This runs the same replay and the same
scoring over a set of corpora and puts them in one table.

The vocabulary ceiling is reported alongside recall on purpose. Our detector
cannot name what is not in its class list, so `in_vocab` is the highest recall
the scene admits; recall well under it is a perception problem, recall at it is
a vocabulary problem, and the two need different fixes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from xiao_hei_vln.perception import replay
from xiao_hei_vln.perception.eval import evaluate, load_gt_from_zip, scene_objects_to_eval
from xiao_hei_vln.perception.vocab import DEFAULT_PRIOR, is_structure

log = logging.getLogger(__name__)

DIST_THR = 1.0
IOU_THR = 0.25


def _normalise(label: str) -> str:
    return label.strip().lower().replace("_", " ")


@dataclass
class SceneResult:
    scene: str
    frames: int = 0
    keyframes: int = 0
    n_gt: int = 0
    in_vocab: float = 0.0
    n_pred: int = 0
    precision: float = 0.0
    recall: float = 0.0
    mAP: float = 0.0
    mean_iou: float = 0.0
    median_center_err: float = 0.0
    counting_mae: float = 0.0
    error: str = ""
    report: dict = field(default_factory=dict)


def _vocab_coverage(gt_objs: list[dict], classes: set[str]) -> float:
    if not gt_objs:
        return 0.0
    hit = sum(1 for g in gt_objs if _normalise(g["label"]) in classes)
    return hit / len(gt_objs)


def run_scene(
    scene: str,
    *,
    frames_dir: Path,
    gt_zip: Path,
    include_structure: bool = False,
    scene_out: Path | None = None,
    **lift_kwargs,
) -> SceneResult:
    """Replay one corpus and score it. Never raises — a broken scene is a row."""
    res = SceneResult(scene=scene)
    try:
        gt = load_gt_from_zip(gt_zip, scene)
    except Exception as exc:
        res.error = f"GT unreadable: {exc}"
        return res
    if not include_structure:
        gt = [g for g in gt if not is_structure(g["label"])]
    res.n_gt = len(gt)
    if not gt:
        res.error = "no GT objects in zip"
        return res
    res.in_vocab = _vocab_coverage(gt, {_normalise(c) for c in DEFAULT_PRIOR})

    try:
        all_frames = replay.load_frames(frames_dir, min_move_m=0.0, min_rot_deg=0.0)
        kept = replay.load_frames(frames_dir, **{
            k: lift_kwargs[k] for k in ("min_move_m", "min_rot_deg") if k in lift_kwargs
        })
        res.frames, res.keyframes = len(all_frames), len(kept)
        scene_dict, _ = replay.replay_lift(frames_dir, frames=kept, **lift_kwargs)
    except Exception as exc:
        res.error = f"replay failed: {exc}"
        return res

    objs = scene_dict["objects"]
    if not include_structure:
        objs = [o for o in objs if not is_structure(o["label"])]
    res.n_pred = len(objs)
    if scene_out is not None:
        scene_out.parent.mkdir(parents=True, exist_ok=True)
        scene_out.write_text(json.dumps({**scene_dict, "objects": objs}))

    report, _ = evaluate(gt, scene_objects_to_eval(objs), [DIST_THR], [IOU_THR])
    op = report["operating_point"][f"dist@{DIST_THR}m"]
    res.precision = op["precision"]
    res.recall = op["recall"]
    res.mAP = report["mAP"][f"dist@{DIST_THR}m"] or 0.0
    res.mean_iou = op.get("mean_3d_iou", 0.0)
    res.median_center_err = op.get("median_center_err_m", 0.0)
    res.counting_mae = report.get("counting_MAE", 0.0)
    res.report = report
    return res


def format_table(results: list[SceneResult]) -> str:
    cols = (
        ("scene", 18, "s"), ("kfrm", 5, "d"), ("GT", 5, "d"), ("inVoc", 6, ".0%"),
        ("pred", 5, "d"), ("P", 6, ".3f"), ("R", 6, ".3f"), ("mAP", 6, ".3f"),
        ("mIoU", 6, ".3f"), ("cErr", 6, ".2f"), ("cMAE", 6, ".2f"),
    )
    head = " ".join(f"{n:>{w}s}" if i else f"{n:<{w}s}" for i, (n, w, _) in enumerate(cols))
    lines = [head, "-" * len(head)]
    ok = [r for r in results if not r.error]
    for r in results:
        if r.error:
            lines.append(f"{r.scene:<18s} {r.error}")
            continue
        vals = (r.scene, r.keyframes, r.n_gt, r.in_vocab, r.n_pred, r.precision,
                r.recall, r.mAP, r.mean_iou, r.median_center_err, r.counting_mae)
        lines.append(" ".join(
            f"{v:<{w}{f}}" if i == 0 else f"{v:>{w}{f}}"
            for i, (v, (_, w, f)) in enumerate(zip(vals, cols))))
    if ok:
        lines.append("-" * len(head))
        # Unweighted across scenes: a benchmark answers "how does this hold up
        # on a new room", and a per-object mean would let the biggest scene
        # answer for all of them.
        mean = ("MEAN", int(np.mean([r.keyframes for r in ok])),
                int(np.mean([r.n_gt for r in ok])), float(np.mean([r.in_vocab for r in ok])),
                int(np.mean([r.n_pred for r in ok])),
                *(float(np.mean([getattr(r, a) for r in ok])) for a in
                  ("precision", "recall", "mAP", "mean_iou", "median_center_err",
                   "counting_mae")))
        lines.append(" ".join(
            f"{v:<{w}{f}}" if i == 0 else f"{v:>{w}{f}}"
            for i, (v, (_, w, f)) in enumerate(zip(mean, cols))))
    return "\n".join(lines)


def run(
    scenes: list[str],
    *,
    frames_root: Path,
    gt_root: Path,
    run_name: str = "run1",
    out_dir: Path | None = None,
    include_structure: bool = False,
    **lift_kwargs,
) -> list[SceneResult]:
    results = []
    for scene in scenes:
        frames_dir = frames_root / f"{scene}_{run_name}"
        if not (frames_dir / "frames.jsonl").is_file():
            results.append(SceneResult(scene=scene, error=f"no corpus at {frames_dir}"))
            continue
        if not (frames_dir / "detections.jsonl").is_file():
            results.append(SceneResult(
                scene=scene, error="no detections.jsonl — run `replay detect` first"))
            continue
        log.info("scoring %s", scene)
        results.append(run_scene(
            scene,
            frames_dir=frames_dir,
            gt_zip=gt_root / f"{scene}.zip",
            include_structure=include_structure,
            scene_out=(out_dir / f"{scene}_scene.json") if out_dir else None,
            **lift_kwargs,
        ))
    return results
