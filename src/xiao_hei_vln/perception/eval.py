"""Quantitative evaluation of the perception scene graph against ground truth.

The online pipeline (``PerceptionResponder`` → ``SceneRepresentation``) builds a
3D object map but has no way to answer *how good* that map is. This module scores
a serialized scene graph against a ground-truth object list and reports the
standard detection metrics, so a run can be graded offline.

Two ground-truth sources feed the SAME evaluator:

* **authoritative** — the VLA-3D ``object_list.txt`` inside a challenge scene zip
  (every object in the scene, with real 3D boxes). Read via
  :func:`xiao_hei_vln.scene.io.read_objects_from_zip`. Measures whole-scene recall.
* **observable** — an object map produced over only the frames the trajectory
  actually saw (our ``objectmap.py`` ``--source gt`` output). Decouples perception
  quality from coverage. Pass such a JSON via ``--gt``.

Metrics (ported verbatim from our offline eval prototype)
--------------------------------------------------------------------------------
Primary  : mAP @ center-distance (nuScenes-style; AABB boxes + thin objects make
           3D IoU brutally strict, so centre distance is the fairer primary).
Aux      : mAP @ 3D IoU, operating-point precision/recall/F1, per-class counting
           error (numerical-question proxy), matched centre error + mean 3D IoU,
           and a label-confusion list (geometric match, label disagreement).

Frames: predictions come from the scene graph in the sim **map** frame; the
authoritative ``object_list`` centers are in the VLA-3D scene frame. The challenge
treats these as aligned (see ``dataset_generator/challenge_gt_gen.py``); box
headings are ignored (axis-aligned approximation) — center-distance is unaffected.

    # score a dumped scene graph against a scene zip:
    python -m xiao_hei_vln.perception.eval \
        --scene run/scene.json --gt-zip scenes/arabic_room.zip \
        --scene-name arabic_room --out metrics.json

    # score against an observable-GT object map (objectmap.py --source gt):
    python -m xiao_hei_vln.perception.eval --scene run/scene.json --gt gt_map.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


# ── geometry ──────────────────────────────────────────────────────────────────

def _box(o):
    b = o["bbox_aabb"]
    return np.asarray(b["min"], float), np.asarray(b["max"], float)


def iou_3d(amin, amax, bmin, bmax) -> float:
    """Axis-aligned 3D IoU (same convention as objectmap.iou_3d)."""
    lo = np.maximum(amin, bmin)
    hi = np.minimum(amax, bmax)
    inter = np.prod(np.clip(hi - lo, 0, None))
    if inter <= 0:
        return 0.0
    va = np.prod(np.clip(amax - amin, 0, None))
    vb = np.prod(np.clip(bmax - bmin, 0, None))
    union = va + vb - inter
    return float(inter / union) if union > 0 else 0.0


def center_dist(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a["center_3d"], float)
                                - np.asarray(b["center_3d"], float)))


# ── matching + AP ─────────────────────────────────────────────────────────────

def _ok(pred, gt, mode, thr) -> bool:
    if mode == "dist":
        return center_dist(pred, gt) <= thr
    pm, pM = _box(pred); gm, gM = _box(gt)
    return iou_3d(pm, pM, gm, gM) >= thr


def _quality(pred, gt, mode) -> float:
    """Higher = better match (for greedy pick among eligible GTs)."""
    if mode == "dist":
        return -center_dist(pred, gt)           # nearest
    pm, pM = _box(pred); gm, gM = _box(gt)
    return iou_3d(pm, pM, gm, gM)               # highest IoU


def average_precision(preds, gts, mode, thr) -> float:
    """COCO-style AP for one class: rank preds by score, greedily match each to
    the best still-unmatched eligible GT, integrate the precision-recall curve."""
    n_gt = len(gts)
    if n_gt == 0:
        return float("nan")
    preds = sorted(preds, key=lambda p: p.get("score", 0.0), reverse=True)
    matched = [False] * n_gt
    tp = np.zeros(len(preds)); fp = np.zeros(len(preds))
    for i, p in enumerate(preds):
        best_j, best_q = -1, None
        for j, g in enumerate(gts):
            if matched[j] or not _ok(p, g, mode, thr):
                continue
            q = _quality(p, g, mode)
            if best_q is None or q > best_q:
                best_q, best_j = q, j
        if best_j >= 0:
            matched[best_j] = True; tp[i] = 1
        else:
            fp[i] = 1
    if not preds:
        return 0.0
    tpc, fpc = np.cumsum(tp), np.cumsum(fp)
    rec = tpc / n_gt
    prec = tpc / np.maximum(tpc + fpc, 1e-9)
    # 101-point interpolation (COCO)
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        p = prec[rec >= r].max() if np.any(rec >= r) else 0.0
        ap += p / 101
    return float(ap)


def operating_point(pred_objs, gt_objs, mode, thr):
    """Greedy one-to-one match over ALL preds (score order) -> TP/FP/FN at the
    natural operating point, plus matched pairs for error stats."""
    by_cls_gt = defaultdict(list)
    for g in gt_objs:
        by_cls_gt[g["label"]].append(g)
    used = {k: [False] * len(v) for k, v in by_cls_gt.items()}
    tp = fp = 0
    pairs = []
    for p in sorted(pred_objs, key=lambda p: p.get("score", 0.0), reverse=True):
        gts = by_cls_gt.get(p["label"], [])
        best_j, best_q = -1, None
        for j, g in enumerate(gts):
            if used[p["label"]][j] or not _ok(p, g, mode, thr):
                continue
            q = _quality(p, g, mode)
            if best_q is None or q > best_q:
                best_q, best_j = q, j
        if best_j >= 0:
            used[p["label"]][best_j] = True; tp += 1
            pairs.append((p, gts[best_j]))
        else:
            fp += 1
    fn = sum(u.count(False) for u in used.values())
    return tp, fp, fn, pairs


def confusion(pred_objs, gt_objs, thr=1.0):
    """Geometric match IGNORING label (centre dist <= thr); report disagreements."""
    used = [False] * len(gt_objs)
    out = []
    for p in sorted(pred_objs, key=lambda p: p.get("score", 0.0), reverse=True):
        best_j, best_d = -1, thr
        for j, g in enumerate(gt_objs):
            if used[j]:
                continue
            d = center_dist(p, g)
            if d <= best_d:
                best_d, best_j = d, j
        if best_j >= 0:
            used[best_j] = True
            if p["label"] != gt_objs[best_j]["label"]:
                out.append((p["label"], gt_objs[best_j]["label"], round(best_d, 2)))
    return out


# ── driver ────────────────────────────────────────────────────────────────────

def evaluate(gt_objs, pred_objs, dist_thr, iou_thr):
    classes = sorted({o["label"] for o in gt_objs} | {o["label"] for o in pred_objs})
    gt_by = defaultdict(list); pr_by = defaultdict(list)
    for o in gt_objs:
        gt_by[o["label"]].append(o)
    for o in pred_objs:
        pr_by[o["label"]].append(o)

    report = {"n_gt": len(gt_objs), "n_pred": len(pred_objs),
              "classes": len(classes), "mAP": {}, "operating_point": {},
              "counting": {}, "confusion": []}

    # mAP at each threshold (mean over classes that have GT)
    for d in dist_thr:
        aps = [average_precision(pr_by[c], gt_by[c], "dist", d) for c in classes]
        aps = [a for a in aps if a == a]   # drop NaN (no GT)
        report["mAP"][f"dist@{d}m"] = round(float(np.mean(aps)), 4) if aps else None
    for t in iou_thr:
        aps = [average_precision(pr_by[c], gt_by[c], "iou", t) for c in classes]
        aps = [a for a in aps if a == a]
        report["mAP"][f"iou@{t}"] = round(float(np.mean(aps)), 4) if aps else None

    # operating-point P/R/F1 + matched error stats at the primary distance thr
    primary = dist_thr[len(dist_thr) // 2]
    for d in dist_thr:
        tp, fp, fn, pairs = operating_point(pred_objs, gt_objs, "dist", d)
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        entry = {"tp": tp, "fp": fp, "fn": fn,
                 "precision": round(prec, 4), "recall": round(rec, 4),
                 "f1": round(f1, 4)}
        if d == primary and pairs:
            ce = [center_dist(p, g) for p, g in pairs]
            ious = [iou_3d(*_box(p), *_box(g)) for p, g in pairs]
            entry["mean_center_err_m"] = round(float(np.mean(ce)), 3)
            entry["median_center_err_m"] = round(float(np.median(ce)), 3)
            entry["mean_3d_iou"] = round(float(np.mean(ious)), 3)
        report["operating_point"][f"dist@{d}m"] = entry

    # per-class counting (numerical-question proxy)
    abs_err = []
    for c in classes:
        g, p = len(gt_by[c]), len(pr_by[c])
        abs_err.append(abs(g - p))
        report["counting"][c] = {"gt": g, "pred": p, "err": p - g}
    report["counting_MAE"] = round(float(np.mean(abs_err)), 3) if abs_err else 0.0
    report["counting_exact_frac"] = round(
        float(np.mean([e == 0 for e in abs_err])), 3) if abs_err else 0.0

    report["confusion"] = confusion(pred_objs, gt_objs, thr=max(dist_thr))
    return report, primary


# ── adapters: scene graph / GT sources → the eval object schema ────────────────
#
# The evaluator consumes objects shaped ``{label, score, center_3d, bbox_aabb}``.
# ``bbox_aabb`` is only used by the IoU-mode metrics; when a predicted object has
# no 3D box (their scene graph leaves it ``None`` until an ObjectMap fills it), we
# synthesize a degenerate point-box so IoU code never KeyErrors — its IoU against
# any real box is 0, so IoU-mAP simply reflects the missing extent while the
# center-distance metrics (the primary) stay exact.

def _point_box(center):
    c = [float(x) for x in center]
    return {"min": c, "max": c, "size": [0.0, 0.0, 0.0]}


def scene_objects_to_eval(objects: Iterable[dict]) -> list[dict]:
    """Adapt ``SceneRepresentation.to_dict()['objects']`` to eval objects.

    Uses ``confidence`` as the ranking score and ``position`` as the center.
    Builds ``bbox_aabb`` from ``bbox_min``/``bbox_max`` when both are present,
    else a degenerate point-box at the center.
    """
    out = []
    for o in objects:
        center = o["position"]
        if center is None:
            continue
        bmin, bmax = o.get("bbox_min"), o.get("bbox_max")
        if bmin is not None and bmax is not None:
            bmin = [float(x) for x in bmin]; bmax = [float(x) for x in bmax]
            box = {"min": bmin, "max": bmax,
                   "size": [bmax[i] - bmin[i] for i in range(3)]}
        else:
            box = _point_box(center)
        out.append({
            "label": o["label"],
            "score": float(o.get("confidence", 1.0)),
            "center_3d": [float(x) for x in center],
            "bbox_aabb": box,
        })
    return out


def object_entries_to_eval(entries) -> list[dict]:
    """Adapt authoritative ``ObjectEntry`` GT (VLA-3D ``object_list``) to eval
    objects. ``entries`` may be the dict returned by ``read_objects_from_zip``
    or any iterable of ``ObjectEntry``. AABB = center ± size/2 (heading ignored)."""
    if isinstance(entries, dict):
        entries = entries.values()
    out = []
    for e in entries:
        cx, cy, cz = e.center.x, e.center.y, e.center.z
        hx, hy, hz = e.size.x / 2.0, e.size.y / 2.0, e.size.z / 2.0
        out.append({
            "label": e.label,
            "score": 1.0,
            "center_3d": [cx, cy, cz],
            "bbox_aabb": {"min": [cx - hx, cy - hy, cz - hz],
                          "max": [cx + hx, cy + hy, cz + hz],
                          "size": [e.size.x, e.size.y, e.size.z]},
        })
    return out


def load_gt_from_zip(zip_path: Path, scene_name: str) -> list[dict]:
    """Read authoritative GT objects from a challenge scene zip."""
    from xiao_hei_vln.scene.io import read_objects_from_zip
    return object_entries_to_eval(read_objects_from_zip(zip_path, scene_name=scene_name))


def _load_pred(scene_json: Path) -> list[dict]:
    """Load predicted objects from a scene-graph dump (SceneRepresentation.to_dict)
    or an objectmap.py-style ``{"objects": [...]}`` file (already eval-shaped)."""
    data = json.load(open(scene_json))
    objs = data["objects"]
    # objectmap.py already emits center_3d + bbox_aabb; scene graph emits position.
    if objs and "position" in objs[0]:
        return scene_objects_to_eval(objs)
    return objs


def _print(report, primary):
    print(f"\nGT objects: {report['n_gt']}   Pred objects: {report['n_pred']}   "
          f"classes: {report['classes']}")
    print("\n== mAP ==")
    for k, v in report["mAP"].items():
        print(f"  {k:12s} {v}")
    print("\n== operating point (greedy, per-class, center-distance) ==")
    print(f"  {'thr':10s} {'P':>6s} {'R':>6s} {'F1':>6s}  {'TP':>3s} {'FP':>3s} {'FN':>3s}")
    for k, e in report["operating_point"].items():
        print(f"  {k:10s} {e['precision']:6.3f} {e['recall']:6.3f} {e['f1']:6.3f}  "
              f"{e['tp']:3d} {e['fp']:3d} {e['fn']:3d}")
    pe = report["operating_point"].get(f"dist@{primary}m", {})
    if "mean_center_err_m" in pe:
        print(f"  matched @ {primary}m: center err mean={pe['mean_center_err_m']} "
              f"median={pe['median_center_err_m']} m, mean 3D IoU={pe['mean_3d_iou']}")
    print(f"\n== counting ==  MAE={report['counting_MAE']}  "
          f"exact={report['counting_exact_frac']:.0%}")
    for c, d in sorted(report["counting"].items(), key=lambda kv: -abs(kv[1]['err'])):
        if d["err"]:
            print(f"  {c:22s} gt={d['gt']:2d} pred={d['pred']:2d} (err {d['err']:+d})")
    if report["confusion"]:
        print("\n== label confusion (pred -> gt, dist) ==")
        for p, g, d in report["confusion"]:
            print(f"  {p:18s} -> {g:18s} ({d} m)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path, required=True,
                    help="predicted scene-graph JSON (SceneRepresentation.to_dict) "
                         "or objectmap.py output")
    gt_src = ap.add_mutually_exclusive_group(required=True)
    gt_src.add_argument("--gt-zip", type=Path,
                        help="authoritative GT: a challenge scene zip")
    gt_src.add_argument("--gt", type=Path,
                        help="observable GT: an objectmap.py object-map JSON")
    ap.add_argument("--scene-name", default=None,
                    help="scene name inside --gt-zip (default: zip stem)")
    ap.add_argument("--dist-thresh", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--iou-thresh", type=float, nargs="+", default=[0.25])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    pred = _load_pred(args.scene)
    if args.gt_zip is not None:
        scene_name = args.scene_name or args.gt_zip.stem
        gt = load_gt_from_zip(args.gt_zip, scene_name)
    else:
        gt_objs = json.load(open(args.gt))["objects"]
        gt = scene_objects_to_eval(gt_objs) if (gt_objs and "position" in gt_objs[0]) else gt_objs

    report, primary = evaluate(gt, pred, sorted(args.dist_thresh), sorted(args.iou_thresh))
    _print(report, primary)
    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)
        print("\nwrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
