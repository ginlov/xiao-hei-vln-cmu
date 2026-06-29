"""Benchmark a predicted 3D object map against the ground-truth object map.

Both maps are produced by ``objectmap.py`` over the SAME frames:

    # GT map (free sim semantics, no torch):
    uv run python perception/objectmap.py --frames 'captures/run3/frames/*' \
        --source gt   --out /tmp/gt.json
    # predicted map (YOLO-World v2 + SAM2.1, needs torch+GPU — run in container):
    python perception/objectmap.py --frames 'captures/run3/frames/*' \
        --source yolo --out /tmp/pred.json

    # then, anywhere (pure numpy):
    uv run python perception/eval_objectmap.py --gt /tmp/gt.json --pred /tmp/pred.json

The GT here is the *observable* GT — objects the trajectory actually saw — so we
don't penalize the detector for objects it never had a chance to see.

Metrics
-------
Primary  : mAP @ center-distance (nuScenes-style; AABB boxes + thin objects make
           3D IoU brutally strict, so centre distance is the fairer primary).
Aux      : mAP @ 3D IoU (0.25), operating-point precision/recall/F1, per-class
           counting error (numerical-question proxy), matched centre error +
           mean 3D IoU, and a label-confusion list (geometric match, label
           disagreement).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

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


def _print(report, primary):
    print(f"\nGT objects: {report['n_gt']}   Pred objects: {report['n_pred']}   "
          f"classes: {report['classes']}")
    print("\n== mAP ==")
    for k, v in report["mAP"].items():
        print(f"  {k:12s} {v}")
    print(f"\n== operating point (greedy, per-class, center-distance) ==")
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", type=Path, required=True, help="GT object-map json")
    ap.add_argument("--pred", type=Path, required=True, help="predicted object-map json")
    ap.add_argument("--dist-thresh", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--iou-thresh", type=float, nargs="+", default=[0.25])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    gt = json.load(open(args.gt))["objects"]
    pred = json.load(open(args.pred))["objects"]
    report, primary = evaluate(gt, pred, sorted(args.dist_thresh), sorted(args.iou_thresh))
    _print(report, primary)
    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)
        print("\nwrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
