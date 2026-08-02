"""Offline scene-graph build + score (Part 2 of the perception benchmark).

Replays a scene's captured viewpoints (image + registered scan + pose) through
the SAME perception components the live pipeline uses — detect (sidecar) → 3D
lift → cross-frame ObjectMap fusion — accumulating one scene graph, then scores
it against the scene's authoritative object_list via perception/eval.py.

Because the inputs are frozen (captures/<scene>/), this is fully reproducible
and lets you A/B perception configs (score_threshold, min_inliers, fusion
params) against identical data without touching the sim.

Requires the perception sidecar running (detection):
    XIAO_HEI_RESPONDER=perception docker/run up -d   # or: docker start xiao_hei_perception

    uv run --extra perception python perception_benchmark/replay_score.py --scene livingroom_3
    uv run --extra perception python perception_benchmark/replay_score.py --all
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

from clustering import STRATEGIES
from viewgen import load_objects, is_occluder
from xiao_hei_vln.messages.common import Quaternion, Vector3
from xiao_hei_vln.perception.client import HTTPPerceptionClient
from xiao_hei_vln.perception.eval import evaluate, object_entries_to_eval
from xiao_hei_vln.perception.lifter import (DEFAULT_CLUSTER_VOXEL_M,
                                            DEFAULT_MIN_INLIERS,
                                            DEFAULT_RANGE_GAP_M, PointLifter)
from xiao_hei_vln.perception.object_map import NMS_DIST, NMS_GAP, ObjectMap
from xiao_hei_vln.perception.scan_accumulator import ScanAccumulator

CAP_DIR = Path(os.environ.get("PERCEPTION_CAP_DIR",
                              "perception_benchmark/captures"))
DEFAULT_BASE_URL = "http://localhost:8001"


def scoreable(label: str) -> bool:
    """Real detectable objects only — drop architecture and the unknown filler,
    which the detector can't meaningfully find and which otherwise skew P/R."""
    lo = label.lower()
    return (lo != "unknown" and "floor" not in lo and "ceiling" not in lo
            and not is_occluder(label))


def load_capture(vp_dir: Path):
    img = np.load(vp_dir / "image.npy")                 # (640,1920,3) bgr8
    scan = np.load(vp_dir / "registered_scan.npy")      # (N,4) map frame
    pose = json.load(open(vp_dir / "pose.json"))
    px, py, pz = pose["position"]
    qx, qy, qz, qw = pose["orientation_xyzw"]
    return img, scan, Vector3(x=px, y=py, z=pz), Quaternion(x=qx, y=qy, z=qz, w=qw)


def build_and_score(scene: str, *, base_url: str, score_threshold: float,
                    min_inliers: int, keep_arch: bool, seed: int, out_dir: Path,
                    accumulate: bool = True, request_timeout_s: float = 60.0,
                    range_gap_m: float = DEFAULT_RANGE_GAP_M,
                    cluster_voxel_m: float = DEFAULT_CLUSTER_VOXEL_M,
                    inlier_filter=None,
                    nms_dist: float = NMS_DIST, nms_gap: float = NMS_GAP,
                    verbose: bool = True):
    np.random.seed(seed)                                # ObjectMap PTS_CAP subsample
    vp_dirs = sorted(glob.glob(str(CAP_DIR / scene / "vp_*")))
    if not vp_dirs:
        print(f"[{scene}] no captures under {CAP_DIR/scene} — run Part 1 first.")
        return None

    objs = load_objects(scene)
    keep = objs if keep_arch else {i: e for i, e in objs.items() if scoreable(e.label)}
    classes = tuple(sorted({e.label for e in keep.values()}))

    # Generous timeout: the sidecar's default 2s is too short for a cold
    # reload_classes (text-encoder warmup) and for SAM on a 1920x640 image.
    client = HTTPPerceptionClient(base_url=base_url, request_timeout_s=request_timeout_s)
    client.wait_until_ready()
    if not client.set_classes(classes):
        print(f"[{scene}] WARNING: set_classes did not confirm — detections may be empty")
    lifter = PointLifter(min_inliers=min_inliers,       # z-buffer on by default
                         range_gap_m=range_gap_m, cluster_voxel_m=cluster_voxel_m,
                         inlier_filter=inlier_filter)
    omap = ObjectMap(nms_dist=nms_dist, nms_gap=nms_gap)
    # Production configuration: densify the scan across keyframes BEFORE lifting
    # (ScanAccumulator) AND fuse the lifted clouds across frames (ObjectMap).
    # Fed in capture order == the live tick order. Fresh accumulator per scene.
    accum = ScanAccumulator() if accumulate else None

    n_det = n_lift = 0
    for vp_dir in vp_dirs:
        img, scan, pos, ori = load_capture(Path(vp_dir))
        cloud = accum.update(scan, pos, ori) if accum is not None else scan
        dets = client.detect(img, score_threshold=score_threshold)
        n_det += len(dets)
        for det in dets:
            res = lifter.lift(det.mask, cloud, pos, ori)
            if res.position is not None:
                omap.add(det.label, det.score, res.inlier_points)
                n_lift += 1
        if verbose:
            print(f"  {os.path.basename(vp_dir)}: {len(dets)} detections, "
                  f"{n_lift} lifted cumulative, {len(omap.nodes)} nodes")

    pred = omap.export()
    gt = object_entries_to_eval(keep)
    report, primary = evaluate(gt, pred, [0.5, 1.0, 2.0], [0.25])

    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump({"scene": scene, "objects": pred}, open(out_dir / f"{scene}_scene.json", "w"))
    json.dump(report, open(out_dir / f"{scene}_metrics.json", "w"), indent=2)

    if verbose:
        print(f"\n[{scene}] viewpoints={len(vp_dirs)} detections={n_det} "
              f"lifts={n_lift} | GT(scoreable)={report['n_gt']} pred={report['n_pred']}")
        m, op = report["mAP"], report["operating_point"].get(f"dist@{primary}m", {})
        print(f"  mAP  d0.5={m['dist@0.5m']}  d1.0={m['dist@1.0m']}  d2.0={m['dist@2.0m']}  "
              f"iou@0.25={m['iou@0.25']}")
        print(f"  @{primary}m  P={op.get('precision')} R={op.get('recall')} F1={op.get('f1')}  "
              f"center_err={op.get('mean_center_err_m')}m")
        print(f"  counting MAE={report['counting_MAE']}  exact={report['counting_exact_frac']:.0%}")
    report["n_viewpoints"] = len(vp_dirs)
    report["n_detections"] = n_det
    report["n_lifts"] = n_lift
    return report


def _row(scene: str, r: dict) -> dict:
    """Flatten one scene's report to a summary row."""
    op = r["operating_point"].get("dist@1.0m", {})
    m = r["mAP"]
    return {
        "scene": scene,
        "n_viewpoints": r.get("n_viewpoints"),
        "n_gt": r["n_gt"], "n_pred": r["n_pred"],
        "detections": r.get("n_detections"), "lifts": r.get("n_lifts"),
        "mAP_dist_0.5m": m["dist@0.5m"], "mAP_dist_1.0m": m["dist@1.0m"],
        "mAP_dist_2.0m": m["dist@2.0m"], "mAP_iou_0.25": m["iou@0.25"],
        "precision@1m": op.get("precision"), "recall@1m": op.get("recall"),
        "f1@1m": op.get("f1"), "center_err_m": op.get("mean_center_err_m"),
        "counting_MAE": r["counting_MAE"], "counting_exact_frac": r["counting_exact_frac"],
    }


def write_results(summary: list[tuple[str, dict]], out_dir: Path, params: dict) -> None:
    """Write results.csv / results.json / results.md over all scored scenes."""
    import csv
    import statistics as st
    rows = [_row(s, r) for s, r in summary]
    cols = list(rows[0].keys())

    # per-scene CSV
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)

    # macro-average across scenes (numeric cols, skipping None)
    def avg(k):
        vals = [row[k] for row in rows if isinstance(row[k], (int, float))]
        return round(st.mean(vals), 4) if vals else None
    agg = {k: avg(k) for k in cols if k != "scene"}

    # full JSON: params + per-scene full reports + aggregate
    json.dump({"params": params,
               "scenes": {s: r for s, r in summary},
               "summary_rows": rows,
               "aggregate_macro_avg": agg},
              open(out_dir / "results.json", "w"), indent=2)

    # Markdown table
    hdr = ["scene", "vps", "gt", "pred", "mAP@1", "P@1", "R@1", "F1@1", "cErr(m)", "cMAE"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in [
            row["scene"], row["n_viewpoints"], row["n_gt"], row["n_pred"],
            row["mAP_dist_1.0m"], row["precision@1m"], row["recall@1m"],
            row["f1@1m"], row["center_err_m"], row["counting_MAE"]]) + " |")
    lines.append("| **macro-avg** | | | | " + f"{agg['mAP_dist_1.0m']} | {agg['precision@1m']} | "
                 f"{agg['recall@1m']} | {agg['f1@1m']} | {agg['center_err_m']} | {agg['counting_MAE']} |")
    (out_dir / "results.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote results.csv / results.json / results.md to {out_dir}/")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--base-url", default=os.environ.get("XIAO_HEI_PERCEPTION_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--score-threshold", type=float, default=0.25)
    ap.add_argument("--min-inliers", type=int, default=DEFAULT_MIN_INLIERS)
    ap.add_argument("--keep-arch", action="store_true",
                    help="score against ALL objects incl. wall/floor/ceiling (default drops them)")
    ap.add_argument("--no-accumulate", dest="accumulate", action="store_false",
                    help="disable ScanAccumulator densification (lift each single sweep)")
    ap.set_defaults(accumulate=True)
    ap.add_argument("--range-gap", type=float, default=DEFAULT_RANGE_GAP_M,
                    help="range-cluster gap (m) for mask-spill rejection; 0 disables")
    ap.add_argument("--cluster-voxel", type=float, default=DEFAULT_CLUSTER_VOXEL_M,
                    help="voxel size (m) for connected-component clustering; 0 disables")
    ap.add_argument("--cluster", choices=sorted(STRATEGIES), default="none",
                    help="inlier-filter strategy (see clustering.py)")
    ap.add_argument("--cluster-eps", type=float, default=None,
                    help="strategy radius/gap override (m)")
    ap.add_argument("--nms-dist", type=float, default=NMS_DIST,
                    help="cross-label suppression: max centre distance (m); 0 disables")
    ap.add_argument("--nms-gap", type=float, default=NMS_GAP,
                    help="cross-label suppression: max box-surface gap (m); 0 disables")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=60.0, help="per-request HTTP timeout (s)")
    ap.add_argument("--out", type=Path, default=Path("perception_benchmark/scores"))
    args = ap.parse_args()

    if args.all:
        scenes = sorted(os.path.basename(os.path.dirname(m))
                        for m in glob.glob(str(CAP_DIR / "*" / "manifest.json")))
    elif args.scene:
        scenes = [args.scene]
    else:
        ap.error("pass --scene <name> or --all")

    factory = STRATEGIES[args.cluster]
    ifilter = (factory() if args.cluster == "none" or args.cluster_eps is None
               else factory(args.cluster_eps))

    summary = []
    for s in scenes:
        print(f"\n======== {s} ========")
        rep = build_and_score(
            s, base_url=args.base_url, score_threshold=args.score_threshold,
            min_inliers=args.min_inliers, keep_arch=args.keep_arch,
            seed=args.seed, out_dir=args.out, accumulate=args.accumulate,
            range_gap_m=args.range_gap, cluster_voxel_m=args.cluster_voxel,
            inlier_filter=ifilter, nms_dist=args.nms_dist, nms_gap=args.nms_gap,
            request_timeout_s=args.timeout)
        if rep is not None:
            summary.append((s, rep))

    if summary:
        print(f"\n{'scene':18s} {'mAP@1':>6s} {'F1@1':>6s} {'cErr':>5s} {'cMAE':>5s} {'gt':>4s} {'pred':>4s}")
        for s, r in summary:
            op = r["operating_point"].get("dist@1.0m", {})
            print(f"{s:18s} {str(r['mAP']['dist@1.0m']):>6s} {str(op.get('f1')):>6s} "
                  f"{str(op.get('mean_center_err_m')):>5s} {str(r['counting_MAE']):>5s} "
                  f"{r['n_gt']:4d} {r['n_pred']:4d}")
        params = {"score_threshold": args.score_threshold, "min_inliers": args.min_inliers,
                  "keep_arch": args.keep_arch, "accumulate": args.accumulate,
                  "range_gap_m": args.range_gap, "cluster_voxel_m": args.cluster_voxel,
                  "cluster": args.cluster,
                  "nms_dist": args.nms_dist, "nms_gap": args.nms_gap,
                  "cluster_eps": args.cluster_eps, "seed": args.seed}
        write_results(summary, args.out, params)
    print(f"\nper-scene scores + results.* written to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
