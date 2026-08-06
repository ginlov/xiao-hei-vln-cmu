"""Phase-by-phase perception debug for captured viewpoints.

For each viewpoint renders the four pipeline phases so you can see exactly what
happened:
  1. INPUT   — the captured equirectangular RGB.
  2. DETECT  — sidecar masks overlaid (label + score, colour per detection).
  3. LIFT    — per detection: lifted (>= min_inliers) or dropped, inlier count,
               3D position; shown spatially in a top-down (BEV) panel with the
               scan cloud, lifted inlier points, GT objects, and robot pose.
  4. FUSE    — cumulative ObjectMap node count after this viewpoint.

Writes perception_benchmark/debug/<scene>/vp_XXX.png (image+BEV) and vp_XXX.json
(the detection/lift table), plus prints a per-viewpoint table.

Needs the perception sidecar up (localhost:8001).

    uv run --extra perception python perception_benchmark/debug_viewpoint.py --scene livingroom_3
    uv run --extra perception python perception_benchmark/debug_viewpoint.py --scene livingroom_3 --vp 0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from viewgen import load_objects, is_occluder
from replay_score import load_capture, scoreable
from xiao_hei_vln.perception.client import HTTPPerceptionClient
from xiao_hei_vln.perception.lifter import DEFAULT_MIN_INLIERS, PointLifter
from xiao_hei_vln.perception.object_map import ObjectMap
from xiao_hei_vln.perception.scan_accumulator import ScanAccumulator

CAP_DIR = Path(os.environ.get("PERCEPTION_CAP_DIR",
                              "perception_benchmark/captures"))
DEBUG_DIR = Path("perception_benchmark/debug")


def _overlay_masks(img_rgb, dets, lifted_flags, node_ids=None):
    """Blend each detection's mask onto the RGB image; return annotated float img
    + list of (u, v, text, color) label anchors.

    ``node_ids`` (optional) is the ObjectMap node each detection fused into.
    When given, the label is prefixed with ``#<id>`` — the same number the
    viewer prints on the 3D box, so a mask on the image can be matched to a
    box in the scene by eye.
    """
    out = img_rgb.astype(np.float32) / 255.0
    cmap = plt.cm.tab20
    anchors = []
    for i, (d, ok) in enumerate(zip(dets, lifted_flags)):
        color = np.array(cmap(i % 20)[:3])
        m = d.mask
        out[m] = 0.5 * out[m] + 0.5 * color               # translucent fill
        ys, xs = np.nonzero(m)
        if len(xs):
            u, v = int(xs.mean()), int(ys.mean())
            mark = "OK" if ok else "x"
            nid = node_ids[i] if node_ids is not None else None
            tag = f"#{nid} " if nid is not None else ""
            anchors.append((u, v, f"{tag}{d.label} {d.score:.2f} [{mark}]", color))
    return np.clip(out, 0, 1), anchors


def debug_scene(scene, *, base_url, score_threshold, min_inliers, accumulate,
                only_vp, request_timeout_s=60.0):
    vp_dirs = sorted(glob.glob(str(CAP_DIR / scene / "vp_*")))
    if only_vp is not None:
        vp_dirs = [d for d in vp_dirs if d.endswith(f"vp_{only_vp:03d}")]
    if not vp_dirs:
        print(f"[{scene}] no captures (or vp {only_vp}) under {CAP_DIR/scene}")
        return

    objs = load_objects(scene)
    keep = {i: e for i, e in objs.items() if scoreable(e.label)}
    classes = tuple(sorted({e.label for e in keep.values()}))
    gt = np.array([[e.center.x, e.center.y] for e in keep.values()]) if keep else np.empty((0, 2))

    client = HTTPPerceptionClient(base_url=base_url, request_timeout_s=request_timeout_s)
    client.wait_until_ready()
    client.set_classes(classes)
    lifter = PointLifter(min_inliers=min_inliers)
    omap = ObjectMap()
    accum = ScanAccumulator() if accumulate else None

    out_dir = DEBUG_DIR / scene
    out_dir.mkdir(parents=True, exist_ok=True)

    for vp_dir in vp_dirs:
        vid = os.path.basename(vp_dir)
        img, scan, pos, ori = load_capture(Path(vp_dir))
        cloud = accum.update(scan, pos, ori) if accum is not None else scan
        dets = client.detect(img, score_threshold=score_threshold)

        # LIFT each detection, record the phase-3 result
        recs, lifted_flags, lift_pts = [], [], []
        for d in dets:
            res = lifter.lift(d.mask, cloud, pos, ori)
            ok = res.position is not None
            recs.append({"label": d.label, "score": round(float(d.score), 3),
                         "n_inliers": int(res.n_inliers), "lifted": ok,
                         "position": [round(float(x), 3) for x in
                                      (res.position.x, res.position.y, res.position.z)] if ok else None})
            lifted_flags.append(ok)
            if ok:
                lift_pts.append((d.label, res.inlier_points))
                omap.add(d.label, d.score, res.inlier_points)

        n_ok = sum(lifted_flags)
        # ---- render the 4 phases ----
        fig = plt.figure(figsize=(20, 11))
        # phase 1+2: equirect image with masks
        ax_img = fig.add_axes([0.03, 0.55, 0.94, 0.42])
        overlay, anchors = _overlay_masks(img[:, :, ::-1], dets, lifted_flags)  # BGR->RGB
        ax_img.imshow(overlay)
        for u, v, txt, color in anchors:
            ax_img.text(u, v, txt, fontsize=7, color="white", ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.1", fc=color * 0.7, ec="none", alpha=0.8))
        ax_img.set_title(f"{scene} {vid}  —  PHASE 1 INPUT + PHASE 2 DETECT "
                         f"({len(dets)} detections)", fontsize=11)
        ax_img.axis("off")

        # phase 3: BEV lift
        ax_bev = fig.add_axes([0.06, 0.05, 0.6, 0.45])
        ax_bev.scatter(cloud[:, 0], cloud[:, 1], s=1, c="0.8", linewidths=0, label="scan cloud")
        if len(gt):
            ax_bev.scatter(gt[:, 0], gt[:, 1], s=40, marker="x", c="black", linewidths=1.2, label="GT objects")
        cmap = plt.cm.tab20
        for i, (lab, pts) in enumerate(lift_pts):
            ax_bev.scatter(pts[:, 0], pts[:, 1], s=6, color=cmap(i % 20), linewidths=0)
            c = pts[:, :2].mean(0)
            ax_bev.scatter([c[0]], [c[1]], s=90, color=cmap(i % 20), edgecolors="black", linewidths=1, zorder=5)
        ax_bev.scatter([pos.x], [pos.y], s=260, marker="*", c="red", edgecolors="black",
                       linewidths=1.2, zorder=6, label="robot")
        ax_bev.set_aspect("equal"); ax_bev.grid(alpha=0.2); ax_bev.legend(loc="upper right", fontsize=8)
        ax_bev.set_title(f"PHASE 3 LIFT  —  {n_ok}/{len(dets)} lifted (min_inliers={min_inliers}); "
                         f"PHASE 4 FUSE cumulative nodes={len(omap.nodes)}", fontsize=11)
        ax_bev.set_xlabel("x (m, map)"); ax_bev.set_ylabel("y (m, map)")

        # phase 3 table (text panel)
        ax_tbl = fig.add_axes([0.68, 0.05, 0.30, 0.45]); ax_tbl.axis("off")
        lines = [f"{'label':16s}{'score':>6s}{'inl':>5s}{'lift':>5s}"]
        for r in sorted(recs, key=lambda r: (-r["lifted"], -r["score"]))[:34]:
            lines.append(f"{r['label'][:16]:16s}{r['score']:6.2f}{r['n_inliers']:5d}"
                         f"{'  OK' if r['lifted'] else '   x':>5s}")
        ax_tbl.text(0, 1, "\n".join(lines), family="monospace", fontsize=7, va="top")

        fig.savefig(out_dir / f"{vid}.png", dpi=95)
        plt.close(fig)
        json.dump({"scene": scene, "viewpoint": vid, "pose": [pos.x, pos.y, pos.z],
                   "n_detections": len(dets), "n_lifted": n_ok,
                   "cumulative_nodes": len(omap.nodes), "detections": recs},
                  open(out_dir / f"{vid}.json", "w"), indent=2)
        print(f"[{scene}] {vid}: {len(dets)} det, {n_ok} lifted, {len(omap.nodes)} nodes -> {out_dir/(vid+'.png')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--vp", type=int, default=None, help="single viewpoint id (default: all)")
    ap.add_argument("--base-url", default=os.environ.get("XIAO_HEI_PERCEPTION_BASE_URL", "http://localhost:8001"))
    ap.add_argument("--score-threshold", type=float, default=0.25)
    ap.add_argument("--min-inliers", type=int, default=DEFAULT_MIN_INLIERS)
    ap.add_argument("--no-accumulate", dest="accumulate", action="store_false")
    ap.set_defaults(accumulate=True)
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()
    debug_scene(args.scene, base_url=args.base_url, score_threshold=args.score_threshold,
                min_inliers=args.min_inliers, accumulate=args.accumulate,
                only_vp=args.vp, request_timeout_s=args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
