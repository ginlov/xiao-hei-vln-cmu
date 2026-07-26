#!/usr/bin/env python3
"""Where does a missed object actually die?

We know 58% of ground-truth objects never reach the scene graph. We do not
know *why*, and every candidate fix targets a different stage — a bigger
vocabulary, a finer face split, a better mask, a looser merge rule — so
choosing between them from a single recall number is guessing.

This walks every ground-truth object through the pipeline and records the
furthest stage it reaches:

    in_vocab   the class prior contains its label at all
    observed   some keyframe's lidar actually returned points off it
    seen_2d    a cached detection's mask covers those points
    named      that detection carries the label ground truth uses
    lifted     the detection lifts to within MATCH_M of the true centre
    node       a node survives export within MATCH_M of it
    recalled   that node also carries the right label

Each stage is a *superset* of the next, so the drop between two rows is the
loss attributable to exactly that step.

Visibility is defined by the lidar rather than by projecting the box corners:
a box corner can be inside the vertical FOV while the object sits behind a
wall, and the returns settle occlusion, range and grazing incidence in one
test without a separate ray-cast.

Run from the repo root, on a box that has the corpora::

    uv run python scripts/recall_funnel.py                 # every scene
    uv run python scripts/recall_funnel.py chinese_room    # just one
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

from xiao_hei_vln.perception import replay
from xiao_hei_vln.perception.geometry import (
    EQUIRECT_H,
    EQUIRECT_W,
    project_camera_points_to_equirect,
    sensor_to_camera_transform,
)
from xiao_hei_vln.perception.eval import load_gt_from_zip
from xiao_hei_vln.perception.lifter import PointLifter, _rotation_from_quaternion
from xiao_hei_vln.perception.vocab import DEFAULT_PRIOR, is_structure

DEFAULT_SCENES = (
    "arabic_room", "chinese_room", "japanese_room", "livingroom_3",
    "loft", "office_1", "office_2",
)
GT_ROOT = Path.home() / "workspace/dataset/unity-scene"
BENCH_DIR = Path("artifacts/bench_v3")

MATCH_M = 1.0           # same association radius the benchmark scores with
MIN_SCORE = 0.35        # shipped detector threshold
MIN_LIDAR_PTS = 8       # returns inside the GT box before we call it observed
BOX_PAD_M = 0.05        # GT boxes are tight; lidar lands just outside
COVER_FRAC = 0.30       # mask must cover this much of the object's pixels

STAGES = ("in_vocab", "observed", "seen_2d", "named", "lifted", "node", "recalled")


def _norm(label: str) -> str:
    return label.strip().lower().replace("_", " ")


def _project(xyz_map, position, orientation, R_sc, t_sc):
    """map-frame points -> (u, v, valid, range). Mirrors PointLifter.lift."""
    R_ms = _rotation_from_quaternion(orientation)
    t_ms = np.array([position.x, position.y, position.z])
    xyz_cam = ((xyz_map - t_ms) @ R_ms) @ R_sc.T + t_sc
    u, v, valid = project_camera_points_to_equirect(xyz_cam)
    return u, v, valid, np.linalg.norm(xyz_cam, axis=1)


def _pixels(u, v, valid):
    """Flat equirect pixel indices for the valid subset of a point cloud."""
    ui = np.mod(np.rint(u[valid]).astype(np.int32), EQUIRECT_W)
    vi = np.rint(v[valid]).astype(np.int32)
    keep = (vi >= 0) & (vi < EQUIRECT_H)
    return np.unique(vi[keep] * EQUIRECT_W + ui[keep])


def run_scene(scene: str, frames_dir: Path, gt_zip: Path, scene_json: Path | None):
    gt = [g for g in load_gt_from_zip(gt_zip, scene) if not is_structure(g["label"])]
    if not gt:
        return None

    prior = {_norm(c) for c in DEFAULT_PRIOR}
    lo = np.array([np.array(g["center_3d"]) - np.array(g["bbox_aabb"]["size"]) / 2
                   - BOX_PAD_M for g in gt])
    hi = np.array([np.array(g["center_3d"]) + np.array(g["bbox_aabb"]["size"]) / 2
                   + BOX_PAD_M for g in gt])
    centres = np.array([g["center_3d"] for g in gt])

    # One independent flag per stage, *not* a single "furthest stage" counter.
    # `node` and `recalled` are read off the exported graph, so an object can
    # pass them without passing the per-frame tests above — a counter would
    # then silently backfill the stages in between and report them as passed.
    flags = np.zeros((len(gt), len(STAGES)), dtype=bool)
    flags[:, 0] = [_norm(g["label"]) in prior for g in gt]
    # Diagnostics that only make sense for objects the robot did observe.
    best_px = np.zeros(len(gt), dtype=np.int64)     # largest apparent size
    best_range = np.full(len(gt), np.inf)           # closest approach
    n_frames_seen = np.zeros(len(gt), dtype=np.int64)
    # How much evidence fusion actually had to work with: an object named in
    # two frames out of three hundred is not one it can be blamed for losing.
    n_frames_named = np.zeros(len(gt), dtype=np.int64)

    frames = replay.load_frames(frames_dir, min_move_m=0.15, min_rot_deg=10.0)
    dets_by_tick = replay.load_detections(frames_dir)
    R_sc, t_sc = sensor_to_camera_transform()
    lifter = PointLifter(min_inliers=10, depth_gap_m=0.3)

    for frame in frames:
        scan = frame.scan()[:, :3].astype(np.float64, copy=False)
        if scan.size == 0:
            continue
        dets = [d for d in dets_by_tick.get(frame.tick_id, [])
                if float(d["score"]) >= MIN_SCORE]

        u, v, valid, rng = _project(scan, frame.position, frame.orientation, R_sc, t_sc)
        # Which objects did the lidar hit this frame? Looping the objects and
        # vectorising the points keeps the intermediate at (P,) rather than the
        # (P, G, 3) a fully broadcast test would allocate every frame.
        hits: dict[int, np.ndarray] = {}
        for gi in range(len(gt)):
            sel = ((scan >= lo[gi]) & (scan <= hi[gi])).all(axis=1)
            if int(sel.sum()) >= MIN_LIDAR_PTS:
                hits[gi] = sel
        if not hits:
            continue

        masks = None                       # decoded lazily; the dominant cost
        for gi, hit in hits.items():
            n_frames_seen[gi] += 1
            sel = hit & valid
            if not sel.any():
                continue
            best_range[gi] = min(best_range[gi], float(rng[sel].min()))
            gt_px = _pixels(u, v, sel)
            best_px[gi] = max(best_px[gi], gt_px.size)
            flags[gi, 1] = True                     # observed
            if not dets:
                continue

            if masks is None:
                masks = [(d, replay.decode_mask(d["mask_rle"])) for d in dets]
            want = _norm(gt[gi]["label"])
            need = COVER_FRAC * gt_px.size
            covering = [(det, mask) for det, mask in masks
                        if mask.reshape(-1)[gt_px].sum() >= need]
            if not covering:
                continue
            flags[gi, 2] = True                     # seen_2d
            named = [(d, m) for d, m in covering if _norm(d["label"]) == want]
            if named:
                flags[gi, 3] = True                 # named
                n_frames_named[gi] += 1
            if flags[gi, 4]:                        # lift already proven; skip the cost
                continue
            for det, mask in named:
                res = lifter.lift(mask=mask, scan_points_map=scan,
                                  pose_position=frame.position,
                                  pose_orientation=frame.orientation)
                if res.position is None:
                    continue
                p = np.array([res.position.x, res.position.y, res.position.z])
                if np.linalg.norm(p - centres[gi]) <= MATCH_M:
                    flags[gi, 4] = True             # lifted
                    break

    # Stages 6 and 7 read the exported graph, so they see every filter at once.
    if scene_json is not None and scene_json.is_file():
        objs = [o for o in json.loads(scene_json.read_text())["objects"]
                if not is_structure(o["label"])]
        if objs:
            pos = np.array([o["position"] for o in objs])
            labels = [_norm(o["label"]) for o in objs]
            for gi, g in enumerate(gt):
                d = np.linalg.norm(pos - centres[gi], axis=1)
                near = np.flatnonzero(d <= MATCH_M)
                if near.size:
                    flags[gi, 5] = True                          # node
                    if any(labels[j] == _norm(g["label"]) for j in near):
                        flags[gi, 6] = True                      # recalled

    return dict(scene=scene, n_gt=len(gt), flags=flags, best_px=best_px,
                best_range=best_range, n_frames_seen=n_frames_seen,
                n_frames_named=n_frames_named,
                labels=[_norm(g["label"]) for g in gt],
                diag=np.linalg.norm([g["bbox_aabb"]["size"] for g in gt], axis=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scenes", nargs="*", default=list(DEFAULT_SCENES))
    ap.add_argument("--frames-root", default="frames")
    ap.add_argument("--gt-root", default=str(GT_ROOT))
    ap.add_argument("--bench-dir", default=str(BENCH_DIR))
    ap.add_argument("--out", default="artifacts/recall_funnel.json")
    args = ap.parse_args()

    results = []
    for scene in args.scenes or DEFAULT_SCENES:
        frames_dir = Path(args.frames_root) / f"{scene}_tour"
        if not frames_dir.is_dir():
            print(f"{scene}: no corpus at {frames_dir}", file=sys.stderr)
            continue
        r = run_scene(scene, frames_dir, Path(args.gt_root) / f"{scene}.zip",
                      Path(args.bench_dir) / f"{scene}_scene.json")
        if r is None:
            print(f"{scene}: no GT objects", file=sys.stderr)
            continue
        results.append(r)
        f = r["flags"]
        print(f"{scene:14s} GT={r['n_gt']:3d}  " +
              " ".join(f"{s}={int(f[:, i].sum())}"
                       for i, s in enumerate(STAGES)), flush=True)

    if not results:
        return 1

    total = sum(r["n_gt"] for r in results)
    flags = np.concatenate([r["flags"] for r in results], axis=0)
    print(f"\n=== funnel over {len(results)} scenes, {total} GT objects ===")
    print(f"{'stage':12s} {'passes':>7s} {'of GT':>7s} {'cumul':>7s} "
          f"{'of GT':>7s} {'lost here':>10s}")
    # `passes` is the stage on its own; `cumul` is that stage *and* every one
    # before it, which is the number the funnel logic actually rests on.
    cum = np.ones(total, dtype=bool)
    prev = total
    for i, stage in enumerate(STAGES):
        cum = cum & flags[:, i]
        n, c = int(flags[:, i].sum()), int(cum.sum())
        print(f"{stage:12s} {n:7d} {n / total:7.1%} {c:7d} {c / total:7.1%} "
              f"{prev - c:10d}")
        prev = c

    px = np.concatenate([r["best_px"] for r in results])
    rng = np.concatenate([r["best_range"] for r in results])
    diag = np.concatenate([r["diag"] for r in results])
    named_frames = np.concatenate([r["n_frames_named"] for r in results])
    obs, seen, named = flags[:, 1], flags[:, 2], flags[:, 3]
    lifted, recalled = flags[:, 4], flags[:, 6]
    print(f"\n=== the {int(obs.sum())} objects the lidar observed ===")
    print(f"{'group':22s} {'n':>4s} {'px(med)':>8s} {'range m':>8s} "
          f"{'GT diag':>8s} {'named frm':>10s}")
    for tag, sel in (("never covered by mask", obs & ~seen),
                     ("covered, never named", seen & ~named),
                     ("named, never lifted", named & ~lifted),
                     ("lifted, not recalled", lifted & ~recalled),
                     ("recalled", recalled)):
        if not sel.any():
            continue
        print(f"{tag:22s} {int(sel.sum()):4d} {np.median(px[sel]):8.0f} "
              f"{np.median(rng[sel]):8.2f} {np.median(diag[sel]):8.2f} "
              f"{np.median(named_frames[sel]):10.0f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([
        {"scene": r["scene"], "labels": r["labels"],
         "stages": STAGES, "flags": r["flags"].astype(int).tolist(),
         "best_px": r["best_px"].tolist(),
         "best_range": [None if not np.isfinite(x) else float(x) for x in r["best_range"]],
         "n_frames_seen": r["n_frames_seen"].tolist(),
         "n_frames_named": r["n_frames_named"].tolist(),
         "diag": [float(x) for x in r["diag"]]}
        for r in results], indent=1))
    print(f"\nwrote {out}")

    # Which labels lose the most objects between `seen_2d` and `named`? That
    # is the vocabulary-wording bill, itemised.
    bill: collections.Counter = collections.Counter()
    for r in results:
        f = r["flags"]
        for gi, lab in enumerate(r["labels"]):
            if f[gi, 2] and not f[gi, 3]:
                bill[lab] += 1
    if bill:
        print("\nseen but never named (top 12):",
              ", ".join(f"{k}x{v}" for k, v in bill.most_common(12)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
