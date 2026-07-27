"""Which points do we take, and which do we leave behind?

TASK 23 measured a lidar oracle at 1.622 of 2 against our 0.191: with the
points selected perfectly the same scans, the same registration and the same
boxes score eight times what they do now. That puts every other repair --
fusion rules, box estimators, vocabulary -- behind one question, and this
script answers it. For every detection that lands on a real object:

    precision = |ours & oracle| / |ours|     how much background we grabbed
    recall    = |ours & oracle| / |oracle|   how much of the object we missed

``oracle`` is every return in the *same scan* that falls inside the
ground-truth box, so both numbers are about selection alone -- the sensor, the
pose and the object are held fixed. Each is measured twice, once on the mask's
returns after the z-buffer gate and once after the dominant-depth-cluster
filter, because the repair differs by regime:

    low  P, high R   mask overshoots, or the cluster keeps the background
                     -> tighten depth_gap_m, erode the mask
    high P, low  R   the cluster cuts into the object
                     -> loosen it
    low  P, low  R   the mask is on the wrong object
                     -> the segmenter is the problem

Only the third row justifies replacing SAM 2.1 Hiera Tiny, which is the
expensive guess and the one everybody reaches for first.

The selection is reimplemented here rather than called, because
``PointLifter.lift`` returns only the surviving points and this needs the two
intermediate sets. It projects once per *frame* instead of once per detection,
which is also why it is minutes rather than an hour. ``--self-check`` replays
the first N detections through the real lifter and asserts the final sets
agree; run it after any change to lifter.py.

Usage::

    python scripts/point_audit.py                       # all seven scenes
    python scripts/point_audit.py japanese_room --self-check 200
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np

from xiao_hei_vln.perception import replay
from xiao_hei_vln.perception.eval import iou_3d, load_gt_from_zip
from xiao_hei_vln.perception.geometry import (
    EQUIRECT_H,
    EQUIRECT_W,
    project_camera_points_to_equirect,
    sensor_to_camera_transform,
)
from xiao_hei_vln.perception.lifter import (
    DEFAULT_DEPTH_GAP_M,
    ZBUF_TOL_M,
    _dominant_depth_cluster,
    _rotation_from_quaternion,
)
from xiao_hei_vln.perception.vocab import is_structure

SCENES = ["arabic_room", "chinese_room", "japanese_room", "livingroom_3",
          "loft", "office_1", "office_2"]
GT_ROOT = Path.home() / "workspace/dataset/unity-scene"

MIN_MOVE_M = 0.15        # the benchmark's keyframe thinning, so the population
MIN_ROT_DEG = 10.0       # audited is the population scored
MIN_SCORE = 0.35         # responder.DEFAULT_SCORE_THRESHOLD
MIN_INLIERS = 10         # lifter.DEFAULT_MIN_INLIERS
PAD = 0.05               # GT boxes are tight; returns land just outside
MIN_ORACLE_PTS = 8       # below this the object is not really in this scan

SIZE_BINS = ((0.0, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 1.8), (1.8, 1e3))

# Ground truth labels some objects `unknown`, and their boxes are not objects:
# an overlay of one showed a single `unknown` AABB spanning half of
# japanese_room, its "oracle" returns smeared across every wall in the scene.
# TASK 23 already excluded them from scoring (53 instances, 8.9% of GT); left
# in here they dominate the largest size bin and drag every average with it.
DROP_LABELS = {"unknown"}


def _scoreable(label: str) -> bool:
    return label.strip().lower().replace("_", " ") not in DROP_LABELS


# ---------------------------------------------------------------------------
# The selection, opened up
# ---------------------------------------------------------------------------


def project_frame(xyz_map, pose_position, pose_orientation, R_sc, t_sc):
    """Everything in ``PointLifter.lift`` that does not depend on the mask.

    Returns ``(front, vi, ui, depth)``: the z-buffer survivors, their equirect
    pixel indices, and their camera-frame ranges.
    """
    R_ms = _rotation_from_quaternion(pose_orientation)
    t_ms = np.array([pose_position.x, pose_position.y, pose_position.z])
    xyz_cam = ((xyz_map - t_ms) @ R_ms) @ R_sc.T + t_sc

    depth = np.linalg.norm(xyz_cam, axis=1)
    u, v, in_fov = project_camera_points_to_equirect(xyz_cam)
    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    np.mod(ui, EQUIRECT_W, out=ui)
    keep = in_fov & (vi >= 0) & (vi < EQUIRECT_H)

    vclip = np.clip(vi, 0, EQUIRECT_H - 1)
    fidx = vclip * EQUIRECT_W + ui
    nearest = np.full(EQUIRECT_H * EQUIRECT_W, np.inf)
    np.minimum.at(nearest, fidx[keep], depth[keep])
    front = keep & (depth <= nearest[fidx] + ZBUF_TOL_M)
    return keep, front, vi, ui, depth


def select(mask, front, vi, ui, depth):
    """``(idx_mask, idx_final)`` — scan indices before and after clustering.

    ``idx_final is idx_mask`` when the cluster would leave too few points, which
    is what the lifter does: the filter is allowed to refuse, not to starve.
    """
    in_mask = np.zeros(front.shape, dtype=bool)
    in_mask[front] = mask[vi[front], ui[front]]
    idx_mask = np.flatnonzero(in_mask)
    if idx_mask.size < MIN_INLIERS:
        return idx_mask, idx_mask
    sel = _dominant_depth_cluster(depth[idx_mask], DEFAULT_DEPTH_GAP_M)
    idx_final = idx_mask[sel] if int(sel.sum()) >= MIN_INLIERS else idx_mask
    return idx_mask, idx_final


def _bias(ours, gt_centre, gt_size, eye):
    """Split the centre error into "too near" and "sideways".

    A cloud taken off an object's front surface has its box centred on that
    surface, half the object's depth short of the true centre, and the
    direction of that error is the viewing ray -- not a fixed direction in the
    map, which is why it survives a median over observations taken from one
    arc. Projecting the error onto the ray separates it from ordinary lateral
    noise: a positive ``radial`` means the truth is further away than we put
    it. ``radial_rel`` divides by the object's own depth along that ray, so a
    value near 0.5 is exactly the front-surface signature.
    """
    c = (ours.max(axis=0) + ours.min(axis=0)) / 2
    eye = np.array([eye.x, eye.y, eye.z])
    u = c - eye
    n = float(np.linalg.norm(u))
    if n < 1e-6:
        return dict(radial=0.0, lateral=0.0, radial_rel=0.0)
    u = u / n
    d = gt_centre - c
    r = float(d @ u)
    depth = float(np.abs(gt_size @ np.abs(u)))       # GT extent along the ray
    return dict(radial=r, lateral=float(np.linalg.norm(d - r * u)),
                radial_rel=r / max(depth, 1e-6))


def dist_to_box(pts, lo, hi):
    """Euclidean distance from each point to an AABB; 0 inside."""
    d = np.maximum(np.maximum(lo - pts, pts - hi), 0.0)
    return np.linalg.norm(d, axis=1)


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def audit_scene(scene, frames_root, self_check=0):
    gt = [g for g in load_gt_from_zip(GT_ROOT / f"{scene}.zip", scene)
          if not is_structure(g["label"]) and _scoreable(g["label"])]
    if not gt:
        return [], collections.Counter()

    gc = np.array([g["center_3d"] for g in gt], dtype=np.float64)
    gs = np.array([g["bbox_aabb"]["size"] for g in gt], dtype=np.float64)
    glo, ghi = gc - gs / 2, gc + gs / 2
    plo, phi = glo - PAD, ghi + PAD
    diag = np.linalg.norm(gs, axis=1)

    frames = replay.load_frames(frames_root / f"{scene}_tour",
                                min_move_m=MIN_MOVE_M, min_rot_deg=MIN_ROT_DEG)
    detections = replay.load_detections(frames_root / f"{scene}_tour")
    R_sc, t_sc = sensor_to_camera_transform()

    lifter = None
    if self_check:
        from xiao_hei_vln.perception.lifter import PointLifter
        lifter = PointLifter(min_inliers=MIN_INLIERS)

    rows, tally = [], collections.Counter()
    checked = mismatched = 0
    # Every observation's points, filed under the object it landed on. Pooling
    # them is an oracle over grouping and estimation *only* -- the masks stay
    # ours -- so the box it yields is the ceiling of all remaining downstream
    # work, and unlike the lidar oracle it is a box we could actually build.
    pools: dict[int, list] = collections.defaultdict(list)

    for frame in frames:
        dets = [d for d in detections.get(frame.tick_id, [])
                if float(d["score"]) >= MIN_SCORE]
        if not dets:
            continue
        scan = frame.scan()
        if scan.size == 0:
            continue
        xyz = scan[:, :3].astype(np.float64, copy=False)

        _keep, front, vi, ui, depth = project_frame(
            xyz, frame.position, frame.orientation, R_sc, t_sc)

        # (N, G) membership in every padded GT box, computed once per frame:
        # it gives both the oracle sets and the per-detection intersections.
        ins = ((xyz[:, None, :] >= plo[None]) & (xyz[:, None, :] <= phi[None])
               ).all(axis=2)
        n_oracle = ins.sum(axis=0)
        visible = n_oracle >= MIN_ORACLE_PTS

        for det in dets:
            mask = replay.decode_mask(det["mask_rle"])
            idx_mask, idx_final = select(mask, front, vi, ui, depth)
            tally["detections"] += 1
            if idx_final.size < MIN_INLIERS:
                tally["dropped_min_inliers"] += 1
                continue
            tally["lifted"] += 1

            if lifter is not None and checked < self_check:
                ref = lifter.lift(mask=mask, scan_points_map=scan,
                                  pose_position=frame.position,
                                  pose_orientation=frame.orientation)
                got = np.sort(xyz[idx_final], axis=0)
                want = np.sort(ref.inlier_points, axis=0)
                if got.shape != want.shape or not np.allclose(got, want):
                    mismatched += 1
                checked += 1

            hit_final = ins[idx_final]                    # (M, G)
            cnt = hit_final.sum(axis=0)
            cnt = np.where(visible, cnt, 0)
            gi = int(np.argmax(cnt))
            if cnt[gi] == 0:
                tally["no_gt_overlap"] += 1
                tally[f"nogt::{det['label']}"] += 1
                continue
            tally["matched"] += 1

            ours = xyz[idx_final]
            pools[gi].append(ours)
            inter_f = int(cnt[gi])
            inter_m = int(ins[idx_mask, gi].sum())
            outside = dist_to_box(ours[~hit_final[:, gi]], plo[gi], phi[gi])

            rows.append(dict(
                scene=scene, gi=gi, label=det["label"], diag=float(diag[gi]),
                n_mask=int(idx_mask.size), n_ours=int(idx_final.size),
                n_oracle=int(n_oracle[gi]),
                p_mask=inter_m / max(idx_mask.size, 1),
                r_mask=inter_m / max(int(n_oracle[gi]), 1),
                p=inter_f / idx_final.size,
                r=inter_f / max(int(n_oracle[gi]), 1),
                iou_ours=iou_3d(ours.min(axis=0), ours.max(axis=0),
                                glo[gi], ghi[gi]),
                iou_oracle=iou_3d(xyz[ins[:, gi]].min(axis=0),
                                  xyz[ins[:, gi]].max(axis=0),
                                  glo[gi], ghi[gi]),
                far=float(np.median(outside)) if outside.size else 0.0,
                far_frac=float(np.mean(outside > 0.5)) if outside.size else 0.0,
                # How much of the ground-truth box our points actually span.
                # High precision with low IoU can only mean an undersized box,
                # and the per-axis span says whether it is undersized in one
                # direction (we see one face) or in all three (we see a part).
                span=float(np.mean((ours.max(axis=0) - ours.min(axis=0))
                                   / np.maximum(gs[gi], 1e-6))),
                span_min=float(np.min((ours.max(axis=0) - ours.min(axis=0))
                                      / np.maximum(gs[gi], 1e-6))),
                span_max=float(np.max((ours.max(axis=0) - ours.min(axis=0))
                                      / np.maximum(gs[gi], 1e-6))),
                **_bias(ours, gc[gi], gs[gi], frame.position),
            ))

    if self_check:
        print(f"  self-check: {checked} detections, {mismatched} mismatched",
              flush=True)

    ceil = []
    for gi, chunks in pools.items():
        pts = np.vstack(chunks)
        raw = iou_3d(pts.min(axis=0), pts.max(axis=0), glo[gi], ghi[gi])
        trim = iou_3d(np.percentile(pts, 5, axis=0), np.percentile(pts, 95, axis=0),
                      glo[gi], ghi[gi])
        ceil.append(dict(scene=scene, gi=gi, diag=float(diag[gi]),
                         raw=raw, trim=trim, n=len(chunks)))
    return rows, tally, ceil


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def quadrants(rows, pt=0.5, rt=0.5):
    q = collections.Counter()
    for r in rows:
        q[(r["p"] >= pt, r["r"] >= rt)] += 1
    return q


def report_ceiling(ceil, rows, n_gt):
    """What our own masks allow, once grouping is taken out of the picture.

    Every observation is filed under the object it actually landed on -- an
    oracle over fusion -- and only the box estimator varies. ``pooled`` and
    ``trimmed`` build one AABB from all of that object's points; ``best view``
    keeps the single observation whose own box scores highest, which upper
    bounds any estimator that chooses among observations rather than blending
    them. All three are scored over *every* ground-truth object, so an object
    no detection reached counts as a zero and the numbers can be read against
    each other directly.
    """
    if not ceil:
        return
    per = collections.defaultdict(float)
    for r in rows:
        k = (r["scene"], r["gi"])
        per[k] = max(per[k], r["iou_ours"])
    best = np.array(list(per.values()))
    raw = np.array([c["raw"] for c in ceil])
    trim = np.array([c["trim"] for c in ceil])
    diag = np.array([c["diag"] for c in ceil])
    print(f"\n=== what our own masks allow ({len(ceil)} of {n_gt} ground-truth "
          f"objects reached) ===")
    print(f"{'variant':12s} {'mIoU':>6s} {'>=.25':>7s} {'>=.5':>7s} {'score/2':>8s}")
    for name, v in (("pooled", raw), ("trimmed", trim), ("best view", best)):
        p25 = float(np.sum(v >= .25) / n_gt)
        p50 = float(np.sum(v >= .5) / n_gt)
        print(f"{name:12s} {v.sum() / n_gt:6.3f} {p25:7.1%} {p50:7.1%} "
              f"{p50 + p25:8.3f}")
    print(f"\n{'GT diagonal':14s} {'n':>5s} {'pooled':>7s} {'trimmed':>8s} "
          f"{'>=.5':>7s}")
    for lo, hi in SIZE_BINS:
        s = (diag >= lo) & (diag < hi)
        if not s.any():
            continue
        print(f"[{lo:g}, {hi:g})".ljust(14) +
              f" {int(s.sum()):5d} {raw[s].mean():7.3f} {trim[s].mean():8.3f} "
              f"{np.mean(trim[s] >= .5):7.1%}")


def report(rows, tally):
    n = len(rows)
    if not n:
        print("no matched detections")
        return
    p = np.array([r["p"] for r in rows])
    r_ = np.array([r["r"] for r in rows])
    pm = np.array([r["p_mask"] for r in rows])
    rm = np.array([r["r_mask"] for r in rows])
    diag = np.array([r["diag"] for r in rows])
    iou_o = np.array([r["iou_ours"] for r in rows])
    iou_c = np.array([r["iou_oracle"] for r in rows])
    n_ours = np.array([r["n_ours"] for r in rows], dtype=float)
    n_orc = np.array([r["n_oracle"] for r in rows], dtype=float)

    print(f"\n=== detections ===")
    for k in ("detections", "dropped_min_inliers", "lifted", "no_gt_overlap",
              "matched"):
        print(f"  {k:22s} {tally[k]:6d}")
    nogt = [(k.split('::', 1)[1], v) for k, v in tally.items()
            if k.startswith("nogt::")]
    if nogt:
        top = ", ".join(f"{lab} {c}" for lab, c in
                        sorted(nogt, key=lambda x: -x[1])[:8])
        print(f"  no-GT labels: {top}")

    print(f"\n=== selection quality over {n} matched observations ===")
    print(f"{'stage':16s} {'P mean':>7s} {'P med':>7s} {'R mean':>7s} {'R med':>7s}")
    print(f"{'after mask':16s} {pm.mean():7.3f} {np.median(pm):7.3f} "
          f"{rm.mean():7.3f} {np.median(rm):7.3f}")
    print(f"{'after cluster':16s} {p.mean():7.3f} {np.median(p):7.3f} "
          f"{r_.mean():7.3f} {np.median(r_):7.3f}")
    print(f"\npoints: ours median {np.median(n_ours):.0f}, "
          f"oracle median {np.median(n_orc):.0f}, "
          f"ratio median {np.median(n_ours / np.maximum(n_orc, 1)):.2f}x")

    q = quadrants(rows)
    print(f"\n=== regime (threshold 0.5 on each) ===")
    print(f"{'':18s} {'R < 0.5':>10s} {'R >= 0.5':>10s}")
    for phigh, name in ((False, "P < 0.5"), (True, "P >= 0.5")):
        cells = " ".join(f"{q[(phigh, rh)] / n:9.1%}" for rh in (False, True))
        print(f"{name:18s} {cells}")
    print("  low P / high R -> mask or cluster keeps background")
    print("  high P / low R -> cluster cuts into the object")
    print("  low P / low  R -> the mask is on the wrong object")

    print(f"\n=== by ground-truth size ===")
    print(f"{'GT diagonal':14s} {'n':>5s} {'P':>6s} {'R':>6s} "
          f"{'IoU ours':>9s} {'IoU orc':>8s} {'ours/orc':>9s}")
    for lo, hi in SIZE_BINS:
        s = (diag >= lo) & (diag < hi)
        if not s.any():
            continue
        print(f"[{lo:g}, {hi:g})".ljust(14) +
              f" {int(s.sum()):5d} {p[s].mean():6.3f} {r_[s].mean():6.3f} "
              f"{iou_o[s].mean():9.3f} {iou_c[s].mean():8.3f} "
              f"{np.median(n_ours[s] / np.maximum(n_orc[s], 1)):9.2f}")

    far = np.array([r["far"] for r in rows])
    farf = np.array([r["far_frac"] for r in rows])
    print(f"\n=== the points we took that we should not have ===")
    print(f"median distance outside the GT box: {np.median(far):.3f} m")
    print(f"share of those points further than 0.5 m: mean {farf.mean():.1%}, "
          f"median {np.median(farf):.1%}")

    span = np.array([r["span"] for r in rows])
    smin = np.array([r["span_min"] for r in rows])
    smax = np.array([r["span_max"] for r in rows])
    print(f"\n=== how much of the GT box our points span ===")
    print(f"mean over the three axes : mean {span.mean():.2f}, "
          f"median {np.median(span):.2f}")
    print(f"worst axis / best axis   : {np.median(smin):.2f} / "
          f"{np.median(smax):.2f}")
    print(f"share spanning <60% on every axis: {np.mean(smax < 0.6):.1%}")

    rad = np.array([r["radial"] for r in rows])
    lat = np.array([r["lateral"] for r in rows])
    rel = np.array([r["radial_rel"] for r in rows])
    print(f"\n=== which way the centre is wrong ===")
    print(f"along the viewing ray : median {np.median(rad):+.3f} m "
          f"(positive = truth is further away)")
    print(f"across it             : median {np.median(lat):.3f} m")
    print(f"share biased toward the camera: {np.mean(rad > 0):.1%}")
    print(f"ray error / GT depth along the ray: median {np.median(rel):+.2f}")

    print(f"\n=== what costs IoU ===")
    print(f"single-observation IoU: ours {iou_o.mean():.3f}, "
          f"oracle {iou_c.mean():.3f}")
    for name, v in (("precision", p), ("recall", r_)):
        print(f"  corr({name}, IoU) = {np.corrcoef(v, iou_o)[0, 1]:+.3f}")

    # Per-object view: an object seen 40 times must not outvote one seen twice.
    per = collections.defaultdict(list)
    for row in rows:
        per[(row["scene"], row["gi"])].append(row)
    pp = np.array([np.median([x["p"] for x in v]) for v in per.values()])
    rr = np.array([np.median([x["r"] for x in v]) for v in per.values()])
    print(f"\nper ground-truth object ({len(per)} objects): "
          f"P {pp.mean():.3f}, R {rr.mean():.3f}")


# ---------------------------------------------------------------------------
# Where the object's returns go
# ---------------------------------------------------------------------------


def decompose_scene(scene, frames_root):
    """Follow the *oracle* points forward instead of ours backward.

    The audit says we keep a tenth of an object's returns at high precision, so
    the loss is a filter dropping good points, not a mask grabbing bad ones.
    Four filters stand between a return inside the ground-truth box and the
    cloud we lift, and they are cheap to separate: field of view, the z-buffer
    gate, the mask, and the min-inlier floor. Each row below is the share of
    that object's returns still alive after the named stage, so the biggest
    drop between two rows is the one worth attacking.
    """
    gt = [g for g in load_gt_from_zip(GT_ROOT / f"{scene}.zip", scene)
          if not is_structure(g["label"]) and _scoreable(g["label"])]
    gc = np.array([g["center_3d"] for g in gt], dtype=np.float64)
    gs = np.array([g["bbox_aabb"]["size"] for g in gt], dtype=np.float64)
    plo, phi = gc - gs / 2 - PAD, gc + gs / 2 + PAD
    diag = np.linalg.norm(gs, axis=1)

    frames = replay.load_frames(frames_root / f"{scene}_tour",
                                min_move_m=MIN_MOVE_M, min_rot_deg=MIN_ROT_DEG)
    detections = replay.load_detections(frames_root / f"{scene}_tour")
    R_sc, t_sc = sensor_to_camera_transform()

    out = []
    for frame in frames:
        dets = [d for d in detections.get(frame.tick_id, [])
                if float(d["score"]) >= MIN_SCORE]
        scan = frame.scan()
        if scan.size == 0:
            continue
        xyz = scan[:, :3].astype(np.float64, copy=False)
        keep, front, vi, ui, depth = project_frame(
            xyz, frame.position, frame.orientation, R_sc, t_sc)
        masks = [replay.decode_mask(d["mask_rle"]) for d in dets]

        ins = ((xyz[:, None, :] >= plo[None]) & (xyz[:, None, :] <= phi[None])
               ).all(axis=2)
        for gi in np.flatnonzero(ins.sum(axis=0) >= MIN_ORACLE_PTS):
            idx = np.flatnonzero(ins[:, gi])
            n = float(idx.size)
            i_fov = idx[keep[idx]]
            i_front = idx[front[idx]]
            best = any_ = 0
            for m in masks:
                hit = int(m[vi[i_front], ui[i_front]].sum()) if i_front.size else 0
                best = max(best, hit)
            if i_front.size and masks:
                union = np.zeros(i_front.size, dtype=bool)
                for m in masks:
                    union |= m[vi[i_front], ui[i_front]]
                any_ = int(union.sum())
            out.append(dict(
                diag=float(diag[gi]), n=n,
                fov=i_fov.size / n, front=i_front.size / n,
                best=best / n, any=any_ / n,
                lifted=float(best >= MIN_INLIERS),
            ))
    return out


def report_decompose(rows):
    if not rows:
        print("nothing visible")
        return
    print(f"\n=== where an object's returns go ({len(rows)} object-frames) ===")
    print(f"{'stage':28s} {'mean':>7s} {'median':>7s}")
    for key, name in (("fov", "in the camera's FOV"),
                      ("front", "survives the z-buffer"),
                      ("any", "inside *some* mask"),
                      ("best", "inside its own detection")):
        v = np.array([r[key] for r in rows])
        print(f"{name:28s} {v.mean():7.1%} {np.median(v):7.1%}")
    lifted = np.array([r["lifted"] for r in rows])
    print(f"{'clears the 10-point floor':28s} {lifted.mean():7.1%}")

    diag = np.array([r["diag"] for r in rows])
    print(f"\n{'GT diagonal':14s} {'n':>6s} {'fov':>7s} {'front':>7s} "
          f"{'mask':>7s} {'lifted':>7s}")
    for lo, hi in SIZE_BINS:
        s = (diag >= lo) & (diag < hi)
        if not s.any():
            continue
        cell = " ".join(
            f"{np.array([r[k] for r in rows])[s].mean():6.1%} "
            for k in ("fov", "front", "best", "lifted"))
        print(f"[{lo:g}, {hi:g})".ljust(14) + f" {int(s.sum()):6d} {cell}")


# ---------------------------------------------------------------------------
# Is the mask small, or is it in the wrong place?
# ---------------------------------------------------------------------------


def geom_scene(scene, frames_root):
    """Compare the mask against the object's own projected footprint.

    The decomposition leaves two candidates for a mask that holds a third of
    the object's returns: it covers a third of the object, or it covers the
    object but sits somewhere else. They are told apart in pixels. For every
    object-frame, project the returns inside the ground-truth box and take the
    (u-unwrapped) pixel bounding box of that footprint, then measure:

      cover = mask pixels / footprint pixels   how much of the object it claims
      hit   = returns inside the mask          how much of it it actually got
      off   = centroid separation, in degrees  whether the two agree on where

    ``cover`` ≈ ``hit`` means an honestly small mask. ``cover`` ≈ 1 with a low
    ``hit`` means the mask is the right size in the wrong place, which is a
    calibration or timing bug and much cheaper to fix than a segmenter.
    """
    gt = [g for g in load_gt_from_zip(GT_ROOT / f"{scene}.zip", scene)
          if not is_structure(g["label"]) and _scoreable(g["label"])]
    gc = np.array([g["center_3d"] for g in gt], dtype=np.float64)
    gs = np.array([g["bbox_aabb"]["size"] for g in gt], dtype=np.float64)
    plo, phi = gc - gs / 2 - PAD, gc + gs / 2 + PAD
    diag = np.linalg.norm(gs, axis=1)

    frames = replay.load_frames(frames_root / f"{scene}_tour",
                                min_move_m=MIN_MOVE_M, min_rot_deg=MIN_ROT_DEG)
    detections = replay.load_detections(frames_root / f"{scene}_tour")
    R_sc, t_sc = sensor_to_camera_transform()
    deg_per_px = 360.0 / EQUIRECT_W

    out = []
    for frame in frames:
        dets = [d for d in detections.get(frame.tick_id, [])
                if float(d["score"]) >= MIN_SCORE]
        if not dets:
            continue
        scan = frame.scan()
        if scan.size == 0:
            continue
        xyz = scan[:, :3].astype(np.float64, copy=False)
        _keep, front, vi, ui, _depth = project_frame(
            xyz, frame.position, frame.orientation, R_sc, t_sc)
        masks = [replay.decode_mask(d["mask_rle"]) for d in dets]

        ins = ((xyz[:, None, :] >= plo[None]) & (xyz[:, None, :] <= phi[None])
               ).all(axis=2)
        for gi in np.flatnonzero(ins.sum(axis=0) >= MIN_ORACLE_PTS):
            idx = np.flatnonzero(ins[:, gi] & front)
            if idx.size < MIN_ORACLE_PTS:
                continue
            pu, pv = ui[idx].astype(np.float64), vi[idx].astype(np.float64)
            # Unwrap u about the footprint's circular mean before taking an
            # extent: an object straddling the seam is otherwise 360 wide.
            th = pu * (2 * np.pi / EQUIRECT_W)
            mu = np.arctan2(np.sin(th).mean(), np.cos(th).mean())
            pu = ((pu - mu * EQUIRECT_W / (2 * np.pi) + EQUIRECT_W / 2)
                  % EQUIRECT_W)
            fw = max(pu.max() - pu.min(), 1.0)
            fh = max(pv.max() - pv.min(), 1.0)
            foot = fw * fh
            cu, cv = pu.mean(), pv.mean()

            best_i, best_hit = -1, -1
            for i, m in enumerate(masks):
                hit = int(m[vi[idx], ui[idx]].sum())
                if hit > best_hit:
                    best_i, best_hit = i, hit
            if best_hit <= 0:
                out.append(dict(diag=float(diag[gi]), cover=0.0, hit=0.0,
                                off=float("nan"), n_det=0, area=0.0, foot=foot))
                continue

            m = masks[best_i]
            mv, mu_px = np.nonzero(m)
            mth = mu_px * (2 * np.pi / EQUIRECT_W)
            mmu = np.arctan2(np.sin(mth).mean(), np.cos(mth).mean())
            mu_un = ((mu_px - mmu * EQUIRECT_W / (2 * np.pi) + EQUIRECT_W / 2)
                     % EQUIRECT_W)
            # Both centroids are expressed in each set's own unwrapped frame,
            # so compare the frames' anchors, not the raw pixel columns.
            dmu = (mmu - mu) * EQUIRECT_W / (2 * np.pi)
            du = ((mu_un.mean() - cu + dmu + EQUIRECT_W / 2) % EQUIRECT_W
                  - EQUIRECT_W / 2)
            dv = mv.mean() - cv

            out.append(dict(
                diag=float(diag[gi]),
                cover=float(m.sum()) / foot,
                hit=best_hit / idx.size,
                off=float(np.hypot(du, dv) * deg_per_px),
                n_det=sum(1 for mm in masks
                          if int(mm[vi[idx], ui[idx]].sum()) >= MIN_INLIERS),
                area=float(m.sum()), foot=float(foot),
            ))
    return out


def report_geom(rows):
    rows = [r for r in rows if r["hit"] > 0]
    if not rows:
        print("no masks on any object")
        return
    cover = np.array([r["cover"] for r in rows])
    hit = np.array([r["hit"] for r in rows])
    off = np.array([r["off"] for r in rows])
    ndet = np.array([r["n_det"] for r in rows])
    diag = np.array([r["diag"] for r in rows])

    print(f"\n=== mask vs the object's projected footprint "
          f"({len(rows)} object-frames) ===")
    print(f"mask pixels / footprint pixels : mean {cover.mean():.2f}, "
          f"median {np.median(cover):.2f}")
    print(f"returns inside the mask        : mean {hit.mean():.1%}, "
          f"median {np.median(hit):.1%}")
    print(f"centroid separation            : median {np.median(off):.2f} deg, "
          f"90th pct {np.percentile(off, 90):.2f} deg")
    print(f"detections covering the object : mean {ndet.mean():.2f}, "
          f"share with >1: {np.mean(ndet > 1):.1%}")

    print(f"\n{'GT diagonal':14s} {'n':>6s} {'cover':>7s} {'hit':>7s} "
          f"{'off deg':>8s} {'n_det':>6s}")
    for lo, hi in SIZE_BINS:
        s = (diag >= lo) & (diag < hi)
        if not s.any():
            continue
        print(f"[{lo:g}, {hi:g})".ljust(14) +
              f" {int(s.sum()):6d} {cover[s].mean():7.2f} {hit[s].mean():7.1%} "
              f"{np.median(off[s]):8.2f} {ndet[s].mean():6.2f}")


# ---------------------------------------------------------------------------
# Are we cutting one object into several detections?
# ---------------------------------------------------------------------------


def split_scene(scene, frames_root):
    """Within a single frame, does merging the detections that agree help?

    The centre error is lateral, not radial: we sit on the object but off to
    one side of it, which is what taking one part of it looks like. If several
    of a frame's detections each land cleanly on the same object, then the
    object is being cut up before fusion ever sees it, and the union of those
    parts is a better observation than the best of them. If instead the best
    part already is the union, the object arrives whole and the missing recall
    is the ground-truth box holding things that are not the object.
    """
    gt = [g for g in load_gt_from_zip(GT_ROOT / f"{scene}.zip", scene)
          if not is_structure(g["label"]) and _scoreable(g["label"])]
    gc = np.array([g["center_3d"] for g in gt], dtype=np.float64)
    gs = np.array([g["bbox_aabb"]["size"] for g in gt], dtype=np.float64)
    glo, ghi = gc - gs / 2, gc + gs / 2
    plo, phi = glo - PAD, ghi + PAD
    diag = np.linalg.norm(gs, axis=1)

    frames = replay.load_frames(frames_root / f"{scene}_tour",
                                min_move_m=MIN_MOVE_M, min_rot_deg=MIN_ROT_DEG)
    detections = replay.load_detections(frames_root / f"{scene}_tour")
    R_sc, t_sc = sensor_to_camera_transform()

    out = []
    for frame in frames:
        dets = [d for d in detections.get(frame.tick_id, [])
                if float(d["score"]) >= MIN_SCORE]
        if not dets:
            continue
        scan = frame.scan()
        if scan.size == 0:
            continue
        xyz = scan[:, :3].astype(np.float64, copy=False)
        _keep, front, vi, ui, depth = project_frame(
            xyz, frame.position, frame.orientation, R_sc, t_sc)
        sets = []
        for det in dets:
            _, fin = select(replay.decode_mask(det["mask_rle"]), front, vi, ui, depth)
            if fin.size >= MIN_INLIERS:
                sets.append((det["label"], fin))

        ins = ((xyz[:, None, :] >= plo[None]) & (xyz[:, None, :] <= phi[None])
               ).all(axis=2)
        for gi in np.flatnonzero(ins.sum(axis=0) >= MIN_ORACLE_PTS):
            n_or = int(ins[:, gi].sum())
            parts = [(lab, fin) for lab, fin in sets
                     if int(ins[fin, gi].sum()) >= max(MIN_INLIERS, 0.5 * fin.size)]
            if not parts:
                continue
            best = max(parts, key=lambda t: int(ins[t[1], gi].sum()))
            bp = xyz[best[1]]
            u = np.unique(np.concatenate([f for _, f in parts]))
            up = xyz[u]
            out.append(dict(
                diag=float(diag[gi]), n_parts=len(parts),
                labels=len({lab for lab, _ in parts}),
                r_best=int(ins[best[1], gi].sum()) / n_or,
                r_union=int(ins[u, gi].sum()) / n_or,
                p_union=int(ins[u, gi].sum()) / u.size,
                iou_best=iou_3d(bp.min(axis=0), bp.max(axis=0), glo[gi], ghi[gi]),
                iou_union=iou_3d(up.min(axis=0), up.max(axis=0), glo[gi], ghi[gi]),
            ))
    return out


def report_split(rows):
    if not rows:
        print("nothing to split")
        return
    npart = np.array([r["n_parts"] for r in rows])
    rb = np.array([r["r_best"] for r in rows])
    ru = np.array([r["r_union"] for r in rows])
    ib = np.array([r["iou_best"] for r in rows])
    iu = np.array([r["iou_union"] for r in rows])
    pu = np.array([r["p_union"] for r in rows])
    diag = np.array([r["diag"] for r in rows])
    print(f"\n=== detections landing on the same object, same frame "
          f"({len(rows)} object-frames) ===")
    print(f"parts per object-frame: mean {npart.mean():.2f}, "
          f"share with >1: {np.mean(npart > 1):.1%}, "
          f"distinct labels among them: {np.mean([r['labels'] for r in rows]):.2f}")
    print(f"{'':16s} {'recall':>8s} {'IoU':>8s}")
    print(f"{'best part':16s} {rb.mean():8.3f} {ib.mean():8.3f}")
    print(f"{'union of parts':16s} {ru.mean():8.3f} {iu.mean():8.3f}")
    print(f"union precision: {pu.mean():.3f}")
    m = npart > 1
    if m.any():
        print(f"\nrestricted to the {int(m.sum())} that actually have >1 part:")
        print(f"{'best part':16s} {rb[m].mean():8.3f} {ib[m].mean():8.3f}")
        print(f"{'union of parts':16s} {ru[m].mean():8.3f} {iu[m].mean():8.3f}")
    print(f"\n{'GT diagonal':14s} {'n':>6s} {'parts':>6s} {'R best':>7s} "
          f"{'R union':>8s} {'IoU best':>9s} {'IoU union':>10s}")
    for lo, hi in SIZE_BINS:
        s = (diag >= lo) & (diag < hi)
        if not s.any():
            continue
        print(f"[{lo:g}, {hi:g})".ljust(14) +
              f" {int(s.sum()):6d} {npart[s].mean():6.2f} {rb[s].mean():7.3f} "
              f"{ru[s].mean():8.3f} {ib[s].mean():9.3f} {iu[s].mean():10.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=None)
    ap.add_argument("--frames-root", type=Path, default=Path("frames"))
    ap.add_argument("--self-check", type=int, default=0,
                    help="verify N detections against the real PointLifter")
    ap.add_argument("--decompose", action="store_true",
                    help="follow the oracle points forward through each filter")
    ap.add_argument("--geom", action="store_true",
                    help="compare each mask against the object's footprint")
    ap.add_argument("--split", action="store_true",
                    help="test whether one object arrives as several detections")
    args = ap.parse_args()

    if args.split:
        rows = []
        for scene in (args.scenes or SCENES):
            r = split_scene(scene, args.frames_root)
            rows.extend(r)
            print(f"{scene:14s} {len(r):6d} object-frames", flush=True)
        report_split(rows)
        return

    if args.geom:
        rows = []
        for scene in (args.scenes or SCENES):
            r = geom_scene(scene, args.frames_root)
            rows.extend(r)
            print(f"{scene:14s} {len(r):6d} object-frames", flush=True)
        report_geom(rows)
        return

    if args.decompose:
        rows = []
        for scene in (args.scenes or SCENES):
            r = decompose_scene(scene, args.frames_root)
            rows.extend(r)
            print(f"{scene:14s} {len(r):6d} object-frames", flush=True)
        report_decompose(rows)
        return

    rows, tally, ceil, n_gt = [], collections.Counter(), [], 0
    for scene in (args.scenes or SCENES):
        r, t, c = audit_scene(scene, args.frames_root, self_check=args.self_check)
        rows.extend(r)
        tally.update(t)
        ceil.extend(c)
        n_gt += sum(1 for g in load_gt_from_zip(GT_ROOT / f"{scene}.zip", scene)
                    if not is_structure(g["label"]) and _scoreable(g["label"]))
        print(f"{scene:14s} {len(r):5d} matched observations", flush=True)
    report(rows, tally)
    report_ceiling(ceil, rows, n_gt)


if __name__ == "__main__":
    main()
