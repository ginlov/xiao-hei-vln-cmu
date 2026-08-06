#!/usr/bin/env python3
"""Can a CLIP embedding decide whether two observations are the same object?

TASK 25 measured that 107 surplus nodes per scene carry a label ground truth
does not have, and that the labels are synonyms -- `couch`/`sofa`,
`lamp`/`lantern`/`wall lamp`. `ObjectMap.add` requires an exact label match, so
those never merge. Replacing the label key with an appearance embedding is the
obvious fix and is what the open-vocabulary 3D mapping line (ConceptFusion,
OpenMask3D, ConceptGraphs) does.

There is an obvious reason it might not work. CLIP encodes *category*, not
*instance*: two different chairs should embed almost identically. Association
needs the opposite -- to tell "two views of one couch" from "two chairs at one
table", which is exactly the discrimination plain geometry already failed at
(TASK 24: relative radius alone runs away). Fixing synonyms while making
over-merging worse is not a fix.

So this measures the discrimination, not the synonym similarity. Ground-truth
boxes supply instance identity; every observation pair is then one of

    same object, same label        -- the ceiling
    same object, different label   -- what the change has to rescue
    different object, same class   -- what it must not break
    different object, different class

and the number that matters is the separation between the middle two, among
pairs close enough that geometry alone is ambiguous.

    uv run python scripts/clip_assoc_probe.py --scene chinese_room
    uv run python scripts/clip_assoc_probe.py --scene all --max-obs 3000
"""

from __future__ import annotations

import argparse
import collections
import itertools
import sys
from pathlib import Path

import numpy as np

from xiao_hei_vln.perception import replay
from xiao_hei_vln.perception.lifter import PointLifter
from xiao_hei_vln.perception.vocab import is_structure
from xiao_hei_vln.scene.io import read_objects_from_zip

DEFAULT_SCENES = ("arabic_room", "chinese_room", "japanese_room",
                  "livingroom_3", "loft", "office_1", "office_2")
LIFT = dict(min_move_m=0.15, min_rot_deg=10.0, min_score=0.35, min_inliers=10)
GT_ROOT = Path.home() / "workspace/dataset/unity-scene"

# Pairs closer than this are usually trivially the same object and pairs beyond
# it are trivially different; the interesting question lives in between, which
# is also where `MERGE_DIST` (0.4 m) currently has to make its call.
AMBIGUOUS_M = (0.15, 1.50)


# ---------------------------------------------------------------------------
# encoder
# ---------------------------------------------------------------------------

def load_encoder(device: str):
    """An image encoder, whichever of the usual packages is installed."""
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k")
        model = model.eval().to(device)

        def encode(pil_batch):
            import torch
            with torch.no_grad():
                x = torch.stack([preprocess(im) for im in pil_batch]).to(device)
                f = model.encode_image(x)
                return (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()

        return encode, "open_clip ViT-B-32 laion2b"
    except ImportError:
        pass

    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        name = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(name).eval().to(device)
        proc = CLIPProcessor.from_pretrained(name)

        def encode(pil_batch):
            with torch.no_grad():
                x = proc(images=list(pil_batch), return_tensors="pt").to(device)
                f = model.get_image_features(**x)
                return (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()

        return encode, name
    except ImportError as exc:
        raise SystemExit(
            "no CLIP image encoder available -- install open_clip_torch or "
            f"transformers into this venv ({exc})") from exc


# ---------------------------------------------------------------------------
# observations
# ---------------------------------------------------------------------------

def crop(equirect_bgr, mask, pad: int, mask_bg: bool):
    """The detection's pixels, as a PIL image.

    Cropped from the equirect rather than from the four perspective faces the
    sidecar renders. Objects sit near the horizon where the distortion is mild,
    and using the faces here would mean re-deriving their LUTs; if the probe
    says the idea is worth pursuing, the real implementation should embed the
    face crop instead.
    """
    from PIL import Image
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, mask.shape[0])
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, mask.shape[1])
    if (y1 - y0) < 8 or (x1 - x0) < 8:
        return None
    patch = equirect_bgr[y0:y1, x0:x1, ::-1].copy()
    if mask_bg:
        m = mask[y0:y1, x0:x1]
        patch[~m] = (patch[~m] * 0.25).astype(patch.dtype)
    return Image.fromarray(patch)


def gt_instance(centre, pts, gt, det_label: str, strict: bool):
    """Which annotated object this observation belongs to, or None.

    Containment of the observation's median point is the strict test; falling
    back to the nearest centre within half a metre keeps observations whose
    cloud sits just outside a tight box, which is common for thin objects.

    Position alone is not enough, because ground-truth boxes nest: a pillow on
    a chair sits inside the chair's box, so the pillow's observations get filed
    under the chair and the "same object" population quietly fills up with
    pairs that are not the same object at all -- which would understate the
    embedding. Under ``strict`` an observation is only accepted when the
    detector's own label agrees with the annotation's, which costs recall but
    makes the population mean what it says.
    """
    med = np.median(pts, axis=0)
    inside = [g for g in gt if np.all(med >= g["lo"]) and np.all(med <= g["hi"])]
    if strict:
        d = det_label.strip().lower()
        inside = [g for g in inside if g["label"] == d]
        if not inside:
            near = [g for g in gt if g["label"] == d
                    and float(np.linalg.norm(g["c"] - centre)) <= 0.5]
            if not near:
                return None
            return min(near, key=lambda g: float(np.linalg.norm(g["c"] - centre)))
    if len(inside) == 1:
        return inside[0]
    cands = inside or gt
    if not cands:
        return None
    best = min(cands, key=lambda g: float(np.linalg.norm(g["c"] - centre)))
    return best if float(np.linalg.norm(best["c"] - centre)) <= 0.5 else None


def collect(scene: str, frames_root: Path, encode, *, max_obs: int, pad: int,
            mask_bg: bool, batch: int, strict: bool) -> list[dict]:
    fdir = frames_root / f"{scene}_tour"
    if not fdir.is_dir():
        print(f"{scene}: no corpus at {fdir}", file=sys.stderr)
        return []
    zp = GT_ROOT / f"{scene}.zip"
    if not zp.is_file():
        print(f"{scene}: no ground truth", file=sys.stderr)
        return []

    gt = []
    for e in read_objects_from_zip(zp, scene_name=scene).values():
        if is_structure(e.label) or e.label.strip().lower() == "unknown":
            continue
        c = np.array([e.center.x, e.center.y, e.center.z])
        s = np.array([e.size.x, e.size.y, e.size.z])
        gt.append({"id": e.object_id, "label": e.label.strip().lower(),
                   "c": c, "lo": c - s / 2, "hi": c + s / 2})

    frames = replay.load_frames(fdir, use_image_pose=True,
                                min_move_m=LIFT["min_move_m"],
                                min_rot_deg=LIFT["min_rot_deg"])
    detections = replay.load_detections(fdir)
    lifter = PointLifter(min_inliers=LIFT["min_inliers"])

    rows, pend_img, pend_row = [], [], []

    def flush():
        if not pend_img:
            return
        for r, v in zip(pend_row, encode(pend_img), strict=True):
            r["emb"] = v
            rows.append(r)
        pend_img.clear()
        pend_row.clear()

    for i, frame in enumerate(frames):
        dets = detections.get(frame.tick_id) or []
        if not dets or len(rows) + len(pend_row) >= max_obs:
            if len(rows) + len(pend_row) >= max_obs:
                break
            continue
        scan, img = frame.scan(), frame.image_bgr()
        for det in dets:
            if float(det["score"]) < LIFT["min_score"]:
                continue
            mask = replay.decode_mask(det["mask_rle"])
            res = lifter.lift(mask=mask, scan_points_map=scan,
                              pose_position=frame.position,
                              pose_orientation=frame.orientation)
            if res.position is None:
                continue
            pts = np.asarray(res.inlier_points, dtype=float)[:, :3]
            centre = (pts.min(axis=0) + pts.max(axis=0)) / 2
            g = gt_instance(centre, pts, gt, det["label"], strict)
            if g is None:
                continue
            im = crop(img, mask, pad, mask_bg)
            if im is None:
                continue
            pend_img.append(im)
            pend_row.append({"scene": scene, "gt_id": g["id"],
                             "gt_label": g["label"],
                             "label": det["label"].strip().lower(),
                             "centre": centre})
            if len(pend_img) >= batch:
                flush()
        if (i + 1) % 40 == 0:
            print(f"  {scene}: frame {i + 1}/{len(frames)}, {len(rows)} kept",
                  flush=True)
    flush()
    print(f"{scene:14s} {len(rows):5d} matched observations over "
          f"{len({r['gt_id'] for r in rows})} ground-truth objects", flush=True)
    return rows


# ---------------------------------------------------------------------------
# the four populations
# ---------------------------------------------------------------------------

def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a random positive scores above a random negative), via rank sum."""
    if not len(pos) or not len(neg):
        return float("nan")
    both = np.concatenate([pos, neg])
    order = both.argsort()
    ranks = np.empty(len(both), dtype=float)
    ranks[order] = np.arange(1, len(both) + 1)
    # average ranks over ties so exact duplicates do not bias the estimate
    _, inv, cnt = np.unique(both, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def describe(name, v):
    if not len(v):
        print(f"  {name:34s}      -- no pairs")
        return
    print(f"  {name:34s} n={len(v):7d}  mean {v.mean():.3f}  "
          f"p10 {np.percentile(v, 10):.3f}  p50 {np.percentile(v, 50):.3f}  "
          f"p90 {np.percentile(v, 90):.3f}")


def analyse(rows: list[dict], sample: int, rng) -> None:
    emb = np.stack([r["emb"] for r in rows])
    ctr = np.stack([r["centre"] for r in rows])
    gid = np.array([f"{r['scene']}/{r['gt_id']}" for r in rows])
    lab = np.array([r["label"] for r in rows])
    glab = np.array([r["gt_label"] for r in rows])
    scene = np.array([r["scene"] for r in rows])

    n = len(rows)
    idx = rng.integers(0, n, size=(min(sample, n * (n - 1) // 2), 2))
    idx = idx[(idx[:, 0] != idx[:, 1]) & (scene[idx[:, 0]] == scene[idx[:, 1]])]
    a, b = idx[:, 0], idx[:, 1]

    sim = np.einsum("ij,ij->i", emb[a], emb[b])
    dist = np.linalg.norm(ctr[a] - ctr[b], axis=1)
    same_obj = gid[a] == gid[b]
    same_lab = lab[a] == lab[b]
    same_cls = glab[a] == glab[b]

    print(f"\n{len(idx)} sampled same-scene pairs\n")
    print("cosine similarity by pair type:")
    describe("same object, same label", sim[same_obj & same_lab])
    describe("same object, DIFFERENT label", sim[same_obj & ~same_lab])
    describe("different object, same class", sim[~same_obj & same_cls])
    describe("different object, different class", sim[~same_obj & ~same_cls])

    lo, hi = AMBIGUOUS_M
    amb = (dist >= lo) & (dist <= hi)
    print(f"\namong the {int(amb.sum())} pairs {lo}-{hi} m apart, where geometry "
          f"alone is ambiguous:")
    pos = sim[amb & same_obj]
    neg_same_cls = sim[amb & ~same_obj & same_cls]
    neg_any = sim[amb & ~same_obj]
    describe("same object", pos)
    describe("different object, same class", neg_same_cls)
    describe("different object, any class", neg_any)

    print(f"\n  AUC same-vs-different (any class)   {auc(pos, neg_any):.3f}")
    print(f"  AUC same-vs-different (same class)  {auc(pos, neg_same_cls):.3f}"
          "   <-- the number that decides it")
    print(f"  AUC from centre distance alone      "
          f"{auc(-dist[amb & same_obj], -dist[amb & ~same_obj & same_cls]):.3f}")

    print("\n  A same-class AUC near 0.5 means the embedding cannot tell one "
          "chair\n  from another and would license exactly the over-merging "
          "that\n  loosening MERGE_DIST already causes.")

    # Which synonym pairs the change would actually rescue.
    pairs = collections.Counter()
    for i, j in zip(a[same_obj & ~same_lab], b[same_obj & ~same_lab], strict=True):
        pairs[tuple(sorted((lab[i], lab[j])))] += 1
    if pairs:
        print("\ncommonest label disagreements on one physical object:")
        for (p, q), c in pairs.most_common(15):
            m = sim[same_obj & ~same_lab]
            print(f"  {p:22s} / {q:22s} {c:5d}")
        print(f"  (their mean similarity {m.mean():.3f})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="chinese_room")
    ap.add_argument("--frames-root", default="frames")
    ap.add_argument("--max-obs", type=int, default=4000,
                    help="observations encoded per scene")
    ap.add_argument("--pad", type=int, default=6)
    ap.add_argument("--no-mask-bg", action="store_true",
                    help="embed the raw bbox crop instead of dimming background")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--pairs", type=int, default=4_000_000)
    ap.add_argument("--strict", action="store_true",
                    help="only keep observations whose detected label agrees "
                         "with the annotation; removes the nested-box "
                         "contamination (a pillow filed under its chair)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    encode, name = load_encoder(args.device)
    print(f"encoder: {name}\n")
    scenes = DEFAULT_SCENES if args.scene == "all" else (args.scene,)
    rows: list[dict] = []
    for s in scenes:
        rows += collect(s, Path(args.frames_root), encode, max_obs=args.max_obs,
                        pad=args.pad, mask_bg=not args.no_mask_bg,
                        batch=args.batch, strict=args.strict)
    if len(rows) < 50:
        print("not enough matched observations", file=sys.stderr)
        return 1
    analyse(rows, args.pairs, np.random.default_rng(0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
