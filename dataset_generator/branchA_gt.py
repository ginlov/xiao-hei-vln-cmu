"""Branch A detection-GT extractor (TASK 10).

The simulator publishes a per-instance semantic image (`/camera/semantic_image`,
1920x640 bgr8) whose colours are a deterministic per-object palette defined in
the sim's `AssetList.csv` (colour -> object name) + `Categories.csv`
(name -> class). This module turns one captured frame into detection GT with
ZERO alignment / projection of external CSV boxes:

  semantic pixel -> snap to 42-grid -> (r,g,b)
    -> AssetList name -> Categories class
  each distinct non-background colour == one VISIBLE instance
    -> connected-component denoise (drop snap-scatter / anti-alias residue)
    -> 2D box (surviving pixels, with 360deg seam wrap)
    -> count (number of surviving instance colours)
  3D centre (map frame) <- back-project the instance's pixels onto the
    frame's own lidar (`/registered_scan`), using the camera convention
    recovered + validated in Step 0:
        az = -(atan2(dy, dx) - yaw),  el = atan2(dz, hypot(dx, dy))
        u  = ((az/2pi + 0.5) mod 1) * W,   v = (0.5 - el/fov_v) * H

Why connected components matter: snapping anti-aliased edge pixels to the
nearest palette level sprinkles a handful of stray pixels of an instance's
colour across the whole panorama. The global bbox of *all* matching pixels is
then garbage (e.g. 70 px spanning 1500 px wide). Keeping only components above a
size floor removes that scatter and collapses each colour to its real blob(s).

Usage (offline, on a captured frameset dir with semantic.npy + scan.npy +
meta.json + AssetList.csv + Categories.csv):

    uv run --extra viz python dataset_generator/branchA_gt.py \
        --sample dataset_generator/branchA_sample --overlay --crops 4
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

LEVELS = np.array([0, 42, 84, 126, 168, 210, 252])
# classes that are scene structure, not countable "objects"
BG_CLASSES = {"floor", "wall", "ceiling", "window", "door", "door frame",
              "carpet", "curtain", "column", ""}
# outdoor scenery seen through windows/glass -- not part of the indoor GT
EXTERIOR_CLASSES = {"building", "exterior structure", "sky", "tree", "road",
                    "grass", "terrain", "mountain"}

# 3D-centre robustness: a single frame's lidar labelling of an instance picks up
# stray/mislabelled far points (and used to include (0,0,0) filler), so a naive
# mean scatters by metres. Require enough inlier points and reject points far
# from the instance's median before averaging.
MIN_LIDAR_3D = 5        # need >= this many inlier lidar pts for a 3D centre
LIDAR_GATE_M = 1.5      # drop labelled pts >this far from the instance median
ZBUF_TOL_M = 0.2        # a lidar pt is "front" if within this of its pixel's nearest


def robust_center(pts: np.ndarray):
    """Median-gate outlier rejection then mean. Returns (centre|None, n_inliers).
    None when too few trustworthy points -> the instance is 2D-only."""
    if len(pts) < MIN_LIDAR_3D:
        return None, len(pts)
    med = np.median(pts, axis=0)
    inl = pts[np.linalg.norm(pts - med, axis=1) <= LIDAR_GATE_M]
    if len(inl) < MIN_LIDAR_3D:
        return None, len(inl)
    return inl.mean(0).tolist(), len(inl)

# denoise defaults (px). Tuned on the living-room sample.
MIN_CC = 20        # drop connected components smaller than this
MIN_PIXELS = 40    # drop an instance whose surviving pixels total below this
MIN_CROP_DIM = 6   # drop crop boxes thinner than this in either dim (anti-sliver)


def snap_palette(bgr: np.ndarray) -> np.ndarray:
    """Snap a bgr8 semantic image to the discrete 42-step palette, return rgb."""
    snapped = LEVELS[np.abs(bgr.astype(int)[..., None] - LEVELS).argmin(-1)]
    return snapped[:, :, ::-1]  # bgr -> rgb


def load_legend(sample: Path):
    """colour(r,g,b) -> (instance_name, cleaned_class, nyu40class)."""
    name2cls = {}
    with open(sample / "Categories.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name2cls[row["name"]] = (row.get("cleaned", ""),
                                     row.get("nyu40class", ""))
    col2obj = {}
    with open(sample / "AssetList.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                col = (int(round(float(row["r"]))),
                       int(round(float(row["g"]))),
                       int(round(float(row["b"]))))
            except ValueError:
                continue
            cleaned, nyu40 = name2cls.get(row["name"], ("", ""))
            col2obj[col] = (row["name"], cleaned, nyu40)
    return col2obj


def yaw_from_quat(o) -> float:
    return math.atan2(2 * (o["w"] * o["z"] + o["x"] * o["y"]),
                      1 - 2 * (o["y"] ** 2 + o["z"] ** 2))


def equirect_uv(d: np.ndarray, yaw: float, W: int, H: int, fov_v_deg: float):
    """Map map-frame offsets (N,3) from the camera to equirect (u,v). Recovered
    convention: az = -(atan2(dy,dx) - yaw)."""
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    az = -(np.arctan2(dy, dx) - yaw)
    el = np.arctan2(dz, np.hypot(dx, dy))
    u = (((az / (2 * math.pi) + 0.5) % 1.0) * W).astype(np.int32) % W
    v = ((0.5 - el / math.radians(fov_v_deg)) * H).astype(np.int32)
    return u, v


def clean_mask(mask: np.ndarray, min_cc: int):
    """Remove connected components smaller than min_cc (8-connectivity). The
    panorama seam is NOT bridged here; a seam-straddling instance survives as
    two components and bbox_with_seam reunites them. Returns (cleaned, n_comp)."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), int))
    if n == 0:
        return mask & False, 0
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = sizes >= min_cc
    cleaned = keep[lab]
    return cleaned, int(keep.sum())


def clean_semantic(S: np.ndarray, gt: dict, min_cc: int) -> np.ndarray:
    """Rebuild a denoised equirect semantic that keeps ONLY pixels belonging to
    a valid extracted instance, with each instance re-cleaned (sub-min_cc specks
    dropped). Colours not in gt['instances'] (palette-collision noise) vanish.

    This must run before crop reprojection: nearest-neighbour resampling
    magnifies single stray pixels into multi-pixel slivers that would otherwise
    pass the crop's own min_pixels filter and produce degenerate/huge boxes."""
    out = np.zeros_like(S)
    for inst in gt["instances"]:
        col = np.array(inst["color_rgb"], dtype=S.dtype)
        mask = np.all(S == col, axis=2)
        mask, _ = clean_mask(mask, min_cc)
        out[mask] = col
    return out


def bbox_with_seam(us: np.ndarray, vs: np.ndarray, W: int):
    """2D bbox of a colour's pixels; if it straddles the +/-180 seam return a
    wrapped box (x may be >= W, meaning it wraps to x-W)."""
    vmin, vmax = int(vs.min()), int(vs.max())
    umin, umax = int(us.min()), int(us.max())
    if umax - umin > W / 2:  # likely wraps the seam
        shifted = np.where(us < W / 2, us + W, us)
        umin, umax = int(shifted.min()), int(shifted.max())
    return [umin, vmin, umax, vmax]  # x1,y1,x2,y2 (x2 may exceed W if wrapped)


def extract(sample: Path, fov_v: float = 120.0, min_cc: int = MIN_CC,
            min_pixels: int = MIN_PIXELS, keep_exterior: bool = False):
    sem = np.load(sample / "semantic.npy")          # bgr8
    S = snap_palette(sem)                            # rgb
    H, W, _ = S.shape
    legend = load_legend(sample)
    meta = json.load(open(sample / "meta.json"))
    pos = meta["pose"]["position"]
    cam = np.array([pos["x"], pos["y"], pos["z"]])
    yaw = yaw_from_quat(meta["pose"]["orientation"])

    # --- lidar -> equirect, label each lidar point by the pixel's instance ---
    scan = np.load(sample / "scan.npy")[:, :3]       # map frame xyz
    scan = scan[~(np.all(scan == 0, axis=1) | np.isnan(scan).any(axis=1))]
    u_l, v_l = equirect_uv(scan - cam, yaw, W, H, fov_v)
    inb = (v_l >= 0) & (v_l < H)
    # occlusion-aware labelling: the semantic pixel shows the NEAREST surface, so
    # only label lidar points at (near) the nearest range for their pixel.
    # Otherwise background points along the same bearing as a near object get
    # that object's colour and drag its 3D centre metres away (the real cause of
    # the per-view centroid scatter).
    rng = np.linalg.norm(scan - cam, axis=1)
    fidx = np.clip(v_l, 0, H - 1) * W + u_l
    nearest = np.full(H * W, np.inf)
    np.minimum.at(nearest, fidx[inb], rng[inb])
    front = inb & (rng <= nearest[fidx] + ZBUF_TOL_M)
    lab_col = np.full((len(scan), 3), -1)
    lab_col[front] = S[v_l[front], u_l[front]]

    flat = S.reshape(-1, 3)
    colours, counts = np.unique(flat, axis=0, return_counts=True)
    instances = []
    dropped = {"background": 0, "exterior": 0, "noise": 0}
    for col, n_px in zip(colours, counts):
        key = tuple(int(c) for c in col)
        name, cleaned, nyu40 = legend.get(key, (None, "", ""))
        if name is None or cleaned in BG_CLASSES:
            dropped["background"] += 1
            continue
        if not keep_exterior and cleaned in EXTERIOR_CLASSES:
            dropped["exterior"] += 1
            continue
        mask = np.all(S == col, axis=2)
        mask, n_comp = clean_mask(mask, min_cc)
        vs, us = np.nonzero(mask)
        if len(us) < min_pixels:
            dropped["noise"] += 1
            continue
        box = bbox_with_seam(us, vs, W)
        # 3D centre from this instance's lidar points (robust, min-count gated)
        lmask = np.all(lab_col == col, axis=1)
        c3d, n_inl = robust_center(scan[lmask])
        dist = (float(np.linalg.norm(np.array(c3d) - cam))
                if c3d is not None else None)
        instances.append({
            "instance": name,
            "raw_label": cleaned,
            "nyu40_label": nyu40,
            "color_rgb": list(key),
            "bbox_2d": [int(x) for x in box],
            "n_pixels": int(len(us)),
            "n_components": n_comp,
            "n_lidar_pts": int(lmask.sum()),
            "n_lidar_inliers": int(n_inl),
            "loc_source": "lidar" if c3d is not None else "2d_only",
            "center_3d": c3d,
            "dist_m": round(dist, 3) if dist is not None else None,
        })

    counts_by_class: dict[str, int] = {}
    for inst in instances:
        counts_by_class[inst["raw_label"]] = counts_by_class.get(inst["raw_label"], 0) + 1

    return {
        "frame": sample.name,
        "image_size": [W, H],
        "pose": meta["pose"],
        "fov_v_deg": fov_v,
        "denoise": {"min_cc": min_cc, "min_pixels": min_pixels,
                    "keep_exterior": keep_exterior},
        "dropped_colours": dropped,
        "n_instances": len(instances),
        "counts_by_class": counts_by_class,
        "instances": instances,
    }, S


def draw_overlay(sample: Path, gt: dict, out: Path):
    from PIL import Image, ImageDraw
    rgb = np.load(sample / "rgb.npy")[:, :, ::-1]    # bgr8 -> rgb
    W = gt["image_size"][0]
    pil = Image.fromarray(rgb.copy())
    dr = ImageDraw.Draw(pil)
    for inst in gt["instances"]:
        x1, y1, x2, y2 = inst["bbox_2d"]
        col = tuple(inst["color_rgb"])
        for seg in ([(x1, x2)] if x2 < W else [(x1, W - 1), (0, x2 - W)]):
            dr.rectangle([seg[0], y1, seg[1], y2], outline=col, width=2)
        dr.text((x1 % W, max(0, y1 - 10)), inst["raw_label"], fill=(255, 255, 0))
    pil.save(out)


# ---------------------------------------------------------------------------
# (b) rectilinear perspective crops: de-warp the 360 panorama into N pinhole
#     views so boxes are detector-friendly (no equirect near-field stretch).
# ---------------------------------------------------------------------------

def _equirect_sample_maps(yaw_c, fov_h_deg, fov_v_deg, cw, ch, W, H, pano_fovv):
    """For a virtual pinhole cam at azimuth yaw_c (rad, panorama-relative),
    return integer (u,v) source maps into the equirect image for an (ch,cw)
    rectilinear output. Camera looks along +z, x right, y down."""
    f = (cw / 2) / math.tan(math.radians(fov_h_deg) / 2)
    xs = (np.arange(cw) - cw / 2)[None, :] / f
    ys = (np.arange(ch) - ch / 2)[:, None] / f
    xs, ys = np.broadcast_arrays(xs, ys)
    # ray (x,y,1): horizontal angle off-centre = atan2(x,1); elevation = -y comp
    az = yaw_c + np.arctan2(xs, 1.0)
    el = np.arctan2(-ys, np.hypot(xs, 1.0))
    u = (((az / (2 * math.pi) + 0.5) % 1.0) * W).astype(np.int32) % W
    v = ((0.5 - el / math.radians(pano_fovv)) * H).astype(np.int32)
    valid = (v >= 0) & (v < H)
    v = np.clip(v, 0, H - 1)
    return u, v, valid


def reproject_crops(sample: Path, gt: dict, S: np.ndarray, out_dir: Path,
                    n: int = 4, fov_h: float = 100.0, fov_v_crop: float = 90.0,
                    cw: int = 640, ch: int = 640, min_pixels: int = MIN_PIXELS,
                    min_cc: int = MIN_CC):
    """Render N overlapping rectilinear crops + per-crop 2D boxes. Boxes are
    recomputed in crop space from the resampled semantic, so they de-warp and
    clip naturally. Returns a list of crop dicts and writes crop_k.png +
    crop_k_overlay.png + crops_gt.json."""
    from PIL import Image, ImageDraw
    out_dir.mkdir(parents=True, exist_ok=True)
    W, H = gt["image_size"]
    pano_fovv = gt["fov_v_deg"]
    rgb = np.load(sample / "rgb.npy")[:, :, ::-1]    # rgb
    legend = load_legend(sample)
    # denoise once in equirect space so resampling can't magnify stray pixels
    S_clean = clean_semantic(S, gt, min_cc)
    crops = []
    for k in range(n):
        yaw_c = 2 * math.pi * k / n
        u, v, valid = _equirect_sample_maps(yaw_c, fov_h, fov_v_crop,
                                            cw, ch, W, H, pano_fovv)
        crop_rgb = rgb[v, u]
        crop_sem = S_clean[v, u]
        crop_rgb[~valid] = 0
        crop_sem[~valid] = 0
        boxes = []
        cols, cnts = np.unique(crop_sem.reshape(-1, 3), axis=0, return_counts=True)
        for col, _n in zip(cols, cnts):
            key = tuple(int(c) for c in col)
            if key == (0, 0, 0):
                continue
            name, cleaned, nyu40 = legend.get(key, (None, "", ""))
            if name is None or cleaned in BG_CLASSES or cleaned in EXTERIOR_CLASSES:
                continue
            mask = np.all(crop_sem == col, axis=2) & valid
            mask, n_comp = clean_mask(mask, min_cc)
            # keep only the largest component: a single crop can still catch two
            # disjoint fragments of one colour whose union bbox would be huge.
            lab, ncc = ndimage.label(mask, structure=np.ones((3, 3), int))
            if ncc > 1:
                sizes = np.bincount(lab.ravel()); sizes[0] = 0
                mask = lab == int(sizes.argmax())
            ys, xs = np.nonzero(mask)
            if len(xs) < min_pixels:
                continue
            bw, bh = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
            if bw < MIN_CROP_DIM or bh < MIN_CROP_DIM:
                continue
            boxes.append({
                "instance": name, "raw_label": cleaned, "nyu40_label": nyu40,
                "color_rgb": list(key),
                "bbox_2d": [int(xs.min()), int(ys.min()),
                            int(xs.max()), int(ys.max())],
                "n_pixels": int(len(xs)),
            })
        Image.fromarray(crop_rgb).save(out_dir / f"crop_{k}.png")
        pil = Image.fromarray(crop_rgb.copy())
        dr = ImageDraw.Draw(pil)
        for b in boxes:
            x1, y1, x2, y2 = b["bbox_2d"]
            dr.rectangle([x1, y1, x2, y2], outline=tuple(b["color_rgb"]), width=2)
            dr.text((x1, max(0, y1 - 10)), b["raw_label"], fill=(255, 255, 0))
        pil.save(out_dir / f"crop_{k}_overlay.png")
        crops.append({"index": k, "yaw_deg": round(math.degrees(yaw_c), 1),
                      "fov_h_deg": fov_h, "fov_v_deg": fov_v_crop,
                      "size": [cw, ch], "n_boxes": len(boxes), "boxes": boxes})
    json.dump({"frame": sample.name, "n_crops": n, "crops": crops},
              open(out_dir / "crops_gt.json", "w"), indent=2)
    return crops


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=Path,
                    default=Path("dataset_generator/branchA_sample"))
    ap.add_argument("--fov-v", type=float, default=120.0)
    ap.add_argument("--min-cc", type=int, default=MIN_CC)
    ap.add_argument("--min-pixels", type=int, default=MIN_PIXELS)
    ap.add_argument("--keep-exterior", action="store_true",
                    help="keep outdoor scenery classes (building/sky/...)")
    ap.add_argument("--overlay", action="store_true")
    ap.add_argument("--crops", type=int, default=0,
                    help="also render N rectilinear perspective crops (0=off)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    gt, S = extract(args.sample, args.fov_v, args.min_cc, args.min_pixels,
                    args.keep_exterior)
    out_json = args.out or (args.sample / "detection_gt.json")
    json.dump(gt, open(out_json, "w"), indent=2)
    print(f"frame={gt['frame']}  instances={gt['n_instances']}  "
          f"dropped={gt['dropped_colours']}")
    print("counts_by_class:")
    for cls, n in sorted(gt["counts_by_class"].items(), key=lambda kv: -kv[1]):
        print(f"  {n:3d}  {cls}")
    has3d = sum(1 for i in gt["instances"] if i["center_3d"])
    print(f"instances with 3D centre (lidar): {has3d}/{gt['n_instances']}")
    print(f"wrote {out_json}")
    if args.overlay:
        ov = args.sample / "overlay_gt.png"
        draw_overlay(args.sample, gt, ov)
        print(f"wrote {ov}")
    if args.crops:
        cdir = args.sample / "crops"
        crops = reproject_crops(args.sample, gt, S, cdir, n=args.crops,
                                min_pixels=args.min_pixels, min_cc=args.min_cc)
        print(f"wrote {args.crops} crops -> {cdir}  "
              f"(boxes: {[c['n_boxes'] for c in crops]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
