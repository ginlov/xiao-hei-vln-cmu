#!/usr/bin/env python3
"""2D open-vocab detection front-end for the perception worker (Goal B).

Produces detections in the SAME contract lift3d consumes -- {label, score,
mask(H,W bool)} in equirectangular pixel coords -- but from real models instead
of the sim's semantic GT:

    YOLO-World v2  (boxes + labels) ── on perspective crops ──┐
                                                              ├─ equirect mask
    SAM 2.1        (box-prompted masks, precise contours) ────┘

Why crops: YOLO/SAM are trained on rectilinear images and break on the 360deg
equirect warp, so we de-warp the panorama into N overlapping pinhole crops
(reusing branchA_gt._equirect_sample_maps), detect there, then scatter each
crop-space mask back to the panorama via the same crop->equirect sample map.

The output plugs straight into lift3d.lift_frame(); the geometry half is
unchanged whether masks come from GT or from YOLO+SAM.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from branchA_gt import _equirect_sample_maps  # noqa: E402


@lru_cache(maxsize=2)
def _yolo(weights: str, names: tuple[str, ...]):
    from ultralytics import YOLO
    m = YOLO(weights)
    m.set_classes(list(names))
    return m


@lru_cache(maxsize=1)
def _sam(weights: str):
    from ultralytics import SAM
    return SAM(weights)


def _largest_cc(m: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component of a boolean mask, dropping
    stray SAM fragments. No-op if scipy is missing or the mask is single-blob."""
    try:
        from scipy import ndimage
    except ImportError:
        return m
    lbl, n = ndimage.label(m)
    if n <= 1:
        return m
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    return lbl == int(sizes.argmax())


def dets_from_yolo_sam(
    frame_dir: Path,
    names: list[str],
    *,
    n_crops: int = 4,
    fov_h: float = 100.0,
    fov_v_crop: float = 90.0,
    pano_fov_v: float = 120.0,
    cw: int = 640,
    ch: int = 640,
    conf: float = 0.25,
    iou: float = 0.6,
    use_sam: bool = True,
    yolo_weights: str = "yolov8x-worldv2.pt",
    sam_weights: str = "sam2.1_b.pt",
    device: str | None = None,
):
    """Run YOLO-World (+ SAM) over N crops and return equirect-mask detections.

    Each detection's mask is the SAM contour (or the YOLO box rectangle if
    use_sam=False) scattered back into the panorama. Objects straddling two
    overlapping crops yield duplicate detections; lift3d + the cross-frame
    merge dedupe them in 3D, so we keep them here."""
    rgb = np.load(frame_dir / "rgb.npy")[:, :, ::-1]   # bgr8 -> rgb
    H, W = rgb.shape[:2]
    yolo = _yolo(yolo_weights, tuple(names))
    sam = _sam(sam_weights) if use_sam else None

    dets = []
    import math
    for k in range(n_crops):
        yaw_c = 2 * math.pi * k / n_crops
        u, v, valid = _equirect_sample_maps(yaw_c, fov_h, fov_v_crop,
                                            cw, ch, W, H, pano_fov_v)
        crop = rgb[v, u].copy()
        crop[~valid] = 0

        r = yolo.predict(crop, imgsz=cw, conf=conf, iou=iou, device=device,
                         verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        scr = r.boxes.conf.cpu().numpy()

        if sam is not None:
            sm = sam.predict(crop, bboxes=xyxy, device=device, verbose=False)[0]
            cmasks = sm.masks.data.cpu().numpy().astype(bool)  # (n, ch, cw)
        else:
            cmasks = np.zeros((len(xyxy), ch, cw), bool)
            for i, (x1, y1, x2, y2) in enumerate(xyxy.astype(int)):
                cmasks[i, y1:y2, x1:x2] = True

        for cm, box, c, s in zip(cmasks, xyxy, cls, scr):
            # Clip the SAM mask to the YOLO box: box-prompted SAM often bleeds
            # onto the co-planar wall *outside* the detection (the wall-decal /
            # picture spill that inflates 3D boxes and spawns phantom wall nodes).
            # Then keep the largest connected blob to drop stray fragments.
            x1, y1, x2, y2 = box.astype(int)
            bx = np.zeros_like(cm)
            bx[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = True
            cm = cm & bx & valid
            if cm.sum() == 0:
                continue
            cm = _largest_cc(cm)
            em = np.zeros((H, W), bool)
            em[v[cm], u[cm]] = True          # crop pixels -> equirect pixels
            dets.append({"label": names[c], "score": float(s), "mask": em,
                         "crop": k})
    return dets
