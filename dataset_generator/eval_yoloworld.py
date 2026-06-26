#!/usr/bin/env python3
"""Zero-shot YOLO-World v2 vs Branch A GT (TASK 10).

Loads `yolov8x-worldv2` (the detector the organizers' SysNav uses), prompts it
with our COCO class names, runs it on the run3 crop images, and scores per-class
mAP against the Branch A GT with COCOeval. Answers "which classes work zero-shot
vs which need fine-tuning" with real numbers instead of a guess.

    uv run python eval_yoloworld.py --coco captures/run3/coco [--device mps]
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def patched_gt(gt_path: Path) -> str:
    """COCOeval needs area + iscrowd on every annotation; add them if absent."""
    d = json.load(open(gt_path))
    for a in d["annotations"]:
        a.setdefault("iscrowd", 0)
        if "area" not in a:
            a["area"] = float(a["bbox"][2] * a["bbox"][3])
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(d, tmp)
    tmp.close()
    return tmp.name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", type=Path, required=True, help="dir with annotations.json + images/")
    ap.add_argument("--weights", default="yolov8x-worldv2.pt")
    ap.add_argument("--device", default=None, help="mps / cpu / 0; default auto")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    gt_file = patched_gt(args.coco / "annotations.json")
    coco = COCO(gt_file)
    cats = sorted(coco.loadCats(coco.getCatIds()), key=lambda c: c["id"])
    names = [c["name"] for c in cats]
    catid_of_index = [c["id"] for c in cats]    # pred class idx -> coco category_id
    print(f"{len(names)} classes prompted to YOLO-World:\n  {names}")

    model = YOLO(args.weights)
    model.set_classes(names)

    preds = []
    imgs = coco.loadImgs(coco.getImgIds())
    for i, im in enumerate(imgs):
        fp = args.coco / im["file_name"]
        r = model.predict(str(fp), imgsz=args.imgsz, conf=args.conf, iou=0.6,
                          max_det=300, device=args.device, verbose=False)[0]
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        scr = r.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c, s in zip(xyxy, cls, scr):
            preds.append({"image_id": im["id"], "category_id": catid_of_index[c],
                          "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                          "score": float(s)})
        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(imgs)} images")
    print(f"total detections: {len(preds)}")

    pred_file = args.coco / "yoloworld_pred.json"
    json.dump(preds, open(pred_file, "w"))
    if not preds:
        print("no detections at all -- check weights/prompts"); return 1

    dt = coco.loadRes(str(pred_file))
    ev = COCOeval(coco, dt, "bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()

    # per-class AP@.5 and AP@[.5:.95]
    prec = ev.eval["precision"]                 # [T,R,K,A,M]
    k_catids = list(ev.params.catIds)
    name_of_catid = {c["id"]: c["name"] for c in cats}
    gt_count = {cid: len(coco.getAnnIds(catIds=[cid])) for cid in k_catids}
    rows = []
    for k, cid in enumerate(k_catids):
        p50 = prec[0, :, k, 0, -1]; p = prec[:, :, k, 0, -1]
        ap50 = p50[p50 > -1].mean() if (p50 > -1).any() else float("nan")
        ap = p[p > -1].mean() if (p > -1).any() else float("nan")
        rows.append((name_of_catid[cid], gt_count[cid], ap50, ap))
    rows.sort(key=lambda r: (-r[2] if r[2] == r[2] else 1))   # NaN last
    print("\nper-class  (sorted by AP@.5):")
    print(f"  {'class':22s} {'#gt':>4s} {'AP@.5':>7s} {'AP@.5:.95':>10s}")
    for nm, ng, ap50, ap in rows:
        print(f"  {nm:22s} {ng:4d} {ap50:7.3f} {ap:10.3f}")
    good = [r[0] for r in rows if r[2] == r[2] and r[2] >= 0.30]
    weak = [r[0] for r in rows if not (r[2] == r[2] and r[2] >= 0.30)]
    print(f"\nzero-shot OK (AP@.5>=0.30): {good}")
    print(f"needs fine-tune (<0.30 / 0): {weak}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
