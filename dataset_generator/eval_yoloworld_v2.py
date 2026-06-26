#!/usr/bin/env python3
"""Zero-cost improvements over the bare-name YOLO-World eval (TASK 10):
  (1) merge synonym classes in the GT (picture/painting/photo; table family),
  (2) prompt YOLO-World with descriptive phrases for weak/ambiguous classes.
Re-scores with COCOeval so we can compare to the 0.369 mAP@.5 baseline.

    uv run python eval_yoloworld_v2.py --coco captures/run3/coco --device mps
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# (1) synonym merges: near-identical concepts the open-vocab model conflates.
MERGE = {
    "painting": "picture", "photo": "picture",   # all wall art
    "coffee table": "table",                       # both tables
    "tv cabinet": "cabinet",                       # both cabinets/consoles
}

# (2) descriptive prompts (canonical class -> CLIP text). Strong classes keep
# their bare noun (descriptive wording can hurt them); only ambiguous/weak ones
# get richer phrases.
PROMPT = {
    "picture": "a framed picture or painting on the wall",
    "wall decal": "a decorative wall sticker or wall decal",
    "book": "a book on a shelf",
    "table": "a table",
    "cabinet": "a wooden cabinet or media console",
    "shelf": "a shelf",
    "stool": "a stool",
    "lamp": "a lamp",
    "flower": "a flower in a vase",
    "dvd": "a dvd case",
    "chess": "a chess board set",
    "magazine": "a magazine",
    "tablecloth": "a tablecloth covering a table",
    "decorative ball": "a decorative ball ornament",
    "newtons cradle": "a newton's cradle desk toy",
    "elephant decoration": "an elephant figurine ornament",
    "buddha decoration": "a buddha statue figurine",
    "tray": "a serving tray",
    "box": "a small box",
}


def build_merged_gt(gt_path):
    """Return (tmp_gt_file, canonical_names, name->canonical_id)."""
    d = json.load(open(gt_path))
    orig = {c["id"]: c["name"] for c in d["categories"]}
    # canonical class list, stable order = first appearance
    canon_order = []
    for c in d["categories"]:
        cn = MERGE.get(c["name"], c["name"])
        if cn not in canon_order:
            canon_order.append(cn)
    canon_id = {cn: i + 1 for i, cn in enumerate(canon_order)}
    d["categories"] = [{"id": canon_id[cn], "name": cn} for cn in canon_order]
    for a in d["annotations"]:
        cn = MERGE.get(orig[a["category_id"]], orig[a["category_id"]])
        a["category_id"] = canon_id[cn]
        a.setdefault("iscrowd", 0)
        a.setdefault("area", float(a["bbox"][2] * a["bbox"][3]))
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(d, tmp); tmp.close()
    return tmp.name, canon_order, canon_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", type=Path, required=True)
    ap.add_argument("--weights", default="yolov8x-worldv2.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    gt_file, canon, canon_id = build_merged_gt(args.coco / "annotations.json")
    coco = COCO(gt_file)
    prompts = [PROMPT.get(c, c) for c in canon]
    catid_of_index = [canon_id[c] for c in canon]
    print(f"{len(canon)} canonical classes (after merge)")
    print("prompts overridden:", [c for c in canon if c in PROMPT])

    model = YOLO(args.weights)
    model.set_classes(prompts)

    preds = []
    imgs = coco.loadImgs(coco.getImgIds())
    for i, im in enumerate(imgs):
        r = model.predict(str(args.coco / im["file_name"]), imgsz=args.imgsz,
                          conf=args.conf, iou=0.6, max_det=300,
                          device=args.device, verbose=False)[0]
        xyxy = r.boxes.xyxy.cpu().numpy(); cls = r.boxes.cls.cpu().numpy().astype(int)
        scr = r.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c, s in zip(xyxy, cls, scr):
            preds.append({"image_id": im["id"], "category_id": catid_of_index[c],
                          "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                          "score": float(s)})
        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(imgs)}")
    pred_file = args.coco / "yoloworld_v2_pred.json"
    json.dump(preds, open(pred_file, "w"))
    print("total detections:", len(preds))

    dt = coco.loadRes(str(pred_file))
    ev = COCOeval(coco, dt, "bbox"); ev.evaluate(); ev.accumulate(); ev.summarize()

    prec = ev.eval["precision"]; catids = list(ev.params.catIds)
    name_of = {canon_id[c]: c for c in canon}
    rows = []
    for k, cid in enumerate(catids):
        p50 = prec[0, :, k, 0, -1]; p = prec[:, :, k, 0, -1]
        ap50 = p50[p50 > -1].mean() if (p50 > -1).any() else float("nan")
        apc = p[p > -1].mean() if (p > -1).any() else float("nan")
        ng = len(coco.getAnnIds(catIds=[cid]))
        rows.append((name_of[cid], ng, ap50, apc))
    rows.sort(key=lambda r: (-r[2] if r[2] == r[2] else 1))
    print(f"\n  {'class':22s} {'#gt':>4s} {'AP@.5':>7s} {'AP@.5:.95':>10s}")
    for nm, ng, ap50, apc in rows:
        print(f"  {nm:22s} {ng:4d} {ap50:7.3f} {apc:10.3f}")


if __name__ == "__main__":
    main()
