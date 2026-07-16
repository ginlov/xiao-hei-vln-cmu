#!/usr/bin/env python3
"""Export captured frames -> COCO detection dataset (TASK 10).

Pipeline (per captured frame dir with semantic.npy + scan.npy + meta.json +
legend CSVs):
  1. branchA_gt.extract  -> per-instance GT on the 360 panorama
  2. branchA_gt.reproject_crops -> N rectilinear perspective crops + boxes
     (de-warped so a normal detector like YOLO can consume them)
  3. drop non-target classes (structure / ceiling fixtures / arch -- the same
     sets the QA generators use: vla3d_num_gen.BAD_LABELS +
     vla3d_nested_gen.NON_TARGET_LABELS) -> closed-set target vocab
  4. write COCO annotations.json + a flat images/ dir of the crop PNGs

Each annotation keeps rich labels (raw_label, nyu40_label, instance name,
instance colour) in `attributes`, so the SAME data can later be re-exported for
an open-vocab / text-conditioned detector without re-capturing. The COCO
category name uses raw_label (the VLM bridges synonyms downstream).

Usage:
    uv run --extra viz python dataset_generator/export_coco.py \
        --frames dataset_generator/captures/run1/frames \
        --out dataset_generator/captures/run1/coco --scene livingroom
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import branchA_gt as B  # noqa: E402

# Closed-set DROP = structure + ceiling fixtures + architectural non-targets.
# Mirrors vla3d_num_gen.{STRUCTURAL,BORING}_LABELS and
# vla3d_nested_gen.NON_TARGET_LABELS so the detector vocab == the QA target
# vocab. Anything not in here is a countable/queryable target.
DROP = {
    # structural
    "wall", "walls", "exterior walls", "interior wall", "ceiling", "celling",
    "floor", "carpet", "unknown", "column", "columns", "ceiling beam",
    "stair", "stairs", "door", "door frame", "doorframe", "window", "windows",
    "curtain", "curtains",
    # ceiling fixtures / boring
    "focus light", "ceiling lamp", "spot light", "light switch",
    "ceiling light", "_",
    # outdoor scenery seen through glass
    "building", "exterior structure", "sky", "tree", "road", "grass",
    "terrain", "mountain",
}


def build(frames_dir: Path, out_dir: Path, scene: str, n_crops: int,
          fov_h: float, min_box_px: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)
    images, anns = [], []
    cats: dict[str, int] = {}
    img_id = ann_id = 0

    def cat_id(name: str) -> int:
        if name not in cats:
            cats[name] = len(cats) + 1
        return cats[name]

    for fr in sorted(p for p in Path(frames_dir).iterdir() if p.is_dir()):
        if not (fr / "semantic.npy").exists():
            continue
        gt, S = B.extract(fr)
        crops = B.reproject_crops(fr, gt, S, fr / "crops", n=n_crops, fov_h=fov_h)
        for c in crops:
            k = c["index"]
            cw, ch = c["size"]
            fname = f"{scene}_{fr.name}_crop{k}.png"
            shutil.copy(fr / "crops" / f"crop_{k}.png", out_dir / "images" / fname)
            images.append({
                "id": img_id, "file_name": f"images/{fname}",
                "width": cw, "height": ch, "scene": scene,
                "frame": fr.name, "crop": k, "yaw_deg": c["yaw_deg"],
            })
            for b in c["boxes"]:
                if b["raw_label"] in DROP:
                    continue
                x1, y1, x2, y2 = b["bbox_2d"]
                w, h = x2 - x1, y2 - y1
                if w * h < min_box_px or w < 2 or h < 2:
                    continue
                anns.append({
                    "id": ann_id, "image_id": img_id,
                    "category_id": cat_id(b["raw_label"]),
                    "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": 0,
                    "attributes": {
                        "raw_label": b["raw_label"],
                        "nyu40_label": b.get("nyu40_label", ""),
                        "instance": b["instance"],
                        "color_rgb": b["color_rgb"],
                    },
                })
                ann_id += 1
            img_id += 1

    categories = [{"id": i, "name": n}
                  for n, i in sorted(cats.items(), key=lambda kv: kv[1])]
    coco = {
        "info": {"description": f"Branch A perspective-crop detection ({scene})",
                 "vocab": "closed-set raw_label; structure/ceiling-fixtures/arch dropped",
                 "crops": {"n": n_crops, "fov_h_deg": fov_h}},
        "images": images, "annotations": anns, "categories": categories,
    }
    json.dump(coco, open(out_dir / "annotations.json", "w"), indent=2)
    return coco


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scene", default="scene")
    ap.add_argument("--crops", type=int, default=4)
    ap.add_argument("--fov-h", type=float, default=100.0)
    ap.add_argument("--min-box-px", type=int, default=64,
                    help="drop boxes with area below this (px^2)")
    args = ap.parse_args()
    coco = build(args.frames, args.out, args.scene, args.crops, args.fov_h,
                 args.min_box_px)
    print(f"images={len(coco['images'])}  annotations={len(coco['annotations'])}"
          f"  categories={len(coco['categories'])}")
    from collections import Counter
    by = Counter(a["category_id"] for a in coco["annotations"])
    name = {c["id"]: c["name"] for c in coco["categories"]}
    print("per-category boxes:")
    for cid, n in by.most_common():
        print(f"  {n:3d}  {name[cid]}")
    print(f"wrote {args.out/'annotations.json'}  + {len(coco['images'])} crop images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
