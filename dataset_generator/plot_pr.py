#!/usr/bin/env python3
"""Plot PR curves (IoU=0.5) for a few classes from the saved YOLO-World eval,
so 'AP = area under the PR curve' is visible.

    uv run python plot_pr.py --coco captures/run3/coco --classes chair book mirror table
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def patched_gt(gt_path):
    d = json.load(open(gt_path))
    for a in d["annotations"]:
        a.setdefault("iscrowd", 0)
        a.setdefault("area", float(a["bbox"][2] * a["bbox"][3]))
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(d, tmp); tmp.close()
    return tmp.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", type=Path, required=True)
    ap.add_argument("--classes", nargs="+", default=["chair", "book", "mirror", "table"])
    args = ap.parse_args()

    coco = COCO(patched_gt(args.coco / "annotations.json"))
    dt = coco.loadRes(str(args.coco / "yoloworld_pred.json"))
    ev = COCOeval(coco, dt, "bbox")
    ev.evaluate(); ev.accumulate()

    prec = ev.eval["precision"]            # [T,R,K,A,M]
    rec = ev.params.recThrs                # 101 recall points (x-axis)
    catids = list(ev.params.catIds)
    name2cat = {c["name"]: c["id"] for c in coco.loadCats(coco.getCatIds())}

    plt.figure(figsize=(7, 6))
    for nm in args.classes:
        cid = name2cat[nm]
        k = catids.index(cid)
        p = prec[0, :, k, 0, -1]           # IoU=.5, area=all, maxDet=100
        p = np.where(p < 0, 0, p)          # -1 means no recall reached -> 0
        ap50 = p.mean()
        ng = len(coco.getAnnIds(catIds=[cid]))
        plt.plot(rec, p, linewidth=2, label=f"{nm}  (AP@.5={ap50:.2f}, #gt={ng})")

    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("YOLO-World v2 zero-shot — PR curves @ IoU=0.5\n(AP@.5 = area under each curve)")
    plt.xlim(0, 1); plt.ylim(0, 1.02)
    plt.grid(alpha=0.3); plt.legend(loc="lower left")
    out = args.coco / "pr_curves.png"
    plt.tight_layout(); plt.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
