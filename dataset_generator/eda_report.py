"""Regenerate the EDA charts for docs/eda_report.md.

Reads dataset/vla3d_ref.jsonl + vla3d_num.jsonl, writes PNGs to
docs/eda_assets/, and prints the tables used in the report. Run after
regen.sh:  uv run python dataset_generator/eda_report.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from vla3d_loader import load_all_vla_scenes  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset"
OUT = ROOT / "docs" / "eda_assets"

PALETTE, PAL2, GREEN, RED = "#4C72B0", "#DD8452", "#55A868", "#C44E52"


def _bar(names, vals, title, fn, horiz=False, color=PALETTE):
    plt.figure(figsize=(7, 3.6))
    if horiz:
        y = range(len(names))
        plt.barh(y, vals, color=color)
        plt.yticks(y, names)
        plt.gca().invert_yaxis()
    else:
        plt.bar(range(len(names)), vals, color=color)
        plt.xticks(range(len(names)), names)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT / fn, dpi=110)
    plt.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line)
            for fn in ("vla3d_ref.jsonl", "vla3d_num.jsonl")
            for line in (DATA / fn).open()]
    ref = [r for r in rows if r["type"] == "object_reference"]
    num = [r for r in rows if r["type"] == "numerical"]
    scenes = load_all_vla_scenes()
    print(f"TOTAL {len(rows)}  ref {len(ref)}  num {len(num)}")
    print("by source", dict(Counter(r["source"] for r in rows)))

    # 1) relation words: official vs ours
    pats = {k: re.compile(v, re.I) for k, v in {
        "on": r"\bon\b", "closest": r"\bclosest\b", "near": r"\bnear\b",
        "between": r"\bbetween\b", "above": r"\babove\b",
        "farthest": r"\b(farthest|furthest)\b", "below": r"\bbelow\b"}.items()}
    g = Counter()
    for r in rows:
        for k, p in pats.items():
            g[k] += len(p.findall(r["question"]))
    total = sum(g.values())
    off = {"on": 42.4, "closest": 28.8, "near": 7.6, "between": 6.1,
           "above": 6.1, "farthest": 4.5, "below": 3.0}
    order = ["on", "closest", "near", "between", "above", "farthest", "below"]
    x = np.arange(len(order))
    w = 0.38
    plt.figure(figsize=(7.5, 3.8))
    plt.bar(x - w / 2, [off[k] for k in order], w, label="Official", color=PAL2)
    plt.bar(x + w / 2, [100 * g[k] / total for k in order], w, label="Ours",
            color=PALETTE)
    plt.xticks(x, order)
    plt.ylabel("% of relation words")
    plt.title("Relation-word distribution: official vs ours")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "relations.png", dpi=110)
    plt.close()

    # 2) target size
    buckets = [("<10cm", 0, .10), ("10-20cm", .10, .20), ("20-40cm", .20, .40),
               ("40-80cm", .40, .80), (">80cm", .80, 99)]
    bc: Counter = Counter()
    for r in ref:
        o = scenes[r["scene"]].by_id.get(r["target"])
        if not o:
            continue
        md = max(o.lx, o.ly, o.lz)
        for name, lo, hi in buckets:
            if lo <= md < hi:
                bc[name] += 1
                break
    _bar([b[0] for b in buckets], [bc[b[0]] for b in buckets],
         "Target object size (largest bbox edge)", "target_size.png")

    # 3) per-scene
    sc = Counter(r["scene"] for r in rows)
    names = [k for k, _ in sc.most_common()]
    _bar(names, [sc[k] for k in names], "Samples per scene", "per_scene.png",
         horiz=True)

    # 4) colours
    colre = re.compile(r"\b(red|blue|green|black|white|orange|yellow|brown|"
                       r"gray|grey|purple|pink)\b", re.I)
    cc: Counter = Counter()
    for r in rows:
        for m in colre.findall(r["question"]):
            cc[m.lower()] += 1
    cn = [k for k, _ in cc.most_common()]
    _bar(cn, [cc[k] for k in cn], "Colour words (all basic + dominant)",
         "colors.png", color=GREEN)

    # 5) numerical answers
    na = Counter(r["answer"] for r in num if isinstance(r["answer"], int))
    ks = sorted(na)
    _bar([str(k) for k in ks], [na[k] for k in ks], "Numerical answer values",
         "num_answers.png", color=RED)

    print("relations:", {k: round(100 * g[k] / total, 1) for k in order})
    print("size:", dict(bc))
    print("colours:", dict(cc.most_common()))
    print("num answers:", dict(sorted(na.items())))
    print("charts ->", OUT)


if __name__ == "__main__":
    main()
