"""Top-down (BEV) visualization of scene viewpoints for review.

Plots walls (occluders), free space, objects (covered vs unobservable), the
floor footprint, and the generated viewpoints (0-indexed labels = their ``id``)
with faint coverage fans. Viewpoints NOT over the room floor are flagged in red
("OUT") — these are exterior free-space points that should be dropped.

    uv run python perception_benchmark/viz_viewpoints.py --scene livingroom_3
    uv run python perception_benchmark/viz_viewpoints.py --all      # 15-scene montage
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from viewgen import (ALL_SCENES, load_objects, load_free_points, load_wall_points,
                     load_floor_xy, over_floor_mask)

VP_DIR = Path("perception_benchmark/viewpoints")


def plot_scene(ax, scene: str) -> dict:
    data = json.load(open(VP_DIR / f"{scene}.json"))
    vps = data["viewpoints"]
    unobs = set(data["report"]["not_observable"])
    objs = load_objects(scene)
    free = load_free_points(scene)
    walls = load_wall_points(scene)
    floor = load_floor_xy(scene)

    if len(walls):
        ax.scatter(walls[:, 0], walls[:, 1], s=1, c="0.35", alpha=0.5, linewidths=0)
    ax.scatter(free[:, 0], free[:, 1], s=6, c="#8fc98a", alpha=0.3, linewidths=0)

    ox = np.array([[o.center.x, o.center.y] for o in objs.values()])
    oid = list(objs.keys())
    cov = np.array([i not in unobs for i in oid])
    ax.scatter(ox[cov, 0], ox[cov, 1], s=12, c="#3b6", alpha=0.8, zorder=3)
    if (~cov).any():
        ax.scatter(ox[~cov, 0], ox[~cov, 1], s=45, c="#d33", marker="x",
                   linewidths=1.5, zorder=4)

    # flag viewpoints outside the room floor
    vp_xy = np.array([[v["x"], v["y"]] for v in vps])
    inside = over_floor_mask(floor, vp_xy)

    cmap = plt.cm.tab10
    ocenter = {o.object_id: (o.center.x, o.center.y) for o in objs.values()}
    n_out = 0
    for k, vp in enumerate(vps):
        col = cmap(k % 10)
        for oc in vp.get("covers", []):
            if oc in ocenter:
                cx, cy = ocenter[oc]
                ax.plot([vp["x"], cx], [vp["y"], cy], c=col, alpha=0.05, lw=0.5, zorder=1)
        if inside[k]:
            ax.scatter([vp["x"]], [vp["y"]], s=220, c=[col], marker="*",
                       edgecolors="black", linewidths=1.1, zorder=5)
        else:                                   # outside room floor -> red flag
            n_out += 1
            ax.scatter([vp["x"]], [vp["y"]], s=300, c="none", marker="*",
                       edgecolors="#d00", linewidths=2.2, zorder=6)
            ax.annotate("OUT", (vp["x"], vp["y"]), fontsize=7, color="#d00",
                        ha="center", va="bottom", xytext=(0, 9),
                        textcoords="offset points", zorder=7)
        ax.annotate(str(vp["id"]), (vp["x"], vp["y"]), fontsize=9, fontweight="bold",
                    ha="center", va="center", zorder=8,
                    color="white" if k % 10 not in (1, 8, 9) else "black")

    r = data["report"]
    ax.set_title(f"{scene} — {len(vps)} vps, {r['coverage_frac']:.0%} cov"
                 + (f", {n_out} OUT" if n_out else ""), fontsize=10)
    ax.set_aspect("equal"); ax.grid(alpha=0.2)
    return {"scene": scene, "n_vps": len(vps), "n_out": n_out,
            "out_ids": [vps[k]["id"] for k in range(len(vps)) if not inside[k]]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.all:
        fig, axes = plt.subplots(5, 3, figsize=(22, 34))
        summary = []
        for ax, scene in zip(axes.flat, ALL_SCENES):
            summary.append(plot_scene(ax, scene))
        for ax in axes.flat[len(ALL_SCENES):]:
            ax.axis("off")
        fig.suptitle("Viewpoint review — all 15 scenes (labels = viewpoint id, 0-based; "
                     "red ★ = OUT = viewpoint not over room floor)", fontsize=14, y=0.998)
        fig.tight_layout(rect=[0, 0, 1, 0.99])
        out = args.out or VP_DIR / "_all_scenes_map.png"
        fig.savefig(out, dpi=85)
        print("wrote", out)
        print(f"\n{'scene':18s} {'vps':>4s} {'OUT':>4s}  out_ids")
        for s in summary:
            print(f"{s['scene']:18s} {s['n_vps']:4d} {s['n_out']:4d}  {s['out_ids']}")
        return 0

    scene = args.scene
    if not scene:
        ap.error("pass --scene <name> or --all")
    fig, ax = plt.subplots(figsize=(12, 12))
    info = plot_scene(ax, scene)
    ax.set_xlabel("x (m, map)"); ax.set_ylabel("y (m, map)")
    fig.tight_layout()
    out = args.out or VP_DIR / f"{scene}_map.png"
    fig.savefig(out, dpi=110)
    print("wrote", out, "| OUT viewpoints:", info["out_ids"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
