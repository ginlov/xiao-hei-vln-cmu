"""Freeze a single viewpoint and show the 2D→3D geometry for one class.

The merge video answers "how did the graph get here"; this answers "at *this*
viewpoint, did the lift land where the 2D detection points". For every
detection of the chosen class in the viewpoint it:

  * reprojects the committed 3D lift back into the panorama (using the exact
    forward projection the lifter uses) and marks the column, so you can see
    whether the 3D point and the 2D mask agree in azimuth;
  * draws a top-down panel with the robot + heading, the bearing wedge the GT
    footprint subtends from the robot, the lift point, and the GT footprints —
    so an *angular* error (lift outside the wedge) is visually distinct from a
    *radial* one (lift inside the wedge but short/long of the surface).

    uv run python perception_benchmark/inspect_vp.py \
        --scene arabic_room --label carpet --vp vp_009 \
        --run perception_benchmark/debug

Saves <run>/<scene>/inspect/<vp>_<label>.png (and _pano.png). Reads a
dump_debug.py directory only — no sidecar, no GPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Wedge
from PIL import Image

matplotlib.use("Agg")

from xiao_hei_vln.perception.geometry import (  # noqa: E402
    EQUIRECT_W,
    project_camera_points_to_equirect,
    sensor_to_camera_transform,
)


def _matches(label: str, wanted: str) -> bool:
    low = label.lower()
    return wanted.lower() == low or wanted.lower() in low


def _reproject(p_map, pose, yaw):
    """Map-frame point → equirect (u, v, valid), via sensor & camera frames."""
    c, s = np.cos(-yaw), np.sin(-yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    p_sensor = Rz @ (np.asarray(p_map, float) - np.asarray(pose, float))
    Rc, tc = sensor_to_camera_transform()
    p_cam = Rc @ p_sensor + tc
    u, v, valid = project_camera_points_to_equirect(p_cam.reshape(1, 3))
    return float(u[0]), float(v[0]), bool(valid[0])


def _footprint(bmin, bmax):
    return bmin[0], bmin[1], bmax[0] - bmin[0], bmax[1] - bmin[1]


def render(args) -> None:
    run = Path(args.run)
    data = json.loads((run / args.scene / "viz.json").read_text())
    vps = data["viewpoints"]
    vp = next((v for v in vps if v["id"] == args.vp), None)
    if vp is None:                                  # allow a bare index too
        try:
            vp = vps[int(args.vp)]
        except (ValueError, IndexError):
            raise SystemExit(f"no viewpoint {args.vp!r} in {args.scene}")
    label = args.label
    pose, yaw = vp["pose"], vp["yaw"]
    R = np.array(pose[:2])

    dets = [d for d in vp["detections"] if _matches(d["label"], label)]
    gts = [g for g in data["gt"] if _matches(g["label"], label)]
    lifts = [d for d in dets if d["lifted"] and d["position"]]
    print(f"{vp['id']}: {len(dets)} {label} detections, {len(lifts)} lifted; "
          f"pose {[round(x, 3) for x in pose]}, yaw {np.degrees(yaw):.1f}°")

    out_dir = run / args.scene / "inspect"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- panorama with reprojected lift columns --------------------------
    png = run / args.scene / f"{vp['id']}.png"
    fig_p, ax_p = plt.subplots(figsize=(args.width, args.width / 3.0),
                               dpi=args.dpi)
    ax_p.axis("off")
    pano_w = None
    if png.is_file():
        img = Image.open(png)
        ax_p.imshow(img)
        pano_w = img.size[0]
    for d in lifts:
        u, v, valid = _reproject(d["position"], pose, yaw)
        frac = u / EQUIRECT_W
        if pano_w:
            x = frac * pano_w
            ax_p.axvline(x, color="red", lw=1.5, alpha=0.9)
            ax_p.text(x, 12, f" lift n{d['node_id']} ({frac * 100:.0f}%)",
                      color="red", fontsize=8, va="top")
    ax_p.set_title(f"{vp['id']} · {label} · red = 3D lift reprojected to 2D "
                   f"(should sit on the mask)", fontsize=9)
    pano_out = out_dir / f"{vp['id']}_{label}_pano.png"
    fig_p.savefig(pano_out, bbox_inches="tight")
    plt.close(fig_p)

    # ---- top-down bearing / range geometry -------------------------------
    fig, ax = plt.subplots(figsize=(args.width * 0.65, args.width * 0.65),
                           dpi=args.dpi)
    # GT footprints + the bearing wedge each subtends from the robot.
    for g in gts:
        fx, fy, fw, fh = _footprint(g["bmin"], g["bmax"])
        ax.add_patch(Rectangle((fx, fy), fw, fh, fill=False, ec="black",
                               ls="--", lw=1.4))
        corners = np.array([[g["bmin"][0], g["bmin"][1]],
                            [g["bmin"][0], g["bmax"][1]],
                            [g["bmax"][0], g["bmin"][1]],
                            [g["bmax"][0], g["bmax"][1]]])
        angs = np.degrees(np.arctan2(corners[:, 1] - R[1], corners[:, 0] - R[0]))
        rng = np.linalg.norm(corners - R, axis=1).max()
        ax.add_patch(Wedge(R, rng, angs.min(), angs.max(), fc="green",
                           alpha=0.10, ec="green", lw=0.6))
    # Every lift in the viewpoint, joined to the robot by its bearing ray.
    for d in lifts:
        p = np.array(d["position"][:2])
        ax.plot([R[0], p[0]], [R[1], p[1]], color="red", lw=0.8, ls=":")
        ax.scatter(*p, s=70, c="red", edgecolors="black", zorder=6)
        ax.text(p[0], p[1], f" n{d['node_id']}", color="red", fontsize=8,
                zorder=7)
        b = np.degrees(np.arctan2(p[1] - R[1], p[0] - R[0]))
        # Is this lift inside the nearest GT's angular wedge?
        note = ""
        if gts:
            g = min(gts, key=lambda g: np.linalg.norm(np.array(g["center"][:2]) - p))
            corners = np.array([[g["bmin"][0], g["bmin"][1]],
                                [g["bmin"][0], g["bmax"][1]],
                                [g["bmax"][0], g["bmin"][1]],
                                [g["bmax"][0], g["bmax"][1]]])
            angs = np.degrees(np.arctan2(corners[:, 1] - R[1],
                                         corners[:, 0] - R[0]))
            inside = angs.min() <= b <= angs.max()
            inx = g["bmin"][0] <= p[0] <= g["bmax"][0]
            iny = g["bmin"][1] <= p[1] <= g["bmax"][1]
            note = (f"lift n{d['node_id']}: bearing {b:.1f}° "
                    f"{'INSIDE' if inside else 'OUTSIDE'} GT wedge "
                    f"[{angs.min():.0f}°,{angs.max():.0f}°]; "
                    f"in GT box x={inx} y={iny}")
            print("  " + note)
    # Robot + heading.
    ax.scatter(*R, s=90, c="royalblue", marker="o", zorder=8, label="robot")
    ax.arrow(R[0], R[1], 0.7 * np.cos(yaw), 0.7 * np.sin(yaw),
             head_width=0.15, color="royalblue", zorder=8)

    ax.set_aspect("equal")
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{vp['id']} top-down · green wedge = GT angular span from "
                 f"robot\nlift inside wedge ⇒ azimuth OK (offset is radial)",
                 fontsize=9)
    top_out = out_dir / f"{vp['id']}_{label}.png"
    fig.savefig(top_out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {pano_out}\nwrote {top_out}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="arabic_room")
    p.add_argument("--run", default="perception_benchmark/debug")
    p.add_argument("--label", default="carpet")
    p.add_argument("--vp", default="vp_009", help="viewpoint id or index")
    p.add_argument("--width", type=float, default=13.0)
    p.add_argument("--dpi", type=int, default=120)
    render(p.parse_args())


if __name__ == "__main__":
    main()
