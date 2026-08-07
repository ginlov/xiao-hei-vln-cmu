#!/usr/bin/env python3
"""Why each false positive was wrong — one annotated panel per case.

The sweep says 22% of claimed sightings land more than 1 m from ground truth,
which is the number blocking the whole approach. "22%" does not say what to
fix, though: a box on the wrong object needs a different remedy from a correct
box whose ray stopped on a nearer surface. This draws every failure so the two
can be told apart by eye, and prints the one number that separates them —
the lifted range against the true range.

Yellow is what the model claimed. Cyan is where ground truth actually sits,
projected into the same face; when cyan is absent the truth was outside this
face entirely, which is its own diagnosis.

Runs entirely off the cached replies, so it costs nothing.

    uv run python scripts/vlm_fp_report.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "perception"))
import geometry as G  # noqa: E402
from vlm_sweep import family_key  # noqa: E402
from vlm_locate import locate, rot_from_quat, scan_to_camera  # noqa: E402
from vlm_probe import to_pixels  # noqa: E402
from vlm_sweep import CHALLENGE, PROMPT_VER, faces_of, scene_frame  # noqa: E402

PANEL = 460
OUT = Path("artifacts/fp")


def gt_to_face(gt_xyz: np.ndarray, pose: dict, face_idx: int):
    """Ground-truth point → pixel in ``face_idx``, or None if not in it."""
    from xiao_hei_vln.perception.geometry import sensor_to_camera_transform
    R_sc, t_sc = sensor_to_camera_transform()
    R_ms = rot_from_quat(pose["orientation"])
    p_sensor = (gt_xyz - np.asarray(pose["position"])) @ R_ms
    p_cam = p_sensor @ R_sc.T + t_sc
    x, y, ok = G.world_dir_to_face_pixel(p_cam[None, :], face_idx)
    if not bool(np.asarray(ok).ravel()[0]):
        return None
    x, y = float(np.asarray(x).ravel()[0]), float(np.asarray(y).ravel()[0])
    if not (0 <= x < G.FACE_SIZE and 0 <= y < G.FACE_SIZE):
        return None
    return x, y


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    rows = json.loads(Path("artifacts/vlm_sweep.json").read_text())
    cache = json.loads(Path("artifacts/vlm_sweep_cache.json").read_text())
    qs = {e["scene"]: e["questions"] for e in
          json.loads((CHALLENGE / "questions/questions.json").read_text())}
    OUT.mkdir(parents=True, exist_ok=True)

    fps = [r for r in rows if r["claimed"] and not r["hit"]]
    panels, report = [], []

    for r in fps:
        scene, phrase = r["scene"], r["phrase"]
        ck = f"{PROMPT_VER}|claude|claude-opus-5|{scene}|{phrase}"
        m = re.search(r"\{.*\}", cache[ck], re.S)
        rep = json.loads(m.group(0))
        i = int(rep["image_index"])

        eq, scan, pose = scene_frame(scene)
        faces = faces_of(eq)
        img = cv2.imdecode(np.frombuffer(faces[i], np.uint8), cv2.IMREAD_COLOR)

        d = json.loads((Path("viz/data") / f"{scene}.json").read_text())
        labels = [x.strip().lower() for x in d["labels"]]
        truth = [np.array(o["c"]) for o in d["gt"]
                 if family_key(labels[o["l"]]) == family_key(phrase)]

        box = rep.get("feature_box_2d") or rep.get("box_2d")
        px = to_pixels(box, rep.get("coord_space"), G.FACE_SIZE)
        res = locate(px, i, scan_to_camera(scan, pose), cone_deg=2.0, pose=pose)

        ymin, xmin, ymax, xmax = [int(v) for v in px]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 255), 3)

        # Ground truth in this face, if it is in this face at all.
        origin = np.asarray(pose["position"])
        gt_here, gt_rng = [], []
        for g in truth:
            gt_rng.append(float(np.linalg.norm(g[:2] - origin[:2])))
            p = gt_to_face(g, pose, i)
            if p:
                gt_here.append(p)
                cv2.drawMarker(img, (int(p[0]), int(p[1])), (255, 220, 0),
                               cv2.MARKER_TILTED_CROSS, 26, 3)
        lift_rng = res.get("range")
        nearest = min(gt_rng) if gt_rng else float("nan")

        cause = ("no lidar return" if not res.get("n") else
                 "truth not in this face" if not gt_here else
                 "ray stopped short" if lift_rng and lift_rng < nearest - 1.0 else
                 "ray overshot" if lift_rng and lift_rng > nearest + 1.0 else
                 "lateral / wrong instance")
        report.append({
            "scene": scene, "q": r["q"], "phrase": phrase, "err": r["err"],
            "conf": r["conf"], "face": i, "lift_range": lift_rng,
            "gt_range_nearest": None if not gt_rng else round(nearest, 2),
            "n_gt": len(truth), "gt_in_face": len(gt_here), "cause": cause,
            "evidence": (rep.get("evidence") or "")[:200],
        })

        p = cv2.resize(img, (PANEL, PANEL))
        bar = np.full((62, PANEL, 3), 22, np.uint8)
        err = "no lift" if r["err"] is None else f"{r['err']:.2f}m"
        cv2.putText(bar, f"{scene} q{r['q']}  {phrase}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(bar, f"err {err}  conf {r['conf']}  lift "
                         f"{'—' if lift_rng is None else f'{lift_rng:.1f}m'}"
                         f" / gt {nearest:.1f}m", (6, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 200, 235), 1, cv2.LINE_AA)
        cv2.putText(bar, cause, (6, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (120, 200, 255), 1, cv2.LINE_AA)
        panels.append(np.vstack([bar, p]))

    cols = 4
    rows_n = (len(panels) + cols - 1) // cols
    ph, pw = panels[0].shape[:2]
    sheet = np.full((rows_n * (ph + 8) + 8, cols * (pw + 8) + 8, 3), 12, np.uint8)
    for k, p in enumerate(panels):
        y, x = 8 + (k // cols) * (ph + 8), 8 + (k % cols) * (pw + 8)
        sheet[y:y + ph, x:x + pw] = p
    cv2.imwrite(str(OUT / "false_positives.jpg"), sheet,
                [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"{'scene':14s} {'phrase':18s} {'err':>7s} {'lift':>6s} {'gt':>6s} "
          f"{'nGT':>3s} {'inface':>6s}  cause")
    print("-" * 92)
    for x in sorted(report, key=lambda z: z["cause"]):
        e = "  —  " if x["err"] is None else f"{x['err']:6.2f}"
        lr = "  —  " if x["lift_range"] is None else f"{x['lift_range']:6.2f}"
        gr = "  —  " if x["gt_range_nearest"] is None else f"{x['gt_range_nearest']:6.2f}"
        print(f"{x['scene']:14s} {x['phrase']:18s} {e} {lr} {gr} "
              f"{x['n_gt']:3d} {x['gt_in_face']:6d}  {x['cause']}")

    print("\ncause counts:")
    for c in sorted({x["cause"] for x in report}):
        print(f"  {sum(1 for x in report if x['cause'] == c):2d}  {c}")

    (OUT / "false_positives.json").write_text(json.dumps(report, indent=1))
    print(f"\nwrote {OUT / 'false_positives.jpg'} and .json")
    for x in sorted(report, key=lambda z: z["cause"]):
        print(f"\n[{x['cause']}] {x['scene']} {x['phrase']}\n  "
              f"{qs[x['scene']]['instruction_following'][x['q'] - 4]}\n  "
              f"VLM: {x['evidence']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
