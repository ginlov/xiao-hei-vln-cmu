#!/usr/bin/env python3
"""Export a live VLM session scene dump for ``gemini.batch --object-source live``.

Reads the latest (or given) question directory under a ``vlm_logs/session_*``
tree and writes::

    <out-dir>/<scene>/scene.json
    <out-dir>/<scene>/panorama.jpg      # if a camera frame was logged
    <out-dir>/<scene>/trajectory.json   # if poses were logged in ticks

Usage::

    uv run python scripts/export_live_scene_for_offline_eval.py \\
        --session vlm_logs/session_20260711_052457 \\
        --scene livingroom_3 \\
        --out artifacts/explored_scenes
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def _latest_question_dir(session: Path) -> Path:
    qs = sorted(p for p in session.glob("q_*") if p.is_dir())
    if not qs:
        raise SystemExit(f"no q_* directories under {session}")
    return qs[-1]


def _load_last_tick(q_dir: Path) -> dict:
    ticks = q_dir / "ticks.jsonl"
    if not ticks.is_file():
        raise SystemExit(f"missing {ticks}")
    last = None
    for line in ticks.read_text().splitlines():
        line = line.strip()
        if line:
            last = json.loads(line)
    if last is None:
        raise SystemExit(f"empty ticks.jsonl: {ticks}")
    return last


def export_scene(
    *,
    session: Path,
    scene_name: str,
    out_root: Path,
    question_dir: Path | None = None,
) -> Path:
    q_dir = question_dir or _latest_question_dir(session)
    tick = _load_last_tick(q_dir)
    scene = tick.get("scene")
    if not isinstance(scene, dict):
        raise SystemExit(
            f"last tick in {q_dir} has no scene dict — ensure the logger "
            "had a scene attached (scene_gemini / perception runs)."
        )

    dest = out_root / scene_name
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "scene.json").write_text(json.dumps(scene, indent=2) + "\n")

    image_rel = tick.get("image_path")
    if image_rel:
        src = q_dir / image_rel
        if src.is_file():
            shutil.copy2(src, dest / "panorama.jpg")

    traj: list[list[float]] = []
    ticks_path = q_dir / "ticks.jsonl"
    for line in ticks_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        pose = rec.get("pose")
        if isinstance(pose, dict) and "x" in pose and "y" in pose:
            traj.append([float(pose["x"]), float(pose["y"])])
        elif isinstance(pose, dict) and "position" in pose:
            p = pose["position"]
            traj.append([float(p["x"]), float(p["y"])])
    if traj:
        (dest / "trajectory.json").write_text(json.dumps(traj) + "\n")

    print(f"Wrote live scene dump → {dest}", file=sys.stderr)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session",
        type=Path,
        required=True,
        help="Path to vlm_logs/session_<ts>/",
    )
    parser.add_argument(
        "--scene",
        required=True,
        help="Unity scene name (directory name under --out).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output root for <scene>/scene.json dumps.",
    )
    parser.add_argument(
        "--question-dir",
        type=Path,
        default=None,
        help="Specific q_* directory (default: latest under --session).",
    )
    args = parser.parse_args()
    export_scene(
        session=args.session,
        scene_name=args.scene,
        out_root=args.out,
        question_dir=args.question_dir,
    )


if __name__ == "__main__":
    main()
