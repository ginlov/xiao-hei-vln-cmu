#!/usr/bin/env python3
"""Freeze the first N VLA-3D questions per scene/split for reproducible benches.

Writes::

    <out>/<scene>_ref.jsonl
    <out>/<scene>_num.jsonl

Usage::

    uv run python scripts/freeze_bench_gt.py \\
      --scenes studio chinese_room livingroom_3 office_2 home_building_1 \\
      --num 10 --ref 10 \\
      --out artifacts/bench_5scene_100q/gt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _head_for_scene(src: Path, scene: str, n: int) -> list[dict]:
    rows: list[dict] = []
    if not src.is_file() or n <= 0:
        return rows
    for line in src.open():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("scene") != scene:
            continue
        rows.append(row)
        if len(rows) >= n:
            break
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("/home/ubuntu/workspace/dataset/xiao-hei-vln-cmu/dataset"),
        help="Directory containing vla3d_ref.jsonl / vla3d_num.jsonl",
    )
    p.add_argument("--scenes", nargs="+", required=True)
    p.add_argument("--ref", type=int, default=10, help="Object-reference Qs per scene")
    p.add_argument("--num", type=int, default=10, help="Numerical Qs per scene")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ref_src = args.gt_dir / "vla3d_ref.jsonl"
    num_src = args.gt_dir / "vla3d_num.jsonl"
    total_ref = total_num = 0
    for scene in args.scenes:
        ref_rows = _head_for_scene(ref_src, scene, args.ref)
        num_rows = _head_for_scene(num_src, scene, args.num)
        ref_out = args.out / f"{scene}_ref.jsonl"
        num_out = args.out / f"{scene}_num.jsonl"
        ref_out.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in ref_rows)
            + ("\n" if ref_rows else "")
        )
        num_out.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in num_rows)
            + ("\n" if num_rows else "")
        )
        print(f"{scene}: ref={len(ref_rows)} -> {ref_out}")
        print(f"{scene}: num={len(num_rows)} -> {num_out}")
        total_ref += len(ref_rows)
        total_num += len(num_rows)
    print(f"total: ref={total_ref} num={total_num} out={args.out}")


if __name__ == "__main__":
    main()
