"""CLI: splice our detected scene graph into the Q&A dataset for a scene.

Usage::

    # from a saved scene-graph JSON (SceneRepresentation.to_dict())
    python -m xiao_hei_vln.detected_dataset \\
        --scene-graph runB_om1_scene.json \\
        --scene arabic_room \\
        --qa dataset/vla3d_num.jsonl dataset/vla3d_ref.jsonl \\
        --out-dir dataset/detected

    # or straight from a VLM-logger run (extracts the last-tick scene graph)
    python -m xiao_hei_vln.detected_dataset \\
        --session vlm_logs/session_20260711_052457 \\
        --scene arabic_room

Outputs ``<out-dir>/<scene>_qa.jsonl`` (Q&A + our detected_object_list, feed
to ``xiao_hei_vln.gemini.batch``) and ``<out-dir>/<scene>_truth.jsonl``
(question + ground-truth answer).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from xiao_hei_vln.detected_dataset.builder import (
    _DEFAULT_VLA3D_DIR,
    build_dataset,
    extract_scene_graph_from_session,
)

_DEFAULT_QA = ["dataset/vla3d_num.jsonl", "dataset/vla3d_ref.jsonl"]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--scene-graph", type=Path, help="scene graph JSON (to_dict)")
    src.add_argument("--session", type=Path, help="VLM-logger session dir")
    ap.add_argument("--scene", required=True, help="scene name (matches the QA 'scene' field)")
    ap.add_argument("--qa", nargs="+", default=_DEFAULT_QA, help="Q&A JSONL files")
    ap.add_argument("--out-dir", type=Path, default=Path("dataset/detected"))
    ap.add_argument(
        "--vla3d-dir",
        default=_DEFAULT_VLA3D_DIR,
        help="VLA-3D Unity dir holding <scene>/<scene>_object_result.csv "
        "for GT object colours; pass '' to skip GT-colour join.",
    )
    args = ap.parse_args(argv)

    if args.session is not None:
        scene_graph = extract_scene_graph_from_session(args.session)
    else:
        scene_graph = json.loads(Path(args.scene_graph).read_text())

    spliced, truth, detected = build_dataset(
        scene_graph, args.scene, args.qa, vla3d_dir=args.vla3d_dir or None
    )
    if not spliced:
        print(
            f"WARNING: no Q&A records for scene {args.scene!r} in {args.qa}",
            file=sys.stderr,
        )

    qa_out = args.out_dir / f"{args.scene}_qa.jsonl"
    truth_out = args.out_dir / f"{args.scene}_truth.jsonl"
    _write_jsonl(qa_out, spliced)
    _write_jsonl(truth_out, truth)

    n_num = sum(1 for r in spliced if r.get("type") == "numerical")
    n_ref = sum(1 for r in spliced if r.get("type") == "object_reference")
    print(f"detected objects: {len(detected)}")
    print(f"spliced records:  {len(spliced)}  ({n_num} numerical, {n_ref} object_reference)")
    print(f"wrote {qa_out}")
    print(f"wrote {truth_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
