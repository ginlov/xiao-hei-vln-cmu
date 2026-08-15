"""Rebuild a scene.json (SceneRepresentation.to_dict shape) OFFLINE from the
dump_lifts.py precompute, at a chosen YOLO score + SAM threshold.

The lifts hold every detection's lifted cloud + score + SAM at the low floor, so
re-fusing at (score>=score_thr AND sam>=sam_thr) reproduces exactly the scene
graph the live responder would build at those thresholds — no sidecar. The
result is written in the same shape gemini.batch's --object-source live loads
(load_live_scene -> scene_from_to_dict), so the offline eval can score it.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from xiao_hei_vln.perception.object_map import ObjectMap
from xiao_hei_vln.scene import SceneRepresentation


def build(lifts_path: Path, out_dir: Path, score_thr: float, sam_thr: float):
    payload = pickle.loads(lifts_path.read_bytes())
    om = ObjectMap()
    kept = dropped = 0
    for _vid, dl in payload["lifts"].items():
        for d in dl:
            if d["score"] < score_thr or d["sam"] < sam_thr:
                dropped += 1
                continue
            om.add(d["label"], d["score"], d["pts"])
            kept += 1
    nodes = om.export(min_pts=15, min_obs=1)      # same NMS+prune as the live export
    scene = SceneRepresentation()
    scene.sync_from_object_map(nodes)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scene.json").write_text(json.dumps(scene.to_dict()))
    n_obj = len([n for n in nodes if not n["is_structure"]])
    print(f"lifts={kept+dropped} kept={kept} (score>={score_thr}, sam>={sam_thr}) "
          f"-> {len(nodes)} nodes ({n_obj} non-structure) -> {out_dir/'scene.json'}")
    return scene, nodes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lifts", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True, help="<...>/<scene>/")
    ap.add_argument("--score-threshold", type=float, default=0.4)
    ap.add_argument("--sam-threshold", type=float, default=0.8)
    args = ap.parse_args()
    build(args.lifts, args.out_dir, args.score_threshold, args.sam_threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
