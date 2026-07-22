# TASK 13 - Merge our perception stack into the main pipeline as enhancements

## Purpose

`main` already carries a complete SysNav perception pipeline (PR #13, giangvl):
sidecar YOLO-World v2 + SAM 2.1 → `PointLifter` (2D mask + LiDAR + pose → 3D
point) → `SceneRepresentation` → `PerceptionResponder`. It runs inference on its
own. This task folds in the two capabilities our `feature/perception-live-stack`
branch had that it lacked, **as opt-in enhancements** — leaving the default
online behaviour byte-for-byte unchanged.

Our `dataset_generator/perception/` stays as the offline benchmark / capture tool;
the reusable cores were ported into the package `src/xiao_hei_vln/perception/`.

## What changed

**Rebase.** `feature/perception-live-stack` rebased onto `origin/main` (23 commits
ahead). Only conflict was `.gitignore` (union of both). Not force-pushed yet —
awaiting confirmation.

**Workstream C — quantitative evaluation** (`src/xiao_hei_vln/perception/eval.py`, new)
- Ported the self-contained numpy evaluator from `eval_objectmap.py`: mAP @
  center-distance + @ 3D-IoU, operating-point P/R/F1, per-class counting error
  (numerical-question proxy), matched center error, label confusion.
- Adapters: `scene_objects_to_eval` (SceneRepresentation.to_dict) and
  `object_entries_to_eval` (authoritative VLA-3D `object_list` via
  `scene.io.read_objects_from_zip`); predictions with no box get a degenerate
  point-box so IoU code never breaks.
- CLI: `python -m xiao_hei_vln.perception.eval --scene scene.json --gt-zip <zip>`.
- Zero runtime risk (offline only).

**Workstream A — PointLifter** (`src/xiao_hei_vln/perception/lifter.py`)
- `LiftResult` gains `inlier_points` (map-frame cloud) for cross-frame fusion.
- Optional **z-buffer occlusion gate** (`enable_zbuffer`, `ZBUF_TOL_M=0.2`): at
  each equirect pixel only the nearest surface survives, so a far wall seen
  through a mask no longer contaminates the object. **Defaults to `False`** — the
  historical median is unchanged; the ObjectMap path opts in.
- Kept their full quaternion + sensor→camera extrinsic projection.

**Workstream B — ObjectMap fusion** (`src/xiao_hei_vln/perception/object_map.py`, new)
- Self-contained port of `objectmap.py` (`robust_center` + `LIDAR_GATE_M` inlined,
  no `dataset_generator` dependency). Cross-frame **point-cloud union → converged
  AABB**, 3D-IoU merge, cross-label NMS, wall-sheet phantom rejection, evidence
  pruning. Each node has a stable `node_id` + colour.
- `SceneRepresentation.sync_from_object_map(nodes)`: rebuilds the object layer
  from a fused export, **filling the `bbox_min`/`bbox_max` the scene graph left
  `None`**, keeping `object_id` stable per `node_id` across ticks, recording the
  current viewpoint edge, preserving spatial relations.
- `PerceptionResponder` gains an optional `object_map`; when set, `_inject_visible`
  fuses each detection's lifted cloud and syncs once per tick instead of calling
  `add_object`.
- `app/main.py`: env flag **`XIAO_HEI_OBJECT_MAP=1`** builds the ObjectMap + a
  z-buffer lifter and injects them. **Default off → pipeline behaves exactly as
  before.**

## Verification

- `uv run pytest tests/ -q` → **302 passed, 1 skipped** (shapely, pre-existing).
  New: `test_perception_eval.py` (8), `test_object_map.py` (7), z-buffer +
  inlier-points cases in `test_perception_lifter.py` (4). Their existing lifter /
  scene / responder tests unchanged and green (default path untouched).
- eval CLI smoke: dumped scene graph vs synthetic GT → sensible mAP/P/R/F1/counting.
- End-to-end C+B: ObjectMap → `sync_from_object_map` → `scene.to_dict()` → eval:
  boxes populated, **IoU-mAP 0.5 (was 0.0 with `None` boxes)** — B unlocks C.
- Not yet run: on-sim A/B (`XIAO_HEI_OBJECT_MAP=0/1`) — needs xiaohei1 + sim.

## Files

New: `src/xiao_hei_vln/perception/eval.py`, `.../object_map.py`,
`tests/test_perception_eval.py`, `tests/test_object_map.py`.
Modified: `src/xiao_hei_vln/perception/lifter.py`, `.../responder.py`,
`src/xiao_hei_vln/scene/representation.py`, `src/xiao_hei_vln/app/main.py`,
`tests/test_perception_lifter.py`.

## Deferred (contribution 3)

rviz live scoreboard / GT boxes / object markers. Not on the challenge-scoring
path; revisit after the ObjectMap A/B on-sim comparison.
