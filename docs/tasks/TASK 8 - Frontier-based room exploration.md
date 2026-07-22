# TASK 8 — Frontier-based room exploration (Phase A)

## Problem

The existing responders (`dummy`, `qwen`) emit one-shot answers but do
nothing to **gather scene information** before answering. In practice
the challenge's living-room-sized environment with furniture occlusion
needs an active exploration policy: the agent has to drive itself
around the room within the 10-minute budget so that downstream
modules (object detection, scene memory — both Phase 2) actually have
something to look at.

This task ships the geometry-only first half of that exploration loop:
a persistent global occupancy grid built from `terrain_ext`, plus a
reactive frontier-based planner that picks the next waypoint each
tick.

## What this PR adds

Three new modules + a small dispatcher patch:

| File | Purpose |
|---|---|
| `src/xiao_hei_vln/perception/global_map.py` | Persistent 2D occupancy grid stitched from `terrain_ext` + pose. Cells are `UNKNOWN`/`FREE`/`OCCUPIED`. Origin anchors on the robot's first observed pose. |
| `src/xiao_hei_vln/exploration/frontier.py` | `FrontierPlanner` that extracts FREE-adjacent-to-UNKNOWN clusters from the grid and scores them per-tick: `w_size·log1p(size) − w_dist·distance − w_loop·anti_loop_penalty`. |
| `src/xiao_hei_vln/perception_responder/responder.py` | `PerceptionResponder` matching the `DummyResponder` API (`respond/is_done/reset/close`). Each tick: update grid → select frontier → emit `WaypointPathResponse`. |
| `src/xiao_hei_vln/app/main.py` | One new dispatcher branch: `XIAO_HEI_RESPONDER=perception`. |

`scipy` is intentionally **not** added as a dependency — `binary_dilation`
and connected-components labelling are reimplemented in ~25 lines of
numpy in `perception/global_map.py`.

## Algorithm summary

Every 500 ms tick:

1. **Stitch** the latest 20 m `terrain_ext` snapshot into the global
   grid using the provided pose. Cells become `FREE` if cost ≤ threshold
   else `OCCUPIED`; previously-observed cells are kept unless
   overwritten. The robot's own cell is always `FREE`.
2. **Find frontier clusters** — connected components of FREE cells that
   border at least one UNKNOWN 4-neighbour.
3. **Score each reachable cluster** with a weighted sum of cluster size
   (positive), straight-line distance (negative), and a soft
   anti-revisit penalty derived from a 5-entry history ring buffer.
4. **Emit a waypoint** toward the highest-scored cluster's representative
   cell. If no frontier remains, emit a "stand still" waypoint at the
   current pose (cue for Phase B / TSP coverage in a future PR).

State (`GlobalMap` + planner history) is persistent across ticks and
wiped via `responder.reset()` when the question text changes (already
handled by the existing main-loop reset hook).

## Tests

25 new unit tests, all numpy-only — no ROS, no fixtures from disk:

- `tests/test_global_map.py` — 12 tests covering empty state, point
  stitching, origin anchoring on a far-away pose, out-of-bounds drop,
  frontier extraction on hand-built grids (open-top boundary, fully
  enclosed → no frontier), `reset()`, and the straight-line
  reachability check.
- `tests/test_frontier_planner.py` — 7 tests covering empty input,
  size-wins-when-symmetric selection, anti-loop history dampening,
  distance and size term sign/monotonicity, `reset()`, and wall-blocked
  reachability filtering.
- `tests/test_perception_responder.py` — 6 tests covering cold-start
  no-output guards, waypoint emission when frontiers exist, fall-back
  to stand-still when frontiers are exhausted, and state reset.

Whole suite: 218 passed (193 existing + 25 new), 5 skipped (pillow not
installed in venv).

## How to verify locally

```sh
.venv/bin/python -m pytest tests/test_global_map.py tests/test_frontier_planner.py tests/test_perception_responder.py -v
```

End-to-end smoke (no GPU needed):

```sh
XIAO_HEI_RESPONDER=perception python -m xiao_hei_vln.app.main
```

The responder will sit quietly until the cache has both `terrain_ext`
and `pose`, then begin emitting `Pose2D` waypoints once per tick.

## Hard prerequisite still to confirm

`terrain_ext` cost-field semantics — the current `obstacle_cost_threshold=0.5`
in `GlobalMap.__init__` is a placeholder. Before live-simulator deploys,
run a one-time `scripts/inspect_terrain.py` (not in this PR) that
prints `np.unique`/`np.histogram` of the actual cost values and pick
the right threshold. The classification of `FREE` vs `OCCUPIED` rides
on this constant.

## Out of scope

These are deliberately left for follow-up PRs so this one stays small
enough to review:

- **Object detection** (`perception/detector.py`) — YOLO-World over the
  1920×640 panorama.
- **3D localisation** (`perception/localizer.py`) — LiDAR backprojection
  of detected boxes; needs camera↔LiDAR extrinsics first.
- **Scene memory** (`perception/scene_memory.py`) — cross-tick object
  merging that will let the responder answer numerical and
  object-reference questions.
- **TSP coverage sweep** (Phase B) — for "behind furniture" zones where
  frontier exploration alone leaves blind spots.
- **VLM-scored frontier ranking** — layering Qwen scoring on top of
  the geometric scorer.

The architecture leaves clean attach points for each of these on the
same `PerceptionResponder`.
