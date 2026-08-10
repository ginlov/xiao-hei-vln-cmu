# perception_benchmark

Offline tooling to benchmark the perception stack deterministically, decoupled
from navigation. **Not part of the challenge submission stack.**

- `VIEWPOINT_ALGORITHM.md` — the algorithm spec.
- `viewgen.py` — viewpoint generator (implements the spec).
- `viewpoints/<scene>.json` — generated output (teleport poses + coverage report).

## Two capture datasets

There are two input sets, and they answer different questions. Neither replaces
the other.

| | driver | root | question it answers |
|---|---|---|---|
| viewpoints | `capture_viewpoints.py` drives to `viewgen.py`'s k-cover poses | `captures/` | how good *could* perception be, given near-ideal coverage? |
| navigation | the real `FrontierExplorer`; `record_navigation.py` records passively at the live 2 Hz tick | `captures_nav/` | what does perception actually see on the trajectory the robot takes? |

The layouts are identical, so every offline script reads either one — point
`PERCEPTION_CAP_DIR` at the root you want (default `captures/`):

```bash
perception_benchmark/run_all_scenes.sh                     # viewpoint capture
perception_benchmark/run_nav_capture.sh                    # navigation capture

export PERCEPTION_CAP_DIR=perception_benchmark/captures_nav
export PERCEPTION_DEBUG_DIR=perception_benchmark/debug_nav
uv run --extra perception python perception_benchmark/replay_score.py --all
```

`record_navigation.py` samples at `RATE_HZ`, which defaults to
`XIAO_HEI_VLM_TICK_HZ` (2.0) — the live node ingests every 0.5 s with no motion
gate, no blur gate and no warm-up, so the recorder has none of those knobs
either. Budget ~500 MB per minute of exploration per scene.

See TASK 23 for the full rationale and what to check in the results.

## Run

```bash
uv run python perception_benchmark/viewgen.py --scene arabic_room     # one scene
uv run python perception_benchmark/viewgen.py --all                   # all 15
# knobs: --r-max 6.0  --k 2   --out perception_benchmark/viewpoints
```

Reads VLA-3D GT from `/home/long/Projects/dataset/vla-3d/Unity/<scene>/`
(override with `VLA3D_UNITY_DIR`)
(`object_list.txt`, `<scene>_free_space_pc_result.ply`, `<scene>_pc_result.ply`
+ `<scene>_object_split.npy`). ~20 s for all 15 scenes.

## How it works (short)

360° camera ⇒ a viewpoint is a *position*; "observed" = object within `[R_min,
R_max]` and unoccluded (2D raycast) — pure geometry, no perception in the loop.
1. Occluder grid (0.1 m) from **actual wall-surface points** (see gotcha below).
2. Candidates = free-space PLY points, dropped if < robot-radius from a wall
   **and** if not over the room floor (`over_floor_mask` — the free-space cloud
   leaks into exterior notches/courtyards; the floor-point footprint is the
   reliable inside-the-room test, since `region_id` is a single region per scene).
3. Visibility matrix (range gate + raycast, `stop_short` so wall-mounted objects
   stay visible from the room side).
4. Greedy **k-cover** (k=2, viewpoints ≥30° apart per object) for box convergence.
5. Patch 0-coverage objects with a close-approach viewpoint; else mark
   `not_observable`.

## Latest results (R_max=6, k=2, floor-filtered — all viewpoints inside the room)

| scene | vps | coverage | unobservable |
|---|---|---|---|
| chinese_room | 5 | 100% | 0 |
| loft | 12 | 100% | 0 |
| office_1 | 7 | 100% | 0 |
| home_building_2 | 17 | 99% | 2 |
| arabic_room | 9 | 98% | 2 |
| hotel_room_1 | 15 | 98% | 2 |
| livingroom_3 | 8 | 98% | 2 |
| livingroom_1 | 9 | 97% | 3 |
| hotel_room_2 | 9 | 97% | 3 |
| office_2 | 9 | 96% | 6 |
| japanese_room | 17 | 95% | 3 |
| home_building_1 | 42 | 94% | 27 |
| livingroom_2 | 5 | 93% | 6 |
| livingroom_4 | 5 | 91% | 11 |
| studio | 9 | 88% | 9 |

Mean views/object ≈ 1.9 (target k=2). All 15 scenes verified 0 viewpoints
outside the room floor (`viz_viewpoints.py --all`). Remaining `not_observable`
are a genuine geometric limit — verified by label: architecture coplanar with
walls (`window`, `column`, `wall`), exterior objects (`tree` — now correctly
unobservable from inside, which raised home_building_1 from 15→27), ceiling
`lamp`, and closet contents (`clothes`, `hanger`, `box`) with no interior line
of sight.

Review montage: `viewpoints/_all_scenes_map.png` (0-based viewpoint labels).

## Gotcha fixed during bring-up

**Wall bboxes are NOT solid occluders.** A wall object's axis-aligned bbox
encloses free space (e.g. home_building_1 `wall` id=25 is 21.6×18.8 m — the
whole-building shell). Filling bboxes solid blocked whole rooms (home_building_1
was 8% covered). Fix: rasterize the **actual wall-surface points** from
`pc_result.ply`, sliced per-object via `object_split.npy` byte ranges (reads
only wall points, not the ~40 M-point cloud). → 8% → 96%.

## Known limitations / next steps

- **2D occlusion** ignores height — see-under-tables, over-low-obstacles not
  modeled. Upgrade to a height-aware raycast only where verification shows misses.
- **Architecture in the target set.** `object_list.txt` includes `wall`,
  `window`, `column`, `floor`, `ceiling`, `door frame` — not real perception
  targets. Filtering them would raise the effective denominator; left in for now
  (a benchmarking decision, not silently applied).
- **Empirical verification loop (Stage G) not yet wired** — the coverage above
  is *geometric* (should-be-visible), not confirmed by teleport+capture.
- **Next:** the teleport-and-capture harness — publish each viewpoint to
  `/unity_sim/set_model_state`, wait for fresh frames, record the 7 legal
  inputs, build the scene graph, score vs `object_list.txt`.
