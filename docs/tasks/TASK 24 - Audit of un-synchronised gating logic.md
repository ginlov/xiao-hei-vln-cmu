# TASK 24 — Audit of un-synchronised gating logic

## Why

The stack answers the question *"is this new?"* in eight different places, with
eight independently-chosen thresholds, and nothing reconciles them. Six of those
are pose-based, so "the robot is somewhere new" is currently defined as 0.25 m,
0.3 m, 1.5 m, 2.0 m, 3.0 m, and a 0.92-valued nav signal — depending on which
layer is asking.

This is an audit, not a change. Nothing here is fixed yet; each item records
what the logic is, what breaks because of it, and what the evidence says. The
point is that picking any of these up later does not mean re-deriving the
diagnosis.

**Explicitly out of scope:** the tick itself (0.5 s, `XIAO_HEI_VLM_TICK_HZ`) and
sensor pairing (`LatestCache.snapshot`). Both live inside `tick()` and are
deliberately left unchanged. See "Deferred" at the end for why, and for the one
measurement worth taking there.

## The inventory

| # | layer | trigger | value | source |
|---|---|---|---|---|
| 3 | scan keyframe | ~~motion since last keyframe~~ → **every tick** | window 10 ticks, voxel 0.05 | `perception/scan_accumulator.py` |
| 4 | viewpoint node | distance from *every* existing node | ≥2.0 m | `scene/representation.py:344` |
| 5 | object merge (`add_object`) | same-label proximity | 1.5 m | `scene/representation.py:116` |
| 6 | object merge (`ObjectMap`) | IoU **or** centre dist | IoU 0.3 / 0.4 m; NMS: IoU 0.5, dist 0.4 + gap 0.05 | `perception/object_map.py:37-59` |
| 7 | lift acceptance | inlier count | `min_inliers=10`, cluster voxel 0.10 | `perception/lifter.py:54,78` |
| 8 | waypoint reached (explorer) | distance | 0.3 m | `exploration/_frontier.py:66` |
| 9 | waypoint reached (nav signal) | a different signal entirely | 0.92 | `app/main.py:483` |
| 10 | explorer give-up | time / count | stuck 12 s, max skips 20, max wp dist 1.5 m | `app/main.py:273-274` |

---

## A — #3 vs #4: two motion gates that cannot be related to each other

> **Superseded, not resolved.** #3's motion gate has since been deleted — the
> accumulator now commits a keyframe on *every* tick, so its window is keyed on
> ticks rather than travel. That removes the mismatch described below by
> removing one of the two gates, but it does not unify the layers: #4 still
> keys on distance. It also introduced two costs measured after the change:
>
> - **Coverage is evicted by time.** `max_keyframes=10` at 2 Hz means the
>   buffer turns over after 5 s. Simulated over 5 distinct regions: driving
>   past all five leaves 5 regions in the merged cloud; 10 stationary ticks
>   leave **1**. On the TASK 23 captures — 66% of `arabic_room` frames below
>   0.05 m/s, `chinese_room` never moving — densification now switches itself
>   off exactly when the robot is stalled.
> - **Every tick pays the merge.** `update()` used to return the cache
>   untouched off-keyframe. It now vstacks and voxel-downsamples on every call:
>   **218 ms** per tick on a 20k-point sweep, against a 500 ms budget, on top
>   of detect + lift + the sidecar round-trip.
>
> Keying eviction on travel rather than ticks would keep the simplification
> and recover both properties. The measurements below predate the change and
> describe the gate as it was.

`ScanAccumulator._should_keyframe` stores a sweep when the robot has moved
≥0.25 m **or** turned ≥15° *since the last stored keyframe* — a chained delta.
`SceneRepresentation._maybe_add_viewpoint` adds a node when the pose is ≥2.0 m
from **every** existing node — a global minimum-distance test. Different
reference frames, different predicates, and rotation counts for one and not the
other.

They are therefore not convertible. Measured over the TASK 23 navigation
captures (15 scenes, 5487 ticks):

| scene | ticks | scan keyframes (#3) | viewpoint nodes (#4, 2 m) | #4's rule at 0.25 m |
|---|---|---|---|---|
| arabic_room | 316 | 147 | 3 | 37 |
| chinese_room | 228 | 1 | 1 | 1 |
| home_building_1 | 767 | 313 | 31 | 241 |
| home_building_2 | 1008 | 696 | 11 | 117 |
| hotel_room_1 | 377 | 211 | 8 | 82 |
| hotel_room_2 | 207 | 59 | 2 | 20 |
| japanese_room | 166 | 64 | 5 | 44 |
| livingroom_1 | 297 | 119 | 5 | 50 |
| livingroom_2 | 378 | 187 | 6 | 57 |
| livingroom_3 | 364 | 128 | 7 | 52 |
| livingroom_4 | 227 | 104 | 6 | 43 |
| loft | 280 | 31 | 2 | 14 |
| office_1 | 328 | 181 | 10 | 88 |
| office_2 | 247 | 91 | 5 | 37 |
| studio | 297 | 118 | 5 | 51 |
| **total** | **5487** | **2450** | **107** | **934** |

The last column is the key evidence: run #4's predicate at #3's threshold and
you get 934, not 2450 — a 2.6× gap from the predicate alone, before any
threshold disagreement. So there is no fixed ratio between "sweeps in the
accumulator" and "viewpoints in the graph", and neither can be inferred from the
other.

**Proposed direction:** collapse both into one keypose. A single tracker owns the
0.25 m / 15° chained test and emits a `keypose_id` per tick; `ScanAccumulator`
consumes it instead of re-deriving motion, and `_maybe_add_viewpoint` fires
exactly when a new keypose is emitted. Three things fall out:

- the accumulator window becomes literally "the last 10 viewpoints";
- `observing_viewpoint_ids` becomes real provenance at ~0.25 m granularity;
- one knob (`XIAO_HEI_SCAN_MIN_MOVE_M`) controls both, so they cannot drift.

Cost: 107 → ~2450 viewpoint nodes across 15 scenes (~163/scene, tens of KB of
JSON). **Blocker to handle first:** `RoomNode.best_image` updates on every new
viewpoint node (`representation.py:354-358`). At keypose density that becomes
per-keyframe image churn, so it needs its own explicit "best coverage" rule
before the layers are merged.

## B — #4: the viewpoint edge points at the *newest* node, not the *nearest*

`_current_viewpoint_id()` (`representation.py:324-342`) returns
`self._viewpoints[-1].tick_id`. Both object paths use it — `add_object` at
line 186 and `sync_from_object_map` at line 236.

Because #4's gate only requires 2 m from *some* node, the robot can sit 0.5 m
from vp0 while every detection is attributed to vp7 several metres away.
Concretely: nodes laid at (0,0), (2,0), (4,0), then the robot returns to
(0.5,0). No new node — it is 0.5 m from vp0 — but the edge still records vp2,
3.5 m off. On the TASK 23 trajectories, which are short and heavily
back-and-forth, this is the common case rather than the corner case.

`observing_viewpoint_ids` is therefore not currently trustworthy as geometry.

**Secondary defect:** dedup only compares against the last entry
(`if not vp_ids or vp_ids[-1] != vp_id`, lines 205 and 237). An object seen from
vp0 → vp1 → vp0 ends up with `[vp0, vp1, vp0]`. So `len(observing_viewpoint_ids)`
counts *visits*, not distinct viewpoints; anything reading it as a view count
needs a `set()`.

**Proposed direction:** return the nearest viewpoint, not the last. ~5 lines, no
dependency on A, and it makes the existing edges correct. Do this one first.

## C — #5 vs #6: not two merge rules, one flag with inconsistent defaults

> **Resolved.** `add_object` and `merge_radius` are deleted; `ObjectMap` fusion
> is unconditional and `XIAO_HEI_OBJECT_MAP` is gone from the code, the four
> config files and the docs. The offline constructors in `gemini/batch.py`
> (which had no point clouds to fuse) now build a merge-disabled `ObjectMap`
> from their boxes and sync it in, so there is exactly one path into the object
> layer. The diagnosis below is kept for its rationale.


These are **mutually exclusive implementations of the same step**, not layers.
`_inject_visible` branches per detection (`responder.py:311-317`): when
`self._object_map is not None` the lifted cloud goes to `object_map.add(...)`
and `continue` skips `scene.add_object` entirely; line 331 then replaces the
whole object layer via `sync_from_object_map`. `merge_radius=1.5` and the
IoU-0.3/0.4 m rule never both apply.

`add_object` is the original — same label + close centre, higher confidence
wins, keeping one frame's median and **no 3D box**. `ObjectMap` is the
replacement: it unions the point clouds so the AABB converges, plus NMS and
wall-sheet rejection. TASK 13 added it as opt-in so the pipeline would "behave as
before" by default, and kept `add_object` as the arm of an on-sim A/B that
TASK 13's own notes record as never run.

The duplication is not the problem. The problem is that the flag defaults
differently per entry point:

| entry point | `XIAO_HEI_OBJECT_MAP` |
|---|---|
| `docker/compose.yml:72` | `${...:-}` → **off** |
| `docker/compose_scene_gemini.yml:125` | `${...:-}` → **off** |
| `docker/compose.eval.yml:59` | `${...:-1}` → **on** |
| `scripts/run_scene_vla3d_eval.sh:175` | `${...:-1}` → **on** |
| `perception_benchmark/replay_score.py:88` | hardcoded `ObjectMap(...)` — no opt-out |

So every perception number on record — TASK 20–22, the TASK 23 navigation
scores, the scan-accumulator A/B — measures the **ObjectMap** path, while the
default `compose.yml` runs the **add_object** path. `replay_score.py` cannot
measure the path the default compose runs even if asked.

The consequence is concrete: on the `add_object` path the responder passes
`bbox_min=None, bbox_max=None` (`responder.py:325-326`). Under `compose.yml`'s
default the submission's scene graph ships **objects with no 3D boxes at all**.

**Proposed direction:** flip the default on and delete `add_object` +
`merge_radius`. `ObjectMap` is strictly more capable and is what every
measurement was taken on; a fallback no benchmark can score is exactly how this
divergence arose. Minimum acceptable fix: make the four defaults agree.

**Open question, must be answered first:** which compose file the challenge
submission actually launches. Both candidates currently default to off.

## D — #7: thresholds in object space, not pose space

`min_inliers=10`, `cluster_voxel=0.10`, and #6's IoU/distance values are tuned
against measured data (TASK 21, TASK 22) and are not in the same units as the
pose deltas above.

**No action.** The un-synchronisation here is intentional and should stay that
way. Recorded so it is not swept into a later "unify the thresholds" change.

One caveat carried from TASK 20: the `min_inliers` sweep was never re-run with
the scan accumulator enabled, so 10 is tuned against single-sweep density while
the live stack lifts against an accumulated cloud. That is a measurement gap,
not a synchronisation one — it stays in `backlog.md`.

## E — #8 vs #9: two unreconciled definitions of "reached"

`FrontierExplorer` has `waypoint_reach_dist=0.3` (`_frontier.py:66`), a distance
in metres. `app/main.py:483` has `_WP_REACHED_THRESHOLD = 0.92` applied to a
`Float32` on `/way_point_reached` — a different signal in different units, with
the comment "nav stack settles between 0.25-0.90 m depending on obstacles".

Nothing reconciles them, and both can gate progress.

## F — #10: the give-up constants are hardcoded and are firing everywhere

`stuck_timeout_s=12.0` and `max_consecutive_skips=20` are literals at
`app/main.py:273-274` (overriding the class defaults of 30.0 and 3). Neither is
env-settable, so `TIMEOUT` in the capture sweep cannot influence them.

Every TASK 23 run ended early:

| scene | visited | skipped | reason |
|---|---|---|---|
| home_building_1 | 26 | 37 | max_consecutive_skips |
| livingroom_1 | 9 | 20 | max_consecutive_skips |
| livingroom_3 | 8 | 25 | max_consecutive_skips |
| office_2 | 6 | 15 | no_frontiers |
| hotel_room_1 | 5 | 26 | no_frontiers |
| japanese_room | 5 | 11 | no_frontiers |
| studio | 5 | 20 | no_frontiers |
| loft | 4 | 22 | max_consecutive_skips |
| arabic_room | 3 | 24 | max_consecutive_skips |
| livingroom_2 | 3 | 27 | max_consecutive_skips |
| office_1 | 3 | 21 | no_frontiers |
| livingroom_4 | 2 | 18 | no_frontiers |
| chinese_room | 1 | 20 | max_consecutive_skips |
| hotel_room_2 | 0 | 18 | no_frontiers |
| home_building_2 | — | — | no DONE line — truncated at the 510 s cap |

`skipped` exceeds `visited` in every scene. `chinese_room` recorded a path
length of **0.0 m** over 228 ticks; `hotel_room_2` visited zero waypoints. This
is the single largest confound on every navigation-capture number, and #8/#9 are
a plausible contributor: if the two "reached" tests disagree, the explorer
re-targets a waypoint the nav stack considers reached, burning a skip.

**Proposed direction:** pull the `WP_SKIP` lines from
`exploration_logs/<scene>/exploration.log` and check whether the frontier
targets land in untraversable space or merely time out short. Treat as an
exploration bug, separate from the perception work.

## Suggested order

1. ~~**C** — resolve the `XIAO_HEI_OBJECT_MAP` default.~~ **Done** — fusion is
   now unconditional, so the submission always ships 3D boxes.
2. **B** — nearest-not-newest viewpoint edge. Small, self-contained, fixes a
   real defect in data already collected.
3. **F/E** — diagnose the explorer stall. Blocks meaningful scoring of the
   navigation captures.
4. **A** — the keypose merge. Largest change, and worth doing only once the
   `best_image` question above is settled.

## Deferred: the two tick-internal items

**#1, the tick (0.5 s).** Unchanged by decision. Object fusion runs
unconditionally every tick — `_inject_visible` skips only when the snapshot
lacks an image, pose or scan. That cadence is the reference the TASK 23
recorder reproduces, and changing it would invalidate the captures.

**#2, sensor pairing.** `cache.snapshot()` is the first statement of `tick()`
(`app/main.py:505`), so the perception stack only ever sees one frozen
`VLMInput` — image, pose and scan are guaranteed same-generation for the whole
tick, and the sidecar round-trip cannot skew them apart mid-tick. But
`snapshot()` (`sync/latest_cache.py:79-92`) merely copies whatever is in each
slot under a lock. There is no timestamp comparison, no tolerance window, no
staleness rejection — it is an atomic *read*, not a synchroniser. With
`/camera/image` at ~10 Hz and the scans at ~5 Hz, channel skew of ~200 ms
(≈20 cm of ego-motion at 1 m/s) is unmodelled.

Two facts for whenever this is picked up:

- `rclpy.spin(node)` (`main.py:679`) uses the default **single-threaded**
  executor, so no callback runs while `tick()` executes. Channel ages are fixed
  at snapshot time. The flip side: a tick longer than ~500 ms stops all
  draining, and the subscriber QoS is `depth=5, RELIABLE`
  (`adapters/ros/subscribers.py:69-73`), so that queue will overflow.
- **Every message already carries its stamp** — `_header_from_ros(msg.header)`
  is preserved on `ImageFrame`, `LidarScan`, `TerrainMap` and `OdomPose`. The
  data needed to measure skew exists and is unused.

The non-invasive next step, if wanted: have `snapshot()` also report per-channel
age (`tick_time - header.stamp`) and log it. Measurement first, no behaviour
change, then decide from real numbers whether a tolerance gate is warranted.

## Related

- TASK 13 — introduced `ObjectMap` as opt-in; its unrun A/B is the origin of C.
- TASK 20–22 — every number they report was measured on the ObjectMap path.
- TASK 23 — source of the navigation captures all measurements above draw on.
- `docs/tasks/backlog.md` — B2, B3 and the `min_inliers` sweep remain open and
  are unaffected by this audit.
