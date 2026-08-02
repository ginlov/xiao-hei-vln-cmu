# TASK 21 — Reject mask spill in the 3D lift

Detections whose segmentation mask spills onto a neighbouring surface were
lifting onto that surface instead of the object, producing scene-graph nodes
hundreds of times larger than ground truth. This task diagnoses the failure,
compares five rejection strategies on frozen captures, and adopts voxel-grid
connected components in `PointLifter`.

## The failure

`arabic_room` / `vp_000`, detection #5, `wall lamp`, score 0.711:

| | value |
|---|---|
| inliers | 52 — bimodal in range |
| low mode | 22 pts at 2.30–2.51 m — the fixture |
| high mode | 30 pts at 2.97–4.37 m — ceiling, a 1.46 × 1.04 m sheet, **0.00 m** thick |
| lifted position | z = 2.761, exactly the ceiling plane |
| resulting node | 1.61 × 1.29 × 0.80 m vs GT 0.19 × 0.10 × 0.45 m (~190×) |

Three failures compound:

1. **The mask overspills** onto ceiling adjacent to the fixture.
2. **The median is a majority vote.** 30 of 52 points are ceiling, so it returns
   the ceiling. A median is robust to ≤50% contamination *by construction*.
3. **The AABB spans everything** — `ObjectMap._recompute` uses raw min/max, so
   even a correct centre would still produce a box enclosing both modes.

The z-buffer gate cannot help: it rejects returns occluded *along a bearing*,
and ceiling adjacent to a ceiling-mounted lamp is not occluded — at those pixels
it is the nearest return. Measured across all 15 scenes the gate fires on 1.4%
of 4,751 detections and removes 0.05% of points. It is close to inert.

Nor does the fusion layer recover: merges union the point clouds and recompute
the AABB, so **a box can only grow**. This node went 1.67 m³ → 2.49 m³ and then
sat unchanged for seven more viewpoints.

## The measurement was wrong before the fix was

The first round ranked strategies on `mAP@1`, which is **distance**-based — it
asks only whether a node's centre lands within 1 m of a GT centre, and is nearly
blind to extent. On that metric the cheapest strategy (a 1-D split on range
gaps) ranked first while still leaving 337 nodes above 1 m³.

`perception_benchmark/box_quality.py` was added to score extent directly: tail
(`over_1m3`, `max_volume`), calibration (`inflation` = predicted ÷ GT volume),
and `pathology_rate` (share above 10× GT).

**Acceptance rule: the tail must improve, inflation must not collapse, and
recall must not regress against what ships today.** Optimising the tail alone
selects for filters that fix box size by eating the object — see the note on
reading `inflation` below, which is why it is a relative signal rather than a
target of 1.0.

## Strategies compared

All five run inside `PointLifter` on identical frozen captures, with the
selection rule held constant (nearest cluster with ≥ `min_inliers` points, else
the largest) so the comparison measures clustering rather than tie-breaking.
14 scenes — `hotel_room_1` has no captures.

| strategy | >1 m³ | max m³ | inflation | >10× GT | mAP@1 |
|---|---|---|---|---|---|
| baseline | 800 | 30808 | 11.97 | 52.9% | 0.2111 |
| range gap 0.3 m | 337 | 493.8 | 1.36 | 18.3% | **0.2324** |
| single-link | 178 | 2534.6 | 0.69 | 10.5% | 0.2243 |
| DBSCAN (`min_samples=4`) | 187 | 2534.6 | 0.69 | 11.2% | 0.2267 |
| **voxel CC** | 109 | **14.4** | 0.42 | 7.2% | 0.2134 |
| plane (RANSAC) + voxel | 54 | 11.9 | 0.28 | 4.7% | 0.2100 |

Baseline's worst node is 30,808 m³ — larger than the building — and 53% of its
nodes exceed 10× their label's GT volume.

## Decision: voxel connected components

- A **range-only split** cannot see a plane viewed at a grazing angle, whose
  range recedes continuously with no gap to cut on. Verified on
  `arabic_room`/`vp_001` det #6: 92 inliers spanning 2.56–4.89 m with a largest
  internal gap of 0.203 m, so a 0.3 m threshold never fires and the 2.33 m³ box
  survives. All four spatial methods fix that same case to 0.0098 m³ against a
  GT of 0.0086 m³.
- **Single-link and DBSCAN** chain through a thin bridge of points. Both produce
  a 2534 m³ worst-case node — *worse* than range-gap's 494 m³ — despite better
  medians. `min_samples=4` was not enough when the bridge is genuinely dense.
- **Voxel adjacency cannot chain** the same way, since a bridge must occupy
  contiguous cubes. It is also O(n) with no pairwise distance matrix.
- **RANSAC plane removal** scored best on extent but was rejected: table tops,
  sofa seats and pictures are planes too, and recall regressed on 13 of 14
  scenes. It wins by deleting evidence.

The `mAP@1` gaps between range, single-link and DBSCAN are inside the noise
(paired: −0.0057 ± 0.0149, 4W/8L), so extent and failure-mode understandability
decide it rather than the headline metric.

### Voxel-size calibration (14 scenes)

| voxel | >1 m³ | max m³ | inflation | mAP@1 | R@1 | mAP-IoU |
|---|---|---|---|---|---|---|
| 0.06 m | 35 | 10.5 | 0.18 | 0.1908 | 0.2299 | 0.0466 |
| 0.08 m | 79 | 10.5 | 0.33 | 0.2062 | 0.2483 | 0.0565 |
| **0.10 m** | 109 | **14.4** | 0.42 | 0.2134 | **0.2552** | 0.0582 |
| 0.12 m | 141 | 39.2 | 0.55 | 0.2142 | 0.2541 | 0.0584 |

Finer voxels fragment the object itself: 0.06 m costs 4 points of recall against
the range-gap arm and drops `mAP@1` **below baseline**. Coarser ones start
re-merging the contaminant — 0.12 m nearly triples the worst-case node
(14.4 → 39.2 m³) for no gain in localisation.

**0.10 m adopted.** Against 0.12 m it is level on `mAP@1` (0.2134 vs 0.2142) and
recall (−0.0163 vs −0.0174 against range-gap), with a much better tail. Against
baseline it improves recall (0.2552 vs 0.2458), so the acceptance rule is met:
tail improves, recall does not regress against what ships today.

**On reading `inflation`:** the 1.0 target in the acceptance rule is too strict
as an absolute. LiDAR samples only the surfaces facing the robot, from a limited
set of viewpoints — the back of a sofa is never observed — so a correct lift
covers less than the full GT box and lands below 1.0 by construction. Treat it
as a *relative* signal between arms (0.18 is clearly over-cut next to 0.42), not
as a target to hit. The reference `wall lamp`, lifted correctly, scores 1.14.

## Changes

| file | change |
|---|---|
| `src/xiao_hei_vln/perception/lifter.py` | `cluster_voxel_m` + `_voxel_components`; `range_gap_m` kept as an alternative; `inlier_filter` hook for offline A/B |
| `perception_benchmark/clustering.py` | new — the five candidate strategies (benchmark only) |
| `perception_benchmark/box_quality.py` | new — extent metrics: tail, inflation, pathology rate |
| `perception_benchmark/replay_score.py` | `--cluster-voxel`, `--cluster`, `--range-gap` |
| `perception_benchmark/dump_debug.py` | `--range-gap`, `--out`, and per-viewpoint `nodes_vp` |
| `perception_benchmark/viz_app.py` | per-viewpoint vs cumulative toggle (was a no-op); `PERCEPTION_DEBUG_DIR` override |
| `perception_benchmark/viewgen.py` | `DATA_ROOT` reads `VLA3D_UNITY_DIR` (was a hardcoded `/home/ubuntu` path) |
| `docs/perception-sidecar.md` | lift algorithm updated with the z-buffer and clustering steps |
| `docs/tasks/backlog.md` | new — B1 (this, resolved) and B2 |

## Known residual

Counting MAE is slightly worse than baseline in every arm except the rejected
plane strategy, and remains unexplained.

**B2** (percentile AABB in `ObjectMap`) is the natural follow-up: box extent is
owned by the fusion layer, so trimming it there attacks the symptom without
touching which points survive — a much smaller blast radius than any point
filter.
