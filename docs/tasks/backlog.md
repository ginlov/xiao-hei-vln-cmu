# Backlog

Known gaps that are understood and reproducible but not yet fixed. Each entry
records the evidence, so picking one up does not mean re-deriving the diagnosis.

Items here are scoped to the offline
[perception benchmark](../../perception_benchmark/README.md), which replays
frozen captures — every one can be implemented and measured without the
simulator.

---

## B1 — Reject mask spill in the 3D lift  *(resolved: voxel clustering)*

**Status:** adopted — voxel connected components in `PointLifter`
**Files:** `src/xiao_hei_vln/perception/lifter.py`

The z-buffer gate rejects returns occluded *along a bearing*. It cannot reject a
co-visible neighbouring surface the detection mask spilled onto — for those
pixels the neighbour is itself the nearest return. When the spill outnumbers the
object, the median tracks the contaminant and the lifted box blows up.

**Evidence** (`arabic_room` / `vp_000` / det #5, `wall lamp`, score 0.711):

- 52 inliers, bimodal in range: 22 points at 2.30–2.51 m (the fixture), 30 at
  2.97–4.37 m (ceiling, a 1.46 × 1.04 m sheet with **0.00 m** thickness).
- 58% of inliers within 5 cm of the ceiling plane, so the median returns
  z = 2.761 — exactly the ceiling.
- Resulting node: 1.61 × 1.29 × 0.80 m against a GT `wall lamp` of
  0.19 × 0.10 × 0.45 m, ~190× oversized.
- The z-buffer removed **0 of 52** points here. Across all 15 scenes it fires on
  1.4% of 4,751 detections and removes 0.05% of points — effectively inert.

### The measurement was wrong before the fix was

The first round was ranked on `mAP@1`, which is **distance**-based: it asks only
whether a node's centre is within 1 m of a GT centre. A 2 m³ contaminated box
and a correct 0.01 m³ box score identically when their centres are equally
close. Ranked that way, the cheapest strategy (a 1-D range split) looked best
while still leaving 337 nodes over 1 m³.

`perception_benchmark/box_quality.py` scores extent instead — tail
(`over_1m3`, `max_volume`), calibration (`inflation` = pred ÷ GT volume), and
`pathology_rate` (share above 10× GT). **Accept a change only if the tail
improves AND inflation does not collapse AND recall does not regress against
what ships today**; optimising the tail alone selects for filters that eat the
object. See the note below on why `inflation` is a relative signal, not a 1.0
target.

### Strategy comparison (14 scenes with captures)

| strategy | >1 m³ | max m³ | inflation | >10× GT | mAP@1 |
|---|---|---|---|---|---|
| baseline | 800 | 30808 | 11.97 | 52.9% | 0.2111 |
| range gap 0.3 m | 337 | 493.8 | 1.36 | 18.3% | **0.2324** |
| single-link | 178 | 2534.6 | 0.69 | 10.5% | 0.2243 |
| DBSCAN | 187 | 2534.6 | 0.69 | 11.2% | 0.2267 |
| **voxel CC** | 109 | **14.4** | 0.42 | 7.2% | 0.2134 |
| plane + voxel | 54 | 11.9 | 0.28 | 4.7% | 0.2100 |

Why **voxel connected components** was chosen despite not topping `mAP@1`:

- A **range-only split** is blind to a plane at a grazing angle, whose range
  recedes continuously with no gap to cut on. It left the reference case at
  2.33 m³ even after the fix.
- **Single-link / DBSCAN** chain through a thin bridge of points — both produced
  a 2534 m³ worst-case node, *worse* than range-gap's 494 m³.
- **Voxel adjacency cannot chain** that way (a bridge must occupy contiguous
  cubes) and is O(n) with no pairwise distance matrix.
- **RANSAC plane removal** scored best on extent but was rejected: real targets
  (table tops, sofa seats, pictures) are planes too, and recall regressed on
  13 of 14 scenes. It wins by deleting evidence.

The `mAP@1` differences between range, single-link and DBSCAN are inside the
noise (paired: −0.0057 ± 0.0149, 4W/8L), so extent and failure-mode
understandability decide it, not the headline metric.

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

## B2 — Outlier-trimmed bounding boxes in ObjectMap

**Status:** not started
**Files:** `src/xiao_hei_vln/perception/object_map.py`

`_Node._recompute` sets the box from raw `pts.min(0)` / `pts.max(0)`, so a
single stray point sets a corner. `robust_center` already exists for the
centre — there is no equivalent for the extent.

Compounding this, merges **union** point clouds and recompute the AABB, so a
box can only ever grow. One contaminated first observation is permanent: the
`arabic_room` node above went 1.67 m³ at `vp_000` → 2.49 m³ at `vp_001`, then
stayed there for the remaining seven viewpoints. A bloated box also trivially
clears `MERGE_IOU = 0.3`, making it a magnet for nearby same-label detections.

**Approach:** replace the raw AABB with a percentile box (e.g. 2–98%), and
consider re-estimating extent from the best-supported points on merge rather
than only ever growing.

**Interaction with B1:** B1 removes much of the contamination upstream, so
measure B2 on top of B1 rather than in isolation — the headroom will be
smaller than the standalone numbers suggest.

---

## B3 — Cross-label duplicate suppression

**Status:** same-label half shipped; cross-label half deferred
**Files:** `src/xiao_hei_vln/perception/object_map.py`

### What shipped

`finalize` now suppresses the weaker of two co-located nodes on either
`IoU >= NMS_IOU` (any label, unchanged) **or** `centre <= NMS_DIST = 0.4 m` and
`box_gap <= NMS_GAP = 0.05 m`, restricted to **identical labels**.

`NMS_DIST` matches `MERGE_DIST` deliberately. `add` already merges same-label
nodes within that radius, but a node's centre **moves as it accumulates
points**, so two nodes created further apart can drift inside the radius with
nothing re-checking. This is that check, deferred until the centres settle —
which also bounds how much it can recover.

Measured over 14 scenes: redundant nodes 216 → 204, counting MAE
2.836 → 2.813 (6 scenes better, 1 worse, 7 unchanged), recall −0.0028,
precision +0.0014. Dropping to 0.3 m removes the effect entirely; the gap term
is not binding at 0.4 m (0.05 and 0.15 give identical results) and is kept only
as a guard against a large box whose centre coincides with a small one.

### Why IoU alone could not do this

Over 14 scenes, node pairs whose centres are within 0.5 m have median IoU
0.055, and **none of 691 reach `NMS_IOU = 0.5`** — the cross-label rule has
never fired. Those pairs are not far apart: their box surfaces sit a median
**3 cm** apart (53% under 5 cm, 82% under 15 cm), and the boxes are thin slabs
— median extents 0.55 × 0.32 × 0.09 m, aspect 7.6:1.

That is not a lifter accuracy problem. A LiDAR sweep sees one *face* of an
object, so two viewpoints yield adjacent, disjoint surface patches. Two boxes
of ~0.01 m³ with centres 0.43 m apart **cannot** overlap, and IoU is a step
function with no gradient below overlap: 3 cm apart and 3 m apart both score
exactly 0. IoU is the wrong similarity measure at this box scale, which is also
why `MERGE_IOU` contributes on only 12% of pairs while `MERGE_DIST` carries
the rest.

Note this got worse when the lifter was fixed (B1): in `arabic_room` the median
IoU between co-located nodes fell 0.158 → 0.010 once boxes tightened. The
thresholds were implicitly calibrated against bloated boxes.

### What is deferred — the cross-label case

It is the larger half. Of duplicated GT objects across 14 scenes:

| cause | count |
|---|---|
| same label — threshold failure (shipped fix targets this) | 26 redundant nodes |
| mixed labels, one **is** the true label | 80 objects |
| mixed labels, **none** is the true label | 42 objects |

**82% involve label disagreement**, and that is a hard gate, not a threshold:
`add` skips any candidate whose label differs (`if nd.label != label: continue`)
before evaluating distance or IoU, so no tuning can merge them.

The confusions split in two, and conflating them is the trap:

- **Synonym / granularity collisions** — `cabinet | shelf`, `chair | couch`,
  `coffee table | table`. Merging would be right.
- **Plain misclassification** — `chair | map wall decal`,
  `computer monitor | map wall decal`, `lamp | picture`. One label is simply
  wrong, and suppressing it hides a detector error rather than fixing a
  duplicate.

Worth knowing before designing the fix: the class list pushed to YOLO-World is
**the scene's own ground-truth labels**, so the detector is handed a closed,
correct vocabulary and still labels one object differently across viewpoints.
This is frame-to-frame instability, not a bad word list — and exact-string
equality in the merge rule converts it directly into duplicate nodes.

**Options when picked up:**

1. Extend the gap/distance rule across labels, recording the loser on the
   survivor. `absorbed_labels` is already plumbed through `to_list()` for
   exactly this, so instability stays visible instead of being swallowed.
2. Prune near-synonyms from the class list before pushing it — if `chair` and
   `couch` are never both offered, the detector cannot split an object between
   them. Trades GT-matching fidelity for graph stability.

Either is measurable on the frozen captures with
`perception_benchmark/box_quality.py` (`dupObj` / `redund`) plus counting MAE.

---

## Related gaps, not yet scheduled

Recorded from the same investigation; no work planned yet.

- **No dispersion check on the inlier set.** Nothing notices that a 52-point
  cloud spans 1.6 m; a bimodal or implausibly large inlier set should demote to
  `lifted=False` rather than silently emit a wrong position.
- **No per-label size prior.** Nothing compares lifted extent against a
  plausible scale for the label. Using GT sizes at inference would be cheating
  for the challenge; a generic or corpus-learned prior would not.
- **`_is_wall_sheet` is too narrowly scoped.** It encodes the right instinct —
  "this cloud's shape is not this label's shape" — but only for vertical sheets
  and nine flat labels. The horizontal (ceiling) analogue does not exist.
- **Mask quality near the equirect top,** where vertical distortion is worst.
  Four of seven `wall lamp` detections at `arabic_room` / `vp_000` failed
  outright with 0–7 inliers.
