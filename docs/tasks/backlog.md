# Backlog

Known gaps that are understood and reproducible but not yet fixed. Each entry
records the evidence, so picking one up does not mean re-deriving the diagnosis.

Items here are scoped to the offline
perception benchmark (`perception_benchmark/README.md`), which replays
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

## B4 — Reject objects beyond a range cap

**Status:** proposed — not yet measured
**Files:** `src/xiao_hei_vln/perception/lifter.py` (a hook already exists),
`src/xiao_hei_vln/perception/responder.py`

Detections far from the robot are lifted from very few LiDAR returns, because
point density on a surface falls ~`1/r²`. Beyond ~4.5 m a whole object is
carried by a handful of points, so its position is noisy and it is prone to the
mask-spill failure of B1 (the few real points are easily outnumbered by spill
onto a far wall). Rejecting objects past a range cap — default **~4.5 m**,
adjustable — should trade a little recall on genuinely distant objects for
cleaner, better-localised nodes.

**Evidence** (`debug_k2` / `arabic_room` / `vp_017`): `potted plant` node #25,
robot at `(-2.74, 1.07, 0.76)`, object at `(2.31, 4.04, 0.69)` — **5.86 m**
away, lifted from only **19 inliers** (min_inliers is 10, so it barely
cleared), `n_obs = 1`. A thin, one-shot, far detection is exactly the profile
this would drop. For context the offline tooling already treats **8 m** as the
edge of usable coverage (`verify_projection --max-range`, the overlay's
GT-observability filter).

**Two ways to do it (decide when picked up):**

1. **Pre-filter the cloud** — drop scan returns farther than the cap from the
   robot *before* lifting. A hook already exists: `PointLifter.max_depth_m`
   (currently `None`, unused) caps camera-frame return distance for exactly
   this reason ("a sparse return through a doorway snaps onto a far wall and
   biases the median"). Wiring it to ~4.5 m and exposing it (env +
   `replay_score`/`dump_debug` flags) is most of the work. Bonus: it also
   starves far mask-spill of points, so it complements B1. Risk: it uses the
   return's own range, so a near object seen past a far surface is unaffected —
   which is correct.
2. **Post-lift gate** — lift as now, then drop nodes whose lifted position is
   farther than the cap from the robot pose that saw them. Simpler and purely
   additive, but the lift already ran on the contaminated cloud (the median may
   already be wrong), and it spends compute on detections it then discards.

Prefer (1): it removes the bad points rather than the symptom, and reuses an
existing, documented mechanism. Whichever is chosen, **measure the recall cost
first** — count GT objects legitimately beyond 4.5 m per scene before setting
the default, since those become guaranteed misses. Measurable on the frozen
captures with `replay_score.py` (recall / cErr / counting MAE) and
`box_quality.py` (tail volume).

---

## B5 — SAM mask quality

**Status:** proposed — not yet measured
**Files:** `perception/pipeline.py` (sidecar)

The masks are not always clean, and a bad mask feeds the lift directly: a mask
that under-covers starves the inlier count, one that over-spills pulls in a
neighbouring surface (the B1 failure mode). The sidecar currently runs the
**smallest** SAM 2.1 checkpoint — `sam2.1_hiera_tiny.pt`
(`SAM_WEIGHTS`/`SAM_CONFIG` at `pipeline.py:251`).

**Two solutions on the table:**

1. **Upgrade the SAM model** — swap the tiny Hiera checkpoint for
   small / base-plus / large. Just a weights + config change, but heavier per
   call (SAM already runs once per YOLO box per face); measure the latency hit
   against the tick budget, and the mask-quality gain, before committing.
2. **Gate on SAM's confidence** — SAM2's `predict()` already returns a
   mask-quality score (predicted IoU) that we currently discard
   (`masks, _, _ = self._sam.predict(...)`, `pipeline.py:383`). Capture it,
   thread it through `_FaceDetection` → `Detection`, and drop or demote
   low-confidence masks. Nearly free, and it gives a per-mask signal the
   pipeline has never had — complementary to the YOLO box score, which says
   nothing about mask fit.

The two are independent and could combine (better model *and* a confidence
gate). Prefer starting with (2): it is cheap, measurable, and tells us how much
of the problem is low-confidence masks in the first place — which also informs
whether (1) is worth its cost. Both are measurable on the frozen captures via
`box_quality.py` (extent tail) and `replay_score.py` (recall / precision),
though note detection masks would change, so the frozen `detections.npz` must
be re-dumped for a fair A/B.

---

## B6 — Large flat objects fragment into many nodes

**Status:** addressed on `arabic_room` (TASK 32, carpet 9→2) — multi-scene sweep
of the new gate still pending
**Files:** `src/xiao_hei_vln/perception/object_map.py`

Large planar objects (carpet, floor, ceiling) shatter into many nodes instead
of one, so no single node's box matches the object's true AABB.

**Evidence** (arabic_room, Task-1 object-reference, 133 questions): carpet is
the reference **target in 26/133 questions**, but the **2 real carpets are
lifted as 25 separate nodes** (floor → 30, ceiling → 39). For GT carpet #32
(center `1.94, -0.58`, a ~2 m slab) the nearest carpet fragment centroid is
0.61 m away and its box is tiny vs. the GT AABB → IoU ≈ 0. This alone drives
~20% of Task-1 misses, and is the same over-segmentation that pushes benchmark
precision down (arabic_room pred 188 vs GT 81).

**Approach:** merge same-label large planar fragments into one node before
export (voxel-adjacent or plane-fit union), and add the horizontal (ceiling)
analogue of `_is_wall_sheet`. Fixes the Task-1 carpet miss *and* the benchmark
precision drop at once. Relates to B2 (box shape) and B3 (dedup).

**Fixed (TASK 32):** the `add()` gate is now `IoU ≥ 0.3 OR surface_gap ≤
0.15·max_footprint`, with `IoU` falling back to xy-footprint (BEV) IoU when both
boxes are flat. On arabic_room this cut carpet nodes 9 → 2 (one per GT carpet)
and total nodes 46 → 30, with no distinct-instance collapse (pillows stayed 2,
potted plants 5/5). Remaining work: sweep `MERGE_GAP_FRAC`/`FLAT_Z_M` across the
14-scene corpus (dense-scene recall is the historical risk — see the constant's
comment) and add the horizontal (ceiling) analogue of `_is_wall_sheet`.

**Why the old merge gates missed it** (measured, arabic_room, TASK 31).
`add()` merges a same-label detection into a node when `iou_3d >= MERGE_IOU
(0.3)` **or** centre `dist <= MERGE_DIST (0.4 m)`. For flat carpet fragments
*both* structurally fail: (1) the boxes have ~0 z-thickness, so volumetric
`iou_3d` collapses to **0.000** even when the footprints overlap; (2) partial
views produce offset centroids, so centres sit >0.4 m apart. Concrete case:
the fragment created at `vp_015` (centre `[-2.9, -0.55]`, box `0.58×1.26`)
against node #0 gave **IoU 0.000, dist 0.565 m, surface-gap 0.000 m** — the
boxes *touch* but neither gate fires. `finalize()`'s suppression path
(`nms_dist=0.4 m` AND `gap<=0.05 m`) also misses because 0.565 > 0.4. So the fix
must key on **footprint overlap / surface-gap for flat labels**, not volumetric
IoU, and use a distance gate larger than 0.4 m (or grow it with object size).

---

## B7 — Small objects never reach the scene graph

**Status:** diagnosed (end-to-end `arabic_room` eval) — not started
**Files:** `src/xiao_hei_vln/perception/lifter.py` (`min_inliers` gate),
`perception/pipeline.py`

Seven **queried** object classes are never lifted into the scene graph in
`arabic_room`: **hookah, hookah wire, coffee pot, glass, tray, focus light,
window**. Any question using them as target or anchor auto-fails.

**Evidence:** these types are in the query vocab but absent from the 188-node
graph; they account for ~30% of Task-1 misses (target/anchor not present).
E.g. *"Find the vase closest to the hookah"* — no hookah node exists, so Gemini
grounded on "Arabic jar … is a type of hookah" and picked the wrong vase.

**Cause:** small/thin items score low for YOLO-World *and* fall below the
lift's `min_inliers = 10` LiDAR gate (return density ∝ 1/r²), so even when
detected they never commit a position.

**Approach:** relax or skip `min_inliers` for known-small classes (lift from
fewer points), and/or a per-class score floor. Measurable on the frozen
captures via `replay_score.py` recall.

---

## B8 — Open-vocab label instability vs. the challenge vocabulary

**Status:** diagnosed (end-to-end `arabic_room` eval) — not started
**Files:** `src/xiao_hei_vln/perception/object_map.py` /
`src/xiao_hei_vln/perception/vocab.py` / responder grounding

The right object is lifted to roughly the right place but under a **neighbouring
label**, so grounding on the question's exact noun fails.

**Evidence** (arabic_room, ~15% of Task-1 misses): recurring swaps
`vase ↔ Arabic jar`, `glass ↔ potted plant`, `window ↔ picture`,
`sofa ↔ pillow`. Note Gemini itself grounds correctly ~91% of the time *given*
the graph — the score is gated by perception label fidelity, not reasoning.

**Approach:** label normalization / alias sets applied at grounding time (or
fed to Gemini as synonym groups), and/or resolve confusable pairs in the
cross-label suppression of B3. Partly responder-side, so unlike B1–B7 it is not
purely a perception-benchmark item.

---

## B10 — Viewpoint redundancy degrades the box (dwell bias)

**Status:** both halves shipped (TASK 33); multi-scene sweep of the two knobs
pending, centre-drift only partly addressed
**Files:** `src/xiao_hei_vln/perception/object_map.py`,
`src/xiao_hei_vln/perception/responder.py`, `src/xiao_hei_vln/app/main.py`

When the robot dwells at one pose it produces many near-identical partial views;
each carries its point-count weight, so a single vantage can own the majority of
a node's box weight (measured: 228/240 obs = 88% of node #13's weight from one
0.5 m cell), collapsing the box to that thin slice and drifting its centre.

**Shipped (both levers):**
- *Estimator* — flat nodes size from the **max** per-view extent, not the mean,
  recovering the reach of the best view (arabic_room carpet GT-coverage 20→56%,
  64→87%; volumetric untouched).
- *Capture-time novelty gate* — `responder._inject_visible`: the accumulator
  runs every tick while detect/lift/fuse fires only when the pose is farther
  than `novel_viewpoint_m` (env `XIAO_HEI_NOVEL_VIEWPOINT_M`) from ALL kept
  `(x,y)` (360° camera, so no heading term). Reproducible offline via
  `replay_score.py --novel-viewpoint-m`. **Default OFF (0.0):** measured on
  arabic_room, a 0.3 m gate cuts compute ~15× and raises box IoU@0.25
  (0.051→0.073) but drops centre-distance mAP@0.5 (0.195→0.173) — it costs
  detection recall, so it stays opt-in until a corpus sweep justifies it.

**Next:** get the gate's box-quality/compute gain WITHOUT the recall cost —
gate the *fusion weight* per viewpoint (perceive every frame, down-weight
redundant poses in `_recompute`) rather than skipping the observation entirely.
Then sweep `novel_viewpoint_m` and `max` vs a high percentile together against
the real metric. See TASK 33.

---

## B9 — Node box centre and extent use inconsistent estimators (points fall outside the box)

**Status:** diagnosed (arabic_room node #0, TASK 31) — deferred, revisit later
**Files:** `src/xiao_hei_vln/perception/object_map.py`
(`_Node._observe` / `_recompute`)

A per-observation box takes its **size** from the raw AABB span of the core
points (`hi - lo`) but its **position** from `robust_center` (a median-gated
*mean*), not from the midpoint `(min+max)/2`. When the cloud is asymmetric the
mean ≠ midpoint, so a size-correct box is re-centred off the extremes and the
sparse tail on the far side falls **outside** the box. Observed directly on
carpet node #0 at `vp_000`: box dims `1.38×2.02` correct, but the box is shifted
so boundary points spill out (IoU 0.834 vs 0.896 for a raw extreme-centred box
of the same size — the whole gap is position, not size).

This is a deliberate trade, not a plain bug: `robust_center` is chosen because
the **primary challenge metric is centre-distance**, and the mean tracks the
true centre far better than `(min+max)/2`, which is set by the two most fragile
points and moves half-way toward any single outlier. Extent only affects
secondary IoU. The open question is whether a **containment-consistent** box
(e.g. centre on the robust centroid but expand the extent symmetrically to
enclose the core points, or clip the size to what the robust centre can
contain) improves IoU without regressing centre-distance. Relates to B2 (box
shape) and the flat-object sizing finding in TASK 31.

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
