# TASK 21 — Offline perception replay harness & 3D lift repair

## Why

Our detected scene graphs were bad enough to be unusable, and nobody could say
*which stage* was at fault. Scored against VLA-3D ground truth, the
`livingroom_3` run we shipped in TASK 20 achieved:

| metric (non-structure objects) | value |
|---|---|
| recall @ label + centre ≤ 1 m | 4 / 87 = **5 %** |
| box size error (median, longest side) | **6.8×** |
| boxes with a dimension > 3 m | **65 / 102** |
| mean 3D IoU (the challenge's Task-2 metric) | **0.003** |

At 0.003 mean IoU the object-reference task scores essentially zero, since the
challenge pays 1 point at IoU ≥ 0.25 and 2 at ≥ 0.5.

Iterating against the live simulator could not fix this: every run explores a
different route, so a metric delta is as likely to be the route as the change
under test, and each experiment costs ~15 minutes.

## What we built

### 1. Frame recorder — `perception/recorder.py`

Captures the raw sensor input of every tick (equirect JPEG, `/registered_scan`
`.npy`, pose, and **all three message stamps**) under
`XIAO_HEI_FRAME_RECORD_DIR`. `VLMLogger` could not serve this: it only writes
from `respond()`, which needs an active question, so a pure exploration run
left 7 images on disk across every session we had.

### 2. Replay harness — `perception/replay.py`, `perception/__main__.py`

Split at the GPU boundary so the geometry loop does not pay for the detector:

- **Stage A** POSTs each recorded frame to the sidecar and caches the masks as
  COCO RLE. ~10 min; only invalidated by a detector change.
- **Stage B** re-lifts the cached masks against the recorded scans and poses,
  fuses through `ObjectMap`, and emits a scene graph that
  `perception.eval` scores unchanged. **38 seconds.**

`--min-move-m` / `--min-rot-deg` thin the corpus to viewpoint keyframes;
`--max-speed` / `--max-yaw-rate` / `--no-image-pose` / `--no-scan-accumulator`
turn the pipeline's open questions into offline ablations.

## What was actually wrong

Reading the sidecar's debug dumps settled the first question: **2D detection
and segmentation are fine.** YOLO-World boxes sat on real objects and SAM's
masks were tight. The failures were all downstream.

### Time synchronisation — the largest single defect

Every subscriber used `RELIABLE` + `depth=5`. When the subscriber falls behind
— and ours does, because the tick blocks ~1 s inside `/detect` — DDS *retains
and replays stale samples in order*, so the steady-state lag is
`depth × message period`:

| topic | predicted | measured |
|---|---|---|
| `/camera/image` (4 Hz) | 5 × 250 ms = 1250 ms | **1200 ms** |
| `/registered_scan` (4.7 Hz) | 5 × 212 ms = 1060 ms | **990 ms** |

Masks were being lifted against a pose from 1.2 s later — 0.3 m of error at
the median exploration speed, ~1 m at peak. Three fixes:

1. `BEST_EFFORT` + `depth=1` on sensor topics (drop stale frames rather than
   queue them). The question topic stays reliable.
2. `MultiThreadedExecutor` + a reentrant callback group, so callbacks keep
   draining while the tick sits in `/detect`.
3. `LatestCache` keeps 512 poses (2.5 s at 200 Hz) and `snapshot()` publishes
   `VLMInput.image_pose`, the pose interpolated at the *camera frame's* stamp.
   Position lerps, orientation nlerps along the short arc. `pose` still
   reports the newest sample, which is what exploration wants.

Result: skew 1200 ms → 355 ms, and the remainder is corrected by
interpolation. Independently, the number of fused objects dropped from 433 to
256 at unchanged recall — half our "objects" had been one object smeared
across several positions.

### Structure classes are not instances

`wall` / `floor` / `ceiling` / `window` and friends are 40 % of all lifted
observations. One mask covers an entire surface, often spanning two rooms, so
they fuse into dozens of overlapping nodes — `floor` came out as **68 separate
objects** against 1 in ground truth. They are now tagged `is_structure`
(`perception/vocab.py:STRUCTURE_LABELS`) and carried through `ObjectMap` →
`SceneRepresentation` → the scene-graph dump. They are still detected (a
counting question may ask about doors) but excluded from the Gemini prompt and
the detection dataset.

### Depth clustering — the lift's missing gate

70 % of *single-frame* observations had inlier clouds spanning more than a
metre, so the fusion was being fed garbage rather than creating it.

The z-buffer only rejects background *behind the object at the same pixel*.
Wherever the mask overshoots the true silhouette, the nearest surface at those
pixels genuinely *is* the background, so it passes the gate and drags the wall
into the object's cloud. Objects and their backdrops separate cleanly in
depth, so `_dominant_depth_cluster` sorts the inlier ranges, cuts wherever
consecutive values jump more than `DEFAULT_DEPTH_GAP_M` (0.3 m), and keeps the
most populous run — ties to the nearer cluster, since an object is in front of
what it leaks onto.

### The box was the least robust statistic available

`_Node._recompute` took the raw `min`/`max` of the union of every observation
ever merged in, while the centre used a median gate — which is how a node
could report a centre 1.3 m outside its own box. Now both come from the same
gated cloud, the box is per-axis 2–98 percentiles, and a cloud still spanning
> 3 m falls back to 26-connected voxel clustering (percentiles shave a tail;
they cannot remove a whole second blob).

### Multi-sweep accumulation now costs more than it gives

`ScanAccumulator` merged the last ten keyframes before lifting, to get small
objects over the `min_inliers` gate. Once the lifter clusters by depth that
trade stops paying: the keyframes span metres of travel, and depth clustering
can separate an object from the wall behind it but not one view of a surface
from another view of the same surface. Disabling it improved every axis on
both corpora, so `XIAO_HEI_SCAN_KEYFRAMES` now defaults to `0`.

| | livingroom_3 on → off | chinese_room on → off |
|---|---|---|
| box size error | 2.28× → **1.28×** | 1.52× → **0.87×** |
| precision @ 1 m | 0.119 → **0.242** | 0.152 → **0.233** |
| mean 3D IoU | 0.166 → **0.214** | 0.256 → **0.323** |

The original motivation (sparse returns on small distant objects) still stands
and now wants a different answer — scaling `min_inliers` with the object's
angular size, rather than inflating the cloud.

## Results — same corpus, same metrics, non-structure objects

Geometry work alone (time sync, structure split, depth clustering, robust
box), measured on `livingroom_3`:

| | before | after |
|---|---|---|
| single-frame inlier spread (median p95) | 1.69 m | **0.84 m** |
| boxes with a dimension > 3 m | 78 / 110 | **0** |
| box size error vs GT (median) | 6.8× | **1.27×** |
| mean 3D IoU | 0.004 | **0.214** |
| precision @ 1 m | 0.173 | **0.242** |
| recall @ 1 m | 0.218 | **0.276** |

End of the day, geometry **and** vocabulary, on both corpora:

| | livingroom_3 | chinese_room |
|---|---|---|
| recall @ 1 m | 0.218 → **0.517** | 0.305 → **0.500** |
| mAP @ 1 m | 0.078 → **0.401** | 0.204 → **0.415** |
| mean 3D IoU | 0.004 → **0.219** | 0.078 → **0.249** |
| boxes > 3 m | 90 → **0** | 46 → **0** |
| centre error (median) | 0.251 → **0.229** | 0.242 → **0.167** |

`chinese_room` is the held-out scene: nothing was tuned on it, and every
conclusion drawn on `livingroom_3` reproduced there.

## Two findings that contradicted the plan

**The class-size prior is now a regression.** `perception/size_prior.py`
(generated by `scripts/gen_size_prior.py`; 172 classes, median over 15 VLA-3D
scenes) was meant to replace the measured box, and would have been right when
boxes were 6.8× oversized. With robust boxes the measurement is better —
mean IoU 0.150 measured vs 0.109 with the prior. It survives only as the
fallback for detections that never got a box, where the alternative is a
zero-volume answer.

**The centre, not the size, is now what limits the score.** Holding one term
at ground truth and measuring the other:

| | mean IoU | IoU ≥ 0.25 |
|---|---|---|
| as-is | 0.150 | 20.9 % |
| our centre + GT size | 0.181 | 27.9 % |
| **GT centre + our size** | **0.296** | **65.1 %** |

Centre error is median 0.30 m / p90 0.93 m, with no systematic per-axis bias
(±0.09 m), so there is no offset to correct — it is fusion noise. Only 43 of
99 predictions match a same-label GT within 1.5 m, which points at duplicate
and mis-merged nodes as the remaining source.

## Vocabulary: the other half of the recall problem

`DEFAULT_PRIOR` was copied from `arabic_room`'s object list, so it carried
`hookah` but not `chair`, and an open-vocabulary detector cannot find what it
is not prompted for. `scripts/gen_vocab_prior.py` regenerates it from the
union of the 15 VLA-3D scenes — every label present in **at least 2 distinct
scenes** (110 labels, 82 % of all ground-truth objects), with plurals
collapsed. Recurrence across scenes is the right criterion because the
challenge scenes are unseen; frequency inside one scene is not evidence of
generality.

Generating it from the scene vocabulary also fixes wording. We had been
detecting the `livingroom_3` sofa correctly and calling it "sofa" while ground
truth calls it "couch" — a perfect detection scored as a miss.

| | livingroom_3 | chinese_room |
|---|---|---|
| recall @ 1 m | 0.276 → **0.517** | 0.329 → **0.500** |
| mAP @ 1 m | 0.106 → **0.401** | 0.226 → **0.415** |
| precision @ 1 m | 0.242 → 0.169 | 0.233 → 0.167 |

A leave-one-scene-out check confirms the gain is not leakage: excluding the
evaluated scene from vocabulary generation costs 1 % of GT coverage on
`livingroom_3` and 6 % on `chinese_room`, against a recall gain of 17–24
points.

Checked against the official question set (75 questions, 15 scenes): the
questions use the exact ground-truth label wording. There are no synonyms
("sofa" for a `couch`) and no hypernym-only questions ("cabinet" for a
`tv cabinet`) — every apparent counterexample turned out to be an artifact of
naive substring matching. So no synonym table and no LLM merge pass is
needed; aligning the detector's vocabulary with the scene's is sufficient.

## Duplicate suppression: tried, measured, rejected

Duplication is real — each ground-truth object we claim is claimed by 1.4–1.7
predictions — but neither remedy survived measurement, and both are reverted.

**Cross-face NMS in the sidecar** was dropped before implementation. The four
unwrapped faces overlap by ~10°, so an object on a seam should be detected
twice; but of all same-label detection pairs within one frame, only 3–5 % lift
to within 0.3 m of each other, and even those have a median mask
intersection-over-smaller of 0.00 — they are different parts of one object,
not the same pixels twice. The "three ceilings in one frame" that prompted
this idea were structure classes, which the split above already removes.

**Same-label consolidation by intersection-over-smaller**, plus a merge
distance that grows with object size, did raise precision (livingroom_3 0.169
→ 0.265) by folding nested duplicate nodes together. But box overlap cannot
tell "two views of one couch" from "two chairs at one table": on the denser
`chinese_room` it merged real instances, taking recall from 0.500 to 0.402 and
mAP from 0.415 to 0.332. Sweeping both thresholds (diagonal fraction
0.15/0.25/0.5 × IoS 0.6/0.8) found no setting that was neutral on both
corpora. The rationale is recorded in `object_map.py` so it is not retried
blind.

## Open items

- **`ScanAccumulator` should probably default to off.** With depth clustering
  in place it costs more than it gives (boxes > 3 m: 52 → 18, precision
  0.134 → 0.220 when disabled). Not flipped yet — it changes online behaviour
  and `livingroom_3` is one open-plan scene.
- A yaw-rate gate (≤ 0.2 rad/s) is a free precision win (0.242 → 0.266) and is
  not wired into the online path yet.
- Duplication is still unsolved (1.4–1.7 predictions per claimed object). The
  approaches that only look at box overlap are exhausted; a working fix
  probably needs appearance or viewpoint evidence — e.g. two nodes seen from
  the same viewpoint at the same time cannot be the same object.
- Precision is the weak axis after the vocabulary change (0.17). Sweeping the
  detector's score threshold is the obvious next knob; it costs a stage-A
  re-run rather than the 38-second loop.
- End-to-end scoring is not wired up: every number here is detection quality,
  not the challenge's per-question 0/1/2. Building the detected dataset from a
  new scene graph and running it through Gemini would say what the geometry
  work is actually worth.

## Things that cost time, recorded so they cost less next time

- **The frontier explorer wedges.** On `livingroom_3` it stopped at
  `visited=3 skipped=29`; on `chinese_room` the robot drove itself into a
  corner and stayed there. Coverage for the `chinese_room` corpus had to be
  driven manually with `ab_results/wp_driver.py` over a nearest-neighbour tour
  of the cells in `traversable_area.ply`. Commanded waypoints must be ≤ ~1.5 m
  apart or the nav stack gives up on them.
- **Two autonomy stacks will silently fight.** Restarting the simulator with
  `pkill` left the old `localPlanner` and `pathFollower` alive; with two of
  each publishing to `/cmd_vel` the robot jitters in place and looks stuck.
  `ros2 node list | sort | uniq -c` is the tell. Restart the container instead.
- **A sidecar restart silently disables detection.** `set_classes` is
  dedup-cached on the responder, so after the sidecar restarts with an empty
  class list the responder never re-pushes and `/detect` returns 0 detections
  forever. Restarting `xiao_hei_ai_module` is the workaround; the real fix is
  for the client to re-push when the sidecar reports an empty list.
- **`PERCEPTION_DEBUG=1` writes ~7 MB per frame.** It reached 7.6 GB in about
  twenty minutes of idle detection. Turn it on for a handful of frames, not a
  run.
- **The frame recorder keeps recording after exploration ends.** A run that
  finishes and sits still adds hundreds of near-identical frames; use
  `--min-move-m` / `--min-rot-deg` when replaying, or stop the node.

## Verification

```
python -m xiao_hei_vln.perception replay detect --frames frames/<run>
python -m xiao_hei_vln.perception replay lift   --frames frames/<run> \
    --out scene.json --min-move-m 0.15 --min-rot-deg 10
python -m xiao_hei_vln.perception.eval --scene scene.json \
    --gt-zip <scene>.zip --scene-name <scene>
```

New tests: `test_frame_recorder.py`, `test_pose_interpolation.py`,
`test_structure_labels.py`, `test_size_prior.py`, plus depth-clustering cases
in `test_perception_lifter.py`.
