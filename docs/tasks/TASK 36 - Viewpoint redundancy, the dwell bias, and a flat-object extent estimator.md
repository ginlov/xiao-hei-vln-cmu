# TASK 36 — Viewpoint redundancy, the dwell bias, and a flat-object extent estimator

## Why

After the merge gate collapsed each carpet to one node ([[TASK 32]]), the right
carpet's box (node #13) was seen to **shrink and drift** as observations
accumulated: GT-coverage peaked at 53% (14 obs) and decayed to **20%** (240 obs),
with the box migrating into the −y/+x corner.

Root cause, measured:

- **Not `_core_points`** — it discarded 0.0% of points and clustering fired on
  0/240 observations (the carpet cloud is clean and coplanar).
- **The estimator + viewpoint redundancy.** Each dimension of a node's box is
  the point-count-weighted *mean* of every view's own `max−min` extent. Of node
  #13's 240 observations, **228 (88% of the weight) came from a single 0.5 m
  robot dwell cell**, each seeing only a thin 0.42 m slice. A handful of earlier
  pass-through views saw the full 1.35 m width (≈ GT 1.44) but at ~3% weight were
  drowned out. So the box averages down to the dwell view and its centroid.

The estimator has no defense against **viewpoint redundancy** — it weights by
points/observations collected, not by how much *new* information a viewpoint
adds. Standing still degrades the box.

## The design (two complementary levers)

**1. Extent estimator — prefer reach over average (implemented, this task).**
For **flat** nodes (`median per-view z-extent ≤ FLAT_Z_M`), size the box from the
**max** per-view extent in x/y — the reach of the single best view that saw the
whole object — instead of the weighted mean. z stays the weighted mean (~0
regardless). Volumetric nodes are untouched: their mean cancels the ~6×
inflation of pooling near-face LiDAR slabs, which `max` would instead amplify.

**2. Capture-time novelty gate (implemented, `responder._inject_visible`).**
The tick is decoupled so the **ScanAccumulator ingests LiDAR every tick** (cloud
stays dense/registered) while **perception (detect → lift → `ObjectMap.add`)
fires only on a novel viewpoint**. With a 360° panorama the frame content
depends only on position, so novelty is pure translation:

    kept = KDTree of accepted (x, y) positions   # ALL previous, not just the last
    novel(pos) := distance-to-nearest-kept > NOVEL_DIST   (~0.3–0.5 m)

Comparing against *all* kept positions (not just the last) is required so a
revisit/loop doesn't re-admit an already-covered pose. This stops dwell
over-counting at the source and cuts perception compute. It is object-agnostic,
so it caps redundant *frames* but does not by itself grow the box — hence it
pairs with lever 1.

## Prototype results (offline, arabic_room, GT-coverage %)

| | right #32 | left #58 |
|---|---|---|
| baseline (weighted mean, no gate) | 20 | 64 |
| **flat-max estimator, no gate (shipped)** | **56** | **87** |
| gate 0.3 m + mean | 56 | 64 |
| gate 0.3 m + p90 | 67 | 76 |
| gate 0.3 m + max | 69 | 77 |

Notes: `p90` alone (no gate) fails on the right carpet (15%) because 95% of its
views are the narrow dwell slice — only `max` escapes extreme redundancy. Once
the gate rebalances the view distribution, the safer high-percentile works.
Gating slightly *hurts* the already-diverse left carpet (87→77) by discarding
useful views — evidence the two levers should be tuned together, not stacked
blindly.

## Shipped here

**Estimator** — `object_map._Node._recompute`, flat-conditional (`median z ≤
FLAT_Z_M`), two levers:
- **Max extent** (xy): the reach of the single best view, not the dwell mean.
  Carpet GT-coverage 20→56% (right), 64→87% (left).
- **Union-midpoint centre** (xy): the midpoint of the per-view boxes' union
  instead of the point-weighted centroid, so a box built from partial reach
  isn't dragged onto the seen half. Right carpet centre err 0.52→0.24 m
  (coverage 69→82%), left 0.30→0.12 m (77→84%).

Volumetric nodes are untouched (weighted mean, which cancels the ~6× near-face
pooling inflation `max`/union would amplify). Why flat-only, not generic: only
coplanar clouds have no near/far-face ambiguity, so their extreme edges are real
surface points; a volumetric object's extremes are offset near-face slabs, so an
extremal statistic (`max`/union-midpoint) trusts the *most wrong* view.

**Novelty gate** — `responder._inject_visible`: scan accumulator runs every
tick; detect → lift → fuse is gated on `_viewpoint_is_novel`, a nearest-neighbour
check over ALL kept `(x,y)` (revisits caught) against `novel_viewpoint_m`. Wired
through `_PerceptionSettings` as `XIAO_HEI_NOVEL_VIEWPOINT_M`. Reproducible
offline: `replay_score.py`/`dump_debug.py --novel-viewpoint-m <m>`. **Default is
0.3 m (ON)** — flipped from off after the box-quality gain and the ~11× compute
saving were judged to outweigh the small recall cost on the live robot (the
accumulator still ingests LiDAR every tick, so the cloud stays dense).

### Measured (arabic_room, frozen dets @0.6, flat estimator on)

| gate | perceived | pred | mAP@0.5 | mAP@1.0 | IoU@0.25 |
|---|---|---|---|---|---|
| off (0.0) | 397 | 29 | 0.221 | 0.263 | 0.051 |
| **0.3 m (shipped)** | 36 | 24 | 0.199 | 0.235 | 0.091 |

The estimator alone (gate off, vs the pre-TASK-36 mean/centroid baseline)
is a clean win: mAP@0.5 0.195→0.221 (+14%), centre_err 0.240→0.221 m, no recall
cost. The gate then trades ~11× compute + IoU@0.25 +80% for a small mAP drop.

Tests: `test_flat_node_box_takes_max_extent_not_mean`,
`test_flat_node_centre_is_union_midpoint_not_weighted_centroid`,
`test_volumetric_node_still_uses_mean_extent`,
`test_novelty_gate_skips_perception_until_robot_moves`,
`test_novelty_gate_disabled_perceives_every_tick`.

## Open / next

- **Sweep `novel_viewpoint_m` and `max` vs high-percentile** across the corpus
  against the real metric; `max` is safe on coplanar clouds but fragile to a
  single spill view. Relates to [[B9]] (box containment) and B6.
- **Recover the gate's box-quality gain without the recall cost** — gate the
  *fusion weight* rather than the *observation* (perceive every frame, down-weight
  redundant viewpoints in `_recompute`). Keeps recall while killing the dwell bias.
- The score/SAM detection thresholds this pairs with are [[TASK 37]].
