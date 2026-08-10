# TASK 22 — Same-label duplicate suppression in ObjectMap

A third of the scene graph was redundant: one physical object represented by
several nodes. This task measures the causes, ships the fix for the tractable
half, and records why the other half was deferred.

## Why the existing dedup could not fire

`ObjectMap` had two mechanisms, both keyed on 3D IoU of axis-aligned boxes:

- merge in `add` (same label): `IoU >= 0.3` **or** `centre <= 0.4 m`
- cross-label NMS in `finalize`: `IoU >= 0.5`

Over 14 scenes, node pairs whose centres are within 0.5 m have median IoU
**0.055**, and **none of 691 reach `NMS_IOU`** — the cross-label rule has never
fired in this dataset. `MERGE_IOU` contributes on 12% of pairs; the 0.4 m
distance clause carries the rest.

This is not a lifter accuracy problem. Those pairs are close: their box surfaces
sit a median **3 cm** apart (53% under 5 cm, 82% under 15 cm), and the boxes are
thin slabs — median extents 0.55 × 0.32 × 0.09 m, aspect 7.6:1. A LiDAR sweep
sees one *face* of an object, so two viewpoints produce adjacent, disjoint
surface patches. Two boxes of ~0.01 m³ whose centres are 0.43 m apart **cannot**
overlap, and IoU is a step function below overlap: 3 cm apart and 3 m apart both
score exactly 0.

TASK 21 made this worse, predictably — in `arabic_room` the median IoU between
co-located nodes fell **0.158 → 0.010** once boxes tightened. The thresholds
were implicitly calibrated against bloated boxes.

## What shipped

`box_gap()` — shortest distance between two AABB surfaces, 0.0 when touching.
Unlike IoU it stays informative below overlap.

`finalize` suppresses the weaker of two co-located nodes on either the legacy
`IoU >= NMS_IOU` (any label) **or** `centre <= NMS_DIST` and
`box_gap <= NMS_GAP`, restricted to **identical labels**. Best-supported node
wins: most observations, then score.

`NMS_DIST = 0.4 m` matches `MERGE_DIST` deliberately. `add` already merges
same-label nodes inside that radius — but a node's centre **moves as it
accumulates points**, so two nodes created further apart can drift inside it
with nothing re-checking. This is that check, deferred until the centres settle,
which also bounds what it can recover.

| threshold | redundant | counting MAE | mAP@1 | R@1 | P@1 |
|---|---|---|---|---|---|
| off | 216 | 2.8356 | 0.2134 | 0.2552 | 0.3253 |
| **d0.4 g0.05** | **204** | **2.8126** | 0.2123 | 0.2524 | 0.3267 |
| d0.4 g0.15 | 204 | 2.8126 | 0.2123 | 0.2524 | 0.3267 |
| d0.3 g0.05 | 215 | 2.8351 | 0.2131 | 0.2546 | 0.3251 |

Counting MAE improves on 6 scenes, worsens on 1, unchanged on 7. Dropping to
0.3 m removes the effect entirely. The gap term is **not binding** at 0.4 m
(0.05 and 0.15 are byte-identical) and is kept only as a guard against a large
box whose centre coincides with a small one.

## What was deferred

Cross-label duplicates are the larger half — 82% of duplicated GT objects — and
a **hard gate** rather than a threshold: `add` skips any candidate whose label
differs before evaluating distance or IoU. Restricting the new rule to identical
labels was a deliberate call: applying it across labels would also collapse
genuinely touching distinct objects (a pillow on its sofa) and would hide
detector label instability behind whichever label won.

Full analysis, the synonym-vs-misclassification split, and two candidate
approaches are recorded in `docs/tasks/backlog.md` B3.

## Changes

| file | change |
|---|---|
| `src/xiao_hei_vln/perception/object_map.py` | `box_gap()`; `_suppresses()`; `NMS_DIST` / `NMS_GAP`; `absorbed_labels` in `to_list()` |
| `perception_benchmark/replay_score.py` | `--nms-dist`, `--nms-gap` |
| `perception_benchmark/box_quality.py` | `dup_objects` / `redundant` columns |
| `tests/test_object_map.py` | 5 tests for the suppression rule |
| `docs/tasks/backlog.md` | B3 |

`absorbed_labels` records a suppressed label on the survivor rather than
discarding it. It is dormant while the rule is same-label-only, and is the hook
B3 needs so label instability stays visible instead of being swallowed.

One implementation trap worth noting: `export()` builds a throwaway view that
**shares the same `_Node` objects** as the live map, so recording absorbed
labels on the nodes would leak into the live map and accumulate every tick.
They are held in a per-map dict keyed by `node_id` instead, and
`test_export_does_not_mutate_the_live_map` pins that.
