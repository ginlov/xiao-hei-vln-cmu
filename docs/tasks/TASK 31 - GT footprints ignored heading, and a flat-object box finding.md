# TASK 31 — GT footprints ignored heading, and a flat-object box finding

## Why

Two things surfaced while reviewing carpet node #0 in `arabic_room` against
ground truth ([[TASK 29]], [[TASK 30]]).

1. **The GT footprint in every debug/viewer view was drawn with the wrong
   orientation.** `dump_debug.py` wrote each GT object's box as
   `center ± size/2`, dropping `heading`. `arabic_room`'s carpets carry a
   heading: #58 (left) is `-1.568 rad ≈ -90°`, #32 (right) is `≈ -180°`. So
   #58's long axis (`size.x = 2.16`) actually runs along **world y**, but the
   viewer drew it along x — the carpet appeared horizontal when it is really
   vertical. The lifted LiDAR cloud confirmed the truth: node #0's points span
   ≈1.4 m in x and ≈2.1 m in y.

2. **For a flat floor object, the averaged box under-sizes the long axis.** The
   ObjectMap box is a point-count-weighted mean of each observation's own box
   (`_Node._recompute`), which is correct for volumetric objects — pooling
   near-face LiDAR slabs across viewpoints inflates them ~6x. But a carpet is
   planar (all points z ≈ 0.04), so every view lifts points in the *same*
   plane with no depth scatter, and no single oblique view sees the full 2.17 m
   length. Averaging the per-view boxes therefore converges *short*.

## What changed

- **`dump_debug.py` now applies heading to GT boxes.** New `_gt_world_aabb(e)`
  rotates the four footprint corners by `e.heading` (Rz) and takes their world
  AABB; z is left unrotated (floor objects). `viz.json`'s `gt[].bmin/bmax` come
  from it, so `merge_video.py` and `inspect_vp.py` — which read those fields —
  now draw the correct orientation with no change of their own.
- Regenerated `debug/arabic_room` (frozen dets, 0.6, keyframes 2, image-lag 0)
  and the carpet merge video. Left carpet GT now renders vertical, right carpet
  horizontal; node #0 sits inside its GT footprint at the right orientation.

## The box finding (measured, not yet fixed)

Node #0 (30 obs, GT #58 = 1.44 × 2.17 m):

| box | dims | IoU vs GT |
|---|---|---|
| current (weighted mean of per-view boxes) | 1.38 × 1.52 | 0.670 |
| **pooled cloud, raw min/max** | 1.41 × 2.09 | **0.948** |
| pooled cloud, 2–98 percentile | 1.31 × 1.58 | 0.662 |

Raw min/max over the accumulated cloud nearly nails the carpet (0.67 → 0.95),
because the flat object has no near-face scatter to inflate it. The percentile
trim fails too — for a large flat object the "2–98% tail" it clips is real
extent. This is exactly the option that is *unsafe* for volumetric furniture,
so the natural fix is **shape-conditional**: size flat floor objects
(carpet/rug/mat, box-z ≈ 0, large footprint) from the pooled raw cloud, keep
the averaged box for volumetric objects. Prototype + re-score is the follow-up.

## Scope note

The scorer (`perception/eval.py`) is unaffected by item 1: its primary metric is
**center-distance**, and it deliberately ignores heading (axis-aligned
approximation). This was purely a visualization defect. The box under-sizing
(item 2) *does* touch IoU-based secondary metrics but not the center-distance
primary.
