# 3D localization needs multi-frame fusion — design note (TASK 10)

Status: **parked for later discussion.** We are focusing on 2D recognition (YOLO)
first. This captures why single-frame 3D failed and the agreed direction.

## The problem
We tried to get each object's 3D position (map frame) from a **single frame** by
back-projecting `/registered_scan` lidar onto the equirectangular semantic image
and labelling each lidar point by the pixel colour it lands on. The per-object
3D centre was then the mean of its labelled points.

This is unreliable for many object types.

## Evidence (run1, 4 frames, livingroom)
- Same physical object's per-frame 3D centre scatters by a **median ~4–6 m**
  (95% > 1 m), even after filtering (0,0,0) filler, requiring ≥5 inlier points,
  and median-gating outliers.
- The **mirror** seen in 4 frames was placed at 4 different room corners:
  (6.45, 0.77), (6.10, −4.51), (6.96, −3.25), (−1.76, 4.74).
- A z-buffer (only label the nearest lidar point per pixel) **did nothing**:
  lidar is so sparse that only **35 / 10543** occupied pixels had ≥2 points, so
  there was nothing to occlude-test against.

## Root cause
1. **Sparse lidar**: ~10k points over a whole room → most objects get a handful
   of points or none.
2. **Single-viewpoint occlusion**: each frame only sees surfaces facing the
   robot; everything behind is blind.
3. **Dense-semantic / sparse-lidar bleed**: the semantic image is dense and shows
   the *nearest* surface, but a lone lidar point landing in an object's silhouette
   may physically be on the **background behind it** → it gets the foreground
   object's colour → drags the 3D centre metres away. Different viewpoints bleed
   different background points → the per-frame centroid jumps around.
4. **Specular / flat objects** (mirror, glass, painting, tv, picture, shelf)
   return little/no lidar of their own → their labelled points are *all* bleed.

This is fundamental to a **sparse-LiDAR + camera** rig, not a bug. (Autonomous
driving, e.g. nuScenes, accumulates ~10 sweeps for exactly this reason.) The fact
that `/registered_scan` is published in the **map frame** is the system telling us
it is meant to be **accumulated across frames**.

## Agreed direction: 3D is a multi-frame / scene-level problem
Two complementary multi-frame mechanisms (single frame cannot do either well):

1. **Accumulate lidar → dense cloud → cluster.** True surfaces are hit repeatedly
   at the *same* map coordinates → dense clusters; bleed points land at different
   random places each frame → sparse noise. DBSCAN (or similar) keeps the dense
   clusters, rejects noise, and fills occlusion shadows. Objects with no real
   return (mirror) yield no dense cluster → honestly "no reliable 3D".
2. **Multi-view triangulation of 2D detections.** An object detected in 2D from
   ≥2 *known* robot poses gives ≥2 bearings → their intersection is the 3D
   position, with **no lidar at all**. This is the route for specular/flat/thin
   objects. Mathematically impossible from one frame; trivial from two.

## Proposed two-stage architecture
| stage | data unit | reliable? | role |
|---|---|---|---|
| **2D detection** | single-frame camera (perspective crop) | yes, single-frame | trains the YOLO sub-model |
| **3D localization** | multi-frame accumulation (lidar pool + 2D triangulation) | needs multi-frame | scene object list + reference-question CUBE |

Key correction to the earlier pipeline: **3D should not be a per-frame attribute**
(`detection_gt.json.center_3d`). It belongs to a **scene-level fusion stage** built
from accumulated data. Per-frame GT stays 2D only (boxes/classes — reliable).

## Capture implication
The coverage scan must give **overlapping nearby viewpoints** so that
(a) lidar accumulates densely and (b) each object is seen from ≥2 poses with a
usable **baseline** for triangulation. So viewpoint spacing matters for 3D, not
just for camera coverage.

## What to build later
1. Split `branchA_gt.py` per-frame output to **2D-only** (drop/relegate center_3d).
2. Scene-level 3D module: accumulate the run's scans (already map frame), pool per
   object, DBSCAN; for no-lidar objects, triangulate their 2D detections across
   frames using the known poses. Output one map-frame 3D per scene object + a
   `loc_source` ∈ {lidar_cluster, triangulated, none} and a confidence.
3. Evaluate 3D only on the reliable subset; report coverage by `loc_source`.

(The colour-based instance id remains the free GT for *associating* detections
across frames during all of the above — sim-only, training/eval only.)
