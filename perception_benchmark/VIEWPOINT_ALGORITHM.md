# Deterministic viewpoint generation for perception benchmarking

> **Scope / status.** Offline evaluation tooling — **not part of the challenge
> submission stack**. It exists to benchmark the perception stack
> (detect → 3D lift → ObjectMap fusion → scene graph) reproducibly, decoupled
> from navigation. Kept out of `src/` deliberately: nothing here runs at test
> time.

## 1. Purpose

At test time the scene graph's quality is confounded by **navigation coverage**
— frontier exploration is non-deterministic, so two runs see the scene
differently and a metric delta can't be attributed to a perception change.

This tool removes navigation from the loop entirely: precompute a fixed set of
**viewpoints** per scene, teleport the robot to each via
`/unity_sim/set_model_state`, capture the sensor inputs, build the scene graph
from exactly those viewpoints, and score it against the scene's
`object_list.txt`. Same viewpoints → identical inputs → attributable metrics.

**Explicitly out of scope:** the *perception capture gate* (detector
sensitivity, `min_inliers`, small/glossy objects). "Observed" here is a purely
**geometric** predicate. We are benchmarking perception *given* good viewpoints,
not end-to-end run performance.

## 2. Key simplifications

1. **360° camera ⇒ a viewpoint is a position `(x, y)`.** The equirectangular
   camera sees all azimuths, so yaw does not affect what is visible. Viewpoint
   selection collapses from 6-DOF view planning to a **2D coverage / set-cover**
   problem over positions.
2. **Capture gate ignored ⇒ visibility is pure geometry**, computable offline
   from the scene files with no perception in the loop:

   > object `o` is **observed** from position `p`  ⟺
   > `R_min ≤ dist(p, o) ≤ R_max`  **and**  the segment `p→o` is unobstructed in
   > the 2D occluder map  **and**  `o` is within the camera's vertical FOV.

## 3. Data sources (all offline)

Per scene at `/home/ubuntu/workspace/dataset/vla-3d/Unity/<scene>/`
(see the `vla3d-scene-gt-data` memory). All 15 scenes present.

| Role | File |
|---|---|
| **Targets** to cover | `object_list.txt` — `id  cx cy cz  sx sy sz  heading  "label"` |
| **Free space** (candidate positions) | `<scene>_free_space_pc_result.ply` — binary Open3D PLY (`x,y,z,rgb,obj_id,region_id`) |
| **Occluders** (line-of-sight) | `<scene>_pc_result.ply` (dense, ~8.5 M pts) **or** wall/architecture entries in `object_list.txt` |

Coordinates are the sim **map frame** — positions planned here feed
`set_model_state` directly (same alignment `eval.py` assumes). Sanity-check once
per scene.

## 4. Scene sizes (measured)

Most scenes are small single rooms; two are large multi-room buildings. Grid /
`R_max` choices below are informed by this.

| tier | scenes | x×y | footprint | objects |
|---|---|---|---|---|
| small room | japanese_room, hotel_room_2, arabic_room, studio, chinese_room, office_1/2, livingroom_1/4 | ~5–10 m/side | 35–70 m² | 63–161 |
| medium | livingroom_2/3, loft, hotel_room_1 | ~10–14 m | 90–133 m² | 86–115 |
| **large building** | home_building_2 | 25×18 m | 442 m² | 227 |
| **large building** | home_building_1 | 33×36 m | 1187 m² | 432 |

## 5. Algorithm

**Input:** scene name. **Output:** `viewpoints[scene]` (validated teleport
poses) + a coverage report.

### Stage A — Occluder grid (line-of-sight map)
Project `pc_result.ply` (voxel-downsampled) — or the wall/architecture bboxes
from `object_list.txt` — to a **2D occupancy grid**. Mark occupied cells.
- **Resolution ~0.1 m.** This grid is the *accuracy-limiting* structure: too
  coarse and raycasts leak through thin walls or falsely block. Finer than the
  candidate grid.

### Stage B — Candidate positions
Use the `free_space_pc_result.ply` points **directly** as candidates — they are
pre-validated free space at ~0.5–0.7 m native spacing. Drop any within
< robot-radius clearance of an occupied cell.
- Densify to ~0.25 m **only near doorways / narrow passages**, where the single
  position that sees through a gap can fall between coarse samples.
- (Skip the concave-hull + regular-grid step; the free-space points already are
  the candidate set.)

### Stage C — Visibility matrix `V[p][o]`
For each candidate `p` × object `o`: set `V=1` iff the **observed** predicate
(§2) holds — range window, 2D raycast unobstructed, vertical FOV. Cheap and
embarrassingly parallel.

### Stage D — Greedy (multi-)set-cover
Repeat: pick the candidate covering the most still-uncovered objects → append to
the viewpoint list → mark those objects covered. Continue until every object is
covered **`k` times from well-separated angles** (`k = 2–3`).
- `k > 1` is required for ObjectMap **box convergence** — a single-view object
  gets a degenerate box, making the `iou@0.25` metric meaningless.
- "Well-separated" = reject a new covering viewpoint whose bearing to the object
  is within ~30° of an already-chosen one.

### Stage E — Patch uncovered objects
Objects no candidate covers:
- **Near-occluded** → add a close-approach viewpoint at
  `object_pos + offset toward the nearest free cell`.
- **Genuinely unobservable** (inside closed containers) → catalogue in the
  report as `not_observable`; do not let them silently tank recall.

### Stage F — Assign teleport poses + validate
Each chosen `(x, y)` → add ground `z` (from free-space/terrain), an upright
orientation, any fixed yaw (irrelevant to coverage). Re-verify the pose sits in
free space with clearance. Emit as the output pose list.

### Stage G — Empirical verification loop  *(what makes it ground-truth)*
Teleport → capture → build scene graph → score object **observability** vs
`object_list.txt`. Objects missed that were *expected* covered ⇒ the 2D
occlusion model was wrong there ⇒ add a viewpoint, repeat until observability
recall ≈ 100% (excluding the `not_observable` set).

## 6. Parameters

| param | default | notes |
|---|---|---|
| `R_max` | 6 m general; **3–4 m in small rooms** | smaller ⇒ closer, multi-angle looks (better box convergence) instead of one distant everything-shot |
| `R_min` | 0.5 m | don't stand inside the object |
| `k` (views/object) | 2–3, ≥30° apart | drives viewpoint count in small rooms |
| candidate spacing | free-space points (~0.5 m); 0.25 m at doorways | coverage is insensitive to this except at occlusion edges |
| occluder grid | **0.1 m** | the accuracy-limiting grid |
| robot clearance | robot radius | drop candidates too close to walls |
| furniture as occluders | on | more realistic sightlines |
| vertical-FOV check | on | matters for very high/low objects at close range |

## 7. Output schema

```jsonc
// viewpoints/<scene>.json
{
  "scene": "arabic_room",
  "frame": "map",
  "viewpoints": [
    {"id": 0, "x": -1.2, "y": 3.4, "z": 0.2, "yaw": 0.0, "covers": [0,3,7,12]}
  ],
  "report": {
    "n_objects": 85,
    "covered": 82,
    "not_observable": [41, 63],       // inside containers etc.
    "mean_views_per_object": 2.4
  }
}
```

Consumed by the teleport-and-capture harness (publishes `set_model_state`,
waits for fresh frames, records the 7 legal inputs per viewpoint).

## 8. Caveats / decisions

- **2D occlusion is approximate.** A low object visible over a low obstacle, or
  sightlines under tables, need a 2.5D/height-aware raycast. Start 2D; upgrade
  only where Stage G shows misses.
- **`region_id` is NOT a usable room decomposition here.** The per-point
  `region_id` in the free-space PLY is mostly a single region even for
  `home_building_1` (33×36 m ⇒ 1 region). Do **not** seed viewpoints by it;
  check `<scene>_region_result.csv` for finer structure, else rely on
  set-cover + occlusion to separate rooms.
- **Coverage certifies geometry, not perception quality** — intended. The number
  is "perception given these viewpoints," not end-to-end challenge performance.
- **Seed RNG** in the downstream ObjectMap (its `PTS_CAP` subsample uses global
  numpy RNG) so the benchmark is bit-reproducible.
- **Big buildings dominate cost & viewpoint count.** `home_building_1/2` will
  need dozens of viewpoints (driven by walls/rooms, not grid fineness); the
  small rooms need ~2–3 each (driven by the `k`-cover requirement).
```
