# TASK 32 — A flat-aware, gap-based merge gate for ObjectMap

## Why

Tracing why the two `arabic_room` carpets shattered into nine nodes ([[TASK 31]],
backlog B6) pinned it on the `ObjectMap.add` merge gate:

> merge same-label detections if `iou_3d ≥ 0.3` **OR** `centre_dist ≤ 0.4 m`

Both clauses fail for flat objects:

1. **3D IoU is structurally 0** for a carpet — its box is ~0 m thick, so the
   volumetric intersection is 0 even when two footprints fully overlap.
2. **A fixed centre-distance gate is the wrong shape.** Partial views produce
   offset centroids, so two fragments of one carpet sit >0.4 m apart (measured:
   node #17 vs #0 were 0.565 m apart with their surfaces *touching*, gap 0.000).
   Meanwhile the same 0.4 m radius wrongly fuses two genuinely distinct small
   objects (two pillows side by side).

## What changed (`src/xiao_hei_vln/perception/object_map.py`)

The gate is now:

> merge if `iou ≥ MERGE_IOU (0.3)` **OR** `surface_gap ≤ MERGE_GAP_FRAC · size`

- **Flat-aware IoU.** `iou` is `iou_3d` by default but falls back to a new
  `iou_2d` (xy-footprint / bird's-eye IoU) when **both** boxes are flat
  (`_is_flat`: z-thickness ≤ `FLAT_Z_M = 0.10 m`). The volumetric path is
  untouched for everything else.
- **Size-scaled surface-gap replaces centre-distance.** `box_gap` (shortest
  distance between AABB surfaces; 0 when they touch/overlap) is compared to
  `MERGE_GAP_FRAC (0.15) · max(footprint_diagonal_a, footprint_diagonal_b)`.
  Scaling by the *larger* footprint lets a big carpet absorb a nearby sliver
  while keeping two small same-label neighbours apart. The old `merge_dist`
  constructor arg / `MERGE_DIST` constant are gone; `merge_gap_frac` replaces
  them (and `gemini/batch.py`'s no-merge config now passes `merge_gap_frac=-1.0`).

The deferred finalize()-time suppression (`NMS_DIST`/`NMS_GAP`) is unchanged.

## Result — `arabic_room` (frozen dets @ 0.6, keyframes 2)

| | carpet nodes | total nodes (exported) |
|---|---|---|
| before (0.6, TASK 30) | 9 | 46 |
| **flat/gap gate** | **2** | **30** |

Two carpet nodes, one per GT carpet (node 0 → left #58, 74 obs; node 13 → right
#32, 240 obs). Distinct instances were **not** collapsed — pillows stayed at 2
nodes, potted plants 5/5 vs GT, wall lamps 4/4, stools 2/2. Carpet
fragmentation (B6) is effectively resolved for this scene.

## Caveats / open

- **Multi-scene sweep still needed.** This is a global change to the fusion
  gate. The code comment near `MERGE_GAP_FRAC` records the historical risk: a
  size-scaled distance gate previously cost `chinese_room` recall (0.500 →
  0.402) by merging distinct instances in dense scenes. This gate is *gap*-based
  (not centre-based) to avoid that, and arabic_room shows no false collapse, but
  `FRAC` and `FLAT_Z_M` should be swept end-to-end across the 14-scene corpus
  before trusting it broadly.
- **Box size is still short on the long axis** — the averaged per-view extent
  ([[B9]]) is untouched; only the *count* of carpet nodes improved, not each
  box's dimensions.

## Tests

`tests/test_object_map.py`: added `iou_2d` unit coverage, flat-fragment merge,
size-scaled gap merging a small fragment into a large flat node, and the
small-neighbour (pillow) non-merge. Finalize-isolation tests reparametrised to
`merge_gap_frac=0.0`. 70 perception tests pass.
