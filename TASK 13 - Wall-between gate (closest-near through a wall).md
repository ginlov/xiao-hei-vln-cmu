# TASK 13 — Drop closest/near questions whose anchor is across a wall

## Problem

`vla3d_ref.jsonl` idx 12: **"Find the cabinet closest to the fire alarm."**
(office_2). The target cabinet (4.61, −0.45) and the fire alarm anchor
(6.08, −2.38) sit on **opposite sides of a glass-wall partition** (id 65,
y ≈ −0.76). The straight-line xy distance (2.42 m) crosses that wall, so the
"closest" cabinet is actually on the far side — you'd have to walk around. A
ground robot scoring a navigation waypoint can't use straight-line-through-wall
proximity.

This is the geodesic-vs-Euclidean problem. VLA-3D referential statements rank by
straight-line distance and ignore walls. Measured over single-layer
`closest`/`near` with an anchor:

| relation | segment crosses a wall |
|----------|-----------------------:|
| closest  | 24% (506/2105)         |
| near     | 19% (74/380)           |
| farthest | 24% — **exempt**, the farthest object is *expected* to be across the room |

## Decision

User picked a **conservative gate** (vs leave-as-is / visualize-first). Scope:
`closest`/`near` only; `farthest` untouched.

Deviation from the option wording (flagged to the user): I **include glass
partitions**. The option said "solid walls only", but idx 12 — the motivating
case — *is* a glass wall, and glass walls are real navigation barriers (see
through, can't walk through). Excluding them would leave the exact complaint
unfixed; they're only ~55 cases.

## Fix

`wall_between_ok(sc, raw)` in `vla3d_ref_to_qa.py`:
- Structural walls = `raw_label` contains "wall", height `lz ≥ 1.2 m` (so flat
  "wall decal" / "wall lamp" / "light switch" fixtures are excluded), glass
  included.
- Clip the target→anchor xy segment against each wall's OBB footprint
  (`_seg_obb_interval`, Liang-Barsky).
- Reject only if the crossing overlaps the segment's **middle**
  (`t ∈ (0.08, 0.92)`). A crossing pinned to an endpoint is a wall-mounted
  anchor (picture / light switch / fire alarm at `t≈1`) or an object backed
  against its own wall (`t≈0`), not a divider strictly between the two.
- Wired into `build_pair` after the superlative-margin gate (drop reason
  `wall_between`). Applies to `closest`/`near` only.

## Result (5,055 pairs — unchanged, 0 type mismatches, 173 tests pass)

- `wall_between` dropped 3,891 candidate statements; clean single-layer supply
  still filled `TARGET_SINGLE = 3500`, so the corpus stayed at 5,055.
- idx 12's cabinet/fire-alarm pair is gone; **residual wall-crossing in the
  final closest/near set: 0/2,445**.
- `ruff` clean on new code.

## Caveat

Detection is noisier than the redundancy (TASK 10) / margin (TASK 12) gates:
it depends on VLA-3D wall geometry and the `lz ≥ 1.2` / `t`-window heuristics.
The conservative middle-crossing rule errs toward keeping borderline cases
(misses, not false drops), and is deliberately limited to `closest`/`near`.

## Files
- `dataset_generator/vla3d_ref_to_qa.py` — `wall_between_ok`,
  `_scene_walls`, `_seg_obb_interval`, `build_pair` hook.
- `dataset_generator/README.md` — "Wall-between gate" subsection.
