# TASK 12 — Drop tied superlatives (closest/farthest must be visibly unique)

## Problem

Inspecting `vla3d_ref.jsonl` idx 1:

```
Find the file nearest to the big plant.
  TARGET     file id=5   (-1.34, 1.34, 0.63)
  distractor file id=54  (-1.34, 1.34, 0.60)   # same (x,y), 3 cm higher
  anchor     plant id=67 (5.67, 1.23)
  -> both files are 7.011 m from the plant (identical xy distance)
```

Two `file` objects are **stacked** at the same horizontal position, so they are
exactly equidistant from any anchor — "the file nearest the plant" has **no
determinate answer**. VLA-3D still tagged one as the answer (it ranks by 3D
distance / arbitrary tiebreak), but for the ground robot (and the marker-IoU
score) the xy-plane is what matters, and the two tie.

Single-layer `closest`/`farthest`/`near` statements come from VLA-3D verbatim
and had **no margin guard** (only the nested generator enforced
`CLOSEST_MARGIN_M = 0.3`). Measured over 2,839 single-layer superlatives with
≥2 candidates:

| target↔2nd-best candidate xy gap | share |
|----------------------------------|------:|
| <5 cm (tied / stacked)           | 13.9% |
| 5–15 cm                          | 10.6% |
| 15–30 cm                         | 10.0% |
| 30–50 cm                         | 11.0% |
| ≥50 cm (clearly separable)       | 54.5% |

~14% are essentially tied; ~35% are within 30 cm. These are unanswerable
supervision, and the robot's low-res 360 camera could not resolve them anyway.

## Fix

`superlative_margin_ok(sc, raw)` in `vla3d_ref_to_qa.py`: for
`closest`/`near`/`farthest`, rank the target + same-class distractors by xy
distance to the anchor and keep the pair only if the target is THE nearest (or
farthest, for `farthest`) by `SUPERLATIVE_MARGIN_M = 0.30` m. Non-superlative
relations, and cases with <2 candidates / a missing anchor, pass through.
Wired into `build_pair` after the redundancy gate (drop reason
`superlative_tie`). Threshold chosen to match the nested generator (user
picked 0.3 m over a looser 0.15 m).

Side benefit: it also drops samples where VLA-3D's answer disagrees with the
xy-geometry (VLA-3D ranks by 3D distance), keeping geometry self-consistent for
a ground robot.

## Result (5,055 pairs — unchanged, 0 type mismatches, 173 tests pass)

- `superlative_tie` dropped 13,478 *candidate* statements in `build_pair`; the
  single-layer pool still had enough clean supply to fill `TARGET_SINGLE = 3500`,
  so the corpus stayed at 5,055.
- Final single-layer superlatives within 0.30 m: **14% → 1.4%** (39/2,785;
  residual = multi-anchor / float edge cases). idx 1's file/plant tie is gone.
- Relation-word mix essentially unchanged (closest 39%, near 10%, farthest 10%,
  between 11%, on 12%).
- `ruff` clean on new code (added `import math`; `zip(..., strict=True)`).

## Files
- `dataset_generator/vla3d_ref_to_qa.py` — `superlative_margin_ok`,
  `SUPERLATIVE_MARGIN_M`, `build_pair` hook, `import math`.
- `dataset_generator/README.md` — "Superlative margin gate" subsection.
