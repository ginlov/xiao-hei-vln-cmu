# TASK 58 — Drive the object-reference responder, and fix what the robot found

*2026-08-24. Six drives across four scenes, on `xiaohei1`.*

Object reference had been built and tested but never driven. Driving it broke it
open: **the orbit had never made a single successful model call**, and four of
the seven defects below could not have been found by any test, because they are
properties of the vehicle and the sensor rather than of the code.

## What the drives found

| # | defect | how it showed up | fix |
|---|---|---|---|
| 1 | Every crop-carrying re-look raised `IndexError` | `view 1: not found (call_failed)` on the first drive | the crop goes through `ask_claude(previous=)`, not as a fifth image |
| 2 | `run_goto` returned `xy=None` on all 17 failure paths | `no binding` → `nowhere to orbit around` → `NO BOX`, while the binding it discarded was **0.02 m** from the truth | `Outcome.bound_xy`, carried by a `_out()` closure so a new failure path cannot forget it |
| 3 | `stand_off` only ever backed off | two drives parked 1.06 m and 2.60 m short; at 2.60 m the model confused two identical potted plants in five views of eight | it now closes in as well, outside a 0.4 m band |
| 4 | The model names a face the binding contradicts | `whole`, confident, zero returns | reject when the face is more than one quadrant from the binding's |
| 5 | Sparse views drag the box | views within 0.2 m of truth carried 15–162 returns; those missing by 0.6–4.7 m carried 3–9 | `MIN_TAKE_RETURNS = 12` |
| 6 | The orbit spends calls on an arc where the object is hidden | four consecutive `hidden` on a free arc, a cabinet in the way | stop after three consecutive misses (corpus drives keep going, so the stop stays replayable) |
| 7 | **`Pose2D.theta` was published as a hard `0.0`** | the vehicle parked facing map-east; on the back face, 4 of 4 views lifted **zero** returns against 17 of 17 elsewhere | the field is now settable — **and the stack ignores it**, so the last hop before each view is driven straight at the target instead |

Defect 7 is the one worth remembering. `/way_point_with_heading` carries a
heading and nothing had ever set it. Setting it correctly changed nothing:
asked to hold −121°, the vehicle arrived at +37°, which is the direction it had
been travelling. Heading follows travel on this stack, so `aim_at` spends the
last 0.5 m driving at the object. Measured on the same scene and question,
three times:

| | zero-return views | views into the box |
|---|---|---|
| baseline | 4 / 8 | 4 |
| `theta` set correctly | 4 / 8 | 4 |
| **last hop aimed at the target** | **2 / 8** | **6** |

Every view with the target within ±66° of the heading lifted 28–62 returns.
Both that still lifted nothing were beyond ±139°.

## What the orbit is for, measured

Three independent drives agree that looking again is worth it. On the loft
plant, one view gives IoU 0.111 and five give 0.358. On chinese_room three
views agree to within 0.11 m. Centre error is no longer the binding
constraint — 0.13–0.14 m on three of four drives, against a one-point budget
of 0.27–0.48 m. **Extent is.**

So the estimator was swept through the real `TargetBox`, varying one thing at
a time (the earlier hand-rolled sweep bypassed `consensus` and its numbers
were wrong — they are not repeated here):

| `size_mode` | trim | points |
|---|---|---|
| `average` | 2% | 3 / 8 |
| `average` | 0 | 3 / 8 |
| `max` | 0 | 4 / 8 |
| **`union`** | **0** | **5 / 8** |

Both knobs point the same way and for the same reason: a view sees the face of
the object turned towards it, so averaging averages slices and shrinks the box,
and on 28–162 points a 2% trim removes the edges that *are* the extent.
`ObjectMap` keeps its own 2% — its cloud is the union of every observation in a
run, where one stray frame would set a corner permanently.

**Caveats, because four objects is not many.** The loft column is scored
against the plant the views actually saw (#109), not the question's target
(#110) — an estimator measurement, not a scoreable result. chinese_room prefers
`average` (0.313 against 0.277), both worth one point. Every cell is replayable
from the recordings; re-run the sweep when more drives land.

## The other measurement: an orbit is not a circle

`scripts/orbit_arc.py` — for every released question, the azimuths on a ring
where a 0.35 m robot fits, against the VLA-3D boxes in its height band:

| radius | longest contiguous arc, median | ≥180° | ≤90° |
|---|---|---|---|
| 1.2 m | 110° | 4/30 | 14/30 |
| 1.5 m | 120° | 1/30 | 11/30 |
| 1.8 m | **130°** | **1/30** | 8/30 |

The shipped orbit asks for 300°. One target in thirty can give 180°. The
numbers are optimistic — VLA-3D's `unknown` and L-shaped `wall` boxes cover
whole rooms and are dropped, so the real arc is smaller.

And standable is not visible: livingroom_3's vase sat against a cabinet, and
four consecutive positions on a free arc had it hidden.

## Files

`scripts/answer_reference.py`, `scripts/target_box.py`,
`scripts/reference_view.py`, `scripts/approach_loop.py`,
`scripts/answer_numerical.py`, `scripts/robot_io.py`,
`scripts/orbit_arc.py` (new), `scripts/score_reference.py`.
Tests: 953 pass.
