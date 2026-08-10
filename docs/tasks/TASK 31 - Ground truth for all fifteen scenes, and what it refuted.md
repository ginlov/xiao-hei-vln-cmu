# TASK 31 — Ground truth for all fifteen scenes, and what it refuted

Half the official question set could not be scored offline, because
`viz/data/<scene>.json` existed for seven scenes out of fifteen. TASK 30 listed
fixing that as one of two next steps and priced it as a large job.

It was not a large job. The estimate was wrong in a way worth recording,
because the same mistake would have deferred it again.

## The export had two halves and only one was expensive

`export_viz.py` writes one manifest from two sources:

| half | source | cost |
|---|---|---|
| perception | a recorded tour under `frames/<scene>_tour` | hours of sim time per scene |
| ground truth | `object_list.txt` + `map.ply` inside the scene model | seconds, reads a zip |

They were welded together by a single early return: no tour, no export. So the
cheap half inherited the expensive half's price, and eight scenes had neither.
All fifteen scene models were already on the laptop the whole time.

Now a scene with no tour exports ground truth only. The eight ran in about a
minute, on the laptop, with the sim host idle:

| scene | gt objects | scene cloud |
|---|---|---|
| `home_building_1` | 432 | 9.5 M → 500 k |
| `home_building_2` | 227 | 5.1 M → 500 k |
| `hotel_room_1` | 86 | 7.6 M → 260 k |
| `hotel_room_2` | 95 | 4.9 M → 146 k |
| `livingroom_1` | 106 | 7.4 M → 262 k |
| `livingroom_2` | 88 | 7.4 M → 263 k |
| `livingroom_4` | 120 | 5.3 M → 168 k |
| `studio` | 73 | 6.7 M → 232 k |

Three other things had to change with it:

- **Several roots.** The models arrived as three download batches, two of whose
  directories have a space in the name, and the sim host keeps them unpacked
  instead. `--gt-root` now takes a list and reads a zip or an unpacked
  directory, rather than requiring a fourth copy that stays in sync with both.
- **A guard against self-harm.** The corpus lives on whichever machine recorded
  it, so running `--scene all` on the laptop finds no tours at all — and would
  have rewritten the seven good exports as ground-truth-only, destroying work
  that cannot be reproduced here. An existing export with frames is now kept
  unless `--force`.
- **The index is the union.** Exporting one scene used to rewrite `index.json`
  to that scene alone, hiding the rest of the page's data.

The viewer tolerates a scene with no tour: no scan, no detections, no robot,
the camera framed on `map.ply` instead of on a lidar cloud it does not have.
Checked in the browser on `studio` — 65 boxes and the scene cloud draw, and
`office_1` still replays frame-by-frame.

## What it confirmed

**Anchor size.** TASK 30 concluded that anchor size is loft's problem rather
than a general one, from 44 anchors over 7 scenes. The sample is now 92 over
15, and the distribution barely moves:

| subtends at start | 7 scenes | 15 scenes |
|---|---|---|
| 0–2° | 5 % | **3 %** |
| 2–5 ° | 11 % | **10 %** |
| > 5° | 84 % | **87 %** |

The conclusion survives doubling the sample. The new hard case is
`home_building_1`'s `clock`, 0.20 m at 8.8 m — 1.3°, smaller than loft's remote.

**How near is near.** `traj_tolerance.py` now measures 136 named-object
distances instead of 63: median 0.59 m against the 0.58 m TASK 30 quotes, 81 %
within a metre, and unnamed objects still sit at 1.95 m. The target we drive at
did not move.

**The questions are answerable against this ground truth.** 74 of the 75
questions name at least one object using the scene's own label. The single miss
is `chinese_room`'s "count the number of chairs with pillows on them" — plural
morphology against a `chair` label, not a missing object.

## What it refuted

The third keep-out question was the one without ground truth. It is
`livingroom_2` q5, and it breaks two things.

**The questions are not written against the scenes' vocabulary.** The sentence
says *"avoiding the path between the TV and the tea table"*. `livingroom_2` has
no `tea table`; it labels that furniture `coffee table`. `chinese_room`'s
question uses the same two words against a real `tea table` label. So an anchor
name cannot be looked up literally, in either direction.

**`KEEPOUT_M = 1.2` is refuted.** With the third case measurable, the bracket
`keepout_radius.py` reports no longer brackets anything:

| | | |
|---|---|---|
| lower bound | 0.86 m | our p90 centre error + robot half-width + slop |
| upper bound | **0.81 m** | tightest official clearance (`livingroom_2` q5) |

There is no single half-width that is both safe for us and legal for the
reference path.

Worse, the loop does not build a corridor at all — `bind_constraints` puts a
1.2 m **disc on each anchor object**. Measured against the official paths:

| anchor | official path passes | 1.2 m disc |
|---|---|---|
| `chinese_room` chair[2] | 0.59 m | forbids it |
| `chinese_room` chair[3] | 0.71 m | forbids it |
| `livingroom_2` coffee table[0] | 0.81 m | forbids it |
| `chinese_room` chair[0] | 1.27 m | ok |
| `chinese_room` chair[5] | 1.98 m | ok |

3 of 11 anchor instances. In `chinese_room` whether it fires depends on which
of six chairs the model binds; `livingroom_2` has exactly one coffee table and
would fail outright.

Measuring to the box surface rather than the centre does not rescue it — the
official paths pass 0.22 m, 0.29 m and 0.30 m from those boxes, roughly the
robot's own half-width.

**The model is wrong, not the number.** "Avoid the path between A and B"
forbids *crossing the gap*, and the reference paths brush right past A while
respecting the corridor. A disc on A forbids the wrong thing. `keepout_radius.py`
already measured the corridor geometry; `ConverterModel(keepout=[(xy, r)])`
implements discs. The measurement script and the implementation have disagreed
about what the constraint is since the constraint was built.

No radius was changed. Picking one is a design decision — corridor versus disc,
and what to do when no radius satisfies both bounds — and the keep-out path has
never worked end to end anyway (TASK 30: one live anchor, lifted onto the
ceiling). It is recorded here rather than patched.

## State

`viz/data/` holds all 15 scenes (gitignored; 160 MB). Changed:
`scripts/export_viz.py`, `scripts/keepout_radius.py`, `viz/app.js`. Uncommitted,
along with everything from TASK 30.

## Open

- **`gate`** — still the largest single item: 10 of 30 instruction questions,
  parsed but not enforced. Now measurable on all 15 scenes.
- **`avoid` as a corridor**, per above; 3 questions.
- Recorded tours for the 8 new scenes, if we want perception numbers there
  rather than ground truth alone. This *is* the expensive half.
- Repeat runs per scene, unchanged from TASK 30.
