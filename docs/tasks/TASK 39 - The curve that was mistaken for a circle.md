# TASK 39 — The curve that was mistaken for a circle

`runs/cr_0811_01` answered *"Go near the potted plant on the table and stop at
the painting near the TV"* and reached neither. Both legs ended the same way:

```
leg 1  circling 2.51 m short of the binding
leg 2  circling 2.83 m short of the binding
```

Two different faults, and only one of them is this branch. Checked against
`questions/chinese_room/trajectory_q4.ply`, the reference path the organisers
ship as the answer:

| leg | our binding | what the reference goes to | error |
|---|---|---|---|
| 2, the painting near the TV | `(-2.28, -3.64)` | `painting` id2 `(-2.27,-3.63)`; the path ends 0.64 m from it | **0.01 m** |
| 1, the potted plant on the table | `(-2.29, +1.88)` | `potted plant` id27 `(+6.20,-2.58)`, on `table` id77; the path turns round 1.06 m from it | **9.59 m** |

So leg 2 was right and stopped short, and leg 1 was in the wrong room. The
circling branch is the whole of leg 2's failure and none of leg 1's.

The first validation of leg 1 asked the wrong question. Matching a binding to
the *nearest* ground-truth object finds `potted plant` id41 at 0.38 m and looks
like a success; the question is which plant the phrase names. Three of the
scene's four potted plants stand on raised furniture, and the supports separate
them:

| plant | base z | stands on | footprint |
|---|---|---|---|
| id27 `(+6.20,-2.58)` | 0.77 | id77 **`table`** | 1.54 × 0.49 |
| id30 `(+2.72,-4.25)` | 0.83 | id79 `stool` | 0.36 × 0.36 |
| id41 `(-1.92,+1.82)` | 0.83 | id73 `stool` | 0.36 × 0.36 |

"the potted plant on the table" is id27, and we bound id41. See below.

Leg 2's target was reachable: filtering the step's own `/terrain_map` to floor
cells with at least 0.40 m of obstacle clearance — enough for a 0.5 m vehicle —
puts standable ground **0.45 m** from the painting, down a corridor 1.31 m wide
at its narrowest, which is the same corridor the reference path descends at
x ≈ −1.2.

## Both legs were killed on the step where they stood closest

(Leg 1's binding was wrong, but the branch did not know that and would have
fired the same way on a right one — which is what it did to leg 2.)

| | step | pose | distance to binding |
|---|---|---|---|
| leg 1 | 1 | `(+0.00,+0.00)` | 2.97 m |
| | 2 | `(+1.02,+1.04)` | 3.42 m |
| | **3** | `(-0.31,+0.35)` | **2.51 m — killed** |
| leg 2 | 4 | `(-0.31,+0.35)` | 4.44 m |
| | 5 | `(-0.93,-0.95)` | 3.00 m |
| | 6 | `(-1.36,-0.64)` | 3.13 m |
| | **7** | `(-1.21,-1.02)` | **2.83 m — killed** |

Leg 1 died on its third call, for passing 0.47 m from the pose it had started
at — `REVISIT_M` is 0.5 m, and `revisited` counts the leg's own starting stand.
Leg 2 died for passing 0.28 m from where it had stood two steps earlier. Each
had just posted its best approach of the leg.

## The test could not tell a curve from a cycle

`revisited` asks a question about shape: did the vehicle come back to ground it
had already covered? An approach round furniture is a curve, and a curve crosses
its own outbound ground — leg 2's route was an L down a corridor and then west,
because the straight line to the painting grazes a couch at 0.00 m clearance.
Shape does not separate the two cases. Whether the vehicle is getting closer
does.

`closing(gap, closest)` is that test: this step is nearer the binding than the
leg has ever been, by more than `PROGRESS_M`. The circling branch now runs only
when a return buys nothing, which is what circling means.

`closest` is kept **per binding**, not per leg. Leg 2 rebound twice while
approaching; measured against the 1.60 m it once stood from a reading it had
already discarded, its 2.83 m from the painting reads as retreat. When the
binding moves the record is about a different point, and carrying it across is
a category error.

The restart value is the distance the binding was made from, not infinity.
Infinity would let the step after any rebind count as progress, and on
`lr_2_0811_06` the binding jumped 5.5 m out on the step before the revisit —
which is exactly the run that reported `arrived, circled back (9.79 m)`.

## Replayed over every recorded run that reached this branch

Fourteen runs, replaying the recorded poses and bindings through the new test:

| run | before | after |
|---|---|---|
| `cr_0811_01` leg 1 | circling 2.51 m short | **survives** |
| `cr_0811_01` leg 2 | circling 2.83 m short | **survives** |
| `cr_0811_02` leg 1 | circling 2.87 m short | **survives** |
| `lr_2_0811_06` leg 2 | arrived, circled back (9.79 m) | **circling 9.79 m short** |
| `exec_studio7` leg 3 | circling 2.65 m short | unchanged (2.65 m from a 2.31 m best) |
| `hm2_v6_2` leg 2 | arrived, circled back (1.53 m) | unchanged (1.53 m from a 1.32 m best) |
| `studio_0811_01` leg 3 | arrived, circled back (1.57 m) | survives — closed 0.34 m, so it keeps going |
| `lr1_0811_02` s12, `lr1_0811_03` s10 | circling ~2.8 m short | unchanged — the `livingroom_1` gap |
| `exec_hb1_q2`, `hm1_q2_v6`, `lr_2_0811_02`, `lr1_0811_01` | loop, nothing bound | unchanged |

The `lr_2_0811_06` row is a second bug fixed by the same change: that run's false
arrival was scored as success. `studio_0811_01` is the one behaviour that
loosens — a leg that had just closed 0.34 m now spends one more call instead of
stopping, which is the trade this whole change makes.

Termination is unaffected: each surviving revisit costs at least `PROGRESS_M` of
the remaining distance, and the leg still ends on `max_steps` or its slice of
the budget.

618 tests pass, 8 new.

## What this does not fix: leg 1 was never looking in the right room

The fix rescues leg 2 and does nothing for leg 1, which with the fix in place
would drive contentedly to the wrong plant and report success. Worth stating
plainly, because it is the more expensive of the two faults.

The model bound on step 2, from the room it started in, with `confidence: 0.82`,
`relation: null` and `candidates: []` — "there is only one, nothing to compare".
The intended plant was 9.59 m away, behind the start, in a part of the house the
robot had not seen and never went to. The reference path spends its first 8 m
driving there.

Its own evidence names the discriminator and gets it wrong:

> A broad-leaved green plant in a white cylindrical pot standing on a dark
> round-topped wooden **side table** against the back wall

That is `stool` id73, 0.36 × 0.36 × 0.84. So a check on the support as the model
reports it would not have caught this — the model answered the support question
and answered it wrongly. The scene's own labels distinguish `stool`, `side
table` and `table`, and the phrase says `table`.

Two things are wrong and only one is cheap:

- `candidates: []` is a claim about the scene made from one viewpoint. On a
  phrase whose discriminator is a *kind of support* — "on the table", where the
  scene holds three plants on raised furniture — one room is not enough to
  conclude there is nothing to compare. Nothing in the loop distinguishes "I
  have looked and there is one" from "I can see one".
- The support is measurable and never measured. A stool is 0.36 m across and a
  table 1.5 m; the loop lifts the plant and never asks how wide the thing under
  it is. This is the same gap as the dice ornament in TASK 38, where the size
  prior was measuring accurately and deciding nothing.

Neither is attempted here. Left as the next thing to look at, and the more
valuable one: the circling fix converts a wrong stop into a right one on the
legs that already found their target, while this decides whether the leg is
aimed at the right object at all.

## What this also does not fix

Leg 2 step 6 asked for `(-1.44, -2.37)`, which is 0.36 m from the couch — inside
`obstacleDisThre` (0.75 m) — so no path in the platform's library was clear and
the drive stalled after 0.41 m of a requested 1.50 m. The aim cut the corner of
an L-shaped route. The loop recovers by asking again from where it stopped, so
this costs calls rather than the leg, but it is the same structural limit as
`studio`'s passage and `livingroom_2`'s bypass.

Leg 1 step 1 also went sideways: the plant sat at −9.9° elevation, below the
scanner's +3° floor, so the waypoint was `blind` and the converter — with only
164 legal points, one capture into the run — settled 2.14 m from the aim and
2.5 m off its bearing. `/terrain_map` is a live view with a 1.75 m memory
bubble, and on the first step of a run it barely exists.

## Driven: `runs/cr_0811_03`

Same question, same scene, with the fix in.

| | before, `cr_0811_01` | after, `cr_0811_03` |
|---|---|---|
| leg 2, robot to `painting` id2 | 2.82 m | **0.83 m** |
| leg 2, robot to the reference path's own stopping point | 2.93 m | **0.51 m** |
| leg 1, robot to `potted plant` id27 | 7.15 m | 8.36 m |

Leg 2 is the fix working end to end. Step 7 drove 2.86 m from `(-1.20,-0.38)`
to `(-1.40,-3.24)` — straight down the 1.31 m corridor the reference path
descends — and stopped 0.51 m from where the reference stops. The step that
killed it last time is the step it drove through.

Leg 1 went to the wrong plant, as expected, and got closer to it: 0.91 m from
`potted plant` id41, the one on the stool.

## The model corrects itself at close range and nothing is listening

Leg 1's `here` field across its five calls, on the support the phrase turns on:

| step | range | what it called the support |
|---|---|---|
| 1 | 5.0 m | "the potted plant on a round **side table**" |
| 3 | 2.5 m | "the round **side table** with white potted plant" |
| 4 | 1.8 m | "the white potted plant on its tall wooden **plant stand**" |
| 5 | 1.8 m | "the white potted plant on its tall dark **plant stand**" |

It stopped calling it a table the moment it got close enough to see, in its own
unprompted prose, with no hint about sizes or vocabulary. The binding was made
at step 2 from 3.42 m out and never revisited, because nothing in the loop reads
the support noun back against the phrase.

That is the whole of the leg-1 bug and it needs no prior and no noun table:
on `"the X on the Y"` — 28 of the 60 released `object_reference` +
`instruction_following` questions, 21 of which have more than one instance of X
so the support is the discriminator — a binding made from far away should not
stay committed once a nearer call names a different support.

Measured against the alternative that was considered and rejected: `table` is
0.40–2.89 m across the 15 scenes (median 1.26) and `stool` is 0.36–0.73
(median 0.39), so 8 of 26 tables are smaller than the largest stool and a stated
size rule is wrong about a third of the time. And size is the wrong axis anyway —
the supports that actually discriminate the released questions are `tv cabinet`
(×3), `cabinet` (×3), `sofa` (×2), `chair` (×2), `nightstand`, `ottoman`,
`shelf`, `display ledge`, `dressing table`. Those are categories, not sizes.
