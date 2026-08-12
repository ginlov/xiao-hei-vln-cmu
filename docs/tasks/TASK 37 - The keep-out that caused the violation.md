# TASK 37 — The keep-out that caused the violation

`livingroom_2` q5: *"First, go to the chair near the window, then stop at the
soccer ball near the couch, **avoiding the path between the TV and the tea
table**."*

Both destinations were reached — the soccer ball binding landed 0.017 m from
ground truth — and the robot drove through the middle of the forbidden gap,
0.14 m from its midpoint. README §175 penalises a trajectory that "passes
through areas it is forbidden to go through in the command".

## The model was not the problem

It reported `avoid` on all twelve calls, naming both objects on ten of them —
including on the step that drove through:

```
s1  TV + tea table      s7   TV + tea table
s2  tv + tea table      s8   TV + tea table
...                     s10  TV + tea table   <- the violating step
```

The information was present, in the right form, at the moment it was needed.

## Where it was lost

`ctx.avoid` held **five** keep-out discs at step 10, not two:

```
TV                        (+2.41,-2.90)
tea table (coffee table)  (+0.41,-2.31)
tv                        (+2.91,-3.73)   <- the same television
TV                        (+2.74,-5.04)   <- the same television
tea table                 (+0.19,-2.29)   <- the same table
```

The lifts drift as the robot moves, `bind_constraints` matched anchors by
distance only, and `JUMP_M` (1.0 m) called each drifted lift a new object. Five
discs of radius `KEEPOUT_M` = 1.2 m then closed every route the leg had, so
`best_waypoint_toward` returned `None` — and **the fallback was `goal = wp.xy`:
publish the raw waypoint with the constraint dropped in silence.**

Replayed on the recorded frame:

| `ctx.avoid` at step 10 | published |
|---|---|
| none | (+1.51, −5.66) |
| the two correct discs | (+0.07, −1.54) — legal |
| **the five drifted discs (what ran)** | **`None` → raw (+2.62, −6.23)** |

So the keep-out did not merely fail to help. It manufactured the violation:
without it the leg would have driven somewhere legal.

Two discs at 1.2 m are no better — on the *lifted* positions they also return
`None`. `KEEPOUT_M` was already refuted in TASK 31 and never replaced, and this
is why it cannot be repaired by retuning: a disc pair wide enough to close a
2.1 m gap is wide enough to close the room around it.

## A keep-out is a corridor, so it is a segment

`ConverterModel` gains `gates` beside `keepout`. A gate is the segment joining
the two anchors, padded `GATE_PAD_M` = 0.6 m at each end, and it forbids
*crossing*, not standing — so it removes nothing from the legal set and is
enforced on the run from the vehicle to wherever a waypoint would settle.

On the same frame it rejects **19 of 928** legal points, exactly those beyond
the line, where the discs rejected every usable one.

The padding is not cosmetic. Anchors lift to whichever face the scanner saw, so
the segment joining them falls short of the furniture at both ends: computed on
the raw pair, a route that "clears" the gate misses the tea table's centre by
**0.03 m**, which is to say drives through it.

## What changed

- `ConverterModel(..., gates=[(a, b), ...])`, checked in
  `best_waypoint_toward` and in `reach_along`, so waypoint choice *and*
  exploration inherit it. `crosses_gate` is the exact orientation test, not a
  sampled one: a keep-out enforced only usually is worse than none.
- `bind_constraints` merges anchors **by name first**, distance second. Five
  discs become two. The model's naming ("TV", "tv", "tea table (coffee table)")
  is the more reliable half of the answer; `same_thing` already knew how to
  compare them and moved to `approach_loop` so both callers can reach it.
- `gates_from` builds the segment from the two furthest-apart, differently
  named anchors — the same rule `gate_point` uses for a passage.
- `Ctx.keepout_is_gate`, set by the executor from the plan: "between X and Y"
  forbids a corridor, "near the stool" forbids a place, and the plan already
  knows which.
- **The silent fallback is gone.** When nothing toward the aim clears the
  constraint, `nearest_allowed_step` takes the legal point nearest the aim
  whose *route from here* is allowed, and the step is recorded. Only when even
  that is empty does the raw waypoint go out, now with
  `constraint_violated: true` in the log.

## Measured

Replayed through the new code on the recorded step 10:

```
keep-out anchors      5  ->  2
gate                  (+2.98,-3.07) -- (-0.17,-2.15)
best_waypoint_toward  None  ->  answers
published             (+2.62,-6.23)  ->  (+0.07,-1.54)
route crosses the true forbidden gate?   True  ->  False
```

The true gate here is the ground-truth pair from `object_list.txt` — TV at
(2.470, −2.895), coffee table at (0.363, −2.929) — not the lifted one, so the
check is independent of the anchors the fix works from.

598 tests pass, 43 new.

## The same cause, a second symptom

Driven on `livingroom_2` q5 with the above in place, the trajectory stayed out
of the forbidden gap — and the leg stopped anyway, `boxed in (no legal move)`,
1.3 m north of it. Replayed on that frame: **721 legal points were both outside
the gate and more than half a metre away.** The constraint was not what stopped
it.

`best_waypoint_toward` ranks candidates by distance to the target and scans the
nearest `search` = 400. That is a pure optimisation until a keep-out exists,
and then it is a bug: when the target lies beyond the thing being avoided, the
candidates nearest the target are exactly the ones the keep-out rejects. All
400 are refused, the loop ends with nothing, and the caller reads "no legal
move" from a frame full of them.

Reproduced without a simulator — 900 legal points, 600 beyond a gate and 300 on
this side, target beyond it:

```
first reachable candidate ranks 600 of 900 by distance to the target
  search=400   -> None          <- reports boxed in
  search=1200  -> (-1.44,+1.02)
```

This is also the other half of the original violation: the five drifted discs
did not need to remove every legal point, only the four hundred nearest the
soccer ball, and `None` then fell through to publishing raw.

The fix is not a wider window, which would put a thousand settle simulations in
the inner loop. Candidates a constraint will reject are dropped *before*
ranking, by the cheap test on the published point, so the window covers four
hundred plausible candidates rather than four hundred doomed ones. An
unconstrained frame is untouched.

## Driven, and a third symptom: the anchors themselves

Re-run with all of the above (`runs/lr_2_0811_03`): the gate was built, from
two anchors, from the right names — and the vehicle drove through the forbidden
gap anyway, 0.12 m from its midpoint.

Not the logic. The **tea table lifted to (+0.65, −4.92)**; it is at
(+0.36, −2.93). Two metres out. The gate was therefore drawn from
(+2.79, −2.33) to (+0.27, −5.38), a diagonal across the wrong part of the room,
and the route did not cross *that* line while crossing the real one. The same
run's sofa lift on `livingroom_1` was 1.23 m out, and the crystal ball's first
lift was 11 m out. Small objects lift well (round table 0.02 m, soccer ball
0.017 m); large low furniture facing the robot does not, because the scanner
returns the one face it can see and we treat that face as the centre.

So the geometry is only ever as good as a coordinate we cannot trust for
exactly the objects keep-outs are anchored on.

### Asking the model which way round instead

The model does not need a coordinate to know which side to pass. It can see the
TV and the tea table and say "the clear floor left of the tea table" — the same
trick as v6's `way`, which turned "a heading that means through that door" into
a box that could be lifted.

`KEEPOUT_BLOCK` now asks for `detour`: the opening or stretch of floor to cross
*next*, boxed, nullable. `lift_way` generalises to `lift_boxed`, so `detour`
inherits the `WAY_MAX_M` cap that keeps a lift through a gap from landing in
the room beyond. The approach branch steers at the detour when there is one,
and `steer` is kept apart from `aim` so the arrival tests still measure against
the target — a detour is deliberately not it, and `may_stop` is false while one
is in force.

Two things this does not fix, and must not be read as fixing:

- **The model still cannot control the path.** It sees four images from one
  pose. `local_planner` chooses the route. So the detour only helps if each
  step is short enough that the straight line is a fair model of the arc:
  `KEEPOUT_STEP_M` = 2.0 m caps a step while a keep-out is in force, against
  the 4.83 m drive that produced the original violation.
- **It is a second opinion, not a replacement.** The geometry is the only part
  that sees the path at all.

`XIAO_HEI_GATES=0` disables the computed corridor, so the model-led detour can
be driven alone and the two compared. On by default.

## Driven again, and a fourth cause: the plan is not the path

`runs/lr_2_0811_05`, with the detour live. Both legs arrived, and the soccer
ball bound **0.043 m** from ground truth where the run before had bound it
3.42 m out — `binding_nearer` fired at step 7, replacing a 9.93 m binding with
a 3.88 m reading. The model answered `detour` on every constrained step and
pointed the right way each time ("clear floor between the tea table and the
sofa", "clear floor just inside the sliding-door").

And the vehicle went through the forbidden gap again, once, at (+1.28, −2.91).

Step 6 is the whole story:

```
published      (-0.02,-3.96)      planned move 2.50 m
predicted      (-0.02,-3.70)      straight down x = 0, crossing nothing
ACTUAL         (+1.37,-3.17)      1.49 m east, and through the middle of the gap
```

The constraint was checked against a path the robot did not take. Split by how
the drive ended, the converter model is fine when it completes and useless when
it stalls:

| | n | median error | max |
|---|---|---|---|
| `why=arrived` | 4 | 0.15 m | 0.22 m |
| `why=settled` | 3 | 1.27 m | 1.49 m |

`settle()` walks the snap fixed-point along a straight line; `local_planner`
curves round obstacles, and where the straight line does not fit, it goes
somewhere we did not model.

### Measured, not guessed

Over the 121 recorded drives that carry a track, the sideways stray from the
planned line:

| | |
|---|---|
| as a fraction of the move | p50 0.18, p90 0.51, **p95 0.60**, max 1.64 |
| move 0.3–1 m | median 0.16, max 0.51 |
| move 1–2 m | median 0.30, max 0.95 |
| move 3–9 m | median 0.49, **p90 2.10, max 2.89** |

Long moves are where it breaks. So two changes, both from that table:

- **`KEEPOUT_STEP_M` now caps the detour too.** It was applied only to the
  branch without one, which is exactly the branch step 6 did not take. Capping
  at 2.0 m takes the worst observed stray from 2.89 m to 0.95 m.
- **`crosses_gate` became a clearance test**, with the margin scaled by the
  length of the move — 0.60 of it, never under 0.5 m, straight off the p95.
  `gate_clearance` uses an exact segment-to-segment distance.

A margin can seal a route, and `livingroom_2`'s only legal way south is a strip
the reference threads 0.8 m from the tea table. So it is a preference: when
nothing clears the margin the filter retries at zero, which is worse and still
not a violation of the constraint as written.

Replayed on step 6: it now publishes a 0.30 m move with 1.62 m of clearance
against a 0.50 m margin, where the p90 stray for a move that short is 0.15 m.

## No violation, and two more things wrong

`runs/lr_2_0811_06`: **zero crossings**. The clearance test and the step cap
held. The leg then shuffled twice inside half a metre and returned
`arrived, circled back (9.79 m)` — with the binding 9.79 m away and the true
ball 6.37 m from where it stopped.

**The false arrival is the worse of the two.** `revisited` + a binding was read
as "the ring around the target has been walked, and this is the floor", which
is only a description of the walk if the target is in the middle of it.
`CIRCLE_ARRIVE_M` = 2.5 m now qualifies it: the platform will not park inside
`obstacleDisThre` of furniture and measured floors run 1.1–1.5 m to an object
centre, so a real ring fits and 9.79 m does not. Beyond it the leg reports
`circling N m short of the binding`, which is what it was.

**The shuffling is not a bug.** Replayed from that pose, every legal point more
than 0.5 m south sits at x ≈ +0.7…+1.5 — which *is* the forbidden corridor —
and the western strip the reference trajectory threads has **no legal points at
all**, because it is under `obstacleDisThre` from furniture on both sides. The
robot was correctly refusing the only way there was. Turning the margin off
changes nothing: with `margin=0` the same frame gives the same answer.

That is the `studio` finding again, on a keep-out instead of a passage: the
corridor is drivable and unwaypointable, and no amount of scoring gets round
`obstacleDisThre`.

Two things do help, and are done:

- **`past`** aims `DETOUR_BEYOND_M` = 1.0 m beyond the floor the model names,
  because that floor is by construction inside the inflation — "the clear floor
  between the tea table and the sofa" cannot hold a waypoint. Same answer
  `through_point` gives for a passage. On the recorded step it takes the move
  from 0.10 m to 0.30 m; it is not what was blocking this leg.
- **A step that is not aimed at the target must move.** A committed approach
  may settle where it stands — that is arrival — but a detour or a capped step
  may not, and two calls went on 0.10 m moves because it was allowed to. The
  `diverted` flag now governs both that and `may_stop`, so the two cannot
  disagree about what a step was for.

## Not fixed

- **Leg 1 spent five of its seven steps not binding a chair it could see the
  whole time.** The model alternated between a full relational answer
  (`relation: closest_to`, 2-3 candidates, a window anchor) and a bare one
  (`relation: null`, no candidates); on the bare calls `adrift` fires and the
  loop explores away instead of keeping what it has. A window is glass, so the
  scanner returns nothing from it and "closest to the window" may not be
  measurable at all — in which case falling back to the model's own nomination
  is better than discarding the answer.
- `KEEPOUT_M` still governs the one-landmark case ("avoid the area near X"),
  and is still the radius TASK 31 refuted.
- Only 3 of the 30 official instruction questions carry a keep-out; 10 carry a
  required passage, where the pass rate is much worse. That is the larger
  prize and is still open.
