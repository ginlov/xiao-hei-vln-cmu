# TASK 36 — The leg boundary that erased the robot's momentum

`home_building_1` q2, reported from the simulator: *"`between` gets turned into
a waypoint and then the robot immediately turns round, goes back, and detours."*

The passage was not the problem. It is the leg after it.

## What the logs say

Five recorded runs of

> First, go to the nightstand with a clock on it, then take the path between
> the dining table and the picture, and stop at the trash can closest to the
> refridgerator.

All five drove the passage correctly, and all five crossed *between* the
anchors rather than round the end of them:

| run | offset along the gap axis at the crossing | half-span | |
|---|---|---|---|
| `exec_hb1_q2` | 0.02 m | 1.38 m | inside |
| `hm1_q2_v6b` | 0.33 m | 1.68 m | inside |
| `hm1_q2_reserve` | 0.19 m | 1.02 m | inside |
| `hm1_q2_v6` | 0.78 m | 1.46 m | inside |
| `exec_hb1_q2b` | 0.74 m | 1.62 m | inside |

The reversal is the *first exploration step of the destination leg that
follows*. The robot comes out of the gap at x ≈ −1.2 and then:

```
exec_hb1_q2     (−1.28,−5.43) → (+0.38,−5.79) → (+3.46,−6.34) → (+3.15,−6.31)
hm1_q2_v6b      (−1.28,−5.24) → (+0.38,−5.67) → (+5.19,−6.03) → … → (+7.29,−5.92)
hm1_q2_reserve  (−0.59,−5.56) → (+1.12,−5.38) → (+3.42,−3.70)
hm1_q2_v6       (−0.91,−5.55) → (−2.38,−0.60)      ← the one that did not
```

Three runs of four turn round on the first call and drive ~1.7 m back east,
straight through the gap. `hm1_q2_v6b` retreats all the way to (+7.29, −5.92),
the pose the passage leg had started from: the whole passage undone.

Under README §175 this is not merely wasted time. The score is on the
trajectory and on the *order* of the constraints in it, and a required passage
driven and then driven back out of reads as through, back, through.

## Why

**The leg boundary erased everything the robot had just done.** `spent` — the
departure memory added in TASK 35 to stop a leg circling — was built fresh
inside `run_goto`, so the direction the robot had arrived from carried no
penalty at all in the leg that followed. That direction is also the one bearing
guaranteed to have open floor behind it, which is exactly what
`explore_direction`'s reach gate rewards. Nothing in the prompt mentioned the
passage either: `mission` carried the plan and the cursor, never what had been
banked.

**And a latent one, found while reading for the first.** Inside `run_pass`,
"the far side" was recomputed from the live pose every step —
`through_point(sides[0], sides[1], o[:2])` and `far_side_goal`'s `s_veh`.
`bound_gap` froze the gap but not the side the leg entered from, so the instant
the vehicle was past the midpoint without `went_between` firing — the planner
rounds the end of the segment, or an anchor lifts half a metre out — the aim
flipped to the near side and the leg oscillated through the gap until its four
steps ran out. It never fired in these five runs because every crossing was
clean. It is one bad `picture` lift away from firing.

## What changed

**Geometry (A).**

- `Ctx` now owns `spent`, `crossed` and `done`. They are facts about the
  trajectory, not about a leg, and belong on the side of the boundary that
  `visited` and `avoid` were already on.
- `run_pass` freezes `entry` alongside `bound_gap` and hands it to
  `through_point` and `far_side_goal`. `far_side_goal` keeps the live pose too,
  for `settle` — two positions with two different jobs.
- A satisfied passage now banks three things for the legs after it: the gap
  itself in `ctx.crossed`, one departure pointing back at it from where the
  robot came out, and a sentence for the prompt.
- `recrosses` asks whether a candidate bearing would take the robot back out
  through a passage already driven, and `explore_direction` discounts it by
  `SPENT_PENALTY`. The side test is what stops this becoming a ban: a crossing
  that lands on the far side from the entry is the passage being *driven*.

**Prompt (B).** A `DONE_BLOCK` inside `MISSION_BLOCK`, listing what has been
banked and stating the rule: a satisfied constraint is spent, so when the
target is not in sight the way onward is not back the way it came — prefer an
opening not yet used, and treat the one the robot arrived through as the last
resort.

This is not a new prompt version. The rule is about state only the mission
block knows, so a v7 would carry a paragraph that is dead on every
single-destination run, and the version distinction would carry no information.
The block renders only when something has been banked, which leaves v5 and v6
byte-for-byte what they were — the condition the 117 cached replies depend on,
and the one `TestDoneBlock.test_nothing_banked_renders_nothing` guards.

## Measured

Replayed against the recorded terrain and the model's own recorded heading, on
the three steps that actually turned round:

| run | as it ran | with both fixes |
|---|---|---|
| `exec_hb1_q2` | +0°, re-crosses | −30°, does not |
| `hm1_q2_v6b` | +0°, re-crosses | −45°, does not |
| `hm1_q2_reserve` | +0°, re-crosses | +30°, does not |

Both halves are load-bearing. The departure seed alone fixes only
`hm1_q2_reserve`: the robot leaves the gap just 0.28 m past the line, so one
step of any size carries it out of the neighbourhood `already_tried`
recognises, while the gap stays where it is. The `crossed` penalty is what
holds after that.

The eastward component does not go to zero — it falls from +1.35/+1.69/+1.65 m
to +0.43/+0.77/+0.65 m. That is the right answer, not a partial one: the gap
line is diagonal, so "east" and "across it" are different questions, and
`recrosses` is the one that matters.

Freezing the entry changes nothing that ever worked. Across all seven recorded
two-sided PASS steps, `|live − frozen|` aim is 0.000 m.

555 tests pass, 23 of them new.

## A second reversal, found by driving the first fix

`runs/lr1_0811_02`, `living_room_1`: *"take the path between the sofa and the
round tables"*. Ran with the fixes above in place — the frozen aim held at
(−1.92, −2.24) for all four steps, no flip — and the passage still failed.

```
step4 (−0.04,+0.28) → drove to (−1.28,+0.13)
step5 (−1.35,+0.12) → drove to (−0.22,−0.62)
step6 (−0.15,−0.69) → drove to (−1.37,+0.04)
step7 (−1.37,+0.04) → drove to (+0.15,−0.52)
```

Shuttling between two poses 1.5 m apart, four steps, never closer than 1.37 m
to a gate 2.14 m wide.

`far_side_goal` answered every step, with the same corner about 2 m north-west
of the gate. Answering was taken as progress, and it is not:

| | far-side waypoint | motion on offer | |
|---|---|---|---|
| step 4 | (−1.63, +0.11) | 1.35 m | driven |
| step 5 | (−1.83, +0.13) | **0.20 m** | guard fires |
| step 6 | (−1.73, +0.02) | 1.45 m — back to step 4's rest pose | driven |
| step 7 | (−1.77, +0.13) | **0.15 m** | guard fires |

On the short ones the TASK 34 `MIN_VIEW_MOVE_M` guard replaced the goal with
`cm.best_waypoint_toward(aim, …)` — which has no side constraint and prefers a
near-side point, by `far_side_goal`'s own argument. Replaying it reproduces the
published waypoints exactly: (+0.04, −0.86) and (+0.04, −0.85).

`far_side_stalled` now names the two ways a far-side answer is not news — the
vehicle is already standing on it, or it settles where this leg has already
been — and both count toward `no_far` instead of resetting it. Replayed on the
recorded poses the leg ends at step 6 with the verdict *gap not drivable by the
stack*, one step earlier and 1.37 m from the gate on the entry side, rather
than being shuttled past it. The `MIN_VIEW_MOVE_M` guard is now reached only by
the fallback path it was written for: a far-side goal that survives
`far_side_stalled` moves the vehicle at least that far by construction.

### Is the gap simply too narrow?

Measured off the obstacle points, as the widest clearance anywhere on the line
between the two anchors:

| gap | span | widest clearance | corridor | |
|---|---|---|---|---|
| `living_room_1` sofa \| round table | 2.14 m | 0.32 m | ~0.63 m | failed |
| `home_building_1` table \| picture | 2.76 m | 0.90 m | ~1.81 m | passed |
| `home_building_1` table \| picture | 3.36 m | 0.85 m | ~1.70 m | passed |
| `home_building_1` table \| picture | 2.03 m | 0.87 m | ~1.74 m | passed |

The vehicle is ~0.60 m wide, so yes — this one is about one vehicle wide with
no margin, against nearly three for the three that worked, and 0.32 m is far
inside `obstacleDisThre` (0.75 m), so no waypoint can ever be placed in it.
But that is not what this run died of. It died 1.37 m away, shuttling.

Judging the corridor from the terrain on the first look — and skipping a gap
that cannot take the vehicle, instead of spending four steps proving it — would
have returned ~130 s to the destination after it. Not done: it introduces a new
"I think this is impassable" judgement calibrated on four samples, and the four
separate cleanly enough to be tempting and not enough to be trusted four days
out.

## Not fixed

- **The passage binds a different `picture` every run** — spans of 2.03, 2.76,
  2.91, 3.24 and 3.36 m across five runs, under four different names. The
  crossing still lands inside the span each time, so it has not cost a leg yet.
- **The trash can is a different trash can every run** — (+3.3, −10.9) in one,
  (−5.4, −1.4) in two others. The model never sees the refrigerator and the
  trash can in one frame, so "closest to the refridgerator" never constrains
  anything. This, not the reversal, is why leg 3 fails; it needs anchor-first
  binding and is a larger piece of work.
- The feature-over-noun false positive from TASK 35.
