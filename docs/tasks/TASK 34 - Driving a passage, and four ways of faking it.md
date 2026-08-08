# TASK 34 — Driving a passage, and four ways of faking it

TASK 33 built the executor and drove it. `chinese_room` (two destinations)
scored 2/2 against ground truth — 0.07 m and 0.11 m from the named objects.
`studio` (destination, passage, destination) reported 3/3 and was **wrong**:
the passage had never been driven and the last destination bound the wrong
object. This is what came out of chasing that.

## The passage was a false positive four ways

`run_pass` declared the constraint satisfied when the vehicle came within
`GATE_REACHED_M` of the gap midpoint. Plotted against the recorded path, the
trajectory had **zero** intersections with the couch-table line: the vehicle
stopped 1.4 m north of the gap, turned, and went round the west end.

Four separate defects, each invisible in the log:

1. **Proximity was standing in for passage.** Being 0.93 m from a gap is not
   going through it, and README §175 scores the trajectory.
2. **Aiming at the midpoint parks the vehicle in front of the gap.** The stack
   settles on whichever side it approached from.
3. **The gap was re-derived every step and wandered.** One leg used three
   different pairs: `couch`+`table` 1.46 m apart, then `couch`+`coffee table`
   1.86 m apart somewhere else, then `couch`+`couch (left view)` — two views of
   one sofa, treated as a doorway through the middle of it.
4. **Legal-point selection always came back to the near side.** 424 legal
   points sat 1.4 m north of the gap and 4 sat 3.3 m south, so
   `best_waypoint_toward`, which minimises distance to the aim, picked north
   every time.

Also corrected: a conceptual error of mine. There is *never* a legal point
inside such a gap, and that is not a symptom. `obstacleDisThre` (0.75 m)
governs where a **waypoint** may be placed, not where the vehicle may drive.
The waypoint belongs on the far side; threading is `local_planner`'s job.

## The fixes

- `robot_io` records the driven track (`/state_estimation`, subsampled at
  0.10 m) and returns it. This is the thing the challenge scores and we were
  throwing it away.
- `went_between` — segment intersection against the anchor-anchor line.
- `through_point` — aim `THROUGH_M` past the gap, on the far side.
- `far_side_goal` — candidates restricted to the far half-plane, within a
  corridor of the doorway.
- The gap is bound on first resolution and defended, as a destination is.
- `same_thing` — `couch` vs `couch (left view)` is one object, not a doorway.

A regression test asserts that the **actual recorded `studio` path** does not
count as a passage.

## CORRECTION — the gap *is* drivable, and the fault was ours

**The section below is wrong and is kept only because the measurements in it
are real and the conclusion drawn from them was not.** With `far_side_goal` in
place, `studio` crossed the couch-table line at **x = +3.46**, inside the gap
(which spans x 2.60–4.47), on a 238-point recorded track. The passage
constraint is satisfiable with the shipped stack.

What misled me: both manual confirmations drove from the origin, and from
there the stack routes round the west end. That says the gap is not reachable
*on that approach*, not that it is not drivable. The run that succeeded came at
it from the vase, which is where the instruction sends the robot first — and
which is exactly the approach the organisers' reference trajectory takes.

So there is nothing to raise with the organisers, and the survey below is not a
list of impassable gaps. It still bounds how tight they get, which is why the
give-up rule stays: two failures to find a far-side legal point, from two
positions, still means move on and spend the time on the destinations.

## Superseded: "it still would not go through"

With all four fixed, `studio` still fails. Two independent manual drives
confirm the gap is not traversable by the shipped stack, the second published
from 5.14 m away — beyond `adjDisThre` — so `waypoint_converter` could not
have snapped it. The vehicle crossed the couch-table line at **x = −0.08**,
round the west end of a table that starts at x = 2.60.

Meanwhile the organisers' own `questions/studio/trajectory_q5.ply` **does**
thread it:

| | |
|-|-|
| crosses the couch-table centre line | yes |
| closest approach to the midpoint | 0.12 m |
| closest approach to the table centre | 0.77 m |
| gap, centre to centre | 1.86 m |
| net drivable corridor (terrain) | ~0.8 m — obstacle points at 0.37–0.51 m height either side |
| `local_planner` `vehicleWidth` | 0.5 m |

So the reference answer is not reproducible with the parameters shipped
alongside it. Worth raising with the organisers.

## Not a `studio` one-off

Centre-to-centre span of every two-sided passage in the released questions:

| scene | anchors | span |
|-|-|-|
| hotel_room_1 | the TV + the bed | 4.34 m |
| home_building_1 | the dining table + the picture | 3.03 m |
| hotel_room_2 | the TV cabinet + the bed | 2.55 m |
| hotel_room_2 | the bench + the bed | 2.27 m |
| home_building_2 | the sofa + the coffee table | 2.20 m |
| **studio** | **the couch + the table** | **1.86 m** |
| livingroom_1 | the sofa + the round tables | 1.55 m |

Three of seven are at or below the span proven undrivable. So `run_pass` now
gives up after failing to find a far-side legal point twice, from two
positions: the time spent proving a gap impassable belongs to the destinations
after it, which still score.

## State

- 479 tests pass, 1 skipped.
- `chinese_room` 2/2 verified against ground truth.
- `studio` destinations 1 and 3 still need work: destination 3 bound the wrong
  window, and `bind_target` **rejected a later reading 0.25 m from the correct
  one** for jumping 3.98 m from an earlier wrong binding. Same class as the
  guitar bug in TASK 30, opposite direction — an early wrong binding defended
  against a correct correction. Unfixed.
- `scripts/drive.sh` added: publish one waypoint by hand and print the whole
  driven track. This is how both manual confirmations were done.

## TASK 35 follows from this run

With the passage crossed, the same run exposed two more, both fixed:

**The binding arbitration refused a correct correction.** Leg 3 bound the wrong
window and then rejected two later readings 0.32 m and 0.82 m from the right
one. The model had said `same_object_as_previous: false` both times, and the
only thing blocking the re-bind was `conf > bound["conf"]` with both at 0.6 —
a strict comparison against a quantised value. Two changes: `>=` there, and a
`corroborated` rule that lets two consecutive refused readings which agree with
each other overrule the binding, independent of anything the model says about
identity. Re-driven, the binding landed **0.19 m** from the correct window
(was 3.22 m).

**The step cap bit before the time budget.** Leg 3 was still closing —
`settle_to_aim` 3.87 → 2.33 → 1.95 → 2.10 → 1.65 m — when it ran out of its
five calls, with 195 s of the 540 unspent. Each leg now gets an equal share of
the time still left, the last inheriting whatever is unspent, and the step
count is only a safety cap.
