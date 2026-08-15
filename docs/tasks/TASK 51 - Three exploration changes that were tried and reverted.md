# TASK 51 — Three exploration changes that were tried and reverted

**None of the code described here is in the branch.** It was written, run
against the simulator, measured, and rolled back. This is the record so nobody
re-derives it.

The shipped state is TASK 50's fixes, validated by the `exp1` sweep: 13 scenes,
frontier **2516 m²** total, +59% median coverage against the pre-fix baseline.
Three subsequent rounds of changes each found a genuine defect and **none beat
that number**.

## The runs

All sweeps are 13 scenes, both strategies, 480 s per scene. Frontier totals:

| sweep | code | frontier coverage |
|---|---|---|
| exp0 | pre-TASK-43 | — (baseline, +59% below exp1 at the median) |
| **exp1** | **TASK 50 — what this branch ships** | **2516 m²** |
| exp2 | + per-target deadlines, reachability, clearance | 2039 m² (−19%) |
| exp3 | + guard rewrite, cap fix (1 scene only) | worse again |

## 1. Distance-scaled waypoint deadlines — inert, then harmful

**The reasoning.** Waypoint reach rate was 24%. Success fell from 65% for
targets under 1 m to 4% beyond 5 m, and the platform moves at 0.15–0.17 m/s, so
the fixed 12 s deadline bought under 2 m while the median commanded target sat
2.55 m away. 81% of the sweep went into approaches that ended in a skip. The
deadline was made proportional to distance.

**What happened.** It changed nothing, because the deadline was never the
binding constraint — the supervisor's odometry guard fired first. In exp2,
`stuck_timeout` skips went 170 → 2 while `no_progress` went to 99% of all
skips, at the same median 12 s. Coverage fell 19%.

**Why it was wrong at the root.** Reach rate is not the objective. On
`home_building_2`, across three runs:

| | reach rate | coverage |
|---|---|---|
| exp1 | 7% | **617 m²** |
| exp2 | 30% | 306 m² |
| exp3 | 16% | 237 m² |

Reaching *more* waypoints went with covering *less* area. The robot maps while
**driving toward** a target, not on arrival — the LiDAR sweeps new floor the
whole way, so a waypoint that is never reached is not waste, it is a heading
that dragged the sensor across unexplored space. The short 12 s timeout keeps
flinging the robot at new far-off targets; it rarely arrives, and it covers the
most. The "81% wasted" figure that motivated all of this was never waste — it
was the mechanism.

**Do not lengthen `stuck_timeout_s`** on the intuition that the robot should be
given time to arrive. That was tried twice and lost coverage both times.

## 2. Odometry guard: "is it closing?" → "is it executing?" — correct, insufficient

The guard skipped a waypoint after 4 s of less than 0.10 m of *closing* on the
target, i.e. it demanded a sustained 0.025 m/s of net approach. Turning in
place, backing off an obstacle and rounding a corner all close zero distance
while the robot is working normally. exp1's short deadline hid this by ending
approaches first; once exp2 let them live longer, the guard began cutting
drivers — of its own skips, those where the robot had clearly been driving rose
from 5% to 35%.

Rewriting it to ask whether the platform is translating *or* rotating (from the
odometry quaternion) is the question the guard exists to answer, and is
genuinely more correct. It did not recover the coverage. Worth revisiting only
alongside a change that makes long approaches pay.

## 3. Reachability, centroid snapping, clearance, path-cost scoring — unvalidated

Real defects, real fixes, never validated at scale:

- The frontier selector scored straight-line distance to a cluster **centroid**,
  with no path check. A frontier 1 m away *through a wall* outranked a reachable
  one 3 m down the corridor, and the centroid of a C-shaped cluster lands in the
  hollow — often inside an obstacle. 59% of `no_progress` skips never saw a
  single nav message.
- Nothing modelled the robot's footprint, so a free cell flush against a wall
  was a legal goal the robot cannot occupy.

Fixing these is defensible on correctness alone, and exp3 confirmed the
selector then runs its intended path (`pick=score`, 26/26). But **the shipped
code tolerates these bugs and covers more**, and the fixes were never run across
13 scenes. One trap if anyone reinstates them: `max_waypoint_dist` was 1.5 m,
tuned for straight-line distance. Compared against *path* cost it excludes
essentially every cluster, the score branch stops running, and selection
silently degrades to greedy-nearest — which is what halved multi-room coverage
in exp2.

## What is actually limiting coverage

Not the waypoint logic. On the last run the reach rate (6%) and distance driven
(70 m) matched exp1 almost exactly, and coverage was still less than half —
same driving, same arrivals, half the new ground. That points at target
*selection*, and beyond that at things the logs cannot see:

- The robot repeatedly parks **1.5–2.8 m short** of its goal and stalls there.
- It spends the back half of a sweep retracing: on one run, a 5.6 × 3.1 m box
  and **7 m² gained in 240 s**, against 227 m² in the first half.
- Early targets push into space the map reports as free and the robot cannot
  enter — the signature of a LiDAR seeing through glass. Not fixable from here.

## Method notes for whoever picks this up

- **Coverage is `free_m2`** from the `MAP` heartbeat or the `DONE` line. Reach
  rate, waypoint counts and path length are all misleading proxies — every one
  of them moved the wrong way relative to coverage at some point above.
- **Compare within a scene, not across.** Cross-scene correlations are dominated
  by scene size.
- **The noise floor is unmeasured.** No configuration has ever been run twice.
  Every single-scene judgement in this document, including the ones that led to
  reverts, rests on n=1. Measuring run-to-run variance on a handful of scenes is
  probably worth more than the next algorithm change.
- **Pick scenes across the size range**, not just the hard ones: 9 of 13 are
  single rooms that both strategies saturate, so they tie and hide nothing but
  regressions. A reasonable 8: `home_building_1`, `home_building_2`,
  `livingroom_2`, `livingroom_3`, `office_1`, `studio`, `loft`, `hotel_room_2`.
- **Ask whether coverage still limits the score** before optimising it further.
  The challenge grades answers, not maps.
