# TASK 28 — The approach loop, driven

TASK 27 built `next_waypoint()` and validated it by replaying one recorded
frame per scene. Its own closing section said what that could not cover: the
closed loop had never run, and "turning fixes the blind cone" was a deduction
from a rigid sensor mount rather than an observation.

This is the loop, and four live runs of it on `chinese_room` against *go near
the tea table with the elephant figurine on it*. It works — and driving found
two things replay could not, one of which contradicts a decision TASK 27 made
deliberately.

## What was built

| file | role |
|---|---|
| `scripts/robot_io.py` | in-container ROS bridge — `capture`, `drive`, `preflight` |
| `scripts/approach_loop.py` | the loop; `--host` puts the docker calls over ssh |

The split exists so the API key stays on the laptop: the sim host runs only
ROS, and needs no key, no `uv` and no checkout of this branch. For the actual
submission everything has to run in-container; this is a development harness.

```
preflight → for each step: capture → 4 faces → VLM → box → bearing
          → blind check → lidar lift → size gate → waypoint → drive → repeat
```

## It runs

Best run, from 3.35 m away: **2 grounding calls, $0.05, arrived.**

```
[1] at (+0.09, +1.15)  conf 0.42  bearing -146°/-5° (BLIND, floor +5°)
      -> step 2.00 m along the bearing            reached, gap 0.13
[2] at (+0.10, -0.75)  conf 0.90  bearing  +50°/-14° (covered, floor -26°)
      state=approaching  same_object_as_previous=True
      -> DESTINATION, lift 1.53 m                 moved 0.11, gap 0.88
      stack will not close the last 0.88 m — as near as it allows
```

`preflight` earned its place immediately: it flagged a publisher on
`/way_point_with_heading` before any money was spent. That turned out to be
`rviz` — the manual Waypoint tool, which emits only on a human click — so the
check now names publishers rather than counting them.

## The blind cone, observed

TASK 27 predicted that driving one leg toward a blind target swings the bearing
toward 0° azimuth, where the elevation floor is −32° instead of +9°. That
happened in every run that started blind:

| run | before | after one leg |
|---|---|---|
| live3 | −120°, floor −2° → blind | +23°, floor −31° → covered |
| live4 | −146°, floor +5° → blind | +50°, floor −26° → covered |

The deduction is now an observation. It also means the step is not a
concession — it is the thing that makes the next lift sound.

## What driving contradicted

**A blind bearing does not merely lose the lift. It can return a confident
wrong one.** TASK 27 decided, explicitly, to use blindness only to explain an
absent lift and never to reject a present one, on the strength of two replayed
cases where blind lifts were still hits. The first live run found the
counterexample:

```
live3 step 2   bearing -120°, floor -2°   lift 4.70 m   true range ~2.9 m
               implied height 0.62 m — a plausible tea table, so the gate passed
               waypoint 4.1 m out; drove 1.42 m and was snapped 2.69 m short
```

The cone widens upward until it finds something, and what it finds is the wall
above the object. Nothing in the size test can catch that, because the wrong
range and a plausible object size are consistent.

`next_waypoint` now never commits a lift taken in a blind direction; it steps
instead. Re-checked against the 52 replayed sightings, this moves 44/8
committed/stepped to 40/12 — five previously-committed hits become steps. In a
static sweep that reads as a loss. In the loop it is a deferral: live4 stepped,
re-observed from a covered bearing, and committed on the next call, using one
call *fewer* than live3 did while overshooting by 2.69 m.

## The arrival predicate was wrong, and the number was already in the log

`/way_point_reached` answers "did you reach the waypoint", where *the waypoint*
is what `waypoint_converter` snapped ours to. The difference is published in
the same message, and the first run ignored it:

```
live1   asked for (-0.03, -1.55)   reached   dist_to_requested 0.69
        actually stopped 1.32 m from the tea table, and we called it ARRIVED
```

Driving to open space shows what a real arrival looks like: `dist_to_requested`
≈ 0.30, which is exactly `waypointXYRadius`. So 0.69 was a snap, not an
arrival. Arrival is now `gap ≤ 0.35` **or** `moved < 0.25` — the second being
the stack declining to go closer, which is the honest definition of as-near-as-
possible and needs no model to decide.

The VLM's `target_state` said `approaching` on that run, which was the accurate
description of a robot 1.32 m away. That is corroboration, not vindication: the
fix is still arithmetic on a number we already had.

## The platform clamps short of the acceptance bar

> **Superseded by TASK 29.** The converter does not clamp — it *discards* an
> illegal waypoint and re-minimises globally, which on `japanese_room` moved
> the robot 1.08 m to the far side of the target. These three runs look like
> clamping only because `chinese_room`'s tea table stands in the open. The
> 0.58 m bar below is also not reachable through this interface: 42% of the
> reference trajectory sits inside `obstacleDisThre`. Read TASK 29 for the
> corrected account and for the model that predicts the outcome to ~0.1 m.

`waypoint_converter.launch` sets `obstacleDisThre = 0.75` — the snap rejects any
traversable point within 0.75 m of an obstacle. **A 0.6 m standoff is below the
platform's floor and can never be honoured**, no matter how often it is asked
for.

Measured the way the acceptance bar itself is measured — `traj_tolerance.py`'s
`point_to_box_xy`, vehicle centre to the object's bounding box, not its centre:

| run | final pose | to centre | **to box** |
|---|---|---|---|
| live1 | (−0.25, −0.90) | 1.32 m | **0.65 m** |
| live3 | (+0.43, −0.81) | 1.46 m | **0.74 m** |
| live4 | (−0.01, −0.81) | 1.38 m | **0.73 m** |

All three sit against the 0.75 m wall, which closes the explanation. But the
bar from TASK 26 — the median distance the reference trajectories keep from the
objects they name — is **0.58 m to the box**. We stop ~0.12 m further out, and
no change to our code recovers it: `obstacleDisThre` is a parameter of the
organiser's stack, and the README scores the trajectory the robot actually
drove, so the converter cannot be bypassed.

Worse, 0.58 m is a *median*: half the reference trajectories are closer still.
Something reached places this waypoint path will not. Two candidates, neither
checked yet — the reference trajectories were not produced through
`/way_point_with_heading`, or terrain analysis dilates the tea table's obstacle
cells beyond its ground-truth box. This is the open question with the most
score attached to it.

## `/way_point_reached` is not ours to use

The loop's first arrival test read `/way_point_reached`, which the autonomy
stack publishes from `waypoint_converter/src/*.cpp:385`. It is real, it is on
the bus, and it is **not allowed at test time**. The README's System Outputs
table lists five topics — `/camera/image`, `/registered_scan`, `/sensor_scan`,
`/terrain_map(_ext)`, `/state_estimation` — and then says:

> While more topics may be available from the system, these are the only ones
> allowed to be used during test time.

This is not only a problem for this loop. `app/main.py` subscribes to it in two
places, and the comment there records a previous encounter with today's
clamping problem plus the wrong conclusion drawn from it:

> `/way_point_reached` … This is the sole advance trigger — distance to our
> commanded waypoint can't be used as a fallback because the autonomy stack
> often steers to a safer nearby point and never actually reaches our literal
> commanded XY.

The premise is right and the conclusion does not follow. A distance test alone
does hang, because the snap means we never reach our literal XY — but the
missing half is not a topic, it is noticing that the vehicle has *stopped*.
Both halves come from pose:

| condition | from | meaning |
|---|---|---|
| `gap ≤ 0.35 m` | `/state_estimation` + our own waypoint | got where we asked |
| no motion > 0.05 m for 4 s | `/state_estimation` | stack will not go closer |

Validated against the very signal it replaces, driving to two places:

| drive | our pose-only gap | `/way_point_reached` |
|---|---|---|
| open space | 0.3487 → `arrived` | `null` — ours fired first |
| into the tea table | 0.95631006 → `settled` | 0.95631009 |

The stack's payload *is* our computation, to seven decimals. `robot_io.py` now
decides from pose and reports the topic only as `stack_said`, never branching
on it. **`app/main.py` still needs the same change and that is
submission-blocking**, but it is a change to the shipped pipeline rather than
to this harness, so it is not made here.

## The two new fields, first data

| field | result |
|---|---|
| `same_object_as_previous` | `True` twice, across 1.4 m and 1.9 m moves with the view changing right→front and back→left |
| `target_state` | `far` at 3.35 m, `approaching` at 1.32–1.45 m, never `adjacent` — correct, since nothing got nearer than 1.32 m |

n=2. Both still gate nothing.

## Operational

Two 529s from the API during the session. `max_retries` is now 8 (SDK default
is 2, which a sustained overload outlasts), and the closing confirm call — pure
logging — no longer aborts a completed run when it fails.

## Open

- One phrase, one scene, four runs. No ordering constraints, no forbidden
  regions, no out-of-vocabulary target.
- The clamp means we have never observed `target_state == "adjacent"`, so the
  near-field regime is still untested.
- `explore` was exercised once and worked, but the heading it chose was not
  independently checked for correctness — only that the robot turned and the
  target then became visible.
- The loop terminates on the stack's clamp, which is right for "go near" and
  wrong for anything that needs the robot to pass *through* a place.
