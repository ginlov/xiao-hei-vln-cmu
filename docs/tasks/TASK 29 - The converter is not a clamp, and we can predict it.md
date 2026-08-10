# TASK 29 — The converter is not a clamp, and we can predict it

TASK 28 measured three runs stopping 0.65–0.74 m from the tea table's box,
noted that `obstacleDisThre` is 0.75, and concluded the explanation was closed:
the platform clamps our approach and no change to our code recovers the
difference. The first sentence is right about `chinese_room`. The conclusion
does not generalise, and on `japanese_room` it is wrong in a way that cost a
whole run.

## What actually happens to a waypoint

`waypointConverter.cpp` (pulled from the sim image; the same node runs on the
real robot in phase 2) does not clamp. Lines 196–245:

```cpp
kdtreeTravArea->radiusSearch(point /* the VEHICLE */, searchDisThre /* 5.0 */, ...);
for (candidate p in travArea) {
    dis2 = |p - waypoint| + vehicleDisWeight /* 0.5 */ * |p - vehicle|;
    if (dis2 < minDis) {
        traversable = (no obstacleArea point within obstacleDisThre /* 0.75 */ of p);
        if (traversable) { minInd = ind; minDis = dis2; }
    }
}
```

A waypoint inside the inflation is not moved *along our approach* until it is
legal. It is **discarded**, and a global re-minimisation picks a replacement
that owes nothing to the direction we meant. The `0.5·|p − vehicle|` term pulls
that replacement back toward the robot.

`travArea` and `obstacleArea` come from splitting `/terrain_map` at
`intensity < obstacleHeightThre = 0.05` — intensity being height above local
ground. `/terrain_map` is on the README's allowed-topic list.

On `japanese_room`, asking for a point 0.19 m from the lantern's box sent the
robot **1.08 m to the other side of it**, onto the return leg of the reference
trajectory. TASK 28's runs looked like clamping only because `chinese_room`'s
tea table stands in the open, where the global minimum happens to sit on the
approach line. Put the object in a corner and the same mechanism produces
something qualitatively different.

## Predicting it

`scripts/waypoint_converter_model.py` reimplements the decision. Every constant
is read off `waypoint_converter.launch`; nothing is fitted.

One subtlety only visible in the source: `poseHandler` re-snaps at 10 Hz, and
the score's vehicle term re-centres as the vehicle moves. The robot chases a
target that moves with it, so the resting place is the **fixed point** of that
iteration, not the first snap. `settle()` iterates it.

| what | predicted | actual | error |
|---|---|---|---|
| TASK 28's recorded run, waypoint (+1.10, −0.44) from (0, 0) | (+0.93, +0.59) | (+1.02, +0.64) | 0.104 m |
| the same drive replayed live today | (+0.93, +0.59) | (+1.01, +0.61) | 0.081 m |
| `runs/jp3_conv` step 2, "you will move 0.05 m" | 0.05 m | 0.057 m | — |

## What this says about the acceptance bar

TASK 26 derived 0.58 m — the median distance the reference trajectories keep
from the objects they name — and TASK 27 made it `STANDOFF_M = 0.6`. Both
treated it as a target we were falling short of. It is not reachable through
this interface at all:

- **42% of `trajectory_q5.ply` sits below `obstacleDisThre`**, its closest
  approach at 0.36 m clearance. The converter would reject half of the
  organisers' own demonstration.
- In the `/terrain_map` the robot can see from its start pose, the nearest
  legal standing point is **1.16 m** from the lantern's box.

The floor is not a global constant either. Floor cells exist right up to the
lantern — 97 of them within 0.5 m of the box — but the best clearance any of
them has is 0.53 m against a required 0.75. Wall, lantern and potted plant
inflate into each other and seal the corner:

| distance to the lantern box | floor cells | best clearance (need 0.75) |
|---|---|---|
| 0.00–0.50 m | 97 | 0.53 |
| 0.50–0.75 m | 43 | 0.60 |
| 0.75–1.00 m | 46 | 0.65 |
| 1.00–1.25 m | 42 | **0.81** ✓ |

**Driving closer makes it worse, not better.** Re-measured after one leg, with
3× the terrain observed, the nearest legal stand recedes from 1.16 m to 1.32 m:
newly seen ground is mostly newly seen obstacle. Where the robot already stands
(1.06 m) beats anything a further waypoint achieves (1.13 m). This question is
finished at 1.06 m, and that is a property of the platform.

## The loop, rewritten around it

Instead of publishing `range − STANDOFF_M` and discovering the result, the loop
now asks the model for the **legal point nearest the target** and publishes
that. A legal point is a local minimum of the converter's score — stepping from
it back toward the vehicle adds `d` and removes only `0.5 d` — so it is
republished as itself.

| | `jp2_v4` (TASK 28 policy) | `jp3_conv` (predict only) | `jp4_conv` (predict and aim) |
|---|---|---|---|
| drive legs | 2 | 2 | **1** |
| ended | stopped by hand | `did not arrive` | **`ARRIVED`** |
| final distance to box | 1.14 m | 1.06 m | 1.06 m |
| calls / cost | — | 2 / $0.05 | 2 / $0.05 |

The distance is the same because the corner is closed; what changed is that the
robot now goes where it asked to go in one leg, and the loop stops on a fact
about the terrain rather than on a stall timeout.

`jp4_conv` also reported `stack_said: null` — `/way_point_reached` never fired,
because the pose-only predicate from TASK 28 got there first. That is a third
piece of evidence that dropping the disallowed topic loses nothing.

## Two bugs found on the way

**Column 3 of `/terrain_map` is padding.** PCL pads `PointXYZI` to 32 bytes, so
intensity sits at offset 16 with three dead floats after it. Inferring columns
from `point_step // 4` reads a constant 1.0 and yields *zero* traversable
points — which the first run of the model duly reported as a confident, wrong
answer. `robot_io.py` now reads the declared field offsets, and
`ConverterModel` refuses to build on an empty `travArea` rather than
extrapolating from it.

**The predict-only version still drove when it knew better.** `jp3_conv` step 2
printed "will settle 0.05 m from here" and then drove anyway, because the
early stop was gated on `wp.committed` and that step was an uncommitted blind-
cone step. A prediction the loop declines to act on is just logging.

## Open

- One object, one scene. `chinese_room`'s 0.65–0.74 m results have not been
  re-derived through the model, and they are the case where the old policy
  looked fine.
- Whether "go near X" scores at 1.06 m is unknown; the README publishes no
  distance threshold, only "whether it follows the path constraints". If the
  platform floor for corner objects really is ~1 m, that is worth asking the
  organisers directly (the README invites GitHub issues labelled `question`).
- `settle()` models motion as a straight line. It matched twice within 0.11 m,
  but the local planner's actual path around an obstacle is not modelled, and a
  case where the two diverge has not been looked for.
- The vehicle coasted 0.17 m after `drive` returned in `jp4_conv`. Harmless
  here, unmodelled in general.
