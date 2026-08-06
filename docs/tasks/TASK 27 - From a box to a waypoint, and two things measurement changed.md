# TASK 27 — From a box to a waypoint, and two things measurement changed

TASK 26 ended with a grounding call that beats our full-coverage tour and a
list of what to do next. This builds the piece that turns its output into
motion, and reports two places where doing the measurement first changed the
design — including one item TASK 26 named as the obvious next step, which the
data refutes.

`scripts/vlm_approach.py` is the new module. `scripts/vlm_probe.py` gained two
optional prompt fields.

## The reframe: never bet on a range you do not have

TASK 26 measured the two channels a grounding call produces and found them
wildly unequal. Bearing is excellent — 0.03° on the hand-checked case, hits
localising to a median 0.11 m. Range is where the approach dies: 9 of the 15
misses are a correct box whose depth lift went wrong, and the model's own
`distance_m` is **worse than a constant** (median error 2.80 m against 2.36 m
for always answering 5.0; 5 of 9 chinese_room targets got the identical answer
`5.0` at true ranges from 2.17 m to 6.64 m).

So `next_waypoint()` commits to a destination only when the lift survives a
size check, and otherwise takes a bounded step along the bearing into space the
scan says is free, expecting to re-observe from there. Across the 52 claimed
sightings: **44 committed destinations, 8 uncommitted steps.**

The step is not a concession. Returns on a target grow as 1/r², so the regime
where depth fails is the regime the robot is about to leave. And a deferred hit
is not a lost hit — TASK 26's gate B *discarded* `loft / cup`; this steps 0.30 m
toward it and looks again.

Free distance comes from `free_range_along()`, which asks the scan how far the
robot can drive down that bearing inside its own width. That is a measurement
we already trust, not an estimate of where the target is.

## Refuted: per-class size priors

TASK 26 called replacing gate B's crude global band with the per-class priors
in `perception/size_prior.py` "the untried next step". Tried, on the same 54
rows:

| gate | hits | false positives |
|---|---|---|
| **A + global band [0.15, 4.0]** | **70.4 %** | **11.1 %** |
| A + per-class, tol 2.5 | 68.5 % | 11.1 % |
| A + per-class, tol 2.0 | 66.7 % | 11.1 % |
| A + per-class, tol 1.75 | 61.1 % | 9.3 % |

The global band dominates. Two structural reasons:

- **12 of 52 claimed phrases have no entry** — `folding screen`, `exit sign`,
  `tea table`, `elephant figurine` — and fall through to the band anyway.
- **Every false positive that survives is a wrong *instance*.** The horse
  figurine's box landed on the elephant; the chair chosen was on the balcony.
  Implied heights against their class priors: 0.38 vs 0.28, 1.92 vs 1.04, 1.08
  vs 0.83. The object picked is the same size as the object wanted, so no size
  test can separate them — the failure is semantic and size is the wrong
  instrument.

Kept behind `USE_CLASS_PRIOR = False` rather than deleted; the argument for it
is sound and only the data refutes it.

## "No lidar return" was never a sampling problem

TASK 26 attributed the empty cones to sampling density — ~976 points per
steradian, so a 2° cone expects under four returns. That is true and it is not
what happened. A 14° cone on `japanese_room / table` also returns nothing, and
at that width it should catch ~190 points.

The nearest return to that ray is **17.5° away**. Measuring return elevation
per azimuth bin over seven scenes shows why, and shows it repeating to within
1–3°:

| bearing | lowest elevation the scanner returns |
|---|---|
| 0° (ahead) | −32° |
| ±60° | −21° |
| ±90° | −15° |
| ±120° | −5° |
| ±180° (behind) | **+9°** |

**The lidar is tilted forward.** There is a large blind cone under and behind
the robot, and how far down it can see is a function of bearing. All four
no-return cases in the sweep point below this floor at their bearing — **4 for
4**, now reported as such by `in_blind_cone()`:

```
japanese_room table   bearing +112° at -19°, floor  -5°  → blind
office_1      table   bearing -145° at -13°, floor  +5°  → blind
loft          cabinet bearing +133° at  -6°, floor  +2°  → blind
arabic_room   plant   bearing +151° at  -4°, floor  +6°  → blind
```

No cone width recovers these. What does is that **the cone is body-fixed and
rotates with the robot**: driving one leg toward the target puts it near 0°
azimuth where the floor is −32°. The step-and-re-observe fallback fixes these
rather than merely deferring them, which is a stronger claim than the one it
was designed on.

Blindness is used only to explain an absent lift, never to reject a present
one: `arabic_room / stool` and `chinese_room / tea table` point marginally below
the floor, still got a lift from the upper part of their boxes, and are both
hits.

## Triangulation, which the approach loop gets for free

Two views bought by stepping give a baseline. `triangulate()` takes the two
bearings and the odometry, and needs neither lidar nor a model estimate.

The estimator itself is checked against synthetic ground truth — exact bearings
recover the range to 4e-15 m, and driving straight at the target gives parallax
0.000° and is correctly refused. What that cannot supply is the noise level,
so the bearing error was measured on the 52 claimed sightings instead, as the
angle between the box's ray and the direction to the nearest ground-truth
instance:

| | median | p90 |
|---|---|---|
| all claimed sightings | 0.66° | 19.2° |
| **hits only** | **0.48°** | **2.52°** |

The tail belongs to the wrong-instance failures, not to imprecise pointing.
Propagating the hits' figures through a 1.81 m baseline at a 5 m target:

| bearing noise | range error, median | p90 |
|---|---|---|
| 0.48° (measured median) | **0.13 m** | 0.32 m |
| 2.52° (measured p90) | 0.70 m | 1.55 m |
| 20° (a wrong instance) | 3.40 m | 6.60 m |

Against 2.80 m for the model's own estimate and 3.72 m for the lift that lost
the folding screen, the first two rows are an order of magnitude better. The
third says what already limits everything else: get the instance wrong and no
estimator helps.

The catch is the degenerate case. The baseline must have a component across the
line of sight, and stepping straight down the bearing has none. Either offset
the approach slightly or accept the 1/r² improvement instead — `parallax_deg`
tells the caller which happened, and it must be checked before believing
`range_m`.

## Two prompt fields, wired to logging only

`build_prompt(..., approach=True)` appends `target_state`
(`far`/`approaching`/`adjacent`, judged from framing, explicitly not converted
to metres) and `same_object_as_previous`, which needs a crop of the previous
view — `crop_face()` supplies it and `ask_claude`/`ask_gemini` take a
`previous=` image.

`same_object_as_previous` is the field the loop actually needs. chinese_room
holds nine chairs; after a 2 m leg, "the chair" may be a different chair, and
geometry cannot notice. That is the one failure mode of an iterative approach
that only the VLM can catch.

Neither field decides anything. `target_state` is the qualitative form of a
question the model already answers badly in metres, so it gets logged and
compared against geometry before going near the controller — the discipline
`same_space` failed when it returned `true` on all 52 claimed sightings
including chairs behind glass.

The base prompt is untouched, so `artifacts/vlm_sweep_cache.json` stays valid
and re-running the sweep is still free.

## What none of this ran on

**Nothing here executed in the simulator.** Every number comes from replaying
`frames_first/` — one frame per scene, tick 24, the start pose, extracted from
seven recorded tours — through the new code, plus the 117 cached API replies
from TASK 26. Real sensor output, but recorded, and one frame of it.

That is enough for the two findings. The blind cone is a property of how the
scanner is mounted, measured across seven independent scenes and repeating to
1–3°; replay does not weaken it. The gate comparison uses real boxes, real
scans and real ground truth. `next_waypoint`'s 44/8 split is computed on real
inputs.

It is not enough for anything downstream of a decision:

- **The closed loop has never run.** Publish a waypoint, let the robot drive,
  re-image, re-ground — none of that has been exercised. Whether the local
  planner even reaches these waypoints is untested.
- **"Turning fixes the blind cone" is a deduction, not an observation.** The
  mount is rigid, so the cone rotates with the body — that part is certain.
  That the re-observation then *succeeds* was never checked, because doing so
  requires driving.
- **Neither new prompt field has been called**, so their evaluation is zero
  data rather than weak data.
- The standoff of 0.6 m is inherited from the reference trajectories, not from
  watching this controller stop anywhere.

## Open

- Run the loop in the simulator: one blind case (`japanese_room / table`) is
  the cheapest test, since it predicts a specific recovery.
- Evaluate `target_state` and `same_object_as_previous` against geometry.
- `COVERAGE_FLOOR_DEG` is hard-coded from seven scenes of this simulator. It
  is a sensor property and will not transfer to different hardware.
- Wrong-instance errors — 4 of 54, the residual after every geometric gate —
  remain unaddressed. Size cannot touch them by construction, and the 20° row
  in the triangulation table says bearing cannot either.
