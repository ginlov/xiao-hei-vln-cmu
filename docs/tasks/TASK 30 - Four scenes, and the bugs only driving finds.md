# TASK 30 — Four scenes, and the bugs only driving finds

TASK 29 built a model of `waypointConverter` and validated it on one scene. This
is what happened when the loop was rewritten around it and driven on four:
`japanese_room`, `chinese_room`, `office_1`, `loft`. Roughly fifteen runs.

Every defect below was already in the code before today. None of them showed up
in replay. Each needed a robot to move.

## Where it ended

| scene | target | before | after | notes |
|---|---|---|---|---|
| `office_1` | potted plant furthest from the projector screen | 0.89 m | **0.61 m** | platform floor is 0.62 m |
| `japanese_room` | lantern closest to the fan decoration | 1.06 m | **0.99 m** | corner; floor is ~1.06 m |
| `chinese_room` | tea table with the elephant figurine on it | 0.48 m | **0.80 m** | seen anywhere in 0.35–0.80 m |
| `loft` | cup near the TV remote | — | **fails** | see below |

Distances are vehicle centre to the object's ground-truth box, the same way
`traj_tolerance.py` measures the reference trajectories, whose median is 0.58 m.

**`chinese_room` is not a regression and not an improvement — it is a spread.**
Across the session it landed at 0.35, 0.47, 0.48, 0.52, 0.55 and 0.80 m. The
runs that scored best did so through the blind-cone branch wandering into a good
spot; binding took that branch over and with it the accidental luck. One or two
samples per scene cannot tell a policy change from that spread, and no
conclusion here rests on a single-run comparison.

## Seven bugs, in the order driving found them

**1. The waypoint policy optimised the wrong quantity.** It picked the legal
point nearest the target. But the vehicle stops `waypointXYRadius` short of its
goal *along the approach*, so of two goals equally far from the target, the one
nearer the vehicle settles further from it. On `office_1` two candidates sat
1.088 m and 1.085 m from the target; preferring the first because it was 0.37 m
nearer the vehicle ended the run at 0.89 m instead of 0.69 m. Now scored on
where the vehicle *settles*. An exact prune (`|goal−target| − 0.3` bounds the
best possible settle distance) plus resolving legality once per frame instead of
per query took that from 6 000 ms to **34 ms**.

**2. The stall test ignored rotation.** The platform turns to face its waypoint
before translating — measured: commanded bearing −142.2°, final yaw −146.3°. A
waypoint 138° behind the vehicle therefore produces seconds of pure rotation,
and a position-only stall test reads that as the stack refusing to move:
`settled, moved 0.0008 m`, while the next frame showed it had turned 56° and was
still turning. **This has been present since TASK 28**, so some of that task's
"as near as it allows" readings were the loop giving up mid-turn.

**3. Arrival still meant "reached our waypoint".** That was right when the
waypoint *was* the target minus a standoff. It became wrong the moment the
waypoint became "the legal point that settles nearest the target", which can sit
metres away: the loop drove to one 2.61 m from the tea table and reported
ARRIVED. Reaching a waypoint is now a better vantage point and nothing more.

**4. Termination asked whether the converter would move us,** not whether moving
would get us closer. The legal points around an object form a ring at roughly
equal distance from it, so there is always another one worth 1.4 m of driving
and 0.08 m of progress. The loop circled the tea table for six calls.

**5. …and the fix stopped too early.** Judged on gain alone, the first step of
`chinese_room` declared the platform's floor at 2.15 m, because a 5 m local
terrain map from the start pose has not seen the ground near the object. Driving
anywhere is still worth it there — it is what makes the map grow. Only within
`NEAR_M` does "cannot improve" mean "cannot get closer". `NEAR_M = 1.5` is
calibrated on three scenes, not measured.

**6. `max_tokens = 2048` truncated the replies.** Three runs died on
"unparseable reply", which reads as a model failure and was a budget one: v4 asks
the model to enumerate candidates, anchors and alternates, and a five-lantern
reply overruns 2048 and comes back with the JSON cut mid-object. TASK 26's
measured 460 output tokens was v3. Now 4096, `stop_reason == "max_tokens"`
raises rather than returning a bad reply, and the raw text is logged on any
parse failure.

**7. `sim.sh up` started a second simulation stack** when the requested scene was
already loaded: `docker/run` correctly does nothing, and the script then ran
`system_simulation.sh` again inside the same container. Two stacks fought over
`/way_point_with_heading` — visible only as doubled publisher counts in
preflight — and the drive timed out. `up` now always tears down first. One run
was discarded.

## Target binding, and why the gate is on distance

Fixing (3) let the loop reach a second grounding for the first time, which
exposed that it had no memory of what it had committed to. On `japanese_room`
step 1 bound the lantern 0.19 m from the truth and step 2 produced one 4.18 m
away; the loop drove to it.

The model cannot arbitrate this. It reported `same_object_as_previous: True` and
*higher* confidence in both the healthy case and the broken one:

| | error, step 1 → 2 | binding moved |
|---|---|---|
| `office_1` | 0.56 m → 0.05 m | 0.52 m |
| `japanese_room` | 0.19 m → 4.18 m | **4.37 m** |

So the gate is on how far the reading moved, not on identity. That is also what
makes it safe against the obvious objection — that binding early locks in an
early mistake. It does not block refinement, only teleportation. The failure it
accepts is a *grossly* wrong first sighting, 4/54 = 7.4 % in TASK 26; a jump back
is allowed only when the model itself reports a different object at higher
confidence, which is untested.

A binding also rescues a lift the blind cone made untrustworthy: the position
was measured once and `/state_estimation` carries it across the move. On
`chinese_room` the rejected blind lift read 1.38 m against 1.36 m from binding
plus odometry — 0.02 m apart.

## Path constraints: built, not demonstrated

`ConverterModel(terrain, keepout=[(xy, r)])`, with prompt v5 asking for `avoid`
and `gate` anchors in the same call as the target.

The geometry is unit-tested and behaves: the allowed set shrinks 836 → 167, a
goal inside a zone is refused, `reach_along` truncates at the near edge rather
than reading straight through, and — deliberately — `snap`/`settle` are
untouched, because they predict the organisers' node and it has never heard of
our constraint.

**It has not worked end to end.** Across six calls carrying an avoid clause the
model reported an anchor once, and that one lifted to (+4.09, −0.29) — 2.2 m
from the nearest real cabinet, with `ceiling` as the closest ground-truth object.
n = 1. The radius, 1.2 m, sits inside the 0.86–1.98 m bounds
`keepout_radius.py` measured.

*Refuted in TASK 31.* Those bounds came from two of the three keep-out
questions; exporting the third inverts them (upper 0.81 m, below the 0.86 m
lower bound), and a 1.2 m disc forbids the official reference path in two of
the three. The disc is also the wrong shape: the constraint is on the corridor
between two anchors, not on the anchors.

`gate` is parsed and logged, not enforced. It is worth more than `avoid`: ten of
the thirty instruction questions need a passage driven *through*, against three
that need a region avoided.

## Exploration, and what loft actually is

Two changes, neither sufficient.

**Reachability.** The explore heading was published as a fixed 0.8 m hop without
ever asking whether the vehicle could go that way. On `loft` the model asked
three times for a heading with **0.00 m of reach** while 5.16 m was available 30°
away, and each refusal cost a call. Legs now run as far as the legal set allows,
capped at 3 m, along the drivable direction closest to the one asked for.

*This corrects a claim made earlier in the session.* Routing exploration through
the converter model was blamed for throttling it; measured, the legal set
extended 5.16 m and the model's chosen heading was simply blocked.

**Memory.** Each reply now writes a `here` clause — its own words for where it is
standing and what it could search — and those are fed back on later calls. In
the model's language, not map coordinates, which it cannot act on.

It did not stop the wandering, because the robot never left a 2 m radius and all
six entries describe the same place from different angles.

### What loft is

The model was right the whole time, and an earlier note in this session claiming
it switched from reasoning about the coffee table to the dining table was wrong:
it named the dining table in **every step of every run**. There is a real cup on
that table — two, in fact:

| | position | on dining table #9 |
|---|---|---|
| `cup` #107 | (+1.57, +2.04) | yes |
| `coffee cup` #113 | (+0.57, +1.56) | yes |
| `cup` #108 — **the answer** | (+6.21, +0.68) | no, 5.64 m away |

The loop bound (+0.59, +1.51), which is **0.06 m from #113**. The recognition
was accurate. What it could not do was check "near the TV remote", because that
anchor is 0.05 × 0.20 × 0.02 m and returned zero liftable anchors on every call
of every run. So it did the reasonable thing: circled the table trying to get a
clear view of a tabletop that is hidden behind chair backs — and cannot be seen
anyway, from a camera 0.75 m up, standing the 0.75 m away that `obstacleDisThre`
enforces, at a cup whose base is 0.43 m.

## How common is that anchor?

`scripts/anchor_size.py` measures the angular size of every relation's anchor at
the start pose, over the seven scenes with exported ground truth.

*Superseded by TASK 31, which repeats this on all fifteen scenes: 92 anchors,
3 % / 10 % / 87 %. The conclusion below holds; the shares shift by a point or
two.*

| subtends at start | n | share | |
|---|---|---|---|
| 0–2° | 2 | 5 % | `tv remote` 1.79°, `sushi` 1.70° |
| 2–5° | 5 | 11 % | resolves on approach |
| > 5° | 37 | **84 %** | liftable from the start pose |

The thresholds are calibrated on observed outcomes, not chosen first: `tv remote`
1.8° failed on 100 % of calls; `fan decoration` 26° and `projector screen` 16°
both lifted, the latter to 0.02 m.

**So anchor size is not a general problem — it is loft's problem.** A two-stage
"find the anchor first" search would serve 5 % of relations and is not worth
building; `resolve_relation` already works on the 84 %.

Three caveats: this is measured from the start pose, so the 2–5 % band improves
by driving; it covers 7 of 15 scenes because the rest have no exported ground
truth; and the matcher is crude — 6 phrases matched nothing, and its first
version resolved "the TV remote" to the 1.02 m `tv`, deleting the very case the
script exists to count. That error mode is one-directional, so the true figure
is likely below 84 %.

## Corrected during the session

Four claims made here and then measured false. Recording them because the
pattern — an explanation that fits, asserted before it was checked — is the
recurring failure mode of this work, not any one bug.

| claim | what measurement said |
|---|---|
| "we left 0.42 m on the table at the japanese_room lantern" | the proxy was a z-band over the dataset cloud; the real terrain floor is 1.16 m and we reached 1.06 m |
| "routing exploration through the converter throttles it" | the legal set reached 5.16 m; the model's heading was blocked |
| "the model switched from the coffee table to the dining table" | it said dining table in every step of every run; a subordinate clause had been quoted as the conclusion |
| "loft fails because objects are not in the first frame" | all four failures were small-or-distant targets and ranging errors, none a visibility failure |

## State

Committed as `dad76b3` on `feature/perception-replay-harness` — 30 files, not
pushed. Everything after that commit (binding, constraints, exploration, v5,
`anchor_size.py`) is uncommitted.

## Open

- `gate`: ten questions, parsed but not enforced.
- Ground truth for the other eight official scenes; half the question set cannot
  currently be scored offline.
- Repeat runs per scene. Every comparison above rests on one or two samples
  against a spread that is 0.45 m wide on `chinese_room`.
- Multi-constraint instructions — *go near A, then take the path near B to C* —
  are not implemented; the loop drives to one object and stops.
- `NEAR_M`, `JUMP_M`, `KEEPOUT_M`, `MAX_EXPLORE_M` are calibrated or bounded,
  not fitted.
