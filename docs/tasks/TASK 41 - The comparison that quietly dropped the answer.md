# TASK 41 — The comparison that quietly dropped the answer

`runs/jr_0812_01`, *"Go to the lantern closest to the fan decoration, then take
the path near the wardrobe doors to the flowers on the display ledge"*.

| leg | reported | truth |
|---|---|---|
| 1, the lantern | `arrived` | **4.25 m from the right lantern** |
| 2, the wardrobe doors | `passed` | correct |
| 3, the flowers | `arrived, circled back (2.06 m)` | 1.61 m from `flowers` id6; the binding was 0.66 m from it |

`japanese_room` has three lanterns. `lantern` id61 at `(+1.40, -0.52)` is
**0.63 m** from `fan decoration` id62; the other two are 6.5 m away in the
tatami hall. The answer is id61.

## Not the grounding, and not the anchor

The anchor was right on every call:

| | true fan bearing | model's anchor | error |
|---|---|---|---|
| step 1 | az −45.6°, el +18.2° | az −43.3°, el +15.8° | 2.3° / 2.4° |
| step 2 | az +177.2°, el +7.0° | az +176.8°, el +7.5° | **0.4° / 0.5°** |

And at step 2 the model listed the correct lantern:

```
model  "black floor lantern beside the low stand under the fan"  az -174.6  el -8.3
truth   lantern id61                                             az -174.8  el -7.8
```

**0.2° and 0.5°.** The identification was not the problem.

## `resolve_relation` dropped it and declared a winner anyway

`boxes()` lifted each candidate to a map position and kept the ones that
lifted:

```python
xy = _lift_xy(px, i, scan_cam, pose)
if xy is not None:
    out.append(...)          # and if it is None, nothing is recorded
```

The floor lantern sits directly behind the robot at azimuth −174.6°, where the
scanner's elevation floor is +9.4°. At −8.3° it is 17.7° under it, so the lift
refused it. `len(cb) >= 2` still held over the other three, and the comparison
returned *"large slat lantern hanging from the coffered ceiling of the tatami
hall = 3.74 m"* — 3.98 m from the answer — with a rationale that reads as
authoritative in the log.

Fewer than two candidates was the only incompleteness the function checked for.
Fewer than *all* is the other one, and it is the dangerous one, because it
still produces a number.

Replaying every relational reply in leg 1 through the new code:

| step | candidates reported | compared | dropped |
|---|---|---|---|
| 1 | 4 | 4 | — (the answer was never listed) |
| **2** | 4 | 3 | **the answer** |
| 3 | 5 | 3 | two ledge lanterns |
| 4 | 5 | 3 | two ledge lanterns |

Across all recorded runs, 243 comparative replies:

| | |
|---|---|
| resolved and complete | 120 (49%) |
| **resolved but partial** | **20 (8%)** — previously indistinguishable from complete |
| unresolvable, falls back to the model's pick | 103 (42%) |

Median one candidate dropped, at most two.

## What changed

`resolve_relation` now returns a `Resolved` dataclass carrying `complete` and
`missed`, and still iterates as `(box, image_index, why)` so both existing call
sites destructure unchanged.

- **The reason string names what was left out** (`"; NOT compared: ..."`), so a
  log reader is not misled the way this one was.
- **The winner is demoted, not discarded.** `verified` is untouched — an
  incomplete comparison is still the best measured evidence there is, and
  refusing to bind on it would keep whatever wrong binding came before, which
  is what this run already did. What is withdrawn is `measured`, the licence
  that lets a reading overrule an earlier binding *at any distance*. A
  comparison missing a candidate has not earned that, because the candidate it
  could not lift is exactly the one that might have won.
- **The dropped candidate's bearing is kept and fed back.** A candidate the
  scanner cannot reach is usually behind the robot, and that is a direction to
  turn rather than a thing to forget. It goes into `ctx.visited` as a lead:
  *"a possible 'black floor lantern...' was seen at bearing −175° but could not
  be measured from here"*.

## Undecided is not unmeasurable

`runs/jr_0812_04`, the same lantern phrase, found the second half of the same
bug. `resolve_relation` returned `None` whenever fewer than two candidates
lifted, and the caller reads `None` as "unverified", which makes `bind_target`
**keep whatever binding it already has**.

At steps 3 and 4 the anchor lifted to **0.11 m** and **0.02 m** of the true fan
decoration, and exactly one candidate lifted — the right lantern, 0.57 m and
0.64 m from it. Both calls produced a committed waypoint **0.05 m and 0.04 m
from the truth**. Both were discarded for a binding carried from step 2 that
sat 3.93 m away, and the leg then reported `arrived, within standoff` because
the robot happened to stop 0.25 m from that wrong binding. The two candidates
that failed to lift were the tokonoma ledge lanterns, behind the robot in the
blind cone — the wrong ones.

One survivor out of several is an *undecided* comparison, not an unmeasurable
one, and it now returns a `Resolved` marked incomplete: the same demotion as a
partial comparison, because it is one. The anchor-never-lifted case is
untouched and still returns `None` — that is the `loft` counterexample the
branch was written for, where the phrase carries no evidence at all.

### Except on `farthest_from`, where the survivor argues against itself

The first driven run after this change, `runs/ar_0812_03`, was made worse by it,
and the reason generalises. A lift fails *because* the object is far, or
occluded, or under the scanner's floor — so "the only candidate that lifted" is
systematically the near one, and `farthest_from` is precisely the question whose
answer is the far one.

Leg 1 asked for *"the potted plant furthest from the hookah"*. One of two
candidates lifted; it was the nearest plant of five, **8.59 m** from the answer.
Binding it locked the leg out of the two steps that followed, whose committed
waypoints sat **0.37 m and 0.55 m** from the truth and were refused by `JUMP_M`
because a binding was already held. Before the change no binding was made and
those two steps were free to drive at the right plant.

`jr_0812_04`, the case the rescue was written for, is `closest_to` — where the
same bias points at the answer. So the rescue applies to `closest_to` and
`between` and returns `None` on `farthest_from`, which is what the code did
before.

Replayed through the real `bind_target` chain, the fix reaches the right
lantern by the route already in the code:

```
step 3  jump 4.49 m > JUMP_M  -> refused, recorded in `pending`
step 4  two readings within 1.0 m of each other and far from the binding
        -> corroborated -> re-bound (-1.83,+1.73) 3.93 m out
                        ->           (+1.93,-0.87) 0.64 m out
```

and the robot is then 4.1 m from its binding, so it keeps driving instead of
declaring arrival.

Regression over every leg with ground truth — 15 legs across `chinese_room`,
`japanese_room` and `livingroom_2`, replaying the whole binding chain both ways:

| | |
|---|---|
| better | **2** (`jr_0812_04` leg 1, 3.93 m → 0.64 m; `jr_0812_05` leg 1, 6.29 m → 0.04 m) |
| worse | **0** |
| unchanged | 14 |

`ar_0812_03` leg 1 never reaches a final binding under either version, so it
falls outside that table; checked separately, it is back to the pre-change
behaviour of not binding at all.

641 tests pass, 7 new — five of them replaying recorded replies against
recorded scans, so the fixtures cannot drift from what happened.

## This does not fix `jr_0812_01`

Stated plainly because the run is the reason the task exists. Step 3 found the
right lantern *and* the geometry picked it — *"small black slat lantern/basket
on the floor beside the low stand under the fan panel = 0.77 m"*, against a
ground truth of 0.63 m — and it was thrown away by a different test:

```
winner bearing   az +113.8°   el -4.38°
scanner floor there            -4.16°
                 -> 0.23° below -> blind -> lift not trusted -> binding carried
```

The lift that was refused would have landed at `(+1.68, −0.91)`, **0.48 m** from
the truth, against the 3.98 m the leg kept.

`COVERAGE_FLOOR_DEG` is not wrong — rebuilt in the sensor frame from 498 scans
across 74 runs it matches the table to within 1° at every bin. It is a 0.5th
percentile, so 0.5% of returns lie below it by construction, and the
scene-to-scene spread is about 2°. Treating it as a knife-edge is the bug. Of
69 blind rejections in the recorded runs the median sits 9.8° below the floor
and the deepest 37.3°, but **11 (16%) are within 3°, and 10 of those 11 had a
lift available**:

| below floor | run | |
|---|---|---|
| 0.23° | `jr_0812_01` s3 | this one |
| 0.23° | `cr5_conv` s3 | |
| 0.38° | `cr_0811_03` s6 | |
| 0.44° | `hm2_v6_2` s3 | |
| 0.77° | `lr_2_0811_07` s8 | |
| 2.5–2.8° | `lr1_0811_0{1,2,3}` s1 | the first step of three runs |

A margin is the obvious fix and is not made here. TASK 27's counterexample is
real — a target 17° into the blind cone lifted 4.70 m against a true 2.9 m
because the cone widened upward onto the wall above it — so the margin should
be chosen after computing the lift error for all eleven, not from this one
case.
