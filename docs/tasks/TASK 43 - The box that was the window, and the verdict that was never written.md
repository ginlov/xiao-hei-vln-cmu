# TASK 43 — The box that was the window, and the verdict that was never written

`runs/o_1_0814_02` has a complete-looking `steps.jsonl`: seven records, every
one carrying its model reply, its waypoint and its drive result. `plan.json`
reports the second leg as

```json
{"k": 2, "ok": false, "why": "boxed in (stack would not move us)", "xy": null}
```

Nothing in the log says so. That is two bugs — one that lost the verdict, and
one that caused it — and they are unrelated except that the first is why the
second went unnoticed.


## The verdict that was never written

`Ctx.record` serialised on call. How a step ended is set on the record *after*
the step decides:

```python
rec["drive"] = res
ctx.record(rec)                                   # line written here
...
return Outcome(False, "boxed in (stack would not move us)")   # verdict set after
```

Every terminal verdict — `arrived`, `stopped` — was therefore absent from
`steps.jsonl`, and a leg that failed logged exactly like a leg that finished.
`record` now holds the object and serialises it when the next one arrives, so
the mutations that say how the step ended are captured. Re-recording the same
object is a no-op rather than a second line, and `Ctx.close()` flushes what is
held. The cost is one step of durability: the window is the microseconds
between the verdict and the next call, against a step that spends tens of
seconds in the model and the drive.


## The box that was the window

The leg was asked for **"the water cooler near the window"**. At step 7 the
model returned, in the same reply:

| box | az | el | lift |
|---|---|---|---|
| `box_2d` `[148,148,553,249]` | +24.3° | −6.0° | **1.17 m** |
| `feature_box_2d` `[0,0,430,30]` | +48.6° | +14.4° | **1.77 m** |

`box_2d` frames the cooler dead centre. `feature_box_2d` frames the window
frame at the left edge of the same face. The loop preferred the feature box
unconditionally:

```python
box = to_pixels(reply.get("feature_box_2d") or reply["box_2d"], ...)
```

so it drove at the window. The model's own evidence names it — *"the dark frame
of the glazed window wall is at the immediate left edge of the same view"* —
and its own `distance_m` (1.0 m) and `target_state` (`adjacent`) both agree
with `box_2d`, not with the box the loop used.

`_RELATIONAL_BRANCH` already warns that returning the anchor is the commonest
way to get this wrong, and routes anchors to `anchors`. But it guards only the
comparative forms it names — *closest*, *farthest*, *between*. "near the
window" is not one of them, `relation` came back `null`, no candidate
comparison ran, and the anchor arrived in `feature_box_2d` instead.

### What that cost

The 0.60 m of extra range put the binding past the cooler and into the window,
1.77 m from the vehicle. `NEAR_M` is 1.5:

```
converter: best legal point closes 0.09 m — there is nothing nearer   ← correct
   ↓ here = 1.77 > NEAR_M = 1.5, so "not close enough to call this the floor"
   ↓ MIN_VIEW_MOVE_M = 0.5 forces a re-query for any point ≥0.5 m away
   ↓ every such point is ≥2.98 m from the aim; the only move is 1.3 m backwards
   ↓ the stack refuses to drive there (asked 1.40 m, moved 0.063 m)
   ↓ here > NEAR_M again → boxed in, not arrived
```

The converter had the right answer at the first branch and `NEAR_M` overruled
it by 0.27 m. With `box_2d` the range is 1.17 m, `here` is inside `NEAR_M`, and
the same branch returns `no legal point closer (predicted)` — an arrival.


## Containment is the obvious test and it is wrong

Over **222 recorded steps carrying both boxes, across 55 runs**, 84% put the
feature box outside the target box and 27% shared no pixels with it at all, a
median 16.7° apart. That looks conclusive until the images are drawn:

| step | `box_2d` | `feature_box_2d` | overlap | correct call |
|---|---|---|---|---|
| `cr6_bind` s4 | tea table | figurine **on** it | 0.27 | keep |
| `cr8_bind` s3 | tea table | figurine **on** it | 0.20 | keep |
| `hm1_q2_v6b` s6 | nightstand | clock **on** it | 0.05 | keep |
| `jr_0812_05` s9 | flowers | ledge beneath | 0.65 | drop |
| `cr_0811_03` s4 | potted plant | table beneath | 0.02 | drop |
| `o_1_0814_02` s7 | water cooler | **window** | 0.00 | drop |

A feature sitting *on* something legitimately pokes out above its box, so an
overlap gate throws away exactly the cases worth keeping. Angular separation
does not divide them either: the clock is 28° from its nightstand and the
table *under* the plant, which is an anchor, is 38°.

An overlap gate at 0.9 was built, measured, and discarded on this evidence. On
the steps it refused it was a coin flip — scored against the model's
independent `distance_m`, the target box was closer on 40 and the feature box
on 41 — and it produced seven arrival flips of which six were cases it should
not have touched.


## The range is the test

A feature on the target is at the target's range; an anchor across the room is
not. Over the 102 steps where both boxes lift:

```
figurine / tea table      0.26 m, 0.21 m      keep
clock / nightstand        0.17 m              keep
window / water cooler     0.50 m, 0.60 m      drop
sofa / guitar             0.47 m              drop
```

`aim_box` now lifts both boxes and refuses the feature box when they disagree
by more than `FEATURE_AGREE_M`. That is not a new threshold: it is
`dominant_cluster`'s `gap_m` of 0.35 m, already this codebase's definition of
when neighbouring returns stop belonging to one object. It costs one extra
`locate` on a scan already in memory, and it tests the property that actually
matters — whether the ray lands on the target — instead of a proxy for it.

Where only one box lifts, the target box wins: it is the one the model was
asked to draw round the target, and taking it costs a commit and buys a step,
which is what we want when the thing we would have committed to might be a
window.


## Replayed over every recorded run

490 steps, no API calls and no simulator — the replies are on disk.

```
no feature box (unchanged) : 293
feature box KEPT by gate   :  92
feature box DROPPED        : 105   (21% of steps, 48 legs across 39 runs)

on the 47 dropped steps where both boxes lift:
  |range old − new|   median 0.61 m   p90 1.90   max 5.60
  waypoint moved      median 0.82 m   p90 2.02   max 5.79
  vs the model's own distance_m:  new closer 29, old closer 18
                                  median |err| 1.58 m vs 2.26 m

arrival test flips 'too far' → 'as near as it allows' : 2
  o_1_0814_02 s7   1.77 → 1.17 m   (model: 1.0 m, adjacent)
commits withdrawn (commits to the feature box, now steps) : 21
```

The five labelled keep/drop cases above all come out right. The 29–18 split is
the difference between this gate and the containment one that preceded it,
which was 40–41.


## What this does not fix

- `distance_m` is the referee here and it is a weak one — measured earlier at
  2.03 m against a 0.48 m lift, with its stated interval containing the truth
  half the time. It is independent of the lidar, which is what makes it worth
  quoting, but it is not ground truth. The object-level GT mapping for the 16
  scored legs was built by hand in an earlier session and was not kept.
- The `NEAR_M` / `MIN_VIEW_MOVE_M` interaction is untouched. A committed
  approach whose converter says "nothing closer" is still overruled whenever it
  stands more than 1.5 m out, and is then forced into a ≥0.5 m move that can
  only go backwards. On `o_1_0814_02` the box fix keeps it out of that path;
  it does not close it.
- 21 withdrawn commits become extra steps, at ~40 s each against a 10-minute
  question budget.


## Changed

- `scripts/vlm_approach.py` — `box_range`, `aim_box`, `FEATURE_AGREE_M`.
- `scripts/approach_loop.py` — `Ctx.record` deferred, `Ctx.flush`, `Ctx.close`,
  `_pending`; the aim-box call site records why a feature box was refused.
- `scripts/execute_plan.py` — `ctx.close()` in place of `log.close()`.
