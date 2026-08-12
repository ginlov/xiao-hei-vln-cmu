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

567 tests pass, 12 new.

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
