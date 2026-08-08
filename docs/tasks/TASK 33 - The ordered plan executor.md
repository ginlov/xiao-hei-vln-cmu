# TASK 33 — The ordered plan executor

`approach_loop` drove to one object and stopped, which is the shape of exactly
one of the thirty official instruction questions. 27 of the other 29 name two
or more destinations that must be reached **in order**, and README §175 scores
the trajectory on whether it "follows the path constraints in the command and
in the correct order". This task builds the carrier: one question in, an
ordered drive out, the vehicle never reset between legs.

## Who decides what

| | holder | why |
|-|-|-|
| decomposition | the model, once, at step 0 | TASK 32 |
| progress cursor | the executor, monotonic | measured below |
| positions | geometry | unchanged from TASK 26–31 |

The model is shown **the whole question on every call**, plus the full plan
with the current step marked, plus the keep-outs — because a leg often cannot
be read alone: "the picture closest to the TV" needs the TV that an earlier leg
named. What it is never asked is *which step to do next*.

That split is not fastidiousness. `bind_target` records the model reporting
`same_object_as_previous: true`, at higher confidence, both when it had refined
a binding by 0.52 m and when it had jumped 4.37 m to a different object.
Progress state has the same shape as identity state — "is this still the thing
I was talking about" — and it is the one judgement the model has been measured
failing at, while the score's largest penalty is for getting the order wrong.

## What was built

**`scripts/execute_plan.py`** — decompose, then walk `steps(plan)` in order,
with `keepouts(plan)` held active throughout.

**`approach_loop.run_goto`** — the per-destination loop, extracted from
`main()` unchanged in behaviour. `main()` is now a thin wrapper that calls it
once, so the single-target CLI still works exactly as before.

**`approach_loop.Ctx`** — the state that outlives one clause. Getting a piece
of state on the wrong side of this line is a bug either way:

- carries: `visited` (places stood in), `avoid` (keep-outs), `calls`, `step`,
  the wall-clock deadline
- stays per-leg: `bound`, `prev_crop`, `misses` — carrying a binding into the
  next clause would aim the next leg at the previous leg's object

**`run_pass`** — a passage is not a destination. It has no standoff and no
binding to defend; the constraint is satisfied by the trajectory crossing the
gap. The gate point is the **midpoint of the two lifted anchors**, because we
publish a point and the organisers' `local_planner` picks the arc to it from
its own path library — so "between the TV and the bed" cannot be expressed as a
destination on the far side and a hope. Both anchors come from one call via the
v5 `gate` field.

Two guards on that midpoint, both tested: the widest reported pair defines the
gap, so a third object cannot shrink it; and two anchors closer than 0.5 m are
one object reported twice, not a gap, so the waypoint is refused rather than
placed inside the furniture.

**Budget** — `--budget 540` against README's 600 s, checked before each leg and
each step, and folded into every drive timeout. A leg that runs out is recorded
as not attempted rather than silently skipped.

**A failed leg does not end the run.** Scoring is per-constraint with partial
credit, so the destinations after a failure are still worth driving from
wherever the robot now stands.

## Keep-outs reach the prompt

`keepouts(plan)` are named in every call's prompt, with an instruction to
report their anchors under `avoid` *even when the current leg's phrase never
mentions them*. Without that the field only ever gets filled on the leg whose
own words mention the region — and by then the robot may have already driven
through it.

## State

- 456 tests pass (15 new), 1 skipped.
- `--plan-only` and the prompt rendering are verified locally.
- **Not yet driven.** Everything above is offline-verifiable and was verified
  offline; TASK 30 is the standing evidence that this is not the same as
  working. Next is two questions on the simulator: one GOTO-only, one with a
  passage.
- `KEEPOUT_M = 1.2` is still the refuted constant from TASK 31 — the keep-out
  is still a disc, not a corridor. Unchanged here deliberately: this task is
  the carrier, and 3 of 30 questions ride on it.
