# TASK 46 — The arrival the model said had not happened

`runs/o_2_0814_02` reports 3/3. All three legs stopped at a binding the model
independently put 2.8x, 3.0x and 3.6x further away, and one of them stopped
6.5 m short of a door on the far side of a glass partition.


## Leg 3, in full

Target: *"the door near the exit sign"*.

```
step  7  binding (5.78,-0.75)  1.64 m from vehicle   model: 5.0 m, far
step  8  refined (5.63,-0.95)  1.72 m                model: 4.5 m, approaching
step  9  lift rejected                               model: 6.5 m, far, behind_glass
step 10  back at step 7's pose -> revisited fires
         not closing, binding 1.81 m <= CIRCLE_ARRIVE_M 2.5  -> ARRIVED
         model on that step: 6.5 m, far
```

The scanner cannot see this door. Every ray toward it returns **one tight
cluster and nothing beyond**:

```
step  7  door  22 returns  1.62-1.67 m   single cluster
step  8  door  25 returns  1.46-1.59 m   single cluster
step  8  sign  22 returns  1.64-1.79 m   single cluster
step 10  door  34 returns  1.63-1.70 m   single cluster
```

Glass returns the beam. No cluster-selection rule reaches a surface that
produced no returns, so TASK 43's nearest-credible-cluster work is irrelevant
here — this is not a choice between readings, it is the absence of one.

A 1 cm coincidence supplied the binding that ended the leg. The box being
lifted on steps 6, 8 and 9 was `feature_box_2d`, which for this phrase is the
**exit sign** — a 3-5° box that TASK 43's range gate correctly keeps, because
the sign really is at the glass's range. `size_gate` then measured it against
the target's expectations:

```
step 6  h=3.1°  lift 2.49 m  implied 0.14 m  REJECTED
step 8  h=5.1°  lift 1.72 m  implied 0.15 m  ACCEPTED   <- SIZE_BAND floor is 0.15
step 9  h=4.2°  lift 1.86 m  implied 0.14 m  REJECTED
```

Applying the *target's* size prior to the *feature's* box is a pre-existing
bug that this task does not fix; it is recorded here because the acceptance it
produced is what `circled back` later read as a ring.


## The rule that looked right and is not

The obvious fix is to refuse a commit when the lift and `distance_m` disagree
by more than the factor of three the prompt says the estimate exists to catch.
Replayed on this run it vetoes **all three legs**, and two of them are right:

```
leg 1  binding 1.05 m  model 3.0 m (2.8x)   10° cone: 0 returns near 3.0 m
leg 2  binding 1.16 m  model 3.5 m (3.0x)   10° cone: 0 returns near 3.5 m
leg 3  binding 1.81 m  model 6.5 m (3.6x)   glass
```

`distance_m` has a median error of 2.80 m — worse than answering with a
constant — so a rule built on its metres inherits that. Do not build one.


## `target_state` separates them

It is the same question asked qualitatively, and the prompt forbids deriving it
from the metres: *"judged only from how the target sits in the frame — how much
of the view it fills, whether the frame cuts it off, whether you can make out
surface detail"*. It needs the model to get "small in the view" right, which is
a judgement about the picture in front of it, and it is the only signal at an
arrival that is independent of the scanner.

On this run legs 1 and 2 read `approaching`; leg 3 read `far`.

Over the 39 recorded arrivals that carry a reply, 8 (21%) were declared on a
`far`, spread across all three arrival paths, with the model putting the target
3.6-8.0 m out. None of the 8 had the model also saying ≤2 m, so none is
self-contradictory.


## What the veto is worth, checked one by one

For each refused arrival, whether the scan holds returns near the model's
figure — the check that decides whether the model or the scanner was right:

| run | model | returns near it | verdict |
|---|---|---|---|
| `jr_0812_04` s4 | 7.5 m | 163 @ 4.5 m | correct — binding was 0.25 m out |
| `jr_0812_01` s4 | 6.0 m | 160 @ 4.7 m | correct — binding was 0.30 m out |
| `studio_d2` s2 | 4.5 m | 190 @ 3.2/5.6 m | correct |
| `lr_2_0811_06` s8 | 8.0 m | none at all, `behind_glass` | correct — binding was 9.79 m |
| `lr1_0811_03` s3 | 3.6 m | none at all, `see_through` | correct |
| `o_2_0814_02` s10 | 6.5 m | none — glass, established above | correct |
| `cr7_bind` s1 | 4.5 m | one cluster at 2.2 m | **possible false veto** |
| `l3002` s2 | 4.0 m | one cluster at 1.2 m | **possible false veto** |

Six hold up, two do not. Both doubtful ones declared arrival on step 1 and
step 2 — legs that finished almost immediately, where `far` is a first-glance
judgement with no second look behind it. Requiring `far` on two consecutive
steps would drop both, and would be a threshold tuned on n=8; it is not done.

The cost of a false veto is bounded: the leg keeps looking under its existing
`leg_deadline` and `max_steps`, and if it runs out it returns a truthful
`ok: false`. A failed leg does not end the question — the score is
per-constraint with partial credit and later legs still drive.


## Changed

- `scripts/approach_loop.py` — `says_far`; the veto at all four arrival paths
  (`circled back`, `within standoff`, the predicted converter stop, and the
  post-drive stack clamp); `rec["far_veto"]` records when it fires.
- `Ctx.note_settings` writes the run's configuration as the first line of
  `steps.jsonl`. Asked whether `runs/o_1_0814_04` was faster than the runs
  before it, the only way to tell what it had been run with was whether `here`
  appeared in the replies. Wall clock without settings is not a measurement.


## Not fixed

- `size_gate` judging a feature box against the target's size prior.
- Nothing here has been driven. The veto changes what 21% of recorded arrivals
  would have done; whether the legs it releases then reach their targets is a
  sim question.
- The two doubtful vetoes above.
