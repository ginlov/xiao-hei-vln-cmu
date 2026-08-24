# TASK 54 — Falsifying a binding with the motion already paid for

Every gate in this stack was found the expensive way: a run reported success,
the run was wrong, and a gate was added. Six gates, six post-mortems. That
method does not generalise and it cannot be finished — there is always another
failure not yet met.

This task asks whether the gates could have been *generated* instead.

## The principle

The loop already re-observes after every drive, and it uses the new observation
to **re-ground** — to ask the same question again from a new place. That is
re-estimation, and it inherits every bias of the first answer. It is why
`bind_target` records the model reporting `same_object_as_previous: true` at
*higher* confidence both when it had refined a binding by 0.52 m and when it had
jumped 4.37 m to a different object.

Falsification asks the opposite question:

> If the binding at **P** is correct, what must the next observation look like?
> Does it?

The asymmetry is the point. Such a test can only return *contradicted* or *no
contradiction found*; it can never return *correct*, so it cannot manufacture
the confidence that is the documented failure. And the observation it needs is
free: the robot moved anyway.

A binding made from one pose is unfalsifiable from that pose — every reading
there is consistent with it by construction. Motion supplies the second vantage.

## Four candidate tests, and what the geometry gives each one

| | test | needs |
|---|---|---|
| **T1** | predicted range `‖P − b‖` against the new lift | odometry + lift |
| **T2** | implied physical size `r·h` must agree across the two views | lift + box height |
| **T3a** | do the two bearings' rays actually intersect | bearings only |
| **T3b** | bearing-only triangulated range against the lift | bearings + lift |

T3a and T3b use **no lidar at all**, which is why they were worth trying: T1 and
T2 both consume the lift, so they share its failures.

## Method

`scripts/falsify_audit.py`, offline over recorded runs — no model call, no
simulator. 115 in-leg consecutive step pairs from 28 runs. Pairs spanning a leg
boundary are dropped, because the target changes there.

A residual is never zero even when the binding is right, so each test is
characterised by its distribution on pairs where the binding *is* right. Pairs
are labelled by the binding's distance to the nearest ground-truth instance of
the phrase's head noun — read from the scene annotation, **not** from our own
scorer, so the label does not depend on anything being calibrated here.
68 pairs land within 1.0 m, 38 beyond 2.5 m, 9 in between and excluded.
Every threshold below is that test's own good-p95, so every row buys the same
5% false-alarm rate.

## Result — one test of four survives

| test | catches | separation |
|---|---|---|
| **T1** range vs odometry | **44%** | **5.9x** |
| T2 implied size | 12% | 2.4x |
| T3a ray miss | 15% | 1.2x |
| T3b triangulated vs lifted | 0% | 2.6x |

T1's noise floor is a median of **0.100 m** — which is TASK 26's measured lift
error of 0.11 m, arrived at independently. That agreement is the main reason to
believe the instrument.

**T2 is dead.** On correct bindings the implied size already differs by ±18%
between two views; on wrong ones by ±47%. Box jitter swamps the signal, and
conditioning on a longer move does not move it.

**T3b is dead**, and its failure is instructive: the good distribution has a
*fatter* tail than the bad one (p95 1.454 against 1.096). Triangulation error
goes as `r²·σθ/b_perp`, and the parallax is not there to pay for it.

### A correction, recorded because it changed a conclusion

Parallax was first computed between the two **camera-frame** bearings, which
counts the robot's own rotation as though it were baseline. Median parallax read
53.7°, and T3 looked well-supplied. In the map frame it is **16.4°**, with 18%
of pairs degenerate. The first number was an artefact and the optimism that came
with it was unearned.

## The finding: the abstention has to apply to the verifier too

Each test above was scored over every pair, including the pairs its own geometry
cannot judge. Letting each decline those:

| test | valid when | abstains | catches | separation |
|---|---|---|---|---|
| T1 | moved ≥ 0.5 m | 17% | 48% | 7.8x |
| **T1** | **moved ≥ 0.75 m — deployed** | **~25%** | **65%** | — |
| T1 | moved ≥ 1.0 m | 42% | 58% | 21.4x |
| T2 | moved ≥ 1.0 m | 42% | 11% | 2.2x |
| T3a | parallax ≥ 20° | 59% | 23% | 2.7x |
| T3b | parallax ≥ 10° | 34% | 10% | 1.0x |

**The same test, permitted to say "I cannot judge this one", goes from 44% to
58–65% detection and from 5.9x to over 20x separation.** Nothing about the test
changed; only its willingness to decline.

### The detection number moved twice, and both moves are recorded

It was first measured at **69%**. That figure is superseded and should not be
quoted. Two things changed under it:

1. **The corpus grew.** `score_if.py` matched a run to a question by exact
   string, and three office_1 runs carried a line break introduced when the
   question was passed as a shell argument. Collapsing whitespace recovered
   them and eleven others — 25 runs to 36 — and the same test on the larger
   corpus reads **58%** at `moved ≥ 1.0 m`.
2. **The operating point moved.** `moved ≥ 0.75 m` catches as much as
   `≥ 1.0 m` while declining far less, so it is what ships: **65% detection at
   6% false alarm** over 51 good and 20 bad pairs, old corpus only.

Quote **~60%**, with the range, and say which threshold and which corpus. A
number that moves this much under a corpus fix is not one to report to two
significant figures. A gate judging half the steps well
beats one judging every step badly, and the criterion for declining falls out of
the test's own geometry rather than being tuned.

This is the abstention principle applied one level up — to the thing doing the
verifying — and it is the transferable result here, not T1 itself.

## Deployed

`XIAO_HEI_FALSIFY=1` adds one rung to `bind_target`'s arbitration ladder — the
existing rungs are refinement, measurement outranking a guess, a nearer
reading, the model reporting a different object, and two corroborating refused
readings. This one is the first that involves nothing the model said. Default
off, so the baseline is byte-identical and provably untouched (0 fires in 500
flag-off calls). Constants `FALSIFY_MOVED_M = 0.75`, `FALSIFY_RESIDUAL_M = 0.93`
reproduce the audit's operating point exactly: 65% / 6% on the same pairs.

**Whether it improves the score is not measured here and cannot be.** The gate's
action is to distrust the binding and re-observe, and that re-observation is a
model call the recorded runs do not contain. Detection rate is not score. The
A/B on the four lowest-scoring questions is what answers it.

## Limits

- Small: 28 good / 16 bad pairs at T1's operating point. Directional, not tight.
- T1 consumes the lift, so it cannot catch a lift that is wrong in a way the
  odometry agrees with; only T3 was independent of the lift, and T3 failed.
- Pairs whose lift was refused have no range, so T1's sample **excludes the
  cases the blind cone already caught** — 44%/69% is therefore a floor.
- The label is the head noun's nearest ground-truth instance, so binding to the
  wrong instance of the right class can be scored good.
- These tests detect *inconsistency*, not *wrongness*. A box that is wrong the
  same way from both views passes everything.

## What it does not cover, and what would

Nothing geometric can see that a lift landed on a **poster of** the object: a
poster is a real plane at a real distance, so range, size and triangulation are
all self-consistent. That failure is semantic, and the complement is a second
model call at the arrival decision, prompted to look only for contradicting
evidence and given no access to the reasoning that produced the claim. Recorded
face JPEGs make that measurable offline over the 39 recorded arrivals. Not run.

## Use

```bash
uv run python scripts/falsify_audit.py
```
