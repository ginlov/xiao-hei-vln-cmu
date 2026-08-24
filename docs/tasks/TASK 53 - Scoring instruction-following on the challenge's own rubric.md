# TASK 53 — Scoring instruction-following on the challenge's own rubric

Every instruction-following result this team has reported was an anecdote:
`chinese_room` 2/2, `studio` 3/3 *and it was wrong*, `home_building_2` q1 2/2.
None of them is the number the challenge pays. README scores the **trajectory**
out of 6 — constraints achieved, in the right order, with penalties for
forbidden regions — and nothing here computed that.

`scripts/score_if.py` computes it, offline, over trajectories already recorded.
No simulator, no API, no new runs.

## The instrument had to be validated before it could be believed

The official `challenge_evaluation_node` is not public, so this is a proxy.
A proxy that cannot award the organisers' own reference answers full marks is
measuring its own thresholds, so `--reference` scores
`questions/<scene>/trajectory_q4.ply` and `q5.ply` with the identical code path.
That number moved four times, and every move was a bug in the scorer:

| | reference at 6/6 |
|---|---|
| first draft | 16 / 30 |
| + passages split by relation; order matched sequentially | 25 / 30 |
| + a gap may be crossed more than once | 28 / 30 |
| + keep-out anchors grounded exactly, or not at all | **29 / 30** |

**Mean reference score 5.97 / 6.** Every number below is conditional on that.

### Full marks on the reference is necessary and not sufficient

A scorer that returned 6.0 for every input would pass the test above perfectly.
The reference answers are all *good* trajectories, so they constrain the proxy
only where it should score high, and say nothing about whether it penalises a
wrong answer. `--negative` supplies the other half: each control corrupts a
reference answer in one known way, so the points it ought to lose are known
before it is scored.

| trajectory | mean /6 | at 6/6 | vs reference |
|---|---|---|---|
| reference | 5.97 | 97% | — |
| truncated to the first half | 3.13 | 10% | −2.83 |
| the other question's path, same scene | 3.50 | 30% | −2.47 |
| straight line, start to end | 4.10 | 33% | −1.87 |
| reference, reversed | 4.62 | 13% | −1.35 |

Reversal is the weakest separation, and running it is what found the last real
defect: **order was anchored to the first entry into the tolerance ball.** A
reference path is a tour and brushes past objects it has not visited yet, so
"first within tau" admits a monotonic assignment in almost any order — reversed
paths were scoring 6/6 with destinations **6.88 m apart**. Anchoring instead to
the *closest approach* — the visit, which is what "stop at the trash can" means
and what a reversal actually moves — took reversed paths from 4.93 to 4.62 and
from 30% at full marks to 13%, with the reference unchanged at 29/30.

The 13% that survive are genuine: `japanese_room` q4's two destinations are
**0.29 m** apart and `livingroom_4` q4's are 1.12 m, both inside `TAU`. When
destinations are closer together than the tolerance, their order is not
observable from a trajectory at all — by this proxy or by any other, the
organisers' included.

### What each fix was

**A passage is two different constraints wearing one word.** Of 13 passages in
the official set, 9 are `between` and **4 are `near`**. "Take the path between
the two columns" is a line to cross; "take the path near the TV" has no line and
no gap — a crossing test can never satisfy it. (`gate_point` in `execute_plan`
already had this right, with `sides=None` for the one-landmark case; only the
scorer had it wrong.) Also `between the two columns` is *one* anchor phrase
naming *two* instances of one label, not two anchors.

**Order is sequential, not first-touch.** Scoring each constraint at its first
approach anywhere on the track marks a destination as achieved when the robot
merely drove past it on the way somewhere else. Constraints are now matched
against a monotonic cursor, and one that can only be satisfied before the
cursor is recorded achieved-but-out-of-order, worth `ORDER_CREDIT` of its share.

**A gap may be crossed more than once, and the in-order crossing is the one
that counts.** `arabic_room` q5 crosses its column line at track index 91 and
again at 517; the destinations bracketing it sit at 253 and 728, so only 517
satisfies the instruction. Returning the first crossing failed the reference.

**A keep-out must not be grounded by widening.** `livingroom_2` q5 forbids "the
path between the TV and the tea table". That scene's annotation contains no
`tea table`; falling back to its `table` put the forbidden corridor on the wrong
furniture and reported the organisers' own path as violating the keep-out it was
drawn to respect. Keep-out anchors now resolve by exact label or not at all —
a keep-out is the only constraint that can *lose* points, so it is the one place
a generous match is unsafe. Ungroundable keep-outs are reported, not scored.

`TAU = 1.25 m` is p90 of how close the reference paths come to the objects their
own instructions name (`traj_tolerance.py`: median 0.59 m, 81% within 1.0 m,
against 26% for objects they do not name). It is read off the reference answers,
not chosen to flatter ours.

**Known limit.** `livingroom_3` q5 is the one reference the proxy does not award
6/6: its path approaches the bowl (track 38–214) before the cabinet it is
instructed to pass first (233–471), so no monotonic matching can satisfy all
three. Left standing rather than tuned away.

## What the recorded runs score

102 run directories, 51 with a driven track, 25 matching an official question
verbatim via `plan.json`:

| | |
|---|---|
| **pooled over 25 runs** | **4.56 / 6** |
| best run per question | 5.44 / 6 |
| latest run per question | 5.22 / 6 |
| reference answers, same scorer | **5.97 / 6** |

| | achieved |
|---|---|
| constraints achieved | 46 / 60 — 77% |
| passages | 8 / 10 — 80% |
| out-of-order constraints | **0** |
| keep-out violations | **0** |

Two results worth separating. The ordering machinery and the keep-out corridor
(TASK 32, 33, 37) cost **nothing** across every recorded run — the constraints
that were hardest to reason about are not where the points go. All 12 lost
destinations are grounding or approach failures, and they are **bimodal**:

| miss | n | distances |
|---|---|---|
| 1.25 – 2.5 m — standoff / tolerance | 5 | 1.25, 1.30, 1.55, 1.64, 1.96 |
| > 2.5 m — the wrong object was bound | 7 | 3.54, 4.11, 4.39, 4.94, 5.51, 5.63, 5.64 |

Nothing lands between 1.96 m and 3.54 m. The near band is a standoff that stops
a little short; the far band is `bind_target` holding a different object. They
are different bugs and only the second is a research problem.

## The corpus is not an evaluation, and this is the finding

The 25 runs cover **9 of the 30 official questions, on 9 of 15 scenes**, with
repeats concentrated on the questions that were being debugged at the time —
`livingroom_2` q5 six times, `home_building_1` q5 five times, `studio` q5 four
times — while `hotel_room_1`, `hotel_room_2`, `livingroom_1`, `livingroom_3`,
`livingroom_4` and `office_1` have never been driven at all.

So **4.56 / 6 is not this stack's score.** It is the mean of a development log,
weighted toward whatever was broken that week, and the three questions with
n = 1 carry as much weight as the six-run scene. Quoting it as a system result
would be wrong.

What it is good for is saying where the next runs belong: the 21 official
questions never attempted, on the 6 scenes never entered. That is one clean
sweep of all 30 with the shipped `ai_module` node, and it is the only thing that
turns this table into an evaluation.

## Use

```bash
uv run python scripts/score_if.py --reference          # necessary condition
uv run python scripts/score_if.py --negative           # sufficient condition
uv run python scripts/score_if.py --runs runs          # score the corpus
uv run python scripts/score_if.py --runs runs --verbose --json scores.json
```
