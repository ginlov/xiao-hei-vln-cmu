# TASK 59 — Which half of the stack the score actually depends on

The system makes two geometric commitments the score could plausibly rest on: it
takes range from a lidar lift rather than from the model, and it publishes the
point it predicts `waypointConverter` will *settle at* rather than the point it
would like the vehicle to reach. Both cost engineering effort, and neither had
ever been priced against the alternative on a driven robot. A third commitment —
handing the model back a log of places it has already searched — had been priced
per call in TASK 45 and never driven.

This task drives all three, plus the baseline that has none of the geometry.


## The arms

All five run the same 26 instruction-following questions over 13 scenes, from the
same spawns, with the same prompt version and model.

| arm | range | converter | visited block | runs |
|---|---|---|---|---|
| **Ours** | lidar lift | modelled | in prompt | 26 |
| **w/o Lift** | model's `distance_m` | modelled | in prompt | 52 (2 passes) |
| **w/o Platform Model** | lidar lift | free-space check only | in prompt | 26 |
| **Naive** | model's `distance_m` | free-space check only | in prompt | 26 |
| **w/o Memory** | lidar lift | modelled | removed | 26 |

Each is one environment variable: `XIAO_HEI_RANGE_FROM`, `XIAO_HEI_SKIP_CONVERTER`,
`XIAO_HEI_VISITED`. Sweeps are `scripts/sweep_{control,naive,arm_c,arm_d,arm_e}.sh`;
`sweep_control.sh` sets the three variables to their shipped defaults explicitly,
so the reference arm's configuration is legible in the file rather than implied.

`w/o Platform Model` and `Naive` remove our *prediction* of the converter, never
the converter: it runs on the robot and re-snaps whatever is published. `Naive`
is not a VLM-only system either — comparative relations are still settled by
lifting candidates and anchors against the scan.


## Results

| arm | score | nDTW | CLS | TL | TL ratio |
|---|---|---|---|---|---|
| Ours | **62.8%** | 0.347 | 0.398 | 12.1 m | 1.23 |
| w/o Lift | 60.6% | 0.285 | 0.318 | 13.6 m | 1.42 |
| w/o Platform Model | 50.6% | 0.335 | 0.383 | 8.3 m | 0.78 |
| Naive | 54.5% | 0.314 | 0.358 | 10.2 m | 1.00 |
| w/o Memory | 67.3% | 0.389 | 0.398 | 12.3 m | 1.19 |

`Ours` is `runs/cn_0825_2316_*`, driven contemporaneously with the ablations.
An earlier corpus of the same configuration on 18 Aug code (`runs/cv_0818_*`,
52 drives over two passes) scores **62.8%** as well — see the code-boundary
paragraph below.

Paired over the 26 questions, repeats averaged first, `|t| > 2.06` at df = 25:

```
Ours - w/o Lift             +2.2 pp   t = +0.56   not resolvable
Ours - w/o Platform Model  +12.2 pp   t = +2.31   RESOLVED
Ours - Naive                +8.3 pp   t = +1.87   not resolvable
Ours - w/o Memory           -4.5 pp   t = -1.13   not resolvable
```

**Only the platform model is resolved on the score.** The design resolves roughly
8-point effects at this sample size; every other gap is smaller than that, which
means the experiment cannot tell, not that the arms are equal.


## The two failures point in opposite directions, and partly cancel

This is the finding that survives every test we applied to it.

```
TL ratio against the ground truth's own length (1.0 = same length)
  Ours                +0.16   t = +1.76   ns
  w/o Lift            +0.42   t = +3.21   RESOLVED   -- measurably long
  w/o Platform Model  -0.22   t = -2.09   RESOLVED   -- measurably short
  Naive               +0.00   t = +0.01   ns

w/o Lift - w/o Platform Model   +0.64   t = 5.97   23+/3- (p = .0001)
                                trimmed of the two extremes: t = 6.25
```

The model's metres overshoot, so `w/o Lift` drives 1.42× the ground truth. The
free-space clamp fires only when the free distance is *shorter* than the aim, so
it is a one-way operator and `w/o Platform Model` drives 0.78×. Put both faults
in one system and `Naive` drives 1.00× — indistinguishable from the ground
truth, and closer to it than we are.

**The effects are not additive.** Additive prediction 48.4%; `Naive` scores
54.5%, above the arm that keeps the lift (50.6%). The difference between those
two is not resolved (`+3.8 pp, t = 0.83`), so this is an observation rather than
a result, but it is consistent with the mechanism: once nothing downstream
predicts where the platform settles, an over-long range is a *corrective* error
against a clamp that can only shorten.

Driving the right distance is not driving to the right place, which is why
`Naive` recovers no score.


## The memory arm: no behavioural effect, and a score we cannot read

Removing `VISITED_BLOCK` is a prompt-only ablation. This was checked rather than
assumed: `Ctx.visited` is read in exactly one place (`visited_for`) and all three
call sites feed the prompt. The navigation-side revisit machinery uses separate
lists — `spent` (`approach_loop:577`), `stood` (`:770`), `reached`
(`execute_plan:250`) — so the robot still refuses to re-drive a spent bearing;
only the model loses the record.

Measured per step rather than per question (`scripts/memory_ablation.py`):

```
                 Ours    w/o Memory    paired
revisit rate      27%        29%       t = -0.57   12+/14-
spread          2.59 m     2.78 m      t = -1.36
steps/question    8.9        9.0       t = -0.10
visible rate      92%        93%       t = -0.32
```

**All four are unresolved and all four sign tests are near-tied**, including the
one the block's own wording predicts. That reproduces TASK 45's per-call result
on a driven robot, and closes the sentence that report ended on.

The score, however, moves the wrong way: 67.3% against 62.8%, `+4.5 pp,
t = 1.13`, sign 4+/2-. Unresolved, but no longer unattributable.

**The code boundary is measured, not argued away.** `runs/cv_0818_*` predates two
commits of 24 Aug (`d2196ea`, `926f4b4`) that rewrote `bind_target` and
`run_goto`; every ablation arm was driven after them, so the +4.5 pp could have
been the commits rather than the block. `scripts/sweep_control.sh` re-drove the
shipped configuration on 25 Aug code over the same 26 questions. The two corpora
score **identically**: `-0.0 pp, t = -0.00`, sign 6+/5-. The week of commits is
worth nothing here, so the 4.5 points belong to the ablation.

Eleven of the 26 questions did change score between the two code versions and
cancelled. That is an independent instance of TASK 57's finding: per-question
results are not reproducible while corpus means are.

**Where the 4.5 points actually come from.** Only six questions differ at all;
twenty tie. Of the six, three are passage constraints flipping (1 for the arm,
2 against — `PASS` is the dominant failure mode in every arm and flips both
ways), one is a 17 cm near-miss past a 1.25 m threshold, and one is both runs
exhausting the step budget with the arm happening to be 1.7 m further along.

**One question has a legible mechanism.** `home_building_2` q4 asks for *the
potted plant on the dressing table*; the scene holds eight potted plants in two
clusters, and the robot spawns in one of them. At the clause switch our arm
reports `visible=True, confidence=0.55`, turns back toward the area it started
in, and stops 0.99 m from the wrong instance — 7.04 m from the reference's.
Without the block the same step reports `visible=False, confidence=0.20`, the
arm explores away from the spawn, and reaches the right instance. The harm, if
it is real, is not that the block makes the robot avoid revisiting: it is that
handing the model a record of where it has been and what it saw there makes it
readier to settle for a candidate it already has. **This is n = 1 and the
confidence signature does not reproduce on the other two flips**, where the arm
without the block is if anything the more confident of the two.


## Failure analysis of the shipped system

17 constraints are missed across the 52 `Ours` runs. Three modes account for
nearly all of them.

1. **Local grounding without a plan that reaches the target.** `loft` q4: the
   model returns a similar object in the wrong part of the room at a confidence
   of 0.40 it never revises, while the referred cup lies ~6 m ahead past the
   sofa. The episode covers 13.7 m of path and ends 0.6 m from its start pose.
2. **Passages resolved but not executed** — 9 of the 17. `hotel_room_2` q4: the
   returned box falls on the floor between bench and bed, and the vehicle
   advances 1.1 m in total. The deficit is waypoint generation, not grounding: a
   passage needs a waypoint placed *beyond* a gap not yet entered.
3. **Premature commitment under a confident detection** — 11 arrivals. `chinese_room`
   q4: confidence rises across five calls to 0.95 on a plant supported by a
   corner stand rather than a table, and the search stops; the referred instance
   is 5.2 m away. These repeat to within 0.1 m across independent passes.
   Confidence reflects head-noun match quality and says nothing about whether the
   qualifying clause holds.

(3) bounds what better ranging can recover: an accurate range to the wrong object
does not help. That is the most plausible reading of `w/o Lift`'s unresolvable
2.2 pp.


## What this does not settle

- **Whether the memory block is harmful.** The control removes the code-drift
  explanation, but `+4.5 pp, t = 1.13` does not clear the threshold, and the
  mechanism above rests on one question. What is established is the negative:
  removing the block does not hurt.
- **Single passes.** Only `w/o Lift` is averaged over two passes. `Ours`,
  `w/o Platform Model`, `Naive` and `w/o Memory` are one pass each; their
  question-level noise is undamped. `w/o Lift`'s two passes are 5.8 points apart
  on their own (57.7% and 63.5%), which is how large that noise is.
- **nDTW cannot rank these arms.** A deliberately truncated ground-truth path
  scores 0.404 on our validation, higher than every one of the five arms manages
  on real driving (best is 0.389), so the measure cannot punish stopping short — which is one arm's
  entire failure mode. It also inverted on a 16-question subset. It is
  corroboration when it agrees and uninformative when it does not.
- **CLS is reported as a negative result.** It gives a *reversed* ground-truth
  path 1.000, because coverage is order-blind, and order carries the heaviest
  penalty in the rubric.
- **SPL is not computable.** It needs the shortest traversable path, which needs
  a navmesh we do not have.


## Two claims withdrawn

Both were made on a 16-question subset driven while the corpus was still filling,
and neither survived the full 26.

- *The two geometry ablations score identically.* They scored 56.2% and 56.2% on
  the subset; on the full corpus they are 54.5% and 50.6%.
- *nDTW ranks `w/o Platform Model` first.* It did on the subset (0.406 against our
  0.372); on the full corpus it does not (0.335 against 0.347).

A ten-question difference flipped both, which is itself the argument for not
leaning on either measure.


## A recording gap this exposed

`steps.jsonl`'s settings header records the backend, model, prompt version and
every ablation flag, but not the commit the loop ran at. That omission is why the
memory arm's result cannot be attributed. Adding `"commit": <sha>` to
`Ctx.record_settings` costs one line and makes any future cross-corpus comparison
self-evident. Deliberately not done while sweeps were in flight, since
`approach_loop.py` is re-imported by every question.


## Changed

- `scripts/sweep_arm_c.sh`, `sweep_arm_d.sh`, `sweep_arm_e.sh`,
  `sweep_arm_d_topup.sh`, `sweep_arm_e_16.sh`, `sweep_control.sh` — the arms.
- `scripts/memory_ablation.py` — per-step behavioural metrics for the memory arm.
- `scripts/path_fidelity.py` — nDTW, CLS, TL against the ground-truth paths.
- `scripts/score_if.py` — `--glob`, so one arm can be scored without pooling it
  with the arm it ablates.
- `scripts/approach_loop.py`, `vlm_approach.py` — `XIAO_HEI_RANGE_FROM` and
  `XIAO_HEI_SKIP_CONVERTER`, with `SKIP_FREE_MIN_M` so an uninformative
  free-space check publishes the aim rather than the vehicle's own position.
- `tests/test_naive_range_arm.py`, `tests/test_skip_converter_arm.py` — 13 tests,
  including the regression for the empty-terrain clamp that invalidated the first
  `w/o Platform Model` sweep.
