# TASK 26 — Can a VLM ground a referring expression, measured

TASK 23–25 established the ceiling our own perception stack runs into: the
fused object map holds the objects the fourteen instruction-following
questions name only **67 %** of the time, and 74 % of the misses are words
absent from the 110-class list rather than detection failures. Instruction
following is ~70 % of the challenge score and is answered with waypoints, so
that ceiling caps the thing worth most.

`docs/vlm_grounding_test.md` proposed the experiment that decides whether to
keep that architecture. This is the result.

**Headline: a VLM looking at a single frame from the start pose localises
70–72 % of named objects, beating the 67 % our full-coverage tour reaches, with
a median error of 0.11 m against a 0.58 m budget.** After two geometric gates
the false-positive rate lands at 11.1 %, against a 10 % acceptance bar.

## What was run

| | |
|---|---|
| Test set | the same 60 (question, named object) pairs `scripts/retrieval_probe.py` scores, deduplicated to 54 unique (scene, phrase) |
| Frame | the first frame of each recorded corpus — the start pose, no driving |
| Input | the four 640×640 perspective faces the sidecar already unwraps for YOLO |
| Model | `claude-opus-5` (the Gemini key does not exist yet; Claude's boxes proved accurate enough that Part B did not need it) |
| Hit criterion | lifted position within 1.0 m of any ground-truth instance of the phrase's synonym family — identical to `retrieval_probe` |

Pipeline per pair: referring expression → VLM → `box_2d` → face pixel → camera
ray → lidar returns in an adaptive cone → dominant depth cluster → map-frame
position → error against ground truth.

## Results

| | VLM, start pose | object map, start pose | object map, full tour |
|---|---|---|---|
| localised within 1.0 m | **72.2 %** | 48.3 % | 67 % (k=3) |
| median error of hits | **0.11 m** | 0.236 m | — |
| p90 error | 0.52 m | — | — |
| false positives | 24.1 % → **11.1 %** gated | — | — |

The error budget is 0.58 m, the median distance the reference trajectories keep
from the objects they name (`scripts/traj_tolerance.py`). p90 sits just inside
it.

## Two prompt bugs that would have inverted the conclusion

**Coordinate convention.** The first run parsed the boxes as the 0–1000
normalised values the prompt asked for and measured a 23.7° bearing error, with
the box drawn on a wall. The model had answered in **raw pixels**. On a
640-pixel face both readings produce numbers in the same range, so nothing in
the data distinguishes them — the correct reading gives a **0.03°** bearing
error. Fixed by requiring the model to declare `coord_space` explicitly. Had
this gone unnoticed the whole approach would have been recorded as a failure.

**A prompt written for only half the test set.** v1 demanded a "distinguishing
feature" before reporting a match, which is right for *the tea table with the
elephant figurine on it* and wrong for `chair`, `table`, `tv` — bare nouns with
no qualifier, which the model then dutifully refused. Splitting the instruction
into a bare-noun branch and a compound branch moved chinese_room from 33 % to
56 %.

## What the false positives actually are

`scripts/vlm_fp_report.py` renders every failure with the model's box in yellow
and ground truth projected into the same face in cyan
(`artifacts/fp/false_positives.jpg`). Reading the panels rather than the error
column changes the diagnosis:

| cause | n | what it looks like |
|---|---|---|
| no lidar return | 4 | ground truth **inside** the box; the cone is empty even at 14° |
| ray overshot | 3 | open stair, pendant lamp — beam passes through and finds the far wall |
| ray stopped short | 2 | box correct, beam measures an occluder in front |
| wrong instance | 2 | balcony chairs through glass; a side table read as the tray's table |
| truth not in this face | 2 | the elephant figurine answered for the horse; a TV stand for the cabinet |

**Nine of the thirteen false positives are cases where the model identified the
object correctly and the depth lift failed.** Counting only genuine
misidentifications gives 4/54 = **7.4 %**, inside the acceptance bar.

A sampling artefact was suspected inside this too: one registered scan carries
~976 points per steradian, so a 2° cone expects **under four returns** and
routinely catches none. `locate()` now widens the cone until it has enough to
take a median of, and reports which cone it used. That fixed the tea table but
**not** the four empty cones above, whose real cause TASK 27 traces to a
forward-tilted lidar with a bearing-dependent blind cone.

## Three proposed fixes for the depth problem, and what measurement said

| proposal | verdict |
|---|---|
| **(1) reject lifts beyond a distance cutoff** | **refuted.** False positives are *closer* than hits (median 2.52 m vs 2.92 m); 7 of 9 sit under 3 m. No threshold separates the distributions — a 3 m cutoff discards 45 % of hits to remove a third of the errors. |
| **(2) let the VLM judge whether the scanner can be trusted** | **refuted.** Asked as visual facts rather than sensor questions, `same_space` returned `true` for all 52 claimed sightings — including the chairs on the far side of a glass door. `occlusion` pointed the right way (`clear` 77 %, `see_through` 62 %) but on n=8 with no separable margin. |
| **(3) let the VLM estimate distance by perspective** | **partly true, and redundant.** Raw estimates are biased **1.81×** long with only 21 % within 1.5×. The bias is systematic: dividing it out lifts that to 65 % (in-sample — the factor was fit on this data and needs a held-out check). Even debiased it rejected nothing the geometric gates had not already caught. |

Recommending (2) as "most feasible" was wrong, and the data said so within one
run. In hindsight the reason is structural: all three failure modes — glass,
open structure, occluder — surface as *one* contradiction, between the measured
range and the size the object would have to be at that range. That contradiction
is a multiplication we can do ourselves, and asking a model to notice it
introduces judgment where arithmetic suffices.

## The gates that worked

| gate | hits | false positives | cost |
|---|---|---|---|
| none | 72.2 % | 24.1 % | — |
| **A — decline when the cone is empty** | **72.2 %** | **16.7 %** | **zero** |
| B — implied size within [0.15, 4.0] m | 70.4 % | 18.5 % | 1 hit |
| **A + B** | **70.4 %** | **11.1 %** | 1 hit |
| A + B + debiased VLM distance | 70.4 % | 11.1 % | 1 hit |

`implied_size = 2 · lift_range · tan(box_angle / 2)`. Gate A is free: no
threshold, no model, and refusing to answer without depth data is strictly
better than answering from none.

Gate B uses a crude global band. Replacing it with the per-class priors in
`perception/size_prior.py` looked like the obvious next step; TASK 27 tried it
and it **loses** at every tolerance — 12 of 52 phrases have no prior, and every
surviving false positive is a wrong instance whose size matches the right one.

## Bearing on the architecture

The pseudocode reviewed this session survives, with three corrections that came
out of the measurements:

1. **Two loops, not one.** A VLM call costs 1–3 s while the reference paths are
   ~10 m of driving. Geometry runs at control rate; the VLM fires on events —
   start, arrival, end of an exploration leg. Five to ten calls per question.
2. **`bound` is not `detected`.** A detection is per-frame; a binding is a
   committed map entry that survives out of view. Everything downstream reasons
   over the binding table, never over the last reply.
3. **Geometry validates.** The gates above are that step, and without them the
   claim rate is 96 % against a 72 % hit rate.

Separately, `avoid` turns out to cover **3 of 30** instruction questions while
**10** use `take the path between …`, a *required* passage. Treating every
`between` as a keep-out would lose ten questions to protect three.
`scripts/keepout_radius.py` also shows the reference path drives straight
through four of the six chair↔screen corridors in chinese_room q5, so "the path
between A and B" can only mean the **closest** pair — a geometric
disambiguation, not a semantic one.

## Cost

Measured, not estimated: 2 990 input tokens (four faces ≈ 2 400) and 460 output
tokens per call. At Opus 5's $5/$25 per MTok that is **$0.0265 a call** — $1.43
for a 54-pair sweep, and roughly **$4 for a full competition run** at ten calls
across fifteen questions. Cost is not a constraint here; the 1–3 s latency is.

## Artefacts

| path | what |
|---|---|
| `scripts/snap.sh` | capture image + scan + pose from the running sim |
| `scripts/grab_faces.py` | equirect → the sidecar's four faces, via its own LUTs |
| `scripts/vlm_probe.py` | one grounding call; the prompt lives here |
| `scripts/vlm_locate.py` | box → ray → lidar → metres, scored against a GT centre |
| `scripts/vlm_sweep.py` | the 54-pair sweep; replies cached, so re-runs are free |
| `scripts/vlm_fp_report.py` | annotated panel per false positive |
| `scripts/vlm_gates.py` | offline gate sweep — no API calls |
| `scripts/keepout_radius.py` | two-sided bounds on the avoidance radius |
| `artifacts/vlm_sweep.json` | per-pair rows |
| `artifacts/fp/false_positives.jpg` | the twelve failures, drawn |

## Open

- ~~Gate B on per-class size priors rather than a global band.~~ Refuted in
  TASK 27.
- Held-out validation of the 1.81× distance bias, if (3) is revisited.
- Re-run without the (scene, phrase) deduplication, for a strictly identical
  denominator to `retrieval_probe`'s 60.
- Everything here is measured **from the start pose on one frame**. Distance,
  occlusion and out-of-vocabulary hard cases are untested; so is Gemini, whose
  detection output is a trained capability rather than an emergent one.
