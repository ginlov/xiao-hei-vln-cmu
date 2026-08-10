# TASK 23 — Where recall actually dies, measured stage by stage

TASK 22 left recall at 0.42 and no account of the missing 0.58. Every candidate
repair — a wider class list, a finer face split, a better mask, a looser merge
rule — targets a different stage, so choosing between them from one aggregate
number is guessing. This task builds the per-stage attribution, follows it, and
retires most of the plan TASK 22 proposed.

## The funnel

`scripts/recall_funnel.py` walks every ground-truth object through the pipeline
and records, independently per stage, how far it got:

| stage | test |
|---|---|
| `in_vocab` | the class prior contains its label at all |
| `observed` | some keyframe's lidar returned ≥8 points inside its box |
| `seen_2d` | a cached detection's mask covers ≥30% of those points' pixels |
| `named` | one of those detections carries the label ground truth uses |
| `lifted` | such a detection lifts to within 1 m of the true centre |
| `node` | a node survives export within 1 m of it |
| `recalled` | that node also carries the right label |

Visibility is decided by lidar returns rather than by projecting the box
corners: a corner can sit inside the vertical FOV while the object is behind a
wall, and the returns settle occlusion, range and grazing incidence in one test
without a separate ray-cast.

**The stages must be independent flags, not a "furthest stage reached"
counter.** The first version used a counter, and `node`/`recalled` are read off
the exported graph rather than from the per-frame pass — so any object with a
node near it had its counter raised to 6, silently backfilling `named` and
`lifted` as passed. That version reported `in_vocab` at 99.8% and a 46.6% loss
at fusion, and both were artefacts. The fingerprint was visible in the output
and missed on the first read: `named == lifted == node` in all seven scenes.

## Result — 595 objects, seven scenes

| stage | passes alone | cumulative | lost here | of GT |
|---|---|---|---|---|
| `in_vocab` | 445 (74.8%) | 445 | **150** | **25.2%** |
| `observed` | 463 (77.8%) | 362 | **83** | **13.9%** |
| `seen_2d` | 433 (72.8%) | 350 | 12 | 2.0% |
| `named` | 225 (37.8%) | 225 | **125** | **21.0%** |
| `lifted` | 221 (37.1%) | 221 | 4 | 0.7% |
| `node` | 533 (89.6%) | 221 | 0 | 0% |
| `recalled` | 256 (43.0%) | 207 | 14 | 2.4% |

**Vocabulary and naming together account for 46.2% of all ground-truth
objects** — nearly half, lost purely to which word we prompt with and which
word comes back. 2-D detection loses 2.0%, the lift 0.7%, fusion 2.4%.

The clean chain ends at 207 while 256 are actually recalled, so the per-frame
tests are conservative and each stage's loss is an upper bound. The ranking is
not sensitive to that: no plausible loosening moves a 21% term below a 2% one.

## The autopsy agrees, and kills the merge hypothesis

A second, independent pass classified why each object failed, by re-running
`export` step by step and attributing every removal:

| outcome | n | share |
|---|---|---|
| recalled | 245 | 41.2% |
| `never_formed` — no node with that label ever existed | 260 | **43.7%** |
| `far` — such a node exists, but not near this instance | 57 | 9.6% |
| `nms` — cross-label NMS deleted it | 15 | 2.5% |
| `taken` — another instance claimed it (under-segmentation) | 11 | **1.8%** |
| `pruned` — dropped as low-evidence | 7 | 1.2% |

`MERGE_DIST` was the leading suspect: it is 0.4 m, our centre error is 0.22 m,
and office scenes are full of identical chairs. It is worth 1.8%. The
supporting statistic is equally deflating — 170 ground-truth pairs share a
label within `MERGE_DIST`, but 117 of them are `unknown`, a label nothing can
name; excluding it leaves 53 pairs across seven scenes.

## The vocabulary, priced

| scene | GT | in vocab | recall | recall / ceiling |
|---|---|---|---|---|
| arabic_room | 60 | 85.0% | 0.48 | 0.56 |
| chinese_room | 82 | 89.0% | 0.56 | 0.63 |
| japanese_room | 43 | 53.5% | 0.33 | 0.62 |
| livingroom_3 | 87 | 87.4% | 0.55 | 0.63 |
| loft | 90 | 90.0% | 0.38 | 0.42 |
| office_1 | 99 | 85.9% | 0.43 | 0.50 |
| office_2 | 134 | **41.8%** | 0.23 | 0.55 |
| total | 595 | 74.8% | | |

Normalised by the ceiling the scenes run 0.42–0.63 rather than 0.23–0.56:
**most of the scene-to-scene spread is vocabulary and coverage, not
perception.** `office_2`'s recall of 0.23 is almost entirely its 0.418 ceiling.

Two facts about the 25.2% outside the prior:

* **`unknown` is 53 instances across five scenes — 8.9% of all ground truth.**
  Ground truth labels the object `unknown`; `gen_vocab_prior.py` drops the word
  deliberately because it names nothing. No label-matching metric can ever
  score these, and they have been sitting in our recall denominator.
* 62 of the 64 missing labels appear in exactly one scene, which is precisely
  what `MIN_SCENES = 2` excludes — 94 instances, 15.8% of ground truth.

## A class list is not a capability list

Of the 110 prior labels, **22 never fire once** at the shipped threshold across
all seven corpora, and they cover 83 ground-truth instances (13.9%):

| label | GT instances | detections ≥0.35 | detections at any score |
|---|---|---|---|
| **`focus light`** | **52** | 0 | **0** |
| `marker` | 8 | 0 | 0 |
| `paper` | 8 | 0 | 12 |
| `file` | 4 | 0 | 0 |
| `chopsticks` | 3 | 0 | 5 |

`focus light` alone is 8.7% of every ground-truth object in the benchmark, and
YOLO-World has never emitted the phrase at any confidence. Twelve further
labels are pure dead weight — ground truth does not use them either
(`bedroom light`, `beer bottle`, `drawer`, `glass`, `handle`, `kitchen
cabinet`, `shower tap`, `sink cabinet`, `stove`, `tap`, `toilet paper`,
`towel rack`) — and they still cost inference time and dilute the scores of
the phrases that do work.

The reverse also holds: **26.8% of detections land on labels ground truth never
uses.** Excluding structure, the largest are `dining table` ×914, `bookcase`
×206, `tv stand` ×162, `sculpture` ×100, `ottoman` ×49, `mattress` ×37. We
detect the object confidently and file it under a word nobody asked for.

These are two different diseases needing two different cures. A dead word means
the object is **absent from the map** and no downstream reasoning can recover
it. A wrong word means the object is **present under another name**, which the
LLM can often still pick — and which the official object-reference metric does
not penalise at all, since it scores 3-D IoU and never compares labels.

## The one repair this task validated

`ObjectMap.finalize` runs cross-label NMS over nodes, deleting the weaker of
two boxes overlapping above `NMS_IOU`, ranked by `(n_obs, score)` with the
label not consulted. When `sofa` is seen 80 times and `couch` 12 times at one
spot, `couch` is deleted — and ground truth calls it a couch.

Two overlapping nodes are one physical object, so collapsing them is right;
discarding the loser's *name* is what costs recall. Three treatments, one lift
pass, seven scenes:

| | #obj | P | R | mIoU | ≥0.25 | cMAE | score/2 |
|---|---|---|---|---|---|---|---|
| A suppress (shipped) | 250 | 0.141 | 0.424 | 0.170 | 29.3% | **4.71** | 0.338 |
| B keep both | 291 | 0.131 | 0.461 | 0.186 | 33.6% | 5.23 | 0.398 |
| **C alias** | 252 | **0.153** | **0.461** | **0.190** | **34.4%** | 5.16 | **0.399** |

C reaches B's recall at A's object count: recall +8.7%, precision +8.5%, mIoU
+12%, challenge score +18% relative. The cMAE regression is a presentation
choice, not an intrinsic cost — it charges the group to every label it carries.
Counting only the primary label leaves the group set identical to A's, so cMAE
should return to 4.71 with the recall retained; that follows from the grouping
being the same and has not yet been measured directly.

## Also measured

* **Admitting openings is free.** Narrowing `is_structure` to surfaces only
  (wall/floor/ceiling/carpet/rug/glass) leaves the furniture metrics
  bit-identical (P 0.141, R 0.424, mAP 0.303) while recovering doors, windows
  and columns at R 0.42 / mAP 0.27 — comparable to furniture. 22 of the 75
  official questions name one.
* **The centre error is scatter, not bias.** Per-axis median offset is
  0.029 / −0.005 / 0.060 m against standard deviations of 0.25 / 0.22 / 0.14.
  Removing the global offset moves the challenge score from 0.420 to 0.437 out
  of 2. An uncalibrated extrinsic is ruled out; the fix must be per-object.
* **Box quality rises steeply with observation count.** Nodes under 16
  observations score 0% at IoU ≥ 0.5 and under 0.10 mIoU; nodes over 64 reach
  0.332 mIoU, 61% at ≥0.25 and 20.5% at ≥0.5, with centre error falling from
  0.45 m to 0.145 m. corr(log n_obs, IoU) = +0.53, corr(log n_pts, IoU) =
  +0.57. Confounded with object size and proximity, but it is the evidence
  behind any "drive closer and re-measure" policy.
* **Evidence gating is a clean trade-off curve, not a free win.** `n_obs ≥ 8`
  takes precision 0.141 → 0.307 and cMAE 4.71 → 2.65 while recall falls 0.424 →
  0.314 and mAP 0.303 → 0.235. mIoU rises throughout, so the survivors are
  better boxes. Choosing a point on this curve needs a task-weighted objective
  — 15 counting questions against 30 reference ones — not mAP.
  (`min_obs` was recorded as inert in TASK 22 because `prune` keeps a node when
  `n_obs > min_obs` **or** `n_pts >= min_pts`; the idea was never tested.)

## Retired

TASK 22 and the plan drafted from it ranked a finer face split first and
observation-weighted fusion second. The funnel prices 2-D detection at 2.0% and
the lift at 0.7%, so both are repairs to rounding error. The detector unwraps
1920×640 into four 640×640 faces at 100° FOV and runs at `imgsz=640`, so no
resolution is being thrown away in the first place.

Also retired: extrinsic calibration (refuted above), `MERGE_DIST` (1.8%),
`prune` (1.2%).

## Next

1. **Prompt–output decoupling.** What we prompt with and what we emit need not
   be the same string. Prompt with a phrase YOLO-World can ground and emit the
   wording ground truth uses. `focus light` is the flagship: 52 objects, 8.7%
   of ground truth, currently absent from the map entirely.
2. **Question-driven prompting.** `Vocabulary.current_classes` already unions
   the prior with question-derived labels, but extracts single words only,
   while the missing labels are largely multi-word (`calligraphy painting`,
   `mouse pad`, `sake bottle`). This is the fix that transfers to unseen test
   scenes; a static prior tuned on seven corpora will not.
3. **Node-level alias merge**, per the validated table above.
4. **Exclude `unknown` from the benchmark denominator** — measurement hygiene,
   not an improvement.
5. **Coverage.** 13.9% never observed: `loft` 43/90 is the known stairs
   problem; `office_1` 70/99 and `office_2` 84/134 need separating into "the
   tour never got there" and "8 lidar returns inside a 10 cm box is a strict
   test".
