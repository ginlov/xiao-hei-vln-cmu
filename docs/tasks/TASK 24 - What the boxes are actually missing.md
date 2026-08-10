# TASK 24 — What the boxes are actually missing

TASK 23 ended with one instruction: before touching the segmenter, find out
whether the points we lift are the wrong points. Its lidar oracle put a perfect
selection at 1.622 of 2 against our 0.191, which made point selection worth
roughly eight times everything else on the list, and SAM 2.1 Hiera *Tiny* the
obvious thing to buy. This is the audit that instruction asked for. It moves
the answer, retires three candidate repairs — two of them predictions of mine —
and revises TASK 23's headline downward.

## The tool

`scripts/point_audit.py`. For every detection that lands on a real object it
measures the selection against an oracle set — every return in the *same* scan
that falls inside the ground-truth box:

```
precision = |ours ∩ oracle| / |ours|      how much background we grabbed
recall    = |ours ∩ oracle| / |oracle|    how much of the object we missed
```

Both are computed twice, once on the mask's returns after the z-buffer gate and
once after the dominant-depth-cluster filter, because the repair differs by
regime: low P with high R means something keeps the background, high P with low
R means the cluster cuts into the object, and low on both means the mask is on
the wrong object. Only the third justifies a new segmenter.

Four modes, one pass each over the corpora:

| mode | question |
|---|---|
| *(default)* | precision/recall per observation, plus the box span and the ceiling |
| `--decompose` | of an object's returns, how many survive each filter |
| `--geom` | is the mask small, or the right size in the wrong place |
| `--split` | does one object arrive as several detections in one frame |

The selection is reimplemented rather than called: `PointLifter.lift` returns
only the survivors and the audit needs the intermediate sets. It also projects
once per *frame* instead of once per detection, which is why it is minutes
rather than an hour. `--self-check N` replays N detections through the real
lifter and asserts the final point sets agree — 150 checked, 0 mismatched. Run
it after any change to `lifter.py`; the reimplementation is the one thing here
that can silently drift.

## What it measures

Seven scenes, detections at the shipped 0.35 threshold, `unknown` ground truth
excluded (see below), 59 369 detections → 27 704 matched observations.

| stage | P mean | P median | R mean | R median |
|---|---|---|---|---|
| after mask + z-buffer | 0.652 | 0.735 | 0.384 | 0.364 |
| after depth cluster | **0.720** | **0.885** | 0.382 | 0.360 |

The depth cluster buys 6.8 points of precision for 0.002 of recall. It is
doing exactly its job and should not be loosened.

The regime table, thresholding each at 0.5:

| | R < 0.5 | R ≥ 0.5 |
|---|---|---|
| **P < 0.5** | 17.9% | 5.2% |
| **P ≥ 0.5** | **45.7%** | 31.1% |

Where the detections go: 23 710 of 59 369 (40%) never lift at all, dropped for
fewer than ten inliers; 7 955 lift but land on no scoreable ground-truth object,
dominated by `door`, `floor`, `ceiling`, `column` and `window` — structure, which
is excluded from ground truth here, so those are not errors. (`picture` 865 and
`lamp` 337 in that bucket are less innocent; some of them will be objects
ground truth labels `unknown`, which this audit drops.)

## `unknown` is not an object

The first overlay rendered showed why the raw numbers could not be trusted: it
picked a ground-truth object labelled `unknown` whose AABB spans half of
`japanese_room`, its "oracle" returns smeared across every wall in the scene,
paired against a `floor` detection. TASK 23 had already found 53 such instances
(8.9% of ground truth) and recommended excluding them from scoring. They are
excluded here. Left in, they dominate the largest size bin and drag every
average with them — the first `japanese_room` run reported P 0.834 / R 0.259
against 0.720 / 0.382 once they were gone.

## The correction to TASK 23

The second overlay, on real furniture, is the important one. Ground truth's
`fan decoration` is a single box enclosing **three** fans plus the blank wall
between them. Being wall-mounted decorations the box hugs the wall, so nearly
every wall return inside that rectangle counts as "oracle". Our mask segments
one fan, correctly: P = 0.98, R = 0.15.

So the lidar oracle is partly circular. Wherever a ground-truth box is filled
edge to edge by returns that are not the object — a flat object against a
surface, or several objects grouped into one box — the oracle's AABB reproduces
that box for free, and no segmenter can follow it, because following it means
guessing ground truth's grouping convention rather than segmenting anything.

An honest ceiling holds our own masks fixed and takes an oracle only over
*which view to keep*. Scored over all 542 ground-truth objects, so the ones no
detection ever reached count as zeros:

| variant | mIoU | ≥0.25 | ≥0.5 | score/2 |
|---|---|---|---|---|
| pooled AABB over all its observations | 0.027 | 2.8% | 0.4% | 0.031 |
| trimmed (5–95 pct) pooled AABB | 0.083 | 10.7% | 3.1% | 0.138 |
| **best single view** | **0.225** | **37.5%** | **22.9%** | **0.603** |

Against our ≈0.22 on the same basis, the remaining prize is about 2.5×, not 8×.

## Three repairs the measurement retired

**The segmenter.** Only 17.9% of observations sit in the low-P/low-R cell.
Median precision after clustering is 0.885 — for more than half of matched
detections *every* point we lift is inside the ground-truth box. `--decompose`
puts the loss at the mask (36% of an object's returns land inside its own
detection, 43% inside any mask) with the field-of-view and z-buffer gates
dropping nothing, but `--geom` then shows the masks are *larger* in pixels than
the object's projected footprint, not smaller. A bigger checkpoint does not fix
a mask that is already big enough and already on the object. The one place SAM
is plainly wrong is objects under 0.3 m: precision 0.130, and 32.5× as many
points as the oracle set. TASK 23 already showed that bin cannot reach IoU 0.5
even when cheating, so it is worth at most one point an object, over 488 of
27 704 observations.

**A front-surface bias in the centre.** My prediction: a cloud taken off an
object's visible face is centred on that face, half the object's depth short of
the truth, and the direction of the error follows the viewing ray, so a median
over observations from one arc would never cancel it. That would have had a
deterministic fix. Measured, splitting the centre error along and across the
ray:

| | median |
|---|---|
| along the viewing ray | **+0.004 m** |
| across it | 0.236 m |
| share biased toward the camera | 50.8% |
| ray error / ground-truth depth along the ray | +0.01 |

A coin flip. The centre error is entirely lateral; there is no depth bias to
correct. What "lateral" looks like is taking one part of an object — one fan of
three, one arm of a sofa — which is also what high P with low R looks like.

**Accumulating views to fill the box.** The natural follow-up, and wrong. Over
the same fused nodes, varying only the final box:

| estimator | mIoU | ≥0.25 | ≥0.5 | score/2 | vol/GT | cErr |
|---|---|---|---|---|---|---|
| median of observations (was shipped) | 0.203 | 37.9% | 6.3% | 0.443 | 0.85 | 0.233 |
| **mean weighted by point count** | **0.217** | **44.2%** | 5.9% | **0.501** | 1.09 | 0.230 |
| trimmed union, 25–75 pct | 0.216 | 41.8% | 7.4% | 0.492 | 1.90 | 0.222 |
| union of the better-seen half | 0.150 | 18.9% | 0.3% | 0.192 | 4.26 | 0.238 |
| union of all | 0.127 | 16.1% | 0.0% | 0.161 | 6.00 | 0.242 |

Union inflates the volume six-fold: the observations scatter by 0.23 m and the
union spans the object *plus* the scatter. Even given an oracle that files every
observation under the correct object, one AABB over the pooled points scores
0.138 against the median estimator's 0.443. Pooling points is not the way, and
the oracle does not rescue it.

The same sweep settles whether the boxes are undersized, by scaling every
extent by a constant k:

| k | 0.8 | 0.9 | **1.0** | 1.1 | 1.2 | 1.5 |
|---|---|---|---|---|---|---|
| score/2 | 0.379 | 0.454 | **0.501** | 0.480 | 0.435 | 0.267 |

k = 1 is the peak and the volume ratio there is 1.09. **The boxes are the right
size.** A size prior has nothing to add, and the residual is placement, which
no estimator over these observations can reach — cErr sits at 0.222–0.242 across
all thirteen variants.

## Merging parts within a frame: real but small

`--split` tests the lateral-error reading directly. In 30.7% of object-frames
more than one detection lands cleanly on the same object (mean 1.51 parts, 1.38
distinct labels among them), so objects genuinely are being cut up before
fusion sees them. Taking the union of those parts, within the frame:

| | recall | IoU |
|---|---|---|
| best part alone | 0.511 | 0.274 |
| union of the parts | 0.538 | 0.286 |
| *restricted to the 4 368 with >1 part* | 0.463 → 0.550 | 0.248 → **0.288** |

Union precision stays at 0.876, so the parts really are all on the object. But
+0.012 IoU overall and +0.040 on the subset that has parts to merge is not the
lever, and it says something more useful: from a single frame, even a perfect
in-frame assembly of our masks reaches IoU ≈0.29. The gap to ground truth is
not a grouping failure inside the frame.

## What shipped

`_Node._recompute` now takes a point-count-weighted mean of its observations'
centres and extents instead of a median, with `_observe` recording each
observation's core-point count in a new `obs_weights` list. Measured twice on
independent implementations (0.436 → 0.507 in TASK 23's estimator scratch,
0.443 → 0.501 here) and then end to end through the real pipeline:

| | P | R | mAP | mIoU | **pts/2** | cErr | cMAE |
|---|---|---|---|---|---|---|---|
| median | 0.141 | 0.424 | 0.303 | 0.172 | **0.152** | 0.27 | 4.71 |
| weighted mean | 0.144 | 0.421 | 0.306 | 0.177 | **0.169** | 0.26 | 4.57 |

Same corpora, same detections, same fusion; only the final box differs. Recall
and precision barely move, which is the point — this is a box change, not a
detection change. Counting error improves 3%.

`pts/2` is new, and adding it was necessary to see this at all: mean IoU moves
2.9% where the score moves **11.2%**, because the gain is concentrated at the
IoU 0.25 step and a mean over matched pairs cannot show a threshold crossing.
`eval.py` now reports `frac_iou_ge_25`, `frac_iou_ge_50` and `obj_ref_points`
per ground-truth object, so an object we never found scores zero rather than
being dropped from the average.

The gain is not uniform: `arabic_room` 0.233 → 0.317 and `loft` 0.089 → 0.133
carry it, `livingroom_3` and `office_1` are flat, and `japanese_room`
(0.046 → 0.023) and `office_2` (0.075 → 0.067) regress. The two that regress
have the lowest baselines in the set and 43 and 134 ground-truth objects
against a `pts/2` of a few hundredths, so a single object changing category
moves them; they are noise, not counter-evidence, but the sweep's predicted
0.443 → 0.501 was over my own reimplementation of fusion and clearly does not
transfer one-for-one to the real pipeline.

The gain is in near misses promoted to 1-pointers; the ≥0.5 rate does not move.
That is the signature of an estimator that has run out of room — 2 points needs
points placed differently, not averaged differently.

And it has run out. The 0.603 best-view ceiling says a better *choice* of
observation is worth a lot, but no runtime-computable proxy for "good view"
reaches it. Weighting by agreement with the other observations, and picking the
medoid outright, were both tried:

| estimator | mIoU | ≥0.25 | ≥0.5 | score/2 |
|---|---|---|---|---|
| **mean weighted by point count** | **0.217** | **44.2%** | 5.9% | **0.501** |
| weights × mutual IoU (consensus) | 0.216 | 43.1% | 6.0% | 0.491 |
| weights squared | 0.216 | 43.1% | 5.7% | 0.488 |
| medoid observation | 0.198 | 36.6% | 4.1% | 0.407 |

Thirteen variants in all; plain point-count weighting wins. The medoid result
also re-confirms TASK 23's: any estimator that stakes the box on one
observation loses, and the ordering is monotone in how many survive. Treat the
estimator search as closed.

## What is left, in order

1. **Price the ground-truth box convention.** How much of the remaining recall
   gap is boxes that group several objects or hug a surface? A per-object
   measure — how much of the box's volume its own returns actually occupy —
   would separate "we missed the object" from "the box was never the object",
   and the second kind is not ours to fix.
2. **Lidar ring sparsity.** The overlays show the returns arranged in a few
   horizontal bands; the vertical extent our points can span is quantized by
   ring spacing. The audit's worst-axis span is 0.53 of the ground-truth box
   against a best axis of 1.18, and nobody has checked whether the worst axis
   is systematically the vertical one. If it is, that is a sensor limit rather
   than a bug.
3. Still validated and still unimplemented from TASK 23: the node alias merge
   (recall +8.7%, precision +8.5%), admitting openings (free), and excluding
   `unknown` from the benchmark's denominator — which this audit is now a
   second argument for.
