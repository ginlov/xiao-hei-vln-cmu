# TASK 22 — Multi-scene perception benchmark, and the box fix it found

TASK 21 tuned on `livingroom_3` and checked on `chinese_room`. Two scenes
cannot tell a real gain from a scene-specific one, and the corpora came from
the frontier explorer, whose coverage varies run to run — so a comparison
between two runs measured the explorer as much as the perception. This task
builds seven comparable corpora, a benchmark that scores them in one command,
and follows the first thing that benchmark found.

## What was built

**Deterministic coverage tours.** `scripts/gen_coverage_tour.py` turns a
scene's `traversable_area.ply` into a route that visits the whole floor. It
routes over the floor graph rather than emitting a straight-line tour, which
is what keeps every commanded hop under `--max-hop` (1.6 m; beyond roughly
that the local planner gives up) and keeps the segment between waypoints off
the walls. Node placement is greedy rather than on a lattice — a lattice
landing on furniture stranded whole rooms, and at 1.2 m spacing five of the
fifteen scenes reached under 70% of their nodes. At 0.8 m every scene reaches
essentially all of them.

**One-command recording.** `scripts/record_scene.sh` brings the stack up on a
scene, waits for terrain, arms the joystick, drives the tour, and reports what
fraction of it the robot actually reached. It runs the `dummy` responder on
purpose: detection happens offline in replay stage A, and calling the sidecar
from the tick blocked each frame for ~1 s — the first `japanese_room` recording
captured 39 frames where the same tour later captured 309.

**`python -m xiao_hei_vln.perception benchmark`** replays every corpus and
prints one table. It reports `inVoc` — the fraction of a scene's ground-truth
objects whose label our class prior can even emit — beside recall, because
recall under that ceiling is a perception problem and recall at it is a
vocabulary problem.

**Stage A can thin to keyframes** (`--min-move-m` / `--min-rot-deg`), and
stage B can filter the cache by confidence (`--min-score`). The second one
matters more than it sounds: the cache stores each detection's score, so
sweeping the detector threshold upwards is a 40-second experiment rather than
the 10-minute stage-A re-run TASK 21 assumed.

## Corpora

Seven scenes under `frames/<scene>_tour/`, all tour-driven, all with
detections cached at the 110-class vocabulary. `chinese_room` reached 94% of
its waypoints and `loft` 69% — `loft` has stairs the 2-D tour planner cannot
see, so its numbers understate coverage-limited recall. The rest reached 100%.

`office_building_1` and `office_building_2` are excluded: their scene zips
contain no ground-truth objects at all, so nothing can be scored on them.

## What the benchmark changed about our own numbers

**The benchmark was replaying a configuration we do not ship.** TASK 21 turned
the scan accumulator off in production (`XIAO_HEI_SCAN_KEYFRAMES` defaults to
0) but `replay_lift` kept defaulting it on, and `benchmark` inherited that.
Measured on the same three scenes, the shipped configuration is better on
every axis — precision 0.077 → 0.108, counting error 7.01 → 5.28, mAP 0.316 →
0.348 — so every number TASK 21 reported understated the system. The default
now matches production and the flag is `--scan-accumulator` to opt in.

## The fix: a node's box is the median of its observations, not their union

The binding constraint is the box. Object-reference questions are 30 of the 45
scoreable questions and
`evaluator/metrics/object_reference.py` scores them by 3-D IoU alone — the
label is never compared. Our mean IoU sat at 0.15 against a 0.25 threshold for
a single point.

Four hypotheses were measured and three were wrong:

| Hypothesis | Measurement | Verdict |
|---|---|---|
| Points leak past the mask; bound the extent by its angular span | cloud extent / mask-implied extent = **1.04** | refuted |
| Boxes grow as observations accumulate | corr(log n_obs, diagonal) = **-0.017** | refuted |
| Wall-mounted objects drive it | among *matched* nodes they score better | refuted (survivorship) |
| The fusion is at fault | single observation **0.95x** GT volume, fused node **3.47x** | confirmed |

A single observation is already the right size. Pooling the observations'
points does not average that out: each observation's centre is off by ~0.25 m
in a direction that depends on where the robot stood, so the union spans the
object *plus* that scatter, and the volume comes out about 3.5x too large. The
node now keeps each observation's own box and takes the median of them.

Two details cost a round trip each:

* The outlier gating had to move to the per-observation path, not disappear.
  A mask spanning two surfaces still has to be cut apart, and a median over
  observations would faithfully return the bimodal box.
* Keeping `_percentile_box` *as well* over-corrected, shrinking boxes to 0.66x
  of ground truth. Its docstring justifies itself by "the node's cloud is the
  union of every observation" — a premise this change removes, which is
  exactly why the trim became harmful rather than merely redundant.

## Results

Seven scenes, non-structure objects, before/after today:

| | before | after |
|---|---|---|
| objects emitted | 454 | **249** |
| precision | 0.078 | **0.141** |
| recall | 0.427 | 0.424 |
| mAP @ 1 m | 0.296 | **0.303** |
| counting MAE | 7.13 | **4.71** |
| box volume / GT | 3.47x | **0.84x** |
| mean challenge score (matched pairs, of 2) | 0.23 | **0.36** |

Holding one term at ground truth now says the size is no longer the problem:
GT size with our centre scores 0.56, our size with GT centre scores **0.83**.

## Rejected, with the measurement that rejected them

* **Cross-label NMS on 2-D masks.** Different labels return literally the same
  mask (median IoU 1.00 for `painting`+`picture`, `couch`+`sofa`, and eight
  more pairs), so suppressing the weaker looked obvious. It cost mAP 4.3% and
  recall 2.4% for no precision gain: the label kept is the higher-scoring one,
  which is not reliably the one the ground truth uses.
* **`min_obs` evidence filtering.** Inert as implemented — `prune` keeps a node
  when `n_obs > min_obs` **or** `n_pts >= min_pts`, and spurious nodes carry a
  median of 254 points.
* **Union centre with median extent.** 0.487 vs 0.473; not worth keeping the
  pooled cloud alive for.
* **Front-face bias compensation.** 70% of observations put the centre nearer
  the robot than the truth, median 0.088 m. Pushing the centre back along the
  view ray lowered median centre error (0.183 → 0.166 m) and lowered IoU
  (0.231 → 0.208): the bias is a median over a wide spread, so a uniform shift
  moves the already-correct nodes off target.

## Open, and now quantified

* **Centre accuracy is the whole remaining gap** — 0.36 today against 0.83 with
  a perfect centre. The obvious estimator-level fixes are refuted above; what
  is left is calibration, mask quality, or triangulating across views instead
  of taking a lidar median.
* **We strip the anchors the questions use.** 22 of the 75 official questions
  reference `window`, `door`, `door frame` or `column`, which `is_structure`
  removes from the object list — 14 of the 30 instruction-following questions.
  Surfaces (floor, ceiling, wall) are rightly excluded; countable, localisable
  openings are not.
* **The vocabulary fragments categories the detector cannot split.** Seven
  words for lights, three for wall art. 59% of our nodes sit within 1 m of a
  real object under the wrong name — well above the 2.3–12.7x chance baseline,
  so this is naming, not luck. It costs counting directly and reference
  retrieval indirectly.
* **We are better than the headline on what actually gets asked.** Restricted
  to the labels the official questions name, recall is 0.61 rather than 0.43
  (0.82 on `chinese_room` and `arabic_room`) — but counting error is *worse*
  (9.4 vs 7.1), because the asked-about classes are the common ones we
  duplicate most.
