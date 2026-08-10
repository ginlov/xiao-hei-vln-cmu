# VLM grounding test — plan for the next session

One question decides whether the perception stack keeps its current shape:

> Can a VLM point at **"the tea table with the elephant figurine on it"** in our
> own 360° imagery, and can that pixel become a metric waypoint through the
> lidar?

If yes, it bypasses in one move the three limits measured over TASK 23–25: the
110-class vocabulary, the synonym splitting, and the 67% retrieval ceiling. If
no, we fall back to a narrower detector vocabulary, which is cheaper but capped.

## Why this is the deciding experiment

Retrieval over the fused object map finds the objects the fourteen
instruction-following questions name only **67% of the time** — and the misses
are not detection failures. Splitting them:

| cause | count | fixable by |
|---|---|---|
| word absent from the 110-class list (`stairs`, `whiteboard`, `exit sign`, `water cooler`, `projector screen`, …) | **14 (74%)** | open vocabulary |
| in the vocabulary, still not found (`cabinet`, `tray`, `tv remote`, `fireplace`, `folder`) | 5 (26%) | detection / association |

Two further measured facts set the bar:

* Instruction following is **6 of the 17 points per scene — about 70% of the
  challenge score** — and it is answered with waypoints, not boxes.
* The reference trajectories pass a median **0.58 m** from the objects they
  name, and our fused object centres are already accurate to **0.236 m**
  laterally. Localisation does not need to be precise. It needs to *exist*.

So the prize is coverage, not accuracy — which is exactly what a VLM offers and
a fixed class list cannot.

## What to run

### Part A — can it identify?

Show the VLM the four perspective faces (`pipeline.py` already unwraps the
equirect with precomputed LUTs) and the referring expression **verbatim** from
the question. Ask for the face index and a `box_2d`, or `NOT_VISIBLE`.

Do **not** pre-detect, pre-filter, or offer candidates. The point is to test the
VLM against the raw image, not to re-rank what YOLO already found.

* **Test set**: the 60 (question, named object) pairs across the 7 scenes that
  have corpora — the same set `scripts/retrieval_probe.py` scores, so the
  numbers are directly comparable.
* **Positions to score against**: ground-truth centres from the scene zips.
* **Sample the same frames** the retrieval probe used as cuts: the start pose,
  then after 1.3 m, 4.8 m and 12.4 m of travel.

### Part B — can the pixel become a waypoint?

**Ask for a box, not a point.** The Gemini API returns object detection
natively as `{"box_2d": [ymin, xmin, ymax, xmax], "mask": [[x,y], …], "label"}`
with coordinates normalised to **[0, 1000]** — descale by the image dimensions.
A box centre is steadier than a single point, it prompts SAM exactly the way
YOLO's box does today, and if the `mask` polygon proves usable it replaces SAM
outright.

Run two variants, because they fail for different reasons:

* **B1 — bare pointing.** Box centre → equirect pixel → ray → median depth of
  the lidar returns in a small angular window. Measures the VLM alone.
* **B2 — production shape.** Box → SAM → mask → `PointLifter`, unchanged.

**Lateral precision is not the risk.** At 1920×640 over 360° the equirect gives
5.33 px/deg, so keeping lateral error under 0.58 m allows:

| object distance | angular budget | pixel budget |
|---|---|---|
| 2 m | 16.2° | **±86 px** |
| 3 m | 11.0° | ±59 px |
| 5 m | 6.6° | ±35 px |
| 10 m | 3.3° | ±18 px |

Named objects sit mostly 2–6 m out, so 35–86 px of slop is tolerable — generous
for a clearly visible object.

**Depth is the risk.** Twenty pixels of drift costs 0.1 m laterally, but if
those twenty pixels slip past the object's silhouette the ray lands on the wall
behind and the error along the ray is unbounded. `PointLifter` already carries
`_dominant_depth_cluster` for exactly this — a mask whose edge catches
background. Which is why **B1 failing while B2 succeeds is an informative
result**: the VLM found the object and only the naive depth lookup was wrong.
Only both failing rules the approach out.

## Acceptance criteria — fix these before looking at results

| | criterion | baseline it must beat |
|---|---|---|
| **A** identification rate | **> 67%** of named objects pointed at correctly | retrieval at k=3 |
| **B** localisation error | median **< 0.58 m** for those it identified | reference-trajectory tolerance |
| **A+B** usable rate | **> 67%** localised within 1.0 m | the ceiling this replaces |
| **false positive rate** | **< 10%** confidently pointing at the wrong object | — |

The last row matters more than it looks. A miss costs one question; a confident
wrong answer sends the robot somewhere else and, if the constraint is marked
satisfied, corrupts the ordering for everything after it. `NOT_VISIBLE` must be
a first-class answer and the prompt must make it easy to give.

**Decision rule.** A ≥ 67% and B < 0.58 m → VLM grounding becomes the primary
path and the detector demotes to fallback. A between 50% and 67% → keep YOLO
with a question-specific vocabulary (`scripts/narrow_vocab_probe.py`, written
and ready) and use the VLM only for compositional disambiguation. A < 50% →
drop the VLM grounding idea and pursue narrow vocabulary alone.

## The worked hard case: chinese_room q5

> First, go near the **tea table with the elephant figurine on it**, then stop
> at the **table with the horse figurine on it**, avoiding the path between the
> **chair** and the **folding screen**.

Retrieval localises **1 of 5** named objects here, and stays at 1 of 5 even
after the full tour. What the map actually holds:

| named object | ground truth | in the map |
|---|---|---|
| tea table | (−0.02, −2.20) | only as part of a 50-node `table` family; correct cluster ranks **3rd** |
| elephant figurine | (0.01, −2.17) | **absent** — too small to lift |
| table (horse) | (6.23, −2.59) | **absent from the top 6 candidates** |
| horse figurine | (6.31, −2.08) | present, best node **0.14 m** away |
| folding screen | (6.29, 0.18) | **absent** |
| chair | ×6 | 13 clusters — duplicated 2× |

Three distinct failures worth watching for separately:

1. **The disambiguator is invisible to geometry but visible in the image.** The
   elephant figurine cannot be lifted, yet a person can see which table it sits
   on. This is the case where a VLM is not merely better but *necessary* — no
   geometric feature separates five tables.
2. **The target itself is missing** (the horse table). Tests whether the VLM
   finds what detection never proposed.
3. **An avoidance anchor is missing** (the folding screen). Without it the
   keep-out region cannot be built, and the challenge penalises entering
   forbidden areas.

If the VLM handles all three here, the architecture question is settled.

## Preparation

1. **Start xiaohei1.** The public IP changes on every restart — update
   `~/.ssh/config` and `ssh-keygen -R <old-ip>` if the host key check fails.
   `uv` is not on the non-interactive PATH: prefix with
   `export PATH=$HOME/.local/bin:$PATH`.
2. **Gemini key** into `.env` as `XIAO_HEI_GEMINI_API_KEY` (`.env` is
   gitignored; `gemini/config.py` requires it and fails loudly without it).
   Official FAQ allows online APIs, but the token must be shipped with the
   submission — so use a revocable key with a quota cap, not a personal one.
3. **Copy the sampled frames locally** (~50–100 MB) so later iterations need no
   server:
   `rsync -a xiaohei1:.../frames/<scene>_tour/tick_00000*.jpg viz/frames/`
   (macOS ships openrsync — GNU-only flags like `--info=progress2` will fail.)

### Known gap to fix first

`scripts/export_viz.py` records only frame **position**, not heading, so any
bearing computed from `viz/data` is in the map frame rather than egocentric.
Add `yaw` (via `replay._yaw` on `image_pose_orientation`) before anything
depends on egocentric angles. The grounding test itself does not need it — the
image is already egocentric, which is why the prompt should identify candidates
by *face index*, not by bearing.

## Also ready to run

`scripts/narrow_vocab_probe.py` — the cheaper competing hypothesis, no API
needed. It re-runs stage A with the question's own words pushed through
`YOLOWorld.set_classes` (open-vocabulary already; the 110 classes are our own
choice, and `detect()` accepts a per-call list at ~50 ms to re-encode). It
writes to `detections_<tag>.jsonl` and never touches the 110-class cache.

Accept if k=3 recall reaches **80%** and k=1 does not fall below **60%** —
recall bought by forcing every table-ish surface into the one class asked for
is not a gain.

**Run this first if GPU time is free**: it is cheaper than the API test and, if
it reaches 85%+, Part A's value drops to compositional disambiguation only.

## Reference numbers

| measurement | value | source |
|---|---|---|
| retrieval recall, k=1 / k=3 / k=10 | 60% / 67% / 67% | `scripts/retrieval_probe.py` |
| named objects localised at the start pose | 48.3% | `scripts/start_pose_visibility.py` |
| … after 1.3 m / 4.8 m / 12.4 m | 60.0% / 63.3% / 66.7% | same |
| reference-trajectory distance to named objects | median 0.58 m, p75 0.81 m | `scripts/traj_tolerance.py` |
| our fused centre error | 0.236 m lateral, +0.004 m radial | TASK 24 |
| reference trajectory length | 5.0–19.1 m, mean ≈ 10 m | `scripts/traj_tolerance.py` |
| instruction following share of score | 12 of 17 points per scene (~70%) | challenge README |

`k` plateaus at 3: k=5 and k=10 add nothing, so three candidates is the right
prompt size wherever candidates are offered at all.

## Open question for the organisers

"No exploration in advance" was the answer to whether we may drive with
`/way_point` before publishing the answer path. Under the strict reading the
robot's entire motion is the answer. Worth confirming, since it decides whether
exploration must satisfy constraints as it goes:

> For instruction-following, after the question is published, may the robot move
> to observe the scene before executing the answer path — and does that motion
> count toward the scored trajectory?

Measured mitigation either way: the reference trajectories are only ~10 m long,
and localisation plateaus by 4.8 m of travel, so executing the answer path
already reveals nearly everything the map will ever hold.
