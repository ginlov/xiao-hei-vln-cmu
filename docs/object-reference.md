# Object reference — where it stands and what to run next

*As of 2026-08-24. Nothing in this workstream has been driven.*

Object reference is **12 of the 51 points** — 3 test scenes × 2 questions × 2
points — scored by 3D IoU against the object's ground-truth box:

| IoU | points |
|---|---|
| ≥ 0.50 | 2 |
| ≥ 0.25 | 1 |
| below | 0 |

---

## The measurement that shapes everything else

The targets are small. Over the 30 released questions, measured in VLA-3D:

| | |
|---|---|
| longest side, median | **0.45 m** |
| under 0.50 m | 17 / 30 |
| under 0.30 m | 6 / 30 |
| largest (`loft` potted plant) | 1.22 m |

For a **perfectly sized** box, the centre error that still earns a point —
measured on each target's thinnest axis, which is where a depth error lands:

| | 1 point | 2 points |
|---|---|---|
| median budget | **0.10 m** | **0.05 m** |
| loosest (`loft` potted plant) | 0.48 m | 0.26 m |
| tightest (`hotel_room_2` picture) | 0.00 m | 0.00 m |

TASK 26 measured VLM grounding from the start pose at a **median of 0.11 m**
with a **p90 of 0.52 m**. The median sits exactly on the one-point line and the
p90 earns nothing on any of the thirty targets.

**That is the whole argument for driving there and looking again.** A modelled
sweep over the real target dimensions, assuming a perfect box size:

| centre error | expected | of 12 points |
|---|---|---|
| 0.11 m (start pose, measured) | 0.80/2 | 4.8 |
| 0.055 m (halved by driving closer) | 1.13/2 | **6.8** |
| 0.03 m | 1.39/2 | 8.3 |

Size matters as much: at 0.7× the true size with the same centre, the same
model gives 2.2 of 12. Both halves have to work.

> The expected-score rows are a **model**, not a measurement — target dimensions
> and the IoU formula are real, the error distribution is a lognormal fitted to
> TASK 26's two published quantiles. The first live runs replace it.

---

## What is built

| file | what it is | driven? |
|---|---|---|
| `scripts/score_reference.py` | the answer key — 30/30 questions resolve to exactly one object | n/a |
| `scripts/target_box.py` | face box → equirect mask → `PointLifter` → multi-view box | no |
| `scripts/reference_view.py` | the orbit re-look prompt and its coercion | no |
| `scripts/answer_reference.py` | the loop: `run_goto` → orbit → box | **no** |

The lift is the perception package's `PointLifter`, with its z-buffer and
cluster gates. `ObjectMap` is deliberately not used — its association is keyed
on the detector's labels — but its **estimator** is, because that part was
measured.

### Three things that are not the obvious design

- **Per-view boxes are averaged, not unioned.** `ObjectMap._observe`: a single
  observation's volume ratio is already 0.95, and pooling points spans the
  object *plus* each view's ~0.23 m centre offset — about 6× too big. With an
  oracle association, an AABB over pooled points scores **0.138 of 2** against
  **0.443** for averaging. *(`ObjectMap`'s class docstring describes the pooling
  version; the implementation does the opposite. The docstring is stale.)*
- **Outliers are settled at read time.** Gating each view against the running
  estimate made a 25-return view from 3.7 m the anchor, which then refused a
  242-return view from 0.8 m. Returns grow as 1/r², so the near views arrive
  last and know most.
- **The consensus is anchored on the binding, not elected by weight.** On the
  `arabic_room` replay the grounding drifted onto a table carrying a coffee pot;
  five of eight views landed there and won 511 to 203, putting the box 0.09 m
  from the coffee pot while the two discarded views sat 0.23 m and 0.20 m from
  the two real stools. Weight measures where the looks went, not what was asked.

---

## Run order

13 of 15 scenes are unpacked. **`hotel_room_1` and `livingroom_4` ship only as
archives** — do not put them in a run list. That leaves 26 questions.

Ordered so each of the first runs isolates one thing:

| # | scene | q | target | 1-pt budget | rivals | what this run is for |
|---|---|---|---|---|---|---|
| 1 | `loft` | 1 | potted plant | **0.48 m** | 2 | Does anything work? Loosest budget of all 26 — a failure here is the pipeline, not precision |
| 2 | `loft` | 0 | chair | 0.30 m | **10** | Same scene, no new variables. Does the re-look **re-identify** or drift? Watch `same_object` and `outliers` |
| 3 | `livingroom_3` | 1 | vase | 0.27 m | 3 | Clean mid-difficulty control, `solid` reading |
| 4 | `hotel_room_2` | 1 | picture | **0.00 m** | 4 | The flat one. It will score 0 — the run is for whether the model says `face_only` |
| 5 | `office_1` | 1 | paper cup | 0.05 m | 2 | The floor: a 0.11 m target. What does the pipeline do when the budget is impossible? |
| 6 | `arabic_room` | 0 | pillow | 0.13 m | **11** | Both hard at once — most rivals *and* a tight budget |

Drive them as **corpus** runs, not answering runs — a superset costs model
calls but no extra simulator time, and it is the only way to get more than one
policy out of one drive:

```bash
scripts/sim.sh restart loft
eval "$(grep -E '^[[:space:]]*export ANTHROPIC_API_KEY=' ~/.zshrc)"
uv run --with anthropic python scripts/answer_reference.py \
    --corpus --views 8 --orbit-deg 40 \
    "Find the potted plant between a vase and the cabinet with a TV on it."
```

`--corpus` disables the settle rule so all 8 views are taken; `--orbit-deg 40`
is finer than the shipped 75°, so both are replayable from the one drive.
Without those flags it answers the question and stops, which is what the
submission does.

Score it against the key:

```bash
uv run python scripts/score_reference.py --why loft     # what the answer should be
```

---

## What each run has to answer

**1. Does the orbit help, and which extent estimator is right?**
Averaging per-view extents is right when each view sees the whole object and
wrong when each sees a different slice. `--size-mode {average,max,union}`; the
default is the measured one. All three can be recomputed from a recorded run.

> ### What replay can and cannot do
>
> **Driving is the only part that cannot be replayed.** The views are a
> function of where the policy chose to stand, so anything that changes a
> viewpoint — the standoff, the orbit step, whether `run_goto` got close
> enough — needs the simulator again. Since the box's accuracy is mostly
> decided *by the viewpoints*, replay is a within-run re-analysis and **not**
> the gate on 3D box accuracy.
>
> Everything downstream of the viewpoint is free, because each step writes its
> four faces and its scan, and `steps.jsonl` carries the model's own `box_px`
> and `image_index`:
>
> | | replayable? |
> |---|---|
> | `size_mode`, view weights, consensus rule | yes, from `answer.json` |
> | **SAM refinement vs the rectangular mask** | yes — same box, new mask, re-lift |
> | `MIN_INLIERS`, `OUTLIER_M`, percentile trim | yes — re-lift |
> | a stopping rule that stops *earlier* | yes, if the drive took more views |
> | a coarser orbit step | yes, if the drive used a finer one |
> | standoff, a finer orbit step, a target never reached | **no — drive again** |
>
> That last row is why the first drives should be **supersets**. `--corpus`
> keeps looking to `--views` after the box has settled, and `--orbit-deg` sets
> the step; a policy that stops at three views or turns 75° instead of 40° is
> then a subset of what was driven and costs nothing more to evaluate. The
> extra cost is model calls, not simulator time.
>
> A test asserts the record is sufficient: a view re-lifted from
> `steps.jsonl` + `stepN_scan.npy` reproduces the driven lift exactly.

**2. Is the re-look re-identifying or drifting?**
Run 2 (`loft` chair, 10 rivals) is the test. The signals are in `steps.jsonl`:
`same_object`, `why_not = cannot_tell_which_instance`, and how many views land
in `outliers`.

**3. Is a rectangular mask good enough, or is SAM needed?**
A rect box pulls background in, which shows up as `size_ratio > 1` against the
key. This decides whether to add a `/segment` endpoint to the sidecar — SAM's
box-prompt path already exists (`pipeline.py:395`), it is just driven only by
YOLO-World today. **The eval machine is a NUC with no discrete GPU**, so the
sidecar would fall back to CPU there; measure before depending on it.

**4. Does `extent` carry real signal?**
Run 4 (`hotel_room_2` picture) should come back `face_only`. If it does, the
four picture questions have a handle; if it comes back `whole`, the field is
noise and the plane-snap has to be triggered some other way.

---

## Known risks, written down rather than fixed

**A box near a face corner lifts almost nothing.** The inverse LUT gives each
equirect pixel to whichever face views it most centrally, so a face's extreme
corner belongs to its neighbour. One 60×60 face box maps to 4814 equirect
pixels near the centre and **186** in the corner; a 41×41 corner box maps to
none. `lift_box` reports `mask_px` and `thin`; nothing acts on it yet.

**The key itself is uncertain on one question.** `studio` q0 — *"the vase
closest to the guitar"* — has a ground-truth box **2.49 m tall** where the
scene's other five vases run 0.17–0.69 m. The *choice* is robust (0.55 m against
4.52 m to the runner-up), but that is where our proxy and the organisers'
scorer are likeliest to disagree. `score_reference.py` prints `ODD GROUND TRUTH`.

**Four questions are geometrically unwinnable as things stand.** The flat
targets — pictures, thickness 0.01–0.02 m — have a 1-point centre budget of
0.00 m. Point-cloud geometry cannot give thickness. The fix is a plane fit with
a minimum thickness on the normal, keyed on `extent == "face_only"`, and it is
not built.

**The axis-aligned ceiling may or may not apply.** TASK 25: our boxes are
world-axis-aligned by construction, and against a ground truth built with the
heading ignored a *perfect* prediction caps at mean IoU 0.783, with 6.5% unable
ever to reach 0.5. But that convention is `perception/eval.py`'s, not
necessarily the organisers'. `answer_reference` computes a PCA heading and
reports it; publishing it is an A/B nobody has run.

**`XIAO_HEI_SCAN_KEYFRAMES` defaults to 10** in `app/main.py` while TASK 21
measured 0 as better on both its corpora. `tests/test_perception_settings.py`
pins 10 deliberately. This path builds `PointLifter` directly and is unaffected
either way, but anything comparing against the perception responder is not.

---

## Not built

- The `/segment` sidecar endpoint (external box → SAM mask, single face).
- The flat-object plane snap.
- The `heading` A/B.
- Wiring `answer_reference` into `challenge_node.py`. PR #1 currently hands
  reference questions to the teammates' Gemini pipeline via `hand_over`; that
  stays until this has been driven and scored.
