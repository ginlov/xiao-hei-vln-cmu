# dataset_generator

Generates the training Q&A corpus for Team Xiao Hei's VLM from the
[VLA-3D](https://github.com/HaochenZ11/VLA-3D) Unity subset. The
generators live here; the data they emit lands in a sibling `dataset/`
directory (gitignored).

The 15 Unity scenes covered by VLA-3D are the same scenes the CMU VLN
Challenge uses for training, with object IDs aligned 1:1 to the
challenge's `object_list.txt`. We exploit that alignment to derive
5,055 grounded Q&A pairs across the challenge's two scoreable runtime
types (the third type, `instruction_following`, requires the official
forbidden-zone labels and is tracked separately):

| Source file            | Pairs  | Composition |
|------------------------|-------:|---|
| `vla3d_ref.jsonl`      | 4,865  | 3,500 single-layer `object_reference` rewrites + 1,365 nested-pattern ref pairs |
| `vla3d_num.jsonl`      |   190  | 150 numerical templates (N1–N8) + 40 nested-pattern num pairs |
| **Total**              | **5,055** | — |

The corpus shrank from an earlier 12,190 when relation-word frequencies
were aligned to the official shape (see *Relation-word usage* below): a
`closest`-led, `near`/`farthest`-minority corpus is supply-capped at ~5k,
whereas the old 12k was only that large because it was `near`/`farthest`-
heavy — the relations VLA-3D supplies in bulk but the official set barely
uses.

### Matching the official phrasing distribution

Both halves are shaped to the official question set
(`../CMU-VLN-Challenge-2026/questions/questions.json`, 30 object_reference
+ 15 numerical graded items). `object_reference` (measured vs target):

| Feature                          | Official | Ours (~5k) |
|----------------------------------|---------:|-----------:|
| compositional (≥2 relations)     |   57%    |   28%      |
| color modifier ("the red X")     |    7%    |    7%      |
| indefinite "a X" anchor          |   13%    |   ~11%     |
| omit-"Find" ("The X …")          |   10%    |   10%      |
| ordinal ("second closest")       |    0%    |    0%      |
| redundant constraint (see below) |   ~3%    |    0%      |

The **compositional** share fell from 50% to 28% as a direct trade-off of
the relation-word reweight: compositional questions are almost all nested,
and nested supply is overwhelmingly `near`/`farthest`, so capping those to
match the official relation mix also caps the nested (compositional) count.
The two official targets — 57% compositional *and* `on`/`closest`-heavy —
cannot both be hit from VLA-3D, whose compositional supply *is* the
`near`/`farthest` geometry. We prioritized the relation shape.

`numerical` (measured vs target):

| Feature                         | Official | Ours |
|---------------------------------|---------:|-----:|
| `on` relation                   |   73%    |  67% |
| `near` relation                 |    7%    |   6% |
| color modifier                  |   13%    |  14% |
| compositional (≥2 relations)    |   20%    |  22% |
| "Count the number of …" phrasing|    7%    |   7% |
| pure totals / refusals          |    0%    |   8% |

The `on` share is capped by VLA-3D supply: only ~84 scene-unique "on"
anchors exist, so num is intentionally small (190) and faithful rather
than padded with abundant `near` counts (the old 591 was 36% pure totals
+ refusals and `near`-heavy). The official set has no pure totals
("How many X in the room?"), so the `N4` emitter is disabled; a token
`N5` refusal slice (answer 0) is kept for robustness.

#### Relation-word usage (ref + num)

Occurrences of each spatial-relation word across all question strings (one
nested question contributes ≥2). The default corpus was supply-driven and
~29% `near` / ~29% `farthest`; the reweight makes `closest` the primary
outer and pushes `near`/`farthest` down to minority shares:

| Relation word       | Official % | Ours (default) | Ours (reweighted) |
|---------------------|-----------:|---------------:|------------------:|
| on                  | 42.4%      |  5.2%          | 11.6%             |
| closest             | 28.8%      | 23.8%          | 40.0%             |
| near                |  7.6%      | 29.2%          |  9.9%             |
| between             |  6.1%      |  3.5%          | 11.1%             |
| above               |  6.1%      |  2.3%          |  5.6%             |
| farthest / furthest |  4.5%      | 29.3%          | 10.4%             |
| below               |  3.0%      |  1.5%          |  4.0%             |
| under / in / hanging on | 1.5%   |  1.1%          |  3.4%             |
| beside / next to / adjacent to | 0% | 4.0%        |  3.9%             |

`on` cannot reach its 42% official share — same ~84-anchor supply ceiling
as numerical — so `closest` (abundant, clean) absorbs the primary-relation
role at 40%. The reweight is **moderate by design**: it fixes the gross
`near`/`farthest` over-representation (a real artefact of using `near`-as-
inner to lift nested supply) and matches the official *ranking*, not the
exact percentages of a 66-occurrence sample. Knobs: `REL_CAP`
(`vla3d_ref_to_qa.py`) and `NEAR_INNER_CAP` / `FARTHEST_TEMPLATE_CAP` /
`NEAR_OUTER_TEMPLATE_CAP` (`vla3d_nested_gen.py`).

The knobs live in `vla3d_ref_to_qa.py` (`TARGET_SINGLE`, `COLOR_QUOTA`,
`A_ANCHOR_QUOTA`, `OMIT_FIND_FRAC`), `vla3d_num_gen.py`
(`NUM_TEMPLATE_CAP`, `COUNT_PHRASING_FRAC`) and `vla3d_nested_gen.py`
(`TARGET_NESTED_REF`, `TARGET_NESTED_NUM`). Nested ref supply is lifted
past its old ~4.7k geometric ceiling by using `near` as an *inner*
relation ("the cup near the laptop closest to the window"), then
pattern-stratified-downsampled to 6,000 for diversity.

Single-layer ref pairs whose anchor is an **ill-formed definite
description** — a bare "the X" when the region holds more than one X and
VLA-3D supplied no color/size disambiguator (e.g. "the keyboard" in an
office with 6 keyboards) — are **softened to "a X"** rather than dropped:
the target stays geometrically unique, and the indefinite article no
longer wrongly presupposes a unique keyboard. This is the official set's
own style ("closest to a window"). Anchors with a disambiguator ("the
BLUE book") keep the definite article.

**Redundant-constraint gate.** A spatial/attribute constraint only earns
its place when the target class is *non-unique* in what the robot sees —
otherwise "Find the dvd beside the chair" is over-specified when there is
just one dvd. In the official set **29/30** object_reference targets are
non-unique (pillow×12, bowl×7, blue chair×11, …); only 1 is unique. Our
old corpus was the opposite: ~37% of pairs had a target class that was
already unique in view, so the relation did no disambiguation work. Both
generators now drop these: a ref pair is emitted only if **≥2 instances of
the target class** sit in the object_list it carries (scene-wide for
nested, which ships a scene-wide list; region-wide for single-layer, whose
list is region-filtered). Result: **0%** redundant (down from 37%).
Counting (`numerical`) questions are exempt — a count is meaningful no
matter how many same-class objects exist.

This gate is what pushed the indefinite "a X" anchor share down to ~11%
(from the 13% target): most single-anchor "a X" statements happen to have
a region-unique target and are now dropped. We kept the gate — matching
the official "constraints disambiguate" semantics matters more than the
secondary "a X" stylistic share. The gate also naturally concentrates
single-layer ref on `closest`/`farthest`/`near`/`between` (relations that
*require* multiple same-class instances), which is itself on-distribution.

**Superlative margin gate** (`superlative_margin_ok`, `SUPERLATIVE_MARGIN_M
= 0.30`). A "closest/farthest/near to Y" question is only well-posed if the
target is *visibly* the nearest/farthest of its same-class candidates. VLA-3D
statements are uniquely-referring but can tie in the xy-plane the ground robot
navigates: e.g. two `file` objects stacked at the same (x,y) are exactly
equidistant from any anchor, so "the file nearest the plant" has no determinate
answer. **~14%** of single-layer superlatives were within 5 cm (and ~24% within
15 cm) — unanswerable supervision, worse for the robot's low-res 360 camera.
The gate keeps a single-layer superlative only if the target is the xy-nearest
(or -farthest) candidate by ≥ 0.30 m (matching the nested generator's
`CLOSEST_MARGIN_M`), dropping ties to **1.4%**. It also drops samples where
VLA-3D's chosen answer disagrees with xy-geometry (it ranks by 3D distance).

**Wall-between gate** (`wall_between_ok`, `closest`/`near` only). Straight-line
xy distance through a wall isn't navigable proximity: "the cabinet closest to
the fire alarm" was picking a cabinet on the far side of a glass partition.
~23% of single-layer `closest`/`near` had the target→anchor segment cross a
structural wall. The gate drops these — it clips the segment against each
structural wall's xy footprint (height-filtered so flat "wall decal" / "wall
lamp" fixtures don't count; glass partitions *do*, they're real navigation
barriers) and rejects a crossing in the segment's middle portion (parameter
`t ∈ (0.08, 0.92)`, so a wall-mounted anchor at `t≈1` or an object backed
against its own wall at `t≈0` is not mistaken for a divider between them).
`farthest` is exempt — the farthest object is naturally across the room.
Residual wall-crossing in the final `closest`/`near` set: **0**.

**Colour gate + basic-word mapping** (`color_gate_and_map`). A colour modifier
should name what the object *looks like*. VLA-3D disambiguates near-identical
objects by a **minority** colour — "the blue book" was a 76%-gray book with 18%
blue (it sits among seven gray books distinguished only by their 2nd colour) —
which perception can't ground; ~37% of colour modifiers covered <30% of the
object. The gate keeps a colour word only if it is the object's **dominant**
colour at **≥ 40%** (`COLOR_DOMINANT_MIN`), using VLA-3D's structured
`target_color_used` / anchor `color_used` rather than text-matching. It then
maps VLA-3D's technical palette to the **basic** words the human-authored
official set uses (`maroon→red`, `navy`/`teal`/`aqua→blue`, `olive→green`,
`beige`/`tan→brown`, `violet→purple`): a maroon-100% pillow becomes "the red
pillow" — exactly how the official set refers to it (VLA-3D labels those red
pillows "maroon"). Net colour share stays at **7%** (enough dominant-colour
supply remains); zero technical-palette or minority-colour modifiers survive.

**Ordinal-ranked phrasings** ("second closest", "third farthest", …) are
dropped: the official set uses only superlatives and contains zero
ordinals, so these were out-of-distribution.

A `~10%` slice of object_reference questions drop the leading "Find"
("The red pillow closest to the sushi.") to mirror the 3/30 official
items phrased that way. This required teaching the runtime
`classify_question()` that a leading "The …" is object_reference (the
official instruction_following items always start with an action verb —
Go / First / Take — never "the").

Both `object_list` labels and the question are normalized to
VLA-3D's class vocabulary so the model can ground the question noun
(object_result.csv's `raw_label` of `tv` is rendered as the statement's
`television`); same-class distractors share the target's label so the
spatial relation, not a unique label, must disambiguate.

Nested patterns (inner relation × outer closest/farthest, plus
`between`) are *not* a runtime type — the official challenge has only
three categories, and runtime `classify_question()` routes by question
prefix only. They live inside the two type-aligned files above and are
identifiable via the `source` field (`source == "vla3d_nested"`), so
nested-only metrics are still possible at evaluation time.

All `object_list` lines and question routing match the runtime
contract in `src/xiao_hei_vln/messages/`:

- `object_list` line schema: `id cx cy cz lx ly lz heading label` —
  matches `src/xiao_hei_vln/dummy/data/object_list.txt`.
- `type` field on every pair agrees 100% with the runtime
  `classify_question()` heuristic (see `check_question_types.py`).

## One-time setup: download VLA-3D

The 2 GB raw data is **not committed** (it isn't in the VLA-3D GitHub
repo either — it lives on CMU AirLab's public bucket). Fetch it once
with the helper, which downloads `Unity.zip` and unpacks it into the
default location:

```bash
uv run python dataset_generator/download_vla3d.py
# → dataset_generator/vla-3d/Unity/<scene>/...   (the loader's default)
```

To put it elsewhere and point the scripts at it instead:

```bash
uv run python dataset_generator/download_vla3d.py --dest /data/vla-3d
export VLA3D_ROOT=/data/vla-3d/Unity
```

The pipeline expects each scene to be a directory under `$VLA3D_ROOT/`
containing at minimum:

```
<scene>_object_result.csv
<scene>_scene_graph.json
<scene>_referential_statements.json
```

## Generate the corpus

One shot:

```bash
./dataset_generator/regen.sh
```

Or step-by-step (equivalent):

```bash
uv run python dataset_generator/vla3d_ref_to_qa.py     # → dataset/vla3d_ref.jsonl     (~67 MB)
uv run python dataset_generator/vla3d_num_gen.py       # → dataset/vla3d_num.jsonl     (~3 MB)
uv run python dataset_generator/vla3d_nested_gen.py    # → dataset/vla3d_nested.jsonl  (~14 MB, intermediate)
uv run python dataset_generator/check_question_types.py # auto-merges nested + sanity-checks
```

Nested samples are not a separate runtime type (`classify_question()`
only knows `numerical` / `object_reference` / `instruction_following`),
so the nested generator's output is an *intermediate file* that gets
folded back into the two type-aligned jsonl files. Both downstream
consumers — `check_question_types.py` and `split_and_dump.py` —
**auto-merge** `vla3d_nested.jsonl` into `vla3d_ref.jsonl` /
`vla3d_num.jsonl` at startup if they find it still on disk, so the user
never has to remember the merge step. `merge_nested.py` also exists as
an explicit CLI entry point for CI / batch-style invocation; it is
idempotent (no-op when nested.jsonl is already absent).

All steps are deterministic — seed `42` is fixed, so the same VLA-3D
source data + this code reproduces byte-identical jsonl. The
`dataset/` output is gitignored; regenerate or fetch it from a release
artifact when training.

## Visualize a sample (sanity-check generation)

Each jsonl row baked an `object_list` (all OBBs in the scene) plus a
`target` / `anchors` / `distractor_ids`. To verify these line up with
real geometry — i.e. target is on the right object, anchors actually
match the `closest to X` relation, heading isn't flipped — render any
row as a 3D OBB wireframe scene:

By default the viewer overlays the scene **point cloud** on a **dark
background** with **2 cm tube** OBB edges and a **0.02 m** voxel
downsample — the most readable setup out of the box. Opt out with
`--no-pointcloud` / `--no-dark-bg` or override the numeric knobs.

```bash
# One-time install (heavy ~400 MB Open3D wheel; opt-in only)
uv sync --extra viz

# Interactive (drag to rotate, scroll to zoom) — point cloud + dark bg by default:
uv run python dataset_generator/visualize_sample.py \
    --jsonl dataset/vla3d_ref.jsonl --idx 42

# Random row:
uv run python dataset_generator/visualize_sample.py \
    --jsonl dataset/vla3d_ref.jsonl --random

# Single PNG for sharing in a PR/issue:
uv run python dataset_generator/visualize_sample.py \
    --jsonl dataset/vla3d_ref.jsonl --idx 42 --save out/

# Spot-check N samples per scene (headless batch):
uv run python dataset_generator/visualize_sample.py \
    --jsonl dataset/vla3d_ref.jsonl --sample-per-scene 20 --save out/

# Lighter boxes-only view (no point cloud, white background):
uv run python dataset_generator/visualize_sample.py \
    --jsonl dataset/vla3d_ref.jsonl --idx 42 --no-pointcloud --no-dark-bg
```

Options:

| Flag | Default | Effect |
|------|---------|--------|
| `--jsonl PATH` | (required) | jsonl file to read rows from |
| `--idx N` / `--random` | — | pick a specific 0-based row / a random one |
| `--sample-per-scene N` | — | batch: up to N rows per scene (needs `--save`) |
| `--seed N` | 42 | RNG seed for `--random` / batch sampling |
| `--pointcloud` / `--no-pointcloud` | **on** | overlay the scene point cloud |
| `--point-size F` | 2.5 | point size in px (bump to 4–5 for chunky) |
| `--voxel-size F` | **0.02** | voxel downsample in m (0 disables) |
| `--gray-points` | off | force uniform gray instead of native RGB |
| `--dark-bg` / `--no-dark-bg` | **on** | dark background so colors/OBBs pop |
| `--ceiling-cut F` | 0.5 | crop top F m of cloud so the ceiling stops occluding |
| `--line-radius F` | **0.02** | OBB edge thickness in m, drawn as tubes (0 = 1px wire) |
| `--show-other` | off | also draw gray context OBBs for every other object |
| `--save PATH` | — | write PNG (dir → auto-named); omit for an interactive window |
| `--scene-root PATH` | bundled | where the VLA-3D Unity scenes live |

In an interactive window: `+` / `-` resize points, `Q` / `Esc` / `Ctrl+C`
close it.

Color code:

- **red** — `target` (the answer object)
- **yellow** — `anchors` (objects referenced in the question)
- **cyan** — `distractor_ids` (same-class non-answers, ref samples only)
- **gray** — everything else in `object_list`, for spatial context

Wireframes are **OBB** (heading-aware), so a wrong `heading` shows up as
a visibly mis-rotated box. Note: our in-repo evaluator computes IoU as
**AABB** (`src/xiao_hei_vln/evaluator/metrics/object_reference.py`); the
official challenge does not publish its evaluation code, so the AABB/OBB
choice on the leaderboard side is not confirmed.

Saved PNGs come with a baked-in **header strip** (`#idx`, `scene`,
`type`, the question, and the answer) plus a **legend** in the
bottom-right corner showing only the role colors actually present in
that sample. Interactive windows show `scene` + question prefix in the
title bar; the full Q/A and role IDs are printed to terminal.

## Build train/val/test splits

```bash
# Single 10/3/2 scene split
uv run python dataset_generator/split_and_dump.py --seed 42

# Or 5-fold scene-level cross-validation
uv run python dataset_generator/split_and_dump.py --kfold 5 --seed 42
```

Splits are written under `dataset/splits/`.

Scene-level (Group K-Fold) splitting prevents leakage: every sample
from a given scene lands in the same split, so test accuracy reflects
true generalization to unseen scenes.

## Layout

```
dataset_generator/          # generator code (committed)
├── download_vla3d.py       Fetch + unpack the VLA-3D Unity subset (~2 GB) into vla-3d/
├── vla3d_loader.py         Unified VLA-3D scene loader (objects, scene_graph, ref statements)
├── vla3d_ref_to_qa.py      Rewrite VLA-3D ref statements → 6,730 "Find …" pairs
├── vla3d_num_gen.py        8 numerical templates (count, color-conditioned, refusal)
├── vla3d_nested_gen.py     Two-stage nested patterns: inner relation × outer closest/farthest
├── merge_nested.py         Fold nested.jsonl into ref/num jsonl by type (auto-called by consumers; CLI also)
├── regen.sh                One-shot driver: runs the three generators + sanity check (merge is implicit)
├── noise_augment.py        Perception-noise library (drop/swap/jitter, target-protected)
├── split_and_dump.py       Scene-level single split + K-Fold
├── check_question_types.py CI sanity: jsonl `type` ↔ runtime classify_question()
├── visualize_sample.py     Render one or many jsonl rows as 3D OBB wireframe (Open3D; opt-in viz extras)
└── vla-3d/Unity/           VLA-3D source data — INPUT (NOT committed; download per above)

dataset/                    # generated output (NOT committed)
├── vla3d_ref.jsonl         single-layer ref + nested ref (identifiable via `source`)
├── vla3d_num.jsonl         numerical templates + nested num (identifiable via `source`)
└── splits/                 train/val/test (or fold_*/) + manifest.json
```

## Integrating into a responder

Each jsonl row carries everything a fine-tuned LLM needs:

```jsonc
{
  "scene": "loft",
  "type": "object_reference",
  "question": "Find the lamp on the table closest to the potted bamboo.",
  "object_list": [
    "0 5.91 -1.46 0.31 0.40 0.23 0.05 -0.02 dvd",
    "5 -0.84 -3.55 4.06 0.71 0.17 0.65 -2.80 lamp",
    // ...
  ],
  "answer": {"object_id": 5, "label": "lamp"},
  "target": 5,
  "anchors": [22, 30]
}
```

At inference time, the responder feeds the perception module's
`object_list` plus the question to the model; the model emits an
`object_id` (or `count` / `waypoints`), and the responder looks the
bbox up in the perception list before publishing
`ObjectReferenceResponse`.

## Assumptions worth knowing

- **Full-scan scene context.** Every `object_list` baked into the
  jsonl is the *whole scene*, on the assumption that the deployed
  robot does a full sweep before answering. If the exploration
  strategy changes later (region-driven, etc.), `noise_augment.py`
  is the seed for a viewpoint sampler that perturbs this view at
  training time.
- **Instruction-following pairs are not generated yet.** Building
  those out is tracked separately.
