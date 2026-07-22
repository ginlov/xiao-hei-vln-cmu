# TASK 9 — Expand ref corpus to 12k & match the official phrasing distribution

## Goal

Grow `object_reference` from 6,350 → ~12,000 **and** reshape its phrasing
to match the official question set
(`../CMU-VLN-Challenge-2026/questions/questions.json`, 30 graded
object_reference items, characterised in TASK 8 Part D). User chose a
12k total with nested ≈ 50%.

## Official vs ours (before → after)

| Feature                       | Official | Before (6.35k) | After (12k) |
|-------------------------------|---------:|---------------:|------------:|
| compositional (≥2 relations)  |   57%    |   15%          |   56%       |
| color modifier                |    7%    |   21%          |    7%       |
| indefinite "a X" anchor       |   13%    |    0%          |   13%       |
| omit-"Find" ("The X …")       |   10%    |    0%          |   10%       |
| ordinal ("second closest")    |    0%    |   20%*         |    0%       |

(*ordinals were already removed in TASK 8 Part D; listed for contrast.)

## What changed

### 1. Nested supply lifted past its geometric ceiling (`vla3d_nested_gen.py`)

The old patterns (`on/above/below × closest/farthest` + `between`) top out
at ~4.7k pairs even uncapped — not enough for a 50% nested share of 12k.
Fix: add **`near` as an inner relation** ("the cup near the laptop closest
to the window") — `near` has 39k scene-graph entries, exploding the raw
pool to ~15k. Also added `hanging_on`/`in` inner relations, `near` as an
outer relation (unique-within-radius, distinct geometry from `closest`),
and a same-label guard (no "the X … near the X"). Architectural labels
(ceiling/`celling`/window/wall/floor/door/curtain) are excluded as
*targets* (vacuous "the ceiling above the spoon") but still allowed as
anchors. The raw pool is then **pattern-stratified-downsampled** to
`TARGET_NESTED_REF = 6000` (seed 42) for per-pattern/anchor diversity.

### 2. Ill-formed definite anchors → "a X" (`vla3d_ref_to_qa.py`)

TASK 8 Part C *dropped* bare "the X" anchors with >1 X in region. Now they
are **softened to "a X"** (`soften_anchor`), recovering the samples and
covering the official 13% indefinite-anchor style. Guards: skip when the
anchor shares the target's class (VLA-3D writes "the other X", well-formed),
skip plural labels (no "a books"), word-boundary match (no "the box" inside
"the boxes").

### 3. Quota sampler replaces per-relation rebalance (`vla3d_ref_to_qa.py`)

`sample_single_layer` draws `TARGET_SINGLE = 6000` from three disjoint
buckets — a-anchor (priority, `A_ANCHOR_QUOTA = 1560`), color
(`COLOR_QUOTA = 840`), plain — so feature shares are controlled directly.
Color detection uses a strict colour-word regex (excludes "dark"/"light",
which matched "light switch"/"spot light").

### 4. omit-"Find" variant (`phrasing.py`, shared)

`apply_omit_find` rephrases ~10% of object_reference from "Find the X …"
to "The X …". Only "Find the" → "The" (never "Find a" → "A", which the
classifier would misroute).

### 5. Runtime classifier fix (`messages/question.py`) — real competition bug

`classify_question()` routed any non-"Find"/"How many" text to
instruction_following, so the **3/30 official object_reference questions
that read "The red pillow closest to …" would be misclassified at
competition time** (wrong response topic → lost points). Fixed: a leading
"the " → object_reference. Verified safe — all 30 official
instruction_following items start with Go / First / Take, never "the".
Mirrored in `check_question_types.py`'s copy of the heuristic.

## Verification

- `check_question_types`: 12,591 pairs, **0** type mismatches.
- Final corpus: `vla3d_ref.jsonl` 12,000 (6,000 single + 6,000 nested),
  `vla3d_num.jsonl` 591. Distribution table above.
- 0 ordinals, 0 ungrammatical "a <plural>".
- `pytest`: 173 passed. `ruff`: new code clean (pre-existing E701/E702/
  B007 one-line style untouched).
- Determinism preserved (seed 42 throughout).

## Numerical (second pass — matched to the 15 official numerical items)

Official numerical is **`on`-dominated** (11/15), color 13% (2/15, all
"on"), compositional 20% (3/15), one "Count the number of …" phrasing,
and **zero** pure totals / refusals. Our old 591 was the opposite: 19%
`on`, 1% color, 36% pure-total (N4) + refusal (N5), `near`-heavy.

Changes:
- **`on` count includes answer 1** (`emit_relation_count` `min_count=1`) —
  official "How many red pillows are on the sofa?" counts 1.
- **`NUM_TEMPLATE_CAP`** reweights per template: keep all `on` (N1) and
  color-on (N6, capped to ~13%), suppress `near` (N2→1/scene), small
  above/below/hanging.
- **N4 disabled** (pure "in the room" totals — OOD); **N5** kept token.
- **Nested num** shrunk (`TARGET_NESTED_NUM` 205→40) and stripped of any
  `near` template, so the merged num stays `on`-leaning and compositional
  ≈ 20%.
- **"Count the number of …"** phrasing (`phrasing.apply_count_phrasing`,
  ~9% pre-merge → 7% final), plus a classifier route for leading "Count"
  (another official-form the heuristic previously misrouted).

Result (190 pairs): on 67% / near 6% / color 14% / compositional 22% /
Count 7% / OOD 8%. num shrank 591→190 deliberately: VLA-3D has only ~84
scene-unique "on" anchors, so a faithful on-heavy set is small — padding
it back to 591 would re-introduce the `near`/totals skew. `check_question_types`:
12,190 pairs, 0 mismatches; pytest 173 passed.

## Not done / follow-ups

- num is **supply-capped at ~190**. Growing it on-distribution would need
  relaxing anchor uniqueness (region-unique instead of scene-unique) to
  mine more "on" anchors — not done.
- Relation-word *shares within* nested ref skew toward `near` (inner). The
  ≥2-relation **structural** target (57%) is met; exact per-relation
  percentages are not separately tuned (over-fitting to small samples).
- `instruction_following` generation still not started (needs official
  forbidden-zone labels).
