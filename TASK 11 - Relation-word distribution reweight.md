# TASK 11 — Reweight the relation-word distribution toward the official shape

## Problem

A PR-description table comparing relation-word usage (official ref+num vs our
generated ref+num) exposed a large mismatch the earlier "distribution
alignment" (TASK 9) never touched — it tuned *structural* features
(compositional %, color %, `a X` %, omit-`Find` %) but not *which relation
words* appear:

| relation | official % | ours (before) |
|----------|-----------:|--------------:|
| on       | 42.4%      |  5.2%         |
| closest  | 28.8%      | 23.8%         |
| near     |  7.6%      | **29.2%**     |
| farthest |  4.5%      | **29.3%**     |

Our corpus was supply-driven: `near` exploded because we used `near`-as-inner
to lift nested supply past its geometric ceiling, and `farthest` because it is
the free symmetric twin of `closest` (every nested question carries a
closest/farthest/near outer selector). The official set is human-authored and
`on`/`closest`-led — humans rarely say "the X farthest from Y".

## Decision

User chose a **moderate reweight** (vs exact-fit / no-change): match the
official *ranking and rough magnitude*, not the exact percentages of a tiny
66-occurrence sample (fitting those would be overfitting to noise). User
accepted the resulting corpus shrink.

## Changes

### `vla3d_nested_gen.py` — per-template caps
Replaced the equal round-robin `_stratified_downsample` with explicit
relation-category caps (`_template_cap`): every `*_closest` template (+
`between`) keeps its full supply; the over-supplied minorities are hard-capped
regardless of available supply:
- `*_farthest` → `FARTHEST_TEMPLATE_CAP = 45` each
- `*_near` (near as outer) → `NEAR_OUTER_TEMPLATE_CAP = 30` each
- `near_closest` / `near_farthest` (near as inner, ~10k combined supply) →
  `NEAR_INNER_CAP = 320` (near_farthest hits the farthest cap first)

(An earlier weighted water-filling attempt was abandoned: it only applied the
weights when supply > target, but the capped supply is far below target, so it
took everything and ignored the weights. Absolute caps are predictable.)

### `vla3d_ref_to_qa.py` — single-layer relation caps
`sample_single_layer` now caps the over-supplied relations before bucketing:
`REL_CAP = {"farthest": 300, "near": 380}`. VLA-3D referential statements are
`farthest`/`near`-heavy; this lets `closest` / `between` / `on`-family fill the
rest. `TARGET_SINGLE` 6000 → 3500, `COLOR_QUOTA` 840 → 350 (~7% of the smaller
corpus).

## Result (5,055 pairs, 0 type mismatches, 173 tests pass)

| relation | official % | before | after |
|----------|-----------:|-------:|------:|
| on       | 42.4%      |  5.2%  | 11.6% |
| closest  | 28.8%      | 23.8%  | 40.0% |
| near     |  7.6%      | 29.2%  |  9.9% |
| farthest |  4.5%      | 29.3%  | 10.4% |
| between  |  6.1%      |  3.5%  | 11.1% |
| above    |  6.1%      |  2.3%  |  5.6% |
| below    |  3.0%      |  1.5%  |  4.0% |

`near`/`farthest` are now minorities and `closest` leads, matching the official
ranking. Two ceilings remain and are documented, not bugs:
- **`on` stays at ~12% (official 42%)** — VLA-3D has only ~84 scene-unique `on`
  anchors; no reweight can manufacture supply. `closest` absorbs the
  primary-relation role at 40%.
- **compositional fell 50% → 28%** (official 57%) — a direct trade-off:
  compositional questions are almost all nested, and nested supply *is*
  `near`/`farthest`, so capping those caps the compositional count. VLA-3D
  cannot be simultaneously 57%-compositional and `on`/`closest`-heavy.

## Cost

Corpus 12,190 → **5,055** (−58%). A `closest`-led / `near`-`farthest`-minority
corpus is supply-capped at ~5k; the old 12k was only that large because it was
`near`/`farthest`-heavy. User accepted this trade.

## Files
- `dataset_generator/vla3d_nested_gen.py` — `_template_cap` + caps.
- `dataset_generator/vla3d_ref_to_qa.py` — `REL_CAP`, smaller `TARGET_SINGLE` /
  `COLOR_QUOTA`.
- `dataset_generator/README.md` — composition + distribution + relation-word
  tables, trade-off notes.
