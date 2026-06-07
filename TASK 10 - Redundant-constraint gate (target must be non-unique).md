# TASK 10 — Drop over-specified ref questions (target must be non-unique)

## Problem

A reviewer flagged `Find the dvd beside the big chair.` in our generated
`object_reference` data: livingroom_3 has exactly **one** dvd, so "the dvd"
already identifies it and "beside the big chair" is a vacuous constraint.

A spatial/attribute constraint is only meaningful when the target class is
**non-unique** in what the robot sees — otherwise it does no disambiguation.

## Evidence: official vs ours

Parsed the 30 graded `object_reference` items in
`../CMU-VLN-Challenge-2026/questions/questions.json` and counted each
target class in its scene's `object_result.csv`:

- **Official: 29/30 (97%) targets are non-unique** (pillow×12, bowl×7,
  blue chair×11, picture×7, …). Only `hotel_room_2 / flowers` is unique.
  → the official design is "constraints disambiguate among same-class
  instances".
- **Ours (before): 37% (4394/12000) targets were unique in view** — the
  constraint was redundant. Worst in nested (2456/6000) and single-layer
  (1938/6000).

## Fix — a redundancy gate in both generators

A ref pair is emitted only if **≥2 instances of the target class** sit in
the `object_list` the pair carries. Scope matches each source's list:

- **`vla3d_nested_gen.py`** — object_list is scene-wide, so gate on the
  scene-wide count `len(by_label[tgt_label]) >= 2`. Added to both the
  inner/outer ref branch (`_emit_inner_outer`) and `between`
  (`_emit_between`). The `num` (counting) branch is left untouched — a
  count is meaningful regardless of class multiplicity.
- **`vla3d_ref_to_qa.py`** — object_list is region-filtered (VLA-3D's
  per-region disambiguators like "the BIG table" are only unique in
  region), so gate on the region count via `_region_info`
  (`region_cnt[target.raw_label] >= 2`) in `build_pair`. New drop reason
  `target_unique_in_view` (16,019 single-layer candidates dropped; 36,977
  still eligible, so the 6,000 quota is unaffected).

## Result

- **Redundant ref pairs: 37% → 0%** (verified: every one of the 12,000
  `object_reference` pairs now has ≥2 same-class objects in its
  object_list). The flagged dvd example is gone.
- Corpus size unchanged: **12,190** (12,000 ref + 190 num) — supply after
  gating still exceeded both 6,000 quotas, so no shrink was needed.
- `check_question_types`: **0** type mismatches. `pytest`: **173 passed**.
  `ruff`: new code clean (pre-existing E701/E702/B007/E501/SIM untouched).
- Determinism preserved (seed 42).

## Distribution side-effects (accepted)

- **Indefinite "a X" anchor: 13% → ~5%.** Most single-anchor "a X"
  statements happen to have a region-unique target, so the gate removes
  them. Supply-capped at 544; can't hit the old 1,560 quota. Kept the gate
  anyway — matching the official "constraints disambiguate" semantics
  outranks the secondary "a X" stylistic share.
- **Compositional: 56% → 50%** (nested is 6,000 / 12,000; single-layer
  contributes ~0 multi-relation pairs). Official is 57%.
- Single-layer ref now concentrates on `closest`/`farthest`/`near`/
  `between` (35% / 35% / 19% / 10%) — relations that inherently need
  multiple same-class instances, which is itself on-distribution; `on`/
  `above`/`below` single-layer fell to ~0 (their targets are usually
  region-unique). Compositional `on` is still well-supplied via nested+num.

## Files changed

- `dataset_generator/vla3d_ref_to_qa.py` — region-scope redundancy gate.
- `dataset_generator/vla3d_nested_gen.py` — scene-scope redundancy gate
  (inner/outer + between ref branches).
- `dataset_generator/README.md` — distribution tables + a new
  "Redundant-constraint gate" subsection.
