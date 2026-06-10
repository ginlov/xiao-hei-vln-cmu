# TASK 8 — Ref dataset ambiguity audit & label-vocabulary fix

## Trigger

`visualize_sample.py` (TASK 7) rendered `vla3d_ref.jsonl[1]`:

    Q: "Find the chair in between the keyboard and the bench."

and the scene had 6 keyboards + 2 benches. That looked like a generation
bug, so we audited the whole corpus for anchor ambiguity.

## Finding 1 — the naive "62.5% ambiguous" was a false positive

A first pass counted, per sample, whether any anchor's class label
occurs more than once in the scene. That flagged 62.5% of ref samples.
**It was the wrong metric.** Re-auditing with the generator's actual
disambiguation logic:

- **nested** (978 ref + 205 num): `0` truly ambiguous. The inner anchor
  is *intentionally* multi-instance, disambiguated by "closest/farthest
  to a unique outer anchor"; `between` uses two singleton anchors. The
  generator already verifies uniqueness geometrically (margin check).
- **single-layer ref**: 6711/6730 have `target + distractor_ids` ==
  every same-class object in the region. VLA-3D guarantees the statement
  uniquely identifies the target among those distractors, so the
  existential phrasing ("the chair between *a* keyboard and *a* bench")
  still picks a unique target even when the anchor class is not unique.

So there is **no systematic position-ambiguity bug**. Only 19/6730
single-layer ref samples (0.28%) have an uncovered same-class object,
and those are ordinal phrasings ("second/third closest") that the
ordinal still disambiguates.

## Finding 2 — the real bug: object_list ↔ question vocabulary mismatch

`vla3d_ref` questions are rewritten from VLA-3D `referential_statements`,
which use a normalized **class** vocabulary (`television`, `plant`,
`picture`). But `object_list` was rendered from `object_result.csv`'s
**raw_label** (`tv`, `potted plant`, `painting`). Result:

    object_list[target] != answer.label  in  3113/6730 (46%) ref samples

The model reads "Find the television…" but `object_list` only has a `tv`
— it cannot ground the noun. This is what *looked* like ambiguity.

`nested` (0/978) and `num` are clean: they build questions from
`raw_label`, the same vocabulary as their `object_list`.

## Fix (two parts, both in the generators; runtime schema unchanged)

### Part A — region-filtered object_list (Step 1)

`render_object_list(sc, region_ids={region_id})` for single-layer ref and
anchor-based num (N1/N2/N3/N6/N7/N8). VLA-3D disambiguators are
region-scoped ("the BIG table" is unique in its region, not scene-wide),
so a scene-wide object_list could reintroduce ambiguity. Audit confirmed
0/6730 ref anchors live outside the target's region → lossless. Nested
keeps a scene-wide object_list (its outer anchor may be in another
region). N4/N5 are scene-wide by design (`region_id=None`).

### Part B — align object_list labels to the statement vocabulary

`render_object`/`render_object_list` gained an optional
`label_overrides: dict[int,str]`. `vla3d_ref_to_qa.build_pair` builds:

    {target} ∪ distractor_ids -> target_class      # answer noun grounds
    each anchor               -> its statement class # anchor noun grounds

Distractors get the *same* label as the target on purpose: the spatial
relation, not a unique label, must disambiguate — otherwise the model
would learn the label-matching shortcut. Direction chosen = VLA-3D class
vocabulary (not raw_label) because the **test-time questions use that
vocabulary**, so training questions must too; we fix `object_list` to
match. (Deployment perception must likewise emit class-vocab labels, or
map to them — separate concern.)

## Verification (after regenerating the corpus)

- `object_list[target] != answer.label`:  3113/6730 → **0/6730**
- distractors NOT sharing target label:    **0/4346** (no label shortcut)
- anchor object_list-label present in question: **8228/8230** (99.98%)
- nested true ambiguity (precise geometric audit): **0** ref + **0** num
- anchors missing from (region-filtered) object_list: **0/10769**
- `check_question_types`: 8299 pairs, 0 mismatches
- `pytest`: 173 passed
- `vla3d_ref.jsonl`: 67 MB → 41 MB (region filter shrinks object_list)

## Files

- `vla3d_loader.py` — `render_object`/`render_object_list` gained
  `label`/`label_overrides`.
- `vla3d_ref_to_qa.py` — region filter + label_overrides in `build_pair`.
- `vla3d_num_gen.py` — region filter on anchor-based emitters; `region_id`
  on every pair (None for scene-wide N4/N5).
- `vla3d_nested_gen.py` — `region_id` field added; object_list stays
  scene-wide (documented why).

## Part C — drop ill-formed definite-anchor samples

The label-vocab fix made object_list groundable, but a separate phrasing
defect remained: an anchor referred to as a bare "the X" while the region
holds several X. "Find the chair in between **the keyboard** and the
bench" in an office with 6 keyboards is an ill-formed definite
description — the target chair is geometrically unique (VLA-3D's
`between` is satisfied by exactly one chair), but "the keyboard" gives
the model no way to know which keyboard is meant, and the definite
article wrongly presupposes uniqueness. At inference there is no
highlight; the model sees only text + object_list.

Precise count (anchor with **no** color/size disambiguator AND >1 object
of that raw_label in the region): 2,643 / 119,831 (2.2%) of all VLA-3D
statements. Anchors carrying a modifier ("the BLUE book", "the BIG
table") are kept — the modifier is what disambiguates (44k such cases).

`vla3d_ref_to_qa.anchor_ill_formed()` drops these in `build_pair`.
Result: single-layer ref 6,730 → 5,372. Verified 0 residual ill-formed
under the exact (color_used/size_used-aware) criterion.

num is unaffected: its single-layer anchors already come from
`label_singletons_in_scene` (unique by construction). nested is
unaffected: its inner anchor is intentionally multi-instance but
well-formed because "closest/farthest to <unique outer anchor>"
disambiguates it; `between` uses two singleton anchors.

Final corpus: ref 6,350 (5,372 single + 978 nested) + num 591 = 6,941.

## Not done (possible follow-ups)

- The 12 raw_label→class **conflicts** (`glass`→glass/bowl,
  `dish`→plate/serving dish/glass dish) are handled per-sample correctly,
  but indicate VLA-3D itself sometimes describes one object with
  different nouns. No action needed unless we see eval noise.
- color/size grounding: `object_list` carries no color/size, so "the
  *blue* book" still relies on perception, not the label. Out of scope
  (the runtime object_list schema never had these fields).

## Part D — drop ordinal-ranked phrasings (align to official distribution)

After locating the organizer's actual question set
(`../CMU-VLN-Challenge-2026/questions/questions.json` — 75 questions: 15
numerical + 30 object_reference + 30 instruction_following, one PDF per
scene), an exact search found **zero** ordinal/ranking words across all
30 object_reference items. The official object_reference style is purely
superlative + compositional, e.g.:

- "Find the bowl on the table closest to the folding screen." (on × closest)
- "Find the vase between the cabinet and the stool." (between)
- "The red pillow closest to the sushi." (color modifier, "Find" omitted)
- "Find the picture closest to a window." (indefinite "a" anchor)

But our `vla3d_ref.jsonl` carried 1,248 ordinal samples ("second/third
closest|farthest|nearest") — out-of-distribution. `vla3d_ref_to_qa.py`
now drops them at the source via `is_ordinal_ranked()` (regex: an ordinal
word immediately followed by a ranking word; all 1,248 matched this form
exactly, 0 false positives). Source-level drop: 46,788 statements.

**Corpus size is unchanged** (single-layer ref still 5,372; total still
6,941). The per-relation rebalance cap (closest 1500, farthest 500) was
already the binding constraint, so removing ordinals from the pool just
refills those caps with plain-superlative samples — same size, better
matched to the official phrasing. Verified: `vla3d_ref.jsonl` 0/6350
ordinal, `vla3d_num.jsonl` 0/591, 173 tests pass.

### Other gaps surfaced vs the official set (not yet acted on)

- **Nested/compositional is the official mainstay** (most of the 30 are
  2-hop "inner relation × outer closest"), yet we have only 978 nested
  ref pairs — likely under-weighted; candidate to expand.
- **Official mixes "a" and "the" anchors** ("closest to *a* window").
  Part C drops non-unique definite anchors outright; a better fix may be
  to *rephrase* them as "a X" rather than drop, to cover this style.
- **Official omits "Find"** in some items ("The red pillow closest…") and
  uses color modifiers — paraphrase coverage worth adding to the rewriter.
