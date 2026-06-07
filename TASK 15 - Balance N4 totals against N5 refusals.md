# TASK 15 — Balance N4 (totals) against N5 (refusals) to kill a 0-answer shortcut

## Problem

N5 (refusal) emits "How many **X** are there in the room?" for a category absent
from the scene → **answer 0**. N4 (total count) emits the **identical** phrasing
for a present category → answer ≥1, but N4 was **disabled** (it's out-of-
distribution; the official set has no totals).

With N4 off, *every* "…are there in the room?" question in the corpus was an N5
with answer **0**. A model can learn the spurious shortcut **"in the room ⇒ 0"**
— pattern-matching the phrasing instead of actually counting. (Flagged by the
user.)

## Decision

User chose to **re-enable a small, balanced N4** rather than drop N5. N4 and N5
share the exact phrasing, so adding N4 (answers ≥1) turns the phrasing from a
shortcut into a genuine grounding task: the wording alone no longer predicts the
answer, the model must look at the scene.

## Changes (`vla3d_num_gen.py`)

- **Re-enabled N4** in `generate_scene` (`bucket["N4"] = emit_total_count(sc)`).
- **Tightened N4's answer range** `[3, 30] → [2, 8]` so totals stay near the
  official small-count range (no "15 books" outliers).
- **`NUM_TEMPLATE_CAP["N4"] = 2`** (vs N5's 1) so N4 (~30) outnumbers N5 (~15) —
  0 becomes a *minority* answer within the shared phrasing, not the only one.

## Result (0 type mismatches, 173 tests pass)

- `"…in the room?"` subspace: **45** questions, answers `{0:15, 2:11, 3:8, 4:5,
  5:4, 6:1, 7:1}` — **0 is now 33%** of that phrasing (was 100%). Shortcut
  broken.
- Numerical corpus **190 → 220** (N4 = 30); total corpus **5,055 → 5,085**.
- "totals + refusals" share is now 20% of num (official 0%) — an accepted move
  *away* from the official distribution, traded for not teaching a wrong cue.
  Both categories are OOD by design; N4 exists only to neutralise N5.

## Files
- `dataset_generator/vla3d_num_gen.py` — re-enable N4, range `[2,8]`,
  `NUM_TEMPLATE_CAP["N4"]=2`, comments.
- `dataset_generator/README.md` — composition (220 / 5,085) + numerical
  distribution notes.
- `docs/eda_report.md` — totals + numerical-answer commentary; charts
  regenerated (`eda_report.py`).
