# TASK 14 — Colour modifiers: dominant-only + map technical palette to basic

## Problem

`vla3d_ref.jsonl` idx 15: **"Find the light switch nearest to the blue book."**
The "blue book" anchor is `colors=('gray','blue',…)`, `pct=(0.76, 0.18, …)` — a
**76%-gray** book with **18% blue**. It sits among seven books in the region,
*all* predominantly gray, that VLA-3D can only tell apart by their **secondary**
colour (blue / aqua / olive / brown / black). Calling it "the blue book" is
valid disambiguation but perceptually wrong — a camera/VLM sees a gray book.

Two systemic findings across the corpus's colour modifiers (350 samples):
- **59% name a non-dominant colour**; **37% name a colour covering <30%** of the
  object (e.g. "brown picture" at 12%, "maroon picture" at 10%).
- **Palette mismatch.** VLA-3D uses a technical palette (maroon / aqua / olive /
  navy / teal / violet / beige / tan); the human-authored official set uses
  **basic** words (red / blue / black / green / brown). The official "red
  pillow" (japanese_room) is **maroon-100%** in VLA-3D; the official "blue
  chair" / "black pillow" (loft) have **no** matching VLA-3D colour at all.
  So our colour vocabulary was off-distribution two ways: minority colours *and*
  technical words the official never uses.

## Fix (`color_gate_and_map`, `vla3d_ref_to_qa.py`)

Uses VLA-3D's structured `target_color_used` / anchor `color_used` (not text
parsing) for each colour modifier in a statement:

1. **Dominant gate** — keep only if the used colour is the object's dominant
   colour (`colors[0]`) covering ≥ `COLOR_DOMINANT_MIN = 0.40`; else drop the
   sample (drop reason `weak_color`).
2. **Basic-word mapping** — rewrite the surviving question's colour word via
   `COLOR_BASIC_MAP` (`maroon→red`, `navy`/`teal`/`aqua→blue`, `olive→green`,
   `beige`/`tan→brown`, `violet→purple`, `gold/golden→red`, `silver→gray`). A
   maroon-100% pillow → "the red pillow", matching the official phrasing exactly.

Wired into `build_pair` after the anchor-softening step (so the mapped word also
benefits the colour-quota sampler's `_has_color` flag).

## Result (5,055 pairs — unchanged, 0 type mismatches, 173 tests pass)

- `weak_color` dropped 2,769 candidate statements; enough dominant-colour supply
  remained to keep the colour quota, so **colour share stays at 7.2%** (official
  ~7%) and the corpus stays at 5,055.
- "blue book" and all minority-colour modifiers are gone; **0 technical-palette
  words** remain. Corpus colours are now all basic and dominant: gray, black,
  brown, red (←maroon), purple (←violet), green (←olive), blue (←aqua/navy/teal),
  pink.
- `ruff` clean on new code.

## Files
- `dataset_generator/vla3d_ref_to_qa.py` — `color_gate_and_map`,
  `COLOR_DOMINANT_MIN`, `COLOR_BASIC_MAP`, `build_pair` hook.
- `dataset_generator/README.md` — "Colour gate + basic-word mapping" subsection.
