# TASK 52 — A 100-label detector prior covering all 15 scenes

## Why

The open-vocabulary detector can only emit labels it is prompted with, and its
`/detect` latency grows with the class count (measured ~2–3 s at ~130 classes —
enough to blow the perception client timeout, see the nav_task1 debugging). The
`DEFAULT_PRIOR` was **110 labels skewed to a single scene's** ground truth, so
recall on the other 14 scenes suffered while the prompt was still large. We want
one bounded prior that serves **all 15** challenge scenes.

## What

Capped the prior at **100 labels** and made the selection cross-scene:

- `scripts/gen_vocab_prior.py` gained `--max-labels` (default `MAX_LABELS = 100`).
  When the `≥ min_scenes` pool exceeds the cap, labels are ranked by
  **cross-scene generality** (number of distinct scenes) then total frequency,
  and the tail is dropped — the rarest scene-specific labels go first, since the
  challenge scenes are unseen and a label common in one scene is not thereby
  general. Plural-collapse and the min-2-scenes floor are unchanged.
- Regenerated `DEFAULT_PRIOR` from the 15 VLA-3D Unity `object_list.txt` dumps
  (`/home/long/Projects/dataset/unity_scenes_extracted/`).

## Result

- **100 labels, 81% of all 1802 ground-truth objects** across the 15 scenes.
- Every scene is served: per-scene GT coverage ranges **67.7% (studio) → 95.7%
  (livingroom_4)**; arabic_room 92.6%.
- Dropped (beyond the cap, lowest generality/frequency): `side table`, `stove`,
  `tv stand`, `tablecloth`, `sink cabinet`, `shower head/tap`, `tap`,
  `potted bamboo`, `water bottle`. Rare, scene-specific words a question still
  reaches through the per-question dynamic-vocab expansion (e.g. `hookah`).
- Distribution for reference: `≥2 scenes` → 113 labels / 83%; `≥3` → 66 / 68%.
  The 100-cap sits just under the `≥2` tier, keeping nearly all of its coverage.

Regenerate with:

    uv run python scripts/gen_vocab_prior.py \
      --vla3d-dir /home/long/Projects/dataset/unity_scenes_extracted

Tests: `tests/test_perception_vocab.py` (12) + perception settings/responder (24) green.
