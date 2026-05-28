# TASK 3 — VLA-3D generated dataset pipeline

## Goal

Bootstrap a training corpus for our VLM by exploiting the
[VLA-3D](https://github.com/HaochenZ11/VLA-3D) Unity subset. The 15
Unity scenes VLA-3D ships are the same scenes the CMU VLN Challenge
trains on, and the object IDs are aligned 1:1 to the challenge's
`object_list.txt`. With three jsonl generators on top of VLA-3D we
end up with **8,299 grounded Q&A pairs** in the same schema the
runtime contract already speaks.

## What was done

### Phase 1 — Source data

- Cloned VLA-3D, kept only the 15-scene Unity subset (~2 GB).
- Inspected the three per-scene files we actually use:
  `<scene>_object_result.csv`, `<scene>_scene_graph.json`,
  `<scene>_referential_statements.json`. The first gives bboxes,
  region IDs, NYU40 categories, and 3 dominant colors per object;
  the second gives 8 precomputed spatial relations (`on`, `above`,
  `below`, `near`, `between`, `beside`, `in`, `hanging_on`) plus
  `closest`/`farthest` chains; the third gives ~120 K precomputed
  referential statements.

### Phase 2 — Loader

`vla3d_loader.py` produces a unified `VLAScene` with:

- `objects`: list of `VLAObject`(id, region_id, raw/nyu40 labels,
  bbox cx/cy/cz/lx/ly/lz/heading, top-3 colors).
- `regions`: `region_id → VLARegion`.
- `relationships`: `region_id → rel_name → src_id → list[tgt_id]`.
  Ternary relations (`between`) carry list-of-pairs.
- `ref_statements`: per-region list of normalized statement dicts.

A single helper `render_object_list(sc, region_ids=None)` formats
objects as one-line records that exactly match the runtime
`src/xiao_hei_vln/dummy/data/object_list.txt` schema:
`id cx cy cz lx ly lz heading label`.

### Phase 3 — Generators

Three deterministic generators (seed `42`):

| Script                  | Output                  | Pairs  | Idea                                                                                  |
|-------------------------|-------------------------|-------:|---------------------------------------------------------------------------------------|
| `vla3d_ref_to_qa.py`    | `vla3d_ref.jsonl`       | 6,730  | Rewrite VLA-3D ref statements ("the X that is above Y") → imperative ("Find the X above Y.") |
| `vla3d_num_gen.py`      | `vla3d_num.jsonl`       |   386  | 8 numerical templates: on / near / above / below / hanging-on / total / refusal / color+on |
| `vla3d_nested_gen.py`   | `vla3d_nested.jsonl`    | 1,183  | Two-stage: scene_graph inner relation × computed closest/farthest outer (+ `between`) |

Highlights:

- `vla3d_ref_to_qa.py` rebalances the 119 K raw statements down to
  6,730. Raw distribution is 48% `farthest`, the official Q&A is 47%
  `on`; per-relation caps + imperative rewrite + word-length filter
  pull it back toward the challenge's mix.
- `vla3d_nested_gen.py` produces the official-style nested phrasing
  ("Find the bowl on the table closest to the screen") with 100%
  geometric ground truth — no LLM, no verifier needed. Closest /
  farthest are computed with a 0.3 m uniqueness margin so the answer
  is unambiguous.
- 7,708 `object_reference` and 591 `numerical` pairs in total.
  Instruction-following pairs are not generated here (separate task).

### Phase 4 — Runtime alignment

`object_list` lines, the `type` field, and `ChallengeQuestion` routing
are all 1:1 with the runtime stack:

- `object_list` line format is identical to the dummy responder's
  fixture file → an `ObjectReferenceResponse` can be assembled
  directly from a perception list + the LLM's chosen `object_id`.
- `type` field is set at generation time. The runtime classifier
  in `src/xiao_hei_vln/messages/question.py` derives the same type
  from the question text. Verified by `check_question_types.py`:

```
Checked 8299 pairs across 3 files
Mismatches: 0
```

  If a future generator emits a question whose first word doesn't
  match the classifier (e.g. "Count …" or "What …"), the script
  exits non-zero — wire it into CI before training.

### Phase 5 — Splits

`split_and_dump.py` does **scene-level Group K-Fold** (default
10/3/2 single split, optional `--kfold 5`). Same-scene samples never
straddle splits, so test accuracy reflects generalization to unseen
scenes — which is what the challenge actually evaluates.

### Phase 6 — Perception noise (library, deferred use)

`noise_augment.py` provides `perturb_scene(objects, protected_ids, …)`
with drop / label-swap / bbox-jitter. Target + anchors are always
protected. Not wired into the current pipeline because we currently
assume full-scan inference; kept as a starting point for when the
exploration strategy is finalized.

## How to reproduce

The data is not committed. See `generated_dataset/README.md` for
the full workflow:

```bash
export VLA3D_ROOT=/path/to/VLA-3D/Unity
uv run python generated_dataset/vla3d_ref_to_qa.py
uv run python generated_dataset/vla3d_num_gen.py
uv run python generated_dataset/vla3d_nested_gen.py
uv run python generated_dataset/check_question_types.py
uv run python generated_dataset/split_and_dump.py --kfold 5 --seed 42
```

## Open questions / next tasks

- **`instruction_following`** Q&A pairs are not generated yet.
  Possible angle: chain 2–3 ref-style anchors into a path
  description ("Take the path near the table to the fridge").
- **Exploration strategy** dictates whether full-scan training is
  representative. Once decided, swap a viewpoint sampler in on top
  of `noise_augment.perturb_scene` so each epoch sees a different
  perception coverage profile.
- **Real-robot data augmentation** is untouched. The challenge ships
  one unlabeled ROS bagfile sample whose layout differs from the
  real evaluation — needs its own pass once a perception module
  exists to profile.
