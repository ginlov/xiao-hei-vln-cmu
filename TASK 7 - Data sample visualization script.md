# TASK 7 — Data sample visualization script

## Problem

The generator produces 8,299 grounded Q&A pairs across `vla3d_ref.jsonl`
and `vla3d_num.jsonl`. Each row carries a baked `object_list` plus a
`target` (the answer), `anchors` (objects referenced in the question)
and `distractor_ids` (same-class non-answers). Up to now there was no
way to actually *see* a row:

- Is `target` pointing at the right object cluster, or off-by-one?
- Does the chosen `anchor` match the relation in the question
  (e.g. "closest to X" but X is actually 8 m away)?
- Is the `heading` field on the OBB sane, or is the box rotated wrong?
- Are distractors actually same-class objects, or did the generator
  pick something unrelated?

Without a visual, the only debugging path was reading raw jsonl —
unworkable at this scale.

## Design

### Output

3D OBB wireframe scene per sample, color-coded:

| Role        | Color  | Source field            |
|-------------|--------|-------------------------|
| target      | red    | `target` / `answer.object_id` |
| anchors     | yellow | `anchors`               |
| distractors | cyan   | `distractor_ids`        |
| other       | gray   | rest of `object_list`   |
| point cloud | l-gray | `<scene>_pc_result.ply` (optional) |

OBB (heading-aware) not AABB — a wrong `heading` shows up as a visibly
mis-rotated wireframe. Note our in-repo `evaluator` computes IoU on AABB
(`src/xiao_hei_vln/evaluator/metrics/object_reference.py:46`); the
official challenge eval code is not published, so AABB/OBB on the
leaderboard side is unconfirmed. Visualization needs OBB regardless,
because we're debugging the generator output (which encodes heading).

### Modes

| Mode              | CLI                                           | Output             |
|-------------------|-----------------------------------------------|--------------------|
| Single interactive| `--idx 42`                                    | Open3D window      |
| Random interactive| `--random`                                    | Open3D window      |
| Single PNG        | `--idx 42 --save out/`                        | `<scene>_00042.png`|
| Spot-check batch  | `--sample-per-scene 20 --save out/`           | 15 × N PNGs        |
| With point cloud  | add `--pointcloud` to any of the above        | overlay `.ply`     |

Point cloud overlay is **off by default** — wireframe-only PNGs are
~40 KB; with point cloud they jump to ~800 KB and load in 1-3 s. For
batch spot-check (e.g. 20 × 15 = 300 PNGs) wireframe-only stays under
15 MB, comfortable for git artifacts / Slack screenshots.

### Files

- `dataset_generator/visualize_sample.py` — new CLI script.
- `pyproject.toml` — new `viz` optional-dependency group with
  `open3d>=0.18` + `pillow>=10`. Heavy (~400 MB wheel), opt-in only;
  training / docker / CI don't need it.
- `dataset_generator/README.md` — added "Visualize a sample" section
  + new line in the layout tree.

### Coordinate system

VLA-3D Unity subset uses **Z-up**, with `heading` interpreted as yaw
about Z (verified against `vla3d_loader.py:47` —
`heading: float  # bbox orientation about Z`). The script builds the
rotation matrix accordingly.

## Verification

Dry-run (no GUI) on the first row of each jsonl:

- `vla3d_ref.jsonl[0]` → 101 OBBs parsed, target id=8 painted red,
  anchor id=9 painted yellow.
- `vla3d_num.jsonl[0]` → 106 OBBs parsed, target=None (correct: numeric
  questions have no single answer object), anchor id=84 painted yellow.

Headless PNG render:

- Single `--idx 0` → `loft_00000.png` (39 KB).
- Batch `--sample-per-scene 3` → 45 PNGs across all 15 scenes, 1.7 MB
  total.

Visual check on `arabic_room_00060.png` confirmed:
- Scene wireframes match the room layout (walls, furniture clusters).
- Red OBB on the lamp (target).
- Yellow OBB on the anchor object.
- Cyan OBBs on the same-class distractors.

Lint: `ruff check dataset_generator/visualize_sample.py` clean.

## Out of scope

- 3D text labels inside the window. Open3D's in-scene text rendering is
  finicky; the script prints `question` / `answer` / role IDs to
  terminal alongside instead.
- Multi-view atlas (top-down + side-by-side perspective in one PNG).
  Doable as a follow-up if review needs it.
- Open3D Web viewer / rerun.io integration. Considered; the existing
  desktop window covers the iteration loop already.

## Follow-up refinements (point-cloud readability)

After running the tool on real samples, four readability tweaks:

- **Colored points are the default.** VLA-3D native per-point RGB
  (gamma-boosted, native mean ~0.16-0.23) renders by default; the white
  dots a user hit were from opting into `--gray-points` (flat gray
  override). Confirmed on `livingroom_4`.
- **`--ceiling-cut` (default 0.5 m)** crops the top of the cloud so the
  roof stops occluding the interior from the isometric view (~28% of
  points on `livingroom_4`). Bump to 0.8 to also drop hanging lights.
- **`--point-size`** (default 2.5) — close-up the 2.5px dots look tiny;
  5-8 makes them solid. Interactive window also responds to `+`/`-`.
- **`--line-radius` (meters, default 0)** — draw OBB edges as solid
  cylinder tubes of real, backend-independent thickness.
  `RenderOption.line_width` is a no-op on the macOS OpenGL backend
  (verified: width 1 vs 8 render byte-identical, 41970 lit px both), and
  is driver-dependent / often capped at 1px on Linux Mesa too, so a
  LineSet can't be reliably thickened. Tubes (`box_to_tube`) scale
  correctly: radius 0/0.01/0.03 -> 41970/56969/132987 lit px. Cost is a
  heavier mesh (8-face cylinder per edge); fine for single/batch render,
  slightly slower interactive rotation at large radius. Suggested 0.015-
  0.02.
