# Recorded corpora for the five-arm ablation

Two files per run, enough to re-score everything in
`docs/tasks/TASK 59` without a simulator and without a model call:

| file | what it carries |
|---|---|
| `steps.jsonl` | the settings header, then one record per decision step: pose, the clause being pursued, the model's reply, the waypoint published, the converter's prediction, and the driven track (`/state_estimation` subsampled at 0.10 m) |
| `plan.json` | the question and the ordered clauses it was decomposed into |

The images, scans and terrain maps each run also produced stay out of git; the
`.gitignore` negation above these two names is deliberate.

## The arms

| prefix | arm | range | converter | visited block |
|---|---|---|---|---|
| `cn_0825_2316_` | **Ours** | lidar lift | modelled | in prompt |
| `nv_0825_0221_` | w/o Lift | model's `distance_m` | modelled | in prompt |
| `cm_0825_1549_` | w/o Platform Model | lidar lift | free-space check only | in prompt |
| `dm_0825_1816_` | Naive | model's `distance_m` | free-space check only | in prompt |
| `vo_0825_1958_` | w/o Memory | lidar lift | modelled | removed |

`nv_` is two passes (52 runs); the rest are one pass (26 each).

## Re-scoring

Needs the challenge repo and `viz/data`, neither of which lives here.

```bash
uv run python scripts/paper_table.py --runs 'cn_0825_*' --strict-instance
uv run python scripts/path_fidelity.py --runs 'cn_0825_*,nv_0825_*,cm_0825_*,dm_0825_*,vo_0825_*'
uv run python scripts/memory_ablation.py
```

`--strict-instance` is not optional if you want the published numbers: without
it a GOTO is satisfied by any object sharing the phrase's head noun, which
scores the corpus about 9 points higher (4.29/6 against 3.77/6).
