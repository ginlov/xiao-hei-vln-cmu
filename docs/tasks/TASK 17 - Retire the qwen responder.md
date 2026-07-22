# TASK 17 - Retire the `qwen` responder

## Purpose

Remove the Qwen3.5 / vLLM responder and its entire support surface. It was
superseded by `scene_gemini` (TASK 14) and the team will not use it, but unlike
the Phase-A exploration stack (TASK 15) it was still **fully wired** — installed
by default in the docker image, documented as the primary run path, and
referenced across 38 files.

## Problem (before)

`qwen` was not dead code — it was the *documented default*:

- `docker/compose.yml` built `ai_module` with `XIAO_HEI_EXTRA=qwen,perception`,
  so every image shipped the `openai` client whether or not it was used.
- `docker/compose_gpu.yml` defaulted `XIAO_HEI_RESPONDER=qwen` and started a
  `vllm/vllm-openai` sidecar reserving the whole GPU — the same GPU the
  perception sidecar and the simulator need.
- `docs/index.md` told readers "the system currently uses **Qwen3.5-4B**",
  and `getting-started/quickstart.md` used it as the worked example. The
  actual submission stack (`scene_gemini`) was documented only in the README.
- `docs/getting-started/configuration.md` documented 12 `XIAO_HEI_QWEN_*` env
  vars and **zero** `XIAO_HEI_GEMINI_*` or `XIAO_HEI_PERCEPTION_*` ones.

So the docs site described a pipeline the team had already replaced.

## What changed

### Deleted

- `src/xiao_hei_vln/qwen/` — `config.py`, `engine.py`, `prompts.py`,
  `responder.py` (`QwenResponder`, `QwenEngine`, `HTTPQwenEngine`, `QwenConfig`)
- `tests/test_qwen_{config,numerical_loop,prompts,responder}.py`,
  `tests/test_http_engine.py` — 47 tests
- `docker/compose_gpu.yml` — the vLLM-first compose stack
- `docs/concepts/qwen-engine.md`, `docs/api/qwen.md`
- `pyproject.toml`: the `qwen` and `qwen-local` optional-dependency groups
  (`openai`, `vllm`, `huggingface-hub`)
- `docker/compose.yml`: the whole `vllm` service, the `qwen` profile, the
  `XIAO_HEI_QWEN_*` env passthrough, and the now-orphaned `hf_cache` volume

### Rewired

| File | Change |
|---|---|
| `app/main.py` | dropped the `qwen` factory branch, node-name mapping, and docstring; error message now lists `dummy, perception, scene_gemini` |
| `gemini/{engine,config}.py` | docstrings cross-referenced `QwenConfig` / `HTTPQwenEngine`; reworded to stand alone |
| `tests/test_vlm_logger.py`, `tests/test_logger_pointclouds.py` | borrowed `QwenConfig` purely as a dataclass for the logger payload → `GeminiConfig(api_key="test-key")` |
| `dataset_generator/project_gt_prototype.py` | usage line said `uv run --extra qwen` but the script only imports **PIL** → `uv run --with pillow` |
| `docker/{Dockerfile,run,README.md}` | default extra `qwen,perception` → `perception`; wrapper no longer maps a `qwen` profile |
| `docs/` (13 files) | see below |
| `mkdocs.yml` | dropped the two deleted pages from nav |

### Docs realigned to the actual stack

`docs/index.md` now describes `scene_gemini` as the system. `architecture.md`'s
container-topology and tick-lifecycle Mermaid diagrams had a `vLLM Server` node
wired to `HTTPQwenEngine`; both now show the perception sidecar and the Gemini
API. `getting-started/configuration.md` swapped 12 `XIAO_HEI_QWEN_*` rows for
the real `XIAO_HEI_GEMINI_*` (6) and `XIAO_HEI_PERCEPTION_*` / `XIAO_HEI_SCAN_*`
(8) variables — **read out of `gemini/config.py` and `app/main.py`**, not
transcribed from the old table, so the defaults match the code.

The two Qwen **design records** (`docs/task3_phase1_framework.md`,
`docs/task3_phase3_prompt.md`) were kept verbatim — same treatment TASK 16 gave
the task reports — but moved out of the "Design Docs" nav group into
"Historical design records" and given an admonition banner. Their
sidecar-vs-in-process rationale still explains why the perception sidecar is
structured the way it is.

## Gotcha worth remembering

Same namespace-package trap as TASK 15, one step nastier: `git rm -r` left
`src/xiao_hei_vln/qwen/` on disk, and the follow-up `find -type d -empty
-delete` **did not remove it** because `__pycache__/` was still inside. The
directory therefore stayed importable as an empty PEP 420 namespace package.
`__pycache__` must be purged *before* the empty-dir sweep, not after.

## Verification

- `uv run pytest -q` → **407 passed, 1 skipped** (454 − 47 qwen tests). The
  logger tests exercise the `GeminiConfig` swap.
- `import xiao_hei_vln.qwen` raises `ModuleNotFoundError`.
- `mkdocs build --strict` → exit 0. Strict mode is what proves the doc edits
  are complete: both deleted pages were in `nav`, so a missed reference fails
  the build (and CI runs the same command).
- `docker compose -f docker/compose.yml config` → OK; `--profiles` now lists
  only `perception`. `compose_scene_gemini.yml` validates with a key set.
- `ruff check src tests` → no new findings.
- Repo-wide grep for `qwen`/`vllm` outside `docs/tasks/` and the two banner-ed
  historical records returns **nothing**.

## Follow-ups this opens

- `XIAO_HEI_RESPONDER` now has three values (`dummy`, `perception`,
  `scene_gemini`) but `docker/run` only knows two profiles — `scene_gemini`
  lives in its own compose file. Folding it into the unified `compose.yml` +
  wrapper would make one entry point for every responder.
- `docs/concepts/frontier-explorer.md` and `concepts/exploration.md` were not
  touched here; they should be checked against the TASK 15 removals.
