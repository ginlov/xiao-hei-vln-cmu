# TASK 6 — Documentation webpage

## Problem

The repository has grown to include multiple modules (messages, adapters,
sync, dummy, qwen, evaluator, eval_sampler, eval_pipeline), Docker
infrastructure, scripts, and a dataset generator. Onboarding new team
members or revisiting the code after a break requires reading scattered
README files and task reports. A unified documentation site will:

- Guide people through common classes and protocols.
- Introduce system architecture focusing on VLM communication and
  configurable parameters of ai_module.
- Tutor for new model development and evaluation.
- Document data generation and current status.

## Target

Static HTML site built with **MkDocs + Material theme** — renders Markdown,
auto-generates API reference from docstrings, and deploys to GitHub Pages
with a single `mkdocs gh-deploy` command. Integrates into an existing
github.io page by being published to a subpath
(`<user>.github.io/xiao-hei-vln-cmu/`).

## Planned structure

```
docs/
  index.md                    # Landing page / introduction
  architecture.md             # System architecture diagram + data flow
  getting-started/
    quickstart.md             # Minimal steps to run the system
    docker.md                 # Docker setup (from docker/README.md, expanded)
    configuration.md          # All env vars with descriptions + defaults
  concepts/
    messages.md               # VLMInput, VLMOutput, sensor types, question types
    responder.md              # Responder protocol, lifecycle, how to add a new one
    tick-loop.md              # 2 Hz loop, LatestCache, snapshot mechanics
    qwen-engine.md            # HTTPQwenEngine vs QwenEngine, config, prompts
  guides/
    new-model.md              # Step-by-step: implement + wire a new VLM
    evaluation.md             # Evaluator pipeline, metrics, eval_sampler
    data-generation.md        # VLA-3D dataset, eval sample generator
    vlm-logging.md            # Tick logger, replay, HTML reports
  api/                        # Auto-generated from docstrings (mkdocstrings)
    messages.md
    qwen.md
    evaluator.md
    sync.md
    adapters.md
  changelog.md                # Per-task progress log
```

## Phases

### Phase 1 — Scaffolding + core content pages

1. Add `docs` optional dependency group in `pyproject.toml`:
   ```toml
   docs = ["mkdocs-material>=9.5", "mkdocstrings[python]>=0.25"]
   ```
2. Create `mkdocs.yml` at the repo root with nav structure, Material
   theme config, and mkdocstrings plugin.
3. Write initial pages:
   - `docs/index.md` — project intro, link to challenge, team overview
   - `docs/architecture.md` — ASCII/mermaid diagram of the full stack
   - `docs/getting-started/quickstart.md`
   - `docs/getting-started/docker.md`
   - `docs/getting-started/configuration.md`

### Phase 2 — Concepts + guides

4. Write concept pages explaining the internal data model and execution flow.
5. Write guides for new model development, evaluation, data generation,
   and VLM logging.
6. Set up `mkdocstrings` auto-generated API reference for key modules.

### Phase 3 — GitHub Pages integration

7. Add a GitHub Actions workflow (`.github/workflows/docs.yml`) that
   builds and deploys the site on push to `main`.
8. Configure the GitHub repo's Pages source to the `gh-pages` branch.
9. Verify the site is live at the expected URL.

## Technology choice: MkDocs + Material

| Criterion | MkDocs + Material | Sphinx | Docusaurus |
|---|---|---|---|
| Language fit | Python native, reads docstrings | Python native | JS/React |
| Setup speed | ~5 min, 2 deps | ~15 min, many extensions | npm project |
| Appearance | Modern, responsive, dark mode | Functional | Modern |
| GitHub Pages | Built-in `mkdocs gh-deploy` | Manual | Manual |
| API reference | mkdocstrings plugin | autodoc (native) | None |

MkDocs + Material wins on speed, appearance, and GH Pages integration.

## Verification

1. `uv run mkdocs serve` — local preview at `http://127.0.0.1:8000/`
2. All internal links resolve (mkdocs strict mode)
3. API reference pages render without errors
4. `mkdocs gh-deploy` successfully publishes to GitHub Pages
