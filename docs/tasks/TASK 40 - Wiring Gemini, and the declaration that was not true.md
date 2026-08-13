# TASK 40 — Wiring Gemini, and the declaration that was not true

`--backend gemini` has been an argument on `execute_plan.py` and
`approach_loop.py` for weeks. It had never completed a single call.

```python
r = genai.Client(api_key=key, http_options=...).models.generate_content(...)
```

The client is a temporary. The SDK closes its `httpx` session when the object
is collected, which happens before the request goes out, so every call raised

```
RuntimeError: Cannot send a request, as the client has been closed
```

It is now bound to a name and cached at module scope — a leg makes ten of these
calls and each new client is a fresh TLS handshake.

## The declaration that was not true

Worse than not running would have been running. `gemini-3.1-pro-preview`
returns 0-1000 normalised boxes and declares them `"pixels"`, and `to_pixels`
trusts the declaration first:

```python
if space == "pixels":
    return list(box)
```

On `runs/cr_0811_03` step 4, asked for *"the potted plant on the table"*, Pro
answered `box_2d [293, 330, 506, 596]`, `feature_box_2d [506, 388, 885, 559]`,
`coord_space "pixels"` — and **885 does not exist on a 640-pixel face**. Drawn
as declared, both boxes sit on bare wall and floor about two metres right of
the plant; drawn as normalised, they land on the plant and on the stand beneath
it. The box centre moves 174 px, which is a waypoint into a wall.

Nine calls, three models, three scenes:

| | declared `pixels` | declared `normalized_1000` | proved normalised by magnitude |
|---|---|---|---|
| `gemini-3.1-pro-preview` | 2/3 | 1/3 | yes, on one of the `"pixels"` ones |
| `gemini-3.6-flash` | 0/3 | 3/3 | yes |
| `gemini-3.1-flash-lite` | 0/3 | 3/3 | — |

No call anywhere was shown to be in pixels.

**A magnitude guard is not enough.** The same sweep had Pro declare `"pixels"`
with a maximum of 494 — a legal pixel value and a legal normalised one, so that
reply is undecidable on its own numbers and would pass straight through. What
settles it is the backend: Gemini's detection output is normalised by
construction, which is what the comment in `to_pixels` already said before
anything depended on it.

`settle_coord_space(reply, backend, size)` runs in `ground()`, before any box
is converted. For Gemini it rewrites `coord_space` outright. For Claude it
leaves the declaration alone but keeps one arithmetic override — a coordinate
above the image is not a pixel on it, whatever the reply calls itself — and
looks in every slot the schema puts a box in, not only `box_2d`.

## The rest of the wiring

- **`google-genai` was declared but never installed.** It is the `gemini`
  extra in `pyproject.toml`; the runbook needed `--with google-genai` next to
  `--with anthropic`. Both are required even on `--backend gemini`, because
  decomposition always calls Claude.
- **The default model 404s.** `gemini-2.5-flash` answers *"no longer available
  to new users"* — while still being listed by `models.list`. The catalogue is
  not evidence that a model can be called. Now `DEFAULT_GEMINI_MODEL =
  "gemini-3.6-flash"`, one constant instead of three literals, pinned rather
  than `gemini-flash-latest` because that alias moved from 3.5 to 3.6 during
  the hour this was written and an A/B whose model changed halfway measured
  nothing. A test asserts the default is not an alias, and that `scripts/` and
  `src/` agree on it.
- **`ask_gemini` had no output budget.** `ask_claude` caps at 4096 and raises
  on `stop_reason == "max_tokens"`, because a truncated reply arrives as
  "unparseable" and reads as a model failure when it is a budget failure.
  Gemini bills thinking against the same ceiling and the 3.x models think
  unasked — 676 thinking tokens against 251 of answer on one Pro call — so
  `XIAO_HEI_GEMINI_MAX_TOKENS` (8192) covers both, and `MAX_TOKENS` now raises
  with the thinking count in the message.

## What a key can call is a billing fact, not a catalogue fact

Free tier gives Pro a **daily quota of zero** — `GenerateRequestsPerDayPerProjectPerModel-FreeTier`
is hit on the first request of the day, so it is not a rate limit to wait out.
Free → Tier 1 is billing linked in AI Studio and takes effect immediately.
After linking, `gemini-3.1-pro-preview`, `gemini-pro-latest` and
`gemini-3-flash-preview` all went 429/403 → 200; the `gemini-2.5-*` family
stayed 404, which is retirement and not billing.

At the measured 6394 input tokens per grounding call:

| model | per call | per run (12 calls) | all 30 released questions |
|---|---|---|---|
| `gemini-3.1-pro-preview` | $0.031 | $0.37 | $11.08 |
| `gemini-3.6-flash` | $0.021 | $0.25 | $7.50 |
| `gemini-3.1-flash-lite` | $0.002 | $0.02 | $0.71 |

## One call each, against Opus 5 on the same image

`runs/cr_0811_03` step 4, *"the potted plant on the table"*, through our own
`ground()`:

| | box in pixels | latency | conf | says |
|---|---|---|---|---|
| `claude-opus-5` | `[183, 207, 315, 396]` | ~15-29 s | 0.93 | 1.8 m |
| `gemini-3.1-pro-preview` | `[188, 211, 323, 381]` | 12.0 s | 0.95 | 1.5 m |
| `gemini-3.6-flash` | `[186, 211, 325, 381]` | 13.2 s | 0.95 | 1.8 m |
| `gemini-3.1-flash-lite` | `[187, 211, 324, 381]` | 2.7 s | 1.00 | 1.5 m |

Box centres agree to about 8 px on a 640 px face. That is one image and one
phrase — it says the plumbing is right, not that the grounding is as good.

## Two things to watch before trusting a Gemini run

- **`gemini-3.1-flash-lite` returned `confidence: 1.00` and no
  `feature_box_2d`.** `bind_target` arbitrates with confidence
  (`switched and conf >= bound["conf"]`), so a constant 1.0 makes every new
  reading outrank the binding; and the lift falls back from `feature_box_2d`
  to the whole box, which moves the implied height that `anchor_size` checks.
- Nothing here has been driven. This is one call per model against recorded
  frames.

628 tests pass, 14 new.
