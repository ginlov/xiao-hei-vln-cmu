# Task 3 — Phase 1: Serving framework for Qwen 3.5

This document records the framework choice that backs the
`xiao_hei_vln.qwen` package added in Phase 2. The decision is driven
by the tick cadence and synchronization strategy defined in Task 1
(`docs/task1_io_spec.md`) and the container layout from Task 2
(`docker/`).

## Constraints

| Constraint | Source | Value |
|---|---|---|
| VLM tick period | `docs/task1_io_spec.md` §3 | 500 ms (2 Hz default, configurable 0.5–5 Hz via `XIAO_HEI_VLM_TICK_HZ`) |
| Snapshot freshness | Phase-1 measurements | ≤ 110 ms behind slowest input (camera @ 9.3 Hz) |
| Inputs available per tick | `VLMInput` | one 1920×640 BGR image, two `(N, 4)` lidar scans, two terrain maps, an `OdomPose`, a `ChallengeQuestion` |
| Output schema | `VLMOutput` (discriminated union) | `numerical | object_reference | waypoint_path` |
| Eval host | User confirmed | single NVIDIA GPU (24 GB-class, e.g. RTX 4090), Ubuntu 24.04 + ROS Jazzy container |
| Process model | User confirmed | **in-process** engine inside `xiao_hei_ai_module` (no HTTP sidecar) |

The hard budget for the VLM body is therefore **~400 ms per tick**
after subtracting overhead for the rclpy spin, `LatestCache.snapshot()`,
publisher serialization, and a comfort margin.

## Model: Qwen3.5 family

Qwen3.5 (released 2026-02-16) is natively multimodal — text, image,
and video are first-class inputs with no separate `-VL` checkpoint.
The HF collection ships dense models at 0.8B / 2B / 4B / 9B / 27B
and MoE models at 35B-A3B / 122B-A10B / 397B-A17.

Footprints on a 24 GB GPU (bf16 weights + KV cache for 4k context + a
single image ≈ 600–1200 vision tokens):

| Variant | Weights (bf16) | Weights (AWQ-Int4) | Headroom on 4090 |
|---|---|---|---|
| Qwen3.5-2B | ~4 GB | ~1.5 GB | very comfortable |
| **Qwen3.5-4B** (default) | ~8 GB | ~3 GB | comfortable, leaves ~14 GB for KV + images |
| Qwen3.5-9B | ~18 GB | ~5 GB | bf16 tight; AWQ-Int4 comfortable |
| Qwen3.5-27B | OOM at bf16 | ~14 GB | AWQ-Int4 only; marginal at our context length |
| 35B-A3B MoE | OOM | ~18 GB | doesn't leave room for KV |

**Initial pick: `Qwen/Qwen3.5-4B-Instruct` at bf16.** It is the
smallest member of the family that has all the capability gains from
the 3.5 native-multimodal training; cheap to bring up; trivially
swappable for `Qwen3.5-9B-Instruct-AWQ` if the eval shows quality is
short.

## Serving runtime: vLLM in-process

Three runtimes were considered for an in-process integration:

| Option | Why we'd pick it | Why we ruled it out (or didn't) |
|---|---|---|
| `transformers` `AutoModelForVision2Seq` | Simplest API; no extra deps | Eager decode is 3–5× slower than vLLM at our context length; no native guided JSON; we'd pay every tick |
| **vLLM `LLM` engine** | PagedAttention → low-latency single-stream decode at 2 Hz; native Qwen3.5 + multimodal image support; built-in `guided_json` / `guided_choice` for structured `VLMOutput` emission; same Python process so no HTTP cost | Heavy deps (CUDA + torch); not yet built for ARM eval hosts (not in scope here) |
| SGLang | Best-in-class prefix-caching and structured outputs | Same dep weight as vLLM; smaller community for Qwen3.5; we already chose in-process so its server story is moot |

Picked **vLLM `LLM` engine**, used as a Python object (not the
`vllm.entrypoints.openai.api_server` HTTP shim), because:

1. Single-stream decode latency at our context length sits in the
   ~200–300 ms range on a 4090 for a ~50-token JSON answer with a
   Qwen3.5-4B prefill of ~1.5k tokens (image + system + question +
   evidence log). That fits the 400 ms body budget.
2. **Guided JSON decoding** (`vllm.SamplingParams(guided_decoding=
   GuidedDecodingParams(json=<pydantic schema>))`) lets us emit a
   well-formed `VLMOutput` directly — no post-hoc parsing, no
   regex-cleanup of model preambles.
3. vLLM has first-class support for Qwen3.5 multimodal (per
   `docs.vllm.ai/projects/recipes/.../Qwen3.5.html`), including the
   pre/post-processor for the vision tokens.
4. Pure-Python embedding: import, construct one `LLM`, call
   `.chat(messages, sampling_params=...)` inside `respond()`. No
   subprocess, no port allocation, no health check, no compose service.

### Latency budget (single tick, Qwen3.5-4B bf16, 4090)

| Stage | Estimate | Notes |
|---|---|---|
| `snapshot()` + JSON-ify inputs | ~5 ms | already validated by Task 1 |
| Image resize + base64 / tensor handoff | ~10–20 ms | downscale 1920×640 → 1280×426 max; ~600 vision tokens |
| Prefill (~1.5k tokens) | ~80–150 ms | one image + system + question + evidence log |
| Decode (~50 tokens, guided JSON) | ~150–250 ms | 5 ms/token, plus FSM step for guidance |
| `parse_vlm_output()` + publish | ~5 ms | already validated |
| **Total** | **~250–430 ms** | Fits the 400 ms body budget; some risk at the top of the range |

If real measurements blow the budget we have, in order of preference:

1. Drop the tick rate to 1 Hz via `XIAO_HEI_VLM_TICK_HZ=1.0` — the
   contract already supports this.
2. Skip ticks while inference is in flight (the responder already
   tolerates `respond()` returning `None`).
3. Switch to `Qwen3.5-4B-Instruct-AWQ` once those checkpoints land
   (≈ 60–70 % of bf16 latency).
4. Step down to `Qwen3.5-2B-Instruct` for the slowest scenes.

## Image handling

Qwen3.5 expects RGB images. Our `ImageFrame.encoding == "bgr8"` and
`data` is raw bytes shaped `(height, step)`. The engine wrapper
converts at the boundary:

```python
arr = np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, 3)
rgb = arr[:, :, ::-1]                       # BGR → RGB, no copy
img = PIL.Image.fromarray(rgb, mode="RGB")  # vLLM's multimodal input format
```

Downscaling to a target long-edge of 1280 px halves prefill time at
no measurable quality loss for object-counting and routing prompts —
done once per tick, before handoff to vLLM.

## Structured output

`VLMOutput` is a Pydantic discriminated union (`kind` field). vLLM's
guided JSON decoding takes the model's JSON schema directly:

```python
schema = TypeAdapter(VLMOutput).json_schema()
params = SamplingParams(
    max_tokens=128,
    temperature=0.0,
    guided_decoding=GuidedDecodingParams(json=schema),
)
result = llm.chat(messages, sampling_params=params)
parsed = parse_vlm_output(json.loads(result[0].outputs[0].text))
```

This guarantees the model's emission is parseable by the existing
`parse_vlm_output()` adapter — no glue code between the engine and
`VLMOutputPublisher`.

## Packaging

vLLM and its CUDA dependencies are too heavy to ship as a hard
requirement of the package (it would break dev installs on the dev
laptop, CI, and any tester without a GPU). They live behind an
**optional dependency group** `qwen`:

```toml
[project.optional-dependencies]
qwen = [
    "vllm>=0.7,<0.10",        # Qwen3.5 multimodal + guided_json
    "pillow>=10",             # image handoff to vLLM
    "huggingface-hub>=0.24",  # checkpoint fetch
]
```

- Dev / CI / tests stay on the lean base install (`pydantic`, `numpy`).
- The docker image installs the `[qwen]` extra; weights are mounted
  via an `HF_HOME` cache volume in compose so a `compose down/up`
  doesn't redownload.
- The dummy responder still ships and remains the default when the
  `XIAO_HEI_RESPONDER` env var isn't `qwen` — useful for compose-only
  smoke tests and for environments without CUDA.

## Open risks / follow-ups

- **vLLM version pin**: Qwen3.5 multimodal support is rolling. We'll
  pin to the vLLM recipe page's recommended minor when wiring the
  Dockerfile in Phase 2.
- **First-call warmup**: the first `llm.chat()` after engine init
  hits a CUDA-graph capture step that can take 5–15 s. The engine
  wrapper will warm it during `__init__()` (off the rclpy thread)
  so the first real tick doesn't stall.
- **MEM-OOM on long instructions**: if the question text or evidence
  log grows unboundedly, we'll add a token-budget cap in the prompt
  builder. Not a Phase 1 concern.
- **Eval host not yet measured**: numbers above are budgeted, not
  observed. Phase 2 will land a `--measure-latency` mode on the app
  entry point so we can validate before the prompt design lands.
