# Qwen Engine

The Qwen engine handles communication with the VLM inference backend. Two
implementations share a common `EngineProtocol`:

```python
class EngineProtocol(Protocol):
    def infer(
        self, system: str, user_text: str, image: ImageFrame | None
    ) -> VLMOutput | None: ...

    def warmup(self) -> None: ...
```

## HTTPQwenEngine (default)

Calls a vLLM OpenAI-compatible HTTP server running in a separate container.

**Advantages:**

- No CUDA dependencies in the ai_module image
- No pip conflicts with the ROS base image
- Can scale the vLLM server independently

**How it works:**

1. Constructs an OpenAI-style message array (system + user with optional base64 image)
2. Sends `POST /v1/chat/completions` with guided decoding (JSON schema)
3. Parses the response JSON into a `VLMOutput`

**Configuration:**

```bash
XIAO_HEI_QWEN_VLLM_BASE_URL=http://localhost:8000/v1
XIAO_HEI_QWEN_MODEL=/models/Qwen3.5-4B
XIAO_HEI_QWEN_TEMPERATURE=0.0
XIAO_HEI_QWEN_MAX_OUTPUT_TOKENS=256
```

## QwenEngine (legacy, in-process)

Loads the vLLM engine directly in the same Python process. Requires CUDA and
`pip install .[qwen-local]`.

**When to use:**

- Single-machine development without docker-compose
- Debugging the inference pipeline directly

**Not recommended** for production because:

- vLLM's pip dependencies conflict with the ROS base image
- Couples GPU memory to the ai_module process
- Harder to debug and profile

## Guided decoding

Both engines use vLLM's guided decoding to force the model output to match
the `VLMOutput` JSON schema. This eliminates parsing failures and ensures
the model always returns a valid discriminated-union response.

The schema includes all three output types:

```json
{
  "oneOf": [
    {"$ref": "#/$defs/NumericalResponse"},
    {"$ref": "#/$defs/ObjectReferenceResponse"},
    {"$ref": "#/$defs/WaypointPathResponse"}
  ]
}
```

## Image preprocessing

Before sending to the engine, camera frames are:

1. Converted from BGR8 to RGB (OpenCV convention → PIL convention)
2. Downscaled if the long edge exceeds `image_long_edge` (default 1280px)
3. Encoded as JPEG and base64 for the HTTP API

This is handled by shared utilities in `xiao_hei_vln.qwen.image_utils`.

## Warmup

On startup, `HTTPQwenEngine.warmup()` polls the vLLM server until it's ready
(retries for up to 5 minutes), then fires a dummy inference to warm the KV
cache. This ensures the first real question doesn't hit cold-start latency.
