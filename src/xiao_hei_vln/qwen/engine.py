"""Inference engines for Qwen3.5.

Two implementations of `EngineProtocol`:

- `HTTPQwenEngine` (recommended) — calls a vLLM OpenAI-compatible HTTP
  server running as a docker-compose sidecar. The ai_module image only
  needs the lightweight `openai` + `pillow` packages. No CUDA deps.
- `QwenEngine` (legacy) — loads vLLM in-process. Requires `pip install
  .[qwen-local]` and a CUDA GPU in the same container. Kept for
  single-process local-GPU development without docker-compose.

Both are lazy-imported — this module can be imported in test / dev /
dummy environments without those packages installed. Tests inject an
`EngineProtocol`-shaped fake into `QwenResponder`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import TypeAdapter

from xiao_hei_vln.messages import VLMOutput, parse_vlm_output
from xiao_hei_vln.messages.sensors import ImageFrame
from xiao_hei_vln.qwen.config import QwenConfig
from xiao_hei_vln.image_utils import image_frame_to_pil, resize_pil

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


class EngineProtocol(Protocol):
    """Minimal surface the responder depends on; used to inject fakes in tests."""

    def infer(self, system: str, user_text: str, image: ImageFrame | None) -> VLMOutput:
        ...


_VLM_OUTPUT_SCHEMA = TypeAdapter(VLMOutput).json_schema()


class QwenEngine:
    """In-process vLLM wrapper around `Qwen/Qwen3.5-*-Instruct` checkpoints."""

    def __init__(self, config: QwenConfig | None = None) -> None:
        self._config = config or QwenConfig()
        # Heavy imports happen here so module import stays cheap.
        from transformers import AutoProcessor  # type: ignore[import-not-found]
        from vllm import LLM  # type: ignore[import-not-found]
        from vllm.sampling_params import (  # type: ignore[import-not-found]
            GuidedDecodingParams,
            SamplingParams,
        )

        self._llm = LLM(
            model=self._config.model,
            dtype=self._config.dtype,
            max_model_len=self._config.max_model_len,
            gpu_memory_utilization=self._config.gpu_memory_utilization,
            trust_remote_code=self._config.trust_remote_code,
            seed=self._config.seed if self._config.seed is not None else 0,
            limit_mm_per_prompt={"image": 4},
        )
        self._processor = AutoProcessor.from_pretrained(
            self._config.model,
            trust_remote_code=self._config.trust_remote_code,
        )
        self._sampling = SamplingParams(
            temperature=self._config.temperature,
            max_tokens=self._config.max_output_tokens,
            seed=self._config.seed,
            guided_decoding=GuidedDecodingParams(json=_VLM_OUTPUT_SCHEMA),
        )

    # --- public surface ---------------------------------------------------

    def warmup(self) -> None:
        """Force CUDA-graph capture + tokenizer JIT before the first real tick."""
        self.infer(
            system="warmup",
            user_text='{"kind":"numerical","value":0}',
            image=None,
        )

    def infer(self, system: str, user_text: str, image: ImageFrame | None) -> VLMOutput:
        pil_image = _prepare_image(image, long_edge=self._config.image_long_edge)
        prompt = self._render_chat_prompt(system, user_text, has_image=pil_image is not None)

        engine_input: dict[str, Any] = {"prompt": prompt}
        if pil_image is not None:
            engine_input["multi_modal_data"] = {"image": pil_image}

        outputs = self._llm.generate([engine_input], sampling_params=self._sampling)
        text = outputs[0].outputs[0].text
        return parse_vlm_output(json.loads(text))

    # --- internals --------------------------------------------------------

    def _render_chat_prompt(self, system: str, user_text: str, *, has_image: bool) -> str:
        user_content: list[dict[str, Any]] = []
        if has_image:
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": user_text})
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        return self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


class HTTPQwenEngine:
    """Calls a vLLM OpenAI-compatible HTTP server (sidecar container).

    The ai_module image needs only `openai` + `pillow` — no vLLM, torch,
    or CUDA wheels. The vLLM server runs in a separate container built
    from the official `vllm/vllm-openai` image.
    """

    def __init__(self, config: QwenConfig, *, client: Any | None = None) -> None:
        self._config = config
        if config.vllm_base_url is None:
            raise ValueError("HTTPQwenEngine requires config.vllm_base_url to be set")

        if client is not None:
            self._client = client
        else:
            from openai import OpenAI  # type: ignore[import-not-found]

            self._client = OpenAI(
                base_url=config.vllm_base_url,
                api_key="EMPTY",
                timeout=60.0,
            )
        self._model = config.model
        self._response_format: dict[str, Any] = {
            "type": "json_schema",
            "json_schema": {
                "name": "VLMOutput",
                "schema": _VLM_OUTPUT_SCHEMA,
            },
        }

    # --- public surface ---------------------------------------------------

    def warmup(self) -> None:
        """Block until the vLLM server is reachable, then run a test inference."""
        import logging
        import time

        log = logging.getLogger(__name__)
        max_wait = 300
        interval = 5
        elapsed = 0

        while elapsed < max_wait:
            try:
                self._client.models.list()
                break
            except Exception:
                log.info(
                    "Waiting for vLLM server at %s… (%ds)",
                    self._config.vllm_base_url,
                    elapsed,
                )
                time.sleep(interval)
                elapsed += interval
        else:
            raise RuntimeError(
                f"vLLM server at {self._config.vllm_base_url} "
                f"not ready after {max_wait}s",
            )

        self.infer(
            system="warmup",
            user_text='{"kind":"numerical","value":0}',
            image=None,
        )
        log.info("HTTPQwenEngine warmup complete")

    def infer(self, system: str, user_text: str, image: ImageFrame | None) -> VLMOutput:
        messages = self._build_messages(system, user_text, image)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._config.temperature,
            max_tokens=self._config.max_output_tokens,
            seed=self._config.seed,
            response_format=self._response_format,
        )
        text = response.choices[0].message.content
        return parse_vlm_output(json.loads(text))

    # --- internals --------------------------------------------------------

    def _build_messages(
        self,
        system: str,
        user_text: str,
        image: ImageFrame | None,
    ) -> list[dict[str, Any]]:
        user_content: list[dict[str, Any]] = []
        if image is not None:
            data_url = _image_frame_to_data_url(
                image,
                long_edge=self._config.image_long_edge,
            )
            user_content.append(
                {"type": "image_url", "image_url": {"url": data_url}},
            )
        user_content.append({"type": "text", "text": user_text})
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]


# --- image helpers --------------------------------------------------------


def _image_frame_to_data_url(frame: ImageFrame, *, long_edge: int) -> str:
    """Convert an `ImageFrame` (BGR8 bytes) to a base64 JPEG data URL."""
    import base64
    import io

    pil_image = _prepare_image(frame, long_edge=long_edge)
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _prepare_image(frame: ImageFrame | None, *, long_edge: int) -> PILImage | None:
    if frame is None:
        return None
    return resize_pil(image_frame_to_pil(frame), long_edge=long_edge)
