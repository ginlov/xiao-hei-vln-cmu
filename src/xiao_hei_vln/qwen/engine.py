"""vLLM-backed inference engine for Qwen3.5.

`QwenEngine` lazily imports `vllm`, `transformers`, and `PIL` only when
constructed, so this module can be imported in test / dev / dummy
environments without those packages installed. Tests inject an
`EngineProtocol`-shaped fake into `QwenResponder` instead of building
a real engine.

Outputs are decoded with vLLM `guided_decoding=GuidedDecodingParams(json=...)`
against the `VLMOutput` JSON schema, so the model's emission is always
parseable by `parse_vlm_output()`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import TypeAdapter

from xiao_hei_vln.messages import VLMOutput, parse_vlm_output
from xiao_hei_vln.messages.sensors import ImageFrame
from xiao_hei_vln.qwen.config import QwenConfig

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
            limit_mm_per_prompt={"image": 1},
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
        pil_image = _image_frame_to_pil(image, long_edge=self._config.image_long_edge)
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


def _image_frame_to_pil(frame: ImageFrame | None, *, long_edge: int) -> PILImage | None:
    if frame is None:
        return None

    import numpy as np
    from PIL import Image  # type: ignore[import-not-found]

    if frame.encoding != "bgr8":
        raise ValueError(f"unsupported image encoding {frame.encoding!r}; expected 'bgr8'")

    arr = np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, 3)
    rgb = arr[:, :, ::-1]
    img = Image.fromarray(rgb, mode="RGB")

    longest = max(img.width, img.height)
    if longest > long_edge:
        scale = long_edge / longest
        img = img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
            Image.Resampling.BILINEAR,
        )
    return img
