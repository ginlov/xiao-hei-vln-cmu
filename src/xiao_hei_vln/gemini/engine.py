"""Multimodal inference engine wrapping the ``google-genai`` SDK.

The :class:`GeminiEngine` is a thin client whose only job is to take a
``(system_prompt, user_text, [images])`` triple and return a parsed
:class:`xiao_hei_vln.messages.VLMOutput`. JSON-mode is enforced via the
SDK's ``response_mime_type='application/json'`` and the canonical
:class:`VLMOutput` JSON schema, so malformed responses are virtually
eliminated.

Tests inject a fake client implementing :class:`GeminiClientProtocol`,
so no real API key or network round-trip is needed for unit testing.
The SDK itself is imported lazily inside ``__init__`` to keep this
module importable in environments without ``google-genai`` installed.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import TypeAdapter

from xiao_hei_vln.gemini.config import GeminiConfig
from xiao_hei_vln.messages import VLMOutput, parse_vlm_output

if TYPE_CHECKING:  # pragma: no cover
    pass

log = logging.getLogger(__name__)


# Same schema the Qwen engine uses for guided decoding — kept in sync via
# the canonical VLMOutput TypeAdapter.
_VLM_OUTPUT_SCHEMA = TypeAdapter(VLMOutput).json_schema()


class GeminiClientProtocol(Protocol):
    """Surface the engine depends on; lets tests inject fakes."""

    def generate_content(
        self,
        *,
        model: str,
        contents: list[Any],
        config: Any,
    ) -> Any:
        ...


class GeminiEngineProtocol(Protocol):
    """The engine surface :class:`GeminiResponder` depends on.

    Wider than :class:`xiao_hei_vln.qwen.engine.EngineProtocol` because
    we always go multimodal (panorama + occupancy map) plus a JSON
    scene-graph dump. The Qwen ``EngineProtocol`` only takes a single
    ``ImageFrame``; this one takes pre-serialised image bytes.
    """

    def infer_multimodal(
        self,
        *,
        system: str,
        user_text: str,
        images: list[bytes],
    ) -> VLMOutput:
        ...

    def warmup(self) -> None:
        ...


class GeminiEngine:
    """Real :class:`google-genai`-backed engine."""

    def __init__(
        self,
        config: GeminiConfig,
        *,
        client: GeminiClientProtocol | None = None,
    ) -> None:
        self._config = config
        if client is not None:
            self._client = client
        else:
            # Lazy import so tests + the `dummy` responder path can run
            # without the gemini extra.
            from google import genai  # type: ignore[import-not-found]

            self._client = genai.Client(api_key=config.api_key)

    # --- public API ----------------------------------------------------

    def warmup(self) -> None:
        """Issue a minimal request to confirm credentials + reachability.

        Fails fast with a clear error if the API key is rejected or the
        SDK can't reach the model — matches the warmup pattern used by
        :class:`xiao_hei_vln.qwen.engine.HTTPQwenEngine`.
        """
        try:
            self.infer_multimodal(
                system="warmup",
                user_text=(
                    "Return JSON: "
                    '{"kind":"numerical","value":0,"rationale":"warmup"}'
                ),
                images=[],
            )
        except Exception:
            log.exception("GeminiEngine warmup failed")
            raise
        log.info("GeminiEngine warmup complete (model=%s)", self._config.model)

    def infer_multimodal(
        self,
        *,
        system: str,
        user_text: str,
        images: list[bytes],
    ) -> VLMOutput:
        """Issue one Gemini call and return a parsed :class:`VLMOutput`.

        ``images`` is a list of raw image bytes (PNG/JPEG). Each becomes
        an inline ``Part`` attached to the user turn; the SDK figures out
        the MIME type from the first few bytes.
        """
        contents = self._build_contents(user_text=user_text, images=images)
        gen_config = self._build_generation_config(system=system)

        response = self._client.generate_content(
            model=self._config.model,
            contents=contents,
            config=gen_config,
        )
        text = _extract_text(response)
        return parse_vlm_output(json.loads(text))

    # --- internals -----------------------------------------------------

    def _build_contents(
        self,
        *,
        user_text: str,
        images: list[bytes],
    ) -> list[Any]:
        """Build the SDK ``contents`` array for one ``generate_content`` call.

        We construct primitive dicts/Parts that the SDK accepts directly,
        rather than depending on a specific google-genai class hierarchy
        (the SDK accepts both ``types.Part`` instances and plain dicts in
        practice). Keeping this loose lets tests pass MagicMock-friendly
        primitives without instantiating real SDK objects.
        """
        parts: list[Any] = []
        for blob in images:
            mime = _sniff_mime(blob)
            parts.append({"inline_data": {"mime_type": mime, "data": blob}})
        parts.append({"text": user_text})
        return [{"role": "user", "parts": parts}]

    def _build_generation_config(self, *, system: str) -> Any:
        """JSON-mode config plus optional response schema.

        Returns a plain dict the SDK accepts as a ``GenerateContentConfig``.
        """
        return {
            "system_instruction": system,
            "temperature": self._config.temperature,
            "max_output_tokens": self._config.max_output_tokens,
            "response_mime_type": "application/json",
            "response_schema": _VLM_OUTPUT_SCHEMA,
        }


# --- helpers ---------------------------------------------------------------


def _sniff_mime(blob: bytes) -> str:
    """Detect MIME type from the first few bytes.

    Supports PNG and JPEG — the only formats produced by
    :mod:`xiao_hei_vln.gemini.scene_rep`. Falls back to JPEG.
    """
    if blob[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if blob[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    return "image/jpeg"


def _extract_text(response: Any) -> str:
    """Pull the response text out of the SDK response object.

    google-genai exposes the convenience attribute ``.text`` on the
    response, but tests use ``MagicMock`` which won't auto-populate it.
    Falls back to ``candidates[0].content.parts[0].text`` if ``.text``
    is missing.
    """
    text = getattr(response, "text", None)
    if isinstance(text, str) and text:
        return text
    try:
        candidate = response.candidates[0]
        return candidate.content.parts[0].text
    except Exception as exc:  # pragma: no cover — defensive
        raise RuntimeError(
            "Could not extract text from Gemini response: "
            f"got type={type(response).__name__}",
        ) from exc
