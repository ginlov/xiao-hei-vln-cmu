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
import time
from typing import TYPE_CHECKING, Any, Protocol

from xiao_hei_vln.gemini.config import GeminiConfig
from xiao_hei_vln.messages import VLMOutput, parse_vlm_output

if TYPE_CHECKING:  # pragma: no cover
    from xiao_hei_vln.gemini.trace import GeminiTracer

log = logging.getLogger(__name__)


class _ModelsNamespaceProtocol(Protocol):
    """The ``client.models`` namespace surface we depend on."""

    def generate_content(
        self,
        *,
        model: str,
        contents: list[Any],
        config: Any,
    ) -> Any:
        ...


class GeminiClientProtocol(Protocol):
    """The ``genai.Client`` surface we depend on.

    The real SDK routes ``generate_content`` through ``client.models`` —
    not the top-level client. We expose only that namespace so tests can
    inject a fake without modelling the rest of the client.
    """

    @property
    def models(self) -> _ModelsNamespaceProtocol:
        ...


class GeminiEngineProtocol(Protocol):
    """The engine surface :class:`SceneGeminiResponder` depends on.

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
        tracer: GeminiTracer | None = None,
    ) -> None:
        self._config = config
        # Optional debug tracer: records the full request + raw response +
        # usage + latency + errors of every call. See gemini.trace.
        self._tracer = tracer
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

        if self._tracer is None:
            response = self._client.models.generate_content(
                model=self._config.model,
                contents=contents,
                config=gen_config,
            )
            text = _extract_text(response)
            return parse_vlm_output(_loads_lenient(text))

        return self._traced_call(
            system=system, user_text=user_text, images=images,
            contents=contents, gen_config=gen_config,
        )

    def _traced_call(
        self,
        *,
        system: str,
        user_text: str,
        images: list[bytes],
        contents: list[Any],
        gen_config: Any,
    ) -> VLMOutput:
        """`infer_multimodal` body with full-fidelity tracing around it."""
        assert self._tracer is not None
        request = {
            "model": self._config.model,
            "system": system,
            "user_text": user_text,
            "num_images": len(images),
            "image_bytes": [len(b) for b in images],
            "images": self._tracer.save_images(images),
            "temperature": self._config.temperature,
            "max_output_tokens": self._config.max_output_tokens,
        }
        t0 = time.perf_counter()
        response: Any = None
        text: str | None = None
        try:
            response = self._client.models.generate_content(
                model=self._config.model,
                contents=contents,
                config=gen_config,
            )
            text = _extract_text(response)
            parsed = parse_vlm_output(_loads_lenient(text))
        except Exception as exc:
            self._tracer.record(
                request=request,
                response=_response_meta(response, text),
                parsed=None,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                error=repr(exc),
            )
            raise
        self._tracer.record(
            request=request,
            response=_response_meta(response, text),
            parsed=parsed.model_dump(),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            error=None,
        )
        return parsed

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
        """JSON-mode config.

        Returns a plain dict the SDK accepts as a ``GenerateContentConfig``.

        We deliberately do **not** pass ``response_schema``:
        ``VLMOutput`` is a discriminated union (``numerical |
        object_reference | waypoint_path``), which the SDK's schema
        translator cannot represent cleanly today (``oneOf`` +
        ``discriminator`` aren't in the OpenAPI-3 subset Gemini accepts).
        Instead we lean on ``response_mime_type='application/json'``
        plus the strict JSON-shape contract in the system prompt —
        Gemini is already very reliable in that mode, and
        :func:`parse_vlm_output` handles the discriminator on our side.

        ``thinking_config`` disables (or bounds) the model's reasoning
        tokens. gemini-2.5-flash otherwise spends most of
        ``max_output_tokens`` on hidden thinking and truncates the JSON
        answer; a structured-output task doesn't need it.
        """
        return {
            "system_instruction": system,
            "temperature": self._config.temperature,
            "max_output_tokens": self._config.max_output_tokens,
            "response_mime_type": "application/json",
            "thinking_config": {"thinking_budget": self._config.thinking_budget},
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


def _loads_lenient(text: str) -> Any:
    """Parse the first JSON object from a model response, tolerating slop.

    gemini models occasionally wrap the JSON in a ```json fence or emit a
    stray trailing character (e.g. an extra ``}``) after a valid object,
    which trips ``json.loads`` ("Extra data"). We strip a fence, seek the
    first ``{``, and use ``raw_decode`` so anything after the first
    complete object is ignored.
    """
    s = text.strip()
    if s.startswith("```"):
        # ```json\n{...}\n``` — take the fenced body.
        s = s[3:]
        if s[:4].lower() == "json":
            s = s[4:]
        end = s.rfind("```")
        if end != -1:
            s = s[:end]
        s = s.strip()
    start = s.find("{")
    if start > 0:
        s = s[start:]
    obj, _end = json.JSONDecoder().raw_decode(s)
    return obj


def _response_meta(response: Any, text: str | None) -> dict[str, Any] | None:
    """Best-effort structured view of a Gemini response for the trace log.

    Pulls the raw text, finish reason, and token usage when present.
    Every field is guarded — a fake/partial response (or ``None`` after a
    failed call) degrades to whatever is available.
    """
    if response is None and text is None:
        return None
    meta: dict[str, Any] = {"raw_text": text}
    try:
        meta["finish_reason"] = str(response.candidates[0].finish_reason)
    except Exception:  # noqa: BLE001 — optional field
        pass
    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        meta["usage"] = {
            "prompt_tokens": getattr(usage, "prompt_token_count", None),
            "candidates_tokens": getattr(usage, "candidates_token_count", None),
            "total_tokens": getattr(usage, "total_token_count", None),
        }
    return meta


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
