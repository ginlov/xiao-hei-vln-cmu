"""Gemini API wrapper with a small testable protocol."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from xiao_hei_vln.messages.sensors import ImageFrame


class GeminiEngineProtocol(Protocol):
    def infer(self, system: str, user_text: str, image: ImageFrame | None) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class GeminiConfig:
    api_key: str
    model: str = "gemini-2.5-flash"
    temperature: float = 0.0
    image_long_edge: int = 1024


class GeminiEngine:
    """Thin wrapper around ``google-genai``.

    Imports are lazy so tests and dummy-only runs do not need the optional
    Gemini dependency installed.
    """

    def __init__(self, config: GeminiConfig) -> None:
        self._config = config
        from google import genai  # type: ignore[import-not-found]

        self._client = genai.Client(api_key=config.api_key)

    def infer(self, system: str, user_text: str, image: ImageFrame | None) -> dict[str, Any]:
        from google.genai import types  # type: ignore[import-not-found]

        contents: list[Any] = []
        if image is not None:
            contents.append(_image_frame_to_gemini_part(image, self._config.image_long_edge))
        contents.append(user_text)

        response = self._client.models.generate_content(
            model=self._config.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                temperature=self._config.temperature,
            ),
        )
        return _parse_json_response(response.text or "")


def _parse_json_response(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        cleaned = cleaned.removesuffix("```").strip()
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("Gemini response must be a JSON object")
    return data


def _image_frame_to_gemini_part(frame: ImageFrame, long_edge: int):
    import io

    from google.genai import types  # type: ignore[import-not-found]

    from xiao_hei_vln.image_utils import image_frame_to_pil, resize_pil

    pil_image = resize_pil(image_frame_to_pil(frame), long_edge=long_edge)
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=90)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")
