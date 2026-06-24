"""Tests for ``xiao_hei_vln.gemini.engine``.

All tests use a fake :class:`GeminiClientProtocol` — no real SDK,
no network calls, no API key required.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from xiao_hei_vln.gemini.config import GeminiConfig
from xiao_hei_vln.gemini.engine import (
    _VLM_OUTPUT_SCHEMA,
    GeminiEngine,
    _sniff_mime,
)
from xiao_hei_vln.messages import NumericalResponse, WaypointPathResponse


def _config() -> GeminiConfig:
    return GeminiConfig(api_key="test-key", model="gemini-2.5-flash")


def _fake_client_returning(json_payload: dict) -> MagicMock:
    """A MagicMock client whose `generate_content` returns the given JSON."""
    client = MagicMock()
    response = MagicMock()
    response.text = json.dumps(json_payload)
    client.generate_content.return_value = response
    return client


class TestSniffMime:
    def test_png(self) -> None:
        assert _sniff_mime(b"\x89PNG\r\n\x1a\n" + b"rest") == "image/png"

    def test_jpeg(self) -> None:
        assert _sniff_mime(b"\xff\xd8\xff" + b"rest") == "image/jpeg"

    def test_unknown_defaults_to_jpeg(self) -> None:
        assert _sniff_mime(b"unknown bytes") == "image/jpeg"


class TestInferMultimodal:
    def test_returns_parsed_numerical_response(self) -> None:
        client = _fake_client_returning({
            "kind": "numerical",
            "value": 3,
            "rationale": "saw 3 cups",
        })
        engine = GeminiEngine(_config(), client=client)
        out = engine.infer_multimodal(
            system="system prompt",
            user_text="user text",
            images=[],
        )
        assert isinstance(out, NumericalResponse)
        assert out.value == 3
        assert out.rationale == "saw 3 cups"

    def test_returns_parsed_waypoint_response(self) -> None:
        client = _fake_client_returning({
            "kind": "waypoint_path",
            "waypoints": [
                {"x": 1.0, "y": 2.0, "heading": 0.0},
                {"x": 3.0, "y": 4.0, "heading": 1.5},
            ],
            "rationale": "two-step path",
        })
        engine = GeminiEngine(_config(), client=client)
        out = engine.infer_multimodal(system="s", user_text="u", images=[])
        assert isinstance(out, WaypointPathResponse)
        assert len(out.waypoints) == 2
        assert out.waypoints[1].x == pytest.approx(3.0)

    def test_call_passes_system_to_config(self) -> None:
        client = _fake_client_returning({
            "kind": "numerical", "value": 1, "rationale": "ok",
        })
        engine = GeminiEngine(_config(), client=client)
        engine.infer_multimodal(system="MY-SYSTEM", user_text="text", images=[])
        # Inspect what was sent to the SDK
        call = client.generate_content.call_args
        config = call.kwargs["config"]
        assert config["system_instruction"] == "MY-SYSTEM"
        assert config["response_mime_type"] == "application/json"
        assert config["response_schema"] == _VLM_OUTPUT_SCHEMA

    def test_call_attaches_images_with_mime(self) -> None:
        client = _fake_client_returning({
            "kind": "numerical", "value": 0, "rationale": "ok",
        })
        engine = GeminiEngine(_config(), client=client)
        png = b"\x89PNG\r\n\x1a\n" + b"fake-png-rest"
        jpg = b"\xff\xd8\xff" + b"fake-jpeg-rest"
        engine.infer_multimodal(
            system="s",
            user_text="here are images",
            images=[png, jpg],
        )
        contents = client.generate_content.call_args.kwargs["contents"]
        assert len(contents) == 1
        parts = contents[0]["parts"]
        # 2 inline images + 1 text part
        assert len(parts) == 3
        assert parts[0]["inline_data"]["mime_type"] == "image/png"
        assert parts[1]["inline_data"]["mime_type"] == "image/jpeg"
        assert parts[2]["text"] == "here are images"

    def test_no_images_still_has_text_part(self) -> None:
        client = _fake_client_returning({
            "kind": "numerical", "value": 0, "rationale": "ok",
        })
        engine = GeminiEngine(_config(), client=client)
        engine.infer_multimodal(system="s", user_text="just text", images=[])
        parts = client.generate_content.call_args.kwargs["contents"][0]["parts"]
        assert len(parts) == 1
        assert parts[0]["text"] == "just text"


class TestWarmup:
    def test_warmup_issues_one_call(self) -> None:
        client = _fake_client_returning({
            "kind": "numerical", "value": 0, "rationale": "warmup",
        })
        engine = GeminiEngine(_config(), client=client)
        engine.warmup()
        assert client.generate_content.call_count == 1

    def test_warmup_propagates_errors(self) -> None:
        client = MagicMock()
        client.generate_content.side_effect = RuntimeError("auth failed")
        engine = GeminiEngine(_config(), client=client)
        with pytest.raises(RuntimeError, match="auth failed"):
            engine.warmup()


class TestExtractTextFallback:
    def test_falls_back_to_candidates_when_text_missing(self) -> None:
        client = MagicMock()
        response = MagicMock(spec=[])  # no `text` attr
        candidate = MagicMock()
        candidate.content.parts = [MagicMock(text=json.dumps({
            "kind": "numerical", "value": 7, "rationale": "from candidates",
        }))]
        response.candidates = [candidate]
        client.generate_content.return_value = response
        engine = GeminiEngine(_config(), client=client)
        out = engine.infer_multimodal(system="s", user_text="u", images=[])
        assert isinstance(out, NumericalResponse)
        assert out.value == 7
