"""Tests for `HTTPQwenEngine` with a mock openai client.

No GPU, vLLM, or network access required — the openai client is
injected via the `client=` kwarg.
"""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from xiao_hei_vln.messages import NumericalResponse, WaypointPathResponse
from xiao_hei_vln.messages.common import Header, Stamp
from xiao_hei_vln.messages.sensors import ImageFrame
from xiao_hei_vln.qwen import QwenConfig
from xiao_hei_vln.qwen.engine import HTTPQwenEngine, _image_frame_to_data_url


def _config(**overrides) -> QwenConfig:
    defaults = {"vllm_base_url": "http://localhost:8000/v1"}
    defaults.update(overrides)
    return QwenConfig(**defaults)


def _mock_client(response_json: dict) -> MagicMock:
    """Build a mock openai client whose chat.completions.create returns `response_json`."""
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=json.dumps(response_json))),
        ],
    )
    client.models.list.return_value = []
    return client


def _image(width: int = 6, height: int = 4) -> ImageFrame:
    """Small synthetic BGR8 image."""
    data = np.zeros((height, width, 3), dtype=np.uint8)
    data[:, :, 0] = 255  # blue channel
    return ImageFrame(
        header=Header(stamp=Stamp(sec=0, nanosec=0), frame_id="camera"),
        width=width,
        height=height,
        encoding="bgr8",
        step=width * 3,
        data=data.tobytes(),
    )


# --- construction ---------------------------------------------------------


class TestConstruction:
    def test_requires_vllm_base_url(self) -> None:
        with pytest.raises(ValueError, match="vllm_base_url"):
            HTTPQwenEngine(QwenConfig())

    def test_accepts_injected_client(self) -> None:
        client = _mock_client({"kind": "numerical", "value": 0})
        engine = HTTPQwenEngine(_config(), client=client)
        assert engine._client is client


# --- infer (text only) ----------------------------------------------------


class TestTextOnlyInfer:
    def test_returns_parsed_vlm_output(self) -> None:
        client = _mock_client({"kind": "numerical", "value": 5})
        engine = HTTPQwenEngine(_config(), client=client)
        out = engine.infer(system="test", user_text="hello", image=None)
        assert isinstance(out, NumericalResponse)
        assert out.value == 5

    def test_messages_have_no_image_url(self) -> None:
        client = _mock_client({"kind": "numerical", "value": 0})
        engine = HTTPQwenEngine(_config(), client=client)
        engine.infer(system="sys", user_text="hello", image=None)
        call_kwargs = client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        user_content = messages[1]["content"]
        assert all(item["type"] != "image_url" for item in user_content)

    def test_system_message_is_first(self) -> None:
        client = _mock_client({"kind": "numerical", "value": 0})
        engine = HTTPQwenEngine(_config(), client=client)
        engine.infer(system="my system prompt", user_text="q", image=None)
        messages = client.chat.completions.create.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "my system prompt"


# --- infer (with image) --------------------------------------------------


class TestImageInfer:
    def test_messages_include_base64_image_url(self) -> None:
        client = _mock_client(
            {"kind": "waypoint_path", "waypoints": [{"x": 1.0, "y": 2.0}]},
        )
        engine = HTTPQwenEngine(_config(), client=client)
        out = engine.infer(system="s", user_text="q", image=_image())
        assert isinstance(out, WaypointPathResponse)
        messages = client.chat.completions.create.call_args.kwargs["messages"]
        user_content = messages[1]["content"]
        image_items = [i for i in user_content if i["type"] == "image_url"]
        assert len(image_items) == 1
        url = image_items[0]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")

    def test_base64_decodes_to_valid_jpeg(self) -> None:
        client = _mock_client({"kind": "numerical", "value": 0})
        engine = HTTPQwenEngine(_config(), client=client)
        engine.infer(system="s", user_text="q", image=_image())
        url = (
            client.chat.completions.create.call_args.kwargs["messages"][1]["content"][0][
                "image_url"
            ]["url"]
        )
        b64_data = url.split(",", 1)[1]
        jpeg_bytes = base64.b64decode(b64_data)
        assert jpeg_bytes[:2] == b"\xff\xd8"  # JPEG magic bytes


# --- guided decoding format -----------------------------------------------


class TestGuidedDecoding:
    def test_response_format_includes_json_schema(self) -> None:
        client = _mock_client({"kind": "numerical", "value": 0})
        engine = HTTPQwenEngine(_config(), client=client)
        engine.infer(system="s", user_text="q", image=None)
        rf = client.chat.completions.create.call_args.kwargs["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "VLMOutput"
        schema = rf["json_schema"]["schema"]
        assert "$defs" in schema or "properties" in schema


# --- sampling params forwarded -------------------------------------------


class TestSamplingParams:
    def test_temperature_and_max_tokens_forwarded(self) -> None:
        cfg = _config(temperature=0.7, max_output_tokens=128, seed=42)
        client = _mock_client({"kind": "numerical", "value": 0})
        engine = HTTPQwenEngine(cfg, client=client)
        engine.infer(system="s", user_text="q", image=None)
        kw = client.chat.completions.create.call_args.kwargs
        assert kw["temperature"] == pytest.approx(0.7)
        assert kw["max_tokens"] == 128
        assert kw["seed"] == 42


# --- warmup ---------------------------------------------------------------


class TestWarmup:
    def test_warmup_calls_models_list_then_infer(self) -> None:
        client = _mock_client({"kind": "numerical", "value": 0})
        engine = HTTPQwenEngine(_config(), client=client)
        engine.warmup()
        client.models.list.assert_called_once()
        assert client.chat.completions.create.call_count == 1

    def test_warmup_retries_on_connection_error(self, monkeypatch) -> None:
        client = _mock_client({"kind": "numerical", "value": 0})
        # Fail twice, then succeed.
        client.models.list.side_effect = [
            ConnectionError("refused"),
            ConnectionError("refused"),
            [],
        ]
        engine = HTTPQwenEngine(_config(), client=client)
        # Patch sleep so the test is instant.
        monkeypatch.setattr("time.sleep", lambda _: None)
        engine.warmup()
        assert client.models.list.call_count == 3

    def test_warmup_gives_up_after_timeout(self, monkeypatch) -> None:
        client = MagicMock()
        client.models.list.side_effect = ConnectionError("refused")
        engine = HTTPQwenEngine(
            _config(),
            client=client,
        )
        monkeypatch.setattr("time.sleep", lambda _: None)
        # The retry loop uses elapsed < max_wait (300) with interval=5.
        # With sleep patched to no-op, elapsed increments by 5 each
        # iteration → 60 iterations → 60 calls.
        with pytest.raises(RuntimeError, match="not ready after"):
            engine.warmup()
        assert client.models.list.call_count == 60


# --- _image_frame_to_data_url -------------------------------------------


class TestImageFrameToDataUrl:
    def test_small_image_roundtrips(self) -> None:
        frame = _image(width=10, height=8)
        url = _image_frame_to_data_url(frame, long_edge=1280)
        assert url.startswith("data:image/jpeg;base64,")
        jpeg_bytes = base64.b64decode(url.split(",", 1)[1])
        assert jpeg_bytes[:2] == b"\xff\xd8"

    def test_large_image_is_downscaled(self) -> None:
        from PIL import Image

        frame = _image(width=1920, height=640)
        url = _image_frame_to_data_url(frame, long_edge=960)
        jpeg_bytes = base64.b64decode(url.split(",", 1)[1])
        import io

        img = Image.open(io.BytesIO(jpeg_bytes))
        assert max(img.width, img.height) == 960
