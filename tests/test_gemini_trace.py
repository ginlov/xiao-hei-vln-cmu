"""Tests for the Gemini call tracer (`xiao_hei_vln.gemini.trace`) and its
integration into `GeminiEngine`.

No real SDK / network: a hand-rolled fake client returns plain objects so
the tracer's JSON serialisation is exercised for real.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from xiao_hei_vln.gemini.config import GeminiConfig
from xiao_hei_vln.gemini.engine import GeminiEngine
from xiao_hei_vln.gemini.trace import GeminiTracer
from xiao_hei_vln.messages import NumericalResponse

PNG = b"\x89PNG\r\n\x1a\n" + b"pngbody"
JPG = b"\xff\xd8\xff" + b"jpgbody"


def _config() -> GeminiConfig:
    return GeminiConfig(api_key="test-key", model="gemini-2.5-flash", temperature=0.2)


def _resp(text: str, *, usage: object | None = None, finish_reason: str = "STOP"):
    return types.SimpleNamespace(
        text=text,
        usage_metadata=usage,
        candidates=[types.SimpleNamespace(finish_reason=finish_reason)],
    )


def _client(resp=None, exc: Exception | None = None):
    def _gen(*, model, contents, config):
        if exc is not None:
            raise exc
        return resp

    return types.SimpleNamespace(models=types.SimpleNamespace(generate_content=_gen))


def _read_lines(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


# --- GeminiTracer standalone -----------------------------------------------


class TestGeminiTracer:
    def test_record_appends_jsonl(self, tmp_path: Path) -> None:
        tracer = GeminiTracer(tmp_path / "trace.jsonl")
        tracer.record(request={"a": 1}, response={"raw_text": "x"}, parsed={"kind": "numerical"},
                      latency_ms=12.34, error=None)
        tracer.record(request={"a": 2}, response=None, parsed=None,
                      latency_ms=5.0, error="boom")

        rows = _read_lines(tmp_path / "trace.jsonl")
        assert len(rows) == 2
        assert rows[0]["call_index"] == 1 and rows[1]["call_index"] == 2
        assert rows[0]["latency_ms"] == 12.3          # rounded to 1 dp
        assert rows[0]["parsed"] == {"kind": "numerical"}
        assert rows[1]["error"] == "boom"
        assert "ts" in rows[0]

    def test_save_images_disabled_returns_empty(self, tmp_path: Path) -> None:
        tracer = GeminiTracer(tmp_path / "trace.jsonl")  # save_images defaults off
        assert tracer.save_images([PNG, JPG]) == []
        assert not (tmp_path / "trace_images").exists()

    def test_save_images_enabled_writes_files(self, tmp_path: Path) -> None:
        tracer = GeminiTracer(tmp_path / "trace.jsonl", save_images=True)
        paths = tracer.save_images([PNG, JPG])
        assert len(paths) == 2
        assert any(p.endswith(".png") for p in paths)
        assert any(p.endswith(".jpg") for p in paths)
        for p in paths:
            assert (tmp_path / p).exists()


# --- GeminiEngine integration ----------------------------------------------


class TestEngineTracing:
    def test_no_tracer_writes_nothing_and_still_parses(self, tmp_path: Path) -> None:
        client = _client(_resp(json.dumps({"kind": "numerical", "value": 3})))
        engine = GeminiEngine(_config(), client=client)  # no tracer
        out = engine.infer_multimodal(system="s", user_text="u", images=[])
        assert isinstance(out, NumericalResponse) and out.value == 3

    def test_successful_call_is_traced_with_raw_and_usage(self, tmp_path: Path) -> None:
        usage = types.SimpleNamespace(
            prompt_token_count=100, candidates_token_count=20, total_token_count=120,
        )
        raw = json.dumps({"kind": "numerical", "value": 5, "rationale": "five"})
        client = _client(_resp(raw, usage=usage, finish_reason="STOP"))
        tracer = GeminiTracer(tmp_path / "trace.jsonl")
        engine = GeminiEngine(_config(), client=client, tracer=tracer)

        out = engine.infer_multimodal(system="SYS", user_text="scene graph here", images=[PNG])
        assert isinstance(out, NumericalResponse) and out.value == 5

        (rec,) = _read_lines(tmp_path / "trace.jsonl")
        # Request captured in full.
        assert rec["request"]["system"] == "SYS"
        assert rec["request"]["user_text"] == "scene graph here"
        assert rec["request"]["num_images"] == 1
        assert rec["request"]["image_bytes"] == [len(PNG)]
        assert rec["request"]["temperature"] == 0.2
        # Raw response + usage captured.
        assert rec["response"]["raw_text"] == raw
        assert rec["response"]["finish_reason"] == "STOP"
        assert rec["response"]["usage"]["total_tokens"] == 120
        # Parsed output + no error.
        assert rec["parsed"]["value"] == 5
        assert rec["error"] is None
        assert rec["latency_ms"] >= 0

    def test_failed_call_is_traced_then_reraised(self, tmp_path: Path) -> None:
        client = _client(exc=RuntimeError("429 RESOURCE_EXHAUSTED"))
        tracer = GeminiTracer(tmp_path / "trace.jsonl")
        engine = GeminiEngine(_config(), client=client, tracer=tracer)

        with pytest.raises(RuntimeError):
            engine.infer_multimodal(system="s", user_text="u", images=[])

        (rec,) = _read_lines(tmp_path / "trace.jsonl")
        assert "429" in rec["error"]
        assert rec["parsed"] is None
        assert rec["response"] is None  # call raised before any response

    def test_parse_failure_is_traced_with_raw_text(self, tmp_path: Path) -> None:
        # Gemini returned non-JSON — the raw text must survive in the trace
        # so we can see exactly what broke the parser.
        client = _client(_resp("not json at all"))
        tracer = GeminiTracer(tmp_path / "trace.jsonl")
        engine = GeminiEngine(_config(), client=client, tracer=tracer)

        with pytest.raises(Exception):
            engine.infer_multimodal(system="s", user_text="u", images=[])

        (rec,) = _read_lines(tmp_path / "trace.jsonl")
        assert rec["parsed"] is None
        assert rec["error"] is not None
        assert rec["response"]["raw_text"] == "not json at all"
