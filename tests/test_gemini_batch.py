"""Tests for the offline Gemini batch evaluation harness.

All cases inject a fake engine implementing ``GeminiEngineProtocol`` — no
API key, no network. The fake records the ``(system, user_text, images)``
it was handed so we can assert the offline prompt shape.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xiao_hei_vln.gemini import batch
from xiao_hei_vln.messages import (
    NumericalResponse,
    ObjectReferenceResponse,
    QuestionType,
    VLMOutput,
)
from xiao_hei_vln.messages.common import Vector3

# A tiny two-object scene in VLA-3D object_list text form:
#   id cx cy cz lx ly lz heading "label"
CHAIR = '0 1.00 2.00 0.50 0.60 0.60 1.00 0.00 "chair"'
TABLE = '1 1.50 2.50 0.40 1.20 0.80 0.70 0.00 "table"'
OBJECT_LIST = [CHAIR, TABLE]


class FakeEngine:
    """Records calls; returns a canned VLMOutput."""

    def __init__(self, output: VLMOutput) -> None:
        self.output = output
        self.calls: list[dict] = []

    def infer_multimodal(self, *, system: str, user_text: str, images: list[bytes]) -> VLMOutput:
        self.calls.append({"system": system, "user_text": user_text, "images": images})
        return self.output

    def warmup(self) -> None:  # pragma: no cover - unused here
        pass


class BoomEngine:
    """Raises on every call, to exercise the error path."""

    def infer_multimodal(self, *, system: str, user_text: str, images: list[bytes]) -> VLMOutput:
        raise RuntimeError("network down")

    def warmup(self) -> None:  # pragma: no cover
        pass


# --- scene reconstruction --------------------------------------------------


def test_build_scene_keeps_every_object_with_bbox() -> None:
    scene = build = batch.build_scene(OBJECT_LIST)
    objs = build.objects
    assert len(objs) == 2
    labels = {o.label for o in objs}
    assert labels == {"chair", "table"}
    for o in objs:
        assert o.bbox_min is not None and o.bbox_max is not None
    # chair centre 1,2,0.5 with size 0.6,0.6,1.0 → bbox spans ±half-size.
    chair = next(o for o in scene.objects if o.label == "chair")
    assert chair.bbox_min.x == pytest.approx(0.70)
    assert chair.bbox_max.z == pytest.approx(1.00)


def test_build_scene_disables_same_label_merge() -> None:
    # Two chairs 0.5 m apart would merge under the default 1.5 m radius;
    # the harness must keep them distinct.
    close_pair = [
        '0 0.00 0.00 0.50 0.60 0.60 1.00 0.00 "chair"',
        '1 0.50 0.00 0.50 0.60 0.60 1.00 0.00 "chair"',
    ]
    scene = batch.build_scene(close_pair)
    assert len(scene.objects) == 2


def test_build_scene_sets_scene_bounds() -> None:
    scene = batch.build_scene(OBJECT_LIST)
    assert scene.room.scene_bounds is not None
    mn, mx = scene.room.scene_bounds
    assert mn.x <= 0.70 and mx.x >= 2.10


def test_scene_to_text_is_compact_object_list() -> None:
    text = batch.scene_to_text(batch.build_scene(OBJECT_LIST))
    assert "```json" in text
    assert "chair" in text
    assert '"center"' in text and '"size"' in text
    # Compact: no pretty-print indentation, and none of the verbose fields.
    assert "\n  " not in text
    assert "bbox_min" not in text
    assert "observing_viewpoint_ids" not in text


def test_compact_objects_prunes_fields() -> None:
    objs = batch._compact_objects(batch.build_scene(OBJECT_LIST))
    assert len(objs) == 2
    keys = set(objs[0])
    assert keys <= {"id", "label", "center", "size", "color"}
    assert "center" in keys and "size" in keys
    assert len(objs[0]["center"]) == 3 and len(objs[0]["size"]) == 3


# --- prompt dispatch -------------------------------------------------------


def test_offline_system_prompt_dispatch() -> None:
    num = batch.offline_system_prompt(QuestionType.NUMERICAL)
    ref = batch.offline_system_prompt(QuestionType.OBJECT_REFERENCE)
    assert "NUMERICAL" in num
    assert "OBJECT-REFERENCE" in ref
    with pytest.raises(ValueError):
        batch.offline_system_prompt(QuestionType.INSTRUCTION_FOLLOWING)


# --- predict_entry ---------------------------------------------------------


def test_predict_entry_numerical() -> None:
    engine = FakeEngine(NumericalResponse(value=2, rationale="two"))
    entry = {"type": "numerical", "question": "How many chairs?", "object_list": OBJECT_LIST}
    result = batch.predict_entry(engine, entry)
    assert result is not None
    assert result.question == "How many chairs?"
    assert isinstance(result.output, NumericalResponse) and result.output.value == 2
    # Offline path sends the compact scene objects, no images.
    call = engine.calls[0]
    assert call["images"] == []
    assert "scene objects" in call["user_text"].lower()
    assert "chair" in call["user_text"]
    assert "NUMERICAL" in call["system"]


def test_predict_entry_object_reference() -> None:
    engine = FakeEngine(
        ObjectReferenceResponse(
            label="chair",
            object_id=1,
            center=Vector3(x=1.0, y=2.0, z=0.5),
            size=Vector3(x=0.6, y=0.6, z=1.0),
        )
    )
    entry = {
        "type": "object_reference",
        "question": "Find the chair near the table.",
        "object_list": OBJECT_LIST,
    }
    result = batch.predict_entry(engine, entry)
    assert result is not None
    assert isinstance(result.output, ObjectReferenceResponse)
    assert "OBJECT-REFERENCE" in engine.calls[0]["system"]
    # Debug payload captures the exact request + parsed prediction.
    assert result.debug["type"] == "object_reference"
    assert "objects" in result.debug["scene_graph"]
    assert "OBJECT-REFERENCE" in result.debug["system_prompt"]
    assert result.debug["prediction"]["kind"] == "object_reference"


def test_predict_entry_skips_instruction_following() -> None:
    engine = FakeEngine(NumericalResponse(value=0))
    entry = {"type": "instruction_following", "question": "Go to the sofa.", "object_list": []}
    assert batch.predict_entry(engine, entry) is None
    assert engine.calls == []


def test_predict_entry_skips_empty_question() -> None:
    engine = FakeEngine(NumericalResponse(value=0))
    assert batch.predict_entry(engine, {"type": "numerical", "question": "  "}) is None


# --- run -------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_run_writes_evaluator_ready_predictions(tmp_path: Path) -> None:
    gt = tmp_path / "gt.jsonl"
    out = tmp_path / "pred.jsonl"
    _write_jsonl(
        gt,
        [
            {"type": "numerical", "question": "How many chairs?", "object_list": OBJECT_LIST},
            {"type": "instruction_following", "question": "Go left.", "object_list": []},
            {"type": "numerical", "question": "How many tables?", "object_list": OBJECT_LIST},
        ],
    )
    engine = FakeEngine(NumericalResponse(value=1, rationale="one"))

    written = batch.run(gt, out, engine)

    assert written == 2  # instruction_following skipped
    lines = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["question"] == "How many chairs?"
    assert lines[0]["prediction"]["kind"] == "numerical"
    assert lines[0]["prediction"]["value"] == 1


def test_task1_cap_counts_scoreable_numerical_only(tmp_path: Path) -> None:
    gt = tmp_path / "gt.jsonl"
    out = tmp_path / "pred.jsonl"
    # Interleave non-scoreable entries — they must NOT consume the budget.
    rows: list[dict] = []
    for i in range(5):
        rows.append({"type": "instruction_following", "question": f"go{i}", "object_list": []})
        rows.append({"type": "numerical", "question": f"n{i}", "object_list": OBJECT_LIST})
    _write_jsonl(gt, rows)
    engine = FakeEngine(NumericalResponse(value=0))

    written = batch.run(gt, out, engine, task1=3)
    # Exactly 3 numerical examples, regardless of the interleaved skips.
    assert written == 3


def test_task2_flag_restricts_to_object_reference(tmp_path: Path) -> None:
    # Mixed file: only --task2 given → numerical entries are excluded.
    gt = tmp_path / "gt.jsonl"
    out = tmp_path / "pred.jsonl"
    rows: list[dict] = []
    for i in range(4):
        rows.append({"type": "numerical", "question": f"n{i}", "object_list": OBJECT_LIST})
        rows.append({"type": "object_reference", "question": f"find{i}", "object_list": OBJECT_LIST})
    _write_jsonl(gt, rows)
    engine = FakeEngine(
        ObjectReferenceResponse(
            label="chair", object_id=0,
            center=Vector3(x=1, y=2, z=0.5), size=Vector3(x=0.6, y=0.6, z=1.0),
        )
    )
    written = batch.run(gt, out, engine, task2=2)
    assert written == 2  # 2 object_reference, zero numerical
    kinds = {json.loads(ln)["prediction"]["kind"] for ln in out.read_text().splitlines()}
    assert kinds == {"object_reference"}


def test_no_caps_processes_all_scoreable(tmp_path: Path) -> None:
    gt = tmp_path / "gt.jsonl"
    out = tmp_path / "pred.jsonl"
    _write_jsonl(
        gt,
        [
            {"type": "numerical", "question": "n0", "object_list": OBJECT_LIST},
            {"type": "object_reference", "question": "r0", "object_list": OBJECT_LIST},
            {"type": "instruction_following", "question": "go", "object_list": []},
        ],
    )
    engine = FakeEngine(NumericalResponse(value=1))
    written = batch.run(gt, out, engine)  # no caps
    assert written == 2  # both scoreable types, instruction_following skipped


def test_run_survives_engine_errors(tmp_path: Path) -> None:
    gt = tmp_path / "gt.jsonl"
    out = tmp_path / "pred.jsonl"
    _write_jsonl(
        gt,
        [{"type": "numerical", "question": "q0", "object_list": OBJECT_LIST}],
    )
    written = batch.run(gt, out, BoomEngine())
    assert written == 0
    assert out.read_text() == ""


# --- rate limiting + retry -------------------------------------------------


def test_parse_retry_delay() -> None:
    assert batch._parse_retry_delay("plain error") is None
    assert batch._parse_retry_delay("429 ... 'retryDelay': '9s'") == 9.0
    assert batch._parse_retry_delay("Please retry in 3.5s") == 3.5


def test_retry_wait_classifies_errors() -> None:
    # Non-retryable → None.
    assert batch._retry_wait(ValueError("bad json"), 0, min_backoff=2.0) is None
    # 429 rate-limit → retryable; server delay floored by exponential backoff.
    w = batch._retry_wait(RuntimeError("429 RESOURCE_EXHAUSTED 'retryDelay': '1s'"), 0, min_backoff=2.0)
    assert w == 2.0  # max(1, 2*2**0)
    # 503 transient → retryable even without a retryDelay.
    assert batch._retry_wait(RuntimeError("503 UNAVAILABLE high demand"), 0, min_backoff=2.0) == 2.0
    # Input-token-per-minute quota → at least a 60s window despite short delay.
    w = batch._retry_wait(
        RuntimeError("429 free_tier_input_token_count ... 'retryDelay': '0s'"), 0, min_backoff=2.0,
    )
    assert w == 60.0
    # Exponential backoff grows with attempt.
    assert batch._retry_wait(RuntimeError("503 UNAVAILABLE"), 3, min_backoff=2.0) == 16.0
    # Per-DAY quota won't clear in-window → not retryable (fail fast).
    assert batch._retry_wait(
        RuntimeError("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier"),
        0, min_backoff=2.0,
    ) is None


def test_throttled_engine_retries_on_429(monkeypatch) -> None:
    monkeypatch.setattr(batch.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    class Flaky:
        def infer_multimodal(self, *, system, user_text, images):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("429 RESOURCE_EXHAUSTED 'retryDelay': '1s'")
            return NumericalResponse(value=7)

        def warmup(self):
            pass

    eng = batch.ThrottledEngine(Flaky(), rpm=0, max_retries=5)
    out = eng.infer_multimodal(system="s", user_text="u", images=[])
    assert isinstance(out, NumericalResponse) and out.value == 7
    assert calls["n"] == 3  # failed twice, succeeded on the third


def test_throttled_engine_propagates_non_rate_limit(monkeypatch) -> None:
    monkeypatch.setattr(batch.time, "sleep", lambda _s: None)

    class Boom:
        def infer_multimodal(self, **_kwargs):
            raise ValueError("malformed json")

        def warmup(self):
            pass

    eng = batch.ThrottledEngine(Boom(), rpm=0, max_retries=5)
    with pytest.raises(ValueError):
        eng.infer_multimodal(system="s", user_text="u", images=[])


def test_throttled_engine_gives_up_after_max_retries(monkeypatch) -> None:
    monkeypatch.setattr(batch.time, "sleep", lambda _s: None)

    class Always429:
        def __init__(self):
            self.n = 0

        def infer_multimodal(self, **_kwargs):
            self.n += 1
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

        def warmup(self):
            pass

    inner = Always429()
    eng = batch.ThrottledEngine(inner, rpm=0, max_retries=2)
    with pytest.raises(RuntimeError):
        eng.infer_multimodal(system="s", user_text="u", images=[])
    assert inner.n == 3  # initial try + 2 retries


# --- debug dir -------------------------------------------------------------


def test_run_debug_dir_writes_full_request(tmp_path: Path) -> None:
    gt = tmp_path / "gt.jsonl"
    out = tmp_path / "pred.jsonl"
    dbg = tmp_path / "debug"
    _write_jsonl(
        gt, [{"type": "numerical", "question": "How many chairs?", "object_list": OBJECT_LIST}]
    )
    engine = FakeEngine(NumericalResponse(value=1, rationale="one"))

    batch.run(gt, out, engine, debug_dir=dbg)

    files = list(dbg.glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["question"] == "How many chairs?"
    assert "objects" in payload["scene_graph"]
    assert payload["system_prompt"]
    assert payload["user_text"]
    assert payload["prediction"]["kind"] == "numerical"
    assert payload["error"] is None


def test_throttled_engine_retries_on_503(monkeypatch) -> None:
    monkeypatch.setattr(batch.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    class Flaky503:
        def infer_multimodal(self, *, system, user_text, images):
            calls["n"] += 1
            if calls["n"] < 2:
                raise RuntimeError("503 UNAVAILABLE: model overloaded")
            return NumericalResponse(value=9)

        def warmup(self):
            pass

    eng = batch.ThrottledEngine(Flaky503(), rpm=0, max_retries=5)
    out = eng.infer_multimodal(system="s", user_text="u", images=[])
    assert isinstance(out, NumericalResponse) and out.value == 9
    assert calls["n"] == 2  # 503 is retried, not treated as permanent


def test_run_debug_dir_captures_failures(tmp_path: Path) -> None:
    gt = tmp_path / "gt.jsonl"
    out = tmp_path / "pred.jsonl"
    dbg = tmp_path / "debug"
    _write_jsonl(
        gt, [{"type": "numerical", "question": "How many chairs?", "object_list": OBJECT_LIST}]
    )

    written = batch.run(gt, out, BoomEngine(), debug_dir=dbg)

    assert written == 0
    assert out.read_text() == ""  # no prediction line for a failed call
    # ...but the failure IS captured for inspection, with the error recorded.
    (fp,) = list(dbg.glob("*.json"))
    payload = json.loads(fp.read_text())
    assert payload["prediction"] is None
    assert payload["error"] is not None
    assert payload["user_text"]  # the exact prompt that failed is preserved
