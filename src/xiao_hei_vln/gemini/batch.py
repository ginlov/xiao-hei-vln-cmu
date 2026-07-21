"""Offline batch evaluation harness for the Gemini responder.

Produces a ``predictions.jsonl`` for Task 1 (numerical) and Task 2
(object_reference) *without* the ROS simulator, so we can score Gemini
with the offline evaluator (``xiao_hei_vln.eval_pipeline``).

The live :class:`~xiao_hei_vln.scene_gemini.SceneGeminiResponder` consumes
raw sensor snapshots (camera / lidar / pose) and renders occupancy maps.
There is no offline dataset of such snapshots, so this harness instead
reconstructs the **same scene-graph representation** Gemini sees at test
time from the ground-truth ``object_list``:

    object_list (VLA-3D text) → SceneRepresentation → to_dict() JSON → Gemini

The JSON is byte-for-byte the shape
:meth:`xiao_hei_vln.scene.SceneRepresentation.to_dict` emits in the live
path (Room → Viewpoints → Objects, each with a 3D bbox and spatial
relations), so what Gemini reasons over here matches production — minus
the rendered panorama / occupancy images, which don't exist offline.

Because we send a scene graph instead of images, the live
image-centric system prompts in :mod:`xiao_hei_vln.gemini.prompts` don't
apply; this module carries scene-graph-driven prompts of its own.

Scoring note: object_reference is scored by 3D bbox IoU, so this
measures Gemini's ability to *pick the referred object and return its
bbox* from the graph; numerical measures spatial-relation filtering +
counting. Both isolate Gemini's reasoning over the scene representation,
with perception assumed perfect.

Usage::

    export XIAO_HEI_GEMINI_API_KEY=<key>
    uv run python -m xiao_hei_vln.gemini.batch \\
        --gt   /path/to/vla3d_ref.jsonl \\
        --out  pred_ref.jsonl \\
        --limit 50
    uv run python -m xiao_hei_vln.eval_pipeline --gt <GT> --pred pred_ref.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from xiao_hei_vln.eval_sampler.object_list import parse_object_list
from xiao_hei_vln.gemini.engine import GeminiEngineProtocol
from xiao_hei_vln.messages import QuestionType, VLMOutput
from xiao_hei_vln.messages.common import Vector3
from xiao_hei_vln.scene import ObjectObservation, SceneRepresentation

log = logging.getLogger(__name__)

# Only these two are scored by our offline evaluator; instruction_following
# needs the official evaluator and is skipped.
SCOREABLE_TYPES = frozenset({"numerical", "object_reference"})

# Free-tier gemini-2.5-flash is capped at 5 requests/minute; default the
# throttle to match so a full batch doesn't 429 itself.
DEFAULT_RPM = 5
DEFAULT_MAX_RETRIES = 5


# --- rate limiting + retry -------------------------------------------------


class ThrottledEngine:
    """Wrap a Gemini engine with request throttling + backoff-retry.

    The free API tier limits both requests/minute and input-tokens/minute,
    and the service occasionally 503s under load. We (a) space calls at
    least ``60/rpm`` seconds apart, and (b) on a *retryable* error
    (429 rate-limit or 5xx transient) sleep — honouring the server's
    ``retryDelay`` but flooring with exponential backoff, and waiting a
    full minute window for input-token-quota errors — then retry.
    Permanent errors propagate immediately. Implements
    :class:`GeminiEngineProtocol` so it drops into ``predict_entry``.
    """

    def __init__(
        self,
        inner: GeminiEngineProtocol,
        *,
        rpm: int = DEFAULT_RPM,
        max_retries: int = DEFAULT_MAX_RETRIES,
        min_backoff_s: float = 2.0,
    ) -> None:
        self._inner = inner
        self._min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self._max_retries = max_retries
        self._min_backoff_s = min_backoff_s
        self._last_call = 0.0

    def warmup(self) -> None:
        self._inner.warmup()

    def infer_multimodal(
        self, *, system: str, user_text: str, images: list[bytes]
    ) -> VLMOutput:
        for attempt in range(self._max_retries + 1):
            self._throttle()
            try:
                return self._inner.infer_multimodal(
                    system=system, user_text=user_text, images=images
                )
            except Exception as exc:  # noqa: BLE001 — inspect for retryability
                wait = _retry_wait(exc, attempt, min_backoff=self._min_backoff_s)
                if wait is None or attempt >= self._max_retries:
                    raise
                log.warning(
                    "retryable error (attempt %d/%d); waiting %.1fs: %s",
                    attempt + 1,
                    self._max_retries,
                    wait,
                    _short_error(exc),
                )
                time.sleep(wait)
        raise RuntimeError("unreachable: retry loop exhausted")  # pragma: no cover

    def _throttle(self) -> None:
        if self._min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()


_RATE_MARKERS = ("429", "RESOURCE_EXHAUSTED")
_TRANSIENT_MARKERS = ("503", "UNAVAILABLE", "500", "INTERNAL", "overloaded", "high demand")
_MAX_BACKOFF_S = 120.0


def _parse_retry_delay(text: str) -> float | None:
    """Extract a server-suggested retry delay (seconds) from an error string."""
    m = re.search(r"[Rr]etry[Dd]elay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", text)
    if m:
        return float(m.group(1))
    m = re.search(r"retry in (\d+(?:\.\d+)?)s\b", text)
    if m:
        return float(m.group(1))
    return None


def _retry_wait(exc: Exception, attempt: int, *, min_backoff: float) -> float | None:
    """Seconds to wait before retrying, or ``None`` if not retryable.

    Retries 429 rate-limits and 5xx transient errors. The wait is the
    server ``retryDelay`` floored by exponential backoff
    (``min_backoff * 2**attempt``, capped). Input-token-per-minute quota
    errors get at least a 60 s window — their ``retryDelay`` is
    misleadingly short (often 0 s) yet the window only clears each minute,
    and each retry re-sends the whole prompt.

    A **per-day** quota (``...PerDay...``) is *not* retried: it won't clear
    within any retry window (it resets at midnight Pacific), so retrying
    just wastes minutes per entry. Fail fast instead.
    """
    text = str(exc)
    if "PerDay" in text or "per_day" in text:
        return None
    rate = any(m in text for m in _RATE_MARKERS)
    transient = any(m in text for m in _TRANSIENT_MARKERS)
    if not (rate or transient):
        return None
    server = _parse_retry_delay(text) or 0.0
    wait = max(server, min_backoff * (2 ** attempt))
    if rate and "input_token" in text:
        wait = max(wait, 60.0)
    return min(wait, _MAX_BACKOFF_S)


def _short_error(exc: Exception) -> str:
    """One-line, truncated error text (the 429 body is huge)."""
    return str(exc).replace("\n", " ")[:160]


@dataclass(frozen=True)
class EntryPrediction:
    """A scoreable entry's prediction plus the full request for debugging.

    ``output`` is ``None`` when the Gemini call or JSON parse failed; the
    error is then recorded in ``debug['error']``.
    """

    question: str
    output: VLMOutput | None
    debug: dict


# --- offline (scene-graph-driven) system prompts ---------------------------
#
# The live prompts in gemini.prompts are written for the panorama +
# occupancy-map path. Offline we hand Gemini the full scene graph as JSON
# and no images, so we tell it explicitly to reason over that graph.

OFFLINE_SYSTEM_NUMERICAL = """\
You are answering a NUMERICAL question ("How many <target>...?") for the
CMU Vision-Language-Navigation Challenge.

You are given the COMPLETE list of objects in the room as a JSON array:
each object has an ``id``, a ``label``, a 3D ``center`` [x, y, z] and a
``size`` [x, y, z] (axis-aligned extent, metres), and sometimes a
``color``. Treat this list as the full and authoritative record of the
scene — there is nothing unobserved. Spatial relations (near, closest,
between, above/below) are computed from the centers and sizes.

Count the objects that satisfy the question, reasoning over the labels,
centers, sizes, and colours. Return EXACTLY one JSON object:

  {"kind": "numerical", "value": <non-negative int>, "rationale": "<=200 chars>"}

Rules:
  - ``value`` MUST be a non-negative integer; ``rationale`` MUST be present.
  - Base the count on the graph, not on prior assumptions about the room.
  - Do not include any text outside the JSON object.
"""

OFFLINE_SYSTEM_OBJECT_REFERENCE = """\
You are answering an OBJECT-REFERENCE question ("Find the <referring
expression>") for the CMU Vision-Language-Navigation Challenge.

You are given the COMPLETE list of objects in the room as a JSON array:
each object has an ``id``, a ``label``, a 3D ``center`` [x, y, z] and a
``size`` [x, y, z] (axis-aligned extent, metres), and sometimes a
``color``. Treat this list as the full and authoritative record of the
scene. Spatial relations (near, closest, between, above/below) are
computed from the centers and sizes.

Identify the single object the expression refers to, using the labels,
centers, sizes, and colours. Then return that object's bounding box.
EXACTLY one JSON object:

  {"kind": "object_reference",
   "label": "<the object's label from the graph>",
   "object_id": <that object's object_id from the graph>,
   "center": {"x": <float>, "y": <float>, "z": <float>},
   "size":   {"x": <float>, "y": <float>, "z": <float>},
   "heading": 0.0,
   "rationale": "<=200 chars>"}

Rules:
  - ``center`` MUST equal the chosen object's ``center`` and ``size`` MUST
    equal its ``size`` from the list — copy them verbatim (as {x, y, z});
    do NOT invent dimensions. Set ``object_id`` to the chosen object's
    ``id``.
  - Pick exactly one object. If the expression is ambiguous, choose the
    best match and explain briefly in ``rationale``.
  - Do not include any text outside the JSON object.
"""


def offline_system_prompt(qtype: QuestionType) -> str:
    """Return the scene-graph-driven system prompt for a scoreable type."""
    if qtype is QuestionType.NUMERICAL:
        return OFFLINE_SYSTEM_NUMERICAL
    if qtype is QuestionType.OBJECT_REFERENCE:
        return OFFLINE_SYSTEM_OBJECT_REFERENCE
    raise ValueError(f"Unsupported question type for offline Gemini eval: {qtype!r}")


# --- scene-graph reconstruction --------------------------------------------


def build_scene(object_list: list[str], *, near_threshold: float = 2.0) -> SceneRepresentation:
    """Rebuild a :class:`SceneRepresentation` from a VLA-3D ``object_list``.

    Each object becomes one :class:`ObjectObservation` with an
    axis-aligned bbox derived from its centre + size. Same-label merging
    is disabled (``merge_radius=0``) so every ground-truth object is kept
    distinct; ``near`` spatial edges are then derived so the graph Gemini
    sees carries relational context.
    """
    scene = SceneRepresentation(merge_radius=0.0)
    entries = parse_object_list(object_list)
    for oid in sorted(entries):
        e = entries[oid]
        bmin, bmax = _aabb(e.center, e.size)
        scene.add_object(
            ObjectObservation(
                label=e.label,
                position=e.center,
                confidence=1.0,
                bbox_min=bmin,
                bbox_max=bmax,
                color_name=e.color,
            )
        )
    scene.derive_near_relations(threshold=near_threshold)
    _set_scene_bounds(scene)
    return scene


def _compact_objects(scene: SceneRepresentation) -> list[dict]:
    """Token-lean per-object view: id, label, center, size, colour.

    The full ``SceneRepresentation.to_dict`` is ~20x larger — it repeats
    each bbox as min+max, plus confidence, viewpoint ids, tick ids, and
    pre-derived ``near`` edges. For a 70-object scene that is ~37k input
    tokens per call, which overruns the free-tier per-minute input-token
    quota. We drop everything Gemini can infer from the coordinates.
    """
    items: list[dict] = []
    for o in scene.objects:
        item: dict = {
            "id": o.object_id,
            "label": o.label,
            "center": [round(o.position.x, 2), round(o.position.y, 2), round(o.position.z, 2)],
        }
        if o.bbox_min is not None and o.bbox_max is not None:
            item["size"] = [
                round(o.bbox_max.x - o.bbox_min.x, 2),
                round(o.bbox_max.y - o.bbox_min.y, 2),
                round(o.bbox_max.z - o.bbox_min.z, 2),
            ]
        # `color_name` only exists on the color-enabled scene rep (live
        # perception path); the offline GT never carries colour. getattr
        # keeps this working on both scene-representation variants.
        color = getattr(o, "color_name", None)
        if color:
            item["color"] = color
        items.append(item)
    return items


def scene_to_text(scene: SceneRepresentation) -> str:
    """Serialise a compact object list for the prompt (token-lean).

    Compact JSON (no indentation, tight separators); each object carries
    only ``id``, ``label``, ``center`` [x,y,z], ``size`` [x,y,z], and an
    optional ``color``.
    """
    objects = _compact_objects(scene)
    return (
        "Scene objects (JSON array; each has id, label, center [x,y,z], "
        "size [x,y,z] in metres, optional color):\n"
        f"```json\n{json.dumps(objects, separators=(',', ':'))}\n```"
    )


def build_user_message(question: str, qtype: QuestionType, scene_text: str) -> str:
    """Compose the user turn: question + scene graph + response cue."""
    return "\n\n".join(
        [
            f"Question (type={qtype.value}): {question}",
            scene_text,
            "Respond now with the JSON object — no prose around it.",
        ]
    )


# --- per-entry prediction --------------------------------------------------


def predict_entry(
    engine: GeminiEngineProtocol,
    entry: dict,
    *,
    near_threshold: float = 2.0,
) -> EntryPrediction | None:
    """Turn one GT entry into an :class:`EntryPrediction`.

    Returns ``None`` for entries that are not scoreable (wrong type,
    empty question). A failed Gemini call / JSON parse is captured, not
    raised: the returned ``EntryPrediction`` has ``output=None`` and
    ``debug['error']`` set, so the failing entry is still inspectable via
    ``--debug-dir``. The ``debug`` dict always captures the exact scene
    graph + prompts handed to Gemini.
    """
    qtype_str = entry.get("type", "")
    if qtype_str not in SCOREABLE_TYPES:
        return None
    question = (entry.get("question") or "").strip()
    if not question:
        return None

    qtype = QuestionType(qtype_str)
    # Prefer our real detections when the record carries them
    # (detected_object_list, injected by xiao_hei_vln.detected_dataset) so
    # the eval measures perception + LLM; fall back to the GT object_list.
    # The ground-truth converter still reads object_list, so scoring stays
    # against the authoritative GT.
    detected = entry.get("detected_object_list")
    raw_list = detected if isinstance(detected, list) else entry.get("object_list")
    object_list = raw_list if isinstance(raw_list, list) else []

    scene = build_scene(object_list, near_threshold=near_threshold)
    scene_text = scene_to_text(scene)
    system = offline_system_prompt(qtype)
    user_text = build_user_message(question, qtype, scene_text)
    debug: dict = {
        "question": question,
        "type": qtype.value,
        "scene_graph": scene.to_dict(),
        "system_prompt": system,
        "user_text": user_text,
        "prediction": None,
        "error": None,
    }

    try:
        output = engine.infer_multimodal(system=system, user_text=user_text, images=[])
    except Exception as exc:  # noqa: BLE001 — record the failure, don't crash the run
        debug["error"] = repr(exc)
        return EntryPrediction(question=question, output=None, debug=debug)

    debug["prediction"] = output.model_dump()
    return EntryPrediction(question=question, output=output, debug=debug)


# --- driver ----------------------------------------------------------------


# GT ``type`` -> Task number, for the per-task caps.
_TYPE_FOR_TASK = {1: "numerical", 2: "object_reference"}


def run(
    gt_path: Path,
    out_path: Path,
    engine: GeminiEngineProtocol,
    *,
    task1: int | None = None,
    task2: int | None = None,
    near_threshold: float = 2.0,
    debug_dir: Path | None = None,
) -> int:
    """Generate predictions and write an evaluator-ready JSONL.

    ``task1`` / ``task2`` cap how many *scoreable* examples of each type
    are attempted — Task 1 = ``numerical``, Task 2 = ``object_reference``
    — counting only real scoreable entries (non-scoreable / duplicate GT
    lines don't consume the budget). Specifying **either** flag restricts
    the run to those task type(s): e.g. ``task2=50`` evaluates 50
    object-reference examples and no numerical ones. With **neither** set,
    every scoreable entry is processed. When ``debug_dir`` is set, dumps
    one JSON per attempted prediction (including failures).

    Returns the number of predictions written.
    """
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    caps: dict[str, int] = {}
    if task1 is not None:
        caps[_TYPE_FOR_TASK[1]] = task1
    if task2 is not None:
        caps[_TYPE_FOR_TASK[2]] = task2
    # A type is collected if it was explicitly requested; with no caps at
    # all, both scoreable types are collected.
    requested = set(caps) if caps else set(SCOREABLE_TYPES)
    attempted: dict[str, int] = {"numerical": 0, "object_reference": 0}

    written = skipped = errors = 0
    with out_path.open("w") as out_f:
        for lineno, line in enumerate(_iter_jsonl(gt_path), 1):
            # Early stop once every requested per-task cap is filled.
            if caps and all(attempted[t] >= c for t, c in caps.items()):
                break
            qtype = line.get("type", "")
            # Not a requested type, or its budget is spent — skip, no Gemini.
            if qtype not in requested:
                skipped += 1
                continue
            if qtype in caps and attempted[qtype] >= caps[qtype]:
                continue
            try:
                result = predict_entry(engine, line, near_threshold=near_threshold)
            except Exception as exc:  # unexpected (e.g. scene build) — keep going
                log.warning("entry %d: %s", lineno, exc)
                errors += 1
                continue
            if result is None:
                skipped += 1
                continue
            attempted[qtype] = attempted.get(qtype, 0) + 1
            # Dump debug for *every* attempted entry — including failures,
            # which are exactly the ones worth inspecting.
            if debug_dir is not None:
                _write_debug(debug_dir, lineno, result)
            if result.output is None:
                errors += 1
                log.warning("entry %d: Gemini call failed: %s", lineno, result.debug["error"])
                continue
            record = {"question": result.question, "prediction": result.output.model_dump()}
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if written % 10 == 0:
                print(f"  ... {written} predictions", file=sys.stderr)

    print(
        f"Wrote {written} predictions to {out_path}  "
        f"(task1/numerical={attempted['numerical']}, "
        f"task2/object_reference={attempted['object_reference']}; "
        f"skipped {skipped} non-scoreable, {errors} errors)",
        file=sys.stderr,
    )
    return written


def _iter_jsonl(path: Path):
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[warn] {path}:{lineno}: {exc}", file=sys.stderr)
                continue


# --- helpers ---------------------------------------------------------------


def _write_debug(debug_dir: Path, lineno: int, result: EntryPrediction) -> None:
    """Dump one prediction's full request/response to ``debug_dir``."""
    slug = re.sub(r"[^a-z0-9]+", "_", result.question.lower())[:50].strip("_")
    path = debug_dir / f"{lineno:05d}_{slug or 'entry'}.json"
    path.write_text(json.dumps(result.debug, ensure_ascii=False, indent=2))


def _aabb(center: Vector3, size: Vector3) -> tuple[Vector3, Vector3]:
    hx, hy, hz = size.x / 2.0, size.y / 2.0, size.z / 2.0
    return (
        Vector3(x=center.x - hx, y=center.y - hy, z=center.z - hz),
        Vector3(x=center.x + hx, y=center.y + hy, z=center.z + hz),
    )


def _set_scene_bounds(scene: SceneRepresentation) -> None:
    objs = scene.objects
    mins = [o.bbox_min for o in objs if o.bbox_min is not None]
    maxs = [o.bbox_max for o in objs if o.bbox_max is not None]
    if not mins or not maxs:
        return
    scene.room.scene_bounds = (
        Vector3(
            x=min(v.x for v in mins),
            y=min(v.y for v in mins),
            z=min(v.z for v in mins),
        ),
        Vector3(
            x=max(v.x for v in maxs),
            y=max(v.y for v in maxs),
            z=max(v.z for v in maxs),
        ),
    )


# --- CLI -------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Offline Gemini predictions for Task 1 (numerical) + Task 2 "
            "(object_reference) from a VLA-3D GT JSONL. Feeds Gemini the "
            "reconstructed scene graph; writes an evaluator-ready predictions JSONL."
        )
    )
    parser.add_argument("--gt", type=Path, required=True, help="VLA-3D GT JSONL file.")
    parser.add_argument("--out", type=Path, required=True, help="Output predictions JSONL path.")
    parser.add_argument(
        "--task1",
        type=int,
        default=None,
        help="Max Task 1 (numerical) examples to evaluate. Default: all in the file.",
    )
    parser.add_argument(
        "--task2",
        type=int,
        default=None,
        help="Max Task 2 (object_reference) examples to evaluate. Default: all in the file.",
    )
    parser.add_argument(
        "--near-threshold",
        type=float,
        default=2.0,
        help="XY distance (m) for deriving `near` spatial edges in the scene graph.",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=DEFAULT_RPM,
        help=(
            "Max requests/minute to Gemini (throttle). Default 5 matches the "
            "free tier; raise it on a paid plan. 0 disables throttling."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Retries on a 429 rate-limit error (honours the server retryDelay).",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help=(
            "If set, dump one JSON per prediction (scene graph + system prompt "
            "+ user text + parsed output) here for inspection."
        ),
    )
    parser.add_argument(
        "--trace-file",
        type=Path,
        default=None,
        help=(
            "If set, append a full-fidelity JSONL trace of every Gemini call "
            "(request + RAW response + token usage + latency + errors) here. "
            "The most complete debug log — captures raw output before parsing."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Build the real engine lazily so --help works without the gemini extra
    # or an API key.
    from xiao_hei_vln.gemini.config import GeminiConfig
    from xiao_hei_vln.gemini.engine import GeminiEngine

    config = GeminiConfig.from_env()
    tracer = None
    if args.trace_file is not None:
        from xiao_hei_vln.gemini.trace import GeminiTracer

        tracer = GeminiTracer(args.trace_file)
        log.info("Tracing every Gemini call to %s", args.trace_file)
    engine: GeminiEngineProtocol = ThrottledEngine(
        GeminiEngine(config, tracer=tracer), rpm=args.rpm, max_retries=args.max_retries,
    )
    log.info(
        "Using Gemini model %s (throttle=%d rpm, max_retries=%d)",
        config.model, args.rpm, args.max_retries,
    )

    run(
        gt_path=args.gt,
        out_path=args.out,
        engine=engine,
        task1=args.task1,
        task2=args.task2,
        near_threshold=args.near_threshold,
        debug_dir=args.debug_dir,
    )


if __name__ == "__main__":
    main()
