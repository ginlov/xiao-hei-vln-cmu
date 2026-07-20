"""Build an LLM-eval dataset from *our* perception instead of ground truth.

The offline Gemini eval (:mod:`xiao_hei_vln.gemini.batch`) reasons over the
scene graph it rebuilds from each Q&A record's ground-truth ``object_list``
(``id cx cy cz lx ly lz heading "label"``). This module lets us swap that
GT for the objects our robot actually detected, so the same eval measures
the *full* pipeline (real perception + LLM) rather than "perception
assumed perfect".

Per scanned scene we:

1. convert our scene graph (``SceneRepresentation.to_dict()``, or the
   offline ``scene_objects/v1`` export) into GT-style ``object_list``
   strings — :func:`scene_graph_to_object_list`;
2. attach them to every Q&A record of that scene as a new
   ``detected_object_list`` field, **keeping** the GT ``object_list`` so the
   reference-answer truth box can still be looked up —
   :func:`splice_detected`;
3. emit a companion question+truth file — :func:`truth_records`.

The Gemini eval prefers ``detected_object_list`` when present (one-line
change in :mod:`xiao_hei_vln.gemini.batch`); the ground-truth converter is
untouched, so scoring still uses the authoritative GT.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from xiao_hei_vln.eval_sampler.object_list import parse_object_list

_DEFAULT_VLA3D_DIR = "dataset_generator/vla-3d/Unity"


def _object_center_size(obj: dict) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Extract (center, size) from a scene-graph object.

    Tolerates both formats we produce:
    - online scene graph: ``position`` + ``bbox_min``/``bbox_max`` (may be null),
    - offline ``scene_objects/v1``: ``center_3d`` + ``bbox_aabb.size``.
    A point-only object (no box) gets size ``(0, 0, 0)``.
    """
    center = obj.get("position") or obj.get("center_3d")
    if center is None:
        raise ValueError(f"object has no position/center_3d: {obj.get('label')!r}")
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])

    bmin, bmax = obj.get("bbox_min"), obj.get("bbox_max")
    aabb = obj.get("bbox_aabb")
    if bmin is not None and bmax is not None:
        sx = abs(float(bmax[0]) - float(bmin[0]))
        sy = abs(float(bmax[1]) - float(bmin[1]))
        sz = abs(float(bmax[2]) - float(bmin[2]))
    elif isinstance(aabb, dict) and aabb.get("size"):
        s = aabb["size"]
        sx, sy, sz = abs(float(s[0])), abs(float(s[1])), abs(float(s[2]))
    else:
        sx = sy = sz = 0.0
    return (cx, cy, cz), (sx, sy, sz)


def scene_graph_to_object_list(
    scene: dict | Sequence[dict], *, precision: int = 4
) -> list[str]:
    """Convert a scene graph into GT-style ``object_list`` strings.

    ``scene`` may be a ``SceneRepresentation.to_dict()`` dict (uses its
    ``objects``), the ``scene_objects/v1`` export dict, or a raw object list.
    Ids are freshly assigned ``0..N-1`` (a local handle for the LLM to point
    at — it does **not** align with GT ids, and does not need to, since
    reference answers are scored by 3D box overlap, not id). Heading is 0
    (we don't estimate orientation).
    """
    objects = scene.get("objects", []) if isinstance(scene, dict) else list(scene)

    lines: list[str] = []
    for i, obj in enumerate(objects):
        (cx, cy, cz), (sx, sy, sz) = _object_center_size(obj)
        label = str(obj.get("label", "unknown")).replace('"', "'")
        # Carry the detection's natural-language colour (already computed live
        # and stored on the scene graph) as a 2nd quoted token, so the LLM
        # eval sees it. Absent/empty → no colour token.
        color = obj.get("color_name")
        color_str = f' "{str(color).replace(chr(34), chr(39))}"' if color else ""
        lines.append(
            f"{i} "
            f"{cx:.{precision}f} {cy:.{precision}f} {cz:.{precision}f} "
            f"{sx:.{precision}f} {sy:.{precision}f} {sz:.{precision}f} "
            f'0.0 "{label}"' + color_str
        )
    return lines


def load_qa_for_scene(qa_paths: Iterable[str | Path], scene: str) -> list[dict]:
    """Load all Q&A records whose ``scene`` matches, across the given JSONL files."""
    records: list[dict] = []
    for path in qa_paths:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("scene") == scene:
                records.append(rec)
    return records


def splice_detected(records: Iterable[dict], detected_object_list: Sequence[str]) -> list[dict]:
    """Attach ``detected_object_list`` to each record, keeping the GT ``object_list``."""
    det = list(detected_object_list)
    return [{**rec, "detected_object_list": det} for rec in records]


def truth_records(records: Iterable[dict]) -> list[dict]:
    """A lean question+truth view. For ``object_reference`` the GT target box
    is resolved from the record's ``object_list`` so the file is self-contained."""
    out: list[dict] = []
    for rec in records:
        item: dict = {
            "scene": rec.get("scene"),
            "type": rec.get("type"),
            "question": rec.get("question"),
            "answer": rec.get("answer"),
        }
        if rec.get("type") == "object_reference":
            gt = parse_object_list(rec.get("object_list") or [])
            ans = rec.get("answer") or {}
            tid = ans.get("object_id")
            entry = gt.get(tid) if isinstance(tid, int) else None
            if entry is not None:
                item["target_center"] = [entry.center.x, entry.center.y, entry.center.z]
                item["target_size"] = [entry.size.x, entry.size.y, entry.size.z]
        out.append(item)
    return out


def extract_scene_graph_from_session(session_dir: str | Path) -> dict:
    """Pull the fullest scene graph (last tick of the last question) from a
    VLM-logger session directory."""
    session_dir = Path(session_dir)
    tick_files = sorted(session_dir.glob("q_*/ticks.jsonl"))
    if not tick_files:
        raise FileNotFoundError(f"no q_*/ticks.jsonl under {session_dir}")
    lines = [ln for ln in tick_files[-1].read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"empty ticks.jsonl in {tick_files[-1]}")
    record = json.loads(lines[-1])
    scene = record.get("scene")
    if scene is None:
        raise ValueError("last tick has no 'scene' — was OBJECT_MAP / logging on?")
    return scene


def load_gt_colors(scene_name: str, vla3d_dir: str | Path) -> dict[int, str]:
    """``object_id → dominant colour`` from VLA-3D ``<scene>_object_result.csv``.

    Uses ``object_color_scheme1`` (the most-dominant named colour). Returns an
    empty dict when the CSV is missing, so GT-colour join is a no-op offline
    when the VLA-3D source isn't present.
    """
    csv_path = Path(vla3d_dir) / scene_name / f"{scene_name}_object_result.csv"
    if not csv_path.exists():
        return {}
    colors: dict[int, str] = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                oid = int(row["object_id"])
            except (KeyError, ValueError, TypeError):
                continue
            c = (row.get("object_color_scheme1") or "").strip()
            if c and c.upper() != "N/A":
                colors[oid] = c
    return colors


def add_colors_to_object_list(lines: Iterable[str], id_color: dict[int, str]) -> list[str]:
    """Append a colour token to each GT ``object_list`` line, keyed by its
    leading ``object_id``. Lines that already carry a colour (a 2nd quoted
    token), or whose id has no colour, pass through unchanged."""
    out: list[str] = []
    for line in lines:
        s = line.rstrip()
        if not isinstance(line, str) or s.count('"') >= 4:
            out.append(line)
            continue
        try:
            oid = int(s.split()[0])
        except (ValueError, IndexError):
            out.append(line)
            continue
        color = id_color.get(oid)
        out.append(f'{s} "{color.replace(chr(34), chr(39))}"' if color else line)
    return out


def build_dataset(
    scene_graph: dict,
    scene_name: str,
    qa_paths: Iterable[str | Path],
    vla3d_dir: str | Path | None = None,
) -> tuple[list[dict], list[dict], list[str]]:
    """End-to-end: scene graph + scene name + Q&A files → (spliced records,
    truth records, detected_object_list). Empty spliced list means no Q&A
    records matched ``scene_name``.

    When ``vla3d_dir`` is given, GT ``object_list`` lines are augmented with
    each object's dominant colour from ``<scene>_object_result.csv`` (matched
    by object_id), so both lists carry natural-language colour.
    """
    detected = scene_graph_to_object_list(scene_graph)
    qa = load_qa_for_scene(qa_paths, scene_name)
    if vla3d_dir is not None:
        id_color = load_gt_colors(scene_name, vla3d_dir)
        if id_color:
            for rec in qa:
                ol = rec.get("object_list")
                if isinstance(ol, list):
                    rec["object_list"] = add_colors_to_object_list(ol, id_color)
    spliced = splice_detected(qa, detected)
    truth = truth_records(qa)
    return spliced, truth, detected
