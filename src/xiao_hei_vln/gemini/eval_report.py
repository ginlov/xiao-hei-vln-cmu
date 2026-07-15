"""Human-readable HTML report for the offline Gemini evaluation.

Turns a ``(GT, predictions)`` pair into a single self-contained
``report.html`` — no server, no external assets. For every scoreable
question it renders a card:

  - the question text and a pass/fail badge (2 / 1 / 0 challenge points),
  - Gemini's answer (label, centre, rationale) next to the ground truth,
  - for object-reference: a **top-down scene plot** of every object in
    the room, with Gemini's box (red) and the ground-truth box (green)
    highlighted so a wrong pick is obvious at a glance,
  - for numerical: the predicted count vs the ground-truth count.

The offline path sends no camera imagery (the scene is a coordinate
graph), so the "images" here are matplotlib top-down renders rather than
photos — but they make a mis-pick visible immediately.

Usage::

    uv run python -m xiao_hei_vln.gemini.eval_report \\
        --gt  dataset/vla3d_ref.jsonl \\
        --pred pred_ref.jsonl \\
        --out report.html
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from dataclasses import dataclass, field
from html import escape
from pathlib import Path

from xiao_hei_vln.eval_sampler.gt_converter import gt_from_entry
from xiao_hei_vln.eval_sampler.object_list import ObjectEntry, parse_object_list
from xiao_hei_vln.evaluator.metrics.object_reference import (
    _center_distance,
    _challenge_score,
    _iou_3d_aabb,
)
from xiao_hei_vln.messages.outputs import (
    NumericalResponse,
    ObjectReferenceResponse,
    parse_vlm_output,
)

SCOREABLE_TYPES = frozenset({"numerical", "object_reference"})

# Badge colours by challenge score (object-ref) / correctness (numerical).
_BADGE = {2: ("#16a34a", "2 pts"), 1: ("#d97706", "1 pt"), 0: ("#dc2626", "0 pts")}


@dataclass
class RefResult:
    question: str
    pred: ObjectReferenceResponse
    gt: ObjectReferenceResponse
    iou: float
    score: int
    center_dist: float
    objects: list[ObjectEntry]


@dataclass
class NumResult:
    question: str
    pred: NumericalResponse
    gt_value: int
    correct: bool


@dataclass
class Report:
    ref: list[RefResult] = field(default_factory=list)
    num: list[NumResult] = field(default_factory=list)
    unmatched: int = 0


# --- assembly --------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[warn] {path}:{lineno}: {exc}", file=sys.stderr)
    return rows


def build_report(gt_path: Path, pred_path: Path, *, limit: int | None = None) -> Report:
    """Match predictions to GT by question text and score each one."""
    gt_lookup: dict[str, dict] = {}
    for entry in _load_jsonl(gt_path):
        q = (entry.get("question") or "").strip()
        if q and q not in gt_lookup:  # keep first on duplicate question text
            gt_lookup[q] = entry

    report = Report()
    preds = _load_jsonl(pred_path)
    if limit is not None:
        preds = preds[:limit]

    for pe in preds:
        q = (pe.get("question") or "").strip()
        entry = gt_lookup.get(q)
        raw = pe.get("prediction")
        if entry is None or raw is None:
            report.unmatched += 1
            continue
        gt_out = gt_from_entry(entry)
        try:
            pred = parse_vlm_output(raw)
        except Exception:  # noqa: BLE001 — malformed prediction row
            report.unmatched += 1
            continue

        if isinstance(pred, ObjectReferenceResponse) and isinstance(
            gt_out, ObjectReferenceResponse
        ):
            iou = _iou_3d_aabb(pred.center, pred.size, gt_out.center, gt_out.size)
            raw_list = entry.get("object_list")
            objects = list(parse_object_list(raw_list if isinstance(raw_list, list) else []).values())
            report.ref.append(
                RefResult(
                    question=q,
                    pred=pred,
                    gt=gt_out,
                    iou=iou,
                    score=_challenge_score(iou),
                    center_dist=_center_distance(pred.center, gt_out.center),
                    objects=objects,
                )
            )
        elif isinstance(pred, NumericalResponse) and isinstance(gt_out, NumericalResponse):
            report.num.append(
                NumResult(
                    question=q,
                    pred=pred,
                    gt_value=gt_out.value,
                    correct=pred.value == gt_out.value,
                )
            )
        else:
            report.unmatched += 1
    return report


# --- top-down scene plot ---------------------------------------------------


def render_topdown_png(result: RefResult, *, dpi: int = 90) -> bytes:
    """Render a top-down (x, y) view: all objects + pred box (red) + GT box (green)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=dpi)

    for o in result.objects:
        ax.plot(o.center.x, o.center.y, "o", color="#cbd5e1", markersize=3, zorder=1)

    def _box(center, size, color: str, label: str) -> None:
        ax.add_patch(
            Rectangle(
                (center.x - size.x / 2, center.y - size.y / 2),
                size.x,
                size.y,
                fill=False,
                edgecolor=color,
                linewidth=2.2,
                label=label,
                zorder=4,
            )
        )
        ax.plot(center.x, center.y, "x", color=color, markersize=9, zorder=5)

    _box(result.gt.center, result.gt.size, "#16a34a", f"ground truth: {result.gt.label}")
    _box(result.pred.center, result.pred.size, "#dc2626", f"gemini: {result.pred.label}")

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"IoU = {result.iou:.2f}   ·   Δcenter = {result.center_dist:.2f} m", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, color="#f1f5f9", zorder=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _png_data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


# --- HTML ------------------------------------------------------------------


def _vec(v) -> str:
    return f"({v.x:.2f}, {v.y:.2f}, {v.z:.2f})"


def _ref_card(r: RefResult) -> str:
    color, pts = _BADGE[r.score]
    img = _png_data_url(render_topdown_png(r))
    return f"""
    <div class="card">
      <div class="chead">
        <span class="q">{escape(r.question)}</span>
        <span class="badge" style="background:{color}">{pts} · IoU {r.iou:.2f}</span>
      </div>
      <div class="cbody">
        <div class="details">
          <div class="row"><span class="k">gemini</span>
            <span class="pred">{escape(r.pred.label)} @ {_vec(r.pred.center)}</span></div>
          <div class="row"><span class="k">truth</span>
            <span class="gt">{escape(r.gt.label)} @ {_vec(r.gt.center)}</span></div>
          <div class="row"><span class="k">Δcenter</span><span>{r.center_dist:.2f} m</span></div>
          <div class="rationale">{escape(r.pred.rationale or "")}</div>
        </div>
        <img src="{img}" alt="top-down scene">
      </div>
    </div>"""


def _num_card(r: NumResult) -> str:
    color, _ = _BADGE[1 if r.correct else 0]
    verdict = "correct" if r.correct else "wrong"
    return f"""
    <div class="card">
      <div class="chead">
        <span class="q">{escape(r.question)}</span>
        <span class="badge" style="background:{color}">{verdict}</span>
      </div>
      <div class="cbody">
        <div class="details">
          <div class="row"><span class="k">gemini</span><span class="pred">{r.pred.value}</span></div>
          <div class="row"><span class="k">truth</span><span class="gt">{r.gt_value}</span></div>
          <div class="rationale">{escape(r.pred.rationale or "")}</div>
        </div>
      </div>
    </div>"""


def _summary(report: Report) -> str:
    bits: list[str] = []
    if report.ref:
        n = len(report.ref)
        mean_iou = sum(r.iou for r in report.ref) / n
        sr50 = sum(1 for r in report.ref if r.iou >= 0.5) / n
        score = sum(r.score for r in report.ref) / n
        bits.append(
            f'<div class="stat"><b>{mean_iou:.3f}</b><span>mean IoU (n={n})</span></div>'
            f'<div class="stat"><b>{sr50:.0%}</b><span>SR@IoU≥0.5</span></div>'
            f'<div class="stat"><b>{score:.2f}/2</b><span>mean score</span></div>'
        )
    if report.num:
        n = len(report.num)
        acc = sum(1 for r in report.num if r.correct) / n
        bits.append(f'<div class="stat"><b>{acc:.0%}</b><span>accuracy (n={n})</span></div>')
    return "".join(bits)


def generate_html(report: Report) -> str:
    cards = [_ref_card(r) for r in report.ref] + [_num_card(r) for r in report.num]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gemini offline eval report</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 14px/1.5 system-ui, sans-serif; margin: 0; padding: 24px;
         background: #f8fafc; color: #0f172a; }}
  @media (prefers-color-scheme: dark) {{ body {{ background:#0b1120; color:#e2e8f0; }}
    .card {{ background:#111827 !important; border-color:#1f2937 !important; }} }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  .sub {{ color:#64748b; margin-bottom:16px; }}
  .stats {{ display:flex; gap:12px; flex-wrap:wrap; margin-bottom:20px; }}
  .stat {{ background:#fff; border:1px solid #e2e8f0; border-radius:10px; padding:10px 16px;
          min-width:110px; }}
  .stat b {{ display:block; font-size:22px; }}
  .stat span {{ color:#64748b; font-size:12px; }}
  .card {{ background:#fff; border:1px solid #e2e8f0; border-radius:12px; margin-bottom:14px;
          overflow:hidden; }}
  .chead {{ display:flex; justify-content:space-between; align-items:center; gap:12px;
           padding:12px 16px; border-bottom:1px solid #e2e8f0; }}
  .q {{ font-weight:600; }}
  .badge {{ color:#fff; border-radius:999px; padding:3px 10px; font-size:12px; white-space:nowrap; }}
  .cbody {{ display:flex; gap:16px; padding:16px; flex-wrap:wrap; align-items:flex-start; }}
  .details {{ flex:1; min-width:240px; }}
  .row {{ display:flex; gap:8px; margin-bottom:4px; }}
  .k {{ color:#64748b; width:64px; flex:none; }}
  .pred {{ color:#dc2626; font-weight:600; }}
  .gt {{ color:#16a34a; font-weight:600; }}
  .rationale {{ margin-top:8px; color:#475569; font-style:italic; }}
  .cbody img {{ max-width:100%; width:360px; border-radius:8px; }}
</style></head>
<body>
  <h1>Gemini offline eval report</h1>
  <div class="sub">{len(report.ref)} object-reference · {len(report.num)} numerical ·
       {report.unmatched} unmatched</div>
  <div class="stats">{_summary(report)}</div>
  {"".join(cards)}
</body></html>"""


# --- CLI -------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a human-readable HTML report from GT + Gemini predictions."
    )
    parser.add_argument("--gt", type=Path, required=True, help="VLA-3D GT JSONL file.")
    parser.add_argument("--pred", type=Path, required=True, help="Predictions JSONL file.")
    parser.add_argument("--out", type=Path, default=Path("report.html"), help="Output HTML path.")
    parser.add_argument("--limit", type=int, default=None, help="Max predictions to include.")
    args = parser.parse_args()

    report = build_report(args.gt, args.pred, limit=args.limit)
    if not report.ref and not report.num:
        print("No matched predictions to report.", file=sys.stderr)
        sys.exit(1)
    args.out.write_text(generate_html(report))
    print(
        f"Wrote {args.out}  ({len(report.ref)} object-reference, "
        f"{len(report.num)} numerical, {report.unmatched} unmatched)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
