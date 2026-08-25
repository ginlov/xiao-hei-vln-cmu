#!/usr/bin/env python3
"""Render a run directory as one self-contained HTML page.

`steps.jsonl` holds the complete model reply for every grounding call, but the
part that decides whether a call was any good is the box drawn on the picture
the model was looking at — and that lives in a separate JPEG, in a coordinate
convention (`[ymin, xmin, ymax, xmax]`, pixels on a 640 face) that nothing
prints. `scripts/show_run.py` deliberately trims to the fields the executor
reads; this shows everything, next to the image it was said about.

    uv run python scripts/build_run_page.py runs/cr_0811_03 -o /tmp/run.html

Images are downscaled and inlined as data URIs so the page needs no server and
no network. Pass `--width` to trade file size against detail.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import cv2
import numpy as np

FACES = ("front 0°", "right 90°", "back 180°", "left 270°")
# `box_2d` is [ymin, xmin, ymax, xmax]; `coord_space: "pixels"` means pixels on
# a face this size, and "normalized_1000" means Gemini's 0-1000 grid. Both
# conventions land in the same numeric range on a 640 face, which is why the
# declaration is trusted before the magnitude. Mirrors `vlm_probe.to_pixels`.
FACE_PX = 640


def to_pixels(box, space, size=FACE_PX):
    if space == "pixels":
        return [float(v) for v in box]
    if space == "normalized_1000":
        return [float(v) / 1000.0 * size for v in box]
    return ([float(v) / 1000.0 * size for v in box] if max(box) > size
            else [float(v) for v in box])


def draw(img, box, colour, label, thick=3):
    y0, x0, y1, x1 = [int(round(v)) for v in box]
    h, w = img.shape[:2]
    x0, x1 = max(0, min(w - 1, x0)), max(0, min(w - 1, x1))
    y0, y1 = max(0, min(h - 1, y0)), max(0, min(h - 1, y1))
    cv2.rectangle(img, (x0, y0), (x1, y1), colour, thick)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ty = y0 - 6 if y0 > th + 10 else y1 + th + 8
        cv2.rectangle(img, (x0, ty - th - 5), (x0 + tw + 8, ty + 4), colour, -1)
        cv2.putText(img, label, (x0 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (12, 14, 18), 2, cv2.LINE_AA)


def face_uri(path: Path, boxes, width: int, quality: int) -> str | None:
    """One face, boxes burnt in, downscaled, as a data URI."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    for box, colour, label in boxes:
        draw(img, box, colour, label)
    if width and img.shape[1] > width:
        s = width / img.shape[1]
        img = cv2.resize(img, (width, int(round(img.shape[0] * s))),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


# BGR, matching the page's tokens: target red, alternates amber, anchors teal.
C_TARGET = (44, 69, 200)
C_FEATURE = (150, 190, 235)
C_ALT = (28, 118, 168)
C_CAND = (154, 130, 47)
C_ANCHOR = (95, 174, 79)


def boxes_for(reply: dict, face_idx: int) -> list:
    """Every box the reply put on this particular face."""
    space = reply.get("coord_space")
    out = []
    if reply.get("image_index") == face_idx and reply.get("box_2d"):
        out.append((to_pixels(reply["box_2d"], space), C_TARGET, "target"))
        if reply.get("feature_box_2d"):
            out.append((to_pixels(reply["feature_box_2d"], space), C_FEATURE,
                        "feature"))
    for group, colour, tag in (("alternates", C_ALT, "alt"),
                               ("candidates", C_CAND, "cand"),
                               ("anchors", C_ANCHOR, "anchor")):
        for i, it in enumerate(reply.get(group) or []):
            if not isinstance(it, dict) or it.get("image_index") != face_idx:
                continue
            if not it.get("box_2d"):
                continue
            name = it.get("name") or it.get("note") or ""
            lab = f"{tag}{i + 1}" + (f" {name[:22]}" if name else "")
            out.append((to_pixels(it["box_2d"], space), colour, lab))
    for it in reply.get("sightings") or []:
        if isinstance(it, dict) and it.get("image_index") == face_idx and it.get("box_2d"):
            out.append((to_pixels(it["box_2d"], space), C_ANCHOR,
                        f"sighting s{it.get('step', '?')}"))
    return out


def esc(x) -> str:
    return html.escape(str(x), quote=True)


def pretty(obj) -> str:
    return esc(json.dumps(obj, indent=2, ensure_ascii=False))


def fmt_xy(p) -> str:
    if isinstance(p, dict):
        p = p.get("position")
    if isinstance(p, (list, tuple)) and len(p) >= 2:
        return f"{float(p[0]):+.2f}, {float(p[1]):+.2f}"
    return "—"


def chips(rec: dict, reply: dict) -> str:
    """The state of one call, as it would read on an instrument panel."""
    out = []
    conf = reply.get("confidence")
    vis = reply.get("visible")
    out.append(f'<span class="chip {"ok" if vis else "no"}">'
               f'{"visible" if vis else "not visible"}</span>')
    if conf is not None:
        band = "ok" if conf >= 0.8 else ("warn" if conf >= 0.6 else "no")
        out.append(f'<span class="chip {band}">conf {float(conf):.2f}</span>')
    if reply.get("distance_m") is not None:
        out.append(f'<span class="chip">says {float(reply["distance_m"]):.1f} m</span>')
    if reply.get("relation"):
        out.append(f'<span class="chip rel">{esc(reply["relation"])} '
                   f'&times;{len(reply.get("candidates") or [])}</span>')
    wp = rec.get("waypoint")
    if wp:
        out.append(f'<span class="chip {"ok" if wp.get("committed") else ""}">'
                   f'{"DESTINATION" if wp.get("committed") else "step point"}</span>')
        if wp.get("blind"):
            out.append('<span class="chip warn">blind lift</span>')
    for key, cls, text in (("arrived", "ok", "arrived"),
                           ("stopped", "no", "stopped"),
                           ("looped", "warn", "looped"),
                           ("binding_nearer", "warn", "rebound nearer"),
                           ("relation_failed", "no", "relation failed"),
                           ("binding_dropped", "no", "binding dropped")):
        if key in rec:
            v = rec[key]
            extra = f": {esc(v)}" if isinstance(v, str) else ""
            out.append(f'<span class="chip {cls}">{text}{extra}</span>')
    return "".join(out)


def render(run: Path, width: int, quality: int) -> str:
    steps = [json.loads(l) for l in (run / "steps.jsonl").open()]
    plan_path = run / "plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.is_file() else {}

    head = [f'<header class="masthead">',
            f'<p class="eyebrow">grounding log</p>',
            f'<h1>{esc(plan.get("question") or run.name)}</h1>',
            f'<p class="sub"><code>{esc(run)}</code> &middot; '
            f'{len(steps)} model calls</p>']
    if plan.get("results"):
        head.append('<ol class="legs">')
        for r in plan["results"]:
            cls = "ok" if r.get("ok") else "no"
            xy = (f'<span class="mono">{fmt_xy(r.get("xy"))}</span>'
                  if r.get("xy") else "")
            head.append(f'<li class="{cls}"><span class="dot"></span>'
                        f'<span class="clause">{esc(r.get("clause"))}</span>'
                        f'<span class="why">{esc(r.get("why"))}</span>{xy}</li>')
        head.append("</ol>")
    head.append("</header>")

    body = []
    leg_seen = set()
    for rec in steps:
        reply = rec.get("reply") or {}
        k = rec.get("clause")
        if k not in leg_seen:
            leg_seen.add(k)
            body.append(f'<h2 class="leg-head">leg {esc(k)} &mdash; '
                        f'<em>{esc(rec.get("phrase"))}</em></h2>')

        n = rec.get("step")
        chosen = reply.get("image_index")
        tiles = []
        for i, name in enumerate(FACES):
            p = run / f"step{n}_face{i}.jpg"
            if not p.is_file():
                continue
            uri = face_uri(p, boxes_for(reply, i), width, quality)
            if uri is None:
                continue
            mark = " chosen" if i == chosen else ""
            tiles.append(
                f'<figure class="tile{mark}">'
                f'<img src="{uri}" alt="{esc(name)} view at step {esc(n)}" '
                f'loading="lazy">'
                f'<figcaption>{esc(name)}'
                f'{" &larr; answered here" if i == chosen else ""}</figcaption>'
                f'</figure>')

        rows = []
        for label, val in (("evidence", reply.get("evidence")),
                           ("here", reply.get("here")),
                           ("explore", (reply.get("explore") or {}).get("why"))):
            if val:
                rows.append(f'<div class="field"><dt>{label}</dt>'
                            f'<dd>{esc(val)}</dd></div>')
        for label, key in (("relation", "relation"), ("binding", "binding"),
                           ("converter", "converter"), ("drive", "drive")):
            if key in rec and rec[key] is not None:
                v = rec[key]
                txt = v if isinstance(v, str) else json.dumps(
                    v, ensure_ascii=False, default=str)
                rows.append(f'<div class="field"><dt>{label}</dt>'
                            f'<dd class="mono small">{esc(txt)}</dd></div>')
        wp = rec.get("waypoint")
        if wp and wp.get("reason"):
            rows.append(f'<div class="field"><dt>waypoint</dt>'
                        f'<dd class="mono small">{esc(wp["reason"])}</dd></div>')

        body.append(
            f'<article class="step" id="step{esc(n)}">'
            f'<div class="step-bar">'
            f'<span class="num">step {esc(n)}</span>'
            f'<span class="pose mono">at {fmt_xy(rec.get("pose"))}</span>'
            f'<span class="chips">{chips(rec, reply)}</span>'
            f'</div>'
            f'<div class="tiles">{"".join(tiles)}</div>'
            f'<dl class="fields">{"".join(rows)}</dl>'
            f'<details class="raw"><summary>complete model reply '
            f'({len(reply)} fields)</summary>'
            f'<pre>{pretty(reply)}</pre></details>'
            f'</article>')

    return TEMPLATE.replace("{{TITLE}}", esc(plan.get("question") or run.name)) \
                   .replace("{{HEAD}}", "".join(head)) \
                   .replace("{{BODY}}", "".join(body))


TEMPLATE = """<title>{{TITLE}}</title>
<style>
:root{
  --ground:#f2f3f6; --panel:#ffffff; --panel-2:#f7f8fa;
  --ink:#171a21; --muted:#5b6373; --faint:#8a92a1;
  --rule:#dde0e7; --rule-strong:#c6cbd6;
  --accent:#b8402a; --ok:#2c7a5c; --warn:#9a6c14; --no:#b03a2b;
  --shadow:0 1px 2px rgba(23,26,33,.06),0 6px 20px rgba(23,26,33,.05);
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#0f1218; --panel:#171b23; --panel-2:#1c212a;
    --ink:#e4e7ed; --muted:#98a1b2; --faint:#6d7688;
    --rule:#272d38; --rule-strong:#39414f;
    --accent:#e2705a; --ok:#4faa84; --warn:#cf9e45; --no:#e0705c;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.3);
  }
}
:root[data-theme="dark"]{
  --ground:#0f1218; --panel:#171b23; --panel-2:#1c212a;
  --ink:#e4e7ed; --muted:#98a1b2; --faint:#6d7688;
  --rule:#272d38; --rule-strong:#39414f;
  --accent:#e2705a; --ok:#4faa84; --warn:#cf9e45; --no:#e0705c;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.3);
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:var(--mono); font-size:14px; line-height:1.6;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1180px; margin:0 auto; padding:32px 20px 80px;
      display:flex; flex-direction:column; gap:28px}
.mono{font-family:var(--mono); font-variant-numeric:tabular-nums}
code{font-family:var(--mono)}

.masthead{display:flex; flex-direction:column; gap:10px;
  padding-bottom:22px; border-bottom:2px solid var(--rule-strong)}
.eyebrow{margin:0; font-size:11px; letter-spacing:.16em; text-transform:uppercase;
  color:var(--accent); font-weight:600}
.masthead h1{margin:0; font-family:var(--sans); font-size:clamp(20px,3vw,28px);
  line-height:1.25; font-weight:650; text-wrap:balance; max-width:34ch}
.sub{margin:0; color:var(--muted); font-size:12.5px}
.legs{list-style:none; margin:8px 0 0; padding:0;
  display:flex; flex-direction:column; gap:7px}
.legs li{display:flex; align-items:baseline; gap:10px; flex-wrap:wrap;
  font-size:12.5px; color:var(--muted)}
.legs .dot{width:8px; height:8px; border-radius:50%; flex:none;
  background:var(--no); transform:translateY(-1px)}
.legs li.ok .dot{background:var(--ok)}
.legs .clause{color:var(--ink); font-weight:600}
.legs .why{color:var(--faint)}

.leg-head{margin:14px 0 -8px; font-family:var(--sans); font-size:15px;
  font-weight:650; letter-spacing:.01em; color:var(--muted)}
.leg-head em{color:var(--ink); font-style:normal}

.step{background:var(--panel); border:1px solid var(--rule); border-radius:8px;
  box-shadow:var(--shadow); overflow:hidden}
.step-bar{display:flex; align-items:center; gap:12px; flex-wrap:wrap;
  padding:11px 16px; background:var(--panel-2);
  border-bottom:1px solid var(--rule)}
.num{font-weight:700; letter-spacing:.02em}
.pose{color:var(--muted); font-size:12.5px}
.chips{display:flex; gap:6px; flex-wrap:wrap; margin-left:auto}
.chip{font-size:11px; padding:2px 8px; border-radius:999px;
  border:1px solid var(--rule-strong); color:var(--muted); white-space:nowrap}
.chip.ok{color:var(--ok); border-color:currentColor}
.chip.no{color:var(--no); border-color:currentColor}
.chip.warn{color:var(--warn); border-color:currentColor}
.chip.rel{color:var(--accent); border-color:currentColor}

.tiles{display:grid; grid-template-columns:repeat(4,1fr); gap:1px;
  background:var(--rule)}
@media (max-width:820px){.tiles{grid-template-columns:repeat(2,1fr)}}
.tile{margin:0; background:var(--panel); position:relative}
.tile img{display:block; width:100%; height:auto}
.tile figcaption{padding:6px 9px; font-size:11px; color:var(--faint);
  letter-spacing:.03em}
.tile.chosen{outline:2px solid var(--accent); outline-offset:-2px; z-index:1}
.tile.chosen figcaption{color:var(--accent); font-weight:600}

.fields{margin:0; padding:14px 16px; display:flex; flex-direction:column; gap:11px}
.field{display:grid; grid-template-columns:88px 1fr; gap:14px; align-items:start}
@media (max-width:640px){.field{grid-template-columns:1fr; gap:3px}}
.field dt{color:var(--faint); font-size:11px; letter-spacing:.09em;
  text-transform:uppercase; padding-top:2px}
.field dd{margin:0; font-family:var(--sans); font-size:13.5px; color:var(--ink);
  max-width:78ch}
.field dd.mono{font-family:var(--mono); font-size:12px; color:var(--muted);
  overflow-x:auto}
.field dd.small{font-size:11.5px}

.raw{border-top:1px solid var(--rule); background:var(--panel-2)}
.raw summary{cursor:pointer; padding:10px 16px; font-size:11.5px;
  color:var(--muted); letter-spacing:.04em; list-style:none}
.raw summary::-webkit-details-marker{display:none}
.raw summary::before{content:"▸ "; color:var(--accent)}
.raw[open] summary::before{content:"▾ "}
.raw summary:hover{color:var(--ink)}
.raw summary:focus-visible{outline:2px solid var(--accent); outline-offset:-2px}
.raw pre{margin:0; padding:0 16px 16px; overflow-x:auto; font-size:11.5px;
  line-height:1.55; color:var(--muted)}
</style>
<div class="wrap">{{HEAD}}{{BODY}}</div>
"""


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--width", type=int, default=440,
                    help="downscale faces to this width (0 keeps full size)")
    ap.add_argument("--quality", type=int, default=72)
    a = ap.parse_args()

    if not (a.run / "steps.jsonl").is_file():
        raise SystemExit(f"no steps.jsonl in {a.run}")
    out = a.out or (a.run / "index.html")
    page = render(a.run, a.width, a.quality)
    out.write_text(page, encoding="utf-8")
    print(f"{out}  ({len(page.encode()) / 1_048_576:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
