#!/usr/bin/env python3
"""Generate self-contained HTML reports from VLM tick-logger sessions.

Usage::

    # All questions in a session
    python scripts/generate_report.py vlm_logs/session_20260530_143022/

    # Single question directory
    python scripts/generate_report.py vlm_logs/session_*/q_001_*/

    # Filter by keyword
    python scripts/generate_report.py vlm_logs/session_*/ -q chairs

Each question directory gets a ``report.html`` with embedded camera
playback, pose trajectory, sensor BEV, per-tick I/O table, and latency
chart.

Requires: ``pip install xiao-hei-vln[replay]`` (matplotlib + pillow).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from html import escape
from pathlib import Path

import numpy as np


def _require_matplotlib():
    try:
        import matplotlib  # noqa: F401

        return True
    except ImportError:
        print(
            "Error: matplotlib is required. Install with:\n"
            "  uv pip install 'matplotlib>=3.8'",
            file=sys.stderr,
        )
        sys.exit(1)


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# --- data loading ---


def load_session_meta(session_dir: Path) -> dict:
    p = session_dir / "session.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def load_question(q_dir: Path) -> list[dict]:
    ticks_file = q_dir / "ticks.jsonl"
    if not ticks_file.exists():
        return []
    ticks = []
    for line in ticks_file.read_text().strip().splitlines():
        if line:
            ticks.append(json.loads(line))
    return ticks


def find_question_dirs(
    path: Path, q_filter: str | None = None,
) -> tuple[Path, list[Path]]:
    """Return (session_dir, [question_dirs])."""
    if (path / "ticks.jsonl").exists():
        return path.parent, [path]

    q_dirs = sorted(
        d for d in path.iterdir()
        if d.is_dir() and d.name.startswith("q_")
    )
    if q_filter:
        q_dirs = [d for d in q_dirs if q_filter in d.name]
    return path, q_dirs


# --- chart generators ---


def render_pose_chart(ticks: list[dict]) -> str:
    """Render pose trajectory + waypoint outputs as a base64 PNG."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))

    xs, ys, labels = [], [], []
    for t in ticks:
        pose = t.get("pose")
        if pose is None:
            continue
        xs.append(pose["position"]["x"])
        ys.append(pose["position"]["y"])
        labels.append(t["tick_id"])

    if xs:
        ax.plot(xs, ys, "o-", color="#2196F3", markersize=4, label="robot path")
        ax.annotate(
            "start", (xs[0], ys[0]), fontsize=8,
            color="green", fontweight="bold",
        )

    for t in ticks:
        output = t.get("output")
        if output and output.get("kind") == "waypoint_path":
            for wp in output.get("waypoints", []):
                ax.plot(
                    wp["x"], wp["y"], "^", color="#FF9800",
                    markersize=8,
                )
                ax.annotate(
                    f"wp t{t['tick_id']}", (wp["x"], wp["y"]),
                    fontsize=6, color="#FF9800",
                )

        if output and output.get("kind") == "numerical":
            pose = t.get("pose")
            if pose:
                ax.plot(
                    pose["position"]["x"], pose["position"]["y"],
                    "*", color="red", markersize=15,
                )
                ax.annotate(
                    f"answer={output['value']}",
                    (pose["position"]["x"], pose["position"]["y"]),
                    fontsize=8, color="red", fontweight="bold",
                )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Pose Trajectory + Waypoints")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


def render_latency_chart(ticks: list[dict]) -> str:
    """Render inference latency bar chart as a base64 PNG."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3))
    tick_ids = [t["tick_id"] for t in ticks]
    latencies = [t.get("inference_ms", 0) for t in ticks]
    colors = [
        "#4CAF50" if ms < 200 else "#FF9800" if ms < 400 else "#F44336"
        for ms in latencies
    ]
    ax.bar(range(len(tick_ids)), latencies, color=colors)
    ax.set_xticks(range(len(tick_ids)))
    ax.set_xticklabels([str(tid) for tid in tick_ids], fontsize=7)
    ax.set_xlabel("tick_id")
    ax.set_ylabel("inference (ms)")
    ax.set_title("Inference Latency per Tick")
    ax.grid(True, alpha=0.3, axis="y")

    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


def render_bev_chart(q_dir: Path, ticks: list[dict]) -> str | None:
    """Render a BEV scatter from the last tick's point clouds."""
    import matplotlib.pyplot as plt

    last_with_pc = None
    for t in reversed(ticks):
        if t.get("pointclouds"):
            last_with_pc = t
            break
    if last_with_pc is None:
        return None

    fig, ax = plt.subplots(figsize=(7, 7))
    pcs = last_with_pc["pointclouds"]

    if "terrain_ext" in pcs:
        pts = np.load(q_dir / pcs["terrain_ext"])
        ax.scatter(
            pts[:, 0], pts[:, 1], s=0.5, c="lightgray",
            alpha=0.4, label="terrain_ext (20m)",
        )

    if "terrain_local" in pcs:
        pts = np.load(q_dir / pcs["terrain_local"])
        sc = ax.scatter(
            pts[:, 0], pts[:, 1], s=1, c=pts[:, 3],
            cmap="RdYlGn_r", alpha=0.6, label="terrain_local (5m)",
        )
        fig.colorbar(sc, ax=ax, label="cost", shrink=0.6)

    if "registered" in pcs:
        pts = np.load(q_dir / pcs["registered"])
        ax.scatter(
            pts[:, 0], pts[:, 1], s=0.3, c="#2196F3",
            alpha=0.5, label="registered_scan",
        )

    pose = last_with_pc.get("pose")
    if pose:
        ax.plot(
            pose["position"]["x"], pose["position"]["y"],
            "^", color="red", markersize=12, label="robot",
        )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"Sensor BEV (tick {last_with_pc['tick_id']})",
    )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")

    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


# --- camera frames ---


def load_camera_frames(q_dir: Path, ticks: list[dict]) -> list[dict]:
    """Load JPEG frames as base64 for the JS player."""
    frames = []
    for t in ticks:
        img_path = t.get("image_path")
        if not img_path:
            continue
        full = q_dir / img_path
        if not full.exists():
            continue
        b64 = base64.b64encode(full.read_bytes()).decode("ascii")
        frames.append({
            "tick_id": t["tick_id"],
            "time": t.get("tick_time", 0),
            "data_url": f"data:image/jpeg;base64,{b64}",
        })
    return frames


# --- HTML rendering ---


def render_tick_rows(ticks: list[dict]) -> str:
    rows = []
    for t in ticks:
        tid = t["tick_id"]
        time_s = t.get("tick_time", 0)
        ms = t.get("inference_ms", 0)
        output = t.get("output")
        kind = output.get("kind", "?") if output else "(none)"
        value = ""
        if output:
            if kind == "numerical":
                value = str(output.get("value", ""))
            elif kind == "object_reference":
                value = output.get("label", "")
            elif kind == "waypoint_path":
                wps = output.get("waypoints", [])
                value = f"{len(wps)} waypoint(s)"

        rationale = ""
        if output and output.get("rationale"):
            rationale = escape(output["rationale"])

        sys_prompt = escape(t.get("system_prompt", ""))
        user_text = escape(t.get("user_text", ""))
        output_json = escape(
            json.dumps(output, indent=2) if output else "null",
        )
        evidence = t.get("evidence", [])
        evidence_html = "<br>".join(escape(e) for e in evidence)

        rows.append(f"""
        <tr>
          <td>{tid}</td>
          <td>{time_s:.2f}</td>
          <td>{ms:.1f}</td>
          <td>{kind}</td>
          <td>{escape(value)}</td>
          <td style="max-width:300px;font-size:11px">{rationale}</td>
        </tr>
        <tr class="detail-row" style="display:none" data-tick="{tid}">
          <td colspan="6">
            <details><summary>System Prompt</summary>
              <pre>{sys_prompt}</pre>
            </details>
            <details><summary>User Text</summary>
              <pre>{user_text}</pre>
            </details>
            <details><summary>Output JSON</summary>
              <pre>{output_json}</pre>
            </details>
            <details><summary>Evidence ({len(evidence)} entries)</summary>
              <div style="font-size:12px">{evidence_html or "(none)"}</div>
            </details>
          </td>
        </tr>""")
    return "\n".join(rows)


def generate_html(
    session_meta: dict,
    q_name: str,
    ticks: list[dict],
    q_dir: Path,
) -> str:
    q_text = ticks[0].get("question_text", "?") if ticks else "?"
    q_type = ticks[0].get("question_type", "?") if ticks else "?"

    final_output = ticks[-1].get("output") if ticks else None
    final_answer = ""
    if final_output:
        if final_output.get("kind") == "numerical":
            final_answer = f"Numerical: {final_output.get('value')}"
        elif final_output.get("kind") == "object_reference":
            final_answer = f"Object: {final_output.get('label')}"
        elif final_output.get("kind") == "waypoint_path":
            final_answer = "Waypoint path"

    latencies = [t.get("inference_ms", 0) for t in ticks]
    avg_ms = sum(latencies) / len(latencies) if latencies else 0

    cfg = session_meta.get("config", {})
    tick_hz = session_meta.get("tick_hz", 2.0)
    frame_interval_ms = int(1000.0 / tick_hz) if tick_hz > 0 else 500

    # Camera frames
    frames = load_camera_frames(q_dir, ticks)
    frames_json = json.dumps([
        {"tick_id": f["tick_id"], "time": f["time"], "src": f["data_url"]}
        for f in frames
    ])

    # Charts
    pose_b64 = render_pose_chart(ticks)
    latency_b64 = render_latency_chart(ticks)
    bev_b64 = render_bev_chart(q_dir, ticks)

    # Tick table
    tick_rows = render_tick_rows(ticks)

    bev_section = ""
    if bev_b64:
        bev_section = f"""
        <h2>Sensor Bird's-Eye View</h2>
        <img src="data:image/png;base64,{bev_b64}"
             style="max-width:100%">
        """

    camera_section = ""
    if frames:
        camera_section = f"""
        <h2>Camera Playback</h2>
        <div id="player-container">
          <img id="camera-img"
               style="max-width:100%;border:1px solid #ddd">
          <div style="margin-top:8px">
            <button onclick="prevFrame()">&#9664;</button>
            <button id="play-btn" onclick="togglePlay()">
              &#9654; Play
            </button>
            <button onclick="nextFrame()">&#9654;</button>
            <input id="slider" type="range" min="0"
                   max="{len(frames)-1}" value="0"
                   oninput="showFrame(this.value)"
                   style="width:300px;vertical-align:middle">
            <span id="frame-label"
                  style="font-family:monospace;margin-left:8px">
            </span>
          </div>
        </div>
        <script>
        const frames = {frames_json};
        let idx = 0, timer = null;
        function showFrame(i) {{
          idx = parseInt(i);
          document.getElementById('camera-img').src = frames[idx].src;
          document.getElementById('slider').value = idx;
          document.getElementById('frame-label').textContent =
            'tick ' + frames[idx].tick_id +
            ' | ' + frames[idx].time.toFixed(2) + 's';
        }}
        function nextFrame() {{
          showFrame(Math.min(idx + 1, frames.length - 1));
        }}
        function prevFrame() {{
          showFrame(Math.max(idx - 1, 0));
        }}
        function togglePlay() {{
          if (timer) {{
            clearInterval(timer);
            timer = null;
            document.getElementById('play-btn').innerHTML = '&#9654; Play';
          }} else {{
            timer = setInterval(() => {{
              if (idx >= frames.length - 1) {{
                clearInterval(timer);
                timer = null;
                document.getElementById('play-btn').innerHTML =
                  '&#9654; Play';
                return;
              }}
              nextFrame();
            }}, {frame_interval_ms});
            document.getElementById('play-btn').innerHTML = '&#9724; Pause';
          }}
        }}
        if (frames.length > 0) showFrame(0);
        </script>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>VLM Report: {escape(q_name)}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                 Roboto, sans-serif;
    max-width: 1100px; margin: 0 auto; padding: 20px;
    color: #333; background: #fafafa;
  }}
  h1 {{ color: #1a237e; border-bottom: 2px solid #1a237e;
       padding-bottom: 8px; }}
  h2 {{ color: #283593; margin-top: 32px; }}
  .meta {{ background: #e8eaf6; padding: 12px 16px; border-radius: 6px;
           margin-bottom: 20px; }}
  .meta td {{ padding: 2px 16px 2px 0; font-size: 14px; }}
  table.ticks {{ border-collapse: collapse; width: 100%;
                 font-size: 13px; }}
  table.ticks th {{ background: #3f51b5; color: white; padding: 8px;
                    text-align: left; }}
  table.ticks td {{ padding: 6px 8px; border-bottom: 1px solid #e0e0e0; }}
  table.ticks tr:hover {{ background: #e8eaf6; cursor: pointer; }}
  table.ticks tr.detail-row {{ background: #f5f5f5; }}
  table.ticks tr.detail-row:hover {{ background: #f5f5f5; }}
  details {{ margin: 4px 0; }}
  details summary {{ cursor: pointer; color: #3f51b5;
                     font-weight: bold; font-size: 12px; }}
  pre {{ background: #263238; color: #eeffff; padding: 10px;
         border-radius: 4px; font-size: 11px; overflow-x: auto;
         max-height: 300px; }}
  .answer {{ font-size: 20px; font-weight: bold; color: #1b5e20;
             background: #e8f5e9; padding: 8px 16px;
             border-radius: 6px; display: inline-block; }}
</style>
</head>
<body>

<h1>VLM Report: {escape(q_name)}</h1>

<div class="meta">
<table>
  <tr><td><b>Question</b></td><td>{escape(q_text)}</td></tr>
  <tr><td><b>Type</b></td><td>{q_type}</td></tr>
  <tr><td><b>Ticks</b></td><td>{len(ticks)}</td></tr>
  <tr><td><b>Avg latency</b></td><td>{avg_ms:.1f} ms</td></tr>
  <tr><td><b>Model</b></td>
      <td>{escape(str(cfg.get('model', 'N/A')))}</td></tr>
  <tr><td><b>Session</b></td>
      <td>{escape(session_meta.get('start_time', 'N/A'))}</td></tr>
</table>
</div>

{"<div class='answer'>" + escape(final_answer) + "</div>" if final_answer else ""}

{camera_section}

<h2>Pose Trajectory + Waypoints</h2>
<img src="data:image/png;base64,{pose_b64}" style="max-width:100%">

{bev_section}

<h2>Tick-by-Tick Detail</h2>
<p style="font-size:12px;color:#666">
  Click a row to expand/collapse prompts and output JSON.
</p>
<table class="ticks">
  <thead>
    <tr>
      <th>tick</th><th>time (s)</th><th>infer (ms)</th>
      <th>output</th><th>value</th><th>rationale</th>
    </tr>
  </thead>
  <tbody>
    {tick_rows}
  </tbody>
</table>
<script>
document.querySelectorAll('table.ticks tbody tr:not(.detail-row)')
  .forEach(row => {{
    row.addEventListener('click', () => {{
      const next = row.nextElementSibling;
      if (next && next.classList.contains('detail-row')) {{
        next.style.display =
          next.style.display === 'none' ? '' : 'none';
      }}
    }});
  }});
</script>

<h2>Latency</h2>
<img src="data:image/png;base64,{latency_b64}" style="max-width:100%">

<footer style="margin-top:40px;padding-top:12px;border-top:1px solid #ddd;
               font-size:11px;color:#999">
  Generated by <code>scripts/generate_report.py</code> &mdash;
  Team Xiao Hei VLN CMU
</footer>

</body>
</html>"""


# --- main ---


def main() -> None:
    _require_matplotlib()

    parser = argparse.ArgumentParser(
        description="Generate HTML reports from VLM tick-logger sessions",
    )
    parser.add_argument(
        "path", type=Path,
        help="Session directory or single question directory",
    )
    parser.add_argument(
        "-q", "--question", type=str, default=None,
        help="Filter questions by substring (e.g. '001' or 'chairs')",
    )
    args = parser.parse_args()

    session_dir, q_dirs = find_question_dirs(args.path, args.question)
    session_meta = load_session_meta(session_dir)

    if not q_dirs:
        print("No question directories found.", file=sys.stderr)
        sys.exit(1)

    for q_dir in q_dirs:
        ticks = load_question(q_dir)
        if not ticks:
            print(f"  {q_dir.name}: no ticks, skipping")
            continue

        html = generate_html(session_meta, q_dir.name, ticks, q_dir)
        out_path = q_dir / "report.html"
        out_path.write_text(html)
        print(f"  {out_path}")

    print(f"\nDone. {len(q_dirs)} report(s) generated.")


if __name__ == "__main__":
    main()
