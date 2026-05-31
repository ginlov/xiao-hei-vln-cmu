#!/usr/bin/env python3
"""Replay a VLM logger session for post-run inspection.

Usage::

    python scripts/replay_session.py vlm_logs/session_20260530_143022
    python scripts/replay_session.py vlm_logs/session_20260530_143022 --images

Prints a summary table of every tick, grouped by question. With
``--images``, opens the JPEG for each tick using the system viewer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_session(session_dir: Path) -> tuple[dict, list[tuple[str, list[dict]]]]:
    """Load session.json and per-question ticks.

    Returns (session_meta, [(question_dir_name, [tick_records])]).
    """
    session_json = session_dir / "session.json"
    if not session_json.exists():
        print(f"Error: {session_json} not found", file=sys.stderr)
        sys.exit(1)

    session = json.loads(session_json.read_text())

    questions: list[tuple[str, list[dict]]] = []
    q_dirs = sorted(
        d for d in session_dir.iterdir()
        if d.is_dir() and d.name.startswith("q_")
    )
    for q_dir in q_dirs:
        ticks_file = q_dir / "ticks.jsonl"
        if not ticks_file.exists():
            continue
        ticks = []
        for line in ticks_file.read_text().strip().splitlines():
            if line:
                ticks.append(json.loads(line))
        questions.append((q_dir.name, ticks))

    return session, questions


def print_session_summary(session: dict) -> None:
    print("=" * 72)
    print("SESSION SUMMARY")
    print("=" * 72)
    print(f"  Start time : {session.get('start_time', 'N/A')}")
    print(f"  Responder  : {session.get('responder', 'N/A')}")
    print(f"  Tick Hz    : {session.get('tick_hz', 'N/A')}")
    cfg = session.get("config", {})
    print(f"  Model      : {cfg.get('model', 'N/A')}")
    print(f"  Max tokens : {cfg.get('max_output_tokens', 'N/A')}")
    print(f"  Temperature: {cfg.get('temperature', 'N/A')}")
    print()


def print_question_ticks(
    q_name: str,
    ticks: list[dict],
    session_dir: Path,
    *,
    show_images: bool = False,
) -> None:
    if not ticks:
        print(f"  {q_name}: (no ticks)")
        return

    q_text = ticks[0].get("question_text", "?")
    q_type = ticks[0].get("question_type", "?")
    print(f"  Question : {q_text!r}")
    print(f"  Type     : {q_type}")
    print(f"  Ticks    : {len(ticks)}")

    cols = [
        f"{'tick':>6}", f"{'time_s':>8}", f"{'infer_ms':>9}",
        f"{'output_kind':<18}", f"{'img':>3}",
    ]
    header = "  ".join(cols)
    print(f"  {header}")
    print(f"  {'-' * len(header)}")

    for t in ticks:
        tick_id = t.get("tick_id", "?")
        tick_time = t.get("tick_time", 0.0)
        inference_ms = t.get("inference_ms", 0.0)
        output = t.get("output")
        kind = output.get("kind", "?") if output else "(none)"
        has_image = "Y" if t.get("image_path") else " "
        row = (
            f"{tick_id:>6}  {tick_time:>8.2f}  {inference_ms:>9.1f}"
            f"  {kind:<18}  {has_image:>3}"
        )
        print(f"  {row}")

    outputs = [t.get("output") for t in ticks]
    final = outputs[-1]
    if final and final.get("kind") == "numerical":
        print(f"  Answer   : {final.get('value')}")
    elif final and final.get("kind") == "object_reference":
        print(f"  Object   : {final.get('label')}")

    latencies = [t.get("inference_ms", 0.0) for t in ticks]
    if latencies:
        print(
            f"  Latency  : min={min(latencies):.1f}ms  "
            f"max={max(latencies):.1f}ms  "
            f"avg={sum(latencies) / len(latencies):.1f}ms",
        )

    for t in ticks:
        output = t.get("output")
        if output and output.get("rationale"):
            print(f"    tick {t['tick_id']}: {output['rationale']}")

    if show_images:
        _show_images(session_dir / q_name, ticks)


def _show_images(q_dir: Path, ticks: list[dict]) -> None:
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError:
        print(
            "  (pillow required for --images)", file=sys.stderr,
        )
        return

    for t in ticks:
        img_path = t.get("image_path")
        if img_path:
            full_path = q_dir / img_path
            if full_path.exists():
                img = Image.open(full_path)
                img.show(title=f"tick_{t['tick_id']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a VLM logger session (text output). "
            "For HTML reports with video and charts, use: "
            "python scripts/generate_report.py <session_dir>"
        ),
    )
    parser.add_argument(
        "session_dir", type=Path, help="Path to session directory",
    )
    parser.add_argument(
        "-q", "--question", type=str, default=None,
        help="Show only questions matching this substring (e.g. '001' or 'chairs')",
    )
    parser.add_argument(
        "--images", action="store_true",
        help="Open JPEG images for each tick",
    )
    args = parser.parse_args()

    session, questions = load_session(args.session_dir)

    print_session_summary(session)

    if not questions:
        print("No questions recorded in this session.")
        return

    if args.question:
        questions = [
            (name, ticks) for name, ticks in questions
            if args.question in name
        ]
        if not questions:
            print(f"No questions matching {args.question!r}.")
            return

    total_ticks = sum(len(ticks) for _, ticks in questions)
    print(f"QUESTIONS: {len(questions)}  |  TOTAL TICKS: {total_ticks}")
    print("=" * 72)

    for q_name, ticks in questions:
        print()
        print(f"--- {q_name} ---")
        print_question_ticks(
            q_name, ticks, args.session_dir, show_images=args.images,
        )

    print()


if __name__ == "__main__":
    main()
