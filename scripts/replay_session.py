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
from collections import defaultdict
from pathlib import Path


def load_session(session_dir: Path) -> tuple[dict, list[dict]]:
    session_json = session_dir / "session.json"
    if not session_json.exists():
        print(f"Error: {session_json} not found", file=sys.stderr)
        sys.exit(1)

    session = json.loads(session_json.read_text())

    ticks_jsonl = session_dir / "ticks.jsonl"
    if not ticks_jsonl.exists():
        print(f"Error: {ticks_jsonl} not found", file=sys.stderr)
        sys.exit(1)

    ticks = []
    for line in ticks_jsonl.read_text().strip().splitlines():
        if line:
            ticks.append(json.loads(line))

    return session, ticks


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


def print_tick_table(ticks: list[dict]) -> None:
    if not ticks:
        print("  (no ticks recorded)")
        return

    cols = [
        f"{'tick':>6}", f"{'time_s':>8}", f"{'infer_ms':>9}",
        f"{'output_kind':<18}", f"{'img':>3}", "question",
    ]
    header = "  ".join(cols)
    print(header)
    print("-" * len(header))

    for t in ticks:
        tick_id = t.get("tick_id", "?")
        tick_time = t.get("tick_time", 0.0)
        inference_ms = t.get("inference_ms", 0.0)
        output = t.get("output")
        kind = output.get("kind", "?") if output else "(none)"
        has_image = "Y" if t.get("image_path") else " "
        question = t.get("question_text", "")
        if question and len(question) > 40:
            question = question[:37] + "..."
        row = (
            f"{tick_id:>6}  {tick_time:>8.2f}  {inference_ms:>9.1f}"
            f"  {kind:<18}  {has_image:>3}  {question}"
        )
        print(row)


def print_grouped_by_question(ticks: list[dict]) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for t in ticks:
        q = t.get("question_text", "(no question)")
        groups[q].append(t)

    for question, group_ticks in groups.items():
        print()
        print(f"  Question: {question!r} ({len(group_ticks)} ticks)")
        print(f"  Type    : {group_ticks[0].get('question_type', '?')}")

        outputs = [t.get("output") for t in group_ticks]
        kinds = [o.get("kind", "?") if o else "(none)" for o in outputs]
        final = outputs[-1]

        print(f"  Outputs : {' → '.join(kinds)}")
        if final and final.get("kind") == "numerical":
            print(f"  Answer  : {final.get('value')}")
        elif final and final.get("kind") == "object_reference":
            print(f"  Object  : {final.get('label')}")

        latencies = [t.get("inference_ms", 0.0) for t in group_ticks]
        if latencies:
            print(
                f"  Latency : min={min(latencies):.1f}ms  "
                f"max={max(latencies):.1f}ms  "
                f"avg={sum(latencies) / len(latencies):.1f}ms",
            )

        for t in group_ticks:
            output = t.get("output")
            if output and output.get("rationale"):
                print(f"    tick {t['tick_id']}: {output['rationale']}")


def show_images(session_dir: Path, ticks: list[dict]) -> None:
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError:
        print("Error: pillow required for --images (pip install pillow)", file=sys.stderr)
        return

    for t in ticks:
        img_path = t.get("image_path")
        if img_path:
            full_path = session_dir / img_path
            if full_path.exists():
                img = Image.open(full_path)
                img.show(title=f"tick_{t['tick_id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a VLM logger session")
    parser.add_argument("session_dir", type=Path, help="Path to session directory")
    parser.add_argument(
        "--images",
        action="store_true",
        help="Open JPEG images for each tick",
    )
    args = parser.parse_args()

    session, ticks = load_session(args.session_dir)

    print_session_summary(session)

    print("TICK TABLE")
    print("-" * 72)
    print_tick_table(ticks)

    print()
    print("GROUPED BY QUESTION")
    print("-" * 72)
    print_grouped_by_question(ticks)
    print()

    if args.images:
        show_images(args.session_dir, ticks)


if __name__ == "__main__":
    main()
