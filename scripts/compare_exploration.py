#!/usr/bin/env python3
"""Compare exploration sweeps from their exploration.log files.

    uv run python scripts/compare_exploration.py exploration_logs

Reads `<root>/<scene>/<strategy>/exploration.log`, and prints a per-strategy
table plus a paired per-scene comparison over the scenes every strategy ran.

Normalises by wall clock. Two sweeps run under different budgets are not
comparable on totals — that confound is what made the first frontier-vs-nbv
comparison hard to read (TASK 43) — so the headline number here is metres per
minute, measured from CLOCK_START rather than node boot.

Also reads the diagnostic fields added for the second sweep, when present:
skip `kind`, `best_odom_dist`, and the MAP heartbeat's free/reachable counters.
Logs without them still parse; the derived columns just read `-`.
"""

from __future__ import annotations

import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_LINE = re.compile(r"\[([\d.]+)\]\s+(\w+)\s*(.*)")
_KV = re.compile(r"(\w+)=(\S+)")
_PT = re.compile(r"\(([-\d.]+),([-\d.]+)\)")


def _num(s: str | None) -> float | None:
    """Parse a log value, treating 'inf' and junk as missing."""
    if s is None:
        return None
    try:
        v = float(s.rstrip("s").rstrip("m"))
    except ValueError:
        return None
    return None if math.isinf(v) or math.isnan(v) else v


def _pt(s: str | None) -> tuple[float, float] | None:
    m = _PT.match(s or "")
    return (float(m.group(1)), float(m.group(2))) if m else None


def parse_log(path: Path) -> dict | None:
    events = []
    for line in path.read_text(errors="replace").splitlines():
        m = _LINE.match(line.strip())
        if m:
            events.append((float(m.group(1)), m.group(2), dict(_KV.findall(m.group(3)))))
    if not events:
        return None

    t_start = t_clock = t_end = None
    poses: list[tuple[float, float]] = []
    done: dict[str, str] = {}
    maps: list[dict[str, str]] = []
    skips: list[dict[str, str]] = []
    advances = 0
    targets: list[tuple[float, float] | None] = []

    for t, kind, kv in events:
        t_end = t
        if kind == "START":
            t_start = t
        elif kind == "CLOCK_START":
            t_clock = t
        elif kind == "DONE":
            done = kv
        elif kind == "MAP":
            maps.append(kv)
        elif kind == "WP_ADVANCE":
            advances += 1
        elif kind == "WP_SKIP":
            skips.append(kv)
        if (p := _pt(kv.get("robot"))) is not None:
            poses.append(p)
        if kind == "WP_SET":
            targets.append(_pt(kv.get("target")))

    t0 = t_clock if t_clock is not None else t_start
    dur = (t_end - t0) if (t0 is not None and t_end is not None) else 0.0

    # path_m from the log when the node recorded it; otherwise reconstruct from
    # the sparse WP_* poses, which undercounts (say so by flagging it).
    path = _num(done.get("path_m"))
    if path is None and maps:
        path = _num(maps[-1].get("path_m"))
    exact = path is not None
    if path is None:
        path = sum(math.dist(poses[i], poses[i + 1]) for i in range(len(poses) - 1))

    repeats = sum(
        1 for i in range(1, len(targets)) if targets[i] is not None and targets[i] == targets[i - 1]
    )
    # A skip is "premature" if the robot was still closing on the goal.
    closing = sum(1 for s in skips if (_num(s.get("best_odom_dist")) or 0) > 0 and s.get("kind"))
    by_kind: dict[str, int] = defaultdict(int)
    for s in skips:
        by_kind[s.get("kind", "unknown")] += 1
    silent_nav = sum(1 for s in skips if s.get("best_nav_dist") == "inf")

    last_map = maps[-1] if maps else {}
    free_m2 = _num(done.get("free_m2")) or _num(last_map.get("free_m2"))
    free = _num(done.get("free")) or _num(last_map.get("free"))
    reach = _num(done.get("reachable")) or _num(last_map.get("reachable"))

    return {
        # A log with a START and nothing else is a run that died before it
        # explored. Scoring it 0.00 m/min would count it against the strategy;
        # it is missing data, so it is excluded from means and pairing.
        "no_data": dur <= 0 or (not skips and advances == 0),
        "dur": dur,
        "path": path,
        "path_exact": exact,
        "visited": int(_num(done.get("visited")) or advances),
        "skipped": int(_num(done.get("skipped")) or len(skips)),
        "silent_nav": silent_nav,
        "repeats": repeats,
        "closing": closing,
        "by_kind": dict(by_kind),
        "free_m2": free_m2,
        "reach_frac": (reach / free) if (free and reach is not None) else None,
        "hatch": int(_num(done.get("hatch_resets")) or 0),
        "reason": done.get("reason", "did-not-finish"),
        "select_why": done.get("select_why", "-"),
        "maps": len(maps),
    }


def _fmt(v, spec=".1f", dash="-"):
    return dash if v is None else format(v, spec)


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else "exploration_logs")
    if not root.is_dir():
        print(f"no such directory: {root}", file=sys.stderr)
        return 1

    runs: dict[str, dict[str, dict]] = defaultdict(dict)
    for log in sorted(root.glob("*/*/exploration.log")):
        scene, strategy = log.parent.parent.name, log.parent.name
        if (r := parse_log(log)) is not None:
            runs[strategy][scene] = r
    # Flat layout (<root>/<scene>/exploration.log) — the pre-TASK-41 sweeps.
    for log in sorted(root.glob("*/exploration.log")):
        if (r := parse_log(log)) is not None:
            runs.setdefault("(unlabelled)", {}).setdefault(log.parent.name, r)

    if not runs:
        print(f"no exploration.log found under {root}", file=sys.stderr)
        return 1

    for strategy, scenes in sorted(runs.items()):
        print(f"\n===== {strategy}  ({len(scenes)} scenes) =====")
        print(f"{'scene':<34}{'dur_s':>7}{'m/min':>7}{'path_m':>8}{'free_m2':>9}"
              f"{'reach%':>8}{'vis':>5}{'skip':>6}{'silent':>7}{'rep':>5}  reason")
        for scene, r in sorted(scenes.items()):
            if r["no_data"]:
                print(f"{scene:<34}{'—':>7}{'—':>7}{'—':>8}{'—':>9}{'—':>8}"
                      f"{'—':>5}{'—':>6}{'—':>7}{'—':>5}  {r['reason']} (no data)")
                continue
            rate = r["path"] / r["dur"] * 60
            print(f"{scene:<34}{r['dur']:>7.0f}{rate:>7.2f}"
                  f"{r['path']:>8.1f}{'' if r['path_exact'] else '~'}"
                  f"{_fmt(r['free_m2']):>{9 if r['path_exact'] else 8}}"
                  f"{_fmt(r['reach_frac'] and r['reach_frac'] * 100, '.0f'):>8}"
                  f"{r['visited']:>5}{r['skipped']:>6}{r['silent_nav']:>7}{r['repeats']:>5}"
                  f"  {r['reason']}")
        scored = [r for r in scenes.values() if not r["no_data"]]
        if scored:
            print(f"{'MEAN':<34}"
                  f"{statistics.mean(r['dur'] for r in scored):>7.0f}"
                  f"{statistics.mean(r['path'] / r['dur'] * 60 for r in scored):>7.2f}"
                  f"{statistics.mean(r['path'] for r in scored):>8.1f}"
                  f"   (over {len(scored)} of {len(scenes)} scenes)")
        kinds: dict[str, int] = defaultdict(int)
        for r in scenes.values():
            for k, n in r["by_kind"].items():
                kinds[k] += n
        if set(kinds) - {"unknown"}:
            total = sum(kinds.values())
            breakdown = "  ".join(f"{k}={n} ({100 * n / total:.0f}%)" for k, n in sorted(kinds.items()))
            print(f"  skip kinds: {breakdown}")
        if not any(r["maps"] for r in scenes.values()):
            print("  (no MAP heartbeats — coverage columns are reconstructed, not measured)")

    if len(runs) > 1:
        names = sorted(runs)
        # Pair only where every strategy produced a usable run, so one crashed
        # container cannot decide a head-to-head.
        common = sorted(
            set.intersection(
                *({s for s, r in runs[n].items() if not r["no_data"]} for n in names)
            )
        )
        print(f"\n===== paired on {len(common)} common scenes =====")
        if not common:
            print("  (no scene produced a usable run under every strategy)")
            return 0
        print(f"{'scene':<34}" + "".join(f"{n[:11]:>12}" for n in names) + "   (metres/min)")
        wins: dict[str, int] = defaultdict(int)
        for scene in common:
            rates = {n: runs[n][scene]["path"] / runs[n][scene]["dur"] * 60 for n in names}
            wins[max(rates, key=lambda k: rates[k])] += 1
            print(f"{scene:<34}" + "".join(f"{rates[n]:>12.2f}" for n in names))
        print(f"{'MEAN':<34}" + "".join(
            f"{statistics.mean(runs[n][s]['path'] / runs[n][s]['dur'] * 60 for s in common):>12.2f}"
            for n in names))
        print("\n  scenes won: " + "  ".join(f"{n}={wins[n]}" for n in names))
        skipped = {n: sorted(s for s, r in runs[n].items() if r["no_data"]) for n in names}
        for n, ss in skipped.items():
            if ss:
                print(f"  excluded from {n} (no data): {', '.join(ss)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
