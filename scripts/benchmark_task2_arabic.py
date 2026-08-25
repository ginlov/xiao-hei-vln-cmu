#!/usr/bin/env python3
"""Benchmark the scene_claude / nav_task1 stack on arabic_room Type-2 questions.

For each object_reference question from the VLA-3D GT set it:
  1. restarts the SIMULATOR (stop system_simulation.sh, relaunch it) so the
     robot starts from spawn again;
  2. restarts the ai_module (fresh nav_task1 explorer + empty scene graph);
  3. publishes the question on /challenge_question;
  4. waits for the /selected_object_marker answer (bounded by --timeout);
  5. scores the answer's centre against the GT target object's position.

Results stream to a JSONL file, so a partial/interrupted run is still useful,
and a summary (hit-rate + mean centre error) prints at the end.

Assumes the scene_claude stack + sim are already up (e.g. via
`docker/run scene_claude up -d` with the arabic scene) and the Anthropic key is
in the ai_module's env.

    uv run python scripts/benchmark_task2_arabic.py --n 30 --timeout 660
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

SYSTEM_CTR = os.environ.get("XIAO_HEI_SYSTEM_CTR", "iros2026_system")
AI_CTR = os.environ.get("XIAO_HEI_AI_CTR", "xiao_hei_ai_module")
DISPLAY = os.environ.get("XIAO_HEI_SIM_DISPLAY", ":0")
GT_PATH = Path("artifacts/scene_vla3d_eval/gt/arabic_room_ref.jsonl")
ROS_ENV = "source /opt/ros/jazzy/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp"
SIM_DIR = "/home/docker/autonomy_stack_mecanum_wheel_platform"
_SIM_PROCS = [
    "system_simulation.sh", "Model.x86_64", "ros2 launch vehicle_simulator",
    "rviz2", "waypointConverter", "localPlanner", "pathFollower", "vehicleSimulator",
]


def _run(cmd: list[str], timeout: float = 60) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, "", "timeout")


def dexec(ctr: str, cmd: str, *, detach: bool = False, timeout: float = 60):
    args = ["docker", "exec"] + (["-d"] if detach else []) + [ctr, "bash", "-lc", cmd]
    return _run(args, timeout=timeout)


def ros(ctr: str, cmd: str, *, detach: bool = False, timeout: float = 60):
    return dexec(ctr, f"{ROS_ENV} && {cmd}", detach=detach, timeout=timeout)


# --- simulator lifecycle ---------------------------------------------------


def stop_sim() -> None:
    kill = "; ".join(f'pkill -f "{p}" 2>/dev/null' for p in _SIM_PROCS)
    dexec(SYSTEM_CTR, f"{kill}; sleep 3; true")


def start_sim() -> None:
    # Run the sim script directly by its absolute path, using the container's own
    # DISPLAY/XAUTHORITY (mirrors the working manual command); just detach + log.
    dexec(
        SYSTEM_CTR,
        f"{SIM_DIR}/system_simulation.sh > /tmp/sim.log 2>&1",
        detach=True,
    )


def wait_for_sim(timeout: float = 120) -> bool:
    """Block until the sim republishes /terrain_map (sensors live again)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = ros(SYSTEM_CTR, "timeout 5 ros2 topic echo --once /terrain_map "
                            ">/dev/null 2>&1 && echo OK", timeout=12)
        if "OK" in r.stdout:
            return True
        time.sleep(3)
    return False


def restart_ai(timeout: float = 90) -> bool:
    # Scope readiness to logs AFTER this restart: the container prints a periodic
    # RViz WARN, so "ready (responder=" quickly leaves any small --tail window,
    # and a large --tail could match a stale ready line from the previous run.
    since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    _run(["docker", "restart", AI_CTR], timeout=60)
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = _run(["docker", "logs", "--since", since, AI_CTR], timeout=15)
        if "ready (responder=" in (r.stdout + r.stderr):
            return True
        time.sleep(2)
    return False


# --- question / answer -----------------------------------------------------


def send_question(text: str) -> None:
    esc = text.replace('"', '\\"')
    ros(SYSTEM_CTR,
        f'ros2 topic pub --once /challenge_question std_msgs/msg/String '
        f'"{{data: \\"{esc}\\"}}"')


def wait_for_answer(timeout: float):
    dexec(SYSTEM_CTR, "rm -f /tmp/bench_ans.txt")
    # Keep stderr OUT of the answer file: a DDS hiccup (e.g. "Failed to find a
    # free participant index") must not masquerade as a marker and end the wait.
    ros(SYSTEM_CTR,
        f"timeout {int(timeout)} ros2 topic echo --once /selected_object_marker "
        f"> /tmp/bench_ans.txt 2>/dev/null", detach=True)
    deadline = time.time() + timeout + 15
    while time.time() < deadline:
        r = dexec(SYSTEM_CTR, "cat /tmp/bench_ans.txt 2>/dev/null")
        marker = parse_marker(r.stdout) if r.stdout.strip() else None
        # Only accept it once it actually parses to a marker (has an id);
        # otherwise keep waiting so partial/garbage content isn't scored.
        if marker and marker.get("id") is not None:
            return marker
        time.sleep(5)
    return None


def parse_marker(txt: str) -> dict:
    """Pull ns(label), id, and the pose.position (first x/y/z block) from echo."""
    label = oid = pos = None
    lines = [ln.rstrip() for ln in txt.splitlines()]
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("ns:") and label is None:
            label = s.split("ns:", 1)[1].strip().strip('"')
        elif s.startswith("id:") and oid is None:
            with contextlib.suppress(ValueError):
                oid = int(s.split("id:", 1)[1].strip())
        elif s.startswith("x:") and pos is None:
            try:
                x = float(s.split("x:", 1)[1])
                y = float(lines[i + 1].strip().split("y:", 1)[1])
                z = float(lines[i + 2].strip().split("z:", 1)[1])
                pos = (x, y, z)
            except (IndexError, ValueError):
                pass
    return {"label": label, "id": oid, "center": pos}


# --- scoring ---------------------------------------------------------------


def gt_target_pos(q: dict):
    tid = str(q.get("target"))
    for o in q.get("object_list", []):
        parts = o.split()
        if parts and parts[0] == tid:
            return (float(parts[1]), float(parts[2]), float(parts[3]))
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=30, help="how many questions")
    ap.add_argument("--start", type=int, default=0, help="skip the first N (resume)")
    ap.add_argument("--timeout", type=float, default=660, help="answer wait (s)")
    ap.add_argument("--hit", type=float, default=1.0, help="centre-dist (m) = hit")
    ap.add_argument("--out", default="artifacts/bench_task2_arabic.jsonl")
    args = ap.parse_args()

    qs = [json.loads(ln) for ln in GT_PATH.read_text().splitlines() if ln.strip()]
    qs = [q for q in qs if q.get("type") == "object_reference"]
    qs = qs[args.start:args.start + args.n]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    results = []
    mode = "a" if args.start else "w"
    with out.open(mode) as f:
        for i, q in enumerate(qs, args.start + 1):
            text = q["question"]
            gt = gt_target_pos(q)
            print(f"\n[{i}] {text}   (GT target #{q.get('target')} @ {gt})", flush=True)

            print("  restarting sim…", flush=True)
            stop_sim()
            start_sim()
            if not wait_for_sim():
                print("  !! sim did not come back; recording as no-sim", flush=True)
            print("  restarting ai_module…", flush=True)
            restart_ai()
            time.sleep(3)

            t0 = time.time()
            send_question(text)
            ans = wait_for_answer(args.timeout)
            dt = time.time() - t0

            dist = None
            hit = False
            if ans and ans.get("center") and gt:
                dx = ans["center"][0] - gt[0]
                dy = ans["center"][1] - gt[1]
                dist = round(math.hypot(dx, dy), 3)
                hit = dist <= args.hit
            rec = {
                "i": i, "question": text, "gt_target": q.get("target"),
                "gt_label": q.get("answer", {}).get("label"), "gt_pos": gt,
                "answer": ans, "dist_m": dist, "hit": hit, "runtime_s": round(dt, 1),
            }
            results.append(rec)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"  -> {ans}  dist={dist}m  hit={hit}  ({dt:.0f}s)", flush=True)

    answered = [r for r in results if r["dist_m"] is not None]
    hits = sum(1 for r in results if r["hit"])
    if answered:
        mean = sum(r["dist_m"] for r in answered) / len(answered)
        med = sorted(r["dist_m"] for r in answered)[len(answered) // 2]
    else:
        mean = med = float("nan")
    print(f"\n=== SUMMARY: {hits}/{len(results)} hits @ {args.hit} m | "
          f"{len(answered)} answered | mean {mean:.2f} m | median {med:.2f} m ===",
          flush=True)
    print(f"results: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
