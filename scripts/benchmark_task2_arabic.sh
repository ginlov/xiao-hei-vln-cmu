#!/usr/bin/env bash
# Benchmark the scene_claude / nav_task1 stack on arabic_room Type-2
# (object_reference) questions. Run this from YOUR OWN terminal session, where
# the X grant (xhost) and DISPLAY are set, because each question restarts the
# Unity simulator (which needs the X display).
#
# Per question it:
#   1. restarts the SIMULATOR (kill system_simulation.sh, relaunch it),
#   2. restarts the ai_module (fresh nav_task1 explorer + empty scene graph),
#   3. publishes the question,
#   4. waits for the /selected_object_marker answer (bounded by TIMEOUT),
#   5. scores the answer centre vs the GT target object.
# Results stream to artifacts/bench_task2_arabic.jsonl; a summary prints at end.
#
# Prereqs: the scene_claude stack + arabic sim already up, Anthropic key in the
# ai_module env, and `xhost +local:` granted on this host.
#
# Usage:
#   scripts/benchmark_task2_arabic.sh [N] [TIMEOUT_S]
#   scripts/benchmark_task2_arabic.sh 30 660
#   START=10 scripts/benchmark_task2_arabic.sh 20      # resume from #11 (appends)
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

SYSTEM_CTR=${XIAO_HEI_SYSTEM_CTR:-iros2026_system}
AI_CTR=${XIAO_HEI_AI_CTR:-xiao_hei_ai_module}
N=${1:-30}
TIMEOUT=${2:-660}
HIT=${HIT:-1.0}
START=${START:-0}
GT=artifacts/scene_vla3d_eval/gt/arabic_room_ref.jsonl
OUT=artifacts/bench_task2_arabic.jsonl
ROS='source /opt/ros/jazzy/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp'
SIM_DIR=/home/docker/autonomy_stack_mecanum_wheel_platform

command -v docker >/dev/null || { echo "docker not found" >&2; exit 1; }
[ -f "$GT" ] || { echo "GT not found: $GT" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"
[ "$START" -eq 0 ] && : > "$OUT"

# Extract questions -> TSV: idx \t question \t gt_x \t gt_y \t target_id
python3 - "$GT" "$N" "$START" > /tmp/bench_qs.tsv <<'PY'
import sys, json
gt, n, start = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
qs = [json.loads(l) for l in open(gt) if l.strip()]
qs = [q for q in qs if q.get("type") == "object_reference"][start:start + n]
def pos(q):
    tid = str(q.get("target"))
    for o in q.get("object_list", []):
        p = o.split()
        if p and p[0] == tid:
            return p[1], p[2]
    return "", ""
for i, q in enumerate(qs, start + 1):
    x, y = pos(q)
    print("\t".join([str(i), q["question"], x, y, str(q.get("target"))]))
PY

start_sim() {
  # Exactly the working manual command — the sim's own launcher auto-starts the
  # whole autonomy stack (rviz, planners, waypointConverter). Just detached +
  # logged so the loop can proceed. Do NOT hand-manage its child nodes.
  docker exec -d "$SYSTEM_CTR" bash -lc \
    "$SIM_DIR/system_simulation.sh > /tmp/sim.log 2>&1"
}
stop_sim() {
  # Clean Ctrl-C: SIGINT the launcher's process group so ros2 launch tears the
  # whole stack down in order (rviz included). No per-node pkill — that left the
  # autonomy half-dismantled (e.g. waypointConverter gone → robot can't move).
  docker exec "$SYSTEM_CTR" bash -lc '
    for pid in $(pgrep -f "system_simulation.sh|launch .*vehicle_simulator"); do
      pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d " ")
      [ -n "$pgid" ] && kill -INT -"$pgid" 2>/dev/null
    done
    for _ in $(seq 1 25); do
      pgrep -f "Model.x86_64|vehicleSimulato|localPlanner|rviz2" >/dev/null 2>&1 || break
      sleep 1
    done
    true' >/dev/null 2>&1
}
# Healthy = the autonomy chain that moves the robot is actually up. Read-only:
# waypointConverter (the node that turns our /way_point_with_heading into the
# planner's /way_point) must be in the ROS graph — its absence is exactly the
# "robot frozen at spawn" failure. No process-counting (pgrep -f double-counts).
nav_healthy() {
  docker exec "$SYSTEM_CTR" bash -lc \
    "$ROS && timeout 8 ros2 node list 2>/dev/null | grep -q waypointConverter" || return 1
  return 0
}
wait_sim() {
  for _ in $(seq 1 40); do
    docker exec "$SYSTEM_CTR" bash -lc \
      "$ROS && timeout 5 ros2 topic echo --once /terrain_map >/dev/null 2>&1" && return 0
    sleep 3
  done
  return 1
}
restart_ai() {
  # Scope the readiness check to logs emitted AFTER this restart: the container
  # prints a periodic RViz WARN every ~30s, so "ready (responder=" quickly falls
  # out of any small --tail window, and a large --tail could match a stale ready
  # line from the previous run. `--since` avoids both.
  local since
  since=$(date -u +%Y-%m-%dT%H:%M:%S)
  docker restart "$AI_CTR" >/dev/null 2>&1
  for _ in $(seq 1 40); do
    docker logs --since "$since" "$AI_CTR" 2>&1 | grep -q "ready (responder=" && return 0
    sleep 2
  done
  return 1
}

while IFS=$'\t' read -r idx q gx gy tid; do
  echo "[$idx] $q   (GT #$tid @ $gx,$gy)"
  # Bring up ONE healthy sim; retry the whole restart if it came up unhealthy
  # (duplicate stack / dead waypoint chain) rather than sending a doomed question.
  for attempt in 1 2 3; do
    echo "  restarting sim… (attempt $attempt)"; stop_sim; start_sim
    wait_sim || echo "  !! sim slow to publish /terrain_map"
    if nav_healthy; then echo "  sim healthy (autonomy live)"; break; fi
    echo "  !! sim unhealthy (dup stack / no cmd_vel) — retrying restart"
  done
  echo "  restarting ai_module…"; restart_ai; sleep 3
  t0=$SECONDS
  esc=${q//\"/\\\"}
  docker exec "$SYSTEM_CTR" bash -lc \
    "$ROS && ros2 topic pub --once /challenge_question std_msgs/msg/String \"{data: \\\"$esc\\\"}\"" \
    >/dev/null 2>&1
  docker exec "$SYSTEM_CTR" bash -lc 'rm -f /tmp/bench_ans.txt' >/dev/null 2>&1
  # Keep stderr OUT of the answer file: a DDS hiccup (e.g. "Failed to find a
  # free participant index") must not masquerade as a marker and end the wait.
  docker exec -d "$SYSTEM_CTR" bash -lc \
    "$ROS && timeout $TIMEOUT ros2 topic echo --once /selected_object_marker > /tmp/bench_ans.txt 2>/dev/null"
  ans=""
  for _ in $(seq 1 $(((TIMEOUT + 15) / 5))); do
    ans=$(docker exec "$SYSTEM_CTR" bash -lc 'cat /tmp/bench_ans.txt 2>/dev/null')
    # Only accept it once it actually looks like a marker (has a numeric id:);
    # otherwise keep waiting so partial/garbage content doesn't score as wrong.
    printf '%s' "$ans" | grep -Eq '^[[:space:]]*id:[[:space:]]*[0-9-]+' && break
    ans=""
    sleep 5
  done
  dt=$((SECONDS - t0))
  # Pass the answer via a FILE, not stdin: `python3 -` reads its program from
  # stdin, so piping "$ans" into it collides with the heredoc script (marker
  # text prepended → SyntaxError, or empty → every field parses as None). The
  # script now reads the marker from argv[1].
  printf '%s' "$ans" > /tmp/bench_ans_host.txt
  python3 - /tmp/bench_ans_host.txt "$q" "$gx" "$gy" "$tid" "$dt" "$HIT" "$OUT" "$idx" <<'PY'
import sys, json, math
txt = open(sys.argv[1], encoding="utf-8", errors="replace").read()
q, gx, gy, tid, dt, hit, out, idx = sys.argv[2:10]
label = oid = None
pos = None
lines = [l.rstrip() for l in txt.splitlines()]
for i, l in enumerate(lines):
    s = l.strip()
    if s.startswith("ns:") and label is None:
        label = s.split("ns:", 1)[1].strip().strip('"')
    elif s.startswith("id:") and oid is None:
        try:
            oid = int(s.split("id:", 1)[1].strip())
        except ValueError:
            pass
    elif s.startswith("x:") and pos is None:
        try:
            pos = (float(s.split("x:", 1)[1]),
                   float(lines[i + 1].strip().split("y:", 1)[1]))
        except (IndexError, ValueError):
            pass
dist = None
ishit = False
if pos and gx and gy:
    dist = round(math.hypot(pos[0] - float(gx), pos[1] - float(gy)), 3)
    ishit = dist <= float(hit)
rec = {"i": int(idx), "question": q, "gt_target": tid, "gt_pos": [gx, gy],
       "answer": {"label": label, "id": oid, "center": pos},
       "dist_m": dist, "hit": ishit, "runtime_s": int(dt)}
open(out, "a").write(json.dumps(rec) + "\n")
print(f"  -> {label} #{oid} @ {pos}  dist={dist}m  hit={ishit}  ({dt}s)")
PY
done < /tmp/bench_qs.tsv

python3 - "$OUT" "$HIT" <<'PY'
import sys, json, statistics as st
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
ans = [r for r in recs if r["dist_m"] is not None]
hits = sum(1 for r in recs if r["hit"])
m = st.mean(r["dist_m"] for r in ans) if ans else float("nan")
md = st.median(r["dist_m"] for r in ans) if ans else float("nan")
print(f"\n=== SUMMARY: {hits}/{len(recs)} hits @ {sys.argv[2]}m | "
      f"{len(ans)} answered | mean {m:.2f}m | median {md:.2f}m ===")
PY
