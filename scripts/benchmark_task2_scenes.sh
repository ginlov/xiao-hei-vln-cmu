#!/usr/bin/env bash
# Benchmark the scene_claude / nav_task1 stack across ALL 15 VLA-3D scenes,
# a few random object_reference questions per scene.
#
# Run this from YOUR OWN terminal session (X grant needed): each scene RECREATES
# the sim container with that scene's Unity mesh bind-mounted, and each question
# relaunches the Unity simulator — both need the host X display and the xhost
# grant. `xhost +local:` must be in effect.
#
# Per scene it:
#   1. brings the stack up with that scene mounted
#      (XIAO_HEI_SCENE_DIR_HOST=<scene> docker/run scene_claude up -d), then
#      waits for the perception sidecar to go green;
#   2. picks Q_PER_SCENE random object_reference questions (seeded → reproducible);
#   3. per question: restarts the sim + ai_module, publishes the question, waits
#      for the /selected_object_marker answer, and scores its centre vs GT.
# Results stream to artifacts/bench_task2_scenes.jsonl; a per-scene + overall
# summary prints at the end.
#
# Usage:
#   scripts/benchmark_task2_scenes.sh [Q_PER_SCENE] [TIMEOUT_S]
#   scripts/benchmark_task2_scenes.sh 3 660
#   SEED=7 scripts/benchmark_task2_scenes.sh 3            # different random pick
#   SCENES="arabic_room studio" scripts/benchmark_task2_scenes.sh 3   # subset
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

SYSTEM_CTR=${XIAO_HEI_SYSTEM_CTR:-iros2026_system}
AI_CTR=${XIAO_HEI_AI_CTR:-xiao_hei_ai_module}
PERC_URL=${XIAO_HEI_PERCEPTION_HEALTH_URL:-http://localhost:8001/healthz}
Q_PER_SCENE=${1:-3}
TIMEOUT=${2:-660}
HIT=${HIT:-1.0}
SEED=${SEED:-0}
SCENES_DIR=${XIAO_HEI_SCENES_DIR:-/home/long/Projects/dataset/unity_scenes_extracted}
REF=${XIAO_HEI_REF_JSONL:-/home/long/Projects/dataset/xiao-hei-vln-cmu/dataset/vla3d_ref.jsonl}
OUT=${OUT:-artifacts/bench_task2_scenes.jsonl}
ROS='source /opt/ros/jazzy/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp'
SIM_DIR=/home/docker/autonomy_stack_mecanum_wheel_platform

# All 15 scenes by default (every dir under SCENES_DIR); SCENES=... to subset.
SCENES=${SCENES:-$(ls -d "$SCENES_DIR"/*/ 2>/dev/null | xargs -n1 basename | tr '\n' ' ')}

command -v docker >/dev/null || { echo "docker not found" >&2; exit 1; }
[ -f "$REF" ] || { echo "ref jsonl not found: $REF" >&2; exit 1; }
[ -d "$SCENES_DIR" ] || { echo "scenes dir not found: $SCENES_DIR" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"
: > "$OUT"

# --- sim lifecycle (identical to the single-scene benchmark's FIXED versions) -
start_sim() {
  # The sim's own launcher auto-starts the whole autonomy stack (rviz, planners,
  # waypointConverter). Just detached + logged. Do NOT hand-manage child nodes.
  docker exec -d "$SYSTEM_CTR" bash -lc \
    "$SIM_DIR/system_simulation.sh > /tmp/sim.log 2>&1"
}
stop_sim() {
  # Clean Ctrl-C: SIGINT the launcher's process group so ros2 launch tears the
  # whole stack down in order. No per-node pkill (that dismantled autonomy).
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
wait_sim() {
  for _ in $(seq 1 40); do
    docker exec "$SYSTEM_CTR" bash -lc \
      "$ROS && timeout 5 ros2 topic echo --once /terrain_map >/dev/null 2>&1" && return 0
    sleep 3
  done
  return 1
}
# Read-only: the node that turns our /way_point_with_heading into the planner's
# /way_point must be in the graph — its absence is the "robot frozen at spawn".
nav_healthy() {
  docker exec "$SYSTEM_CTR" bash -lc \
    "$ROS && timeout 8 ros2 node list 2>/dev/null | grep -q waypointConverter" || return 1
  return 0
}
restart_ai() {
  # Scope readiness to logs AFTER this restart (periodic WARNs push the ready
  # line out of any small --tail; a big --tail could match a stale one).
  local since; since=$(date -u +%Y-%m-%dT%H:%M:%S)
  docker restart "$AI_CTR" >/dev/null 2>&1
  for _ in $(seq 1 60); do
    docker logs --since "$since" "$AI_CTR" 2>&1 | grep -q "ready (responder=" && return 0
    sleep 2
  done
  return 1
}

# --- per-scene stack bring-up -------------------------------------------------
bring_up_scene() {
  # Recreate the sim container with this scene's mesh bind-mounted. Only the
  # `system` service's volumes change, so compose recreates just the sim; the
  # perception + ai_module stay up (scene-agnostic). Then wait for perception.
  local scene="$1"
  export XIAO_HEI_SCENE_DIR_HOST="$SCENES_DIR/$scene"
  echo "  bringing up stack with scene '$scene' mounted…"
  ./docker/run scene_claude up -d >/tmp/bench_up.log 2>&1 || {
    echo "  !! 'docker/run scene_claude up -d' failed — see /tmp/bench_up.log" >&2
    tail -5 /tmp/bench_up.log >&2; return 1; }
  # perception sidecar green (host network → reachable from here). Model load
  # can take a while on a cold start.
  for _ in $(seq 1 90); do
    curl -sf "$PERC_URL" >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "  !! perception sidecar not green at $PERC_URL" >&2
  return 1
}

# --- pick N random object_reference questions for a scene -> TSV --------------
pick_questions() {  # arg: scene ; prints idx \t question \t gt_x \t gt_y \t target
  python3 - "$REF" "$1" "$Q_PER_SCENE" "$SEED" <<'PY'
import sys, json, random
ref, scene, n, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
qs = [json.loads(l) for l in open(ref) if l.strip()]
qs = [q for q in qs if q.get("scene") == scene and q.get("type") == "object_reference"]
# Seed is (seed, scene) so the pick is reproducible AND independent per scene.
random.Random(f"{seed}:{scene}").shuffle(qs)
qs = qs[:n]
def pos(q):
    tid = str(q.get("target"))
    for o in q.get("object_list", []):
        p = o.split()
        if p and p[0] == tid:
            return p[1], p[2]
    return "", ""
for i, q in enumerate(qs, 1):
    x, y = pos(q)
    print("\t".join([str(i), q["question"], x, y, str(q.get("target"))]))
PY
}

# --- main loop ----------------------------------------------------------------
for scene in $SCENES; do
  echo "================ SCENE: $scene ================"
  bring_up_scene "$scene" || { echo "  skipping scene $scene"; continue; }
  pick_questions "$scene" > /tmp/bench_qs.tsv
  [ -s /tmp/bench_qs.tsv ] || { echo "  no questions for $scene, skipping"; continue; }

  while IFS=$'\t' read -r idx q gx gy tid; do
    echo "[$scene #$idx] $q   (GT #$tid @ $gx,$gy)"
    # Bring up ONE healthy sim; retry the restart if it came up unhealthy.
    for attempt in 1 2 3; do
      echo "  restarting sim… (attempt $attempt)"; stop_sim; start_sim
      wait_sim || echo "  !! sim slow to publish /terrain_map"
      if nav_healthy; then echo "  sim healthy (autonomy live)"; break; fi
      echo "  !! sim unhealthy (no waypointConverter) — retrying restart"
    done
    echo "  restarting ai_module…"; restart_ai; sleep 3
    t0=$SECONDS
    esc=${q//\"/\\\"}
    docker exec "$SYSTEM_CTR" bash -lc \
      "$ROS && ros2 topic pub --once /challenge_question std_msgs/msg/String \"{data: \\\"$esc\\\"}\"" \
      >/dev/null 2>&1
    docker exec "$SYSTEM_CTR" bash -lc 'rm -f /tmp/bench_ans.txt' >/dev/null 2>&1
    # stderr OUT of the answer file: a DDS hiccup must not masquerade as a marker.
    docker exec -d "$SYSTEM_CTR" bash -lc \
      "$ROS && timeout $TIMEOUT ros2 topic echo --once /selected_object_marker > /tmp/bench_ans.txt 2>/dev/null"
    ans=""
    for _ in $(seq 1 $(((TIMEOUT + 15) / 5))); do
      ans=$(docker exec "$SYSTEM_CTR" bash -lc 'cat /tmp/bench_ans.txt 2>/dev/null')
      # Accept only once it actually looks like a marker (numeric id:).
      printf '%s' "$ans" | grep -Eq '^[[:space:]]*id:[[:space:]]*[0-9-]+' && break
      ans=""
      sleep 5
    done
    dt=$((SECONDS - t0))
    # Answer via a FILE, not stdin: `python3 -` reads its program from stdin, so
    # piping "$ans" collides with the heredoc script.
    printf '%s' "$ans" > /tmp/bench_ans_host.txt
    python3 - /tmp/bench_ans_host.txt "$scene" "$q" "$gx" "$gy" "$tid" "$dt" "$HIT" "$OUT" "$idx" <<'PY'
import sys, json, math
txt = open(sys.argv[1], encoding="utf-8", errors="replace").read()
scene, q, gx, gy, tid, dt, hit, out, idx = sys.argv[2:11]
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
rec = {"scene": scene, "i": int(idx), "question": q, "gt_target": tid,
       "gt_pos": [gx, gy], "answer": {"label": label, "id": oid, "center": pos},
       "dist_m": dist, "hit": ishit, "runtime_s": int(dt)}
open(out, "a").write(json.dumps(rec) + "\n")
print(f"  -> {label} #{oid} @ {pos}  dist={dist}m  hit={ishit}  ({dt}s)")
PY
  done < /tmp/bench_qs.tsv
done

# --- summary: per scene + overall ---------------------------------------------
python3 - "$OUT" "$HIT" <<'PY'
import sys, json, statistics as st, collections
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
hitm = float(sys.argv[2])
byscene = collections.defaultdict(list)
for r in recs:
    byscene[r["scene"]].append(r)
print("\n=== PER-SCENE ===")
for sc in sorted(byscene):
    rs = byscene[sc]
    ans = [r for r in rs if r["dist_m"] is not None]
    hits = sum(1 for r in rs if r["hit"])
    med = st.median(r["dist_m"] for r in ans) if ans else float("nan")
    print(f"  {sc:16s} {hits}/{len(rs)} hits | {len(ans)} answered | median {med:.2f} m")
ans = [r for r in recs if r["dist_m"] is not None]
hits = sum(1 for r in recs if r["hit"])
m = st.mean(r["dist_m"] for r in ans) if ans else float("nan")
md = st.median(r["dist_m"] for r in ans) if ans else float("nan")
print(f"\n=== OVERALL: {hits}/{len(recs)} hits @ {hitm}m | {len(ans)} answered | "
      f"mean {m:.2f}m | median {md:.2f}m ===")
PY
