#!/usr/bin/env bash
# Per-scene: live explore → dump scene graph → Gemini batch offline on VLA-3D Qs.
#
# Usage:
#   export XIAO_HEI_GEMINI_API_KEY=<key>
#   scripts/run_scene_vla3d_eval.sh --num-scenes 1
#   scripts/run_scene_vla3d_eval.sh --num-scenes 3 --limit 20
#   scripts/run_scene_vla3d_eval.sh livingroom_3 chinese_room
#   scripts/run_scene_vla3d_eval.sh --skip-explore --num-scenes 2   # reuse dumps
#   scripts/run_scene_vla3d_eval.sh --gt-only --limit 5 studio      # GT upper bound
#
# Env (optional):
#   SCENES_DIR   Unity scenes root (default: ~/Downloads/unity_env_models)
#   GT_DIR       VLA-3D JSONL dir (auto-detected, or set explicitly)
#   OUT_DIR      artefacts root (default: <repo>/artifacts/scene_vla3d_eval)
#   MAX_WAYPOINTS / MAX_SECONDS / STRATEGY / TIMEOUT / SPLITS / LIMIT_Q
#   COMPOSE_PROJECT / XIAO_HEI_EVAL_PREFIX / ROS_DOMAIN_ID / PERCEPTION_PORT
#     — isolated eval stack (see docker/compose.eval.yml)

set -uo pipefail

REPO=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
SCENES_DIR=${SCENES_DIR:-$HOME/Downloads/unity_env_models}
OUT_DIR=${OUT_DIR:-$REPO/artifacts/scene_vla3d_eval}
MAX_WAYPOINTS=${MAX_WAYPOINTS:-100}
MAX_SECONDS=${MAX_SECONDS:-540}
STRATEGY=${STRATEGY:-nbv}
TIMEOUT=${TIMEOUT:-$(( MAX_SECONDS + 600 ))}   # explore cap + sim startup slack
SPLITS=${SPLITS:-ref,num}                      # comma-separated: ref and/or num
LIMIT_Q=${LIMIT_Q:-}                           # keep first N GT Qs per scene/split
SKIP_EXPLORE=0
GT_ONLY=0
NUM_SCENES=""
DECLARED_SCENES=()

# Isolated eval stack (does not clobber a default compose_scene_gemini stack).
COMPOSE_PROJECT=${COMPOSE_PROJECT:-xiao_hei_eval}
XIAO_HEI_EVAL_PREFIX=${XIAO_HEI_EVAL_PREFIX:-xiao_hei_eval}
ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
PERCEPTION_PORT=${PERCEPTION_PORT:-8002}
SYSTEM_CTR=${SYSTEM_CTR:-${XIAO_HEI_EVAL_PREFIX}_iros2026_system}
AI_CTR=${AI_CTR:-${XIAO_HEI_EVAL_PREFIX}_ai_module}
DISPLAY_VAL=${DISPLAY:-:0}
export DISPLAY="$DISPLAY_VAL"
export XIAO_HEI_EVAL_PREFIX ROS_DOMAIN_ID PERCEPTION_PORT
COMPOSE=(docker compose
  -p "$COMPOSE_PROJECT"
  -f "$REPO/docker/compose_scene_gemini.yml"
  -f "$REPO/docker/compose.scene.yml"
  -f "$REPO/docker/compose.eval.yml")

# Resolve GT_DIR: explicit env wins, else first existing candidate.
if [[ -z "${GT_DIR:-}" ]]; then
  for _cand in \
    "${XIAO_HEI_GT_DIR:-}" \
    "$REPO/../dataset/xiao-hei-vln-cmu/dataset" \
    "$HOME/workspace/dataset/xiao-hei-vln-cmu/dataset" \
    "/home/ubuntu/workspace/dataset/xiao-hei-vln-cmu/dataset"
  do
    [[ -n "$_cand" && -f "$_cand/vla3d_ref.jsonl" ]] && GT_DIR=$_cand && break
  done
fi
GT_DIR=${GT_DIR:-}

usage() {
  sed -n '2,17p' "$0" | sed 's/^# \?//'
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-scenes) NUM_SCENES=$2; shift 2 ;;
    --limit)      LIMIT_Q=$2; shift 2 ;;
    --skip-explore) SKIP_EXPLORE=1; shift ;;
    --gt-only)    GT_ONLY=1; SKIP_EXPLORE=1; shift ;;
    --splits)     SPLITS=$2; shift 2 ;;
    --help|-h)    usage ;;
    --*)          echo "unknown flag: $1" >&2; usage ;;
    *)            DECLARED_SCENES+=("$1"); shift ;;
  esac
done

if [[ -z "${XIAO_HEI_GEMINI_API_KEY:-}" ]]; then
  echo "error: export XIAO_HEI_GEMINI_API_KEY first" >&2
  exit 2
fi

mkdir -p "$OUT_DIR/explored_scenes" "$OUT_DIR/gt" "$OUT_DIR/preds" \
  "$OUT_DIR/metrics" "$OUT_DIR/traces" "$OUT_DIR/debug"
EXPLORE_LOGS=$REPO/exploration_logs
VLM_LOGS=$REPO/vlm_logs
mkdir -p "$EXPLORE_LOGS" "$VLM_LOGS"

# --- scene list: intersection of Unity dirs ∩ VLA-3D scenes ---------------

[[ -n "$GT_DIR" && ( -f "$GT_DIR/vla3d_ref.jsonl" || -f "$GT_DIR/vla3d_num.jsonl" ) ]] || {
  echo "error: set GT_DIR to a directory containing vla3d_ref.jsonl / vla3d_num.jsonl" >&2
  echo "  (tried XIAO_HEI_GT_DIR and a few common dataset paths)" >&2
  exit 2
}

export GT_DIR
mapfile -t VLA_SCENES < <(python3 - <<'PY'
import json
from pathlib import Path
import os
gt = Path(os.environ["GT_DIR"])
scenes = set()
for name in ("vla3d_ref.jsonl", "vla3d_num.jsonl"):
    p = gt / name
    if not p.is_file():
        continue
    for line in p.open():
        line = line.strip()
        if not line:
            continue
        scenes.add(json.loads(line)["scene"])
print("\n".join(sorted(scenes)))
PY
)

available=()
if [[ ${#DECLARED_SCENES[@]} -gt 0 ]]; then
  available=("${DECLARED_SCENES[@]}")
else
  for s in "${VLA_SCENES[@]}"; do
    [[ -d "$SCENES_DIR/$s/environment" ]] && available+=("$s")
  done
fi

if [[ -n "$NUM_SCENES" ]]; then
  available=("${available[@]:0:$NUM_SCENES}")
fi

[[ ${#available[@]} -gt 0 ]] || {
  echo "error: no scenes selected (SCENES_DIR=$SCENES_DIR GT_DIR=$GT_DIR)" >&2
  exit 2
}

echo "=== scene VLA-3D eval ==="
echo "scenes (${#available[@]}): ${available[*]}"
echo "skip_explore=$SKIP_EXPLORE gt_only=$GT_ONLY strategy=$STRATEGY splits=$SPLITS limit=${LIMIT_Q:-all}"
echo "out=$OUT_DIR"
echo

xhost +local: >/dev/null 2>&1 || echo "warning: xhost failed; RViz may not start" >&2

# --- helpers ---------------------------------------------------------------

kill_sim() {
  docker exec "$SYSTEM_CTR" bash -lc '
    pkill -f system_simulation.sh 2>/dev/null || true
    pkill -f Model.x86_64 2>/dev/null || true
    pkill -f "ros2 launch vehicle_simulator" 2>/dev/null || true
    pkill -f rviz2 2>/dev/null || true
    pkill -f waypointConverter 2>/dev/null || true
    pkill -f localPlanner 2>/dev/null || true
    pkill -f pathFollower 2>/dev/null || true
    pkill -f vehicleSimulator 2>/dev/null || true
    pkill -f keep_autonomy_joy 2>/dev/null || true
    sleep 2
  ' >/dev/null 2>&1 || true
}

start_sim() {
  # Unity must have DISPLAY set at launch or lidar/camera never publish
  # (pose can still tick from vehicleSimulator → empty terrain → NBV visited=0).
  docker exec -d "$SYSTEM_CTR" bash -lc "
    export DISPLAY=$DISPLAY_VAL
    export ROS_DOMAIN_ID=$ROS_DOMAIN_ID
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    cd /home/docker/autonomy_stack_mecanum_wheel_platform
    nohup ./system_simulation.sh >/tmp/xiao_hei_system_simulation.log 2>&1 &
  "
}

arm_autonomy() {
  docker exec "$SYSTEM_CTR" bash -lc "
    source /opt/ros/jazzy/setup.bash
    source /home/docker/autonomy_stack_mecanum_wheel_platform/install/setup.bash
    export ROS_DOMAIN_ID=$ROS_DOMAIN_ID
    pkill -f keep_autonomy_joy_inner 2>/dev/null || true
    nohup bash -c \"
      # keep_autonomy_joy_inner
      source /opt/ros/jazzy/setup.bash
      source /home/docker/autonomy_stack_mecanum_wheel_platform/install/setup.bash
      export ROS_DOMAIN_ID=$ROS_DOMAIN_ID
      while true; do
        ros2 topic pub --once /joy sensor_msgs/msg/Joy \\\"{axes: [0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0], buttons: [0,0,0,0,0,0,0,0,0,0,0]}\\\" >/dev/null 2>&1
        sleep 5
      done
    \" >/tmp/autonomy_joy_keeper.log 2>&1 &
  " >/dev/null 2>&1 || true
}

wait_for_terrain() {
  local deadline=$((SECONDS + ${1:-90}))
  while (( SECONDS < deadline )); do
    if docker exec "$SYSTEM_CTR" bash -lc "
      source /opt/ros/jazzy/setup.bash
      export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=$ROS_DOMAIN_ID
      timeout 3 ros2 topic hz /terrain_map_ext 2>&1 | grep -q \"average rate\"
    " >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

filter_gt() {
  local scene=$1 split=$2 out=$3
  local src=$GT_DIR/vla3d_${split}.jsonl
  [[ -f "$src" ]] || { echo "  missing $src" >&2; return 1; }
  # LIMIT_Q keeps the first N questions for this scene/split (not just a pred cap).
  GT_DIR="$GT_DIR" SCENE="$scene" SPLIT="$split" OUT="$out" LIMIT_Q="${LIMIT_Q:-}" python3 - <<'PY'
import json, os
from pathlib import Path
src = Path(os.environ["GT_DIR"]) / f"vla3d_{os.environ['SPLIT']}.jsonl"
out = Path(os.environ["OUT"])
scene = os.environ["SCENE"]
limit = os.environ.get("LIMIT_Q") or ""
n = int(limit) if limit.strip() else None
rows = []
for line in src.open():
    line = line.strip()
    if not line:
        continue
    row = json.loads(line)
    if row.get("scene") == scene:
        rows.append(row)
        if n is not None and len(rows) >= n:
            break
out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""))
print(f"  filtered {len(rows)} {os.environ['SPLIT']} questions -> {out}")
PY
}

wait_exploration_done() {
  local scene=$1
  local log=$EXPLORE_LOGS/$scene/exploration.log
  local started=$SECONDS
  echo "  waiting for exploration DONE (log=$log, timeout=${TIMEOUT}s)..."
  while ! grep -qE ' DONE |HARD_STOP' "$log" 2>/dev/null; do
    if (( SECONDS - started > TIMEOUT )); then
      echo "  timed out waiting for exploration after ${TIMEOUT}s" >&2
      return 1
    fi
    if [[ "$(docker inspect -f '{{.State.Running}}' "$AI_CTR" 2>/dev/null || true)" != "true" ]]; then
      echo "  ai_module not running ($AI_CTR)" >&2
      return 1
    fi
    sleep 5
  done
  # PNGs / final log lines may trail DONE by a second or two.
  sleep 5
  echo "  exploration finished: $(grep -E ' DONE |HARD_STOP' "$log" | tail -1)"
}

publish_question() {
  local q=${1:-"How many chairs are in the room?"}
  # Escape for ros2 YAML-ish string payload.
  local escaped=${q//\"/\\\"}
  docker exec "$SYSTEM_CTR" bash -lc "
    source /opt/ros/jazzy/setup.bash
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=$ROS_DOMAIN_ID
    ros2 topic pub --once /challenge_question std_msgs/msg/String \"{data: \\\"${escaped}\\\"}\"
  " >/dev/null
}

wait_for_session_scene() {
  local started=$SECONDS
  local timeout=${1:-180}
  echo "  waiting for vlm_logs session with a scene dump (timeout=${timeout}s)..." >&2
  while (( SECONDS - started < timeout )); do
    local sess
    sess=$(ls -1d "$VLM_LOGS"/session_* 2>/dev/null | sort | tail -1 || true)
    if [[ -n "$sess" ]]; then
      local qdir
      qdir=$(ls -1d "$sess"/q_* 2>/dev/null | sort | tail -1 || true)
      if [[ -n "$qdir" && -f "$qdir/ticks.jsonl" ]]; then
        if python3 -c "
import json,sys
p=sys.argv[1]
last=None
for line in open(p):
    line=line.strip()
    if line: last=json.loads(line)
sys.exit(0 if last and isinstance(last.get('scene'), dict) else 1)
" "$qdir/ticks.jsonl" 2>/dev/null; then
          echo "  session ready: $sess" >&2
          printf '%s\n' "$sess"
          return 0
        fi
      fi
    fi
    sleep 3
  done
  echo "  timed out waiting for logged scene" >&2
  return 1
}

explore_scene() {
  local scene=$1
  local dir=$SCENES_DIR/$scene
  if [[ ! -d "$dir/environment" ]]; then
    echo "  skip explore: missing $dir/environment" >&2
    return 1
  fi

  rm -f "$EXPLORE_LOGS/$scene"/exploration.log "$EXPLORE_LOGS/$scene"/*.png 2>/dev/null || true

  export XIAO_HEI_SCENE_DIR_HOST=$dir
  export XIAO_HEI_EXPLORATION_MAX_WAYPOINTS=$MAX_WAYPOINTS
  export XIAO_HEI_EXPLORATION_MAX_SECONDS=$MAX_SECONDS
  export XIAO_HEI_EXPLORATION_STRATEGY=$STRATEGY
  export XIAO_HEI_EXPLORATION_LOG_DIR=/exploration_logs
  export XIAO_HEI_VLM_LOG_DIR=/vlm_logs
  export XIAO_HEI_RESPONDER=scene_gemini
  export XIAO_HEI_OBJECT_MAP=${XIAO_HEI_OBJECT_MAP:-1}
  export XIAO_HEI_VIEWPOINT_RADIUS=${XIAO_HEI_VIEWPOINT_RADIUS:-0.8}
  export XIAO_HEI_WP_REACHED_M=${XIAO_HEI_WP_REACHED_M:-1.35}
  export XIAO_HEI_REF_SPATIAL=${XIAO_HEI_REF_SPATIAL:-1}
  # Periodic scene dump so we can export even if question publish / ticks fail.
  export XIAO_HEI_SCENE_DUMP_PATH=${XIAO_HEI_SCENE_DUMP_PATH:-/exploration_logs/${scene}/scene_live.json}

  echo "  compose up (scene=$scene strategy=$STRATEGY max_s=$MAX_SECONDS DISPLAY=$DISPLAY_VAL)..."
  kill_sim
  "${COMPOSE[@]}" up -d || { echo "  compose up failed" >&2; return 1; }

  # Wait for perception health briefly (eval overlay uses PERCEPTION_PORT).
  local i=0
  while (( i < 60 )); do
    if curl -sf "http://127.0.0.1:${PERCEPTION_PORT}/healthz" >/dev/null 2>&1 \
       || curl -sf "http://127.0.0.1:${PERCEPTION_PORT}/docs" >/dev/null 2>&1; then
      break
    fi
    sleep 2
    i=$((i + 1))
  done

  start_sim
  if ! wait_for_terrain 120; then
    echo "  ERROR: /terrain_map_ext never published — Unity likely missing DISPLAY" >&2
    docker exec "$SYSTEM_CTR" bash -lc 'tail -40 /tmp/xiao_hei_system_simulation.log 2>/dev/null' || true
    return 1
  fi
  arm_autonomy
  # Recreate AI after sim+terrain so NBV starts with a live belief map.
  : > "$EXPLORE_LOGS/exploration.log" 2>/dev/null || true
  mkdir -p "$EXPLORE_LOGS/$scene"
  : > "$EXPLORE_LOGS/$scene/exploration.log" 2>/dev/null || true
  # If root-owned from a prior container write, clear via truncate best-effort.
  chmod u+w "$EXPLORE_LOGS/$scene/exploration.log" 2>/dev/null || true
  "${COMPOSE[@]}" up -d --force-recreate ai_module || {
    echo "  ai_module recreate failed" >&2; return 1
  }
  sleep 8
  arm_autonomy

  # Publish early so the question is deferred until exploration finishes;
  # that guarantees ticks.jsonl includes the final scene graph.
  publish_question "How many chairs are in the room?" || true

  wait_exploration_done "$scene" || return 1

  local sess=""
  if sess=$(wait_for_session_scene 90); then
    echo "  exporting live scene dump from session..."
    uv run python "$REPO/scripts/export_live_scene_for_offline_eval.py" \
      --session "$sess" \
      --scene "$scene" \
      --out "$OUT_DIR/explored_scenes" || return 1
  else
    # Fallback: periodic scene dump written by the AI module.
    local live_dump=$EXPLORE_LOGS/$scene/scene_live.json
    if [[ ! -f "$live_dump" ]]; then
      echo "  ERROR: no VLM session scene and no $live_dump" >&2
      return 1
    fi
    echo "  exporting from SCENE_DUMP_PATH fallback ($live_dump)..."
    mkdir -p "$OUT_DIR/explored_scenes/$scene"
    cp -f "$live_dump" "$OUT_DIR/explored_scenes/$scene/scene.json"
  fi

  # Copy exploration occupancy plot as occupancy.png when present (multimodal).
  if [[ -f "$EXPLORE_LOGS/$scene/exploration.png" ]]; then
    cp -f "$EXPLORE_LOGS/$scene/exploration.png" \
      "$OUT_DIR/explored_scenes/$scene/occupancy.png"
  fi

  echo "  tearing down compose..."
  kill_sim
  "${COMPOSE[@]}" down || true
  # Give DDS / GPU a moment before the next scene.
  sleep 5
}

run_offline_for_scene() {
  local scene=$1
  local object_source=live
  if [[ "$GT_ONLY" -eq 1 ]]; then
    object_source=gt
  else
    local dump=$OUT_DIR/explored_scenes/$scene/scene.json
    if [[ ! -f "$dump" ]]; then
      echo "  missing live dump $dump — run without --skip-explore first" >&2
      return 1
    fi
  fi

  local IFS=','
  local split
  for split in $SPLITS; do
    split=${split// /}
    [[ -n "$split" ]] || continue
    local gt=$OUT_DIR/gt/${scene}_${split}.jsonl
    local pred=$OUT_DIR/preds/${scene}_${split}.jsonl
    local metrics=$OUT_DIR/metrics/${scene}_${split}.json
    local trace=$OUT_DIR/traces/${scene}_${split}.jsonl
    local debug_dir=$OUT_DIR/debug/${scene}_${split}
    filter_gt "$scene" "$split" "$gt" || continue
    if [[ ! -s "$gt" ]]; then
      echo "  no $split questions for $scene — skip"
      continue
    fi

    # Fresh trace/debug for this run (trace file is append-only otherwise).
    rm -f "$trace"
    rm -rf "$debug_dir"
    mkdir -p "$debug_dir"

    local src_args=(--object-source "$object_source")
    if [[ "$object_source" == "live" ]]; then
      src_args+=(--live-scenes-dir "$OUT_DIR/explored_scenes")
    fi

    echo "  gemini.batch ($split, object-source=$object_source)..."
    export XIAO_HEI_REF_SPATIAL=${XIAO_HEI_REF_SPATIAL:-1}
    uv run python -m xiao_hei_vln.gemini.batch \
      --gt "$gt" \
      --out "$pred" \
      "${src_args[@]}" \
      --trace-file "$trace" \
      --debug-dir "$debug_dir" \
      --rpm "${RPM:-5}" || {
        echo "  gemini.batch failed for $scene/$split" >&2
        continue
      }

    echo "  scoring $split..."
    uv run python -m xiao_hei_vln.eval_pipeline \
      --gt "$gt" --pred "$pred" --out "$metrics" \
      || echo "  eval_pipeline failed for $scene/$split" >&2
  done
}

# --- main loop -------------------------------------------------------------

summary=$OUT_DIR/summary.csv
echo "scene,explore_ok,ref_pred,num_pred" > "$summary"

for scene in "${available[@]}"; do
  echo
  echo "=== $scene ==="
  explore_ok=0
  if [[ "$GT_ONLY" -eq 1 ]]; then
    explore_ok=1
    echo "  GT-only — skipping explore"
  elif [[ "$SKIP_EXPLORE" -eq 1 ]]; then
    if [[ -f "$OUT_DIR/explored_scenes/$scene/scene.json" ]]; then
      explore_ok=1
      echo "  reusing existing dump"
    else
      echo "  --skip-explore but no dump — attempting explore"
      if explore_scene "$scene"; then explore_ok=1; fi
    fi
  else
    if explore_scene "$scene"; then explore_ok=1; fi
  fi

  if [[ "$explore_ok" -ne 1 ]]; then
    echo "$scene,0,," >> "$summary"
    echo "  FAILED explore — skipping offline eval"
    "${COMPOSE[@]}" down >/dev/null 2>&1 || true
    continue
  fi

  run_offline_for_scene "$scene"

  ref_n=0
  num_n=0
  if [[ -f "$OUT_DIR/preds/${scene}_ref.jsonl" ]]; then
    ref_n=$(wc -l < "$OUT_DIR/preds/${scene}_ref.jsonl" | tr -d ' ')
  fi
  if [[ -f "$OUT_DIR/preds/${scene}_num.jsonl" ]]; then
    num_n=$(wc -l < "$OUT_DIR/preds/${scene}_num.jsonl" | tr -d ' ')
  fi
  echo "$scene,1,${ref_n},${num_n}" >> "$summary"
done

echo
echo "=== finished ==="
echo "summary: $summary"
echo "dumps:   $OUT_DIR/explored_scenes/"
echo "preds:   $OUT_DIR/preds/"
echo "metrics: $OUT_DIR/metrics/"
echo "traces:  $OUT_DIR/traces/"
echo "debug:   $OUT_DIR/debug/"
cat "$summary"
