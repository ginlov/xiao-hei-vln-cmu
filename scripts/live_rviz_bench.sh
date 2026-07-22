#!/usr/bin/env bash
# Live Unity+RViz bench: top-3 map-free algos × all scenes, end-of-run screenshots.
#
# Uses Aryan-only compose project (aryan_nbv / aryan_* containers, ROS_DOMAIN_ID=42)
# so Rajath's iros2026_system / xiao_hei_ai_module are not recreated.
#
# End-of-run images
# -----------------
# 1) Desktop grab of DISPLAY=:0 via Pillow ImageGrab (Unity + RViz as shown on DCV).
# 2) Optional copy of exploration_logs/exploration.png (matplotlib plot the AI saves).
#
# Each (scene, algo) is hard-stopped at RUN_TIMEOUT_S (default 480 = 8 min):
# ai_module is stopped so exploration freezes, then the screenshot is taken.
#
# Prefer the detached launcher (survives closing the terminal):
#   ./scripts/start_live_rviz_bench_detached.sh
#
# Or run foreground:
#   export XIAO_HEI_GEMINI_API_KEY=...
#   ./scripts/live_rviz_bench.sh
#
# Optional:
#   SCENES="chinese_room arabic_room" ALGOS="nbv rrt wall_follow" \
#   RUN_TIMEOUT_S=480 ./scripts/live_rviz_bench.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

SCENES_ROOT="${SCENES_ROOT:-/home/ubuntu/Downloads/unity_env_models}"
OUT_DIR="${OUT_DIR:-$REPO/exploration_logs/live_rviz}"
ALGOS=(${ALGOS:-nbv rrt wall_follow})
RUN_TIMEOUT_S="${RUN_TIMEOUT_S:-480}"   # 8 min per (scene, algo)
BOOT_WAIT_S="${BOOT_WAIT_S:-90}"
DISPLAY_VAL="${DISPLAY:-:0}"

export DISPLAY="$DISPLAY_VAL"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
xhost +local: >/dev/null 2>&1 || true

: "${XIAO_HEI_GEMINI_API_KEY:?export XIAO_HEI_GEMINI_API_KEY first}"
export XIAO_HEI_VLM_LOG_DIR=/vlm_logs
export XIAO_HEI_EXPLORATION_LOG_DIR=/exploration_logs
export XIAO_HEI_EXPLORATION_MAX_WAYPOINT_DIST=3.0
export XIAO_HEI_EXPLORATION_MAX_WAYPOINTS=80

if [[ -n "${SCENES:-}" ]]; then
  read -r -a SCENE_LIST <<< "$SCENES"
else
  mapfile -t SCENE_LIST < <(
    for d in "$SCENES_ROOT"/*/traversable_area.ply; do
      basename "$(dirname "$d")"
    done | sort
  )
fi

mkdir -p "$OUT_DIR"
MANIFEST="$OUT_DIR/manifest.jsonl"
: > "$MANIFEST"

compose() {
  docker compose -p aryan_nbv \
    -f docker/compose_scene_gemini.yml \
    -f docker/compose.scene.yml \
    -f docker/compose.aryan.yml \
    "$@"
}

kill_sim() {
  docker exec aryan_iros2026_system bash -lc '
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
  docker exec -d aryan_iros2026_system bash -lc "
    export DISPLAY=$DISPLAY_VAL
    export ROS_DOMAIN_ID=42
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    cd /home/docker/autonomy_stack_mecanum_wheel_platform
    nohup ./system_simulation.sh >/tmp/aryan_system_simulation.log 2>&1 &
  "
}

arm_autonomy() {
  docker exec aryan_iros2026_system bash -lc '
    source /opt/ros/jazzy/setup.bash
    source /home/docker/autonomy_stack_mecanum_wheel_platform/install/setup.bash
    export ROS_DOMAIN_ID=42
    # One-shot then rare re-arm (> joyToSpeedDelay=2s).
    pkill -f keep_autonomy_joy_inner 2>/dev/null || true
    nohup bash -c "
      # keep_autonomy_joy_inner
      source /opt/ros/jazzy/setup.bash
      source /home/docker/autonomy_stack_mecanum_wheel_platform/install/setup.bash
      export ROS_DOMAIN_ID=42
      while true; do
        ros2 topic pub --once /joy sensor_msgs/msg/Joy \"{axes: [0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0], buttons: [0,0,0,0,0,0,0,0,0,0,0]}\" >/dev/null 2>&1
        sleep 5
      done
    " >/tmp/autonomy_joy_keeper.log 2>&1 &
  ' >/dev/null 2>&1 || true
}

wait_for_odom() {
  local deadline=$((SECONDS + BOOT_WAIT_S))
  while (( SECONDS < deadline )); do
    if docker exec aryan_iros2026_system bash -lc '
      source /opt/ros/jazzy/setup.bash
      source /home/docker/autonomy_stack_mecanum_wheel_platform/install/setup.bash
      export ROS_DOMAIN_ID=42
      timeout 2 ros2 topic echo /state_estimation --once >/dev/null 2>&1
    '; then
      return 0
    fi
    sleep 2
  done
  return 1
}

wait_done_or_timeout() {
  local timeout_s="$1"
  local deadline=$((SECONDS + timeout_s))
  local log="$REPO/exploration_logs/exploration.log"
  local start_lines=0
  [[ -f "$log" ]] && start_lines=$(wc -l < "$log")
  while (( SECONDS < deadline )); do
    if [[ -f "$log" ]]; then
      if tail -n +"$((start_lines + 1))" "$log" 2>/dev/null | grep -q ' DONE '; then
        return 0
      fi
    fi
    if docker logs aryan_xiao_hei_ai_module 2>&1 | tail -40 | grep -qE 'Exploration complete|Exploration plot saved|FORCE_DONE'; then
      return 0
    fi
    sleep 5
  done
  return 1
}

force_done_and_dump() {
  # Ask AI to dump exploration_state.json, then wait briefly for it.
  rm -f "$REPO/exploration_logs/exploration_state.json"
  docker exec aryan_xiao_hei_ai_module bash -lc 'touch /exploration_logs/FORCE_DONE' 2>/dev/null \
    || touch "$REPO/exploration_logs/FORCE_DONE"
  local i
  for i in $(seq 1 20); do
    if [[ -f "$REPO/exploration_logs/exploration_state.json" ]]; then
      return 0
    fi
    sleep 1
  done
  return 1
}

score_run() {
  local scene="$1"
  local algo="$2"
  local status="$3"
  local stamp="$4"
  local scene_dir="$5"
  local out_json="$OUT_DIR/$scene/${algo}_${status}_${stamp}_score.json"
  if [[ ! -f "$REPO/exploration_logs/exploration_state.json" ]]; then
    echo "WARN no exploration_state.json — skip score"
    return 1
  fi
  cp -f "$REPO/exploration_logs/exploration_state.json" \
    "$OUT_DIR/$scene/${algo}_${status}_${stamp}_state.json"
  local py="$REPO/.venv/bin/python"
  [[ -x "$py" ]] || py=python3
  "$py" "$REPO/scripts/score_live_run.py" \
    --state "$REPO/exploration_logs/exploration_state.json" \
    --scene "$scene_dir" \
    --done-reason "$status" \
    --out "$out_json" || return 1
  echo "score → $out_json"
}

screenshot() {
  local out="$1"
  python3 - <<PY
import os
os.environ["DISPLAY"] = "$DISPLAY_VAL"
from PIL import ImageGrab
img = ImageGrab.grab()
img.save("$out")
print("saved", "$out", img.size)
PY
}

echo "Live RViz bench → $OUT_DIR"
echo "algos=${ALGOS[*]}  scenes=${#SCENE_LIST[@]}  timeout=${RUN_TIMEOUT_S}s"
echo "NOTE: uses aryan_* only (ROS_DOMAIN_ID=42). Does not recreate Rajath containers."

# Ensure perception is up once.
export XIAO_HEI_SCENE_DIR_HOST="$SCENES_ROOT/${SCENE_LIST[0]}"
export XIAO_HEI_EXPLORATION_STRATEGY="${ALGOS[0]}"
compose up -d perception >/dev/null
sleep 5

idx=0
total=$(( ${#SCENE_LIST[@]} * ${#ALGOS[@]} ))

for scene in "${SCENE_LIST[@]}"; do
  scene_dir="$SCENES_ROOT/$scene"
  if [[ ! -d "$scene_dir/environment" ]]; then
    echo "SKIP missing scene dir: $scene_dir"
    continue
  fi
  mkdir -p "$OUT_DIR/$scene"

  for algo in "${ALGOS[@]}"; do
    idx=$((idx + 1))
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    echo
    echo "===== [$idx/$total] scene=$scene algo=$algo ($stamp) ====="

    export XIAO_HEI_SCENE_DIR_HOST="$scene_dir"
    export XIAO_HEI_EXPLORATION_STRATEGY="$algo"

    kill_sim
    # Truncate exploration log for clean DONE detection.
    docker exec aryan_xiao_hei_ai_module bash -lc '
      : > /exploration_logs/exploration.log
      rm -f /exploration_logs/FORCE_DONE /exploration_logs/exploration_state.json
    ' 2>/dev/null || true

    compose up -d --force-recreate system ai_module
    sleep 3
    start_sim

    if ! wait_for_odom; then
      echo "FAIL boot odom scene=$scene algo=$algo"
      screenshot "$OUT_DIR/$scene/${algo}_BOOTFAIL_${stamp}.png" || true
      echo "{\"scene\":\"$scene\",\"algo\":\"$algo\",\"status\":\"boot_fail\",\"ts\":\"$stamp\"}" >> "$MANIFEST"
      kill_sim
      continue
    fi

    arm_autonomy
    # Recreate AI after sim is live so exploration starts cleanly.
    compose up -d --force-recreate ai_module
    sleep 8
    arm_autonomy

    status="timeout"
    if wait_done_or_timeout "$RUN_TIMEOUT_S"; then
      status="done"
      # Natural finish should already have dumped state; wait a beat if needed.
      for _ in 1 2 3 4 5; do
        [[ -f "$REPO/exploration_logs/exploration_state.json" ]] && break
        sleep 1
      done
    else
      echo "HARD STOP at ${RUN_TIMEOUT_S}s — FORCE_DONE dump"
      force_done_and_dump || true
      status="timeout"
    fi

    shot="$OUT_DIR/$scene/${algo}_${status}_${stamp}.png"
    screenshot "$shot" || true
    # Also keep the in-container exploration plot if written.
    if [[ -f "$REPO/exploration_logs/exploration.png" ]]; then
      cp -f "$REPO/exploration_logs/exploration.png" "$OUT_DIR/$scene/${algo}_exploration_${stamp}.png"
    fi
    if [[ -f "$REPO/exploration_logs/exploration.log" ]]; then
      cp -f "$REPO/exploration_logs/exploration.log" "$OUT_DIR/$scene/${algo}_${stamp}.log"
    fi

    score_json=""
    if score_run "$scene" "$algo" "$status" "$stamp" "$scene_dir"; then
      score_json="$OUT_DIR/$scene/${algo}_${status}_${stamp}_score.json"
    fi

    echo "{\"scene\":\"$scene\",\"algo\":\"$algo\",\"status\":\"$status\",\"screenshot\":\"$shot\",\"score\":\"$score_json\",\"ts\":\"$stamp\"}" >> "$MANIFEST"
    echo "→ $status  shot=$shot"
    kill_sim
    sleep 2
  done
done

echo
echo "All runs finished. Manifest: $MANIFEST"
echo "Screenshots under: $OUT_DIR/<scene>/"
PY="$REPO/.venv/bin/python"
[[ -x "$PY" ]] || PY=python3
"$PY" "$REPO/scripts/aggregate_live_scores.py" --root "$OUT_DIR" --out "$OUT_DIR/live_summary.json" || true
echo "Summary: $OUT_DIR/live_summary.json"
