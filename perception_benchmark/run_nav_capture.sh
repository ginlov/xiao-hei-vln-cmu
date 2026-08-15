#!/usr/bin/env bash
# Record the sensor stream during REAL navigation, across all scenes.
#
# The sibling sweep (run_all_scenes.sh) drives the robot to geometrically-chosen
# viewpoints — ideal coverage, not a real trajectory. This one runs the actual
# frontier explorer (the same one the submission stack uses) and passively
# records keyframes along whatever path it takes, so the offline scene-graph
# build sees the inputs the live stack would really have had.
#
#   perception_benchmark/run_nav_capture.sh                    # every scene
#   perception_benchmark/run_nav_capture.sh arabic_room        # one scene first
#   FORCE=1 perception_benchmark/run_nav_capture.sh            # re-capture existing
#   MAX_WAYPOINTS=80 TIMEOUT=900 perception_benchmark/run_nav_capture.sh
#
# Per scene: extract the scene, bring up a FRESH sim + ai_module (dummy
# responder, so no GPU and no sidecar needed during capture), launch the sim,
# start the passive recorder inside the sim container, then wait for the
# explorer to log DONE. On DONE we touch the recorder's stop file, let it flush,
# and copy the captures to the host.
#
# Output: perception_benchmark/captures_nav/<scene>/vp_XXX/ — same layout as
# captures/, so the offline tooling reads it with:
#
#   export PERCEPTION_CAP_DIR=perception_benchmark/captures_nav
#   uv run --extra perception python perception_benchmark/replay_score.py --all
#
# Env: ZIP_DIR, EXTRACT_DIR, OUT_DIR, MAX_WAYPOINTS, TIMEOUT, RATE_HZ, FORCE,
#      KEEP_UP, DISPLAY.

# No -e: one bad scene must not abort the sweep.
set -uo pipefail

REPO="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
ZIP_DIR="${ZIP_DIR:-/home/long/Projects/dataset/unity_scenes}"
EXTRACT_DIR="${EXTRACT_DIR:-/home/long/Projects/dataset/unity_scenes_extracted}"
OUT_DIR="${OUT_DIR:-$REPO/perception_benchmark/captures_nav}"
LOGS="$REPO/exploration_logs"
CONTAINER=iros2026_system
SIM_SH=/home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh
ROS='source /opt/ros/jazzy/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp'
STOP_FILE=/tmp/stop_capture

MAX_WAYPOINTS="${MAX_WAYPOINTS:-500}"
# Per scene, excluding sim boot. A ceiling, not a target: the explorer usually
# ends first on its own (max_consecutive_skips), and this only truncates a run
# that would otherwise overrun. At ~500 MB/min it also caps a scene at ~4.3 GB.
TIMEOUT="${TIMEOUT:-480}"         # 8 min cap on exploration per scene
BOOT_TRIES="${BOOT_TRIES:-40}"    # x3s = up to 2 min for the sim to come up
SETTLE_S="${SETTLE_S:-12}"        # extra wait after nav is up (Unity + terrain)
# 1 Hz matches the live node's XIAO_HEI_VLM_TICK_HZ default, i.e. one recorded
# frame per exploration tick. At ~4.2 MB/frame that is ~250 MB per minute of
# exploration per scene — check free space before a full sweep, or lower RATE_HZ.
RATE_HZ="${RATE_HZ:-1.0}"
export DISPLAY="${DISPLAY:-:0}"

ALL_SCENES=(arabic_room chinese_room home_building_1 home_building_2 hotel_room_1 \
  hotel_room_2 japanese_room livingroom_1 livingroom_2 livingroom_3 livingroom_4 \
  loft office_1 office_2 studio)
SCENES=("$@"); [ ${#SCENES[@]} -eq 0 ] && SCENES=("${ALL_SCENES[@]}")

COMPOSE=(docker compose -f "$REPO/docker/compose.yml" -f "$REPO/docker/compose.scene.yml")

cd "$REPO"
# Create the bind-mount target ourselves. Otherwise Docker auto-creates it as
# root and the next run's log cleanup fails with permission denied.
mkdir -p "$OUT_DIR" "$EXTRACT_DIR" "$LOGS"
declare -A RESULT

# Remove by name, not `compose down`: compose.scene.yml declares
# XIAO_HEI_SCENE_DIR_HOST with `:?`, so any compose call outside a scene loop
# (the EXIT trap, the pre-scene reset) would abort on the unset variable.
# ai_module is `restart: unless-stopped`, so it must be rm'd, not stopped.
teardown() { docker rm -f "$CONTAINER" xiao_hei_ai_module >/dev/null 2>&1 || true; }

cleanup() {
  if [ "${KEEP_UP:-0}" = "1" ]; then
    echo "KEEP_UP=1 — leaving the stack up."; return
  fi
  echo "cleaning up containers ..."; teardown
}
on_signal() { echo; echo "interrupted — stopping."; exit 130; }
trap on_signal INT TERM
trap cleanup EXIT

# The containers run as a different user than the desktop session, so X refuses
# them without this and RViz dies at startup.
xhost +local: >/dev/null 2>&1 || echo "warning: xhost failed; RViz may not start" >&2

wait_ready() {
  for _ in $(seq 1 "$BOOT_TRIES"); do
    if docker exec "$CONTAINER" bash -lc \
        "$ROS && timeout 4 ros2 topic echo --once /state_estimation >/dev/null 2>&1"; then
      return 0
    fi
    sleep 3
  done
  return 1
}

extract_scene() {   # $1 = scene, $2 = scene_dir
  local scene="$1" scene_dir="$2" need=0 f
  [ -d "$scene_dir/environment" ] || need=1
  for f in map.ply map.jpg traversable_area.ply object_list.txt; do
    [ -f "$scene_dir/$f" ] || need=1
    [ -d "$scene_dir/$f" ] && need=1        # docker-created bogus dir
  done
  [ "$need" = 0 ] && return 0
  echo "[$scene] extracting $ZIP_DIR/$scene.zip -> $EXTRACT_DIR"
  for f in map.ply map.jpg traversable_area.ply object_list.txt; do
    [ -d "$scene_dir/$f" ] && { rm -rf "$scene_dir/$f" 2>/dev/null || sudo rm -rf "$scene_dir/$f"; }
  done
  unzip -oq "$ZIP_DIR/$scene.zip" -d "$EXTRACT_DIR"
}

capture_scene() {   # $1 = scene
  local scene="$1" scene_dir="$EXTRACT_DIR/$scene" log="$LOGS/$scene/exploration.log"
  if [ -f "$OUT_DIR/$scene/manifest.json" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[$scene] already captured — skipping (FORCE=1 to redo)."; RESULT[$scene]="skipped"; return 0
  fi
  extract_scene "$scene" "$scene_dir" || { RESULT[$scene]="unzip-fail"; return 1; }

  # Don't rm the scene log dir: the node opens exploration.log with mode "w", and
  # deleting it while a lingering container holds it open orphans the file.
  mkdir -p "$LOGS/$scene"; rm -f "$log"

  echo "[$scene] starting fresh sim + ai_module ..."
  teardown
  XIAO_HEI_SCENE_DIR_HOST="$scene_dir" \
  XIAO_HEI_RESPONDER=dummy \
  XIAO_HEI_EXPLORATION_STRATEGY=frontier \
  XIAO_HEI_EXPLORATION_MAX_WAYPOINTS="$MAX_WAYPOINTS" \
    "${COMPOSE[@]}" up -d || { RESULT[$scene]="compose-up-fail"; return 1; }

  docker exec -d "$CONTAINER" bash -lc "$SIM_SH"

  echo "[$scene] waiting for sim to come up ..."
  if ! wait_ready; then
    echo "[$scene] sim never came up — skipping."; RESULT[$scene]="sim-timeout"; return 1
  fi
  sleep "$SETTLE_S"

  docker exec "$CONTAINER" rm -f "$STOP_FILE" >/dev/null 2>&1
  docker cp "$REPO/perception_benchmark/capture_viewpoints.py" "$CONTAINER:/tmp/capture_viewpoints.py" >/dev/null
  docker cp "$REPO/perception_benchmark/record_navigation.py"  "$CONTAINER:/tmp/record_navigation.py"  >/dev/null

  echo "[$scene] recording while the explorer drives ..."
  docker exec -d "$CONTAINER" bash -lc \
    "$ROS && python3 /tmp/record_navigation.py --scene $scene --out /tmp/nav_captures \
       --rate-hz $RATE_HZ \
       --stop-file $STOP_FILE --max-seconds $((TIMEOUT + 120)) > /tmp/record_$scene.log 2>&1"

  # Wait for the explorer to finish. The recorder is passive, so the explorer's
  # DONE line is the only authoritative end-of-trajectory signal.
  local started=$SECONDS reason=done
  while ! grep -q " DONE " "$log" 2>/dev/null; do
    if (( SECONDS - started > TIMEOUT )); then reason=timeout; break; fi
    if [ "$(docker inspect -f '{{.State.Running}}' xiao_hei_ai_module 2>/dev/null)" != "true" ]; then
      reason=ai-module-exited; break
    fi
    sleep 5
  done
  echo "[$scene] exploration ended ($reason) — stopping the recorder ..."

  docker exec "$CONTAINER" touch "$STOP_FILE"
  # The recorder polls the stop file at ~20 Hz and then flushes. It also rewrites
  # the manifest after every keyframe, so the copy below is safe even if this
  # wait gives up early (e.g. no pgrep in the image) — at worst we lose the
  # frame in flight.
  sleep 2
  for _ in $(seq 1 20); do
    docker exec "$CONTAINER" pgrep -f record_navigation.py >/dev/null 2>&1 || break
    sleep 1
  done
  docker exec "$CONTAINER" cat "/tmp/record_$scene.log" 2>/dev/null | tail -3

  rm -rf "$OUT_DIR/$scene"
  if docker cp "$CONTAINER:/tmp/nav_captures/$scene" "$OUT_DIR/" >/dev/null 2>&1; then
    local n skip sz
    n=$(ls -d "$OUT_DIR/$scene"/vp_* 2>/dev/null | wc -l)
    # A non-zero skip count means ticks the live stack would have ingested but
    # we could not record — worth seeing in the summary, not buried in the log.
    skip=$(python3 -c "import json;s=json.load(open('$OUT_DIR/$scene/manifest.json')).get('skipped',{});print(sum(s.values()))" 2>/dev/null || echo "?")
    sz=$(du -sh "$OUT_DIR/$scene" 2>/dev/null | cut -f1)
    echo "[$scene] copied $n frames ($skip ticks skipped, $sz) -> $OUT_DIR/$scene"
    RESULT[$scene]="$reason:${n}f/${skip}skip/$sz"
  else
    RESULT[$scene]="copy-fail"
  fi

  teardown   # remove before the next scene so RViz/ROS/Unity start clean
  return 0
}

echo "==== nav-capturing ${#SCENES[@]} scene(s): ${SCENES[*]} ===="
for scene in "${SCENES[@]}"; do
  echo; echo "======================== $scene ========================"
  capture_scene "$scene" || true
done

echo; echo "==================== SUMMARY ===================="
for scene in "${SCENES[@]}"; do printf "  %-18s %s\n" "$scene" "${RESULT[$scene]:-unknown}"; done
echo "captures under: $OUT_DIR"
echo
echo "score them with:"
echo "  export PERCEPTION_CAP_DIR=$OUT_DIR"
echo "  uv run --extra perception python perception_benchmark/replay_score.py --all"
# cleanup() runs here via the EXIT trap.
