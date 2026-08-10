#!/usr/bin/env bash
# Orchestrate the drive-and-capture benchmark across all 15 scenes.
#
# Each scene runs in a FRESH sim container (recreated per scene) with that
# scene's files bind-mounted via docker/compose.scene.yml. A fresh container
# guarantees clean RViz + ROS + Unity state every scene — no leftover mesh /
# overlay from the previous room (process-level kill+relaunch left RViz cruft).
# Only the `system` service is started, so ai_module never runs and nothing
# drives the robot but our capture. Each scene's captures are copied to the host
# as it finishes.
#
#   perception_benchmark/run_all_scenes.sh                    # all 15
#   perception_benchmark/run_all_scenes.sh chinese_room loft  # subset
#   FORCE=1 perception_benchmark/run_all_scenes.sh            # re-capture existing
#   KEEP_UP=1 perception_benchmark/run_all_scenes.sh          # leave last sim up
#
# The sim container is removed after each scene and on exit (normal finish OR
# Ctrl-C / kill). Resumable: scenes whose captures/<scene>/manifest.json exists
# are skipped unless FORCE=1. Per-scene failures are logged and skipped.

set -uo pipefail

REPO="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
ZIP_DIR="${ZIP_DIR:-/home/long/Projects/dataset/unity_scenes}"
EXTRACT_DIR="${EXTRACT_DIR:-/home/long/Projects/dataset/unity_scenes_extracted}"
OUT_DIR="$REPO/perception_benchmark/captures"
VP_DIR="$REPO/perception_benchmark/viewpoints"
CONTAINER=iros2026_system
SIM_SH=/home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh
ROS='source /opt/ros/jazzy/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp'
WP_TIMEOUT="${WP_TIMEOUT:-60}"
BOOT_TRIES="${BOOT_TRIES:-40}"    # x3s = up to 2 min for the sim to come up
SETTLE_S="${SETTLE_S:-12}"        # extra wait after nav is up (Unity + terrain)

ALL_SCENES=(arabic_room chinese_room home_building_1 home_building_2 hotel_room_1 \
  hotel_room_2 japanese_room livingroom_1 livingroom_2 livingroom_3 livingroom_4 \
  loft office_1 office_2 studio)
SCENES=("$@"); [ ${#SCENES[@]} -eq 0 ] && SCENES=("${ALL_SCENES[@]}")

cd "$REPO"
mkdir -p "$OUT_DIR" "$EXTRACT_DIR"
declare -A RESULT

teardown() { docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }

cleanup() {
  if [ "${KEEP_UP:-0}" = "1" ]; then
    echo "KEEP_UP=1 — leaving sim container up ($CONTAINER)."; return
  fi
  echo "cleaning up sim container ($CONTAINER) ..."; teardown
}
on_signal() { echo; echo "interrupted — stopping."; exit 130; }
trap on_signal INT TERM
trap cleanup EXIT

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

capture_scene() {   # $1 = scene
  local scene="$1" scene_dir="$EXTRACT_DIR/$scene"
  if [ -f "$OUT_DIR/$scene/manifest.json" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[$scene] already captured — skipping (FORCE=1 to redo)."; RESULT[$scene]="skipped"; return 0
  fi
  if [ ! -f "$VP_DIR/$scene.json" ]; then
    echo "[$scene] no viewpoints file — skipping."; RESULT[$scene]="no-viewpoints"; return 1
  fi
  # Re-extract unless ALL mounted paths are present and the right type. A partial
  # extraction (interrupted unzip) leaves some files missing; a failed earlier
  # `up` then auto-creates the missing bind source as a (root-owned) directory,
  # which breaks the file mount. Checking only environment/ misses both.
  local need=0
  [ -d "$scene_dir/environment" ] || need=1
  for f in map.ply map.jpg traversable_area.ply object_list.txt; do
    [ -f "$scene_dir/$f" ] || need=1
    [ -d "$scene_dir/$f" ] && need=1        # docker-created bogus dir
  done
  if [ "$need" = 1 ]; then
    echo "[$scene] extracting $ZIP_DIR/$scene.zip -> $EXTRACT_DIR"
    for f in map.ply map.jpg traversable_area.ply object_list.txt; do
      [ -d "$scene_dir/$f" ] && { rm -rf "$scene_dir/$f" 2>/dev/null || sudo rm -rf "$scene_dir/$f"; }
    done
    unzip -oq "$ZIP_DIR/$scene.zip" -d "$EXTRACT_DIR" || { RESULT[$scene]="unzip-fail"; return 1; }
  fi

  # fresh container with THIS scene mounted (clean RViz/ROS/Unity)
  echo "[$scene] starting fresh sim container ..."
  teardown
  XIAO_HEI_SCENE_DIR_HOST="$scene_dir" docker compose \
    -f docker/compose.yml -f docker/compose.scene.yml up -d system \
    || { RESULT[$scene]="compose-up-fail"; return 1; }

  docker exec -d "$CONTAINER" bash -lc "$SIM_SH"

  echo "[$scene] waiting for sim to come up ..."
  if ! wait_ready; then
    echo "[$scene] sim never came up — skipping."; RESULT[$scene]="sim-timeout"; teardown; return 1
  fi
  sleep "$SETTLE_S"

  docker cp "$REPO/perception_benchmark/capture_viewpoints.py" "$CONTAINER:/tmp/cap.py" >/dev/null
  docker cp "$VP_DIR/$scene.json" "$CONTAINER:/tmp/vps.json" >/dev/null

  echo "[$scene] capturing ..."
  docker exec "$CONTAINER" bash -lc \
    "$ROS && python3 /tmp/cap.py /tmp/vps.json --scene $scene --out /tmp/captures --wp-timeout $WP_TIMEOUT"

  rm -rf "$OUT_DIR/$scene"
  if docker cp "$CONTAINER:/tmp/captures/$scene" "$OUT_DIR/" >/dev/null 2>&1; then
    local n ok
    n=$(ls -d "$OUT_DIR/$scene"/vp_* 2>/dev/null | wc -l)
    ok=$(python3 -c "import json;m=json.load(open('$OUT_DIR/$scene/manifest.json'));print(sum(v['arrived'] and v['clean'] for v in m['viewpoints']))" 2>/dev/null || echo "?")
    echo "[$scene] copied $n viewpoints ($ok clean) -> $OUT_DIR/$scene"; RESULT[$scene]="ok:$ok/$n"
  else
    RESULT[$scene]="copy-fail"
  fi

  teardown   # remove before the next scene so RViz/ROS start clean
  return 0
}

echo "==== capturing ${#SCENES[@]} scene(s): ${SCENES[*]} ===="
for scene in "${SCENES[@]}"; do
  echo; echo "======================== $scene ========================"
  capture_scene "$scene" || true
done

echo; echo "==================== SUMMARY ===================="
for scene in "${SCENES[@]}"; do printf "  %-18s %s\n" "$scene" "${RESULT[$scene]:-unknown}"; done
echo "captures under: $OUT_DIR"
# cleanup() runs here via the EXIT trap (removes the sim container).
