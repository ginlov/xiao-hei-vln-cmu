#!/usr/bin/env bash
# Record one replay corpus: drive a scene's coverage tour, capture every frame.
#
# The frontier explorer is not used here. It wedges, and its coverage varies
# run to run, so a corpus recorded with it measures the explorer as much as the
# perception stack. A deterministic tour (scripts/gen_coverage_tour.py) makes
# the corpora comparable across scenes and reusable across experiments.
#
# Usage:
#   scripts/record_scene.sh loft
#   scripts/record_scene.sh --stride 2 --run run2 home_building_2
#
# Env: SCENES_DIR, TOURS_DIR, FRAMES_DIR, DISPLAY, WP_TIMEOUT_S
#
# Leaves the stack down and ./frames/<scene>_<run>/ owned by the caller.

set -uo pipefail

REPO=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
cd "$REPO"

SCENES_DIR=${SCENES_DIR:-$HOME/Downloads/unity_env_models}
TOURS_DIR=${TOURS_DIR:-$REPO/tours}
FRAMES_DIR=${FRAMES_DIR:-$REPO/frames}
DISPLAY_VAL=${DISPLAY:-:0}
STRIDE=${STRIDE:-1}
RUN=run1
SYSTEM_CTR=iros2026_system
AI_CTR=xiao_hei_ai_module
# Whole-tour ceiling, so a wedged robot cannot hold the machine overnight.
TOUR_TIMEOUT_S=${TOUR_TIMEOUT_S:-2400}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stride) STRIDE=$2; shift 2 ;;
    --run) RUN=$2; shift 2 ;;
    --timeout) TOUR_TIMEOUT_S=$2; shift 2 ;;
    -h|--help) sed -n '2,17p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) SCENE=$1; shift ;;
  esac
done
[[ -n "${SCENE:-}" ]] || { echo "usage: $0 [--stride N] [--run NAME] <scene>" >&2; exit 2; }

SCENE_DIR=$SCENES_DIR/$SCENE
TOUR=$TOURS_DIR/$SCENE.json
OUT=$FRAMES_DIR/${SCENE}_${RUN}
[[ -d "$SCENE_DIR/environment" ]] || { echo "no such scene: $SCENE_DIR" >&2; exit 2; }
[[ -f "$TOUR" ]] || { echo "no tour: $TOUR (run scripts/gen_coverage_tour.py)" >&2; exit 2; }

ros_env='source /opt/ros/jazzy/setup.bash
source /home/docker/autonomy_stack_mecanum_wheel_platform/install/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0} RMW_IMPLEMENTATION=rmw_cyclonedds_cpp'

say() { echo "[$(date +%H:%M:%S)] $*"; }

cleanup() {
  say "tearing down"
  docker exec "$SYSTEM_CTR" bash -lc 'pkill -f wp_driver.py; pkill -f autonomy_joy' \
    >/dev/null 2>&1 || true
  docker/run down >/dev/null 2>&1 || true
  # The container writes frames as root; hand them back so replay can read them.
  [[ -d "$OUT" ]] && sudo chown -R "$(id -u):$(id -g)" "$OUT" 2>/dev/null || true
}
trap cleanup EXIT

say "=== $SCENE -> $OUT (stride=$STRIDE) ==="
rm -rf "$OUT"
mkdir -p "$FRAMES_DIR"

docker/run down >/dev/null 2>&1 || true

# Unity needs a real X display, and the container needs a cookie for it.
xhost +local: >/dev/null 2>&1 || say "warning: xhost failed"
XAUTH=/tmp/.docker.xauth
rm -f "$XAUTH"; touch "$XAUTH"
for src in "${XAUTHORITY:-}" "$HOME/.Xauthority" /run/user/1000/gdm/Xauthority; do
  [[ -n "$src" && -f "$src" ]] || continue
  xauth -f "$src" nlist "$DISPLAY_VAL" 2>/dev/null \
    | sed -e 's/^..../ffff/' | xauth -f "$XAUTH" nmerge - 2>/dev/null && break
done
chmod 644 "$XAUTH"

export DISPLAY=$DISPLAY_VAL
export XIAO_HEI_SCENE_DIR_HOST=$SCENE_DIR
export XIAO_HEI_FRAME_RECORD_DIR=/frames/${SCENE}_${RUN}
export XIAO_HEI_FRAME_RECORD_STRIDE=$STRIDE
# The recorder lives in the tick, but the explorer must not publish /way_point
# or it fights the tour driver for control of the robot.
export XIAO_HEI_EXPLORATION_MAX_WAYPOINTS=0
# Sample at the camera's rate rather than the default 2 Hz. Ticks that outpace
# the camera re-record a frame, which stage B drops when it thins to keyframes.
export XIAO_HEI_VLM_TICK_HZ=${XIAO_HEI_VLM_TICK_HZ:-4}

# Recording deliberately runs the `dummy` responder: detection happens offline
# in replay stage A, and calling the sidecar from the tick costs ~1 s per frame
# — which halved the corpus on the first run and put YOLO on the GPU for
# results nobody reads.
say "compose up"
docker/run dummy up -d >/dev/null || { say "compose up failed"; exit 1; }
docker cp "$XAUTH" "$SYSTEM_CTR:/tmp/.docker.xauth" >/dev/null 2>&1 || true

say "starting sim"
docker exec -d "$SYSTEM_CTR" bash -lc "
  export DISPLAY=$DISPLAY_VAL XAUTHORITY=/tmp/.docker.xauth
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  cd /home/docker/autonomy_stack_mecanum_wheel_platform
  nohup ./system_simulation.sh >/tmp/sim.log 2>&1 &
"

say "waiting for terrain"
deadline=$((SECONDS + 150))
until docker exec "$SYSTEM_CTR" bash -lc "$ros_env
  timeout 3 ros2 topic hz /terrain_map_ext 2>&1 | grep -q 'average rate'" >/dev/null 2>&1
do
  if (( SECONDS > deadline )); then
    say "ERROR: /terrain_map_ext never published — Unity probably has no display"
    docker exec "$SYSTEM_CTR" bash -lc 'tail -30 /tmp/sim.log' || true
    exit 1
  fi
  sleep 3
done
say "terrain up"

# The local planner ignores waypoints until the joystick reports "autonomy on".
docker exec -d "$SYSTEM_CTR" bash -lc "$ros_env
  while true; do
    ros2 topic pub --once /joy sensor_msgs/msg/Joy \
      '{axes: [0,0,-1,0,0,1,0,0], buttons: [0,0,0,0,0,0,0,0,0,0,0]}' >/dev/null 2>&1
    sleep 5
  done" >/dev/null 2>&1

sleep 5
docker restart "$AI_CTR" >/dev/null 2>&1 || true
say "ai_module restarted (recorder armed)"
sleep 10

docker cp "$REPO/scripts/wp_driver.py" "$SYSTEM_CTR:/tmp/wp_driver.py" >/dev/null
docker cp "$TOUR" "$SYSTEM_CTR:/tmp/tour.json" >/dev/null
n_wp=$(python3 -c "import json;print(len(json.load(open('$TOUR'))['waypoints']))")
say "driving $n_wp waypoints (timeout ${TOUR_TIMEOUT_S}s)"

timeout "$TOUR_TIMEOUT_S" docker exec "$SYSTEM_CTR" bash -lc "$ros_env
  python3 /tmp/wp_driver.py /tmp/tour.json" 2>&1 \
  | stdbuf -oL grep -E 'WP [0-9]+/|DRIVER_DONE' \
  | awk 'NR % 5 == 1 || /DRIVER_DONE/'
rc=${PIPESTATUS[0]}
[[ $rc -eq 124 ]] && say "tour hit the ${TOUR_TIMEOUT_S}s ceiling — corpus is partial"

# The recorder writes on the tick; let the last few frames land.
sleep 8
n_frames=$(docker exec "$AI_CTR" bash -lc "ls /frames/${SCENE}_${RUN}/*.jpg 2>/dev/null | wc -l" | tr -d '\r')
say "recorded $n_frames frames"
[[ "${n_frames:-0}" -gt 0 ]] || { say "ERROR: nothing recorded"; exit 1; }

# Reaching every waypoint is what makes two corpora comparable. A tour that
# stalled halfway looks like a perception failure later, so say so now.
sudo chown -R "$(id -u):$(id -g)" "$OUT" 2>/dev/null || true
python3 - "$TOUR" "$OUT/frames.jsonl" <<'PY'
import json, math, sys
tour = json.load(open(sys.argv[1]))["waypoints"]
pos = [tuple(json.loads(l)["position"][:2]) for l in open(sys.argv[2])]
seen = sum(1 for w in tour
           if any(math.dist((w["x"], w["y"]), p) < 0.8 for p in pos))
print(f"tour coverage: {seen}/{len(tour)} waypoints ({seen / len(tour):.0%})"
      + ("" if seen == len(tour) else "  <-- PARTIAL"))
PY
exit 0
