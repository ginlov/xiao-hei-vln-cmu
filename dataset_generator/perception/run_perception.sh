#!/usr/bin/env bash
# Bring up the live perception stack on a server (run THIS ON THE SERVER HOST).
#
#   bash dataset_generator/perception/run_perception.sh
#
# Idempotent: starts the sim if it isn't up, installs perception deps only the
# first time, always re-syncs the code, and (re)starts the worker. Self-locates
# the repo from its own path.
#
# Env overrides:
#   CONTAINER=iros2026_system   sim/perception container name
#   DEVICE=cuda                 torch device for YOLO+SAM
#   SKIP_SIM=1                  don't touch the sim (already running)
#   SKIP_INSTALL=1              don't install deps (already present)
#   REINSTALL=1                 force-reinstall deps
set -uo pipefail

CONTAINER=${CONTAINER:-iros2026_system}
DEVICE=${DEVICE:-cuda}
SIM_SH=/home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh
# Comprehensive results for the VLM stage land here (bind-mounted to the host on
# xiaohei1, so they persist + are readable from the VLM container). BENCHMARK=1
# also builds the live GT map + scoreboard from /camera/semantic_image.
RESULT_DIR=${RESULT_DIR:-/percep_out}
BENCHMARK=${BENCHMARK:-1}
LEGEND_DIR=${LEGEND_DIR:-/home/docker/autonomy_stack_mecanum_wheel_platform/install/vehicle_simulator/share/vehicle_simulator/mesh/unity/environment}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # <repo>/dataset_generator/perception
DG=$(dirname "$HERE")

say(){ printf "\n\033[1;36m== %s ==\033[0m\n" "$*"; }
dexec(){ docker exec "$CONTAINER" bash -lc "$*"; }
dros(){ docker exec "$CONTAINER" bash -lc \
  "source /opt/ros/jazzy/setup.bash; export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp; $*"; }
# The sim is "up" only if the Unity env process is alive — a topic name can linger
# in DDS discovery after a restart, so grepping topics gives false positives and we
# wrongly skip launching system_simulation.sh (Unity + rviz never come up).
sim_up(){ dexec "pgrep -f Model.x86_64 >/dev/null"; }

############################ 1. sim ############################
if [ "${SKIP_SIM:-0}" != 1 ]; then
  say "1/4  sim container + simulation"
  export DISPLAY=:0
  xhost +local: >/dev/null 2>&1 || true
  docker start "$CONTAINER" >/dev/null
  if ! sim_up; then
    docker exec -d "$CONTAINER" bash -lc "DISPLAY=:0 $SIM_SH > /tmp/sim.log 2>&1"
    echo "  launched simulation, waiting for Unity + topics..."
    for _ in $(seq 1 40); do
      sim_up && dros "ros2 topic list 2>/dev/null | grep -q /registered_scan" && break
      sleep 1
    done
  fi
  sim_up && echo "  Unity alive" || echo "  WARNING: Unity (Model.x86_64) not running"
  n=$(dros "ros2 topic list 2>/dev/null | grep -cE 'camera|registered_scan|way_point'" 2>/dev/null || echo 0)
  echo "  sim topics visible: $n"
fi

########################## 2. deps (one-time) ##########################
if [ "${SKIP_INSTALL:-0}" = 1 ]; then
  say "2/4  skipping deps (SKIP_INSTALL=1)"
elif [ "${REINSTALL:-0}" != 1 ] && dexec "test -x /opt/percep/bin/python"; then
  say "2/4  deps already present (/opt/percep) — skipping"
else
  say "2/4  installing perception deps into container (~3GB, one-time)"
  docker exec -u root "$CONTAINER" bash -lc "
    apt-get update -qq && apt-get install -y -qq python3-pip python3-venv &&
    python3 -m venv --system-site-packages /opt/percep &&
    /opt/percep/bin/pip install -q ultralytics open3d scipy pillow ftfy regex \
      'git+https://github.com/ultralytics/CLIP.git'
  "
fi

############################ 3. code ############################
say "3/4  syncing perception code into container"
docker exec "$CONTAINER" mkdir -p /tmp/percep/perception
docker cp "$DG/branchA_gt.py" "$CONTAINER":/tmp/percep/branchA_gt.py
for f in lift3d detect objectmap eval_objectmap viz3d live_perception; do
  docker cp "$DG/perception/$f.py" "$CONTAINER":/tmp/percep/perception/"$f".py
done
docker cp "$DG/perception/names.json" "$CONTAINER":/tmp/percep/names.json
echo "  synced branchA_gt.py + 5 modules + names.json"

############################ 4. worker ############################
say "4/4  (re)starting live perception worker on $DEVICE (result-dir=$RESULT_DIR$([ "$BENCHMARK" = 1 ] && echo ', benchmark'))"
dexec "pkill -f live_perception" >/dev/null 2>&1 || true
sleep 1
WORKER_ARGS="--names /tmp/percep/names.json --device $DEVICE --result-dir $RESULT_DIR"
[ "$BENCHMARK" = 1 ] && WORKER_ARGS="$WORKER_ARGS --benchmark --legend-dir $LEGEND_DIR"
docker exec -d "$CONTAINER" bash -lc "
  source /opt/ros/jazzy/setup.bash; export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  cd /tmp/percep/perception
  /opt/percep/bin/python live_perception.py $WORKER_ARGS > /tmp/live_percep.log 2>&1
"
sleep 4
dexec "grep -vE 'MiB/s|[0-9]+%' /tmp/live_percep.log | tail -3" 2>/dev/null || true

cat <<EOF

== ready ==  (first keyframe ~15s while YOLO/SAM weights download on a fresh container)

rviz  (server desktop, Add -> By topic):
  /perception/objects        -> MarkerArray
  /perception/object_points  -> PointCloud2   (set Color Transformer = RGB8)

control (server terminal; first: source /opt/ros/jazzy/setup.bash; export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp):
  filter one class : ros2 topic pub --once /perception/filter std_msgs/msg/String "{data: couch}"
  show all         : ros2 topic pub --once /perception/filter std_msgs/msg/String "{data: all}"
  reset map        : ros2 topic pub --once /perception/reset  std_msgs/msg/Empty  "{}"
  drive            : ros2 topic pub --once /way_point geometry_msgs/msg/PointStamped "{header: {frame_id: map}, point: {x: 3.0, y: -1.0, z: 0.0}}"

logs: docker exec $CONTAINER tail -f /tmp/live_percep.log
EOF
