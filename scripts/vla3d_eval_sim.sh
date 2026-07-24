#!/usr/bin/env bash
# Sim helpers sourced by run_scene_vla3d_eval.sh
# Expects: SYSTEM_CTR AI_CTR ROS_DOMAIN_ID DISPLAY_VAL PERCEPTION_PORT COMPOSE

ros_env() {
  # shellcheck disable=SC2016
  echo "source /opt/ros/jazzy/setup.bash
source /home/docker/autonomy_stack_mecanum_wheel_platform/install/setup.bash
export ROS_DOMAIN_ID=$ROS_DOMAIN_ID RMW_IMPLEMENTATION=rmw_cyclonedds_cpp"
}

kill_sim() {
  docker exec "$SYSTEM_CTR" bash -lc '
    for p in system_simulation.sh Model.x86_64 "ros2 launch vehicle_simulator" \
             rviz2 waypointConverter localPlanner pathFollower vehicleSimulator \
             keep_autonomy_joy; do
      pkill -f "$p" 2>/dev/null || true
    done
    sleep 2
  ' >/dev/null 2>&1 || true
}

start_sim() {
  # Unity needs DISPLAY at launch or lidar/camera never publish.
  docker exec -d "$SYSTEM_CTR" bash -lc "
    export DISPLAY=$DISPLAY_VAL ROS_DOMAIN_ID=$ROS_DOMAIN_ID RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    cd /home/docker/autonomy_stack_mecanum_wheel_platform
    nohup ./system_simulation.sh >/tmp/xiao_hei_system_simulation.log 2>&1 &
  "
}

arm_autonomy() {
  # Keep /joy armed so the local planner accepts waypoints.
  docker exec "$SYSTEM_CTR" bash -lc "
    $(ros_env)
    pkill -f keep_autonomy_joy_inner 2>/dev/null || true
    nohup bash -c \"
      $(ros_env)
      while true; do
        ros2 topic pub --once /joy sensor_msgs/msg/Joy \
          \\\"{axes: [0,0,-1,0,0,1,0,0], buttons: [0,0,0,0,0,0,0,0,0,0,0]}\\\" >/dev/null 2>&1
        sleep 5
      done
    \" >/tmp/autonomy_joy_keeper.log 2>&1 &
  " >/dev/null 2>&1 || true
}

wait_for_terrain() {
  local deadline=$((SECONDS + ${1:-90}))
  while (( SECONDS < deadline )); do
    if docker exec "$SYSTEM_CTR" bash -lc "
      $(ros_env)
      timeout 3 ros2 topic hz /terrain_map_ext 2>&1 | grep -q 'average rate'
    " >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

wait_perception() {
  local i=0
  while (( i < 60 )); do
    curl -sf "http://127.0.0.1:${PERCEPTION_PORT}/healthz" >/dev/null 2>&1 \
      || curl -sf "http://127.0.0.1:${PERCEPTION_PORT}/docs" >/dev/null 2>&1 \
      && return 0
    sleep 2
    i=$((i + 1))
  done
  return 1
}

publish_question() {
  local q=${1:-"How many chairs are in the room?"}
  local escaped=${q//\"/\\\"}
  docker exec "$SYSTEM_CTR" bash -lc "
    $(ros_env)
    ros2 topic pub --once /challenge_question std_msgs/msg/String \"{data: \\\"${escaped}\\\"}\"
  " >/dev/null
}

wait_exploration_done() {
  local scene=$1 log=$EXPLORE_LOGS/$scene/exploration.log started=$SECONDS
  echo "  waiting for exploration DONE (log=$log, timeout=${TIMEOUT}s)..."
  while ! grep -qE ' DONE |HARD_STOP' "$log" 2>/dev/null; do
    (( SECONDS - started > TIMEOUT )) && {
      echo "  timed out waiting for exploration after ${TIMEOUT}s" >&2; return 1; }
    [[ "$(docker inspect -f '{{.State.Running}}' "$AI_CTR" 2>/dev/null || true)" == "true" ]] \
      || { echo "  ai_module not running ($AI_CTR)" >&2; return 1; }
    sleep 5
  done
  sleep 5
  echo "  exploration finished: $(grep -E ' DONE |HARD_STOP' "$log" | tail -1)"
}
