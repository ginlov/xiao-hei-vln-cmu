#!/usr/bin/env bash
# Bring up chinese_room + NBV on Aryan-only containers (won't replace Rajath's).
set -euo pipefail
cd "$(dirname "$0")/.."

export DISPLAY="${DISPLAY:-:0}"
xhost +local: >/dev/null 2>&1 || true

export XIAO_HEI_GEMINI_API_KEY="${XIAO_HEI_GEMINI_API_KEY:?export XIAO_HEI_GEMINI_API_KEY first}"
export XIAO_HEI_VLM_LOG_DIR=/vlm_logs
export XIAO_HEI_EXPLORATION_LOG_DIR=/exploration_logs
export XIAO_HEI_EXPLORATION_STRATEGY=nbv
export XIAO_HEI_EXPLORATION_MAX_WAYPOINT_DIST=3.0
export XIAO_HEI_EXPLORATION_MAX_WAYPOINTS=100
export XIAO_HEI_SCENE_DIR_HOST=/home/ubuntu/Downloads/unity_env_models/chinese_room

docker compose -p aryan_nbv \
  -f docker/compose_scene_gemini.yml \
  -f docker/compose.scene.yml \
  -f docker/compose.aryan.yml \
  up -d --force-recreate

echo
echo "SSH/stack is up (project=aryan_nbv)."
echo "Containers: aryan_iros2026_system, aryan_xiao_hei_perception, aryan_xiao_hei_ai_module"
echo "ROS_DOMAIN_ID=42 (isolated from Rajath's default domain)."
echo
echo ">>> In DCV run:"
echo "export DISPLAY=:0"
echo "xhost +local:"
echo "docker exec -it aryan_iros2026_system \\"
echo "  /home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh"
echo
echo ">>> After Unity is up, start waypoint converter:"
echo "docker exec aryan_iros2026_system bash -lc '"
echo "  source /opt/ros/jazzy/setup.bash"
echo "  source /home/docker/autonomy_stack_mecanum_wheel_platform/install/setup.bash"
echo "  export ROS_DOMAIN_ID=42"
echo "  nohup ros2 run waypoint_converter waypointConverter >/tmp/waypoint_converter.log 2>&1 &'"
echo
echo ">>> Recreate AI so exploration starts against live sim:"
echo "docker compose -p aryan_nbv -f docker/compose_scene_gemini.yml -f docker/compose.scene.yml -f docker/compose.aryan.yml up -d --force-recreate ai_module"
echo
echo ">>> Logs: docker logs -f aryan_xiao_hei_ai_module"
