#!/usr/bin/env bash
# Re-arm autonomyMode on the CMU stack (axes[2] <= -0.1).
# Interval must be > joyToSpeedDelay (2s) so /speed can set joySpeed.
set -euo pipefail
docker exec aryan_iros2026_system bash -lc '
source /opt/ros/jazzy/setup.bash
source /home/docker/autonomy_stack_mecanum_wheel_platform/install/setup.bash
pkill -f "keep_autonomy_joy_inner" 2>/dev/null || true
nohup bash -c "
# keep_autonomy_joy_inner
source /opt/ros/jazzy/setup.bash
source /home/docker/autonomy_stack_mecanum_wheel_platform/install/setup.bash
while true; do
  ros2 topic pub --once /joy sensor_msgs/msg/Joy \"{axes: [0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0], buttons: [0,0,0,0,0,0,0,0,0,0,0]}\" >/dev/null 2>&1
  sleep 5
done
" >/tmp/autonomy_joy_keeper.log 2>&1 &
echo started
'
