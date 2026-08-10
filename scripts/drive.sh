#!/usr/bin/env bash
# scripts/drive.sh — publish one waypoint by hand and report what happened.
#
#   scripts/drive.sh 3.54 -2.30           drive to a map-frame point
#   scripts/drive.sh 3.54 -2.30 45        ... with a 45 s timeout
#   scripts/drive.sh where                just print the current pose
#
# Same channel the loop uses: a `Pose2D` on `/way_point_with_heading`, which is
# the topic the challenge scores. So a point this reaches is a point the AI
# module could have asked for, and a point it cannot reach is not a bug in the
# module.
#
# It prints the whole track the vehicle drove, not just the endpoint, because
# `local_planner` picks its own arc from a path library — where the robot went
# is not the straight line between the two waypoints, and for a passage
# question the difference is the entire answer.
#
#   XIAO_HEI_SIM_HOST       ssh host        (default: xiaohei1)
#   XIAO_HEI_SIM_CONTAINER  sim container   (default: iros2026_system)
#
# Same variables `sim.sh` and the loop read, so one export points all three at
# the same box.

set -euo pipefail

HOST="${XIAO_HEI_SIM_HOST:-xiaohei1}"
CTR="${XIAO_HEI_SIM_CONTAINER:-iros2026_system}"
# `local` means this machine is the sim host: docker exec, no ssh. `sh_run`
# then wraps the same command string either way, so the three call sites below
# stay identical.
case "$HOST" in local|localhost|127.0.0.1|"") HOST="";; esac
sh_run() {
  if [ -n "$HOST" ]; then ssh "$HOST" "$1"; else bash -lc "$1"; fi
}
HERE="$(cd "$(dirname "$0")" && pwd)"
ROS_ENV='source /opt/ros/jazzy/setup.bash && source /home/docker/autonomy_stack_mecanum_wheel_platform/install/setup.bash && export ROS_DOMAIN_ID=0 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && '

usage() { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 1; }
[ $# -ge 1 ] || usage

# The bridge may not be in the container yet if the loop has not run since the
# last restart, and pushing it costs nothing when it is.
sh_run "docker exec -i $CTR tee /tmp/robot_io.py >/dev/null" < "$HERE/robot_io.py"

report() {
  # Formatting lives in a file rather than inline, so the quoting survives
  # being nested inside ssh + docker exec + bash -lc.
  python3 "$HERE/_drive_report.py" "$@"
}

if [ "$1" = "where" ]; then
  sh_run "docker exec $CTR bash -lc '${ROS_ENV}python3 /tmp/robot_io.py capture'" \
    2>/dev/null | grep '^{' | report where
  exit 0
fi

[ $# -ge 2 ] || usage
X="$1"; Y="$2"; T="${3:-40}"

sh_run "docker exec $CTR bash -lc '${ROS_ENV}python3 /tmp/robot_io.py drive $X $Y --timeout $T'" \
  2>/dev/null | grep '^{' | report drive
