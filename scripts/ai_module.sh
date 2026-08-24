#!/usr/bin/env bash
# scripts/ai_module.sh — drive the *submission* module on a sim box.
#
# `sim.sh` starts the simulator; this starts the thing being graded, the same
# way the graders will: `ros2 launch dummy_vlm dummy_vlm.launch` inside
# `iros2026/ai_module:latest`, one question per container.
#
#   scripts/ai_module.sh sync            copy ai_module/ to the box
#   scripts/ai_module.sh build           sync, then build the image there
#   scripts/ai_module.sh selftest        capture + drive 2 m; no API key, no cost
#   scripts/ai_module.sh ask "<question>"  launch, publish, follow the log
#   scripts/ai_module.sh logs            follow the running module
#   scripts/ai_module.sh pull [dir]      copy the run artefacts back
#   scripts/ai_module.sh down            stop the module (leaves the sim up)
#
#   XIAO_HEI_SIM_HOST   ssh host (default: xiaohei1)
#   XIAO_HEI_CHALLENGE  challenge repo, local (default: ../CMU-VLN-Challenge-2026)
#
# The API key is read from the environment and handed over on **stdin**, never
# as an argument: an argument is visible in `ps` to every user on a shared box,
# and these boxes are shared. It lands in a 0600 file that is deleted as soon as
# the container has taken it.
set -euo pipefail

HOST="${XIAO_HEI_SIM_HOST:-xiaohei1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHALLENGE="${XIAO_HEI_CHALLENGE:-$(cd "$HERE/../CMU-VLN-Challenge-2026" && pwd)}"
CTR=xh_vln
IMAGE=iros2026/ai_module:latest
SIM=iros2026_system

# Every ros2 call needs this; the image sets RMW in its own env but a
# `docker exec` into the *sim* container does not inherit it.
ROSENV='source /opt/ros/jazzy/setup.bash && export ROS_DOMAIN_ID=0 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp'
IMGENV='source /opt/ros/jazzy/setup.bash && source /home/docker/ai_module/install/setup.bash'

say() { printf '\033[1m%s\033[0m\n' "$*" >&2; }
die() { printf 'ai_module.sh: %s\n' "$*" >&2; exit 1; }

cmd_sync() {
  say "sync ai_module/ -> $HOST:~/ai_module/"
  rsync -a --delete --exclude __pycache__ --exclude .DS_Store \
        "$CHALLENGE/ai_module/" "$HOST:~/ai_module/"
}

cmd_build() {
  cmd_sync
  say "build $IMAGE on $HOST"
  ssh "$HOST" 'cd ~/ai_module && docker/build.sh' 2>&1 | grep -E '^#|ERROR|error' | tail -12
}

cmd_down() {
  ssh "$HOST" "docker rm -f $CTR >/dev/null 2>&1 && echo '  $CTR removed' || true"
}

cmd_selftest() {
  ssh "$HOST" "docker run --rm --network=host -e ROS_DOMAIN_ID=0 \
      -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp $IMAGE \
      bash -c '$IMGENV && python3 /opt/xiao_hei/vlm/challenge_node.py --selftest'"
}

# Start the module the way the graders do, then publish one question at it.
cmd_ask() {
  local q="${1:-}"
  [ -n "$q" ] || die 'ask needs a question: ai_module.sh ask "Go to the ..."'
  [ -n "${ANTHROPIC_API_KEY:-}" ] || die 'ANTHROPIC_API_KEY is not set in this shell'
  ssh "$HOST" "docker ps --format '{{.Names}}' | grep -qx $SIM" \
    || die "$SIM is not running on $HOST — bring the simulator up first (scripts/sim.sh up)"

  cmd_down
  printf 'ANTHROPIC_API_KEY=%s\n' "$ANTHROPIC_API_KEY" \
    | ssh "$HOST" 'umask 077 && cat > /tmp/xh.env'
  say "launching the module on $HOST"
  ssh "$HOST" "docker run -d --name $CTR --network=host --env-file /tmp/xh.env \
        -e ROS_DOMAIN_ID=0 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
        ${XIAO_HEI_CLASSIFY:+-e XIAO_HEI_CLASSIFY=$XIAO_HEI_CLASSIFY} \
        ${XIAO_HEI_IMAGE_TOPIC:+-e XIAO_HEI_IMAGE_TOPIC=$XIAO_HEI_IMAGE_TOPIC} \
        ${XIAO_HEI_BUDGET_S:+-e XIAO_HEI_BUDGET_S=$XIAO_HEI_BUDGET_S} \
        $IMAGE bash -c '$IMGENV && ros2 launch dummy_vlm dummy_vlm.launch' >/dev/null \
     && sleep 6 && rm -f /tmp/xh.env"

  say "asking: $q"
  # The evaluation node publishes at 1 Hz; --once is enough because the module
  # subscribes RELIABLE and is already up by here.
  ssh "$HOST" "docker exec $SIM bash -lc '$ROSENV && timeout 20 ros2 topic pub --once \
        /challenge_question std_msgs/msg/String \"{data: \\\"$q\\\"}\"'" >/dev/null 2>&1
  say "following the log — ctrl-c stops watching, not the run"
  cmd_logs
}

cmd_logs() {
  ssh "$HOST" "docker logs -f $CTR 2>&1" | grep -vE '^\[INFO\] \[launch\]|process started'
}

cmd_pull() {
  local dest="${1:-runs/aimodule_$(date +%m%d_%H%M%S)}"
  mkdir -p "$HERE/$dest"
  ssh "$HOST" "docker exec $CTR bash -c 'cd /tmp/xiao_hei_run && tar cf - .'" \
    | tar -x -C "$HERE/$dest"
  ssh "$HOST" "docker logs $CTR 2>&1" > "$HERE/$dest/node.log"
  say "pulled to $dest"
  ls "$HERE/$dest"
}

case "${1:-}" in
  sync)     cmd_sync ;;
  build)    cmd_build ;;
  selftest) cmd_selftest ;;
  ask)      shift; cmd_ask "$*" ;;
  logs)     cmd_logs ;;
  pull)     shift; cmd_pull "${1:-}" ;;
  down)     cmd_down ;;
  *)        sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 1 ;;
esac
