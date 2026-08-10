#!/usr/bin/env bash
# scripts/sim.sh — drive the simulator on the sim host, from the laptop.
#
#   scripts/sim.sh up [scene]        bring up sim + base autonomy, wait until ready
#   scripts/sim.sh down              stop everything, including the sidecar
#   scripts/sim.sh restart [scene]   down + up; this is also how you reset the pose
#   scripts/sim.sh status            what is running, and is it answering
#   scripts/sim.sh scenes            which scenes are unpacked, which are still zipped
#
# `up` blocks until /terrain_map is publishing, so the next command in a script
# can assume a live stack rather than sleeping and hoping.
#
#   XIAO_HEI_SIM_HOST       ssh host                 (default: xiaohei1)
#   XIAO_HEI_SIM_SCENE      scene when none is given (default: japanese_room)
#   XIAO_HEI_SIM_REPO       repo path on the box, relative to $HOME
#   XIAO_HEI_SIM_SCENES     unpacked scene dirs      (default: discovered)
#   XIAO_HEI_SIM_ARCHIVE    scene .zip dir           (default: discovered)
#   XIAO_HEI_SIM_CONTAINER  sim container name       (default: iros2026_system)
#
# Set XIAO_HEI_SIM_HOST once and it governs the loop too — `execute_plan.py`
# and `approach_loop.py` default `--host` to it, so a scene cannot be restarted
# on one box and driven on the other.
#
# The API key never comes here: this file only starts ROS. The loop itself runs
# on the laptop — see docs/guides/drive-loop-runbook.md.
#
# Every heredoc below is quoted, and paths reach the remote as arguments rather
# than by interpolation. An unquoted heredoc expands on *this* machine, which
# during development silently ran a backticked `docker/run down` from a comment
# against the laptop's own docker.

set -euo pipefail

HOST="${XIAO_HEI_SIM_HOST:-xiaohei1}"
DEFAULT_SCENE="${XIAO_HEI_SIM_SCENE:-japanese_room}"
# Relative to the remote $HOME, so they survive being passed as plain arguments.
REPO_REL="${XIAO_HEI_SIM_REPO:-workspace/chengkai/xiao-hei-vln-cmu}"
CTR="${XIAO_HEI_SIM_CONTAINER:-iros2026_system}"
READY_TIMEOUT=120

# Where the scenes live, which the two boxes do not agree on:
#
#   xiaohei1  workspace/dataset/unity-scene            unpacked dirs AND zips
#   xiaohei2  workspace/dataset/unity_scenes_extracted unpacked dirs
#             workspace/dataset/unity_scenes           zips
#
# So it is two paths, not one, and they are discovered on the box rather than
# assumed. Hardcoding xiaohei1's single directory is what made this script
# xiaohei1-only: every other part of it was already host-agnostic.
# An array, not a newline-joined string: `remote` ends up as `ssh host bash -s
# -- "$@"`, and ssh concatenates argv into ONE remote command line, so an
# embedded newline stops being data and becomes a command separator. That fails
# loudly on one box ("No such file or directory") and quietly on the other.
SCENE_CANDIDATES=(
  workspace/dataset/unity-scene
  workspace/dataset/unity_scenes_extracted
  workspace/dataset/unity_scenes
)

SCENES_REL="${XIAO_HEI_SIM_SCENES:-}"     # unpacked scene directories
ARCHIVE_REL="${XIAO_HEI_SIM_ARCHIVE:-}"   # the .zip files

say()  { printf '\033[1m%s\033[0m\n' "$*" >&2; }
die()  { printf 'sim.sh: %s\n' "$*" >&2; exit 1; }
usage() { sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 1; }

# The script body arrives on stdin; arguments follow `--`. When HOST names this
# machine there is no ssh at all, which is what lets the same script be run from
# a terminal on the box — see scripts/on_host.sh.
is_local() { case "${HOST:-}" in local|localhost|127.0.0.1|"") return 0;; *) return 1;; esac; }
remote() {
  if is_local; then bash -s -- "$@"; else ssh "$HOST" bash -s -- "$@"; fi
}

# Find the scene directories on whichever box this is, once per invocation.
# A candidate counts as the scenes directory if it holds at least one
# subdirectory, and as the archive if it holds at least one zip — which lets one
# directory be both, as it is on xiaohei1, without special-casing either box.
discover_scenes() {
  [ -n "$SCENES_REL" ] && [ -n "$ARCHIVE_REL" ] && return 0
  local found
  found="$(remote "${SCENE_CANDIDATES[@]}" <<'SH'
scenes=; archive=
for rel in "$@"; do
  d="$HOME/$rel"
  [ -d "$d" ] || continue
  if [ -z "$scenes" ] && [ -n "$(find "$d" -mindepth 1 -maxdepth 1 -type d -print -quit 2>/dev/null)" ]; then
    scenes="$rel"; echo "scenes=$rel"
  fi
  if [ -z "$archive" ] && [ -n "$(find "$d" -mindepth 1 -maxdepth 1 -name '*.zip' -print -quit 2>/dev/null)" ]; then
    archive="$rel"; echo "archive=$rel"
  fi
done
SH
)" || true
  [ -n "$SCENES_REL" ]  || SCENES_REL="$(printf '%s\n' "$found" | sed -n 's/^scenes=//p'  | head -1)"
  [ -n "$ARCHIVE_REL" ] || ARCHIVE_REL="$(printf '%s\n' "$found" | sed -n 's/^archive=//p' | head -1)"
  # An archive-only box can still be unpacked into; a box with neither cannot
  # be used at all, and saying so beats a mount of a path that does not exist.
  [ -n "$SCENES_REL" ] || SCENES_REL="$ARCHIVE_REL"
  [ -n "$ARCHIVE_REL" ] || ARCHIVE_REL="$SCENES_REL"
  [ -n "$SCENES_REL" ] || die "no unity scene directory on $HOST — looked in:
$(printf '  ~/%s\n' "${SCENE_CANDIDATES[@]}")
set XIAO_HEI_SIM_SCENES (and XIAO_HEI_SIM_ARCHIVE) to point at it"
}

cmd_scenes() {
  discover_scenes
  say "scenes on $HOST: ~/$SCENES_REL${ARCHIVE_REL:+, zips in ~/$ARCHIVE_REL}"
  remote "$SCENES_REL" "$ARCHIVE_REL" <<'SH'
scenes="$HOME/$1"; archive="$HOME/$2"
echo "unpacked:"
ls -d "$scenes"/*/ 2>/dev/null | xargs -n1 basename | sed 's/^/  /' || echo "  (none)"
echo "zipped only:"
for z in "$archive"/*.zip; do
  [ -e "$z" ] || continue
  n=$(basename "$z" .zip)
  [ -d "$scenes/$n" ] || echo "  $n"
done
SH
}

# Is the simulation actually publishing? Exit status only, so callers can poll.
# Two topics: /terrain_map and /terrain_map_ext.
ready() {
  remote "$CTR" <<'SH' >/dev/null 2>&1
n=$(docker exec "$1" bash -lc 'source /opt/ros/jazzy/setup.bash && ROS_DOMAIN_ID=0 \
      RMW_IMPLEMENTATION=rmw_cyclonedds_cpp timeout 8 ros2 topic list 2>/dev/null' \
    | grep -c '^/terrain_map') || n=0
[ "$n" -ge 2 ]
SH
}

cmd_status() {
  discover_scenes
  remote "$CTR" "$(basename "$SCENES_REL")" <<'SH'
ctr="$1"; scenes_dir="$2"
docker ps --format '{{.Names}}\t{{.Status}}' | sed 's/^/  /'
docker ps --format '{{.Names}}' | grep -qx "$ctr" || { echo "  simulation: not running"; exit 0; }

n=$(docker exec "$ctr" bash -lc 'source /opt/ros/jazzy/setup.bash && ROS_DOMAIN_ID=0 \
      RMW_IMPLEMENTATION=rmw_cyclonedds_cpp timeout 10 ros2 topic list 2>/dev/null' \
    | grep -c '^/terrain_map') || n=0
if [ "$n" -lt 2 ]; then
  echo "  simulation: container up but no topics — system_simulation.sh not running?"
  exit 0
fi
echo "  simulation: publishing"
p=$(docker exec "$ctr" bash -lc 'source /opt/ros/jazzy/setup.bash && ROS_DOMAIN_ID=0 \
      RMW_IMPLEMENTATION=rmw_cyclonedds_cpp timeout 10 \
      ros2 topic echo --once --field pose.pose.position /state_estimation 2>/dev/null' \
    | tr -d ' ' | grep -E '^[xyz]:' | paste -sd' ' -) || p=
[ -n "$p" ] && echo "  pose: $p"
scene=$(docker inspect "$ctr" --format '{{range .Mounts}}{{println .Source}}{{end}}' 2>/dev/null \
        | grep -F "/$scenes_dir/" | head -1 | xargs -r basename) || scene=
[ -n "$scene" ] && echo "  scene: $scene"
exit 0
SH
}

cmd_down() {
  say "stopping everything on $HOST"
  remote "$REPO_REL" <<'SH'
cd "$HOME/$1" && docker/run down 2>&1 | tail -3
# docker/run takes no responder here, so it carries no --profile perception and
# leaves the sidecar running. Stop it explicitly.
docker stop xiao_hei_perception >/dev/null 2>&1 &&
  echo " Container xiao_hei_perception Stopped" || true
exit 0
SH
}

# The scene currently bind-mounted, so `restart` with no argument reuses it
# instead of silently switching to the default.
current_scene() {
  discover_scenes
  remote "$CTR" "$(basename "$SCENES_REL")" <<'SH' 2>/dev/null || true
docker inspect "$1" --format '{{range .Mounts}}{{println .Source}}{{end}}' 2>/dev/null \
  | grep -F "/$2/" | head -1 | xargs -r basename
SH
}

cmd_up() {
  local scene="${1:-}"
  [ -n "$scene" ] || scene="$(current_scene)"
  [ -n "$scene" ] || scene="$DEFAULT_SCENE"
  # Again here, and not only inside `current_scene`: that one runs in a command
  # substitution, so what it discovers never reaches this scope. Without this
  # the scene directory would be empty and the mount would be "$HOME/" — a
  # container that starts, mounts the home directory, and renders nothing.
  discover_scenes

  # Always tear down first. `docker/run dummy up -d` is a no-op when the
  # container already runs the requested scene, but this script then launches a
  # *second* system_simulation.sh inside it — two stacks fighting over
  # /way_point_with_heading, which showed up as doubled publisher counts and a
  # drive that timed out. Idempotent beats fast here.
  cmd_down
  say "starting $scene on $HOST"
  remote "$scene" "$REPO_REL" "$SCENES_REL" "$ARCHIVE_REL" <<'SH'
set -eu
scene="$1"; repo="$HOME/$2"; data="$HOME/$3"; archive="$HOME/$4"

if [ ! -d "$data/$scene" ]; then
  if [ -f "$archive/$scene.zip" ]; then
    # Unpacked into the scenes directory, which is not always the one the zip
    # came from: on xiaohei2 the archives and the extracted scenes are separate.
    echo "  unpacking $scene.zip -> $data/"
    mkdir -p "$data"
    (cd "$data" && unzip -q "$archive/$scene.zip")
  else
    echo "sim.sh: no scene '$scene' — try: scripts/sim.sh scenes" >&2
    exit 2
  fi
fi

export DISPLAY=:0
# Unity renders to :0. Without a display it starts, publishes nothing, and
# looks healthy while doing it.
xhost +local: >/dev/null 2>&1 || true

cd "$repo"
XIAO_HEI_SCENE_DIR_HOST="$data/$scene" docker/run dummy up -d 2>&1 | tail -4
sleep 3

# Our own responder also publishes /way_point_with_heading and will fight the
# approach loop for control of the vehicle.
docker stop xiao_hei_ai_module >/dev/null 2>&1 && echo "  ai_module stopped" || true

# Bringing the container up does not start the simulation. This does.
docker exec -d iros2026_system bash -lc \
  'cd ~/autonomy_stack_mecanum_wheel_platform && DISPLAY=:0 ./system_simulation.sh > /tmp/sim.log 2>&1'
exit 0
SH

  say "waiting for topics"
  local waited=0
  while [ "$waited" -lt "$READY_TIMEOUT" ]; do
    if ready; then
      say "ready — $scene, robot at origin"
      return 0
    fi
    sleep 3
    waited=$((waited + 3))
  done
  die "no /terrain_map after ${READY_TIMEOUT}s — check: ssh $HOST 'docker exec $CTR cat /tmp/sim.log'"
}

case "${1:-}" in
  up)      shift; cmd_up "${1:-}" ;;
  down)    cmd_down ;;
  restart) shift
           cmd_up "${1:-$(current_scene)}" ;;   # `up` tears down first anyway
  status)  cmd_status ;;
  scenes)  cmd_scenes ;;
  *)       usage ;;
esac
