#!/usr/bin/env bash
# Run the frontier explorer across Unity scenes, one after another.
#
# Per scene: mount it into the sim, launch, wait for the explorer to log DONE,
# tear down. The node itself writes exploration.log, exploration.png and
# rviz.png into exploration_logs/<scene>/ — this just drives the loop and
# collates a results.csv at the end.
#
#   scripts/exploration_sweep.sh chinese_room        # one scene (do this first)
#   scripts/exploration_sweep.sh                     # every scene found
#
# Env: SCENES_DIR, MAX_WAYPOINTS, TIMEOUT, DISPLAY.
#
# If rviz.png is missing, RViz's window title may not match DEFAULT_MATCH in
# exploration/_capture.py. List the real titles with:
#   uv run python -c "from xiao_hei_vln.exploration import list_windows
#   print('\n'.join(list_windows()))"

# No -e: one bad scene must not abort the sweep.
set -uo pipefail

REPO=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
SCENES_DIR=${SCENES_DIR:-$HOME/Downloads/unity_env_models}
MAX_WAYPOINTS=${MAX_WAYPOINTS:-500}
TIMEOUT=${TIMEOUT:-1800}          # per scene, incl. the 90-190s of sim startup
export DISPLAY=${DISPLAY:-:0}

LOGS=$REPO/exploration_logs
# Create it ourselves before any `compose up`. Otherwise Docker auto-creates
# the bind-mount target as root, and the final `results.csv` write — plus the
# next run's `rm -rf $LOGS/<scene>` — fail with permission denied.
mkdir -p "$LOGS"
SIM_LAUNCH=/home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh
COMPOSE=(docker compose -f "$REPO/docker/compose.yml" -f "$REPO/docker/compose.scene.yml")

# Scenes: the ones named on the command line, else every dir that has a Unity
# player in it.
scenes=("$@")
if [ ${#scenes[@]} -eq 0 ]; then
  for d in "$SCENES_DIR"/*/; do
    [ -d "$d/environment" ] && scenes+=("$(basename "$d")")
  done
fi
[ ${#scenes[@]} -gt 0 ] || { echo "no scenes under $SCENES_DIR" >&2; exit 2; }

# The containers run as a different user than the desktop session, so X refuses
# them without this and RViz dies at startup.
xhost +local: >/dev/null 2>&1 || echo "warning: xhost failed; RViz may not start" >&2

echo "sweeping ${#scenes[@]} scene(s): ${scenes[*]}"

for scene in "${scenes[@]}"; do
  dir=$SCENES_DIR/$scene
  if [ ! -d "$dir/environment" ]; then
    echo "=== $scene: skipped, no $dir/environment" >&2
    continue
  fi

  echo && echo "=== $scene"
  # Don't rm the scene dir: the node opens exploration.log with mode "w" (which
  # truncates), and deleting it here while a lingering container still holds it
  # open orphans the file — writes vanish and only the PNGs (re-mkdir'd on DONE)
  # survive. Just clear the previous run's stale artefacts.
  rm -f "$LOGS/$scene"/exploration.log "$LOGS/$scene"/*.png 2>/dev/null
  log=$LOGS/$scene/exploration.log

  # The node names its log dir after this scene, so exploration_logs/<scene>/
  # comes for free — no separate log-dir export.
  export XIAO_HEI_SCENE_DIR_HOST=$dir
  export XIAO_HEI_EXPLORATION_MAX_WAYPOINTS=$MAX_WAYPOINTS
  export XIAO_HEI_RESPONDER=dummy

  "${COMPOSE[@]}" up -d || { echo "$scene: compose up failed" >&2; continue; }

  # Detached, unlike the `docker exec -it` used by hand — that would block here.
  docker exec -d iros2026_system "$SIM_LAUNCH"

  started=$SECONDS
  while ! grep -q " DONE " "$log" 2>/dev/null; do
    if (( SECONDS - started > TIMEOUT )); then
      echo "  timed out after ${TIMEOUT}s" >&2
      break
    fi
    if [ "$(docker inspect -f '{{.State.Running}}' xiao_hei_ai_module 2>/dev/null)" != "true" ]; then
      echo "  ai_module exited early" >&2
      break
    fi
    sleep 5
  done

  # The node writes its PNGs just after logging DONE — let them land.
  sleep 5
  "${COMPOSE[@]}" down
done

# Collate. Duration comes from the log's own START/DONE stamps, not wall clock,
# which would include the sim startup we did not spend exploring.
csv=$LOGS/results.csv
{
  echo "scene,duration_s,visited,skipped,reason"
  for scene in "${scenes[@]}"; do
    awk -v scene="$scene" '
      /\] START/ { gsub(/[][]/, "", $1); t0 = $1 }
      /\] DONE/  {
        gsub(/[][]/, "", $1); t1 = $1
        for (i = 3; i <= NF; i++) { split($i, kv, "="); f[kv[1]] = kv[2] }
        printf "%s,%.1f,%s,%s,%s\n", scene, t1 - t0, f["visited"], f["skipped"], f["reason"]
        done = 1
      }
      END { if (!done) printf "%s,,,,did-not-finish\n", scene }
    ' "$LOGS/$scene/exploration.log" 2>/dev/null || echo "$scene,,,,no-log"
  done
} > "$csv"

echo && echo "wrote $csv" && column -t -s, "$csv"
