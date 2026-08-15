#!/usr/bin/env bash
# Run an exploration strategy across Unity scenes, one after another.
#
# Per scene: mount it into the sim, launch, wait for the explorer to log DONE,
# tear down. The node itself writes exploration.log, exploration.png and
# rviz.png into exploration_logs/<scene>/<strategy>/ — this just drives the
# loop and collates a results_<strategy>.csv at the end.
#
#   scripts/exploration_sweep.sh chinese_room        # one scene (do this first)
#   scripts/exploration_sweep.sh                     # every scene found
#   STRATEGY=nbv scripts/exploration_sweep.sh chinese_room
#
# The strategy is part of the artefact path, so sweeping one scene with a
# second STRATEGY leaves the first one's logs and PNGs intact.
#
# Env: SCENES_DIR, MAX_WAYPOINTS, STRATEGY, TIMEOUT, DISPLAY. SCENES_DIR is
# discovered when unset, so normally you set nothing but STRATEGY.
#
# If rviz.png is missing, RViz's window title may not match DEFAULT_MATCH in
# exploration/_capture.py. List the real titles with:
#   uv run python -c "from xiao_hei_vln.exploration import list_windows
#   print('\n'.join(list_windows()))"

# No -e: one bad scene must not abort the sweep.
set -uo pipefail

REPO=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)

# Scene location differs per box, so discover it — the same candidate list
# scripts/sim.sh walks. First directory holding an unpacked scene wins;
# SCENES_DIR overrides the search entirely.
_discover_scenes_dir() {
  local cand d
  for cand in \
      "$HOME/workspace/dataset/unity-scene" \
      "$HOME/workspace/dataset/unity_scenes_extracted" \
      "$HOME/workspace/dataset/unity_scenes" \
      "$HOME/Downloads/unity_env_models"; do
    [ -d "$cand" ] || continue
    for d in "$cand"/*/; do
      [ -d "$d/environment" ] && { echo "$cand"; return 0; }
    done
  done
  echo "$HOME/Downloads/unity_env_models"   # legacy default, for the error message
}
SCENES_DIR=${SCENES_DIR:-$(_discover_scenes_dir)}
MAX_WAYPOINTS=${MAX_WAYPOINTS:-500}
STRATEGY=${STRATEGY:-frontier}    # also the artefact subdirectory, see below
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
  rm -f "$LOGS/$scene/$STRATEGY"/exploration.log "$LOGS/$scene/$STRATEGY"/*.png 2>/dev/null
  log=$LOGS/$scene/$STRATEGY/exploration.log

  # The node names its log dir after this scene and strategy, so
  # exploration_logs/<scene>/<strategy>/ comes for free — no log-dir export.
  export XIAO_HEI_SCENE_DIR_HOST=$dir
  export XIAO_HEI_EXPLORATION_MAX_WAYPOINTS=$MAX_WAYPOINTS
  export XIAO_HEI_EXPLORATION_STRATEGY=$STRATEGY
  export XIAO_HEI_RESPONDER=dummy

  # --remove-orphans: the project name is pinned, so a container left by another
  # stack (e.g. the scene_gemini sidecar) would otherwise survive the whole
  # sweep, holding GPU memory the sim needs to render.
  "${COMPOSE[@]}" up -d --remove-orphans \
    || { echo "$scene: compose up failed" >&2; continue; }

  # `compose up -d` returns as soon as the containers are *started*, which is
  # before the docker daemon will accept an exec into them. Firing the sim
  # launch immediately loses it, the node then ticks with free=0 forever, and
  # the watchdog below reads the module as not-running and tears the whole
  # scene down within seconds — a run that dies in 3 s and reports
  # did-not-finish, with the containers (and their logs) already gone.
  for _ in $(seq 30); do
    [ "$(docker inspect -f '{{.State.Running}}' iros2026_system 2>/dev/null)" = "true" ] && break
    sleep 1
  done
  # Detached, unlike the `docker exec -it` used by hand — that would block here.
  if ! docker exec -d iros2026_system "$SIM_LAUNCH"; then
    echo "  sim launch failed — skipping scene" >&2
    "${COMPOSE[@]}" down
    continue
  fi

  # Same race on the other side: don't start the watchdog until the module is
  # actually up, or its first inspect decides the run is already over.
  for _ in $(seq 30); do
    [ "$(docker inspect -f '{{.State.Running}}' xiao_hei_ai_module 2>/dev/null)" = "true" ] && break
    sleep 1
  done

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

# Collate. Duration is measured from CLOCK_START (first tick with a pose) when
# the log has one, because the robot cannot explore before /state_estimation
# arrives and that wait is 90-190 s of dead time we did not spend exploring.
# Falls back to START for logs predating the clock.
#
# visited/skipped count decisions, not coverage. path_m and free_m2 are the
# ones to compare strategies on; reachable dropping away from free is a
# strategy walling itself in.
csv=$LOGS/results_$STRATEGY.csv
{
  echo "scene,duration_s,visited,skipped,path_m,free_m2,reachable,hatch_resets,select_why,reason"
  for scene in "${scenes[@]}"; do
    awk -v scene="$scene" '
      /\] START/       { gsub(/[][]/, "", $1); t_start = $1 }
      /\] CLOCK_START/ { gsub(/[][]/, "", $1); t_clock = $1 }
      /\] DONE/  {
        gsub(/[][]/, "", $1); t1 = $1
        t0 = (t_clock != "") ? t_clock : t_start
        for (i = 3; i <= NF; i++) { split($i, kv, "="); f[kv[1]] = kv[2] }
        printf "%s,%.1f,%s,%s,%s,%s,%s,%s,%s,%s\n", scene, t1 - t0,
               f["visited"], f["skipped"], f["path_m"], f["free_m2"],
               f["reachable"], f["hatch_resets"], f["select_why"], f["reason"]
        done = 1
      }
      END { if (!done) printf "%s,,,,,,,,,did-not-finish\n", scene }
    ' "$LOGS/$scene/$STRATEGY/exploration.log" 2>/dev/null || echo "$scene,,,,,,,,,no-log"
  done
} > "$csv"

echo && echo "wrote $csv" && column -t -s, "$csv"
