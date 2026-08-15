#!/usr/bin/env bash
# Run frontier then nbv back-to-back over the same scenes, then compare them.
#
#   scripts/exploration_ab.sh                  # every scene found
#   scripts/exploration_ab.sh chinese_room     # one scene (do this first)
#   LABEL=exp2 scripts/exploration_ab.sh       # name the archive
#
# Artefacts do not collide on their own — the node keys them on scene *and*
# strategy (exploration_logs/<scene>/<strategy>/), and each sweep writes its own
# results_<strategy>.csv. What this adds is:
#
#   * one scene list, resolved once and passed to both sweeps explicitly, so the
#     paired comparison is over identical scenes even if the scenes dir changes
#     underneath a long run;
#   * one budget, pinned here rather than left to a compose default, because the
#     previous frontier sweep ran without a wall-clock cutoff and nbv ran with
#     one — which was the biggest confound in that comparison (TASK 50);
#   * an archive per strategy under exploration_logs_<strategy>_<LABEL>/, so the
#     *next* A/B does not overwrite this one;
#   * a transcript, and the comparison table at the end.
#
# Env: LABEL, SCENES_DIR, MAX_SECONDS, MAX_WAYPOINTS, TIMEOUT — plus anything
# scripts/exploration_sweep.sh reads, which this passes straight through.

# No -e: a failing strategy must not skip the archive, the comparison, or the
# other strategy. An overnight run that dies silently at 2am is the failure
# mode this whole script exists to avoid.
set -uo pipefail

REPO=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
LOGS=$REPO/exploration_logs
LABEL=${LABEL:-exp1}
STRATEGIES=(frontier nbv)

# Both strategies get the same cutoff. 480s is the node default; naming it here
# means the transcript records what this run actually used.
export XIAO_HEI_EXPLORATION_MAX_SECONDS=${MAX_SECONDS:-480}

# Resolve the scene list once. Same candidate list as exploration_sweep.sh —
# duplicated rather than sourced because that script runs a sweep on load.
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
  echo "$HOME/Downloads/unity_env_models"
}
SCENES_DIR=${SCENES_DIR:-$(_discover_scenes_dir)}
export SCENES_DIR

scenes=("$@")
if [ ${#scenes[@]} -eq 0 ]; then
  for d in "$SCENES_DIR"/*/; do
    [ -d "$d/environment" ] && scenes+=("$(basename "$d")")
  done
fi
[ ${#scenes[@]} -gt 0 ] || { echo "no scenes under $SCENES_DIR" >&2; exit 2; }

mkdir -p "$LOGS"
transcript=$LOGS/ab_$LABEL.log

# Refuse to start on top of an existing archive. The per-strategy archive is
# rm -rf'd at the end of each sweep, so a re-run under the same LABEL silently
# destroys the previous experiment's evidence — and it would do it five hours
# in, long after the operator stopped watching. Checked up front, before the
# first container starts.
clashes=()
for strategy in "${STRATEGIES[@]}"; do
  [ -e "$REPO/exploration_logs_${strategy}_${LABEL}" ] \
    && clashes+=("exploration_logs_${strategy}_${LABEL}")
done
if [ ${#clashes[@]} -gt 0 ] && [ "${FORCE:-0}" != "1" ]; then
  {
    echo "refusing to run: LABEL='$LABEL' would overwrite ${clashes[*]}"
    echo "  pick a new label:   LABEL=exp2 $0 $*"
    echo "  or overwrite:       FORCE=1 LABEL=$LABEL $0 $*"
  } >&2
  exit 3
fi

# Per scene: the exploration budget, plus 90-190s waiting for /state_estimation
# before the clock even starts, plus sim boot and teardown.
per_scene_min=$(( (XIAO_HEI_EXPLORATION_MAX_SECONDS + 300) / 60 ))
total_h=$(( ${#scenes[@]} * ${#STRATEGIES[@]} * per_scene_min / 60 ))

{
  echo "=============================================================="
  echo "A/B sweep '$LABEL' starting $(date '+%F %T')"
  echo "  strategies : ${STRATEGIES[*]}"
  echo "  budget     : ${XIAO_HEI_EXPLORATION_MAX_SECONDS}s per scene, both"
  echo "  scenes (${#scenes[@]}) : ${scenes[*]}"
  echo "  scenes dir : $SCENES_DIR"
  echo "  rough ETA  : ~${total_h}h (~${per_scene_min} min/scene incl. sim startup)"
  echo "=============================================================="
} | tee -a "$transcript"

for strategy in "${STRATEGIES[@]}"; do
  echo | tee -a "$transcript"
  echo "##### $strategy — started $(date '+%F %T')" | tee -a "$transcript"

  STRATEGY=$strategy "$REPO/scripts/exploration_sweep.sh" "${scenes[@]}" 2>&1 \
    | tee -a "$transcript"

  # The sweep tears down per scene, but a scene that timed out can leave the
  # stack up, and the next strategy needs the GPU the sim is holding.
  docker compose -f "$REPO/docker/compose.yml" -f "$REPO/docker/compose.scene.yml" \
    down --remove-orphans >/dev/null 2>&1

  # Archive this strategy's tree, preserving <scene>/<strategy>/ so
  # compare_exploration.py reads the archive directly.
  dest=$REPO/exploration_logs_${strategy}_${LABEL}
  rm -rf "$dest"
  for scene in "${scenes[@]}"; do
    [ -d "$LOGS/$scene/$strategy" ] || continue
    mkdir -p "$dest/$scene"
    cp -a "$LOGS/$scene/$strategy" "$dest/$scene/"
  done
  cp -a "$LOGS/results_$strategy.csv" "$dest/" 2>/dev/null
  echo "##### $strategy — finished $(date '+%F %T'), archived to $dest" \
    | tee -a "$transcript"
done

{
  echo
  echo "=============================================================="
  echo "A/B sweep '$LABEL' finished $(date '+%F %T')"
  echo "=============================================================="
  echo
  # Reads exploration_logs/, which now holds both strategies side by side.
  cd "$REPO" && uv run python scripts/compare_exploration.py "$LOGS" 2>&1
} | tee -a "$transcript"

echo
echo "transcript: $transcript"
