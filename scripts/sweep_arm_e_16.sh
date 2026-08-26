#!/usr/bin/env bash
# Arm E, restricted to the 8 scenes kept for the early result.
#
#   XIAO_HEI_STAMP=0825_1958 ./scripts/sweep_arm_e_16.sh [passes]
#
# Same ablation as sweep_arm_e.sh -- `XIAO_HEI_VISITED=off`, everything else at
# the shipped default -- over 16 of the 26 questions. The 16 are NOT a prefix of
# the sweep order: home_building_1 sits at positions 5-6 and loft at 19-20, so
# running the first N questions would not produce them and the list has to be
# filtered by scene instead.
#
# The excluded scenes are excluded for RUN TIME, not for anything measured about
# them. The other arms have all 26 and those numbers stay reported; this is an
# early read, and `补齐` means running DROP_SCENES here rather than re-running
# what is already done.
#
# STAMP is inherited so a resumed sweep lands in the same `vo_<stamp>_*` family
# as the questions already driven, and any question whose directory already
# exists is skipped rather than driven twice.
set -uo pipefail
cd "$(dirname "$0")/.."
eval "$(grep -E '^[[:space:]]*export ANTHROPIC_API_KEY=' ~/.zshrc)"
[ -n "${ANTHROPIC_API_KEY:-}" ] || { echo "no ANTHROPIC_API_KEY in ~/.zshrc"; exit 1; }
export XIAO_HEI_SIM_HOST="${XIAO_HEI_SIM_HOST:-xiaohei1}"
export XIAO_HEI_FALSIFY=0
export XIAO_HEI_VISITED=off
# Overridable so the same script can FINISH the arm: run it with
# XIAO_HEI_DROP_SCENES=" " and the 16 already driven are skipped by the
# directory check, leaving exactly the 10 that were never run.
DROP_SCENES="${XIAO_HEI_DROP_SCENES- home_building_1 loft office_1 office_2 studio }"
STAMP="${XIAO_HEI_STAMP:-$(date +%m%d_%H%M)}"; mkdir -p sweep_logs
PASSES="${1:-1}"
source "$(dirname "$0")/ab_questions.sh"

run() {
  local scene="$1" tag="$2" pass="$3" q="$4"
  local name="${scene}_${tag}_p${pass}"
  local out="runs/vo_${STAMP}_${name}"
  [ -d "$out" ] && { echo "=== $(date +%H:%M:%S)  $name  ALREADY DONE, skipping"; return 0; }
  echo "=== $(date +%H:%M:%S)  $name"
  ./scripts/sim.sh restart "$scene" >>"sweep_logs/${STAMP}_vo_sim.log" 2>&1 || {
      echo "    sim restart FAILED"; return 1; }
  timeout 900 uv run --with anthropic python \
      scripts/execute_plan.py "$q" --host "$XIAO_HEI_SIM_HOST" --out "$out" \
      >>"sweep_logs/${STAMP}_vo_${name}.log" 2>&1
  echo "    exit=$? -> $out"
}

echo "arm e (16): visited=$XIAO_HEI_VISITED  stamp=$STAMP  dropping:$DROP_SCENES"
for pass in $(seq 1 "$PASSES"); do
  echo "########## arm e-16 pass $pass  $(date +%H:%M:%S)"
  for i in "${!CV_Q[@]}"; do
    case "$DROP_SCENES" in *" ${CV_S[$i]} "*) continue;; esac
    run "${CV_S[$i]}" "${CV_T[$i]}" "$pass" "${CV_Q[$i]}"
  done
done
echo "########## done $(date +%H:%M:%S)  -> runs/vo_${STAMP}_*"
