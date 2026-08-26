#!/usr/bin/env bash
# Arm D -- the baseline: model range, converter not modelled.
#
# The 2x2 with sweep_cv.sh (lidar+modelled) and sweep_naive.sh (model+modelled).
# Same 26 questions, same 13 scenes, same everything else.
#
#   ./scripts/sweep_arm_d.sh [passes]      default 1 pass = 26 runs, ~3 h
#
# 'free' is the steel-manned setting: the aim is clamped back along the ray to
# floor the terrain map can see. That is a free-space check, not a model of the
# converter's re-minimisation -- a naive team writes the first and not the
# second. Set XIAO_HEI_SKIP_CONVERTER=raw to measure what the check is worth.
set -uo pipefail
cd "$(dirname "$0")/.."
eval "$(grep -E '^[[:space:]]*export ANTHROPIC_API_KEY=' ~/.zshrc)"
[ -n "${ANTHROPIC_API_KEY:-}" ] || { echo "no ANTHROPIC_API_KEY in ~/.zshrc"; exit 1; }
export XIAO_HEI_SIM_HOST="${XIAO_HEI_SIM_HOST:-xiaohei1}"
export XIAO_HEI_FALSIFY=0
export XIAO_HEI_RANGE_FROM=model
export XIAO_HEI_SKIP_CONVERTER="${XIAO_HEI_SKIP_CONVERTER:-free}"
STAMP=$(date +%m%d_%H%M); mkdir -p sweep_logs
PASSES="${1:-1}"
source "$(dirname "$0")/ab_questions.sh"

run() {
  local scene="$1" tag="$2" pass="$3" q="$4"
  local name="${scene}_${tag}_p${pass}"
  local out="runs/dm_${STAMP}_${name}"
  echo "=== $(date +%H:%M:%S)  $name"
  ./scripts/sim.sh restart "$scene" >>"sweep_logs/${STAMP}_dm_sim.log" 2>&1 || {
      echo "    sim restart FAILED"; return 1; }
  timeout 900 uv run --with anthropic python \
      scripts/execute_plan.py "$q" --host "$XIAO_HEI_SIM_HOST" --out "$out" \
      >>"sweep_logs/${STAMP}_dm_${name}.log" 2>&1
  echo "    exit=$? -> $out"
}

echo "arm d: range_from=$XIAO_HEI_RANGE_FROM skip_converter=$XIAO_HEI_SKIP_CONVERTER"
for pass in $(seq 1 "$PASSES"); do
  echo "########## arm d pass $pass  $(date +%H:%M:%S)"
  for i in "${!CV_Q[@]}"; do run "${CV_S[$i]}" "${CV_T[$i]}" "$pass" "${CV_Q[$i]}"; done
done
echo "########## done $(date +%H:%M:%S)  -> runs/dm_${STAMP}_*"
