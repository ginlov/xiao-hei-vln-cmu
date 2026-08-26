#!/usr/bin/env bash
# Arm E -- the memory ablation: our system, with the visited-places block
# removed from the prompt.
#
#   ./scripts/sweep_arm_e.sh [passes]      default 1 pass = 26 runs, ~2.2 h
#
# WHAT THIS REMOVES, AND WHAT IT DOES NOT
#
# `XIAO_HEI_VISITED=off` makes `Ctx.visited_for` return None, so `VISITED_BLOCK`
# -- "PLACES ALREADY SEARCHED ... Prefer somewhere it has not stood" -- is never
# built into the prompt. Nothing else changes: the range still comes from the
# lidar lift and the converter is still modelled, exactly as in the shipped
# system.
#
# It is a PROMPT-ONLY ablation, which was checked rather than assumed:
# `Ctx.visited` is read in exactly one place (`visited_for`), and the three
# call sites all feed the prompt. The navigation-side revisit machinery uses
# separate lists -- `spent` (approach_loop:577), `stood` (:770) and `reached`
# (execute_plan:250) -- and none of them touch `Ctx.visited`. So the robot
# still refuses to re-drive a spent bearing; only the MODEL loses the record.
#
# PRIOR EVIDENCE (docs/tasks/TASK 45), all of it per-call, none of it driven:
#   * the model reads the block -- removing it perturbs the explore heading
#     more than re-rolling the identical prompt does, 22/30 steps, p = 0.016
#   * but on the block's own stated goal, keeping away from places already
#     stood, the advantage is 0.06 m -- and two samples of the SAME prompt also
#     differ by 0.06 m
# TASK 45 closes with "Nothing here has been driven." This sweep is that.
set -uo pipefail
cd "$(dirname "$0")/.."
eval "$(grep -E '^[[:space:]]*export ANTHROPIC_API_KEY=' ~/.zshrc)"
[ -n "${ANTHROPIC_API_KEY:-}" ] || { echo "no ANTHROPIC_API_KEY in ~/.zshrc"; exit 1; }
export XIAO_HEI_SIM_HOST="${XIAO_HEI_SIM_HOST:-xiaohei1}"
export XIAO_HEI_FALSIFY=0
export XIAO_HEI_VISITED=off
STAMP=$(date +%m%d_%H%M); mkdir -p sweep_logs
PASSES="${1:-1}"
source "$(dirname "$0")/ab_questions.sh"

run() {
  local scene="$1" tag="$2" pass="$3" q="$4"
  local name="${scene}_${tag}_p${pass}"
  local out="runs/vo_${STAMP}_${name}"
  echo "=== $(date +%H:%M:%S)  $name"
  ./scripts/sim.sh restart "$scene" >>"sweep_logs/${STAMP}_vo_sim.log" 2>&1 || {
      echo "    sim restart FAILED"; return 1; }
  timeout 900 uv run --with anthropic python \
      scripts/execute_plan.py "$q" --host "$XIAO_HEI_SIM_HOST" --out "$out" \
      >>"sweep_logs/${STAMP}_vo_${name}.log" 2>&1
  echo "    exit=$? -> $out"
}

echo "arm e: visited=$XIAO_HEI_VISITED  (range=lidar, converter=modelled)"
for pass in $(seq 1 "$PASSES"); do
  echo "########## arm e pass $pass  $(date +%H:%M:%S)"
  for i in "${!CV_Q[@]}"; do run "${CV_S[$i]}" "${CV_T[$i]}" "$pass" "${CV_Q[$i]}"; done
done
echo "########## done $(date +%H:%M:%S)  -> runs/vo_${STAMP}_*"
