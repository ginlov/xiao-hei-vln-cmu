#!/usr/bin/env bash
# The contemporaneous control: the SHIPPED system, re-driven on today's code.
#
#   ./scripts/sweep_control.sh [passes]    default 1 pass = 26 runs, ~2.2 h
#
# WHY THIS EXISTS
#
# `runs/cv_0818_*` is the baseline every ablation is measured against, and it
# was driven on 18 Aug. Two commits landed on 24 Aug --
#   d2196ea  two ways to refute a binding, and a parking brake
#   926f4b4  seven defects the object-reference orbit only showed on a robot
# -- and between them they rewrote `bind_target` and `run_goto` in
# `approach_loop.py`. `bind_target` is the function that decides a destination
# has been reached, so this is not cosmetic drift.
#
# Arms nv, C and D were all driven on 25 Aug, AFTER those commits. So every
# published difference against cv confounds the ablated factor with a week of
# code change, of unknown sign and size. The fixes plausibly HELP, which would
# make the ablation effects underestimates -- but that is a guess, and this
# sweep replaces it with a measurement.
#
# It does double duty: it is also the control arm E is read against, so that
# the memory ablation is never compared across a code boundary.
#
# Identical settings to sweep_cv.sh; only the run prefix differs, so the new
# corpus cannot be swept up by a `cv_0*` glob and silently pooled with the old.
set -uo pipefail
cd "$(dirname "$0")/.."
eval "$(grep -E '^[[:space:]]*export ANTHROPIC_API_KEY=' ~/.zshrc)"
[ -n "${ANTHROPIC_API_KEY:-}" ] || { echo "no ANTHROPIC_API_KEY in ~/.zshrc"; exit 1; }
export XIAO_HEI_SIM_HOST="${XIAO_HEI_SIM_HOST:-xiaohei1}"
export XIAO_HEI_FALSIFY=0
# Everything at the shipped default: lidar range, modelled converter, prose
# visited block. Stated rather than left implicit, so the control's
# configuration is legible in this file and not only in the defaults.
export XIAO_HEI_RANGE_FROM=lidar
export XIAO_HEI_SKIP_CONVERTER=off
export XIAO_HEI_VISITED=prose
STAMP=$(date +%m%d_%H%M); mkdir -p sweep_logs
PASSES="${1:-1}"
source "$(dirname "$0")/ab_questions.sh"

run() {
  local scene="$1" tag="$2" pass="$3" q="$4"
  local name="${scene}_${tag}_p${pass}"
  local out="runs/cn_${STAMP}_${name}"
  echo "=== $(date +%H:%M:%S)  $name"
  ./scripts/sim.sh restart "$scene" >>"sweep_logs/${STAMP}_cn_sim.log" 2>&1 || {
      echo "    sim restart FAILED"; return 1; }
  timeout 900 uv run --with anthropic python \
      scripts/execute_plan.py "$q" --host "$XIAO_HEI_SIM_HOST" --out "$out" \
      >>"sweep_logs/${STAMP}_cn_${name}.log" 2>&1
  echo "    exit=$? -> $out"
}

echo "control: range=$XIAO_HEI_RANGE_FROM converter=$XIAO_HEI_SKIP_CONVERTER visited=$XIAO_HEI_VISITED"
for pass in $(seq 1 "$PASSES"); do
  echo "########## control pass $pass  $(date +%H:%M:%S)"
  for i in "${!CV_Q[@]}"; do run "${CV_S[$i]}" "${CV_T[$i]}" "$pass" "${CV_Q[$i]}"; done
done
echo "########## done $(date +%H:%M:%S)  -> runs/cn_${STAMP}_*"
