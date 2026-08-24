#!/usr/bin/env bash
# Coverage sweep -- every instruction question on every unpacked scene, at the
# SHIPPED configuration. This is the corpus the paper's main table is built on.
#
#   ./scripts/sweep_cv.sh [passes]        default 1 pass = 26 runs, ~4 h
#
# The arm is pinned to XIAO_HEI_FALSIFY=0 on purpose: the table has to describe
# the system as it ships, and the gate is default-off. That costs nothing for
# the falsification question, because `falsify_audit.py` re-derives offline --
# from these same `steps.jsonl` files -- how often the gate WOULD have fired and
# how often the arbitration ladder even reaches the branch it lives in. A live
# A/B is only worth server time if that rate turns out not to be ~0.
#
# Scene order is grouped, so the sim restarts between questions but never
# between scenes more than once.
set -uo pipefail
cd "$(dirname "$0")/.."
eval "$(grep -E '^[[:space:]]*export ANTHROPIC_API_KEY=' ~/.zshrc)"
[ -n "${ANTHROPIC_API_KEY:-}" ] || { echo "no ANTHROPIC_API_KEY in ~/.zshrc"; exit 1; }
export XIAO_HEI_SIM_HOST="${XIAO_HEI_SIM_HOST:-xiaohei1}"
export XIAO_HEI_FALSIFY=0
STAMP=$(date +%m%d_%H%M); mkdir -p sweep_logs
PASSES="${1:-1}"

source "$(dirname "$0")/ab_questions.sh"

# CV_T repeats across scenes (`home_bq4` is both home_building_1 and _2), so the
# scene goes in the run name too -- otherwise two different questions land under
# names that only a timestamp tells apart.
run() {                                    # run <scene> <tag> <pass> <question>
  local scene="$1" tag="$2" pass="$3" q="$4"
  local name="${scene}_${tag}_p${pass}"
  local out="runs/cv_${STAMP}_${name}"
  echo "=== $(date +%H:%M:%S)  $name"
  ./scripts/sim.sh restart "$scene" >>"sweep_logs/${STAMP}_sim.log" 2>&1 || {
      echo "    sim restart FAILED"; return 1; }
  timeout 900 uv run --with anthropic python \
      scripts/execute_plan.py "$q" --host "$XIAO_HEI_SIM_HOST" --out "$out" \
      >>"sweep_logs/${STAMP}_${name}.log" 2>&1
  echo "    exit=$? -> $out"
}

for pass in $(seq 1 "$PASSES"); do
  echo "########## pass $pass  $(date +%H:%M:%S)"
  for i in "${!CV_Q[@]}"; do
    run "${CV_S[$i]}" "${CV_T[$i]}" "$pass" "${CV_Q[$i]}"
  done
done
echo "########## done $(date +%H:%M:%S)"
