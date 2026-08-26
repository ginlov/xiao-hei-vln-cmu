#!/usr/bin/env bash
# Naive-arm sweep -- the same 26 questions as `sweep_cv.sh`, with one thing
# changed: the waypoint's metres come from the model's own `distance_m` instead
# of the lidar lift. Bearing, standoff, converter prediction and everything
# downstream are held identical, so the pair of sweeps is an ablation of the
# range channel and nothing else.
#
#   ./scripts/sweep_naive.sh [passes]     default 1 pass = 26 runs, ~2-4 h
#
# Score the two arms against each other with the same scorer:
#   uv run python scripts/score_if.py --runs runs            # ours   (cv_*)
#   uv run python scripts/score_if.py --runs runs --glob 'nv_*'   # naive
#
# Why two passes matter here: our own Sec. 6.2 result is that 8 of 26 questions
# swing by half their marks between identical passes, so a single-pass arm is
# not comparable to a two-pass arm. Run the same number of passes as the corpus
# it is being compared against.
#
# The lift's refusals go with the lift -- the elevation-floor test and the
# implied-size gate have nothing to judge when the number is a model's guess --
# so this arm never abstains. That is part of the effect, not a confound.
set -uo pipefail
cd "$(dirname "$0")/.."
eval "$(grep -E '^[[:space:]]*export ANTHROPIC_API_KEY=' ~/.zshrc)"
[ -n "${ANTHROPIC_API_KEY:-}" ] || { echo "no ANTHROPIC_API_KEY in ~/.zshrc"; exit 1; }
export XIAO_HEI_SIM_HOST="${XIAO_HEI_SIM_HOST:-xiaohei1}"
export XIAO_HEI_FALSIFY=0
export XIAO_HEI_RANGE_FROM=model          # <-- the only difference from sweep_cv.sh
STAMP=$(date +%m%d_%H%M); mkdir -p sweep_logs
PASSES="${1:-1}"

source "$(dirname "$0")/ab_questions.sh"

run() {                                    # run <scene> <tag> <pass> <question>
  local scene="$1" tag="$2" pass="$3" q="$4"
  local name="${scene}_${tag}_p${pass}"
  local out="runs/nv_${STAMP}_${name}"
  echo "=== $(date +%H:%M:%S)  $name"
  ./scripts/sim.sh restart "$scene" >>"sweep_logs/${STAMP}_naive_sim.log" 2>&1 || {
      echo "    sim restart FAILED"; return 1; }
  timeout 900 uv run --with anthropic python \
      scripts/execute_plan.py "$q" --host "$XIAO_HEI_SIM_HOST" --out "$out" \
      >>"sweep_logs/${STAMP}_naive_${name}.log" 2>&1
  echo "    exit=$? -> $out"
}

for pass in $(seq 1 "$PASSES"); do
  echo "########## naive pass $pass  $(date +%H:%M:%S)"
  for i in "${!CV_Q[@]}"; do
    run "${CV_S[$i]}" "${CV_T[$i]}" "$pass" "${CV_Q[$i]}"
  done
done
echo "########## done $(date +%H:%M:%S)  -> runs/nv_${STAMP}_*"
