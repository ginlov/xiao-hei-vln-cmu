#!/usr/bin/env bash
# A/B for the falsification gate (XIAO_HEI_FALSIFY), driven from the laptop.
#
# Both arms run BYTE-IDENTICAL code and differ only in the environment variable,
# which is what makes this an A/B rather than two experiments.
#
# Arms are interleaved PER QUESTION -- f0 then f1 back to back on the same
# question, before moving on. An earlier version ran all questions on one arm
# and then all on the other, so being cut short left only the control arm and
# no comparison at all. Any prefix of this one is balanced.
set -uo pipefail
cd "$(dirname "$0")/.."
eval "$(grep -E '^[[:space:]]*export ANTHROPIC_API_KEY=' ~/.zshrc)"
[ -n "${ANTHROPIC_API_KEY:-}" ] || { echo "no ANTHROPIC_API_KEY in ~/.zshrc"; exit 1; }
export XIAO_HEI_SIM_HOST="${XIAO_HEI_SIM_HOST:-xiaohei1}"
STAMP=$(date +%m%d_%H%M); mkdir -p sweep_logs
PASSES="${1:-3}"

run() {                                    # run <scene> <tag> <arm> <question>
  local scene="$1" tag="$2" arm="$3" q="$4"
  local out="runs/ab_${STAMP}_${tag}_f${arm}_$(date +%H%M%S)"
  echo "=== $(date +%H:%M:%S)  $tag  falsify=$arm  ($scene)"
  ./scripts/sim.sh restart "$scene" >>"sweep_logs/${STAMP}_sim.log" 2>&1 || {
      echo "    sim restart FAILED"; return 1; }
  XIAO_HEI_FALSIFY="$arm" timeout 900 uv run --with anthropic python \
      scripts/execute_plan.py "$q" --host "$XIAO_HEI_SIM_HOST" --out "$out" \
      >>"sweep_logs/${STAMP}_${tag}_f${arm}.log" 2>&1
  echo "    exit=$? -> $out"
}

source "$(dirname "$0")/ab_questions.sh"

for pass in $(seq 1 "$PASSES"); do
  echo "########## pass $pass  $(date +%H:%M:%S)"
  for i in "${!AB_Q[@]}"; do
    run "${AB_S[$i]}" "${AB_T[$i]}" 0 "${AB_Q[$i]}"     # control
    run "${AB_S[$i]}" "${AB_T[$i]}" 1 "${AB_Q[$i]}"     # falsification
  done
done
echo "########## done $(date +%H:%M:%S)"
