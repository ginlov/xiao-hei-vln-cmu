#!/usr/bin/env bash
# Finish arm D: the questions the 20-run cut left behind.
#
#   XIAO_HEI_STAMP=0825_1816 ./scripts/sweep_arm_d_topup.sh
#
# Arm D was stopped at 20 of 26 to save time, which dropped office_1, office_2
# and studio. Those three scenes are where predicting the platform pays most --
# cutting to the 16 moved the w/o-Platform-Model arm by +5.6 pp against our own
# +1.3 -- so their absence is what costs the strongest comparison we have its
# significance. This restores them.
#
# Identical configuration to sweep_arm_d.sh; STAMP is inherited so the runs join
# the existing `dm_0825_1816_*` family rather than forming a second corpus that
# a later glob would have to remember to union. Any question whose directory
# already exists is skipped, so this is safe to re-run.
set -uo pipefail
cd "$(dirname "$0")/.."
eval "$(grep -E '^[[:space:]]*export ANTHROPIC_API_KEY=' ~/.zshrc)"
[ -n "${ANTHROPIC_API_KEY:-}" ] || { echo "no ANTHROPIC_API_KEY in ~/.zshrc"; exit 1; }
export XIAO_HEI_SIM_HOST="${XIAO_HEI_SIM_HOST:-xiaohei1}"
export XIAO_HEI_FALSIFY=0
export XIAO_HEI_RANGE_FROM=model
export XIAO_HEI_SKIP_CONVERTER=free
STAMP="${XIAO_HEI_STAMP:?set XIAO_HEI_STAMP to the stamp of the corpus being finished}"
mkdir -p sweep_logs
source "$(dirname "$0")/ab_questions.sh"

run() {
  local scene="$1" tag="$2" q="$3"
  local name="${scene}_${tag}_p1"
  local out="runs/dm_${STAMP}_${name}"
  [ -d "$out" ] && { echo "=== $(date +%H:%M:%S)  $name  ALREADY DONE"; return 0; }
  echo "=== $(date +%H:%M:%S)  $name"
  ./scripts/sim.sh restart "$scene" >>"sweep_logs/${STAMP}_dm_sim.log" 2>&1 || {
      echo "    sim restart FAILED"; return 1; }
  timeout 900 uv run --with anthropic python \
      scripts/execute_plan.py "$q" --host "$XIAO_HEI_SIM_HOST" --out "$out" \
      >>"sweep_logs/${STAMP}_dm_${name}.log" 2>&1
  echo "    exit=$? -> $out"
}

echo "arm d top-up: range=$XIAO_HEI_RANGE_FROM converter=$XIAO_HEI_SKIP_CONVERTER stamp=$STAMP"
for i in "${!CV_Q[@]}"; do run "${CV_S[$i]}" "${CV_T[$i]}" "${CV_Q[$i]}"; done
echo "########## arm d complete: $(ls -d runs/dm_${STAMP}_* | wc -l | tr -d ' ') runs  $(date +%H:%M:%S)"
