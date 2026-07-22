#!/usr/bin/env bash
# Detachable launcher for the live Unity+RViz bench.
# Survives closing the SSH/Cursor terminal (nohup + disown).
#
# Usage:
#   export XIAO_HEI_GEMINI_API_KEY='...'   # or put it in repo .env
#   ./scripts/start_live_rviz_bench_detached.sh
#
# Optional env (same as live_rviz_bench.sh):
#   SCENES="chinese_room arabic_room"
#   ALGOS="nbv rrt wall_follow"
#   RUN_TIMEOUT_S=480          # hard-stop each run at 8 minutes (default)
#   DISPLAY=:0
#
# Control:
#   tail -f exploration_logs/live_rviz/bench_runner.log
#   kill "$(cat exploration_logs/live_rviz/bench.pid)"
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$REPO/exploration_logs/live_rviz}"
LOG="$OUT_DIR/bench_runner.log"
PID_FILE="$OUT_DIR/bench.pid"
BENCH="$REPO/scripts/live_rviz_bench.sh"

mkdir -p "$OUT_DIR"

# Load repo .env if present (KEY=VAL lines only).
if [[ -f "$REPO/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO/.env"
  set +a
fi

: "${XIAO_HEI_GEMINI_API_KEY:?export XIAO_HEI_GEMINI_API_KEY (or put it in $REPO/.env)}"

export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
export RUN_TIMEOUT_S="${RUN_TIMEOUT_S:-480}"
export XIAO_HEI_GEMINI_API_KEY

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Bench already running (pid=$(cat "$PID_FILE"))."
  echo "Log: $LOG"
  exit 1
fi

chmod +x "$BENCH"
# Strip accidental CRLF if the file was edited on Windows.
sed -i 's/\r$//' "$BENCH" 2>/dev/null || true

# Close stdin; redirect all output; detach from terminal.
nohup "$BENCH" < /dev/null >> "$LOG" 2>&1 &
echo $! > "$PID_FILE"
disown || true

echo "Started live RViz bench in background."
echo "  pid:  $(cat "$PID_FILE")"
echo "  log:  $LOG"
echo "  out:  $OUT_DIR/<scene>/"
echo "  stop: kill \$(cat $PID_FILE)"
echo
echo "Follow progress:"
echo "  tail -f $LOG"
