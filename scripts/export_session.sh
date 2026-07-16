#!/usr/bin/env bash
# Export a VLM tick-logger session from the sim box to a local folder for
# offline LLM / perception debugging, and (optionally) build the HTML report.
#
# The online pipeline logs each session to vlm_logs/session_<ts>/ on the box
# (config + per-tick VLM I/O + scene graph + camera frames + point clouds +
# predictions). This pulls one down and runs scripts/generate_report.py so you
# get a self-contained report.html (camera playback, per-tick I/O table, pose
# trajectory, sensor BEV, latency).
#
# NOTE: ticks are only logged while a question is active — fire a few
# /challenge_question messages during the run or the session holds just
# session.json.
#
# Usage:
#   scripts/export_session.sh [SESSION]
#     SESSION   session dir name (session_20260711_052457) or its timestamp
#               (20260711_052457). Default: the latest session on the box.
#
# Env overrides:
#   REMOTE_HOST   ssh host of the sim box   (default: xiaohei1)
#   REMOTE_REPO   repo path on the box (home-relative) that holds vlm_logs/
#                 (default: workspace/chengkai/xiao-hei-vln-cmu)
#   OUT_ROOT      local output root         (default: ~/Downloads)
#   NO_REPORT=1   skip building report.html (raw data only)
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-xiaohei1}"
REMOTE_REPO="${REMOTE_REPO:-workspace/chengkai/xiao-hei-vln-cmu}"
OUT_ROOT="${OUT_ROOT:-$HOME/Downloads}"
SESSION="${1:-}"

# Resolve the session directory name on the box.
if [[ -z "$SESSION" ]]; then
  SESSION=$(ssh "$REMOTE_HOST" \
    "ls -td $REMOTE_REPO/vlm_logs/session_* 2>/dev/null | head -1 | xargs -r -n1 basename")
  [[ -z "$SESSION" ]] && { echo "No sessions found on $REMOTE_HOST:$REMOTE_REPO/vlm_logs" >&2; exit 1; }
elif [[ "$SESSION" != session_* ]]; then
  SESSION="session_$SESSION"
fi

TS="${SESSION#session_}"
DEST="$OUT_ROOT/percep_out_$TS"
echo "Exporting $SESSION  ->  $DEST"
mkdir -p "$DEST"
rsync -az "$REMOTE_HOST:$REMOTE_REPO/vlm_logs/$SESSION/" "$DEST/"

# Optional HTML report — needs this repo + uv + the 'replay' extra.
if [[ "${NO_REPORT:-}" != "1" ]]; then
  REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
  if command -v uv >/dev/null 2>&1; then
    echo "Building report.html ..."
    (cd "$REPO_ROOT" && uv run --extra replay python scripts/generate_report.py "$DEST/") \
      || echo "Report generation failed (raw data is still exported)." >&2
  else
    echo "uv not on PATH; skipping report (raw data exported)." >&2
  fi
fi

echo "Done: $DEST"
ls -1 "$DEST"
