#!/usr/bin/env bash
# scripts/on_host.sh — run the drive loop from a terminal ON the sim box.
#
#   scripts/on_host.sh setup                    one-off: venv + deps, no root needed
#   scripts/on_host.sh check                    is this box ready to drive?
#   scripts/on_host.sh sim restart <scene>      the sim, without ssh
#   scripts/on_host.sh run "<question>" [args]  drive a whole question
#   scripts/on_host.sh one "<phrase>"  [args]   drive to a single object
#
# The normal way to drive is from the laptop over ssh; see
# docs/guides/drive-loop-runbook.md. This is for when you want the run to
# survive your laptop closing, or the network dropping mid-question — a leg is
# 30 s per step and a question is up to ten minutes, and an interrupted run
# leaves the robot parked wherever it was and the log half-written.
#
# Pair it with tmux so the run outlives the ssh session too:
#
#   tmux new -s drive
#   export ANTHROPIC_API_KEY=...
#   scripts/on_host.sh run "Go near the magazine on the ottoman, ..."
#   # detach with ctrl-b d; come back with `tmux attach -t drive`
#
# THE API KEY. These boxes are shared — ~/workspace has several people's
# directories in it. Export the key for the session and let it die with the
# shell. Do NOT put it in ~/.bashrc, ~/.profile or any file in the repo: a key
# in a dotfile on a shared machine is readable by everyone with an account, and
# outlives the reason you needed it. This script never writes it anywhere.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
VENV="$REPO/.venv-drive"
CTR="${XIAO_HEI_SIM_CONTAINER:-iros2026_system}"
PY_MIN="3.12"

# Everything below drives the *local* docker. Exported so `sim.sh`, `drive.sh`
# and the loop all agree without being told twice.
export XIAO_HEI_SIM_HOST=local

say()  { printf '\033[1m%s\033[0m\n' "$*" >&2; }
warn() { printf '\033[33m%s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31mon_host.sh: %s\033[0m\n' "$*" >&2; exit 1; }
usage() { sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 1; }

# A venv of its own, deliberately not the repo's `.venv`: that one belongs to
# whatever else is set up on this box, and adding opencv to someone's working
# environment because you wanted to drive is not a trade worth making.
py() { "$VENV/bin/python" "$@"; }

# `ssh box 'cmd'` runs a non-login shell, which on these boxes does not put
# ~/.local/bin on PATH — so uv is invisible to a scripted check while being
# perfectly present for the person typing at the terminal. Look there directly
# rather than trust PATH.
find_uv() {
  if command -v uv >/dev/null 2>&1; then command -v uv; return 0; fi
  for c in "$HOME/.local/bin/uv" /usr/local/bin/uv; do
    [ -x "$c" ] && { echo "$c"; return 0; }
  done
  return 1
}

need_venv() {
  [ -x "$VENV/bin/python" ] ||
    die "no venv — run: scripts/on_host.sh setup"
}

cmd_setup() {
  local uv; uv="$(find_uv)" || die \
"no uv on this box, and it is the only way in: the system python has neither
pip nor the venv module (\`python3 -m venv\` fails asking for python3-venv,
which needs root). uv is a single static binary and installs into ~/.local/bin
without root:

  curl -LsSf https://astral.sh/uv/install.sh | sh

Then re-run this. If you would rather not pipe a script into a shell, the
release binaries are at https://github.com/astral-sh/uv/releases — put \`uv\`
on your PATH and this will find it."

  # Three states, not two. A *half-made* venv is the one that bites: a failed
  # `python3 -m venv` leaves the directory behind without a python in it, and
  # `uv venv` then refuses because something is already there.
  if [ -x "$VENV/bin/python" ]; then
    say "reusing $VENV"
  elif [ -e "$VENV" ]; then
    say "replacing a broken $VENV (no python in it)"
    "$uv" venv --clear --python "$PY_MIN" "$VENV"
  else
    say "creating $VENV (python $PY_MIN, via $uv)"
    "$uv" venv --python "$PY_MIN" "$VENV"
  fi

  # Only what runs on this side. `robot_io.py` is copied *into* the container
  # and run there, so rclpy and the ROS message packages are the container's
  # problem, not this box's — which is why this is a short list and not a ROS
  # install. Headless opencv for the same reason: nothing here opens a window,
  # and the GUI build drags in X libraries the box need not have.
  say "installing deps"
  # scipy is the converter model's: it puts every legal terrain point in a
  # cKDTree to answer "where would the organisers' node settle this waypoint",
  # which is what makes the loop's arrival test a prediction rather than a
  # guess.
  #
  # pillow and pydantic are not the loop's own doing: `vlm_locate` imports
  # `xiao_hei_vln.perception.geometry` for one transform, and that package's
  # __init__ eagerly pulls in the ROS responder and its HTTP client, which
  # import PIL. The dependency is on the import graph, not on anything this
  # side actually runs.
  VIRTUAL_ENV="$VENV" "$uv" pip install --python "$VENV/bin/python" \
    "numpy>=1.26" \
    "opencv-python-headless>=4.8" \
    "anthropic>=0.40" \
    "pillow>=10" \
    "pydantic>=2.7" \
    "scipy>=1.11"
  say "done — now: scripts/on_host.sh check"
}

cmd_check() {
  local bad=0
  printf 'repo      %s\n' "$REPO"
  # `--show-current` prints nothing on a detached HEAD, which is exactly what a
  # `git worktree add origin/<branch>` gives you — and an empty line there reads
  # as "no idea", when the commit below is perfectly informative.
  local br; br="$(git -C "$REPO" branch --show-current 2>/dev/null || true)"
  printf 'branch    %s\n' "${br:-(detached)}"
  printf 'head      %s\n' "$(git -C "$REPO" log --oneline -1 2>/dev/null | cut -c1-60 || echo '?')"

  if [ -x "$VENV/bin/python" ]; then
    printf 'venv      %s\n' "$(py --version 2>&1)"
    for m in numpy cv2 anthropic PIL pydantic scipy; do
      if py -c "import $m" 2>/dev/null; then printf '  ok      %s\n' "$m"
      else printf '  MISSING %s\n' "$m"; bad=1; fi
    done
  else
    printf 'venv      MISSING — scripts/on_host.sh setup\n'; bad=1
  fi

  if command -v docker >/dev/null && docker ps >/dev/null 2>&1; then
    if docker ps --format '{{.Names}}' | grep -qx "$CTR"; then
      printf 'sim       %s is up\n' "$CTR"
    else
      printf 'sim       %s NOT running — scripts/on_host.sh sim up <scene>\n' "$CTR"
      bad=1
    fi
    if docker ps --format '{{.Names}}' | grep -qx xiao_hei_ai_module; then
      warn 'ai_module is up and will fight the loop for /way_point_with_heading'
      warn '  docker stop xiao_hei_ai_module'
      bad=1
    fi
  else
    printf 'docker    unavailable — is this actually the sim box?\n'; bad=1
  fi

  if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    printf 'api key   set in this shell (%d chars)\n' "${#ANTHROPIC_API_KEY}"
  else
    printf 'api key   NOT set — export ANTHROPIC_API_KEY=... (this shell only)\n'
    bad=1
  fi
  [ "$bad" -eq 0 ] && say "ready" || warn "not ready — see above"
  return 0
}

require_key() {
  [ -n "${ANTHROPIC_API_KEY:-}" ] || die \
"ANTHROPIC_API_KEY is not set in this shell.

  export ANTHROPIC_API_KEY=...

Do not add it to ~/.bashrc on a shared box; export it per session so it dies
with the shell."
}

# The scene must be reset between runs — two runs from different starting poses
# are not comparable — and `sim.sh` already knows how. It reads the same
# XIAO_HEI_SIM_HOST this script exported, so it skips ssh and drives the local
# docker.
cmd_sim() { exec "$HERE/sim.sh" "$@"; }

cmd_run() {
  need_venv; require_key
  [ $# -ge 1 ] || die 'usage: on_host.sh run "<question>" [extra args]'
  local q="$1"; shift
  py "$HERE/execute_plan.py" "$q" "$@"
}

cmd_one() {
  need_venv; require_key
  [ $# -ge 1 ] || die 'usage: on_host.sh one "<object phrase>" [extra args]'
  local p="$1"; shift
  py "$HERE/approach_loop.py" "$p" "$@"
}

case "${1:-}" in
  setup) shift; cmd_setup "$@" ;;
  check) shift; cmd_check "$@" ;;
  sim)   shift; cmd_sim "$@" ;;
  run)   shift; cmd_run "$@" ;;
  one)   shift; cmd_one "$@" ;;
  *)     usage ;;
esac
