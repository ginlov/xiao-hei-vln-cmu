#!/usr/bin/env bash
# scripts/on_host.sh — build a python on the sim box that can run the loop.
#
#   scripts/on_host.sh setup     create .venv-drive and install the deps
#   scripts/on_host.sh check     what is missing, if anything
#
# That is all this does, because it is the only part with no alternative: the
# system python on these boxes has neither pip nor the venv module, so nothing
# can be installed into it without root.
#
# Running the loop needs no wrapper. Two exports and the venv's python:
#
#   export XIAO_HEI_SIM_HOST=local        # this machine is the sim box
#   export ANTHROPIC_API_KEY=sk-ant-...
#
#   ./scripts/sim.sh restart home_building_2
#   .venv-drive/bin/python scripts/execute_plan.py "<question>" --out runs/x
#
# `XIAO_HEI_SIM_HOST=local` is what makes sim.sh, drive.sh and the loop talk to
# the local docker instead of ssh-ing somewhere. Use tmux if you want the run to
# outlive the ssh session.
#
# The key is a plain export, and on a shared box that is the right default: it
# dies with the shell. If you tire of retyping it, put it in a file only you can
# read — NOT ~/.bashrc, which leaks it into every process you start:
#
#   mkdir -p ~/.config/xiao-hei && chmod 700 ~/.config/xiao-hei
#   ( umask 077; printf 'export ANTHROPIC_API_KEY=%s\n' 'sk-ant-...' \
#       > ~/.config/xiao-hei/env )
#   source ~/.config/xiao-hei/env
#
# Use a key you can revoke separately from your laptop's: ~/workspace on these
# boxes holds several people's directories.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
VENV="$REPO/.venv-drive"
CTR="${XIAO_HEI_SIM_CONTAINER:-iros2026_system}"
PY="3.12"

say()  { printf '\033[1m%s\033[0m\n' "$*" >&2; }
warn() { printf '\033[33m%s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31mon_host.sh: %s\033[0m\n' "$*" >&2; exit 1; }
usage() { sed -n '2,33p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 1; }

# `ssh box 'cmd'` runs a non-login shell, which on these boxes does not put
# ~/.local/bin on PATH — so a scripted check reports uv missing while it is
# perfectly present for the person at the terminal.
find_uv() {
  command -v uv 2>/dev/null && return 0
  for c in "$HOME/.local/bin/uv" /usr/local/bin/uv; do
    [ -x "$c" ] && { echo "$c"; return 0; }
  done
  return 1
}

cmd_setup() {
  local uv; uv="$(find_uv)" || die \
"no uv, and it is the only way in: the system python has neither pip nor the
venv module (\`python3 -m venv\` fails asking for python3-venv, which needs
root). uv is one static binary and installs without root:

  curl -LsSf https://astral.sh/uv/install.sh | sh"

  # Three states, not two. A half-made venv is the one that bites: a failed
  # `python3 -m venv` leaves the directory behind with no python in it, and
  # `uv venv` then refuses because something is already there.
  if [ -x "$VENV/bin/python" ]; then say "reusing $VENV"
  elif [ -e "$VENV" ]; then
    say "replacing a broken $VENV"
    "$uv" venv --clear --python "$PY" "$VENV"
  else
    say "creating $VENV"
    "$uv" venv --python "$PY" "$VENV"
  fi

  # Its own venv, not the repo's `.venv`: that one belongs to whatever else is
  # set up on this box.
  #
  # No ROS. `robot_io.py` is copied *into* the container and run there, so rclpy
  # and the message packages are the container's problem. scipy is the converter
  # model's cKDTree; pillow and pydantic are here only because `vlm_locate`
  # imports `xiao_hei_vln.perception.geometry` and that package's __init__
  # eagerly pulls in the ROS responder and its HTTP client.
  say "installing deps"
  VIRTUAL_ENV="$VENV" "$uv" pip install --python "$VENV/bin/python" \
    "numpy>=1.26" "opencv-python-headless>=4.8" "anthropic>=0.40" \
    "pillow>=10" "pydantic>=2.7" "scipy>=1.11"
  say "done — now: scripts/on_host.sh check"
}

cmd_check() {
  local bad=0
  printf 'repo    %s\n' "$REPO"
  printf 'head    %s\n' "$(git -C "$REPO" log --oneline -1 2>/dev/null | cut -c1-55 || echo '?')"

  if [ -x "$VENV/bin/python" ]; then
    printf 'python  %s\n' "$("$VENV/bin/python" --version 2>&1)"
    for m in numpy cv2 anthropic PIL pydantic scipy; do
      "$VENV/bin/python" -c "import $m" 2>/dev/null ||
        { printf '  MISSING %s\n' "$m"; bad=1; }
    done
    [ "$bad" -eq 0 ] && printf '  deps    all six present\n'
  else
    printf 'python  MISSING — scripts/on_host.sh setup\n'; bad=1
  fi

  if docker ps >/dev/null 2>&1; then
    if docker ps --format '{{.Names}}' | grep -qx "$CTR"; then
      printf 'sim     %s up\n' "$CTR"
    else
      printf 'sim     %s NOT running — ./scripts/sim.sh up <scene>\n' "$CTR"; bad=1
    fi
    docker ps --format '{{.Names}}' | grep -qx xiao_hei_ai_module &&
      warn 'ai_module is up and will fight the loop for /way_point_with_heading'
  else
    printf 'docker  unavailable — is this the sim box?\n'; bad=1
  fi

  if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    printf 'key     set (%d chars)\n' "${#ANTHROPIC_API_KEY}"
  else
    printf 'key     NOT set — export ANTHROPIC_API_KEY=...\n'; bad=1
  fi

  case "${XIAO_HEI_SIM_HOST:-}" in
    local|localhost|127.0.0.1) printf 'host    local — talks to this docker\n' ;;
    "") printf 'host    NOT set — export XIAO_HEI_SIM_HOST=local, or the scripts\n'
        printf '        will try to ssh to xiaohei1 from here\n'; bad=1 ;;
    *)  warn "host    XIAO_HEI_SIM_HOST=${XIAO_HEI_SIM_HOST} — not this box" ;;
  esac

  [ "$bad" -eq 0 ] && say "ready" || warn "not ready — see above"
  return 0
}

case "${1:-}" in
  setup) shift; cmd_setup "$@" ;;
  check) shift; cmd_check "$@" ;;
  *)     usage ;;
esac
