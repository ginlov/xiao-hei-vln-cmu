#!/usr/bin/env bash
# Thin wrapper around run_scene_vla3d_eval.sh for documented e2e offline eval.
#
# Examples:
#   scripts/run_e2e_offline_eval.sh --gt-only --limit 5 --splits ref,num studio
#   scripts/run_e2e_offline_eval.sh --limit 5 studio          # explore + live offline
#   scripts/run_e2e_offline_eval.sh --skip-explore --limit 5 studio
set -uo pipefail
REPO=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
exec "$REPO/scripts/run_scene_vla3d_eval.sh" "$@"
