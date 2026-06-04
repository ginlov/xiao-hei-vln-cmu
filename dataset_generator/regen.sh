#!/usr/bin/env bash
# Regenerate the full VLA-3D Q&A corpus end-to-end:
#
#   ref_to_qa  →  num_gen  →  nested_gen  →  check_question_types
#
# `check_question_types` auto-merges the nested intermediate file before
# validating, so the merge step does not appear here explicitly.
#
# Path-independent: works from any cwd, always runs uv from the repo root.
# Fail-fast on any step.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

cd "$ROOT"

for step in vla3d_ref_to_qa vla3d_num_gen vla3d_nested_gen check_question_types; do
  echo "=== $step ==="
  uv run python "dataset_generator/$step.py"
done

echo
echo "Done. Outputs in $ROOT/dataset/"
