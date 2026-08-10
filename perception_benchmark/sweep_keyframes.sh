#!/usr/bin/env bash
# Sweep the ScanAccumulator window and tabulate the effect on scores.
#
#   perception_benchmark/sweep_keyframes.sh arabic_room            # score each k
#   DUMP=1 perception_benchmark/sweep_keyframes.sh arabic_room      # viewer dumps
#   K_VALUES="0 1 5 10" DUMP=1 perception_benchmark/sweep_keyframes.sh arabic_room
#
# DUMP=1 writes a viewer dump per k (debug_k<N>/) instead of scores. Point the
# viewer at one with PERCEPTION_DEBUG_DIR. Dumps cost ~2.5 min/scene more than
# scoring (matplotlib overlays) and ~320 MB each, so pick the k values you
# actually want to look at.
#
# The sidecar is touched exactly once, to freeze the detection masks
# (dump_detections.py). Detection depends only on the image, the class list and
# the score threshold — none of which this sweep varies — so every run after
# that replays those masks and needs no GPU.
#
# Cost is dominated by the LIFT, not the detector, and lift scales with the
# accumulated cloud: ~5 ms/detection at k=1 rising to ~76 ms at k=40. A large-k
# run is therefore slower than a live baseline even with the sidecar removed.
# Budget roughly 1.5 min per k-step around k=5 on a 316-frame scene.
#
# Env: K_VALUES, CAP_DIR, OUT_ROOT, EXTRA (extra replay_score flags).

set -uo pipefail   # no -e: one bad k must not abandon the rest of the sweep

REPO="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
SCENE="${1:?usage: sweep_keyframes.sh <scene>}"
K_VALUES="${K_VALUES:-0 1 2 3 4 5 6 7 8 9 10}"
CAP_DIR="${CAP_DIR:-perception_benchmark/captures_nav}"
OUT_ROOT="${OUT_ROOT:-perception_benchmark/sweeps/keyframes}"

cd "$REPO"
export PERCEPTION_CAP_DIR="$CAP_DIR"

n_frames=$(ls -d "$CAP_DIR/$SCENE"/vp_* 2>/dev/null | wc -l)
[ "$n_frames" = 0 ] && { echo "no captures under $CAP_DIR/$SCENE" >&2; exit 1; }

# 1. Freeze the masks once (skipped when they are already there).
missing=$(( n_frames - $(find "$CAP_DIR/$SCENE" -name detections.npz | wc -l) ))
if [ "$missing" -gt 0 ]; then
  echo "==== freezing detection masks for $SCENE ($missing/$n_frames missing) ===="
  uv run --extra perception python perception_benchmark/dump_detections.py \
    --scene "$SCENE" || { echo "dump_detections failed" >&2; exit 1; }
else
  echo "==== masks already frozen for $SCENE ($n_frames frames) ===="
fi

# 2. Sweep. Each run reads the frozen masks, so no sidecar is involved.
for k in $K_VALUES; do
  echo; echo "==== $SCENE  scan-keyframes=$k ===="
  start=$SECONDS
  if [ "${DUMP:-0}" = "1" ]; then
    uv run --extra perception python perception_benchmark/dump_debug.py \
      --scene "$SCENE" --scan-keyframes "$k" \
      --out "${DUMP_ROOT:-perception_benchmark}/debug_k$k" \
      ${EXTRA:-} 2>&1 | tail -2
  else
    uv run --extra perception python perception_benchmark/replay_score.py \
      --scene "$SCENE" --scan-keyframes "$k" --out "$OUT_ROOT/k$k" \
      ${EXTRA:-} 2>&1 | tail -5
  fi
  echo "  (took $((SECONDS-start))s)"
done

if [ "${DUMP:-0}" = "1" ]; then
  echo; echo "==== dumps written ===="
  for k in $K_VALUES; do
    d="${DUMP_ROOT:-perception_benchmark}/debug_k$k"
    printf "  k=%-3s %s  %s\n" "$k" "$(du -sh "$d" 2>/dev/null | cut -f1)" "$d"
  done
  echo
  echo "view one with:"
  echo "  PERCEPTION_DEBUG_DIR=${DUMP_ROOT:-perception_benchmark}/debug_k5 \\"
  echo "    uv run --with streamlit --with plotly streamlit run perception_benchmark/viz_app.py"
  exit 0
fi

# 3. One table across the sweep.
echo; echo "==================== SUMMARY: $SCENE ===================="
uv run python - "$OUT_ROOT" "$SCENE" "$K_VALUES" <<'PY'
import json, sys
from pathlib import Path

root, scene, ks = Path(sys.argv[1]), sys.argv[2], sys.argv[3].split()
print(f"{'k':>3} {'mAP@1':>7} {'P@1':>7} {'R@1':>7} {'F1@1':>7} "
      f"{'cErr':>6} {'cMAE':>7} {'pred':>5}")
for k in ks:
    f = root / f"k{k}" / f"{scene}_metrics.json"
    if not f.is_file():
        print(f"{k:>3}  (missing)")
        continue
    r = json.loads(f.read_text())
    op = r["operating_point"].get("dist@1.0m", {})
    print(f"{k:>3} {r['mAP']['dist@1.0m']:>7} {op.get('precision'):>7} "
          f"{op.get('recall'):>7} {op.get('f1'):>7} "
          f"{op.get('mean_center_err_m'):>6} {r['counting_MAE']:>7} "
          f"{r['n_pred']:>5}")
print("\nk=0 is no accumulation (raw sweep). GT count:",
      json.loads((root / f"k{ks[0]}" / f"{scene}_metrics.json").read_text())["n_gt"]
      if (root / f"k{ks[0]}" / f"{scene}_metrics.json").is_file() else "?")
PY
