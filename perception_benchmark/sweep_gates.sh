#!/usr/bin/env bash
# Sweep the B4 (range cap) and B5 (SAM mask-quality) gates together, at a fixed
# scan-keyframe window, and tabulate the score trade-off.
#
#   perception_benchmark/sweep_gates.sh arabic_room
#   K=2 RANGE_CAPS="none 4.5 3.5" SAM_THRESH="0 0.85 0.90" \
#       perception_benchmark/sweep_gates.sh arabic_room
#
# Both gates are pure replay filters over the frozen masks (detections.npz must
# carry sam_scores — re-run dump_detections after the B5 pipeline change), so
# every cell replays the same masks and needs no GPU. "none" = no range cap;
# SAM threshold 0 = keep all masks. The (none, 0) cell is the baseline.

set -uo pipefail
REPO="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
SCENE="${1:?usage: sweep_gates.sh <scene>}"
K="${K:-2}"
RANGE_CAPS="${RANGE_CAPS:-none 6.0 4.5 3.5}"
SAM_THRESH="${SAM_THRESH:-0 0.80 0.85 0.90}"
CAP_DIR="${CAP_DIR:-perception_benchmark/captures_nav}"
OUT_ROOT="${OUT_ROOT:-perception_benchmark/sweeps/gates}"

cd "$REPO"
export PERCEPTION_CAP_DIR="$CAP_DIR"

for cap in $RANGE_CAPS; do
  for sam in $SAM_THRESH; do
    capflag=(); [ "$cap" != "none" ] && capflag=(--range-cap "$cap")
    tag="r${cap}_s${sam}"
    echo; echo "==== $SCENE  k=$K  range-cap=$cap  sam-thresh=$sam ===="
    uv run --extra perception python perception_benchmark/replay_score.py \
      --scene "$SCENE" --scan-keyframes "$K" --sam-thresh "$sam" "${capflag[@]}" \
      --out "$OUT_ROOT/$tag" 2>&1 | tail -3
  done
done

echo; echo "==================== SUMMARY: $SCENE (k=$K) ===================="
uv run python - "$OUT_ROOT" "$SCENE" "$RANGE_CAPS" "$SAM_THRESH" <<'PY'
import json, sys
from pathlib import Path
root, scene, caps, sams = Path(sys.argv[1]), sys.argv[2], sys.argv[3].split(), sys.argv[4].split()
print(f"{'range':>6} {'sam':>5} {'mAP@1':>7} {'P@1':>7} {'R@1':>7} {'F1@1':>7} "
      f"{'cErr':>6} {'cMAE':>7} {'pred':>5}")
gt = None
for cap in caps:
    for sam in sams:
        f = root / f"r{cap}_s{sam}" / f"{scene}_metrics.json"
        if not f.is_file():
            print(f"{cap:>6} {sam:>5}  (missing)"); continue
        r = json.loads(f.read_text()); gt = r["n_gt"]
        op = r["operating_point"].get("dist@1.0m", {})
        print(f"{cap:>6} {sam:>5} {r['mAP']['dist@1.0m']:>7} {op.get('precision'):>7} "
              f"{op.get('recall'):>7} {op.get('f1'):>7} {op.get('mean_center_err_m'):>6} "
              f"{r['counting_MAE']:>7} {r['n_pred']:>5}")
print(f"\nGT count: {gt}   (range 'none' + sam 0 = baseline, no gates)")
PY
