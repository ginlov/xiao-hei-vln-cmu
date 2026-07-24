#!/usr/bin/env bash
# Per-scene: live explore → dump scene graph → Gemini offline on VLA-3D Qs.
#
# Usage:
#   export XIAO_HEI_GEMINI_API_KEY=<key>
#   scripts/run_scene_vla3d_eval.sh --gt-only --limit 5 --splits ref,num studio
#   scripts/run_scene_vla3d_eval.sh --limit 5 studio
#   scripts/run_scene_vla3d_eval.sh --skip-explore --limit 5 studio
#
# Env: SCENES_DIR, GT_DIR / XIAO_HEI_GT_DIR, OUT_DIR,
#      MAX_WAYPOINTS, MAX_SECONDS, STRATEGY, TIMEOUT, SPLITS, LIMIT_Q,
#      COMPOSE_PROJECT, XIAO_HEI_EVAL_PREFIX, ROS_DOMAIN_ID, PERCEPTION_PORT
# See docker/compose.eval.yml for the isolated eval stack.

set -uo pipefail

REPO=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
# shellcheck source=vla3d_eval_sim.sh
source "$REPO/scripts/vla3d_eval_sim.sh"

SCENES_DIR=${SCENES_DIR:-$HOME/Downloads/unity_env_models}
OUT_DIR=${OUT_DIR:-$REPO/artifacts/scene_vla3d_eval}
MAX_WAYPOINTS=${MAX_WAYPOINTS:-100}
MAX_SECONDS=${MAX_SECONDS:-540}
STRATEGY=${STRATEGY:-nbv}
TIMEOUT=${TIMEOUT:-$(( MAX_SECONDS + 600 ))}
SPLITS=${SPLITS:-ref,num}
LIMIT_Q=${LIMIT_Q:-}
SKIP_EXPLORE=0
GT_ONLY=0
NUM_SCENES=""
DECLARED_SCENES=()

COMPOSE_PROJECT=${COMPOSE_PROJECT:-xiao_hei_eval}
XIAO_HEI_EVAL_PREFIX=${XIAO_HEI_EVAL_PREFIX:-xiao_hei_eval}
ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
PERCEPTION_PORT=${PERCEPTION_PORT:-8002}
SYSTEM_CTR=${SYSTEM_CTR:-${XIAO_HEI_EVAL_PREFIX}_iros2026_system}
AI_CTR=${AI_CTR:-${XIAO_HEI_EVAL_PREFIX}_ai_module}
DISPLAY_VAL=${DISPLAY:-:0}
export DISPLAY="$DISPLAY_VAL" XIAO_HEI_EVAL_PREFIX ROS_DOMAIN_ID PERCEPTION_PORT
COMPOSE=(docker compose -p "$COMPOSE_PROJECT"
  -f "$REPO/docker/compose_scene_gemini.yml"
  -f "$REPO/docker/compose.scene.yml"
  -f "$REPO/docker/compose.eval.yml")

if [[ -z "${GT_DIR:-}" ]]; then
  for _cand in "${XIAO_HEI_GT_DIR:-}" \
    "$REPO/../dataset/xiao-hei-vln-cmu/dataset" \
    "$HOME/workspace/dataset/xiao-hei-vln-cmu/dataset" \
    "/home/ubuntu/workspace/dataset/xiao-hei-vln-cmu/dataset"
  do
    [[ -n "$_cand" && -f "$_cand/vla3d_ref.jsonl" ]] && GT_DIR=$_cand && break
  done
fi
GT_DIR=${GT_DIR:-}

usage() { sed -n '2,14p' "$0" | sed 's/^# \?//'; exit 2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-scenes) NUM_SCENES=$2; shift 2 ;;
    --limit) LIMIT_Q=$2; shift 2 ;;
    --skip-explore) SKIP_EXPLORE=1; shift ;;
    --gt-only) GT_ONLY=1; SKIP_EXPLORE=1; shift ;;
    --splits) SPLITS=$2; shift 2 ;;
    --help|-h) usage ;;
    --*) echo "unknown flag: $1" >&2; usage ;;
    *) DECLARED_SCENES+=("$1"); shift ;;
  esac
done

[[ -n "${XIAO_HEI_GEMINI_API_KEY:-}" ]] || {
  echo "error: export XIAO_HEI_GEMINI_API_KEY first" >&2; exit 2; }
[[ -n "$GT_DIR" && ( -f "$GT_DIR/vla3d_ref.jsonl" || -f "$GT_DIR/vla3d_num.jsonl" ) ]] || {
  echo "error: set GT_DIR to a dir with vla3d_ref.jsonl / vla3d_num.jsonl" >&2; exit 2; }

mkdir -p "$OUT_DIR"/{explored_scenes,gt,preds,metrics,traces,debug}
EXPLORE_LOGS=$REPO/exploration_logs
VLM_LOGS=$REPO/vlm_logs
mkdir -p "$EXPLORE_LOGS" "$VLM_LOGS"
export GT_DIR

mapfile -t VLA_SCENES < <(python3 - <<'PY'
import json, os
from pathlib import Path
scenes = set()
for name in ("vla3d_ref.jsonl", "vla3d_num.jsonl"):
    p = Path(os.environ["GT_DIR"]) / name
    if p.is_file():
        for line in p.open():
            if line.strip():
                scenes.add(json.loads(line)["scene"])
print("\n".join(sorted(scenes)))
PY
)

available=()
if [[ ${#DECLARED_SCENES[@]} -gt 0 ]]; then
  available=("${DECLARED_SCENES[@]}")
else
  for s in "${VLA_SCENES[@]}"; do
    [[ -d "$SCENES_DIR/$s/environment" ]] && available+=("$s")
  done
fi
[[ -n "$NUM_SCENES" ]] && available=("${available[@]:0:$NUM_SCENES}")
[[ ${#available[@]} -gt 0 ]] || {
  echo "error: no scenes selected (SCENES_DIR=$SCENES_DIR GT_DIR=$GT_DIR)" >&2; exit 2; }

echo "=== scene VLA-3D eval ==="
echo "scenes (${#available[@]}): ${available[*]}"
echo "skip_explore=$SKIP_EXPLORE gt_only=$GT_ONLY strategy=$STRATEGY splits=$SPLITS limit=${LIMIT_Q:-all}"
echo "out=$OUT_DIR"
echo
xhost +local: >/dev/null 2>&1 || echo "warning: xhost failed; RViz may not start" >&2

filter_gt() {
  local scene=$1 split=$2 out=$3 src=$GT_DIR/vla3d_${split}.jsonl
  [[ -f "$src" ]] || { echo "  missing $src" >&2; return 1; }
  GT_DIR="$GT_DIR" SCENE="$scene" SPLIT="$split" OUT="$out" LIMIT_Q="${LIMIT_Q:-}" python3 - <<'PY'
import json, os
from pathlib import Path
src = Path(os.environ["GT_DIR"]) / f"vla3d_{os.environ['SPLIT']}.jsonl"
out, scene = Path(os.environ["OUT"]), os.environ["SCENE"]
limit = os.environ.get("LIMIT_Q") or ""
n = int(limit) if limit.strip() else None
rows = []
for line in src.open():
    if not line.strip():
        continue
    row = json.loads(line)
    if row.get("scene") == scene:
        rows.append(row)
        if n is not None and len(rows) >= n:
            break
out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""))
print(f"  filtered {len(rows)} {os.environ['SPLIT']} questions -> {out}")
PY
}

export_live_dump() {
  local scene=$1
  local live_dump=$EXPLORE_LOGS/$scene/scene_live.json
  # Prefer periodic AI dump (reliable); session export is optional bonus.
  if [[ -f "$live_dump" ]]; then
    echo "  exporting $live_dump"
    mkdir -p "$OUT_DIR/explored_scenes/$scene"
    cp -f "$live_dump" "$OUT_DIR/explored_scenes/$scene/scene.json"
  else
    echo "  ERROR: missing $live_dump" >&2
    return 1
  fi
  [[ -f "$EXPLORE_LOGS/$scene/exploration.png" ]] && \
    cp -f "$EXPLORE_LOGS/$scene/exploration.png" \
      "$OUT_DIR/explored_scenes/$scene/occupancy.png"
  return 0
}

explore_scene() {
  local scene=$1 dir=$SCENES_DIR/$scene
  [[ -d "$dir/environment" ]] || {
    echo "  skip explore: missing $dir/environment" >&2; return 1; }

  mkdir -p "$EXPLORE_LOGS/$scene"
  rm -f "$EXPLORE_LOGS/$scene"/exploration.log "$EXPLORE_LOGS/$scene"/*.png 2>/dev/null || true
  : > "$EXPLORE_LOGS/$scene/exploration.log" 2>/dev/null || true

  export XIAO_HEI_SCENE_DIR_HOST=$dir
  export XIAO_HEI_EXPLORATION_MAX_WAYPOINTS=$MAX_WAYPOINTS
  export XIAO_HEI_EXPLORATION_MAX_SECONDS=$MAX_SECONDS
  export XIAO_HEI_EXPLORATION_STRATEGY=$STRATEGY
  export XIAO_HEI_EXPLORATION_LOG_DIR=/exploration_logs
  export XIAO_HEI_VLM_LOG_DIR=/vlm_logs
  export XIAO_HEI_RESPONDER=scene_gemini
  export XIAO_HEI_OBJECT_MAP=${XIAO_HEI_OBJECT_MAP:-1}
  export XIAO_HEI_VIEWPOINT_RADIUS=${XIAO_HEI_VIEWPOINT_RADIUS:-0.8}
  export XIAO_HEI_WP_REACHED_M=${XIAO_HEI_WP_REACHED_M:-1.35}
  export XIAO_HEI_REF_SPATIAL=${XIAO_HEI_REF_SPATIAL:-1}
  export XIAO_HEI_SCENE_DUMP_PATH=/exploration_logs/${scene}/scene_live.json

  echo "  compose up (scene=$scene strategy=$STRATEGY max_s=$MAX_SECONDS)..."
  kill_sim
  "${COMPOSE[@]}" up -d || { echo "  compose up failed" >&2; return 1; }
  wait_perception || echo "  warning: perception health check timed out" >&2

  start_sim
  if ! wait_for_terrain 120; then
    echo "  ERROR: /terrain_map_ext never published — Unity likely missing DISPLAY" >&2
    docker exec "$SYSTEM_CTR" bash -lc 'tail -40 /tmp/xiao_hei_system_simulation.log' || true
    return 1
  fi
  arm_autonomy
  # Recreate AI after terrain is live so exploration starts with a belief map.
  "${COMPOSE[@]}" up -d --force-recreate ai_module || {
    echo "  ai_module recreate failed" >&2; return 1; }
  sleep 8
  arm_autonomy
  publish_question || true

  wait_exploration_done "$scene" || return 1
  export_live_dump "$scene" || return 1

  echo "  tearing down compose..."
  kill_sim
  "${COMPOSE[@]}" down || true
  sleep 5
}

run_offline_for_scene() {
  local scene=$1 object_source=live
  if [[ "$GT_ONLY" -eq 1 ]]; then
    object_source=gt
  elif [[ ! -f "$OUT_DIR/explored_scenes/$scene/scene.json" ]]; then
    echo "  missing live dump — run without --skip-explore first" >&2
    return 1
  fi

  local IFS=',' split
  for split in $SPLITS; do
    split=${split// /}
    [[ -n "$split" ]] || continue
    local gt=$OUT_DIR/gt/${scene}_${split}.jsonl
    local pred=$OUT_DIR/preds/${scene}_${split}.jsonl
    local metrics=$OUT_DIR/metrics/${scene}_${split}.json
    local trace=$OUT_DIR/traces/${scene}_${split}.jsonl
    local debug_dir=$OUT_DIR/debug/${scene}_${split}
    filter_gt "$scene" "$split" "$gt" || continue
    [[ -s "$gt" ]] || { echo "  no $split questions for $scene — skip"; continue; }

    rm -f "$trace"; rm -rf "$debug_dir"; mkdir -p "$debug_dir"
    local src_args=(--object-source "$object_source")
    [[ "$object_source" == "live" ]] && src_args+=(--live-scenes-dir "$OUT_DIR/explored_scenes")

    echo "  gemini.batch ($split, object-source=$object_source)..."
    export XIAO_HEI_REF_SPATIAL=${XIAO_HEI_REF_SPATIAL:-1}
    uv run python -m xiao_hei_vln.gemini.batch \
      --gt "$gt" --out "$pred" "${src_args[@]}" \
      --trace-file "$trace" --debug-dir "$debug_dir" --rpm "${RPM:-5}" \
      || { echo "  gemini.batch failed for $scene/$split" >&2; continue; }

    echo "  scoring $split..."
    uv run python -m xiao_hei_vln.eval_pipeline \
      --gt "$gt" --pred "$pred" --out "$metrics" \
      || echo "  eval_pipeline failed for $scene/$split" >&2
  done
}

summary=$OUT_DIR/summary.csv
echo "scene,explore_ok,ref_pred,num_pred" > "$summary"

for scene in "${available[@]}"; do
  echo; echo "=== $scene ==="
  explore_ok=0
  if [[ "$GT_ONLY" -eq 1 ]]; then
    explore_ok=1; echo "  GT-only — skipping explore"
  elif [[ "$SKIP_EXPLORE" -eq 1 && -f "$OUT_DIR/explored_scenes/$scene/scene.json" ]]; then
    explore_ok=1; echo "  reusing existing dump"
  elif explore_scene "$scene"; then
    explore_ok=1
  fi

  if [[ "$explore_ok" -ne 1 ]]; then
    echo "$scene,0,," >> "$summary"
    echo "  FAILED explore — skipping offline eval"
    "${COMPOSE[@]}" down >/dev/null 2>&1 || true
    continue
  fi

  run_offline_for_scene "$scene"
  ref_n=0; num_n=0
  [[ -f "$OUT_DIR/preds/${scene}_ref.jsonl" ]] && ref_n=$(wc -l < "$OUT_DIR/preds/${scene}_ref.jsonl" | tr -d ' ')
  [[ -f "$OUT_DIR/preds/${scene}_num.jsonl" ]] && num_n=$(wc -l < "$OUT_DIR/preds/${scene}_num.jsonl" | tr -d ' ')
  echo "$scene,1,${ref_n},${num_n}" >> "$summary"
done

echo
echo "=== finished ==="
echo "summary: $summary"
echo "dumps:   $OUT_DIR/explored_scenes/"
echo "preds:   $OUT_DIR/preds/"
echo "metrics: $OUT_DIR/metrics/"
cat "$summary"
