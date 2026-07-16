#!/usr/bin/env bash
# Phase 0 / Step 0 probe for the CMU VLN challenge simulator (TASK 10).
#
# Run this ON xiaohei, inside the ROS 2 environment (challenge docker, or a
# shell with the challenge workspace sourced), AFTER the sim is up
# (system_simulation.sh). It answers the questions that gate the whole
# capture pipeline:
#   - which topics exist, their type / rate / frame_id
#   - the camera's vertical FOV source (camera_info, if any)
#   - whether /camera/depth and /camera/semantic_image are available
#     (these decide Branch A "GT nearly free" vs Branch B "CSV projection")
#   - terrain cost metadata (discrete vs continuous)
#
# Output: a timestamped folder ./sim_probe_<ts>/ with report.txt + raw dumps.
#
# Usage:
#   bash scripts/probe_sim.sh            # default 8s rate window
#   HZ_WIN=12 bash scripts/probe_sim.sh  # longer rate window
set -uo pipefail

if ! command -v ros2 >/dev/null 2>&1; then
  echo "ERROR: 'ros2' not found. Source the ROS 2 / challenge workspace first." >&2
  exit 1
fi

OUT="sim_probe_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
REPORT="$OUT/report.txt"
log() { echo "$@" | tee -a "$REPORT"; }

HZ_WIN=${HZ_WIN:-8}   # seconds to measure each topic's publish rate

KEY_TOPICS=(
  /camera/image
  /camera/camera_info
  /camera/depth
  /camera/semantic_image
  /registered_scan
  /sensor_scan
  /terrain_map
  /terrain_map_ext
  /state_estimation
)

log "# sim probe $(date -u +%Y-%m-%dT%H:%M:%SZ)  (HZ_WIN=${HZ_WIN}s)"
log ""
log "=== ros2 topic list ==="
ros2 topic list 2>/dev/null | tee "$OUT/topic_list.txt" >>"$REPORT"

log ""
log "=== key-topic presence / type ==="
AVAIL=()
for t in "${KEY_TOPICS[@]}"; do
  if ros2 topic list 2>/dev/null | grep -qx "$t"; then
    typ=$(ros2 topic type "$t" 2>/dev/null | head -1)
    log "[present] $t  ($typ)"
    AVAIL+=("$t")
    ros2 topic info -v "$t" >"$OUT/info_${t//\//_}.txt" 2>&1
  else
    log "[MISSING] $t"
  fi
done

is_avail() { printf '%s\n' "${AVAIL[@]}" | grep -qx "$1"; }

log ""
log "=== publish rates (ros2 topic hz, ${HZ_WIN}s window) ==="
for t in "${AVAIL[@]}"; do
  r=$(timeout "${HZ_WIN}s" ros2 topic hz "$t" 2>/dev/null \
        | grep -m1 "average rate" || echo "n/a")
  log "  $t : ${r:-n/a}"
done

log ""
log "=== camera_info (vertical-FOV source) ==="
if is_avail /camera/camera_info; then
  timeout 5s ros2 topic echo --once /camera/camera_info >"$OUT/camera_info.txt" 2>&1
  log "  saved camera_info.txt — inspect K (fx,fy,cx,cy), distortion, width/height."
  log "  CAUTION: a panoramic/equirect camera often publishes a meaningless"
  log "  pinhole K (or nothing). If so, get vertical FOV from the sim camera"
  log "  plugin / URDF, or calibrate empirically (known object at known range)."
else
  log "  /camera/camera_info NOT published. Vertical FOV must come from the sim"
  log "  camera plugin / URDF, or empirical calibration. (Our prototype assumes"
  log "  120 deg derived from the 1920x640 aspect ratio — CONFIRM this here.)"
fi

log ""
log "=== image-like topics: encoding / size ==="
for t in /camera/image /camera/depth /camera/semantic_image; do
  if is_avail "$t"; then
    timeout 5s ros2 topic echo --once --no-arr "$t" 2>/dev/null \
      | grep -E "width|height|encoding|step|is_bigendian" \
      | sed "s|^|  ${t}: |" | tee -a "$REPORT"
  fi
done

log ""
log "=== terrain cost metadata (discrete vs continuous) ==="
if is_avail /terrain_map; then
  timeout 5s ros2 topic echo --once --no-arr /terrain_map >"$OUT/terrain_meta.txt" 2>&1
  log "  saved terrain_meta.txt (fields, point_step)."
  log "  To read actual intensity/cost values run, e.g.:"
  log "    ros2 topic echo --once /terrain_map | head -60"
fi

log ""
log "=== GT-strategy verdict ==="
if is_avail /camera/semantic_image && is_avail /camera/depth; then
  log "  -> BRANCH A: semantic_image + depth present."
  log "     2D label GT (semantic) + 3D/visibility (depth) are nearly free."
  log "     NEXT: confirm semantic encoding & whether its labels map to CSV raw_label."
else
  log "  -> BRANCH B: semantic_image and/or depth missing."
  log "     Fall back to CSV projection + alignment T + lidar visibility"
  log "     (dataset_generator/project_gt_prototype.py + solve_alignment.py)."
fi

log ""
log "Done. Full report + raw dumps in: $OUT/"
