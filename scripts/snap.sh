#!/usr/bin/env bash
# Snapshot what the robot is looking at right now, as four perspective faces.
#
# Run it on the sim host while driving. Grabs one frame off
# `/camera/image/compressed`, unwraps it with the sidecar's own LUTs, and
# leaves the panorama plus four faces in a timestamped directory.
#
#   scripts/snap.sh                 # -> snaps/<HHMMSS>/
#   scripts/snap.sh -o look_here    # -> look_here/
#   scripts/snap.sh --face-size 1024
#
# The compressed topic is used deliberately: it is already JPEG on the wire, so
# the container side is a byte copy and needs neither cv2 nor numpy — the sim
# image has neither.

set -uo pipefail

REPO=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
SYSTEM_CTR=${SYSTEM_CTR:-iros2026_system}
OUT=""
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--out) OUT=$2; shift 2 ;;
    -h|--help) sed -n '2,16p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done
[[ -n "$OUT" ]] || OUT="$REPO/snaps/$(date +%H%M%S)"

docker exec "$SYSTEM_CTR" bash -lc '
source /opt/ros/jazzy/setup.bash
source /home/docker/autonomy_stack_mecanum_wheel_platform/install/setup.bash
export ROS_DOMAIN_ID=0 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
python3 - <<PY
import json
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, PointCloud2
from nav_msgs.msg import Odometry

class Snap(Node):
    def __init__(self):
        super().__init__("snap")
        self.img = self.scan = self.pose = None
        # Best-effort matches the sim publishers; a reliable subscription
        # would silently never match and just hang here.
        self.create_subscription(CompressedImage, "/camera/image/compressed",
                                 self.on_img, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, "/registered_scan",
                                 self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/state_estimation",
                                 self.on_pose, qos_profile_sensor_data)

    def on_img(self, m):
        if self.img is None:
            open("/tmp/snap.jpg", "wb").write(bytes(m.data)); self.img = True

    def on_scan(self, m):
        if self.scan is not None:
            return
        # PointXYZI, 16 bytes per point. Assert rather than assume: a layout
        # change would otherwise reshape into plausible-looking garbage.
        assert m.point_step % 4 == 0, m.point_step
        a = np.frombuffer(bytes(m.data), dtype=np.float32)
        np.save("/tmp/snap_scan.npy", a.reshape(-1, m.point_step // 4))
        self.scan = True

    def on_pose(self, m):
        if self.pose is not None:
            return
        p, q = m.pose.pose.position, m.pose.pose.orientation
        json.dump({"position": [p.x, p.y, p.z],
                   "orientation": [q.x, q.y, q.z, q.w],
                   "stamp": m.header.stamp.sec}, open("/tmp/snap_pose.json", "w"))
        self.pose = True

    def done(self):
        return self.img and self.scan and self.pose

rclpy.init()
n = Snap()
for _ in range(300):                      # ~15 s; scan and image are both ~4 Hz
    rclpy.spin_once(n, timeout_sec=0.05)
    if n.done():
        break
if n.done():
    print("captured image + scan + pose")
else:
    print(f"TIMEOUT img={bool(n.img)} scan={bool(n.scan)} pose={bool(n.pose)}")
rclpy.shutdown()
PY' || { echo "capture failed — is $SYSTEM_CTR up?" >&2; exit 1; }

mkdir -p "$OUT"
docker cp "$SYSTEM_CTR:/tmp/snap.jpg" "$OUT/_raw.jpg" >/dev/null || exit 1
docker cp "$SYSTEM_CTR:/tmp/snap_scan.npy" "$OUT/scan.npy" >/dev/null || exit 1
docker cp "$SYSTEM_CTR:/tmp/snap_pose.json" "$OUT/pose.json" >/dev/null || exit 1

cd "$REPO"
# Only the unwrap needs opencv, and only the laptop checkout happens to have
# it. Borrow it for this one command rather than adding a heavyweight
# dependency to the project for a debug tool.
UV_RUN=(uv run)
uv run python -c "import cv2" >/dev/null 2>&1 \
  || UV_RUN=(uv run --with opencv-python-headless)
"${UV_RUN[@]}" python scripts/grab_faces.py "$OUT/_raw.jpg" -o "$OUT" \
  --contact-sheet "${EXTRA[@]+"${EXTRA[@]}"}" || exit 1
rm -f "$OUT/_raw.jpg"
echo
echo "pull them to your laptop with:"
echo "  rsync -a ${XIAO_HEI_SIM_HOST:-xiaohei1}:${OUT#$HOME/} ./"
