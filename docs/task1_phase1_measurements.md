# Task 1 — Phase 1 measurements

Captured 2026-05-24 from the official challenge docker stack
(`zhangjicmu/ubuntu24_ros:cmu_vla_challenge_simulation` running
`system_simulation.sh` with the bundled Unity scene). Measurements
were taken via `ros2 topic hz` (≥10 s window) and `ros2 topic echo
--once`.

The vehicle was idle in the spawn pose; rates may shift slightly when
the platform is moving, but the publishers all run on fixed timers so
the changes should be small.

## Sensor outputs (system → AI module)

| Topic | Type | README rate | Observed rate | Frame | Notes |
|---|---|---|---|---|---|
| `/camera/image` | `sensor_msgs/msg/Image` | 10 Hz | **~9.3 Hz** | `camera` | 1920×640, `bgr8`, `step=5760`, `data` length 3 686 400 bytes (~3.5 MB/frame). Publisher: `sim_image_repub`. |
| `/camera/image/compressed` | `sensor_msgs/msg/CompressedImage` | — | ~9.8 Hz | `camera` | Same scene, JPEG-compressed. Not listed as a challenge-time topic, but available. |
| `/registered_scan` | `sensor_msgs/msg/PointCloud2` | 5 Hz | **~9.8 Hz** | `map` | Fields `(x, y, z, intensity)` all FLOAT32; `point_step=16`. ~10 k points per scan in the spawn pose. |
| `/sensor_scan` | `sensor_msgs/msg/PointCloud2` | 5 Hz | **~9.8 Hz** | `sensor_at_scan` | Fields `(x, y, z)` FLOAT32, `point_step=16` (one float of padding/intensity slot). `is_dense=true`. |
| `/terrain_map` | `sensor_msgs/msg/PointCloud2` | 5 Hz | **~9.7 Hz** | `map` | Fields `(x, y, z, intensity)` FLOAT32, `point_step=32` (16 bytes after `intensity` are reserved by the terrain-analysis module). Intensity ≡ traversability cost. 5 m radius around vehicle. |
| `/terrain_map_ext` | `sensor_msgs/msg/PointCloud2` | 5 Hz | **~9.8 Hz** | `map` | Same schema as `/terrain_map` but 20 m radius. |
| `/state_estimation` | `nav_msgs/msg/Odometry` | 100–200 Hz | **~200.0 Hz** | `map → sensor` | `pose.pose.{position, orientation}` plus `twist.twist.{linear, angular}`. `child_frame_id=sensor`. Spawn z ≈ 0.75 m. |

All seven topics use QoS `RELIABLE` + `VOLATILE`.

### Discrepancies vs the README

1. **Lidar/terrain rates are ~10 Hz, not 5 Hz** as documented. We design for the higher rate (it just means our latest-cache holds fresher data); a future challenge update could revert to 5 Hz, in which case the VLM tick at 2 Hz still works.
2. **Camera frame rate is ~9.3 Hz** — close enough to the documented 10 Hz, but consistently below.
3. Extra sim-only topics observed: `/camera/depth`, `/camera/depth/compressed`, `/camera/semantic_image`, `/camera/semantic_image/compressed`, `/traversable_area`, `/overall_map`. **Per the README these MUST NOT be relied on at test time.** We will not subscribe to them from the AI module.

## AI-module inputs (system → AI module) — control plane

| Topic | Type | Rate | Notes |
|---|---|---|---|
| `/challenge_question` | `std_msgs/msg/String` | 1 Hz | Published by the `challenge_evaluation_node` (not started by `system_simulation.sh` alone — we'll wire it in when the eval node is launched). |

## AI-module outputs (AI module → system)

| Topic | Type | Notes |
|---|---|---|
| `/way_point_with_heading` | `geometry_msgs/msg/Pose2D` | `(x, y, theta)`. `theta` is the requested heading; per README the heading is currently ignored. RViz publishes here too; multiple publishers are allowed. |
| `/selected_object_marker` | `visualization_msgs/msg/Marker` | `CUBE`, `frame_id="map"`. Pose is the object center; scale is L × W × H. |
| `/numerical_response` | `std_msgs/msg/Int32` | Integer answer to a numerical question. Not used for navigation; read by the evaluation node. |

## Conclusion

The schema is exactly what the README describes; only the rates are
higher than documented. The Phase 2 spec and Phase 3 implementation can
therefore proceed using the published field definitions, with the VLM
tick rate chosen on the more conservative (lower) end of the measured
range so we stay safely below the slowest input.
