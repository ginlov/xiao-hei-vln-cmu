# Online perception + scoring

How to run the live perception pipeline against the simulator (360 RGB +
LiDAR → detect → 3D lift → scene graph → answer) and score the resulting
scene graph against the authoritative VLA-3D ground truth.

At test time the robot exposes **only** the 360 camera, the LiDAR clouds
(`/registered_scan`, `/sensor_scan`, `/terrain_map*`) and `/state_estimation`
— no depth image, no GT semantics. This pipeline uses exactly those.

## 0. Prerequisites

- Docker images built (`xiao-hei/ai_module`, `xiao-hei/perception`, the sim
  container). See [Docker](../getting-started/docker.md).
- A scene extracted on the host so the sim can load it and `eval` can read
  its `object_list.txt`:
  ```bash
  SCENES=/path/to/CMU-VLN-Challenge-data/unity_env_models
  unzip -oq $SCENES/arabic_room.zip -d $SCENES/
  ```

## 1. Bring up the perception stack

```bash
xhost +local:
export XIAO_HEI_SCENE_DIR_HOST=$SCENES/arabic_room   # sim loads this scene
export XIAO_HEI_RESPONDER=perception
export XIAO_HEI_OBJECT_MAP=1                          # cross-frame fusion + 3D boxes
XIAO_HEI_RESPONDER=perception docker/run up -d
```

The `perception` sidecar (YOLO-World v2 + SAM 2.1) loads its weights on
start; wait for it:

```bash
docker logs -f xiao_hei_perception   # look for: "Application startup complete"
```

## 2. Launch the simulator

```bash
docker exec -d iros2026_system \
  /home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh
```

Confirm the robot is publishing:

```bash
docker exec iros2026_system bash -lc \
  'source /opt/ros/jazzy/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && \
   ros2 topic hz /registered_scan'
```

## 3. Move the robot so the scene graph grows

The scene is built as the robot observes the room. Pick one:

**a) Autonomous frontier exploration (default).** Nothing to do — the
`ai_module` drives the robot with `FrontierExplorer` until it stops finding
new frontiers, ingesting detections every tick. A question fired mid-run is
deferred until exploration completes, then answered.

**b) Deterministic fixed path (reproducible A/B).** Disable exploration and
drive a pre-planned coverage trajectory so two runs see the same thing —
the only way to attribute a metric delta to a config change:

```bash
export XIAO_HEI_EXPLORATION_MAX_WAYPOINTS=0           # turn frontier off
# plan a coverage path once (needs the trajectory extra)
uv run --extra trajectory python -m xiao_hei_vln.trajectory \
    $SCENES/arabic_room.zip --out trajectories
# drive it deterministically (pose-based arrival) while the responder
# ingests each tick. The driver needs rclpy, so run it in the sim container:
docker cp scripts/wp_driver.py iros2026_system:/tmp/wp_driver.py
docker cp trajectories/arabic_room.json iros2026_system:/tmp/traj.json
docker exec -d iros2026_system bash -lc \
  'source /opt/ros/jazzy/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && \
   python3 /tmp/wp_driver.py /tmp/traj.json'
```

> Frontier exploration is **non-deterministic** (occupancy grid + nav
> timing), so two frontier runs cover the scene differently — fine for a
> single evaluation, not for an A/B that isolates one variable.

## 4. Ask a question (triggers logging + the answer)

```bash
docker exec iros2026_system bash -lc \
  'source /opt/ros/jazzy/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && \
   ros2 topic pub --once /challenge_question std_msgs/msg/String \
   "{data: \"How many stools are there?\"}"'
```

The numerical answer path counts objects of the queried class straight from
the scene graph (no VLM needed). Watch for `Response complete` in
`docker logs xiao_hei_ai_module`.

## 5. Where the run lands

The tick logger writes to the host-mounted `vlm_logs/`:

```
vlm_logs/session_<ts>/
  session.json                     # config used (incl. object_map flag)
  q_001_<slug>/ticks.jsonl         # one JSON line / tick; each has ["scene"]
  q_001_<slug>/predictions.jsonl   # the final answer
```

> Keep `XIAO_HEI_VLM_LOG_DIR` **unset** so logs land in the mounted
> `/vlm_logs` (host `./vlm_logs`). Setting it to a host path writes inside
> the container instead.

### Export a session for offline debugging

Pull a session off the box and build its HTML report (camera playback,
per-tick VLM I/O table, pose, BEV, latency) in one command:

```bash
scripts/export_session.sh                    # latest session -> ~/Downloads/percep_out_<ts>/
scripts/export_session.sh 20260711_052457    # a specific session
```

Overridable via `REMOTE_HOST` / `REMOTE_REPO` / `OUT_ROOT` / `NO_REPORT=1`.
Ticks (the VLM I/O) are only logged while a question is active — fire a few
questions during the run or the session holds just `session.json`.

## 6. Score the scene graph against ground truth

Extract the final (fullest) scene from the last tick, then score it:

```bash
# extract the last tick's scene → scene.json
uv run python - <<'PY'
import json, glob
sess = sorted(glob.glob("vlm_logs/session_*"))[-1]
f    = sorted(glob.glob(sess + "/q_*/ticks.jsonl"))[-1]
scene = json.loads(open(f).read().splitlines()[-1])["scene"]
json.dump(scene, open("scene.json", "w"))
print("wrote scene.json from", f)
PY

# score vs the scene's authoritative object_list.txt
uv run --extra perception python -m xiao_hei_vln.perception.eval \
    --scene scene.json \
    --gt-zip $SCENES/arabic_room.zip \
    --scene-name arabic_room \
    --out metrics.json
```

`metrics.json` reports, per distance/IoU threshold:

| field | meaning |
|---|---|
| `mAP` `dist@{0.5,1,2}m` / `iou@0.25` | mean AP by centre-distance and by 3D-IoU match |
| `operating_point` `precision/recall/f1` | at each match gate |
| `mean_center_err_m` | localisation error of matched objects |
| `counting_MAE`, `counting_exact_frac` | per-class count error (numerical-question proxy) |
| `confusion` | label confusion of matched pairs |

`iou@0.25` is only meaningful with 3D boxes, i.e. `XIAO_HEI_OBJECT_MAP=1`
(the baseline emits a single point per object, so its IoU is 0).

## Tear down

```bash
docker/run down
```

## Key environment flags

| var | default | effect |
|---|---|---|
| `XIAO_HEI_RESPONDER` | `dummy` | set `perception` for this pipeline |
| `XIAO_HEI_OBJECT_MAP` | off | `1` → cross-frame fusion, converged 3D boxes, NMS, wall-sheet rejection |
| `XIAO_HEI_SCENE_DIR_HOST` | — | extracted scene dir the sim loads |
| `XIAO_HEI_EXPLORATION_MAX_WAYPOINTS` | `100` | `0` disables frontier (for the deterministic path) |
| `XIAO_HEI_PERCEPTION_SCORE_THRESHOLD` | `0.25` | sidecar detection confidence cut |
| `XIAO_HEI_PERCEPTION_MIN_INLIERS` | `10` | LiDAR returns a mask needs before a 3D point is committed |
| `XIAO_HEI_SCAN_KEYFRAMES` | `10` | multi-frame LiDAR accumulation window |
| `XIAO_HEI_SCAN_MIN_MOVE_M` / `MIN_ROT_DEG` | `0.25` / `15` | keyframe trigger (move or turn) |
| `XIAO_HEI_SCAN_VOXEL_M` | `0.05` | accumulated-cloud downsample resolution |

The 3D lift always applies the z-buffer occlusion gate (a camera can't see
through a foreground object) and densifies the sparse single sweep with the
keyframe accumulator so small objects clear `min_inliers` with genuine
on-surface returns. See
[Perception (Sidecar + Responder)](../perception-sidecar.md).
