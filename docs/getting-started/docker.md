# Docker Setup

## Container images

| Container | Image | Purpose |
|---|---|---|
| `iros2026_system` | `zhangjicmu/ubuntu24_ros:system_simulation` | Challenge simulator + ROS 2 |
| `xiao_hei_ai_module` | Built from `docker/Dockerfile` | Python VLM responder + ROS node |
| `xiao_hei_perception` | Built from `perception/Dockerfile` | YOLO-World + SAM 2.1 detection sidecar |

## Compose files

| File | Use case |
|---|---|
| `docker/compose.yml` | Single unified stack. Profile-gated `perception` (only starts under `--profile perception`). |
| `docker/compose_scene_gemini.yml` | The submission stack: sim + perception sidecar + `scene_gemini` node, Gemini env wired. |
| `docker/run` | Wrapper that maps `XIAO_HEI_RESPONDER` → the right compose profile. |

## Driving the stack

One env var picks the responder; the wrapper handles profile selection
and prerequisite validation:

```bash
XIAO_HEI_RESPONDER=dummy      docker/run up -d    # system + ai_module
XIAO_HEI_RESPONDER=perception docker/run up -d    # + perception sidecar (YOLO-World + SAM 2.1)
```

All other args pass through verbatim: `docker/run logs -f ai_module`,
`docker/run down`, `docker/run build ai_module`.

### Scene selection (any responder)

Setting `XIAO_HEI_SCENE_DIR_HOST` to an extracted scene directory makes
the wrapper layer `compose.scene.yml`, which bind-mounts
`<dir>/environment/` over the sim's prebaked Unity environment. The same
mechanism works for any responder — handy for testing a specific scene
without `docker cp`.

```bash
SCENES=/path/to/CMU-VLN-Challenge-data/unity_env_models
unzip -oq $SCENES/arabic_room.zip -d $SCENES/                # one-time
export XIAO_HEI_SCENE_DIR_HOST=$SCENES/arabic_room
XIAO_HEI_RESPONDER=perception docker/run up -d              # sim loads arabic_room
```

## Building

```bash
# Build the ai_module image (always installs the lightweight `perception`
# extra — httpx + pillow + pycocotools — so the same image serves both
# responders).
docker/run build ai_module
```

The Dockerfile extends `zhangjicmu/ubuntu24_ros:ai_module`, installs
`python3-pip`, then editable-installs this repo into the system Python 3.12.

## Architecture: why sidecars

When `XIAO_HEI_RESPONDER=perception`, the wrapper activates the matching
compose profile, which brings up a sidecar container alongside `system` +
`ai_module`. The `ai_module` calls it over HTTP — no CUDA dependencies are
installed in the ai_module image, which avoids pip/apt conflicts with the
ROS base.

| Responder | Sidecar | Image |
|---|---|---|
| `perception` | `perception` on `:8001` | `xiao-hei/perception:latest` (built from `perception/Dockerfile`, based on `ultralytics/ultralytics:8.4.72`) |
| `scene_gemini` | `perception` on `:8001`, plus the Gemini API over HTTPS | same sidecar image; reasoning runs in the cloud, so no local inference server |

See [Perception Sidecar](../perception-sidecar.md) for the YOLO-World
+ SAM 2.1 setup specifically.

## Publishing for submission

```bash
docker tag xiao-hei/ai_module:latest <your-dockerhub-user>/xiao-hei-ai-module:latest
docker push <your-dockerhub-user>/xiao-hei-ai-module:latest
```

## Dropping into the challenge repo

In `CMU-VLN-Challenge-2026/docker/compose.yml`, replace the `ai_module:`
block:

```yaml
ai_module:
  image: <your-dockerhub-user>/xiao-hei-ai-module:latest
  container_name: iros2026_ai_module
  network_mode: host
  environment:
    - RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  stdin_open: true
  tty: true
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ai_module` blocks at startup waiting on the sidecar | sidecar not started or weights still downloading | Check `docker logs xiao_hei_perception` |
| `ai_module` crash-loops on `scene_gemini` | invalid/missing `XIAO_HEI_GEMINI_API_KEY` — `warmup()` validates it at boot | Export a valid key before `up` |
| Subscription count: 0 | DDS mismatch | Verify `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` on both containers |
| Echo returns nothing | VOLATILE QoS + late subscriber | Run echo *before* publishing |
