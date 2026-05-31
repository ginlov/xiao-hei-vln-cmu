# Docker Setup

## Container images

| Container | Image | Purpose |
|---|---|---|
| `iros2026_system` | `zhangjicmu/ubuntu24_ros:system_simulation` | Challenge simulator + ROS 2 |
| `xiao_hei_ai_module` | Built from `docker/Dockerfile` | Python VLM responder + ROS node |
| `xiao_hei_vllm` | `vllm/vllm-openai:latest` | vLLM OpenAI-compatible server |

## Compose files

| File | Use case |
|---|---|
| `docker/compose.yml` | CPU-only / dummy responder (no GPU) |
| `docker/compose_gpu.yml` | Full GPU stack with vLLM sidecar |

## Building

```bash
# Build the ai_module image
docker compose -f docker/compose_gpu.yml build ai_module
```

The Dockerfile extends `zhangjicmu/ubuntu24_ros:ai_module`, installs
`python3-pip`, then editable-installs this repo into the system Python 3.12.

## Architecture: sidecar vs in-process

### Sidecar (default, recommended)

The `compose_gpu.yml` stack runs three containers. The `vllm` service uses
the official `vllm/vllm-openai` image and exposes an OpenAI-compatible API.
The `ai_module` calls it via HTTP using the lightweight `openai` Python SDK.
No CUDA dependencies are installed in the ai_module image.

### In-process (legacy)

Set `XIAO_HEI_QWEN_VLLM_BASE_URL=""` and build with
`XIAO_HEI_EXTRA=qwen-local` to load vLLM directly inside the ai_module
container. This requires manually resolving pip/apt conflicts and is not
recommended.

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
| "Waiting for vLLM server..." indefinitely | vLLM not started or model download stalled | Check `docker logs xiao_hei_vllm` |
| vLLM OOM | Model too large | Use smaller model or lower `XIAO_HEI_QWEN_GPU_MEM_UTIL` |
| Subscription count: 0 | DDS mismatch | Verify `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` on both containers |
| Echo returns nothing | VOLATILE QoS + late subscriber | Run echo *before* publishing |
