# System Architecture

## High-level data flow

```mermaid
graph LR
    subgraph Challenge Platform
        SIM[Simulator]
        ROS[ROS 2 Topics]
    end

    subgraph ai_module Container
        SUB[Subscribers]
        CACHE[LatestCache]
        TICK[2 Hz Timer]
        RESP[Responder]
        PUB[VLMOutputPublisher]
    end

    subgraph vllm Container
        VLLM[vLLM Server<br/>OpenAI API]
    end

    SIM --> ROS
    ROS -->|7 topics| SUB
    SUB -->|overwrite latest| CACHE
    TICK -->|snapshot| CACHE
    CACHE -->|VLMInput| RESP
    RESP -->|HTTP /v1/chat| VLLM
    VLLM -->|JSON| RESP
    RESP -->|VLMOutput| PUB
    PUB -->|ROS publish| ROS
    ROS --> SIM
```

## Container topology (GPU compose)

```mermaid
graph TB
    subgraph Docker Compose
        SYS[iros2026_system<br/>Challenge simulator + ROS]
        AI[xiao_hei_ai_module<br/>Python responder + ROS node]
        VLLM[xiao_hei_vllm<br/>vLLM OpenAI server]
    end

    SYS <-->|ROS 2 DDS<br/>network_mode: host| AI
    AI -->|HTTP :8000/v1| VLLM
```

All three containers share `network_mode: host` so ROS 2 DDS discovery and the vLLM HTTP API work without port mapping.

## Input topics

| ROS topic | Rate | Python type | Description |
|---|---|---|---|
| `/camera/image` | ~10 Hz | `ImageFrame` | 1920x640 BGR8 panoramic |
| `/registered_scan` | ~10 Hz | `LidarScan` | (x,y,z,intensity) in map frame |
| `/sensor_scan` | ~10 Hz | `LidarScan` | (x,y,z) in sensor frame |
| `/terrain_map` | ~10 Hz | `TerrainMap` | Local 5m traversability |
| `/terrain_map_ext` | ~10 Hz | `TerrainMap` | Extended 20m traversability |
| `/state_estimation` | ~200 Hz | `OdomPose` | Robot pose in map frame |
| `/challenge_question` | 1 Hz | `ChallengeQuestion` | Natural language question |

## Output topics

| Question type | Python class | ROS topic |
|---|---|---|
| Numerical | `NumericalResponse` | `/numerical_response` |
| Object reference | `ObjectReferenceResponse` | `/selected_object_marker` |
| Instruction following | `WaypointPathResponse` | `/way_point_with_heading` |

## Module map

```
src/xiao_hei_vln/
├── messages/      # Pydantic models for all I/O types
├── sync/          # LatestCache — thread-safe sensor buffer
├── adapters/      # ROS 2 subscribers + publishers
├── app/           # rclpy entry point, tick loop, responder factory
├── logger.py      # VLM tick logger (model-agnostic)
├── image_utils.py # Shared image conversion helpers
├── dummy/         # Reference responder (no GPU)
├── qwen/          # Qwen3.5 responder, engine, prompts
├── evaluator/     # Offline metrics (numerical, object reference)
├── eval_sampler/  # Ground-truth ↔ prediction pairing
└── eval_pipeline/ # End-to-end evaluation CLI
```

## Tick lifecycle

```mermaid
sequenceDiagram
    participant Timer as 2 Hz Timer
    participant Cache as LatestCache
    participant Resp as Responder
    participant Engine as HTTPQwenEngine
    participant vLLM as vLLM Server

    Timer->>Cache: snapshot(tick_id, timestamp)
    Cache-->>Timer: VLMInput
    Timer->>Resp: respond(VLMInput)
    Resp->>Resp: build prompts (system + user)
    Resp->>Engine: infer(system, user_text, image)
    Engine->>vLLM: POST /v1/chat/completions
    vLLM-->>Engine: JSON response
    Engine-->>Resp: VLMOutput
    Resp->>Resp: update evidence, check done
    Resp-->>Timer: VLMOutput
    Timer->>Timer: publisher.publish(output)
```
