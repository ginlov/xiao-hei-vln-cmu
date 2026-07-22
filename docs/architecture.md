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

    subgraph perception Container
        PERCS[YOLO-World + SAM 2.1<br/>FastAPI]
    end

    SIM --> ROS
    ROS -->|challenge topics| SUB
    SUB -->|overwrite latest| CACHE
    TICK -->|snapshot| CACHE
    CACHE -->|VLMInput| RESP
    RESP -->|HTTP /detect| PERCS
    PERCS -->|detections + masks| RESP
    RESP -->|VLMOutput| PUB
    PUB -->|ROS publish| ROS
    ROS --> SIM
```

## Container topology

```mermaid
graph TB
    subgraph Docker Compose
        SYS[iros2026_system<br/>Challenge simulator + ROS]
        AI[xiao_hei_ai_module<br/>Python responder + ROS node]
        PERC["xiao_hei_perception<br/>YOLO-World + SAM 2.1<br/>(profile: perception)"]
        GEM["Gemini API<br/>(external, HTTPS)"]
    end

    SYS <-->|ROS 2 DDS<br/>network_mode: host| AI
    AI -.->|HTTP :8001| PERC
    AI -.->|HTTPS| GEM
```

`system` and `ai_module` always start. The `perception` sidecar is
profile-gated — the wrapper maps `XIAO_HEI_RESPONDER=perception` to
`--profile perception`, so it only runs when a responder needs it. The
submission stack (`scene_gemini`) brings up the sim, the sidecar and our
node together via `docker/compose_scene_gemini.yml`, and reaches Gemini
over the network rather than running a local inference server. See
[Docker setup](getting-started/docker.md) and
[Perception Sidecar](perception-sidecar.md).

All containers share `network_mode: host` so ROS 2 DDS discovery and
the sidecar HTTP APIs work without port mapping.

## Input topics

| ROS topic | Rate | Python type | Description |
|---|---|---|---|
| `/camera/image` | ~10 Hz | `ImageFrame` | 1920x640 BGR8 panoramic |
| `/registered_scan` | ~5 Hz | `LidarScan` | (x,y,z,intensity) in map frame |
| `/sensor_scan` | ~5 Hz | `LidarScan` | (x,y,z) in sensor frame |
| `/terrain_map` | ~5 Hz | `TerrainMap` | Local 5m traversability |
| `/terrain_map_ext` | ~5 Hz | `TerrainMap` | Extended 20m traversability |
| `/state_estimation` | 100–200 Hz | `OdomPose` | Robot pose in map frame |
| `/challenge_question` | 1 Hz | `ChallengeQuestion` | Natural language question |
| `/way_point_reached` | continuous | `Float32` | Nav stack distance to current waypoint (exploration only) |

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
├── app/           # rclpy entry point, tick loop, explorer + responder factory
├── exploration/   # Exploration strategies (FrontierExplorer + protocol)
├── logger.py      # VLM tick logger (model-agnostic)
├── image_utils.py # Shared image conversion helpers
├── dummy/         # Reference responder (no GPU)
├── gemini/        # Gemini engine, prompts, offline batch evaluator, tracer
├── scene_gemini/  # Submission responder: exploration + scene graph + Gemini
├── scene/         # Three-level scene graph (Room/Viewpoint/Object) + renderer
├── trajectory/    # Offline coverage-trajectory planner (Task 7)
├── perception/    # PerceptionResponder + HTTP client + lifter + vocabulary
├── evaluator/     # Offline metrics (numerical, object reference)
├── eval_sampler/  # Ground-truth ↔ prediction pairing
└── eval_pipeline/ # End-to-end evaluation CLI
```

## Tick lifecycle

The tick loop runs in two phases.  Exploration runs first; the responder
only runs once exploration is complete or a question is active.

```mermaid
sequenceDiagram
    participant Timer as 2 Hz Timer
    participant Cache as LatestCache
    participant Exp as Explorer
    participant Resp as Responder
    participant Engine as GeminiEngine
    participant Gemini as Gemini API

    Timer->>Cache: snapshot(tick_id, timestamp)
    Cache-->>Timer: VLMInput

    alt exploration active (no question + not complete)
        Timer->>Exp: update(VLMInput)
        Exp-->>Timer: Waypoint | None
        Timer->>Timer: publisher.publish(WaypointPathResponse)
    else question active or exploration done
        Timer->>Resp: respond(VLMInput)
        Resp->>Resp: build prompts (system + user)
        Resp->>Engine: infer_multimodal(system, user_text, images)
        Engine->>Gemini: generate_content
        Gemini-->>Engine: JSON response
        Engine-->>Resp: VLMOutput
        Resp-->>Timer: VLMOutput
        Timer->>Timer: publisher.publish(output)
    end
```
