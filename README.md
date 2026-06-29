# xiao-hei-vln-cmu

Team Xiao Hei's stack for the [CMU Vision-Language-Navigation
Challenge 2026](https://www.ai-meets-autonomy.com/cmu-vln-challenge).

**[Documentation](https://ginlov.github.io/xiao-hei-vln-cmu/)**

The repo provides a typed Python contract between the challenge's
ROS 2 sensors and a VLM, a dummy VLM that satisfies the contract end-to-end,
and a docker image that drops straight into the official challenge
compose stack. **A real VLM plugs in by changing one line.**

## Inputs (challenge system → VLM)

The challenge platform exposes seven topics that the VLM is allowed
to consume at test time. Each one is wrapped in a pydantic model under
`xiao_hei_vln.messages.*`; the ROS adapter
(`xiao_hei_vln.adapters.ros.subscribers.bind_subscribers`) populates a
thread-safe `LatestCache` from these.

| ROS topic | ROS type | Observed rate | Frame | Python class |
|---|---|---|---|---|
| `/camera/image` | `sensor_msgs/Image` (bgr8, 1920×640) | ~10 Hz | `camera` | `ImageFrame` |
| `/registered_scan` | `sensor_msgs/PointCloud2` (x,y,z,intensity) | ~10 Hz | `map` | `LidarScan(source="registered")` |
| `/sensor_scan` | `sensor_msgs/PointCloud2` (x,y,z) | ~10 Hz | `sensor_at_scan` | `LidarScan(source="sensor")` |
| `/terrain_map` | `sensor_msgs/PointCloud2` (x,y,z,cost) | ~10 Hz | `map` | `TerrainMap(range="local_5m")` |
| `/terrain_map_ext` | `sensor_msgs/PointCloud2` (x,y,z,cost) | ~10 Hz | `map` | `TerrainMap(range="ext_20m")` |
| `/state_estimation` | `nav_msgs/Odometry` | ~200 Hz | `map → sensor` | `OdomPose` |
| `/challenge_question` | `std_msgs/String` | 1 Hz | — | `ChallengeQuestion` |

Each tick the VLM receives a single `VLMInput` snapshot bundling the
latest of each channel (or `None` if a topic hasn't ticked yet).
Full rate measurements and message-field details are in
[`docs/task1_phase1_measurements.md`](docs/task1_phase1_measurements.md).

## Outputs (VLM → challenge system)

Responses are a **discriminated union** — emitting one routes to
exactly one topic, matching the challenge's three question types:

| Question type | Python class | ROS topic | ROS type |
|---|---|---|---|
| Numerical (`How many …`) | `NumericalResponse` | `/numerical_response` | `std_msgs/Int32` |
| Object reference (`Find …`) | `ObjectReferenceResponse` | `/selected_object_marker` | `visualization_msgs/Marker` (CUBE, `map` frame) |
| Instruction following (else) | `WaypointPathResponse` | `/way_point_with_heading` | `geometry_msgs/Pose2D` (one per waypoint) |

Routing is handled by `xiao_hei_vln.adapters.ros.publishers.VLMOutputPublisher`.

## Tick frequency

**The VLM runs at 2 Hz** (configurable via `XIAO_HEI_VLM_TICK_HZ`).
Sensor topics are 10–200 Hz; their callbacks just overwrite the
relevant slot in a thread-safe `LatestCache`. On each 500 ms tick we
atomically snapshot every slot into one `VLMInput`. Rationale and
trade-offs in [`docs/task1_io_spec.md`](docs/task1_io_spec.md).

```
ROS topics ──► subscribers ──► LatestCache ──snapshot──► VLMInput
                                                              │
                                                       (your VLM here)
                                                              │
                                                              ▼
ROS topics ◄── VLMOutputPublisher ◄────────────────────  VLMOutput
```

## Develop / integrate your VLM

Everything around the model is already wired up. To plug in a real
VLM you only touch the `respond()` body.

### 1. Implement a responder

The contract is a single method that turns a snapshot into a response:

```python
# src/xiao_hei_vln/your_model/responder.py
from xiao_hei_vln.messages import (
    NumericalResponse, ObjectReferenceResponse, WaypointPathResponse,
    VLMInput, VLMOutput, Vector3, Waypoint,
)
from xiao_hei_vln.messages.question import QuestionType

class MyVLMResponder:
    def __init__(self) -> None:
        ...  # load weights, build prompt template, etc.

    def respond(self, snapshot: VLMInput) -> VLMOutput | None:
        if snapshot.question is None or not snapshot.is_ready:
            return None  # cold start, wait for next tick

        # snapshot.image, snapshot.registered_scan, snapshot.pose, ... are populated
        # snapshot.question.type tells you which response to emit
        match snapshot.question.type:
            case QuestionType.NUMERICAL:
                return NumericalResponse(value=...)
            case QuestionType.OBJECT_REFERENCE:
                return ObjectReferenceResponse(label=..., object_id=..., center=Vector3(...), size=Vector3(...))
            case QuestionType.INSTRUCTION_FOLLOWING:
                return WaypointPathResponse(waypoints=[Waypoint(x=..., y=...), ...])

    def is_done(self) -> bool: ...   # True once the response is final; tick clears the question
    def reset(self) -> None: ...     # called on each new question
```

Use [`src/xiao_hei_vln/dummy/responder.py`](src/xiao_hei_vln/dummy/responder.py)
as a working reference — same interface, trivial logic.

### 2. Wire it into the app

`src/xiao_hei_vln/app/main.py` is the only file that needs to know
your responder exists. Swap one line:

```python
# from xiao_hei_vln.dummy import DummyResponder
from xiao_hei_vln.your_model import MyVLMResponder

...
responder = MyVLMResponder()   # was DummyResponder()
```

`LatestCache`, the ROS subscribers, the publisher, the 2 Hz timer,
and the docker image stay exactly as they are.

### 3. Test without ROS

The whole input/output contract is pure Python + pydantic + numpy, so
unit tests don't need a ROS install. Pattern in
[`tests/test_dummy_responder.py`](tests/test_dummy_responder.py):
build a `VLMInput` by hand, call `respond()`, assert on the returned
class. Run with `uv run pytest -q`.

### 4. Run end-to-end

```bash
# one-time setup
uv sync
xhost +local:

# bring up the challenge sim + our VLM container
docker compose -f docker/compose_gpu.yml up -d --build

# inside iros2026_system: start the sim
docker exec -it iros2026_system /home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh

# from any container, ask a question
docker exec iros2026_system bash -lc \
  'source /opt/ros/jazzy/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && \
   ros2 topic pub --once /challenge_question std_msgs/msg/String "{data: \"Find the red cup\"}"'

# follow your VLM's logs
docker logs -f xiao_hei_ai_module
```

Build / push / drop-into-challenge-compose details are in
[`docker/README.md`](docker/README.md).

## Offline evaluation

After a live run the session directory contains `predictions.jsonl`.
Score it against the official challenge questions with:

```bash
uv run xiao-hei-eval \
  --gt dataset/challenge_gt.jsonl \
  --pred vlm_logs/<session>/predictions.jsonl
```

`challenge_gt.jsonl` (45 entries, one per scoreable challenge question) is
generated once from the VLA-3D scene graphs:

```bash
uv run python dataset_generator/challenge_gt_gen.py
```

See [Evaluation guide](docs/guides/evaluation.md) and
[Data Generation guide](docs/guides/data-generation.md) for details.

## Repository layout

```
src/xiao_hei_vln/
├── messages/       pydantic models for every input/output type
├── sync/           LatestCache + tick snapshot
├── adapters/       ROS 2 subscribers + publishers (lazy rclpy import)
├── dummy/          reference responder ported from dummyVLM.cpp
├── qwen/           Qwen2.5-VL responder (vLLM-backed, separate container)
├── evaluator/      offline metrics (numerical + object-reference)
├── eval_sampler/   GT ↔ prediction matcher, GT format converter
├── eval_pipeline/  CLI entry point (xiao-hei-eval)
├── trajectory/     waypoint helpers
├── logger.py       VLMLogger — per-tick log writer + predictions.jsonl
└── app/            rclpy entry point (xiao-hei-dummy-vlm console script)
dataset_generator/  GT generation scripts + VLA-3D scene loaders
dataset/            generated JSONL files (challenge_gt, vla3d_ref, vla3d_num)
docker/             Dockerfile + compose + README
docs/               MkDocs site (architecture, guides, API reference)
tests/              pytest suite (no ROS required)
```

## Working principles

- Always run Python via `uv` (Python 3.12 venv created on first
  `uv sync`).
- After each numbered task, write a short report at the repo root:
  `TASK N - <purpose>.md`.
