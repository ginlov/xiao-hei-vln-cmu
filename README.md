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

# bring up the challenge sim + our VLM container (vllm sidecar auto-starts
# because XIAO_HEI_RESPONDER=qwen activates the `qwen` compose profile)
XIAO_HEI_RESPONDER=qwen docker/run up -d --build

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

### 4b. Run end-to-end — the submission stack (`scene_gemini`)

The **`scene_gemini`** responder is the team's full pipeline:

```
frontier exploration  →  perception scene graph (YOLO-World + SAM)  →  Gemini answer
```

One shared frontier sweep drives the robot; on every tick the perception
sidecar lifts detections into the scene graph (`ingest()`). When the sweep
completes, the **populated** object graph — plus a panorama JPEG and an
occupancy/trajectory PNG — is handed to Gemini for the final numerical /
object-reference answer (Task 1) or the route plan (Task 2). Reasoning
runs in the cloud and perception runs in the sidecar; the GPU is shared by
the simulator and the perception sidecar. Use the dedicated compose file
[`docker/compose_scene_gemini.yml`](docker/compose_scene_gemini.yml)
(simulator + perception sidecar + our node, Gemini env wired):

```bash
export XIAO_HEI_GEMINI_API_KEY=<your-key>   # required — see note below
export XIAO_HEI_VLM_LOG_DIR=/vlm_logs       # write predictions.jsonl for scoring
xhost +local:

docker compose -f docker/compose_scene_gemini.yml up -d --build

# inside iros2026_system: start the sim
docker exec -it iros2026_system /home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh

# ask a question (Task 1 example)
docker exec iros2026_system bash -lc \
  'source /opt/ros/jazzy/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && \
   ros2 topic pub --once /challenge_question std_msgs/msg/String "{data: \"How many chairs are in the room\"}"'

docker logs -f xiao_hei_ai_module   # "ready (responder=scene_gemini ...)" means it is up
```

The answer is **deferred until the frontier sweep completes** (the robot
finishes exploring before it commits), so keep the `echo` from §4c
attached and watch the logs. Set `XIAO_HEI_EXPLORATION_MAX_WAYPOINTS=0` to
answer from the spawn pose without exploring.

> **The API key must be valid at startup.** `GeminiEngine.warmup()`
> issues one live `generate_content` call when the container boots, so an
> invalid/missing key (or no outbound 443 to
> `generativelanguage.googleapis.com`) makes `ai_module` crash-loop.

Optional knobs (all have defaults; just `export` to override):

| Env var | Default | Purpose |
|---|---|---|
| `XIAO_HEI_EXPLORATION_MAX_WAYPOINTS` | `100` | frontier sweep budget; `0` disables exploration |
| `XIAO_HEI_OBJECT_MAP` | (off) | `1` fuses detections into converged 3D boxes (NMS + wall-sheet rejection) |
| `XIAO_HEI_PERCEPTION_SCORE_THRESHOLD` | `0.25` | YOLO-World detection score gate |
| `XIAO_HEI_GEMINI_MODEL` | `gemini-2.5-flash` | model id |
| `XIAO_HEI_GEMINI_TEMPERATURE` | `0.2` | sampling temperature |
| `XIAO_HEI_GEMINI_MAX_OUTPUT_TOKENS` | `2048` | response token cap |
| `XIAO_HEI_GEMINI_THINKING_BUDGET` | `0` | thinking tokens; `0` disables (keeps the JSON answer from being truncated), `-1` = dynamic |
| `XIAO_HEI_GEMINI_IMAGE_LONG_EDGE` | `1280` | downscale long-edge before send |

### 4c. Receiving the response

The VLM never replies on `/challenge_question`. It routes the answer to
**one** of three topics depending on the question type (the
[Outputs](#outputs-vlm--challenge-system) discriminated union):

| Question type | Answer topic | ROS type |
|---|---|---|
| Numerical (`How many …`) | `/numerical_response` | `std_msgs/Int32` |
| Object reference (`Find …`) | `/selected_object_marker` | `visualization_msgs/Marker` |
| Instruction following (else) | `/way_point_with_heading` | `geometry_msgs/Pose2D` |

Start the `echo` **before** you `pub` the question, otherwise you miss the
message (responses are latched at ~2 Hz, not replayed on subscribe). If you
don't know the type ahead of time, listen on all three:

```bash
# subscribe first — leave this running in its own terminal
docker exec iros2026_system bash -lc \
  'source /opt/ros/jazzy/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && \
   ros2 topic echo /selected_object_marker'        # or /numerical_response, /way_point_with_heading

# …then pub the question (§4 / §4b) in another terminal.
```

The response is **not immediate**: the VLM ticks at 2 Hz and runs the full
frontier sweep before committing an answer (`scene_gemini` defers until
exploration completes; budget = `XIAO_HEI_EXPLORATION_MAX_WAYPOINTS`), so
keep the `echo` attached and watch `docker logs -f xiao_hei_ai_module`
for progress. `WaypointPathResponse` emits one `Pose2D` per waypoint, so
`/way_point_with_heading` prints a burst of messages, one per path point.

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

### Offline Gemini evaluator (Task 1 + Task 2)

To measure Gemini directly — **without the simulator** —
`xiao_hei_vln.gemini.batch` reconstructs the scene graph from each GT
entry's `object_list` (the same `SceneRepresentation.to_dict()` JSON the
live responder feeds Gemini), asks Gemini for the answer, and writes an
evaluator-ready predictions JSONL.

The workflow is **three steps** — generate predictions once, then score
and/or visualise from that same `pred_ref.jsonl`:

```bash
export XIAO_HEI_GEMINI_API_KEY=<your-key>

# 1. Generate predictions (Task 2 / object_reference here).
#    --debug-dir and --trace-file are optional debug logs (see below).
XIAO_HEI_GEMINI_MODEL=gemini-2.5-flash uv run python -m xiao_hei_vln.gemini.batch \
  --gt   dataset/vla3d_ref.jsonl \
  --out  pred_ref.jsonl \
  --task2 10 --rpm 4 \
  --debug-dir debug_ref --trace-file trace_ref.jsonl

# 2. Score it — metrics to the terminal (mean IoU, SR@IoU, challenge score).
uv run xiao-hei-eval --gt dataset/vla3d_ref.jsonl --pred pred_ref.jsonl

# 3. Render the human-readable HTML report, then open it in a browser.
uv run python -m xiao_hei_vln.gemini.eval_report \
  --gt dataset/vla3d_ref.jsonl --pred pred_ref.jsonl --out eval_report.html
```

Task 1 (numerical) is the identical flow with `dataset/vla3d_num.jsonl`
(scored by exact-match accuracy instead of IoU).

This isolates Gemini's reasoning over the scene graph (perception assumed
perfect); the live stack (§4b) measures perception + exploration + Gemini
together. Question texts come from the GT, so they align exactly.

#### How the prompt is composed

The batch does **not** send Gemini the raw `SceneRepresentation.to_dict()`
— that graph is ~20–25× larger (it repeats each bbox as min+max, plus
confidence, viewpoint/tick ids, and pre-derived `near` edges) and
overruns the free-tier input-token/minute quota. Each call is **text
only** (no images), built as:

- **System prompt** (`offline_system_prompt`, one per task type) — the
  role, the object-list schema, and the exact `VLMOutput` JSON to return.
- **User message** (`build_user_message`) — three blocks:
  1. `Question (type=object_reference): Find the pillow closest to the book.`
  2. the **scene objects** as compact JSON — one entry per object, only
     the fields Gemini needs (proximity / relations are inferred from the
     coordinates, not pre-listed):
     ```json
     [{"id":0,"label":"window","center":[-6.4,-1.56,2.12],"size":[0.12,6.1,4.17]},
      {"id":2,"label":"pillow","center":[1.94,-2.09,0.41],"size":[0.42,0.21,0.36]}]
     ```
  3. `Respond now with the JSON object — no prose around it.`

So an object-reference answer is essentially "pick the right `id` and copy
its `center` / `size`". The `--debug-dir` JSON stores the *full*
`scene_graph` for inspection, but the prompt itself uses the compact form
above — see `request.user_text` in the `--trace-file`.

> The **live** responder differs: it sends the full `to_dict()` graph plus
> a panorama JPEG and an occupancy-map PNG (`gemini/scene_rep.py`).

`gemini.batch` flags:

| Flag | Default | Purpose |
|---|---|---|
| `--task1 N` | all | evaluate N Task 1 (numerical) examples — counts real scoreable entries, not raw GT lines |
| `--task2 M` | all | evaluate M Task 2 (object_reference) examples |
| `--rpm N` | `5` | throttle to N requests/min — `5` matches the free tier, raise on a paid plan, `0` disables |
| `--max-retries N` | `5` | retries on a 429 rate-limit (honours the server `retryDelay`) |
| `--debug-dir DIR` | – | dump one JSON per prediction (scene graph + prompts + parsed output) |
| `--trace-file FILE` | – | append a full-fidelity JSONL trace of every Gemini call (see below) |
| `--near-threshold M` | `2.0` | XY radius for `near` edges in the reconstructed graph |

Passing **either** `--task1` or `--task2` restricts the run to those task
type(s) — e.g. `--task2 10` evaluates 10 object-reference examples and no
numerical ones. With neither set, every scoreable entry is processed.

### Visual eval report

Step 3 above (`xiao_hei_vln.gemini.eval_report`) renders a
**self-contained `eval_report.html`** — no server, no external assets, so
just open it in a browser. It shows one card per question: a pass/fail
badge, Gemini's answer + rationale next to the ground truth, and — for
object-reference — a top-down scene plot with Gemini's box (red) vs the
ground-truth box (green), so a wrong pick is obvious at a glance.
Numerical questions get a predicted-vs-truth card. Add `--limit N` to
cap how many predictions are included.

### Debugging Gemini calls

`GeminiTracer` (`xiao_hei_vln.gemini.trace`) records **every** Gemini call
— full request + raw response + token usage + latency + errors — as
append-only JSONL. It hooks `GeminiEngine`, so it covers both the offline
batch (`--trace-file`) and the live responder (pass `tracer=` to
`GeminiEngine`). Each line carries `request` (model, system prompt, user
text incl. the scene graph, image sizes, sampling knobs), `response`
(`raw_text` *before* parsing, `finish_reason`, `usage`), `parsed`,
`latency_ms`, and `error`:

```bash
uv run python -m xiao_hei_vln.gemini.batch ... --trace-file trace_ref.jsonl

jq -r 'select(.error != null)' trace_ref.jsonl        # failed calls (with raw output)
jq -r '.response.usage.total_tokens' trace_ref.jsonl  # token cost per call
jq -r '.parsed // .response.raw_text' trace_ref.jsonl # parsed result, else raw text
```

The trace file is **append-only** — it accumulates across runs, so only
the tail belongs to the latest run (or `rm trace_ref.jsonl` before a run).

`--debug-dir` instead writes **one JSON per prediction** (named by the
question), carrying the full scene graph, prompts, and parsed output —
and, for failures, the `error` too (exactly the ones worth inspecting):

```bash
jq '.error'    debug_ref/*.json   # which entries failed, and why
jq -r '.question, .prediction.rationale' debug_ref/00004_*.json   # one entry's reasoning
```

For the **live** ROS run, per-tick logs (system prompt, user text, output,
camera frames, point clouds) instead go to `vlm_logs/` via `VLMLogger` —
see the [VLM logging guide](docs/guides/vlm-logging.md).

## Repository layout

```
src/xiao_hei_vln/
├── messages/       pydantic models for every input/output type
├── sync/           LatestCache + tick snapshot
├── adapters/       ROS 2 subscribers + publishers (lazy rclpy import)
├── dummy/          reference responder ported from dummyVLM.cpp
├── qwen/           Qwen2.5-VL responder (vLLM-backed, separate container)
├── gemini/         Gemini engine + offline batch evaluator + call tracer
├── scene_gemini/   submission responder: exploration + perception graph + Gemini
├── perception/     YOLO-World + SAM sidecar client, 3D lifter, scene-graph fusion
├── exploration/    frontier exploration strategies (occupancy grid + planner)
├── scene/          SceneRepresentation (Room → Viewpoints → Objects graph)
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
