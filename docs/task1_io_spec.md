# Task 1 — VLM I/O Specification

This document defines the contract between the ROS 2 stack provided by
the CMU VLN Challenge organizers and Team Xiao Hei's VLM. It covers:

1. Inputs the VLM receives from the system.
2. Outputs the VLM publishes back to the system.
3. The unified VLM execution frequency and how multi-rate inputs are
   resampled onto it.
4. The Python type contract used in this repository.

## 1. Inputs (system → VLM)

The system publishes seven topics that the VLM may rely on at test
time. The "VLM class" column points to the pydantic types in
`xiao_hei_vln.messages.*` (see Section 4).

| ROS topic | ROS type | Published rate (observed) | Frame | VLM class | Logical channel |
|---|---|---|---|---|---|
| `/camera/image` | `sensor_msgs/Image` | ~9.3 Hz | `camera` | `ImageFrame` | `image` |
| `/registered_scan` | `sensor_msgs/PointCloud2` | ~9.8 Hz | `map` | `LidarScan(source="registered")` | `registered_scan` |
| `/sensor_scan` | `sensor_msgs/PointCloud2` | ~9.8 Hz | `sensor_at_scan` | `LidarScan(source="sensor")` | `sensor_scan` |
| `/terrain_map` | `sensor_msgs/PointCloud2` | ~9.7 Hz | `map` | `TerrainMap(range="local_5m")` | `terrain_local` |
| `/terrain_map_ext` | `sensor_msgs/PointCloud2` | ~9.8 Hz | `map` | `TerrainMap(range="ext_20m")` | `terrain_ext` |
| `/state_estimation` | `nav_msgs/Odometry` | ~200 Hz | `map → sensor` | `OdomPose` | `pose` |
| `/challenge_question` | `std_msgs/String` | 1 Hz | — | `ChallengeQuestion` | `question` |

PointCloud2 fields are flattened to `numpy.ndarray` of shape `(N, 4)`
(x, y, z, intensity) at the adapter boundary. For `/sensor_scan` only
three named fields exist (no intensity) so the trailing column is
filled with zeros.

## 2. Outputs (VLM → system)

The VLM emits exactly one response per question, sent over the topic
matching the question type:

| Question type | ROS topic | ROS type | VLM class | Scoring (per README) |
|---|---|---|---|---|
| Numerical | `/numerical_response` | `std_msgs/Int32` | `NumericalResponse` | /1 (exact integer) |
| Object reference | `/selected_object_marker` | `visualization_msgs/Marker` (CUBE, frame `map`) | `ObjectReferenceResponse` | /2 (bbox overlap) |
| Instruction following | `/way_point_with_heading` | `geometry_msgs/Pose2D` (one per waypoint) | `WaypointPathResponse` | /6 (trajectory adherence) |

`VLMOutput` is a discriminated union over the three response classes
keyed by the `kind` field, so the publisher layer can route each
response to exactly one topic without conditional logic on the question
string.

## 3. Execution frequency & synchronization

### Chosen VLM frequency

**The VLM main loop runs at 2 Hz** (period = 500 ms), configurable in
the range 0.5 – 5 Hz.

Rationale:

- The slowest sensor topic publishes at ~9.3 Hz (camera). At a 2 Hz
  VLM tick we are guaranteed to have at least 4 fresh camera frames
  and 4 fresh lidar scans between ticks, so the snapshot is never
  more than ~110 ms stale.
- A 2 Hz cadence allows ~400–500 ms of inference per tick, which is
  realistic for transformer-based VLMs running on the 4090 in the
  evaluation NUC.
- 1 Hz (the question publish rate) would also work and is the
  minimum sensible rate, but 2 Hz halves the worst-case reaction
  latency to scene changes during exploration.
- 5 Hz would force the VLM to run sub-200 ms inference, which is not
  feasible for current VLM architectures.

### Synchronization strategy: latest-cache + tick snapshot

```
ROS topics ──► ROS adapter (subscribers) ──► LatestCache (one slot per channel)
                                                       │
                                                       ▼
                                            VLM main loop (2 Hz)
                                                       │
                                       cache.snapshot() ─►  VLMInput
                                                       │
                                                       ▼
                                              VLM inference
                                                       │
                                                       ▼
                                                  VLMOutput
                                                       │
                                            (discriminator-routed)
                                                       ▼
ROS topics ◄── ROS adapter (publishers) ◄── VLMOutput
```

- The `LatestCache` keeps **one mutex-guarded slot per logical
  channel**. Writers (ROS subscriber callbacks) overwrite; readers
  (VLM tick) snapshot all slots into a single `VLMInput` atomically.
- No history, no time-windowing — the simplest semantics that work
  for a slow VLM consuming high-frequency inputs.
- Each `VLMInput` carries `tick_id` and `tick_time` so downstream
  components can correlate model outputs with the snapshot that
  produced them.
- Channels that have not yet received data (cold-start) are
  represented as `None` in the snapshot; the VLM must tolerate this.

Why not `message_filters.ApproximateTimeSynchronizer`? With our
input rates (camera ~9 Hz vs lidar ~10 Hz vs odometry 200 Hz), an
ATS would routinely drop bundles waiting for a tightly aligned tuple,
and we don't need sub-tick alignment. We keep the door open for ATS
in a follow-up by making `VLMInput` a plain pydantic model that any
synchronizer can populate.

### Question lifecycle

`ChallengeQuestion` is **sticky**: once received it remains in the
cache (and therefore appears in every `VLMInput.snapshot()`) until
the application clears it after publishing a response. The 1 Hz
publish rate of the evaluation node means the same question is
re-asserted continuously; we de-duplicate by text.

## 4. Python type contract (summary)

| Module | Key types |
|---|---|
| `xiao_hei_vln.messages.common` | `Stamp`, `Header`, `Vector3`, `Quaternion` |
| `xiao_hei_vln.messages.sensors` | `ImageFrame`, `LidarScan`, `TerrainMap`, `OdomPose` |
| `xiao_hei_vln.messages.question` | `QuestionType`, `ChallengeQuestion`, `classify_question` |
| `xiao_hei_vln.messages.inputs` | `VLMInput` |
| `xiao_hei_vln.messages.outputs` | `NumericalResponse`, `ObjectReferenceResponse`, `Waypoint`, `WaypointPathResponse`, `VLMOutput`, `parse_vlm_output` |
| `xiao_hei_vln.sync.latest_cache` | `LatestCache` |
| `xiao_hei_vln.adapters.ros.subscribers` | `bind_subscribers(node, cache)` — lazy `rclpy` import |
| `xiao_hei_vln.adapters.ros.publishers` | `publish(node, output)` — routes `VLMOutput` to the correct topic |

All `messages.*` and `sync.*` modules import only stdlib + numpy +
pydantic, so the contract is testable without ROS.

## 5. Open questions for later tasks

- How do we encode multi-step `WaypointPathResponse` semantics when
  the system only accepts one Pose2D per publish? The publisher will
  serialize the path and rely on `/way_point_reached` to advance, as
  the dummy VLM does today; this is implementation detail handled in
  the publisher adapter, not in the data model.
- Question-type classification currently uses simple keyword heuristics
  ("Find" → object_reference, "How many" → numerical, else
  instruction_following), matching the dummy VLM's behaviour. A learned
  classifier may replace this later without changing the contract.
