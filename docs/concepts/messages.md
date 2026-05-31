# Messages & I/O

The `xiao_hei_vln.messages` package defines the complete data contract between
the challenge platform and the VLM. Every type is a Pydantic model — fully
typed, serializable, and testable without ROS.

## VLMInput

A single snapshot of all sensor state at one tick:

```python
@dataclass
class VLMInput:
    tick_id: int
    tick_time: Stamp
    question: ChallengeQuestion | None
    pose: OdomPose | None
    image: ImageFrame | None
    registered_scan: LidarScan | None
    sensor_scan: LidarScan | None
    terrain_local: TerrainMap | None
    terrain_ext: TerrainMap | None
```

Fields are `None` if the corresponding ROS topic hasn't published yet. The
`is_ready` property returns `True` when at least pose + question are available.

## VLMOutput (discriminated union)

The model's response is one of three types:

| Type | When to use | Key fields |
|---|---|---|
| `NumericalResponse` | "How many..." questions | `value: int`, `rationale: str` |
| `ObjectReferenceResponse` | "Find the..." questions | `label: str`, `center: Vector3`, `size: Vector3` |
| `WaypointPathResponse` | Instruction-following | `waypoints: list[Waypoint]`, `rationale: str` |

All three share an optional `rationale` field for chain-of-thought reasoning.

The `kind` field acts as a discriminator for JSON (de)serialization:

```json
{"kind": "numerical", "value": 5, "rationale": "I counted 5 chairs"}
{"kind": "object_reference", "label": "red_cup", "center": {"x": 1.0, ...}}
{"kind": "waypoint_path", "waypoints": [{"x": 3.5, "y": -1.2}]}
```

## Sensor types

### ImageFrame

BGR8 camera image (1920x640 panoramic). Stored as raw bytes matching the ROS
`sensor_msgs/Image` layout.

### LidarScan

Point cloud as `(N, 4)` float32 numpy array — columns are `x, y, z, intensity`.
The `source` field distinguishes `"registered"` (map frame) from `"sensor"`
(sensor frame at scan time).

### TerrainMap

Traversability point cloud as `(N, 4)` float32 — columns are `x, y, z, cost`.
The `range` field is `"local_5m"` or `"ext_20m"`.

### OdomPose

Robot position + orientation in the map frame. Contains `Vector3` position and
`Quaternion` orientation.

## Question classification

`ChallengeQuestion` wraps the raw question string and adds a `type` field:

| Type | Detection rule | Example |
|---|---|---|
| `NUMERICAL` | Contains "how many" | "How many chairs are in the room?" |
| `OBJECT_REFERENCE` | Starts with "find" | "Find the red cup" |
| `INSTRUCTION_FOLLOWING` | Everything else | "Go to the kitchen" |

!!! note
    The current classifier uses keyword heuristics. A planned improvement
    is VLM-based classification on the first tick.
