# TASK 10 — Scene Representation

## Purpose

Build an incrementally maintained scene representation from `VLMInput` to give the VLM structured, accumulated knowledge of the environment across ticks — replacing the flat text rationale chain with a proper graph of what has been seen, from where, and what remains unexplored.

The design follows SysNav: *Multi-Level Systematic Cooperation Enables Real-World, Cross-Embodiment Object Navigation* (arXiv 2603.06914) and adapts its three-level hierarchical graph to the sensor suite and constraints of the CMU VLN Challenge.

---

## Original SysNav design

SysNav builds a three-level graph ℛ updated incrementally as the robot navigates an indoor building:

| Level | Node type | Key attributes |
|---|---|---|
| High | **Room** | category label, room mask, best representative RGB image |
| Mid | **Viewpoint** | position, coverage region (observed voxels), panoramic image |
| Low | **Object** | label, confidence, 3D point cloud, bounding box, representative image |

**Edge types:** room–room (doorway connectivity), room–viewpoint (affiliation), room–object (containment), viewpoint–object (visibility), object–object (spatial relations, task-specific).

**Sensor inputs:** 360° panoramic RGB + LiDAR point cloud at each step. Object detection uses YOLOv8x + SAM2.1; room boundaries are identified via vertical planar surface fitting on the LiDAR point cloud (wall detection). Viewpoints are created when new coverage exceeds a threshold ε relative to the accumulated coverage union.

**VLM query modes:**
- *Room-query*: context contains uncovered room attributes + trajectory. VLM picks the next room to enter.
- *Early-stop*: context contains current room's objects + task goal. VLM decides whether to terminate local exploration.

The representative room image is central to VLM reasoning — it gives the model a wide-angle memory of the room to reason about object layout, even when the robot is currently looking at a different part of the space.

---

## Adaptations for the CMU VLN Challenge

### What changes and why

**Room level — one room instead of many.**
The challenge environment is a single bounded scene (not a multi-room building), so room-to-room routing and wall-detection-based segmentation are not needed. However, the Room node's core value — storing the best representative image of the scene — is retained. The single `RoomNode` acts as the global scene memory: it holds the best-coverage panoramic frame seen so far, updated whenever the robot reaches a new viewpoint.

**Viewpoint level — pose-driven instead of coverage-driven.**
SysNav creates viewpoints by comparing the current observation's voxel coverage against the accumulated union and adding a node when the new area exceeds threshold ε. This requires a 3D occupancy map. In our stack, the only readily available spatial signal is the robot's pose from `/state_estimation`. Viewpoint nodes are therefore created when the robot moves more than `viewpoint_radius` metres from every existing node — a simpler proxy for "the robot is now seeing a meaningfully different slice of the scene." The threshold (default 2.0 m) is tunable.

**Object level — externally populated instead of detector-driven.**
SysNav runs YOLOv8x + SAM2.1 on every frame to maintain the object graph autonomously. Our stack has no onboard detector; object information comes from the VLM's own output (the `rationale` field or a parsed `ObjectReferenceResponse`). The `add_object()` method is therefore a manual insertion point called by the responder after inference, rather than an automatic pipeline stage. See the *Object detection* note in the adaptations section for the upgrade path.

**Object detection — VLM-output-driven.**
SysNav runs YOLOv8x + SAM2.1 on every frame to autonomously maintain the object graph. Our stack has no onboard detector; object information comes from the VLM's own output (the `rationale` field or a parsed `ObjectReferenceResponse`). The `add_object()` method is therefore a manual insertion point called by the responder after inference. A dedicated detector could be added later (lightweight YOLO in-process or a separate sidecar) once camera–LiDAR extrinsics are confirmed available from the challenge platform.

**Wall detection — replaced by scene bounds.**
SysNav fits vertical planes on the LiDAR point cloud to segment rooms and define coverage boundaries. With only one room, room segmentation is unnecessary. Coverage boundaries are instead derived for free from the registered scan: a numpy `min`/`max` over the XY extent of the point cloud gives an approximate scene bounding box (`scene_bounds`) stored on the `RoomNode`. This tells the VLM how large the search area is — enough to reason about whether full coverage has been achieved — without any model inference.

**Edges — storage and entry points now, reasoning logic later.**
Full graph edge reasoning is deferred, but the storage fields and entry point hooks are wired now so the interface does not need to change later. Three edge types are prepared:

- *Room → Viewpoint* (affiliation): `RoomNode.viewpoint_tick_ids: list[int]` — appended in `_maybe_add_viewpoint()`. Trivial with one room; present for API completeness.
- *Viewpoint → Object* (visibility, stored as a reverse edge on the object): `ObjectObservation.observed_from_tick_ids: list[int]` — appended in `add_object()` on both creation and merge. To query "which objects were visible from viewpoint N": filter objects where N is in `observed_from_tick_ids`. Logic can later be enriched with LiDAR ray-casting.
- *Object → Object* (spatial relations): `ObjectObservation.spatial_relations: list[SpatialRelation]` — populated via a new `add_spatial_relation(label_a, label_b, relation)` method. The caller (responder) extracts relations from VLM rationale and calls this method; the relation type is a free string (e.g. `"left_of"`, `"near"`, `"on_top_of"`).

---

## Final data structure

```
SceneRepresentation
│
├── room: RoomNode
│     label: str                              # "scene" or question-derived
│     best_image: ImageFrame | None           # best-coverage frame
│     best_image_tick_id: int | None
│     best_image_position: Vector3 | None
│     scene_bounds: (Vector3, Vector3) | None # (min_xyz, max_xyz) from registered scan
│     viewpoint_tick_ids: list[int]           # ── edge: Room → Viewpoint (affiliation)
│
├── viewpoints: list[ViewpointNode]
│     tick_id: int
│     position: Vector3                       # map frame
│     yaw: float                              # heading in radians
│
├── objects: list[ObjectObservation]
│     label: str
│     position: Vector3                       # map frame, estimated
│     confidence: float
│     bbox_min / bbox_max: Vector3 | None     # from LiDAR, optional
│     first_tick_id / last_tick_id: int
│     observed_from_tick_ids: list[int]       # ── edge: Viewpoint → Object (visibility, reverse)
│     spatial_relations: list[SpatialRelation]# ── edge: Object → Object (spatial)
│
└── SpatialRelation  (frozen dataclass)
      target_label: str
      target_index: int                       # index into objects list
      relation: str                           # e.g. "left_of", "near", "on_top_of"
```

---

## Public API

```python
rep = SceneRepresentation(
    viewpoint_radius=2.0,          # m — min travel before new viewpoint node
    merge_radius=1.5,              # m — max distance to merge same-label objects
)

rep.update(snapshot: VLMInput)                             # call every tick before inference
rep.add_object(obs: ObjectObservation)                     # call after inference if object identified
rep.add_spatial_relation(label_a, label_b, relation: str)  # call after inference for object–object relations
```

---

## Design choices and trade-offs

| Decision | Choice | Alternative | Reason |
|---|---|---|---|
| Number of rooms | 1 fixed | Multi-room with wall detection | Challenge is single-scene; wall detection needs occupancy map we don't have |
| Viewpoint novelty | Distance threshold on pose | Voxel coverage delta (SysNav) | No 3D map available; pose is cheap and always present |
| Best image selection | Updated at every new viewpoint | Updated only if new position is farther from prior best | Simpler; new viewpoint = novel coverage by definition |
| Object population | VLM-output-driven (`add_object()`) | Auto-detect with YOLO + SAM | No onboard detector now; upgrade path exists once camera–LiDAR extrinsics are confirmed |
| Wall detection | Replaced by `scene_bounds` (numpy min/max of registered scan) | RANSAC plane fitting | One room only; terrain map already encodes obstacles; bounds are free from the scan |
| Graph edges | Storage + entry points now; reasoning logic deferred | Fully wired now | Storage fields and hooks added to avoid later interface changes; reasoning filled in once object population matures |
| Engine interface | Unchanged (single image) | Multi-image (room best + current) | Engine change deferred; `best_image` stored on `RoomNode` ready to be used |

---

## Test plan

### 1. ViewpointNode creation
- First `update()` with a pose always creates one viewpoint.
- Second `update()` with the robot < `viewpoint_radius` away → no new viewpoint added.
- Second `update()` with the robot > `viewpoint_radius` away → new viewpoint added.
- Stored `position` and `yaw` match the pose input.
- `update()` with `pose=None` → no viewpoint added, no crash.

### 2. RoomNode best image
- Initialises with `best_image=None`.
- Updated with the current frame when a new viewpoint is created.
- NOT updated when the robot hasn't moved enough (no new viewpoint node).
- `best_image_tick_id` and `best_image_position` track the correct tick and position.

### 3. ObjectObservation merging
- Adding to empty list → creates new node.
- Same label, within `merge_radius`, higher confidence → updates position and confidence in place.
- Same label, within `merge_radius`, lower confidence → existing node unchanged.
- Same label, outside `merge_radius` → creates a second node (different spatial instance).
- Different label, within `merge_radius` → creates a new node (different object type).

### 4. Edge storage

**Room → Viewpoint:**
- `room.viewpoint_tick_ids` is empty on init.
- Appended with the correct tick_id each time a new viewpoint is created.
- Not appended when robot hasn't moved enough.

**Viewpoint → Object (observed_from_tick_ids):**
- Empty on a newly created object.
- Tick_id appended when object is first added.
- Tick_id appended again when a higher-confidence observation merges into the existing node.
- Not appended when a lower-confidence observation is rejected.

**Object → Object (spatial_relations):**
- `add_spatial_relation(label_a, label_b, relation)` appends a `SpatialRelation` to label_a's node.
- No-op when label_a or label_b is not found.
- Multiple relations on the same object accumulate correctly.

### 5. Yaw extraction from quaternion
- Identity quaternion (`w=1, x=y=z=0`) → yaw = 0.
- Pure Z rotation 90° → yaw ≈ π/2.
- Pure Z rotation -90° → yaw ≈ -π/2.

---

## Integration plan (next steps)

1. **Phase 3 (this task):** implement `SceneRepresentation` and tests.
2. **Responder wiring:** `QwenResponder` calls `rep.update(snapshot)` before inference and embeds the representation into the user message via `build_user_message`.
3. **Object population:** after parsing `ObjectReferenceResponse` or extracting entity mentions from the numerical rationale, call `rep.add_object()`.
4. **Engine multi-image:** pass `scene.room.best_image` alongside the current frame once the engine interface is extended to accept multiple images.
