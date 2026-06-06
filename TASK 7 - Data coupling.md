# Task 7 — Data Coupling (Sub-task 1: Coverage Trajectory Generation)

## 1. Dataset Exploration

### 1.1 Scene Inventory

18 scene zips (15 dev + 3 eval holdout) at `CMU-VLN-Challenge-data/unity_env_models/`. Each contains:

| File | Description |
|------|-------------|
| `traversable_area.ply` | ASCII PLY — walkable floor surface as a 2D point cloud. Furniture/obstacles appear as **holes** in the surface. |
| `object_list.txt` | 3D bounding boxes: `id cx cy cz length width height heading "label"` |
| `Categories.csv` | Maps Unity object names → cleaned labels + NYU-40 semantic classes |
| `Dimensions.csv` | Per-object local-frame bounding box extents |
| `map.ply` | Full 3D point cloud (binary, 60–114 MB) |
| `map.jpg` / `render.jpg` | Top-down segmented screenshot / first-person render |

### 1.2 Traversable Area

The point cloud covers the walkable floor. Z-axis span < 0.12 m across all scenes — the problem is purely 2D. Holes in the surface correspond to furniture and obstacles, confirmed by overlaying object bounding boxes.

| Scene | Points | Objects |
|-------|--------|---------|
| studio | 64,159 | 73 |
| arabic_room | 43,016 | 85 |
| chinese_room | 43,980 | 96 |
| japanese_room | 43,160 | 63 |
| hotel_room_1 | 32,999 | 86 |
| hotel_room_2 | 27,772 | 95 |
| livingroom_1 | 37,418 | 106 |
| livingroom_2 | 33,067 | 88 |
| livingroom_3 | 54,934 | 108 |
| livingroom_4 | 30,286 | 120 |
| loft | 30,950 | 115 |
| office_1 | 39,702 | 112 |
| office_2 | 32,484 | 161 |
| home_building_1 | 62,496 | 432 |
| home_building_2 | 46,356 | 227 |
| office_building_1 | 171,368 | 0 |
| office_building_2 | 369,553 | 0 |
| office_building_2_no360 | 369,553 | 0 |

### 1.3 Objects

1,967 total objects across all scenes, 276 unique labels. Each has a 3D center, oriented bounding box, and text label. Heights range from floor-level (tables z≈0.2m) to wall-mounted (pictures z≈1.7m) to structural (ceiling z≈3.0m). The 3 office_building scenes have zero objects.

### 1.4 Challenge Questions

Each scene has 5 questions across 3 types:
- **Numerical**: "How many sofas are below a window?" — counting with spatial relations
- **Object reference**: "Find the pillow closest to the book on the stool." — identify by spatial context
- **Instruction following**: "Go near the stool under the picture..." — navigate between landmarks

All types require the VLM to recognize objects and reason spatially, so the trajectory must capture every object at sufficient resolution.

### 1.5 Robot and Sensor Specs

| Parameter | Value |
|-----------|-------|
| Camera | 360° panoramic, 1920×640 px, 360° HFOV, 120° VFOV, 10 Hz |
| Lidar | 3D Livox Mid360, 5 Hz |
| Max speed | 0.875 m/s |
| Platform | Mecanum wheel (~0.3m radius) |
| Time limit | 10 minutes per question |
| Waypoint interface | `Pose2D` on `/way_point_with_heading` |

Camera effective range: a 30cm object subtends ~32px at 3m (adequate), ~18px at 5m (marginal), ~9px at 10m (insufficient). **Effective observation radius: 3m.**

## 2. Problem Statement

### Input
- `traversable_area.ply`: 2D walkable floor point cloud
- `object_list.txt`: 3D bounding boxes of all objects

### Output
- Ordered list of waypoints `[(x₁, y₁, θ₁), ...]` compatible with `Waypoint(x, y, heading)`
- JSON file per scene with waypoints + statistics
- Visualization PNG

### Metrics
1. **Object coverage** (PRIMARY): % of objects visible from at least one waypoint (within 3m AND clear line-of-sight through 2D raycasting against furniture holes)
2. **Floor coverage**: % of walkable area within 3m of at least one waypoint
3. **Path length**: total distance (reported, not a constraint)
4. **Path validity**: all waypoints inside the walkable polygon

### Constraints
- Path must not place waypoints in non-traversable areas
- Must account for robot size (erode polygon by robot radius ~0.3m)
- 2D raycasting: furniture holes block line-of-sight to objects behind them
- No time budget constraint — priority is maximum coverage
- Must handle scenes with zero objects (floor coverage only)

## 3. Literature Review

### 3.1 Viewpoint Generation via Visibility + Set Cover (Art Gallery Problem)

**Idea** (from Gemini's recommendation in TASK.md): Discretize traversable area into candidate viewpoints. For each, compute visibility polygon via 2D raycasting against furniture/walls. Solve Minimum Set Cover to select smallest subset of viewpoints covering 100% of free space.

**Pros**: Directly optimizes for coverage. Handles arbitrary room topology. Minimal viewpoints = efficient trajectory. Well-studied problem with greedy O(log n) approximation.

**Cons**: Computing full visibility polygons is expensive for dense point clouds. Set cover is NP-hard (but greedy approximation has 1 + ln(n) guarantee).

### 3.2 TSP-Based Trajectory from Selected Viewpoints

**Idea**: Given a set of target viewpoints from Set Cover, compute pairwise distances (Euclidean or A*-based), solve Traveling Salesman Problem for optimal visit order.

**Pros**: Produces a short tour visiting all viewpoints. Nearest-neighbor + 2-opt heuristic runs fast for < 500 points. The robot's local planner handles actual obstacle avoidance.

**Cons**: TSP is NP-hard but heuristics work well at this scale.

### 3.3 Next-Best-View (NBV) Heuristic

**Idea**: Start at robot position, iteratively move to position with highest information gain: `Gain = New Area Visible / Travel Cost`. Repeat until 100% coverage.

**Pros**: Online/iterative, adapts to partial information. Natural for exploration.

**Cons**: Greedy, no global optimality. Requires online execution — less suitable for offline trajectory generation. Can produce long redundant paths.

### 3.4 Polygon Sweep Lines (Previous Attempt)

**Idea**: Horizontal sweep lines clipped to polygon, connected in serpentine order.

**Pros**: Simple, deterministic. **Cons**: Failed in practice — 86% object coverage on multi-room scenes. Ignores objects. Doesn't adapt to room topology. Complex boundary routing.

### Chosen Approach

**Algorithm 3.1 + 3.2: Object-Weighted Set Cover + TSP**

Simplification for 360° camera: visibility from any point is a circle of radius R, not a directional polygon. However, 2D raycasting against furniture holes is used to determine if objects are occluded. This means:
- Candidate viewpoint coverage = objects within 3m with clear line-of-sight
- Floor coverage = floor cells within 3m (no raycasting needed for floor)

The greedy set cover weights object coverage 10× higher than floor coverage, ensuring objects drive viewpoint selection. Floor fill handles the 3 zero-object scenes.

## 4. Algorithm Detail

1. Parse traversable_area.ply → 2D points; object_list.txt → object positions
2. Build polygon via `concave_hull(ratio=0.1, allow_holes=True)`
3. Erode polygon inward by robot_radius (0.3m) for safe viewpoint placement
4. Sample candidate viewpoints on a 0.5m grid inside the eroded polygon
5. For each candidate, compute visible objects AND floor cells: within 3m AND `LineString([viewpoint, target])` does not intersect any furniture hole (2D raycasting applied to both objects and floor)
6. Greedy weighted set cover: score = `10 × new_objects_covered + new_floor_cells_covered`
7. Build visibility graph (waypoints + eroded polygon vertices), compute geodesic distances via Dijkstra
8. Order selected viewpoints via nearest-neighbor TSP + 2-opt on geodesic distances
9. Route collision-free path through visibility graph, inserting intermediate polygon vertices at turns
10. Compute headings: atan2(dy, dx) toward next waypoint
11. Output waypoints + metrics

## 5. Results

### 5.1 Coverage Summary (all 18 scenes)

| Scene | Objects | Covered | ObjCov% | FloorCov% | Waypoints | Path (m) |
|-------|---------|---------|---------|-----------|-----------|----------|
| arabic_room | 85 | 79 | 92.9% | 88.1% | 14 | 17.1 |
| chinese_room | 96 | 96 | 100.0% | 99.1% | 11 | 15.2 |
| home_building_1 | 432 | 381 | 88.2% | 99.7% | 48 | 121.9 |
| home_building_2 | 227 | 218 | 96.0% | 99.6% | 41 | 106.8 |
| hotel_room_1 | 86 | 75 | 87.2% | 99.4% | 10 | 14.5 |
| hotel_room_2 | 95 | 95 | 100.0% | 99.7% | 10 | 14.1 |
| japanese_room | 63 | 63 | 100.0% | 99.2% | 5 | 8.7 |
| livingroom_1 | 106 | 104 | 98.1% | 99.4% | 6 | 10.2 |
| livingroom_2 | 88 | 88 | 100.0% | 99.8% | 12 | 25.3 |
| livingroom_3 | 108 | 108 | 100.0% | 99.1% | 24 | 36.3 |
| livingroom_4 | 120 | 119 | 99.2% | 94.3% | 16 | 16.4 |
| loft | 115 | 107 | 93.0% | 97.9% | 8 | 9.8 |
| office_1 | 112 | 112 | 100.0% | 97.7% | 10 | 16.2 |
| office_2 | 161 | 150 | 93.2% | 100.0% | 10 | 20.4 |
| office_building_1 | 0 | 0 | 100.0% | 95.0% | 40 | 175.6 |
| office_building_2 | 0 | 0 | 100.0% | 95.3% | 73 | 347.2 |
| office_building_2_no360 | 0 | 0 | 100.0% | 95.3% | 73 | 347.2 |
| studio | 73 | 72 | 98.6% | 99.6% | 8 | 15.8 |

### 5.2 Analysis

**8 of 15 object scenes achieve 100% or ≥98% object coverage.** 16 of 18 scenes achieve ≥95% floor coverage; arabic_room (88.1%) and livingroom_4 (94.3%) are lower due to large furniture holes occluding floor cells behind them.

Uncovered objects fall into two categories:

1. **Beyond 3m from walkable area** (majority): outdoor objects (trees at 9-12m, lamps at 8-10m, gates, exterior doors) that are physically unreachable from inside the building. These are inherent limitations — no trajectory through the walkable area can observe them at 3m resolution.

2. **Within 3m but occluded by non-host furniture** (6 objects total across all scenes): objects behind one piece of furniture when viewed from the nearest candidate, where 2D raycasting correctly identifies the occlusion. In 3D, the camera might see over low furniture — a limitation of the 2D raycasting model.

### 5.3 Key Design Decisions

- **2D raycasting for both objects and floor**: line-of-sight against furniture holes is checked for both object visibility and floor cell visibility, ensuring coverage metrics are consistent and realistic.
- **Host-hole exclusion**: objects sitting ON furniture (within 0.5m of a hole boundary) skip raycasting against that hole. Without this, studio drops from 98.6% to ~87% because wall-mounted and tabletop objects are always "blocked" by their own furniture.
- **Visibility-graph pathfinding**: collision-free routes between viewpoints via a visibility graph built from eroded polygon vertices. TSP uses geodesic distances (shortest collision-free path) rather than Euclidean, producing realistic visit ordering. Intermediate polygon vertices are inserted as turn-points where needed.
- **concave_hull ratio=0.1**: tighter than the initial 0.3, captures more furniture holes (studio: 2 holes at 0.1 vs 0 at 0.3).
- **2-opt TSP improvement**: reduces path length ~10-20% over pure nearest-neighbor on larger scenes.
- **Floor subsampling**: every 10th traversable point as floor cells — sufficient resolution without excessive computation.

### 5.4 Verification

- 27 unit + integration tests pass (polygon, coverage, raycasting, floor occlusion, TSP, pathfinding, full pipeline)
- All path segments validated via explicit raycasting against furniture holes — zero crossings across all 18 scenes
- All waypoints are inside the eroded polygon (safe placement accounting for robot radius)
- Visual inspection of PNG plots confirms: paths navigate around furniture, coverage circles overlap objects, trajectory visits all rooms
- Lint clean (ruff)
