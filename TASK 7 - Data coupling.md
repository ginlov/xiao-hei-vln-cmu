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

Simplification for 360° camera: visibility from any point is a circle of radius R, not a directional polygon. However, 2D raycasting against furniture holes is used to determine if objects and floor cells are occluded. This means:
- Candidate viewpoint coverage = objects/floor within 3m with clear line-of-sight
- Floor coverage also uses raycasting — cells behind furniture are not counted as visible

The greedy set cover weights object coverage 10× higher than floor coverage, ensuring objects drive viewpoint selection. Floor fill handles the 3 zero-object scenes.

## 4. Algorithm Detail

1. Parse traversable_area.ply → 2D points; object_list.txt → object positions
2. Build polygon via `concave_hull(ratio=0.1, allow_holes=True)`
3. Erode polygon inward by robot_radius (0.3m) for safe viewpoint placement
4. Sample candidate viewpoints on a 0.25m grid inside the eroded polygon
5. For each candidate, compute visible objects AND floor cells: within 3m AND `LineString([viewpoint, target])` does not intersect any furniture hole (2D raycasting applied to both objects and floor)
6. Greedy weighted set cover: score = `10 × new_objects_covered + new_floor_cells_covered`
7. Build visibility graph (waypoints + eroded polygon vertices), compute geodesic distances via Dijkstra
8. Order selected viewpoints via nearest-neighbor TSP + 2-opt on geodesic distances
9. Route collision-free path through visibility graph, inserting intermediate polygon vertices at turns
10. Compute headings: atan2(dy, dx) toward next waypoint
11. Output waypoints + metrics

## 4.1 Potential Post-Processing Optimizations

The current trajectories are functional but not fully optimized for path length. Several post-processing algorithms could reduce path length and waypoint count:

### 4.1.1 Greedy Shortcutting

For each consecutive triplet (A, B, C) in the path, check if the direct segment A→C is collision-free (stays inside the eroded polygon). If so, remove B. Repeat until no more removals are possible. This eliminates unnecessary polygon-corner detours inserted by the visibility graph routing.

**Pros**: Simple to implement, provably correct (each removal is validated), directly targets the most visible inefficiency (zigzag corners). **Cons**: Greedy — the removal order may not be globally optimal.

### 4.1.2 Funnel Algorithm (String-Pulling)

Classic computational geometry technique for finding the shortest path through a channel (sequence of adjacent triangles or portals). After routing through the visibility graph, the path passes through a sequence of polygon "corridors." The funnel algorithm finds the tightest path through these corridors by maintaining a deque of tangent lines.

**Pros**: Produces the true shortest path through the same corridor sequence — optimal, not just greedy. Standard in game/robotics navmesh pathfinding. **Cons**: Requires decomposing the polygon into a triangulation or portal sequence, more complex to implement.

### 4.1.3 Or-opt / 3-opt (TSP Improvement)

Extensions to the current 2-opt local search for TSP ordering. **Or-opt** moves subsequences of 1–3 consecutive waypoints to a better position in the tour. **3-opt** considers removing 3 edges and reconnecting the tour in all possible ways. These can escape local minima that 2-opt misses.

**Pros**: Better tour ordering reduces total travel distance. **Cons**: Diminishing returns at our scale (5–73 coverage viewpoints). The routing step (visibility graph) dominates path length more than visit order at this scale.

## 5. Results

### 5.1 Coverage Summary (all 18 scenes)

| Scene | Objects | Covered | ObjCov% | FloorCov% | Waypoints | Path (m) |
|-------|---------|---------|---------|-----------|-----------|----------|
| arabic_room | 85 | 83 | 97.7% | 97.8% | 12 | 18.3 |
| chinese_room | 96 | 96 | 100.0% | 98.8% | 8 | 14.8 |
| home_building_1 | 432 | 381 | 88.2% | 99.7% | 48 | 122.1 |
| home_building_2 | 227 | 222 | 97.8% | 99.7% | 47 | 103.5 |
| hotel_room_1 | 86 | 84 | 97.7% | 99.8% | 16 | 26.4 |
| hotel_room_2 | 95 | 95 | 100.0% | 99.6% | 11 | 14.6 |
| japanese_room | 63 | 63 | 100.0% | 99.2% | 6 | 9.0 |
| livingroom_1 | 106 | 104 | 98.1% | 99.4% | 7 | 11.4 |
| livingroom_2 | 88 | 88 | 100.0% | 99.7% | 13 | 25.4 |
| livingroom_3 | 108 | 108 | 100.0% | 99.4% | 21 | 34.2 |
| livingroom_4 | 120 | 120 | 100.0% | 97.7% | 7 | 10.2 |
| loft | 115 | 109 | 94.8% | 98.9% | 9 | 11.5 |
| office_1 | 112 | 112 | 100.0% | 97.9% | 12 | 17.2 |
| office_2 | 161 | 161 | 100.0% | 100.0% | 12 | 22.0 |
| office_building_1 | 0 | 0 | 100.0% | 95.5% | 41 | 184.0 |
| office_building_2 | 0 | 0 | 100.0% | 95.1% | 74 | 330.8 |
| office_building_2_no360 | 0 | 0 | 100.0% | 95.1% | 74 | 330.8 |
| studio | 73 | 72 | 98.6% | 99.6% | 8 | 17.1 |

### 5.2 Analysis

**11 of 15 object scenes achieve ≥97.7% object coverage** (7 at 100%). All 18 scenes achieve ≥95% floor coverage. The 0.25m candidate grid (vs initial 0.5m) significantly improved coverage by providing more viewpoint angles to see around furniture.

Uncovered objects fall into two categories:

1. **Beyond 3m from walkable area** (majority): outdoor objects (trees at 9-12m, lamps at 8-10m, gates, exterior doors) that are physically unreachable from inside the building. These are inherent limitations — no trajectory through the walkable area can observe them at 3m resolution.

2. **Within 3m but occluded by non-host furniture** (6 objects total across all scenes): objects behind one piece of furniture when viewed from the nearest candidate, where 2D raycasting correctly identifies the occlusion. In 3D, the camera might see over low furniture — a limitation of the 2D raycasting model.

### 5.3 Objects Outside the Polygon Boundary

A large majority of objects in each scene have their 2D center **outside** the traversable polygon boundary. This is expected: the polygon represents the walkable floor surface, while objects sit on, against, or above furniture and walls that form the boundary itself.

| Scene | Outside / Total | Near (<1m) | Mid (1–3m) | Far (>3m) |
|-------|----------------|------------|------------|-----------|
| arabic_room | 67 / 85 | 63 | 4 | 0 |
| chinese_room | 85 / 96 | 84 | 1 | 0 |
| home_building_1 | 385 / 432 | 280 | 54 | 51 |
| home_building_2 | 182 / 227 | 149 | 31 | 2 |
| hotel_room_1 | 76 / 86 | 63 | 12 | 1 |
| hotel_room_2 | 86 / 95 | 79 | 7 | 0 |
| japanese_room | 58 / 63 | 52 | 6 | 0 |
| livingroom_1 | 95 / 106 | 84 | 10 | 1 |
| livingroom_2 | 79 / 88 | 74 | 5 | 0 |
| livingroom_3 | 102 / 108 | 92 | 10 | 0 |
| livingroom_4 | 116 / 120 | 112 | 4 | 0 |
| loft | 111 / 115 | 75 | 36 | 0 |
| office_1 | 102 / 112 | 88 | 14 | 0 |
| office_2 | 156 / 161 | 145 | 11 | 0 |
| studio | 67 / 73 | 60 | 7 | 0 |

Objects outside the polygon fall into five categories:

1. **Wall-mounted items** (pictures, windows, TVs, wall lamps, posters, light switches): positioned on walls that form the polygon boundary, typically 0.3–1.5m outside at z=1.0–2.1m. Present in every scene.

2. **Furniture against walls** (sofas, beds, nightstands, chairs, tables): large items whose center of mass is beyond the traversable boundary because the robot cannot drive under or through them. The walkable floor ends at the furniture edge, so the object center is 0.3–0.7m outside. Most common category across all scenes.

3. **Items on furniture** (pillows, bottles, books, vases, bowls, glasses, keyboards): sit on top of wall-adjacent furniture. Their 2D projection falls outside the polygon because the host furniture extends past the walkable boundary. Typically 0.3–1.2m outside at z=0.2–1.7m.

4. **Ceiling-mounted items** (ceiling lights, focus lights, spot lights): directly above rooms but their 2D center may be over a wall or furniture hole. Typically 0.0–1.6m outside at z=2.6–4.0m.

5. **Truly outdoor items** (trees, gates, exterior doors, outside lamps): only found in home_building_1 (51 objects) and home_building_2 (2 objects). These are 5–12m from the polygon boundary and physically unreachable from inside the building.

**Impact on coverage**: despite being outside the polygon, most objects in categories 1–4 are still **within the 3m observation radius** of nearby candidate viewpoints and are successfully covered. The coverage algorithm does not require objects to be inside the polygon — only that a viewpoint inside the polygon can see them within 3m with clear line-of-sight. The 55 truly unreachable objects (category 5, >3m from any candidate) account for the majority of uncovered objects in the results.

### 5.4 Key Design Decisions

- **2D raycasting for both objects and floor**: line-of-sight against furniture holes is checked for both object visibility and floor cell visibility, ensuring coverage metrics are consistent and realistic.
- **Host-hole exclusion**: objects sitting ON furniture (within 0.5m of a hole boundary) skip raycasting against that hole. Without this, studio drops from 98.6% to ~87% because wall-mounted and tabletop objects are always "blocked" by their own furniture.
- **Visibility-graph pathfinding**: collision-free routes between viewpoints via a visibility graph built from eroded polygon vertices. TSP uses geodesic distances (shortest collision-free path) rather than Euclidean, producing realistic visit ordering. Intermediate polygon vertices are inserted as turn-points where needed.
- **concave_hull ratio=0.1**: tighter than the initial 0.3, captures more furniture holes (studio: 2 holes at 0.1 vs 0 at 0.3).
- **2-opt TSP improvement**: reduces path length ~10-20% over pure nearest-neighbor on larger scenes.
- **Floor subsampling**: every 10th traversable point as floor cells — sufficient resolution without excessive computation.

### 5.5 Verification

- 27 unit + integration tests pass (polygon, coverage, raycasting, floor occlusion, TSP, pathfinding, full pipeline)
- All path segments validated via explicit raycasting against furniture holes — zero crossings across all 18 scenes
- All waypoints are inside the eroded polygon (safe placement accounting for robot radius)
- Visual inspection of PNG plots confirms: paths navigate around furniture, coverage circles overlap objects, trajectory visits all rooms
- Lint clean (ruff)
