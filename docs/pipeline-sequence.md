# Ground, Lift, Drive

One grounding step, end to end.

The division the whole pipeline is built on: **the model proposes semantics,
geometry decides metres, the stack decides motion** — and no stage is allowed to
do another's job. Every measured failure in the task reports is a case of one
actor having been trusted with another's decision.

| Actor | Decides | Never decides |
|---|---|---|
| **Claude** — semantics | Which pixels are the thing. Names, boxes, occlusion, candidates and anchors. | The winner of a comparison, or a distance anything acts on. |
| **Lidar** — metres | Where that is, in the map frame: a ray through the box centre, the returns inside a cone, one depth cluster. | What the thing is. |
| **Stack** — motion | How the vehicle gets there. We model `waypointXYRadius` and the terrain ourselves so the goal we publish is one the converter will not move somewhere we did not mean. | Whether we have arrived. |

---

## One question becomes an ordered plan

The sentence is decomposed once, by the model, at step 0. The executor holds the
progress cursor and only ever moves it forward — the model is shown which step is
current but is never asked which step to do next.

```mermaid
sequenceDiagram
    autonumber
    participant U as operator
    participant X as execute_plan
    participant C as Claude
    participant R as run_goto / run_pass

    U->>X: "First go to the nightstand with a clock on it, then take the path between the dining table and the picture, and stop at the trash can…"
    X->>C: decompose(sentence)
    C-->>X: ordered clauses
    Note over X,C: GOTO · PASS · GOTO — one cached call, 3.3 s
    X->>X: preflight() — topics live, no rival publisher on /way_point_with_heading

    loop each clause, in order, vehicle never reset
        X->>X: leg_deadline = share of the 540 s budget minus RESERVE_S per leg still to come
        X->>R: run_goto(phrase, k)
        R-->>X: Outcome(arrived, why, xy)
        Note over X,R: a failed leg does not end the question:<br/>the score is per-constraint with partial credit
    end
```

`scripts/execute_plan.py` · `scripts/decompose.py` · budget 540 s, RESERVE_S 70 s
per remaining leg.

---

## One grounding step

This is the loop that costs money and time — roughly **35 seconds and one Claude
call per iteration**, against a ten-minute budget for the whole question.

```mermaid
sequenceDiagram
    autonumber
    participant L as run_goto
    participant B as Robot bridge
    participant S as ROS / sim
    participant C as Claude
    participant F as lift
    participant V as ConverterModel

    loop until arrived · max_steps · leg_deadline
        L->>B: capture()
        B->>S: docker exec — snapshot four topics together
        S-->>B: /camera/image/compressed · /registered_scan · /terrain_map · /state_estimation
        B-->>L: equirect 1920×640 · scan (map frame) · terrain · pose

        L->>L: faces_of(eq) — gnomonic remap, 4 × 640² at 0° 90° 180° 270°, 100° FOV, ~10° overlap

        L->>C: build_prompt(phrase, visited, mission) + 4 faces + previous target crop
        C-->>L: visible · box_2d · feature_box_2d · occlusion · distance_m · confidence · target_state · candidates · anchors · explore · here
        L->>L: settle_coord_space — a box outside the image is not a pixel box, whatever it says

        alt visible = false
            L->>V: explore_direction(model's heading, terrain)
            V-->>L: nearest drivable bearing that clears MIN_EXPLORE_M
            Note over L,V: reach is a gate, not an objective — maximising it<br/>prefers open floor to the doorway, every time
        else visible = true
            L->>F: resolve_relation(candidates, anchors)
            F-->>L: winner measured, not asked for
            L->>F: aim_box(reply, scan, pose)
            F-->>L: target box — feature box refused when the two lift more than 0.35 m apart
            L->>F: next_waypoint(box, scan, pose)
            F->>F: ray_from_box → scan_to_camera → locate(cone) → dominant_cluster → range
            F->>F: in_blind_cone · size_gate
            F-->>L: Waypoint(committed, range) or a bounded step
            L->>L: bind_target — refine · overrule · reject
        end

        L->>V: best_waypoint_toward(aim, vehicle)
        V->>V: legal_points(terrain), then settle() — the fixed point the C++ converter chases
        V-->>L: goal · settles_at · reach · will_move

        L->>L: arrival gates
        L->>B: drive_to(goal)
        B->>S: /way_point_with_heading (Pose2D)
        S-->>B: /state_estimation at 100–200 Hz until settled · stalled · timeout
        B-->>L: moved_m · dist_to_requested_m · why
    end
```

`approach_loop.run_goto` · one Claude call per iteration · ~35 s wall clock per
step, measured over the 0814 runs.

---

## Inside the lift — a box becomes metres

The single most error-prone hop in the system. A box edge that slips past the
object's silhouette puts the ray on the wall behind it, and the error along the
ray is unbounded.

`next_waypoint` never guesses a range. It either commits to a measured one or
takes a step whose length is itself measured.

```mermaid
flowchart TB
    A["box_2d — pixels on one face"] --> B["ray_from_box<br/>face pixel → camera dir → map dir"]
    B --> C["scan_to_camera<br/>registered scan into the camera frame"]
    C --> D["cone = clip(min(w,h)/4, 1°, 5°)<br/>widen ×1.5 ×2.5 ×4 until ≥12 returns"]
    D --> E["dominant_cluster(gap 0.35 m)<br/>→ median range"]
    E --> G1{"in_blind_cone?"}
    G1 -- yes --> R1["never commit — step and turn instead"]
    G1 -- no --> G2{"size_gate<br/>implied height in 0.15–4.0 m"}
    G2 -- no --> R2["reject the range, take a bounded step"]
    G2 -- yes --> G3{"bind_target<br/>jump ≤ 1.0 m · nearer reading · corroborated"}
    G3 -- no --> R3["hold the old binding, file this one as pending"]
    G3 -- yes --> OK["committed<br/>waypoint = ray × (range − 0.6 m standoff)"]
    R1 --> S["free_range_along — how far the scan says<br/>we can drive before something is in the way"]
    R2 --> S
    S --> T["step = min(2.0, max(free − 0.6, 0.3))"]
```

### Why the fallback step is not a guess

`free_range_along` does not estimate where the target is. It answers a different
question the scan can already answer: *how far along this bearing can the vehicle
drive before something is in the way.* It takes the returns between 0.05 m and
1.2 m above the vehicle — the same population `local_planner` refuses to drive
into — and finds the nearest one within 0.35 m of the ray.

The blind cone heals itself by stepping. `COVERAGE_FLOOR_DEG` is body-fixed, so
it rotates with the robot: a bearing at −120° has a floor of −2°, but driving one
leg toward the target swings it near 0° azimuth where the floor is −32.4°, and
the next lift is sound.

`scripts/vlm_approach.py` · `scripts/vlm_locate.py` · `approach_loop.bind_target`

---

## Where the pipeline is allowed to say no

Each of these exists because a run reported success and was wrong.

| Gate | Refuses | Found by |
|---|---|---|
| `in_blind_cone` | A bearing below the scanner's elevation floor. It does not merely lose the lift — it returns a confident wrong one. | 4.70 m against a true ~2.9 m |
| `size_gate` | A range whose implied object height falls outside 0.15–4.0 m. | passed a 0.15 m reading by 1 cm on `o_2_0814_02` |
| `aim_box` | A `feature_box_2d` that lifts more than 0.35 m from the target box — it is the anchor, not a feature. | drove at the window it was told the cooler was *near* |
| `bind_target` | A reading that jumps more than 1.0 m without a nearer measurement or a second reading agreeing with it. | one wrong binding steering the rest of a leg |
| `says_far` | An arrival declared while the model still reads the target as `far`, on any of the four arrival paths. | 3/3 reported, all short; one by 6.5 m through glass |
| `crosses_gate` | A goal whose *settled* position, or the run to it, crosses a forbidden corridor. Tested on the trajectory, not the waypoint. | a keep-out that caused the violation |

---

## What is measured in what

Four frames, and every bug worth the name lives in a conversion between two of
them. `/registered_scan` arriving already in the map frame is the one piece of
luck here — a stale scan is wrong about *when*, not about *where*.

| Frame | What it is | Units |
|---|---|---|
| **equirect** | The raw panorama, cropped vertically by the sim. | 1920 × 640 px · 360° × 120° |
| **face** | Gnomonic reprojection. Angular size must be measured between edge directions, not by pixels × FOV. | 4 × 640² · 100° FOV |
| **camera** | Where the ray and the returns are compared. One `sensor_to_camera_transform` from the scan. | metres, right-handed |
| **map** | Where bindings, keep-outs, waypoints and the score all live. Heading inverts yaw — `explore_goal` is the only place that knows. | metres · yaw CCW |

### Platform constants the converter model reproduces

| Name | Value | Why it is modelled rather than assumed |
|---|---|---|
| `waypointXYRadius` | 0.30 m | A goal the vehicle settles inside of produces zero motion, which the post-drive test reads as the stack clamping an approach. |
| `vehicleWidth` | 0.50 m | With `obstacleDisThre` 0.75 m, sets which terrain cells can hold a legal waypoint at all. |
| `STANDOFF_M` | 0.60 m | Aim at the target, not at a standoff from it — the inflation is what the standoff is *for*. |
| `NEAR_M` | 1.50 m | Below this, "the converter has nothing closer" means arrival. Above it, it means a terrain map that has not seen the ground near the object. |
| `PROGRESS_M` | 0.25 m | Moving less than this toward a destination is the stack declining, not the vehicle arriving. |

---

## What the diagrams do not show

Three things are drawn as single arrows that are not simple in practice.

**The lift can be blind and not know it.** Glass returns the beam. On
`o_2_0814_02` every ray toward the door produced one tight cluster at 1.65 m and
nothing beyond, so no cluster-selection rule reaches the door — the only signals
that the target is further out are `occlusion` and `target_state`, and until the
`says_far` gate nothing read either of them.

**`dominant_cluster` takes the biggest cluster, not the nearest.** At range the
biggest is often the wall behind the object. Replacing it with the nearest
credible cluster was measured at 4–6 m: 0.40 m → **0.16 m**; beyond 6 m: 3.33 m →
1.69 m. That change has not been made.

**The four faces are re-encoded every call.** They are 2189 of the prompt's
~6000 input tokens and they dominate prefill, which is why prompt caching
measured as a cost win and not a latency one — 2.81 s uncached against 2.96 s
cached.

**An exploration step costs the same as an approach step.** On `hb_1_0814_01`
seven grounding calls — about 280 s, 70% of that leg's budget — were spent inside
a 2.1 × 0.8 m box in a doorway where the drivable reach was under a metre.

---

Drawn from `scripts/execute_plan.py`, `approach_loop.py`, `vlm_approach.py`,
`vlm_locate.py`, `waypoint_converter_model.py`, `robot_io.py`.
