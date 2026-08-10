# Driving the robot from a VLM

How Team Xiao Hei turns a sentence like *go to the lantern closest to the fan
decoration* into waypoints, and what each part of it has been measured to do.

The design rule, which every decision below follows:

> **The VLM proposes semantics. Geometry decides metrics.**

The model is asked only what it is good at — which pixels are the thing the
sentence names. Every number that reaches the robot comes from the lidar, the
pose, or arithmetic on the two. Where we broke this rule the data punished us:
TASK 26 asked the model to judge whether the scanner could be trusted and it
answered `same_space: true` for chairs on the far side of a glass door; asked
it for distance and got a systematic 1.81× overestimate.

---

## 1. Why a VLM at all

Our own perception stack — YOLO-World + SAM 2.1, fused into an object map — is
a fair comparison: both are RGB-only recognition feeding the *identical* lidar
lift. The difference is only *when vocabulary is committed*. YOLO-World needs a
class list before it sees the question; the VLM reads the phrase and the image
together.

| | VLM, one frame from the start pose | object map, start pose | object map, full tour |
|---|---|---|---|
| named object localised within 1.0 m | **72.2 %** | 48.3 % | 67 % |
| median error of hits | **0.11 m** | 0.236 m | — |

A single glance from where the robot spawns beats a complete tour of the scene.
That is the whole case for the architecture.

### What the start-pose result is made of

54 unique (scene, phrase) pairs, one frame each, no driving. A hit is a lifted
position within 1.0 m of any ground-truth instance of the phrase.

| | n | share |
|---|---|---|
| localised within 1.0 m | 39 | **72.2 %** |
| declined outright — target not visible from the start pose | 2 | 3.7 % |
| claimed and wrong | 13 | 24.1 % |

Hit error: median **0.11 m**, p90 0.52 m, worst 0.90 m — against the 0.58 m
budget the reference trajectories imply.

The 13 wrong claims are the interesting part, because reading the annotated
panels (`artifacts/fp/false_positives.jpg`) rather than the error column
changes what they are:

| cause | n | what it looks like |
|---|---|---|
| no lidar return | 4 | ground truth **inside** the box; the cone is empty even at 14° |
| ray overshot | 3 | open stair, pendant lamp — the beam passes through and finds the far wall |
| ray stopped short | 2 | box correct, the beam measures an occluder in front |
| wrong instance | 2 | balcony chairs through glass; a side table read as the tray's table |
| truth not in this face | 2 | the elephant figurine answered for the horse |

**Nine of the thirteen are cases where the model identified the object
correctly and the depth lift failed.** Counting only genuine misidentification
gives 4/54 = **7.4 %**. The recognition is not the weak part; the ranging is —
which is why §5 is mostly about deciding when *not* to believe a range.

Per scene, on a small denominator:

| scene | hits |
|---|---|
| office_2 | 8/9 · 88.9 % |
| japanese_room | 7/8 · 87.5 % |
| office_1 | 7/8 · 87.5 % |
| arabic_room | 5/7 · 71.4 % |
| chinese_room | 6/9 · 66.7 % |
| livingroom_3 | 4/7 · 57.1 % |
| loft | 2/6 · 33.3 % |

`loft` is the outlier, and its four failures are worth naming individually
because none of them is a recognition failure:

| phrase | what happened |
|---|---|
| *tv remote* | declined, confidence 0.03 — 6.4 m away and a few pixels wide |
| *sphere decoration* | declined, confidence 0.20 — 7.1 m away |
| *cabinet* | found at 1.4 m, **no lidar return** in the cone |
| *stairs* | found, confidence 0.94, lifted 5.28 m against a true 1.86 m — the beam went through an open stair to the far wall |

Two declines on small distant targets, and two ranging failures on objects the
model had correctly identified. Both kinds are exactly what the loop's step-
and-re-observe fallback exists for, and neither is visible in a one-frame test.

Details and method in
[TASK 26](tasks/TASK%2026%20-%20Can%20a%20VLM%20ground%20a%20referring%20expression,%20measured.md).

---

## 2. Two loops, not one

A grounding call costs 1–3 s and $0.0265. The reference paths are ~10 m of
driving. So the VLM cannot sit in the control loop:

| loop | rate | job |
|---|---|---|
| geometry | control rate | lift, gate, choose a waypoint, watch for arrival |
| VLM | on events — start, arrival, end of an exploration leg | which pixels are the target |

Five to ten calls per question, ~$4 for a full competition run of fifteen
questions. Cost is not the constraint; latency is.

---

## 3. The loop

```
STRATEGY drive_to(phrase):

    prev_crop ← none                      # last view of the target, for continuity
    repeat up to MAX_STEPS:

        # ---- observe -------------------------------------------------------
        equirect, scan, terrain, pose ← capture()      # 1920×640, /registered_scan,
                                                       # /terrain_map, /state_estimation
        faces ← unwrap(equirect)                        # 4 × 640² gnomonic, FOV 100°,
                                                        # yaw 0/90/180/270

        # ---- ground (the only model call) ----------------------------------
        reply ← VLM(prompt(phrase), faces, previous = prev_crop)
        if reply is unparseable: stop

        if not reply.visible:
            # the model names a heading to look at; note the frames disagree
            # in sign — the prompt counts faces clockwise, the map measures
            # yaw counter-clockwise
            θ ← yaw(pose) − radians(reply.explore.heading_deg)
            drive_to_point(pose.xy + TURN_STEP · [cos θ, sin θ])
            prev_crop ← none
            continue

        box, face ← reply.feature_box_2d or reply.box_2d, reply.image_index

        # ---- decide comparatives by measuring, not by asking ---------------
        if reply.relation in {closest_to, farthest_from, between}:
            box, face ← resolve_relation(reply, scan, pose)   # §4

        # ---- box → waypoint ------------------------------------------------
        wp ← next_waypoint(box, face, scan, pose, phrase)     # §5

        # ---- predict what the platform will do with it ---------------------
        aim  ← pose.xy + unit(wp.xy − pose.xy) · wp.range     # the target itself
        goal ← nearest point the converter will accept, to aim # §6
        if predicted_movement(goal) < PROGRESS:
            declare "as near as the platform allows"; stop     # no drive, no call

        # ---- drive ---------------------------------------------------------
        result ← drive_to_point(goal)                          # §7
        prev_crop ← crop(faces[face], box)

        if wp.committed:
            if result.gap ≤ ARRIVE_TOL:      declare arrived; stop
            if result.moved < PROGRESS:      declare clamped; stop
            # else: settled short — re-observe from here
        else if result.moved < PROGRESS:
            stop            # a step that went nowhere; re-grounding from an
                            # unchanged pose asks the same question
```

Constants, all in `scripts/`: `MAX_STEPS = 6`, `TURN_STEP = 0.8 m`,
`PROGRESS = 0.25 m`, `ARRIVE_TOL = 0.35 m` (just over the stack's
`waypointXYRadius = 0.3`).

---

## 4. Comparative relations

The dominant phrasing in the official question set — 47–60 % of object
reference and instruction following questions use *closest / nearest / farthest
/ between*. Asked for the winner directly, the model answered the first live
relational phrase with **the anchor**: given *the lantern closest to the fan
decoration* it returned the fan decoration.

So the prompt (v4) asks it to enumerate instead, and geometry compares:

```
resolve_relation(reply, scan, pose):
    candidates ← [lift(box) for box in reply.candidates]   # ≥ 2 needed
    anchors    ← [lift(box) for box in reply.anchors]      # ≥ 1 needed
    if too few of either: return none        # fall back to the model's own pick
                                             # — the only answer available

    score(c) = Σ |c − a| over anchors     if relation is "between"
             = |c − anchors[0]|           otherwise
    return argmin score      (argmax for "farthest_from")
```

Measured on that phrase: bearing error **22.5° → 1.5°**, and the chosen
lantern's distance to the fan came out 0.61 m against a ground truth of 0.63 m
(planar, centre to centre). The next-nearest candidate measured 3.97 m, so the
comparison had a wide margin — as it usually will, since these phrases exist to
disambiguate objects a person can tell apart.

---

## 5. Box to waypoint

```
next_waypoint(box, face, scan, pose, phrase):

    d_cam ← direction of the box centre                    # gnomonic LUT
    d_map ← camera → map rotation applied to d_cam
    if d_map is vertical:  return "target is overhead"      # a ceiling lamp

    (w°, h°) ← true angular size of the box                 # arccos between edge
                                                            # directions, not
                                                            # pixels/size × FOV
    cone  ← clip(min(w°, h°) / 4, 1°, 5°)
    lift  ← median depth of scan returns inside that cone   # widened until
                                                            # enough returns

    az, el ← bearing of d_cam in the sensor frame
    blind  ← el < elevation_floor(az)                       # §5.1

    if lift exists and not blind and size_gate(lift, h°):   # §5.2
        return DESTINATION at  pose.xy + d_map · lift       # committed

    # Never guess a range. Step into space the scan says is free and look again;
    # the lift gets easier as 1/r².
    free ← distance along d_map to the first obstacle-band return within ±0.35 m
    return STEP at pose.xy + d_map · min(MAX_STEP, free − standoff)
```

### 5.1 The blind cone

The simulator's lidar is tilted forward, so how far below horizontal it sees
depends on bearing. Measured as the 0.5th percentile of return elevation per
15° azimuth bin across seven scenes, repeating to within 1–3°:

| azimuth | 7° | 52° | 82° | 112° | 142° | 172° |
|---|---|---|---|---|---|---|
| lowest elevation returned | −32° | −26° | −16° | −5° | +4° | +9° |

This is the entire explanation for "no lidar return": all four such cases in
the TASK 26 sweep point below the floor at their bearing, 4 for 4.

**A blind bearing does not merely lose the lift — it can return a confident
wrong one.** The cone widens upward until it finds something, and what it finds
is the wall above the object. On the first live run a target at −120° lifted to
4.70 m against a true ~2.9 m, with an implied height of 0.62 m that the size
gate happily accepted. So a blind lift is never committed.

The cone is body-fixed, so driving one leg toward the target swings the bearing
toward 0° where the floor is −32°. Observed twice live: −120° → +23°, and
−146° → +50°. The step is not a concession; it is what makes the next lift
sound.

### 5.2 The size gate

```
size_gate(range, h°):  0.15 m ≤ 2 · range · tan(h° / 2) ≤ 4.0 m
```

Three failure modes — glass, open structure, an occluder in front — all surface
as one contradiction: between the measured range and the size the object would
have to be at that range. That contradiction is a multiplication we can do
ourselves.

| gate | hits | false positives |
|---|---|---|
| none | 72.2 % | 24.1 % |
| A — decline when the cone is empty | 72.2 % | 16.7 % |
| **A + B (size band)** | **70.4 %** | **11.1 %** |

Gate A is free: no threshold, no model, and refusing to answer without depth
data is strictly better than answering from none. Per-class size priors instead
of the global band looked like the obvious improvement and **lose** at every
tolerance — every surviving false positive is a *wrong instance* whose size
matches the right one.

---

## 6. Predicting the platform

`/way_point_with_heading` is not a position command. `waypointConverter`
**discards** any waypoint inside the obstacle inflation and re-minimises
globally — the replacement owes nothing to the direction we meant. On
`japanese_room` a waypoint 0.19 m from the target sent the robot 1.08 m to the
far side of it.

We reimplement that decision from `/terrain_map`, which is on the challenge's
allowed-topic list, so we know the answer before spending a drive and a call:

```
travArea     ← terrain points with height-above-ground <  0.05 m   (voxel 0.05)
obstacleArea ← the rest                                            (voxel 0.05)
legal(p)     ← no obstacleArea point within 0.75 m of p

snap(waypoint, vehicle):
    candidates ← travArea within 5.0 m of the VEHICLE
    return argmin over legal candidates of
               |p − waypoint| + 0.5 · |p − vehicle|

settle(waypoint, vehicle):        # the C++ re-snaps at 10 Hz and the vehicle
    v ← vehicle                   # term re-centres as the robot moves, so the
    while |snap(waypoint, v) − v| ≥ 0.3:      # resting place is the fixed
        v ← v stepped toward snap(waypoint, v) # point, not the first snap
    return v
```

So instead of publishing `range − standoff` and discovering the result, publish
**the legal point nearest the target**. A legal point is a local minimum of the
converter's score — stepping back toward the vehicle by *d* adds *d* and
removes only *0.5 d* — so it is republished as itself.

Prediction error against three drives: 0.104 m, 0.081 m, and 0.05 vs 0.057 m.
Full account in
[TASK 29](tasks/TASK%2029%20-%20The%20converter%20is%20not%20a%20clamp,%20and%20we%20can%20predict%20it.md).

---

## 7. Knowing we arrived

The autonomy stack publishes `/way_point_reached`, and it is **not** on the
README's list of five topics an AI module may use at test time. Both halves of
the replacement come from `/state_estimation`:

| condition | meaning |
|---|---|
| `gap ≤ 0.35 m` to our own waypoint | got where we asked |
| no motion > 0.05 m for 4 s | the stack will not go closer |

Validated against the very signal it replaces: driving into the tea table, our
pose-only number read 0.95631006 against the stack's 0.95631009. In the latest
japanese_room run `/way_point_reached` never fired at all — our predicate got
there first.

---

## 8. What is measured, and what is not

Measured:

- 72.2 % localisation from a single start-pose frame; 11.1 % false positives
  after gates; 0.11 m median error (54 unique scene–phrase pairs).
- The blind cone, on seven scenes, and its prediction that turning fixes it —
  confirmed live twice.
- The converter's behaviour, to ~0.1 m, on three drives.
- Cost: 2 990 in / 460 out tokens = $0.0265 a call.

Not measured, and load-bearing:

- Everything in §1 is **one frame from the start pose**. Distance, occlusion
  and out-of-vocabulary targets are untested.
- The relational branch has one live phrase behind it, not a sweep — and
  relational phrasings are 47–60 % of the official set.
- `target_state` and `same_object_as_previous` are logged and gate nothing.
- Multi-constraint instructions (*go near A, then take the path near B to C*)
  are not implemented; the loop drives to one object and stops.
- Whether "go near X" scores at the ~1 m the platform allows for objects in
  corners is unknown; the README publishes no distance threshold.

---

## Where the code is

| file | role |
|---|---|
| `scripts/vlm_probe.py` | one grounding call; both prompt versions live here |
| `scripts/vlm_locate.py` | box → ray → lidar cone → metres |
| `scripts/vlm_approach.py` | blind cone, size gate, relation resolution, `next_waypoint` |
| `scripts/waypoint_converter_model.py` | what the platform will do with a waypoint |
| `scripts/robot_io.py` | in-container ROS bridge: `capture`, `drive`, `preflight` |
| `scripts/approach_loop.py` | the loop above |
| `scripts/vlm_sweep.py`, `vlm_gates.py`, `vlm_fp_report.py` | the offline measurements |
