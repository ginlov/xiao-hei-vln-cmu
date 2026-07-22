# Perception (sidecar + responder)

A real, model-driven detector → segmenter → 3D-lift perception path feeding the scene graph. Two halves:

- **Sidecar** (`perception/` at the repo root) — FastAPI service wrapping YOLOv8x-World v2 + SAM 2.1 Hiera Tiny. Equirect frames in, equirect masks out.
- **Responder** (`src/xiao_hei_vln/perception/`) — `PerceptionResponder` + supporting modules: HTTP client, 2D→3D lifter, hybrid open-vocab vocabulary. Runs a Phase A (walk coverage trajectory) / Phase B (answer the question) split, writing into the shared `SceneRepresentation` consumed by the report + logger tooling.

## What's inside

| Component | Choice | Why |
|---|---|---|
| Detector | **YOLOv8x-World v2** (Ultralytics) | Open-vocabulary — accepts class names at inference time without retraining. ~70M params (largest YOLO-World variant; matches SysNav's YOLOv8x). Heavier than the `m`/`l` tiers but more accurate, and comfortable at 2 Hz on a mid-range GPU. |
| Segmenter | **SAM 2.1 Hiera Tiny** (Meta) | Converts each detection bbox into a precise pixel mask. Smallest backbone (~38M); we use bbox prompts. |
| Image transport | equirectangular ↔ 4 perspective faces | The camera publishes a 360°×120° equirectangular image; YOLO and SAM are trained on perspective views. We unwrap to four 640×640 faces (100° FOV, ~10° overlap), detect per face, then reproject masks back to equirect coords on the way out. |
| 3D lifting | LiDAR-filtered (responder side, Phase 3) | Each mask is passed through the wire as a bitmap; the responder filters the registered scan by mask and takes the median XYZ. |
| Container | profile-gated sidecar (`docker/compose.yml`) | One env var (`XIAO_HEI_RESPONDER=perception`) brings it up; other responders never instantiate this container. |

## Why a sidecar (not in-process)

The GPU stack (torch+CUDA+ultralytics+sam2) is ~5 GB and conflicts badly with the ROS base image. Keeping it isolated:

- The `ai_module` image stays small (~9 GB ROS+Python, no CUDA stack).
- The sidecar can be swapped (different YOLO variant, different segmenter) without rebuilding the responder.
- A crash inside the model code only restarts the sidecar; the responder keeps running.

## Architecture

```text
┌─ ai_module ───────────────┐   HTTP    ┌─ perception sidecar ─────────┐
│  PerceptionResponder      │ ───────►  │  FastAPI on :8001            │
│                           │           │   ├─ YOLOv8x-World v2  (~146 MB)
│  per tick:                │           │   └─ SAM 2.1 Hiera Tiny (~155 MB)
│   - send {image_jpg,      │           │                              │
│     classes:[...]}        │ ◄───────  │  returns:                    │
│   - LiDAR-filter each     │           │   [{label, score,            │
│     mask → 3D position    │           │     bbox_xyxy, mask_rle}, …] │
│   - scene.add_object(...) │           │                              │
└───────────────────────────┘           └──────────────────────────────┘
```

### Per-request pipeline

```text
equirect 1920×640 BGR8
        │
        ▼ unwrap (cv2.remap × 4 LUTs, BORDER_WRAP)
    ┌───┴───┐  ┌───────┐  ┌──────┐  ┌──────┐
    │ front │  │ right │  │ back │  │ left │      each 640×640, 100° FOV
    └───┬───┘  └───┬───┘  └───┬──┘  └───┬──┘
        └─────┬────┴──────────┴─────────┘
              ▼
        YOLO-World batched (one forward pass for all 4 faces)
              │  ~5–15 boxes total
              ▼  per detection
        SAM 2.1 (bbox prompt → mask in face coords)
              │
              ▼  face_mask_to_equirect_mask(inverse LUT)
        masks back in equirectangular pixel coords
              │
              ▼  pycocotools.mask.encode → base64 RLE
        wire-ready
```

No cross-face NMS in v1 — the 10° face-overlap may double-detect a seam-spanning object, but the responder-side `SceneRepresentation.merge_radius` collapses 3D duplicates anyway. Revisit if quality data shows it matters.

### Camera model

The CMU sim's `/camera/image` topic is a 360°×120° equirectangular panorama at 1920×640, BGR8. We use the standard equirect projection (no intrinsic matrix needed — just two FOV constants):

```text
forward projection (camera-frame direction → equirect pixel)

    λ = atan2(d_x, d_z)                          # longitude
    φ = atan2(-d_y, sqrt(d_x² + d_z²))           # latitude
    u = (λ / 2π + 0.5) * 1920
    v = (0.5 - φ / (2π/3)) * 640
```

The vertical crop to ±60° means we never hit the pole singularity that breaks naive equirectangular detection at the top/bottom. All projection math lives in `perception/geometry.py` (pure numpy, 33 unit tests).

## HTTP API

| Route | Description |
|---|---|
| `GET /healthz` | `{ model_loaded, gpu_available, schema_version, notes }` — readiness probe. `model_loaded` is `false` until lifespan startup finishes (~3–10 s on a 4090). |
| `POST /reload_classes` | JSON `{ classes: [str, ...] }`. Caches the class list AND refreshes YOLO-World's prompt embeddings (~50 ms for 50 classes). Idempotent on the same list. |
| `POST /detect` | `multipart/form-data`: `image` (file), `classes` (optional CSV), `score_threshold` (float, default 0.25), `iou_threshold` (float, default 0.5). Returns `{ detections, inference_ms, schema_version }`. |

### Detection wire format

```json
{
  "detections": [
    {
      "label": "sofa",
      "score": 0.92,
      "bbox_xyxy": [816, 223, 1051, 374],
      "mask_rle": "<base64 of COCO-style RLE>"
    }
  ],
  "inference_ms": 224,
  "schema_version": "2.0.0"
}
```

- `bbox_xyxy` is in **equirectangular pixel coordinates** (the 1920×640 frame), not per-face.
- `mask_rle` is a base64-encoded COCO RLE of a `(640, 1920)` boolean mask, also in equirect coordinates. Decode with `pycocotools.mask.decode` after b64-decoding the `counts` field. Helper: `perception.pipeline.decode_mask_rle(rle_str, height=640, width=1920)`.

## Running the sidecar

The wrapper handles compose-profile activation; one env var picks the responder:

```bash
# Standalone (no responder yet — sidecar serves curl directly)
XIAO_HEI_RESPONDER=perception docker/run up -d --build perception

# Wait for the model load
docker logs -f xiao_hei_perception            # look for "pipeline ready"
curl -s http://localhost:8001/healthz         # → "model_loaded":true

# Stand it up alongside the rest of the stack
XIAO_HEI_RESPONDER=perception docker/run up -d
```

### Smoke test from the host

```bash
# 1. Make a 1920×640 test frame
python3 -c "from PIL import Image; Image.new('RGB', (1920, 640), (128, 128, 128)).save('/tmp/eq.jpg', 'JPEG')"

# 2. Push a class list
curl -sX POST http://localhost:8001/reload_classes \
  -H 'Content-Type: application/json' \
  -d '{"classes": ["chair", "table", "door", "sofa", "lamp"]}'

# 3. Detect
curl -sX POST http://localhost:8001/detect \
  -F image=@/tmp/eq.jpg \
  -F score_threshold=0.25
```

A grey frame returns `{"detections": []}`. A real scene render (e.g.
`CMU-VLN-Challenge-data/unity_env_models/arabic_room/render.jpg`,
padded to 1920×640) typically returns 5–20 detections.

## Debug mode — inspect YOLO + SAM per step

Set `PERCEPTION_DEBUG=1` and every `/detect` call dumps the intermediate
visualisations to the bind-mounted `perception/debug/` directory on the host — one
folder per call:

```text
perception/debug/detect_000001/
  00_equirect.png          # original equirect input
  01_face{0..3}.png        # the 4 perspective faces after unwrap
  02_face{0..3}_bboxes.png # YOLO-World boxes (green) drawn on each face
  03_face{0..3}_masks.png  # SAM masks (red overlay) on each face
  04_equirect_overlay.png  # masks + boxes reprojected onto the equirect frame
```

This walks the exact pipeline order — equirect → 4 faces → bboxes → segmentation
→ reprojected equirect — so you can see what YOLO detected per face and how well
SAM segmented each box.

```bash
# Enable and restart the sidecar (pipeline.py is bind-mounted, no rebuild).
PERCEPTION_DEBUG=1 XIAO_HEI_RESPONDER=perception \
  docker/run up -d --no-deps --force-recreate perception

# Each subsequent /detect (live ticks or a manual curl) writes a detect_NNNNNN/ folder.
ls perception/debug/
```

Notes:

- A single toggle — `PERCEPTION_DEBUG` (truthy: `1`/`true`/`yes`/`on`). Output path
  is fixed at `/opt/perception/debug` (override with `PERCEPTION_DEBUG_DIR` if needed).
- One folder **per `/detect`**, so a live run at 2 Hz produces many folders quickly;
  turn it off (`PERCEPTION_DEBUG=` + restart) when done.
- Rendering is wrapped in a try/except — debug output never breaks `/detect`.
- The sidecar writes as root; `sudo chown -R "$(id -u):$(id -g)" perception/debug` to
  take ownership. The folder is git-ignored.

## Performance

Measured on `arabic_room/render.jpg` padded to 1920×640, RTX 4090, schema_version 2.0.0:

| Phase | Time |
|---|---|
| Cold latency (first /detect after restart) | 660 ms |
| Warm latency (steady-state) | **224 ms** |
| Tick budget at 2 Hz | 500 ms |

The warm-state budget has ~275 ms of headroom for the responder-side LiDAR projection + 3D lift. SAM dominates per-detection cost (~10 ms each, called for every YOLO box); YOLO itself is ~50 ms batched across 4 faces.

## Build details

| Layer | Source | Why |
|---|---|---|
| Base image | `ultralytics/ultralytics:8.4.72` | Ships torch+CUDA+cuDNN+ultralytics+opencv-headless pre-installed. Saved ~3 GB of pip downloads vs starting from `nvidia/cuda`. |
| Pip layer | `sam2`, `fastapi`, `uvicorn`, `python-multipart`, `pycocotools` | Web layer + segmentation model + RLE encoder. |
| Model weights | `yolov8x-worldv2.pt` (146 MB), `sam2.1_hiera_tiny.pt` (155 MB) | Pre-downloaded during build so the first `/detect` doesn't stall on a network fetch. Cached at `/opt/perception/models/`. |
| Application | `server.py`, `pipeline.py`, `geometry.py` | Bind-mounted from the host (`../perception/*.py`) so dev edits take effect on container restart without rebuilding. |

Image size: ~5 GB content / ~14 GB on-disk (the base ships with full CUDA libs).

## Tests

Pure-numpy geometry round-trip (no GPU needed):

```bash
uv run --with pytest --with numpy pytest perception/tests/test_geometry.py -q
# → 33 passed
```

End-to-end smoke (requires the built image + GPU):

```bash
XIAO_HEI_RESPONDER=perception docker/run up -d perception
# wait for healthz, then curl /reload_classes + /detect — see "Smoke test" above
```

## Responder side (Phase 3)

The Python that drives the sidecar lives in `src/xiao_hei_vln/perception/`. It runs a Phase A (walk coverage trajectory) / Phase B (answer the question) split: per tick it calls the sidecar's `/detect`, lifts each mask to a 3D map-frame point through the registered LiDAR scan, and pushes into the shared `SceneRepresentation`. The report + logger tooling consume that scene graph directly.

### Layout

```
src/xiao_hei_vln/perception/
├── __init__.py           # re-exports PerceptionResponder
├── geometry.py           # responder-side equirect forward projection
│                         # + sensor→camera static extrinsic
├── vocab.py              # Vocabulary: scene-prior + question-derived nouns
├── client.py             # HTTPPerceptionClient + COCO RLE decoder
├── lifter.py             # PointLifter: 2D mask + LiDAR → 3D map-frame point
└── responder.py          # PerceptionResponder
```

### Vocabulary — open-vocab class management

YOLO-World accepts a new class list per call but re-encoding the prompt embeddings takes ~50 ms for ~50 classes, so we keep a stable hybrid list:

- A **prior** — the fixed `DEFAULT_PRIOR` tuple in `perception/vocab.py`, pushed every tick so the scene graph accumulates objects between questions.
- **Question-derived nouns** merged on top — the tokenizer drops stopwords, depluralises common forms (`chairs` → `chair`), and dedupes against the prior. YOLO-World ignores junk tokens cheaply (low embeddings), so we don't need a strict NLP parser.

Because the detector can only ever emit labels from this list, the **prior controls how well the output matches a scene's ground truth**. It is currently **pinned to the arabic_room ground-truth labels** (its `object_list.txt`, minus the `unknown` placeholder) so detection precision/recall can be measured against GT without vocabulary mismatch. To target a different scene, edit `DEFAULT_PRIOR`:

- `src/` is bind-mounted read-only into the `ai_module` container and installed editable, so the change takes effect on `docker/run up -d --no-deps --force-recreate ai_module` — no image rebuild needed.
- Caveat: question-derived nouns are still merged on top, so generic words like `room` from *"how many chairs are in the room"* enter the class list and YOLO-World may emit spurious `room` boxes. Filtering non-object nouns is a known follow-up.

```python
vocab = Vocabulary()
vocab.current_classes(None)
# → ('focus light', 'pillow', 'potted plant', 'wall', 'window', …)  # arabic_room GT
vocab.current_classes("Find the lamp near the sushi")
# → (… prior …, 'lamp', 'sushi')
```

`current_classes` returns a stable `tuple[str, ...]` so the client can compare it against the last set and skip the network call when unchanged.

### Lifter — 2D mask + LiDAR → 3D position

```python
result = lifter.lift(
    mask=detection.mask,                       # (640, 1920) bool, equirect
    scan_points_map=snapshot.registered_scan.points,
    pose_position=snapshot.pose.position,
    pose_orientation=snapshot.pose.orientation,
)
if result.position is not None:                # else: not enough LiDAR support
    scene.add_object(ObjectObservation(label=detection.label,
                                       position=result.position, ...))
```

Algorithm:

1. Transform `registered_scan` from the map frame to the sensor frame using the inverse of the robot's current pose (the sensor sits at the vehicle origin per the sim's static transforms).
2. Apply the static sensor → camera extrinsic (`(0, 0, 0.1)` translation + a 90° axis-swap rotation).
3. Project each camera-frame point into equirect pixel coords using the same formula the sidecar uses to *create* the mask.
4. Keep points whose projected pixel falls inside the mask.
5. Return the **median XYZ in the map frame** if `n_inliers >= min_inliers` (default 10), else `None`.

The median is robust to mask-edge noise and to scan returns that snap through a doorway or window. `max_depth_m` is available for the case where a sparse return at a far wall biases the median; default is `None` (no cap).

### Client — HTTP wrapper around the sidecar

```python
client = HTTPPerceptionClient(base_url="http://localhost:8001")
client.wait_until_ready(timeout_s=60)             # blocks until /healthz

# Detect — set_classes is dedup-cached, so passing the same vocab is free.
detections = client.detect(
    image_bgr=snapshot_image_as_bgr,              # (640, 1920, 3) uint8
    classes=vocab.current_classes(question_text),
    score_threshold=0.25,
)
# detections[i] has .label, .score, .bbox_xyxy, .mask (already decoded to ndarray)
```

`mask_rle` is decoded eagerly into a `(EQUIRECT_H, EQUIRECT_W)` bool ndarray so the lifter doesn't need `pycocotools` at runtime. Decoder prefers the `pycocotools` C path when installed (matches the sidecar encoder exactly); falls back to a small pure-Python decoder otherwise — useful for CI without native deps.

Error model is intentionally permissive: any HTTP / network / parse failure logs and returns `[]`. A tick with no detections leaves the scene graph unchanged, same as a real perception system on a glare frame.

### Responder — composes everything

`PerceptionResponder` runs a two-phase loop:

| Phase | Perception |
|---|---|
| A (walk) | Trajectory walker emits coverage waypoints; advances on `/way_point_reached` |
| Per tick | Call `/detect` + lift each mask + `scene.add_object(...)` |
| B (numerical) | Count matching labels from `scene.objects` |
| B (object_reference) | Pick the closest match from `scene.objects` |
| B (instruction_following) | Trajectory itself is the answer |

Per-tick, `VLMLogger.log_tick` records the system prompt (`"PerceptionResponder"`), question text, output, and inference latency.

The numerical handler returns `0` when the question's label isn't in `scene.objects` (we don't have ground truth to fall back on). That's the correct behaviour for a real perception system: "I haven't seen any of those" *is* the answer.

## Running the perception responder

The wrapper handles compose-profile activation and the sidecar HTTP wiring. The ai_module image installs the perception client extras (`httpx`, `pillow`, `pycocotools`) by default — no extra build args needed.

```bash
# Bring up the full stack
SCENES=/path/to/CMU-VLN-Challenge-data/unity_env_models
unzip -oq $SCENES/arabic_room.zip -d $SCENES/

export XIAO_HEI_SCENE_DIR_HOST=$SCENES/arabic_room
export XIAO_HEI_TRAJECTORY_JSON_HOST=$PWD/trajectories/arabic_room.json   # optional
XIAO_HEI_RESPONDER=perception docker/run up -d --build

# Wait for the perception sidecar to be ready, then launch Unity
docker logs -f xiao_hei_perception        # look for "pipeline ready"
docker exec -it iros2026_system \
    /home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh

# Fire a question
docker exec iros2026_system bash -lc \
  'source /opt/ros/jazzy/setup.bash && \
   export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && \
   ros2 topic pub --once /challenge_question std_msgs/msg/String \
     "{data: \"Find the sofa\"}"'
```

### Configuration

| Env var | Default | Description |
|---|---|---|
| `XIAO_HEI_PERCEPTION_BASE_URL` | `http://localhost:8001` | Where the responder reaches the sidecar. |
| `XIAO_HEI_PERCEPTION_SCORE_THRESHOLD` | `0.25` | Forwarded to YOLO-World on every `/detect`. Lower → more detections, more noise. |
| `XIAO_HEI_PERCEPTION_MIN_INLIERS` | `10` | LiDAR return count below which a mask is dropped (no 3D point committed). |
| `XIAO_HEI_TRAJECTORY_JSON` (via `_HOST` bind mount) | unset | Optional pre-planned coverage trajectory; the responder walks it during Phase A. |

## Tests

```bash
uv sync --extra perception
uv run pytest tests/test_perception_*.py -q
# → 32 passed
```

- `test_perception_vocab.py` — prior return, question-derived merging, plural dedup, stopword filtering, stable order, mixed case + punctuation handling. (10 tests)
- `test_perception_lifter.py` — empty scan / mask, min_inliers floor, median XYZ recovery, mask-isolated cluster selection, rotated-pose projection shift, translated-pose map-frame correctness, input validation. (9 tests)
- `test_perception_responder.py` — `_inject_visible` skip without inputs, detections lifted + added, dropped when lifter returns `None`, set_classes pushed from current vocab, Phase A waypoint advance on reach, Phase B numerical / object_reference paths, lifecycle (close / reset). (13 tests, all with a mocked `_FakeClient` + `_FakeLifter`).

Plus 33 pre-existing tests for the sidecar-side geometry (`perception/tests/test_geometry.py`). Both halves test their own geometry constants independently — if they ever drift, both suites fail.

## Phase 4 (next)

- Run perception against `arabic_room`, `livingroom_2`, `studio` and compare scene graphs vs each scene's ground-truth `object_list.txt`.
- Tune `score_threshold`, `min_inliers`, `merge_radius`.
- Optionally add cross-face NMS in the sidecar if seam-spanning duplicates hurt quality.
- Add a "Perception" tab to the HTML report (overlay detections + masks on the camera image).

See [TASK 11 - Perception responder](tasks/TASK%2011%20-%20Perception%20responder.md) for the full design doc + open questions.
