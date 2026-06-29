# TASK 11 — Perception Responder (YOLOv8x-World v2 + SAM 2.1 Hiera Tiny)

## Purpose

Replace the **OracleResponder**'s ground-truth perception (reading `object_list.txt`) with a real, model-driven detector→segmenter→3D-lift pipeline. The output still feeds `SceneRepresentation.add_object(...)` exactly as the oracle does today, so the scene graph, the reporting tooling, and the question-answering logic all stay put. This is the bridge from the validation harness to a submission-grade perception stack.

Concretely:

- **Detection**: YOLOv8x-World v2 (open-vocabulary YOLO from Ultralytics) — accepts class names at inference time without retraining; ~70M params (largest YOLO-World variant, matching SysNav's YOLOv8x); comfortable at 2 Hz on a mid-range GPU. (Originally scoped as the `m` tier; upgraded to `x` for accuracy.)
- **Segmentation**: SAM 2.1 Hiera Tiny (Meta) — ~38M params; converts each YOLO bbox into a precise mask, which we'll use to filter the LiDAR projection in 3D-lift.
- **3D lifting**: project the registered LiDAR scan into image pixels using the camera+sensor extrinsic, keep returns that land inside each SAM mask, take the median XYZ as the object's position. No neural depth estimator needed.
- **Vocabulary**: hybrid — a small fixed "common indoor" prior plus noun phrases extracted from the current question, refreshed each new question.

## Why this design (chosen, with tradeoffs)

| Question | Choice | Why |
|---|---|---|
| **Where do the models run?** | **HTTP sidecar** container (mirrors the qwen/vllm pattern) | Keeps `ai_module` image lean (no CUDA conflicts with the ROS base); lets us swap or upgrade models without rebuilding the responder. Same compose-profile gating story (`profiles: [perception]`). |
| **What's the open-vocab class list?** | **Hybrid: scene-prior + question-derived** | YOLO-World accepts a changing class list per call. A small prior (chair, table, door, lamp, …) means the scene graph still grows between questions; question-derived merging means we don't miss task-specific labels. |
| **How are 2D masks lifted to 3D?** | **LiDAR points projected into image, filtered by mask, median XYZ** | Robust to mask-edge noise; no neural-depth dependency; uses the data we already have (`registered_scan`). |
| **Container deployment cost** | One extra container only when `XIAO_HEI_RESPONDER=perception` (or its docker-compose profile) is active | Dummy/oracle/qwen runs never touch this container. |

## Architecture

```
┌──────────────────────────┐         ┌─────────────────────────────┐
│  ai_module               │  HTTP   │  perception (sidecar)       │
│  (PerceptionResponder)   │ ──────► │  FastAPI on :8001           │
│                          │         │   ├─ YOLOv8x-World v2 .pt   │
│  every tick:             │         │   └─ SAM 2.1 Hiera Tiny .pt │
│   - send {image_jpg,     │         │                              │
│     classes:[...]}       │ ◄────── │  returns:                    │
│   - send {tick_id,       │         │   [{class, bbox, mask_rle,   │
│     pose,                │         │     conf}, ...]              │
│     registered_scan}     │         │                              │
│   - LiDAR project + lift │         │                              │
│     to 3D                │         │                              │
│   - scene.add_object(...)│         │                              │
└──────────────────────────┘         └─────────────────────────────┘
```

Why HTTP and not shared memory: HTTP keeps the boundary clean across container restarts, lets us run sidecar+responder on different hosts later, and matches the qwen path. The image payload (~600 KB JPEG @ 1920x640) over loopback is negligible at 2 Hz.

## Sub-systems

### 1. Perception sidecar

**Location:** `perception/` (new top-level dir, mirrors how the vllm sidecar lives outside `src/`).

**Files:**

```
perception/
├── Dockerfile                  # FROM nvidia/cuda:…-runtime, installs ultralytics + sam2
├── server.py                   # FastAPI app: /detect, /healthz, /reload_classes
├── pipeline.py                 # YOLO-World → SAM2 → response builder
├── requirements.txt            # ultralytics>=8.3, sam2 (from Meta repo), fastapi, uvicorn
└── README.md                   # build/run/test
```

**API (FastAPI):**

```python
POST /detect
  body: multipart {
    image: file (JPEG),
    classes: list[str],        # open-vocab labels
    score_threshold: float = 0.25,
    iou_threshold: float = 0.5,
  }
  response: {
    detections: [
      {
        label: str,
        score: float,
        bbox_xyxy: [x1, y1, x2, y2],
        mask_rle: str,          # base64-encoded run-length, decoded shape (H, W)
      },
      ...
    ],
    inference_ms: float,
  }

POST /reload_classes          # cache YOLO-World prompt embeddings between calls
  body: {classes: list[str]}
  response: {ok: true}

GET /healthz
  response: {model_loaded: bool, gpu_available: bool}
```

**Why mask_rle (not raw bytes):** the 1920x640 mask is ~1.2 MB raw; RLE compresses 100x+ for typical sparse masks. We decode with `pycocotools` on the responder side.

**Compose:** new `perception` service under `profiles: [perception]`, same GPU reservation pattern as `vllm`. The wrapper maps `XIAO_HEI_RESPONDER=perception` → `--profile perception`.

### 2. PerceptionResponder

**Location:** `src/xiao_hei_vln/perception/` (new package).

**Files:**

```
src/xiao_hei_vln/perception/
├── __init__.py
├── responder.py                # PerceptionResponder (mirrors OracleResponder's shape)
├── client.py                   # HTTPPerceptionClient: posts to the sidecar
├── lifter.py                   # 2D mask + LiDAR → 3D position
├── vocab.py                    # class-list management (prior + question-derived)
└── geometry.py                 # camera/sensor extrinsic, project_points helpers
```

**Shape (mirroring OracleResponder):**

```python
class PerceptionResponder:
    def __init__(
        self,
        scene: SceneRepresentation,
        *,
        client: PerceptionClient,
        vocabulary: Vocabulary,
        lifter: PointLifter,
        observation_radius: float = 8.0,
        near_threshold: float = 2.0,
        trajectory_path: Path | None = None,
        logger: VLMLogger | None = None,
        score_threshold: float = 0.25,
    ): ...

    def respond(self, snapshot: VLMInput) -> VLMOutput | None:
        self._inject_visible(snapshot)        # ← replaces the oracle's ground-truth read
        # Same Phase A (walk trajectory) / Phase B (answer) as OracleResponder.
```

`_inject_visible(snapshot)` is the only line whose body changes vs OracleResponder:

```python
def _inject_visible(self, snapshot: VLMInput) -> None:
    if snapshot.image is None or snapshot.registered_scan is None:
        return
    classes = self._vocabulary.current_classes(snapshot.question)
    detections = self._client.detect(snapshot.image, classes,
                                     score_threshold=self._score_threshold)
    for det in detections:
        position, n_inliers = self._lifter.lift(
            mask=det.mask, scan=snapshot.registered_scan, pose=snapshot.pose,
        )
        if position is None:
            continue                          # mask had no LiDAR support
        self._scene.add_object(ObjectObservation(
            label=det.label,
            position=position,
            confidence=det.score,
            bbox_min=..., bbox_max=...,       # optional, computed from mask 3D extent
        ))
```

`scene.add_object` already merges duplicates inside `merge_radius` and stamps the stable `object_id` we just added in the prior refactor.

### 3. 2D→3D lifter

**Algorithm:**

1. Snapshot has `registered_scan.points` (N, 4) in the **map** frame (because it's `registered_scan`, not `sensor_scan`).
2. Transform map → sensor frame using `snapshot.pose`. The sensor is at the vehicle origin (confirmed from the static_transform_publisher log).
3. Apply sensor → camera extrinsic (0.1 m up, 90° rotation; see the *Camera model* section).
4. Project camera-frame points to image pixels using the equirectangular formula above. Drop points with `|φ| > π/3` (outside vertical FOV) and points with `z_c ≤ 0` after we've handled the wrap (defensive — equirect doesn't strictly require this, but it culls behind-camera returns when something is exactly at the boundary).
5. For each detection mask: find scan returns whose projected pixel lies inside the mask. If `n_inliers >= min_lift_support` (default 10), take the median XYZ in the map frame.

**Edge cases:**

- **Mask with no LiDAR support** (e.g. transparent objects, deep shadows): skip the detection. Logged for offline inspection.
- **Mask spanning multiple objects in depth**: the median is reasonably robust to bimodal returns; if both modes are within `merge_radius` of an existing scene object, `scene.add_object` collapses them naturally.
- **Sparse LiDAR coverage on small objects**: tunable `min_lift_support` knob. Lower it for small-label scenarios.

### 4. Vocabulary

**Default prior** — the fixed `DEFAULT_PRIOR` tuple in `perception/vocab.py`. Originally ~50 generic indoor labels (chair, table, door, lamp, sofa, …). It is **currently pinned to the arabic_room ground-truth labels** (from its `object_list.txt`, minus the `unknown` placeholder) so detector precision/recall can be measured against GT without vocabulary mismatch. Since the detector can only emit labels from this list, the prior controls how well output matches a scene's GT; swap it (edit `DEFAULT_PRIOR`; `src/` is bind-mounted editable, so restart `ai_module` — no rebuild) to target a different scene.

**Question-derived noun extraction:**

- Stage 1 (cheap): regex / spaCy `en_core_web_sm` noun-chunking on `snapshot.question.text`.
- Stage 2 (later, if quality lags): an LLM-prompted extractor that handles attribute-rich queries like "the red cup near the sushi".

**API:**

```python
class Vocabulary:
    def __init__(self, prior: list[str], extractor: NounExtractor): ...
    def current_classes(self, question: ChallengeQuestion | None) -> list[str]:
        # Returns prior + any question-derived nouns, deduped.
```

Vocabulary changes invalidate the sidecar's cached YOLO-World text embeddings; the responder calls `/reload_classes` only when the set changes (cheap dedup on the responder side).

## Camera model (confirmed)

The `/camera/image` topic publishes a **360° equirectangular image cropped to ±60° vertical** (so vertical FOV ≈ 120°), at 1920×640 BGR8. This is consistent with a Unity 360° camera rig that crops the poles. Concretely:

- horizontal: 1920 px / 360° → 5.33 px/° → 0.1875°/px
- vertical:   640 px / 120° →  5.33 px/° (same density, no aspect distortion in pixel/° units)

**Forward projection (camera-frame point → pixel)** for a 3D point `(x_c, y_c, z_c)` in the camera frame (z forward, y down, x right):

```
λ = atan2(x_c, z_c)                               # longitude, (-π, π]
φ = atan2(-y_c, sqrt(x_c² + z_c²))                # latitude,  (-π/2, π/2)

if |φ| > π/3:                                     # outside vertical FOV → invisible
    skip
u = (λ / (2π) + 0.5) * 1920
v = (0.5 - φ / (2π/3)) * 640
```

No intrinsic matrix needed — just two constants: `H_FOV = 2π` and `V_FOV = 2π/3`. The projection is exact and wrap-aware. The vertical crop to ±60° means we never hit the pole singularity that plagues full-equirect detection.

### Detection strategy: 4-face perspective unwrap

YOLOv8x-World v2 and SAM 2.1 are trained on perspective imagery and degrade on equirectangular input (≈10–15% AP loss + horizontal-seam objects split in two). Rather than absorb that, the sidecar unwraps the equirect frame into **four 90°×90° perspective faces** before detection:

```
face_idx ∈ {front, right, back, left}
yaw_center[face_idx] ∈ {0°, 90°, 180°, 270°}
pitch_center = 0°
each face: 640×640 px, ~90° FOV (with 10° horizontal overlap between adjacent faces)
```

**Pipeline per tick:**

1. Server receives the equirect JPEG.
2. Unwrap into 4 perspective faces (precomputed pixel-remap lookup tables).
3. YOLO-World on each face independently (batchable on GPU → 1× inference cost in wall time).
4. NMS in equirectangular angle space across faces: map each bbox back to `(λ, φ, Δλ, Δφ)`, then dedup by angular IoU. Resolves the 10°-overlap region.
5. For each surviving bbox: run SAM 2.1 on the face it came from, then re-project the mask back to equirectangular pixel coordinates (the same lookup table, inverted).
6. Return masks in equirectangular space → the responder-side lifter consumes them with the equirectangular projection above. The lifter never needs to know about faces.

**Why mask re-projection back to equirect (vs forwarding the per-face mask):** the lifter projects LiDAR points using the equirect formula above; if we forwarded per-face masks, the lifter would also need per-face extrinsics and re-project once per face — coupling boundaries with no benefit. Doing the conversion server-side keeps the wire format flat: one mask per detected object, in equirect coordinates.

### Sensor → camera extrinsic

From the sim's static_transform_publisher: `translation (0, 0, 0.1)`, `rotation (-0.5, 0.5, -0.5, 0.5)`. The quaternion is a 90° rotation that swaps the axes from a robotics convention (x forward) to a camera convention (z forward, x right, y down). Encoded as a 4×4 transform in `geometry.py`. Validation in Phase 2: project a known LiDAR return and verify it lands at the expected pixel.

## Remaining open questions

These don't block the plan; they get resolved in code during the build.

1. **SAM 2.1 prompt mode**: bbox prompts (what we'd use) work directly. Point prompts could give slightly better quality but cost an extra round-trip per detection — defer to Phase 4 tuning.
2. **Tick budget**: target ≤ 300 ms per detect call on the sidecar (4 faces × YOLO + N × SAM). If YOLO-World-m + SAM 2.1 Hiera Tiny exceed this on the target GPU, fall back to YOLO-World-s + run SAM only on top-K detections by score.

## Test plan

### Unit tests (no GPU required — mocked client)

- `tests/test_perception_lifter.py`
  - Returns `None` when no LiDAR points project inside the mask.
  - Returns the median when ≥ `min_lift_support` inliers are present.
  - Correctly transforms map-frame scan to camera-frame using a synthetic pose.
  - Robust to scan with zero rows.
- `tests/test_perception_vocab.py`
  - Prior alone when question is `None`.
  - Adds question nouns, dedupes against the prior.
  - Stable order across calls (so the sidecar can cache embeddings on the unchanged subset).
- `tests/test_perception_responder.py`
  - Same Phase A trajectory-walking behaviour as OracleResponder (regression).
  - Calls the (mocked) client only when `snapshot.image` and `snapshot.registered_scan` are both present.
  - Wires the responder's logger exactly like the oracle does.

### Sidecar tests (GPU required, gated behind an extra)

- `perception/tests/test_pipeline_smoke.py`: load both models, run a single inference on a fixture frame, assert output schema.

### End-to-end smoke (manual)

- `XIAO_HEI_RESPONDER=perception docker/run up -d` against `arabic_room` with a known question; verify a non-empty scene graph in the report.html.

## Phased build order

1. **Phase 1 — sidecar skeleton** (one to two sessions):
   - `perception/Dockerfile` + `requirements.txt` + `server.py` returning a stub `[]`.
   - Compose profile + wrapper integration (`XIAO_HEI_RESPONDER=perception` → `--profile perception`).
   - Healthz, model load on startup, no detection logic yet.
2. **Phase 2 — perception pipeline**:
   - Implement YOLO-World + SAM 2.1 on the server side.
   - Confirm the panoramic projection model (the open question above).
   - Verify with a synthetic fixture image.
3. **Phase 3 — responder + lifter**:
   - `PerceptionResponder`, `HTTPPerceptionClient`, `PointLifter`, `Vocabulary`.
   - Unit tests with mocked client.
   - Wire into `app/main.py` `_build_responder`.
4. **Phase 4 — end-to-end + tuning**:
   - Run against `arabic_room` + `livingroom_2` + `studio` (three dev scenes).
   - Compare scene graphs vs the oracle's ground-truth graphs from the same questions.
   - Tune `score_threshold`, `min_lift_support`, `merge_radius`.
5. **Phase 5 — docs + report integration**:
   - Update `docs/scene-representation.md` to mark the SysNav-style detector path as live.
   - Add a "Perception" tab to the HTML report (per-tick detection overlay on the camera image).

## File layout (target end state)

```
perception/                                  # NEW — sidecar
├── Dockerfile
├── requirements.txt
├── server.py
├── pipeline.py
└── README.md

src/xiao_hei_vln/
├── perception/                              # NEW — responder + client + lifter
│   ├── __init__.py
│   ├── responder.py
│   ├── client.py
│   ├── lifter.py
│   ├── vocab.py
│   └── geometry.py
└── app/main.py                              # MODIFIED — adds "perception" branch

docker/
├── compose.yml                              # MODIFIED — adds `perception` service
└── run                                      # MODIFIED — maps responder → --profile

tests/
├── test_perception_lifter.py                # NEW
├── test_perception_vocab.py                 # NEW
└── test_perception_responder.py             # NEW

docs/
└── perception-responder.md                  # NEW — design + run instructions
```

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Sidecar startup time (model weights ~500 MB total) | Bake weights into the image build (download once); responder has retry-with-backoff on the first few ticks. |
| LiDAR/camera time-sync drift in the sim | Use the snapshot's `tick_time` as the join key; the existing `LatestCache` already enforces tick-aligned reads. |
| YOLO-World class-list change cost | Cache embeddings in the sidecar keyed by the sorted class tuple; `/reload_classes` is only called when the set changes. |
| Open-vocab quality on weird labels ("ottoman", "tatami") | Vocabulary prior keeps the common case strong; weak detections at low score are filtered by `score_threshold` and won't pollute the scene graph. |
| Image projection model wrong | Phase 2's synthetic-fixture step catches this before responder integration. |

## What stays unchanged

- `SceneRepresentation` and its API (we just polished it — perfect fit).
- The trajectory walker (`PerceptionResponder` reuses the Phase A/B split — same `_load_waypoints`, `_compute_output` skeleton).

> **Update:** the `OracleResponder` referenced throughout this spec was a
> ground-truth scaffold used to validate the pipeline; it has since been
> **removed** now that the perception path is the production responder. The
> design notes below describe the original migration from it.
- `VLMLogger` integration (perception responder gets the same logger wiring).
- The HTML report (the Scene Representation pane already shows what the perception responder will produce; Phase 5 adds an overlay tab).
