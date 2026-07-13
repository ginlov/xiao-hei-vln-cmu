# Perception sidecar

HTTP service that wraps the open-vocabulary perception pipeline
(YOLOv8x-World v2 + SAM 2.1 Hiera Tiny) used by
`xiao_hei_vln.perception.PerceptionResponder`. Same sidecar pattern as
the `vllm` service — keeps the heavy CUDA deps off the ai_module image
and lets us swap models without rebuilding the responder.

This is **Phase 2**: real models. Equirect frames come in; equirect
masks come out. The 4-face perspective unwrap, batched YOLO-World
detection, SAM 2.1 segmentation, and mask reprojection happen
inside the container; callers never see per-face geometry.

## Endpoints

| Route | Description |
|---|---|
| `GET /healthz` | `{ model_loaded, gpu_available, schema_version, notes }`. |
| `POST /reload_classes` | JSON `{ classes: [str, ...] }` → cached class list for subsequent `/detect` calls. Phase 2 also re-computes YOLO-World prompt embeddings here. |
| `POST /detect` | Multipart upload (`image` file + form fields). Returns `{ detections: [{label, score, bbox_xyxy, mask_rle}, ...], inference_ms, schema_version }`. |

`mask_rle` is COCO-style RLE (base64), in equirectangular pixel
coordinates. The 4-face perspective unwrap happens inside the sidecar;
callers never see per-face geometry.

## Build & run standalone

```bash
docker build -t xiao-hei/perception:latest .
docker run --rm --network host xiao-hei/perception:latest
# Then:
curl -s http://localhost:8001/healthz
```

## Run as part of the stack

The sidecar is profile-gated. Use the wrapper:

```bash
XIAO_HEI_RESPONDER=perception docker/run up -d --build
docker logs -f xiao_hei_perception      # tail
curl -s http://localhost:8001/healthz   # smoke
```

The wrapper maps `XIAO_HEI_RESPONDER=perception` to
`docker compose --profile perception`, which is what starts the
sidecar. Other responders (`dummy`, `qwen`) leave the
sidecar dormant.

## Smoke test the round trip

```bash
# 1. up the stack
XIAO_HEI_RESPONDER=perception docker/run up -d

# 2. wait for healthz
until curl -sf http://localhost:8001/healthz; do sleep 1; done

# 3. tell it which classes to look for
curl -sX POST http://localhost:8001/reload_classes \
  -H 'Content-Type: application/json' \
  -d '{"classes": ["chair", "table", "door"]}'

# 4. send a frame (any JPEG will do for the stub)
curl -sX POST http://localhost:8001/detect \
  -F image=@/path/to/any.jpg \
  -F classes=chair,table \
  -F score_threshold=0.25
# → {"detections": [], "inference_ms": 0.42, "schema_version": "1.0.0-skeleton"}
```

## Pipeline internals

```
equirect 1920×640 BGR8
        │
        ▼ unwrap (cv2.remap × 4 LUTs, BORDER_WRAP)
    ┌───┴───┐   ┌───────┐   ┌──────┐   ┌──────┐
    │ front │   │ right │   │ back │   │ left │     each 640×640, 100° FOV
    └───┬───┘   └───┬───┘   └───┬──┘   └───┬──┘
        └─────┬─────┴───────────┴──────────┘
              ▼
        YOLO-World (batched 4 faces, single forward pass)
              │
              ▼  per detection
        SAM 2.1 (bbox prompt → mask in face coords)
              │
              ▼  face_mask_to_equirect_mask(inverse LUT)
        masks back in equirectangular pixel coords
              │
              ▼  pycocotools.mask.encode → base64 RLE
        wire-ready
```

- **Geometry** is in `geometry.py` (pure numpy, tested separately).
- **No cross-face NMS in v1.** The 10° overlap region may double-detect;
  the responder's `SceneRepresentation.merge_radius` collapses
  duplicates in 3D, so the wire isn't noisy by the time it reaches
  scene state.
- **Camera model**: equirectangular 360° horizontal × 120° vertical
  (cropped to ±60°), confirmed against the CMU sim's `/camera/image`
  output. Pixel ↔ angle formulas in `geometry.py`.

## Testing

Pure-numpy geometry round-trips run without the docker image:

```bash
uv run --with pytest --with numpy pytest perception/tests/test_geometry.py -q
# → 33 passed
```

End-to-end model smoke (requires the built image and a GPU):

```bash
XIAO_HEI_RESPONDER=perception docker/run up -d --build
docker logs -f xiao_hei_perception     # wait for "pipeline ready"

# Synthetic image — any 1920×640 JPEG works
python3 -c "
from PIL import Image
Image.new('RGB', (1920, 640), (128, 128, 128)).save('/tmp/empty.jpg', 'JPEG')
"
curl -sX POST http://localhost:8001/reload_classes \
  -H 'Content-Type: application/json' \
  -d '{"classes": ["chair", "table", "door", "potted plant"]}'

curl -sX POST http://localhost:8001/detect \
  -F image=@/tmp/empty.jpg \
  -F score_threshold=0.25
```
