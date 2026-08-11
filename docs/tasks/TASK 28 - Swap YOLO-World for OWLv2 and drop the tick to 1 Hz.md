# TASK 28 — Swap YOLO-World for OWLv2 (large) and drop the tick to 1 Hz

> **Outcome: OWLv2 reverted; back to YOLO-World.** The on-GPU A/B on
> `arabic_room` (397-frame `captures_nav`, same frozen captures, GT = 69
> objects) did not justify the swap:
>
> | detector @ thresh | mAP@1m | P | R | F1@1m | cErr (m) | pred count |
> |---|---|---|---|---|---|---|
> | **YOLO-World @ 0.25** | 0.397 | **0.443** | 0.449 | **0.446** | **0.160** | **70** ≈ GT |
> | OWLv2-large @ 0.1 | **0.649** | 0.191 | **0.768** | 0.306 | 0.198 | 277 (4× over) |
>
> OWLv2 wins ranking (mAP) and recall but **massively over-produces** at the
> 0.1 floor — precision 0.19, F1 and counting both worse, and the predicted
> node count (277) is ~4× the ground truth (69), which is what made the maps
> "look worse". On top of that OWLv2-large ran **~4.6–6 s/frame** on the A10G,
> which does not fit even the relaxed 1 Hz tick. The high mAP says the ranking
> is good, so threshold tuning *might* have recovered precision — but not the
> latency — so we reverted the detector and kept the two orthogonal wins.
>
> **Kept from this task (not reverted):** the **1 Hz tick** (a good change on
> its own), the **SAM 2.1 Hiera Large** upgrade (detector-agnostic, closes the
> old B5 SAM-Tiny gap), the `sam_score` (SAM predicted-IoU) plumbing, and the
> `viz_app.py` / `sweep_gates.sh` / `dump_debug.py` benchmark tooling.
> **Reverted:** the box detector (OWLv2 → YOLO-World), `transformers` →
> `ultralytics`, the Dockerfile's OWLv2 bake, and the score-threshold floor
> (0.1 → 0.25) across the request path and benchmark scripts.

## Why

Two coupled changes to raise perception quality:

1. **Detector.** YOLO-World was chosen for speed, which stops mattering once
   Task 1/2 answer within a 10-minute budget rather than in real time. Swapping
   it for a heavier open-vocab detector trades latency (which we have) for
   recall on the challenge's object vocabulary (which we need). We keep the
   box→SAM architecture and only replace the box stage — this is the
   Grounded-SAM-2 pattern.
2. **Tick rate.** OWLv2 + SAM-Large no longer fits the old 500 ms / 2 Hz tick,
   so the default drops to **1 Hz** (1000 ms). The lower rate is what *buys*
   the latency budget for the heavier detector; the two changes reinforce.

## What changed

### Detector: OWLv2 large (`google/owlv2-large-patch14-ensemble`)

- `perception/pipeline.py`
  - Load `Owlv2Processor` + `Owlv2ForObjectDetection` in place of
    `ultralytics.YOLOWorld` (lazy import inside `__init__`, as before).
  - `set_classes` now just caches the class list — OWLv2 has no persistent
    prompt to pre-encode; queries are tokenised + encoded jointly with the
    image on every `detect` forward (cheap for a handful of classes at 1 Hz).
  - `detect` runs OWLv2 on the 4-face batch, post-processes with
    `post_process_object_detection` (boxes come back in each face's own pixel
    coords), then applies **per-face NMS** (`torchvision.ops.nms`) — OWLv2 has
    no built-in NMS, unlike YOLO-World's `iou=` arg. Boxes are clamped to
    `FACE_SIZE`. The SAM-per-box path and seam merge are unchanged.
  - Faces are square (`FACE_SIZE`), so OWLv2's square-pad is a no-op and the
    known non-square box-offset caveat does not apply.
- The box→SAM interface, `sam_score` (B5), range cap (B4), and seam merge are
  all detector-agnostic and unchanged.

### Score threshold rescaled to OWLv2's range

OWLv2's query-match scores run lower than YOLO-World's box confidence, so the
detection floor moved **0.25 → 0.1** consistently across the request path:
`responder.DEFAULT_SCORE_THRESHOLD`, `server.py` form default, `client.detect`,
`pipeline.detect`, and the two benchmark scripts
(`replay_score.py`, `dump_detections.py`). This is a starting point — tune on
the frozen captures with `replay_score.py --score-threshold`.

### Sidecar image

- `perception/requirements.txt`: `ultralytics` → `transformers==4.46.3`.
- `perception/Dockerfile`:
  - pip install `transformers`; dropped the OpenAI-CLIP install (it was
    YOLO-World's text encoder) and the `COPY ViT-B-32.pt` step.
  - Bake **SAM 2.1 Hiera Large** (was Tiny — this also closes the B5 gap where
    the code referenced `sam2.1_hiera_large.pt` but the image only fetched
    Tiny) and OWLv2 weights into the HF cache.
  - `HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1` set *after* the bake, so a live
    tick never blocks on the network during the challenge.

### Tick rate: 2 Hz → 1 Hz

- `src/xiao_hei_vln/app/main.py`: `XIAO_HEI_VLM_TICK_HZ` default `2.0` → `1.0`
  (timer period is `1.0 / TICK_HZ`). Override still honoured.
- `perception_benchmark/record_navigation.py` + `run_nav_capture.sh`: mirror
  the default so recorded captures match the live cadence.
- Docs: `README.md`, `docs/getting-started/configuration.md`,
  `docs/concepts/tick-loop.md` updated to the 1 Hz default (incl. the
  "Why 1 Hz?" budget table).

## Status

**Measured on-GPU, then reverted — see the Outcome banner at the top.** OWLv2
was swapped in, the image rebuilt, and the `arabic_room` A/B run: OWLv2's 4×
over-production at the 0.1 floor (277 vs 69 GT) and ~5 s/frame latency lost to
YOLO-World on precision, F1, counting, and tick fit despite a higher mAP. The
detector was reverted to YOLO-World; the 1 Hz tick and SAM-Large upgrade were
kept. Sidecar image rebuilt on the YOLO+SAM-Large definition; all unit tests
pass (perception responder/client/sync + sidecar geometry/seam-merge:
76 + 51 = 127).

## Next

1. If OWLv2 is revisited, first **tune the score threshold** (the 0.649 mAP
   says ranking is good) and **quantise / use a smaller OWLv2 variant** to get
   under the tick budget before re-running the A/B.
2. With YOLO-World restored, layer the B4/B5 gate sweep (`sweep_gates.sh`) —
   now backed by real `sam_score` values from SAM-Large — on the frozen
   `captures_nav` dumps.
3. Re-baseline `arabic_room` with YOLO + SAM-Large (vs the old YOLO + SAM-Tiny)
   to isolate the mask-quality gain the upgrade should bring.

## Related

- TASK 27 — time-sync + extrinsic fix (the lift this detector feeds).
- Backlog B4 (range cap) / B5 (SAM mask quality + SAM-Large) — the gates this
  composes with.
