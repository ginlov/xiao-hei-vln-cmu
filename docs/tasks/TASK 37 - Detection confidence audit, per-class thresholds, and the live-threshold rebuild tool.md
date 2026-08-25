# TASK 37 — Detection confidence audit, score/SAM thresholds, and the live-threshold rebuild tool

## Why

The score floor sat at 0.6 (raised there to keep low-confidence fragments out of
the face-seam merge). A per-class confidence audit on arabic_room showed 0.6 is
too strict for real objects whose YOLO score peaks below it, while never
recovering classes the detector can barely see at all — so one global floor is
the wrong knob.

## What we measured

Re-ran the sidecar at a 0.05 floor over the 36 novelty-gated viewpoints
(`perception_benchmark`, `yolo_conf_stats.py`), 2136 detections. Only 291 (13.6%)
clear 0.6. Per-class:

- **0.6 silently drops real objects**: `door` (GT 4, max 0.52), `table`
  (GT 2, max 0.60), `door frame` (GT 2, max 0.59).
- **Detector-limited, no threshold recovers them**: `focus light` (GT 13, max
  0.13), `window` (GT 4, max 0.20), `glass`/`tray` (never detected).
- **Over-detection at a low floor**: `wall lamp` 228 dets for GT 4, `picture`
  216 for GT 2 — lowering the floor multiplies false positives, so a mask-quality
  gate is needed alongside.

Interactive report (form + per-class explorer): published Artifact
"Detection Confidence Audit".

## Shipped — defaults changed (live pipeline)

`perception/responder.py` + `app/main.py` `_PerceptionSettings`:

| knob | was | now | env |
|---|---|---|---|
| `DEFAULT_SCORE_THRESHOLD` | 0.6 | **0.4** | `XIAO_HEI_PERCEPTION_SCORE_THRESHOLD` |
| `DEFAULT_SAM_THRESHOLD` | — (no gate) | **0.8** | `XIAO_HEI_PERCEPTION_SAM_THRESHOLD` |
| `scan_keyframes` (main default) | 10 | **2** | `XIAO_HEI_SCAN_KEYFRAMES` |

The **SAM mask-quality gate is new in the live responder** (previously only in
the benchmark scripts): detections with `sam_score < 0.8` are dropped before
lifting. It pairs with the 0.4 floor — 0.4 alone adds recall but also weak/
bleeding masks; SAM 0.8 keeps the extra detections clean. `scan_keyframes` moved
to 2 so the live default matches what every offline sweep was measured at.

Perception score (arabic_room, gate 0.3 m, live sidecar), 0.6 → **0.4 + SAM 0.8**:
mAP@1.0 0.235→**0.337** (+43%), recall 0.235→0.383, IoU@0.25 0.091→**0.134**, at
a precision cost (0.79→0.62).

## Shipped — offline tooling

- **`perception_benchmark/dump_lifts.py`** — precompute each detection's lifted
  cloud once at a low floor (0.05), honouring the novelty gate. Lifting is
  threshold-independent (mask + scan only); only fusion depends on the score cut.
- **`viz_app.py` "Live threshold" tab** — re-fuses those cached lifts into an
  `ObjectMap` at any global + per-class score cut and SAM gate, **instantly**
  (fusion is ms). Full graph by default, opt-in to scrub by viewpoint. Reproduces
  the pipeline exactly (24 nodes @0.6, matching the frozen replay).
- **`scripts/build_scene_json_from_lifts.py`** — the same re-fuse, exported to a
  `SceneRepresentation.to_dict()` `scene.json` the offline Gemini eval consumes.

## End-to-end validation (arabic_room, object_reference, n=133)

| | mean challenge score | IoU | SR@0.25 | SR@0.5 | centre dist |
|---|---|---|---|---|---|
| baseline (live explore, old params) | 0.293 | 0.142 | 0.203 | 0.090 | 1.573 m |
| captures_nav, old-thr 0.6 | 0.128 | 0.049 | 0.128 | 0.000 | 2.993 m |
| **captures_nav, new 0.4/SAM0.8** | **0.165** | 0.064 | 0.150 | 0.015 | 2.859 m |

Two separable conclusions:

1. **The threshold change helps, controlled for trajectory.** On the *same*
   `captures_nav` replay, 0.4/SAM0.8 beats 0.6 on every metric — challenge score
   +29%, IoU +31%. The offline mAP gain carries through to grounding.
2. **The dominant e2e bottleneck is exploration coverage, not perception.** Both
   `captures_nav` runs sit far below the baseline because that replay sees only
   19–35 objects vs the live explore's 188. Beating 0.293 needs the new config on
   a *proper live explore* (the Docker sim path), which the offline replay caps.
   The first naive before/after (baseline 0.293 vs new 0.165) was this trajectory
   confound, not the thresholds.

## Open / next

- Re-validate against the baseline with a **live-sim explore** on the new config.
- SAM 0.8 also filtered `door` out of the arabic_room graph — revisit whether the
  SAM gate should be per-class too, or lower (0.7).
- Per-class score thresholds are supported in the tools; not yet wired into the
  live responder (a per-class override map defaulting to 0.4 is the next step).
- Sweep score/SAM across the corpus before locking 0.4/0.8 globally.
