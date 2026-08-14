# TASK 30 — Raising the detection floor to 0.6 and gating the seam merge

## Why

Reviewing the carpet fragments frame by frame ([[TASK 29]]) surfaced a real
bug in *where* the score threshold was applied. The panorama face-seam merge
(`merge_seam_duplicates`, TASK 25) runs inside the sidecar, right after YOLO
detects each face. But the offline benchmark froze detections at a **0.25**
floor and re-filtered to a higher floor *after the fact* — which is after the
seam merge already ran. So a 0.9-confidence carpet detection could be unioned
with a 0.3-confidence fragment, inheriting the high score and carrying the
fragment's mask contamination past every offline filter. `arabic_room` vp_002
showed exactly such a spurious carpet.

## What changed

- **Score threshold raised to 0.6 across the stack.** The per-face gate that
  matters was always there — it is YOLO's `conf=score_threshold` in
  `PerceptionPipeline.detect` (`perception/pipeline.py`), applied before SAM and
  before the seam merge. Raising the floor to 0.6 means low-confidence fragments
  never enter the seam merge. Defaults updated in: `responder.py`
  (`DEFAULT_SCORE_THRESHOLD`, the live value), `pipeline.detect`,
  `client.detect`, `server.py` (the `/detect` Form default), `replay.py`,
  `perception/__main__.py`, and the benchmark tools `dump_detections.py`,
  `dump_debug.py`, `replay_score.py`, `debug_viewpoint.py`.
- **Frozen detections re-dumped at 0.6.** `captures_nav/arabic_room` was
  re-run through the sidecar (A10G) at `--score-threshold 0.6`, so YOLO + SAM +
  seam merge all operate on the ≥0.6 set. The old 0.25 frozen dets are gone;
  offline floor sweeps below 0.6 now need a fresh sidecar re-dump.

## Result — `arabic_room` carpet nodes

| configuration | carpet nodes | total nodes |
|---|---|---|
| 0.25 floor | 27 | 199 |
| 0.25 dump, filtered to 0.6 offline (after the merge) | 13 | 50 |
| **0.6 dump, seam merge run at 0.6** | **9** | 46 |

The step from 13 → 9 is the fix: four carpet fragments existed only because
low-confidence detections had been eligible for the seam merge. vp_002's
spurious carpet is gone; its one remaining carpet detection lifts inside the GT
box. The residual (node 0 + node 26 over GT #58, node 53 + node 14 over GT #32)
is cross-frame [[B6 large-object fragmentation]], which the seam merge — a
within-frame operation — does not address.

## Caveat

Raising the **live** floor to 0.6 trades recall for precision on small/faint
objects; the 0.35 value it replaces was swept for recall. This wants an
end-to-end re-score to confirm it is a net win on the challenge metric, not just
on node counts. Regenerated debug dump, both merge videos, and the vp_002 /
node-0 inspect images all reflect the 0.6 pipeline.
