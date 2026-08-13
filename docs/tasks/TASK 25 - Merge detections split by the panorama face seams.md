# TASK 25 — Merge detections split by the panorama face seams

## Why

The sidecar unwraps the 360° panorama into 4 perspective faces (100° each,
~10° overlap) and runs YOLO-World + SAM per face. An object straddling a seam
is detected twice — once per face — and if it is wider than the overlap it is
*clipped* in both, so neither detection covers the whole object.

`perception/pipeline.py` knew about this and deliberately declined to handle
it:

> *"No cross-face NMS for v1 — the responder-side `SceneRepresentation` merges
> duplicates by 3D distance (`merge_radius`), which handles the ~10°
> face-overlap region naturally."*

That contract broke in TASK 24 §C. `add_object` and its 1.5 m `merge_radius`
were deleted; fusion is now `ObjectMap`, whose `MERGE_DIST` is **0.4 m**. The
tolerance absorbing seam duplicates shrank 3.75× and nothing recorded the
dependency.

## The evidence

Reported from the viewer: `picture` and `carpet` visibly split at
`arabic_room` vp_024. The raw per-frame detections confirm it — one carpet, two
detections:

```
carpet  0.937  [-2.962,  0.029, 0.037]
carpet  0.950  [-2.500, -0.134, 0.037]     0.48 m apart, identical z (floor)
```

0.48 m is just past `MERGE_DIST`, so `ObjectMap.add` leaves them as two nodes.

A face seam sits at a fixed *robot-relative* bearing (faces centred at
0/90/180/270°, so seams recur every 90°). Over 316 frames, 1439 same-label
detection pairs lying <1 m apart **within a single frame**:

| bearing mod 90° | share | | bearing mod 90° | share |
|---|---|---|---|---|
| 0–10° | 1.9% | | 40–50° | **29.7%** |
| 10–20° | 1.3% | | 50–60° | **26.1%** |
| 20–30° | 1.5% | | 60–70° | **32.1%** |
| 30–40° | 3.1% | | 70–80° | 1.5% |
| | | | 80–90° | 2.8% |

Uniform would be 11.1% per bin. **88% of duplicates fall in one 30° band**
around a seam; max/min bin ratio 24.3.

### The measurement that was wrong first

The same test run on *fused* `ObjectMap` nodes came out flat (ratio 1.26), and
on that basis this was initially written off as "not a seam problem". That was
the wrong level. Fusion unions clouds across 316 frames covering all 36 of 36
ten-degree yaw bins, which moves node centres off the seam bearing and erases
the signature — while leaving the duplicate nodes it created in place. Only the
per-frame detections show it. Worth remembering: **measure a per-frame artefact
per frame.**

## What was built

`merge_seam_duplicates()` in `perception/pipeline.py` — a pure function over
already-projected equirect masks, so it is testable with no GPU, no torch and
no weights. `detect()` now collects `_FaceDetection` records carrying the face
index and face-space bbox, merges, then serialises; the face metadata is
dropped before the wire, so no client change.

Grouping is union-find, not pairwise: an object wide enough to cross two seams
is clipped across three faces, and merging only neighbouring pairs would leave
a fragment behind. The survivor keeps the group's best score and the **union**
of the masks — not the best fragment, because when the object is wider than the
10° overlap *neither* fragment is complete.

Merging requires all four of: different faces, same label, vertically-clipped
fragment, and adjacent masks.

### The two things that were not obvious

**Cross-face masks never overlap.** The first implementation required the
equirect masks to intersect, on the reasoning that the faces overlap ~10° so
the fragments must share pixels. They do not: `build_inverse_lut` assigns each
equirect pixel exactly one owning face, so cross-face masks *partition* the
sphere. Measured on the vp_024 carpet:

```
carpet f3  → equirect cols 340..719   (clipped at RIGHT face edge)
carpet f0  → equirect cols 720..873   (clipped at LEFT face edge)
column gap = 1 px      row gap = -169 px (rows overlap)
```

The intersection test could not fire on any input, and the first A/B scoring
run came back **bit-identical** to baseline. The rule is adjacency, not
overlap: gap measured on the column axis as a cylinder (both directions modulo
the width, smaller wins), plus a confirmation that both masks occupy
overlapping *rows* in the band around the junction — two same-label objects
stacked one above the other share a column boundary without being one object.

**Only vertical clipping counts.** `_touches_face_border` checks the left and
right edges only. Seams are vertical; a top/bottom clip is the panorama's ±60°
vertical crop, which every floor-level detection hits. Counting it would make
`carpet` mergeable with anything it touched — the reported symptom would have
"gone away" for entirely the wrong reason.

Writing the wrap-seam test found a third bug: at λ=±π the junction band ran
backwards from column 1919 to column 0 and evaluated empty, so anything split
at the rear seam would never have merged.

## Guardrails

The risk is over-merging: collapsing two genuinely adjacent same-label objects
(two pictures on a wall, two chairs side by side) into one. The clipped-fragment
requirement is what prevents it, and it is asserted directly —
`test_adjacent_objects_inside_one_face_are_left_alone` uses abutting masks with
neither fragment clipped and requires two survivors.

`PERCEPTION_MERGE_SEAMS=0` disables the merge without a rebuild, so the A/B
stays reproducible.

## Verification

`perception/tests/test_seam_merge.py` — 17 tests, no GPU. Both directions:

- a clipped pair across faces merges, using the **real column ranges measured
  at vp_024**, and the result is the union of both masks;
- two interior same-label objects with abutting masks stay separate;
- same-face duplicates, different labels, and disjoint masks are left alone;
- a three-way split collapses to one (transitivity);
- vertically-stacked fragments sharing a column boundary do **not** merge;
- fragments across the λ=±π wrap seam do merge;
- input masks are not mutated in place (the caller still holds them for the
  debug dump).

`perception/tests/` 51 passed. To make the module importable on the host,
`import cv2` moved into the two methods that use it (`_dump_debug`,
`_unwrap_to_faces`) — matching how `torch`/`sam2`/`ultralytics` are already
lazily imported in `__init__`. Without that the pure function could not be
tested outside the container.

Live check on vp_024 (restricted class list): **21 detections → 19**.

## Results

Scored on `captures_nav/arabic_room`, 316 frames, via
`perception_benchmark/replay_score.py`.

| | baseline | seam merge | |
|---|---|---|---|
| mAP@1 | 0.2928 | **0.3103** | +0.018 |
| P@1 | 0.0971 | **0.1003** | +0.003 |
| R@1 | 0.4783 | 0.4783 | unchanged |
| F1@1 | 0.1614 | **0.1658** | +0.004 |
| cErr | 0.469 m | **0.449 m** | −0.020 |
| cMAE | 15.409 | **14.909** | −0.500 |
| pred nodes | 340 | **329** | −11 (GT 69) |

Every metric moved the right way, and recall is **exactly unchanged** — which
is the result to check first. Merging fragments can only help precision if it
does not lose objects; identical recall says the 11 removed nodes were
duplicates, not real detections that got absorbed into a neighbour.

The effect is modest because `arabic_room` is a stalled trajectory (66% of
frames below 0.05 m/s), so most of its 316 frames re-observe the same few
seam-split objects rather than presenting new ones. The remaining 329-vs-69
gap is fragmentation from other causes — see the TASK 22 note below.

## Related

- TASK 24 §C — deleted the `merge_radius` this design depended on.
- TASK 22 — `NMS_DIST`/`NMS_GAP` same-label suppression. Complementary: that
  rule works on fused 3D nodes, this one on 2D detections before the lift.
  Measured on `arabic_room`, **zero** same-label node pairs with touching
  surfaces fall inside `NMS_DIST=0.4` — the threshold sits below the entire
  population (median centre distance 0.95 m), so that rule is currently inert
  and remains open.
