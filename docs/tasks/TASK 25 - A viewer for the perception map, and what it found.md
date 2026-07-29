# TASK 25 — A viewer for the perception map, and what it found

Every number in TASK 22–24 is an average over hundreds of observations, and an
average cannot show *why* a box is wrong. Twice already a picture has overturned
a conclusion the aggregates supported — the equirect overlay is what caught the
`fan decoration` ground truth being one box around three fans plus the wall
between them, which is what made TASK 23's 1.622 lidar oracle partly circular.

So this task built a browser viewer for the replayed scene, and then used it.
Four measurements came out of the exercise. **Three of them contradict things
this repository currently believes**, including one that says our reported
counting error has been inflated by a bug in the metric rather than by the
pipeline.

## The viewer

`scripts/export_viz.py` replays a scene through the real `PointLifter` and
`ObjectMap` at the benchmark's gating (`min_move 0.15`, `min_rot 10°`,
`min_score 0.35`, `min_inliers 10`) and dumps a JSON manifest plus one
`float32` blob of `xyz` triples. The manifest indexes into the blob by point
offset, so the page fetches geometry as an `ArrayBuffer` and never parses
coordinates out of text. **A box on screen is the same box in the benchmark
table**, not a re-derivation free to drift from it.

`scripts/serve_viz.py` serves `viz/` and fetches three.js r128 into
`viz/vendor/` on first run. r128 is the last release whose builds are plain
`<script>` globals rather than ES modules, which keeps the page dependency-free.

Layers: accumulated lidar cloud, the current frame's scan, the lifted inlier
points per detection (coloured per label), per-frame detection boxes, our fused
objects, ground-truth boxes, the ground-truth scene cloud, and the robot path.
Frame slider with play, filters on label / score / observation count, and
click-to-inspect on any box.

Two things worth knowing about how it works:

* **Frame provenance.** `ObjectMap.add` picks or creates a node without telling
  the caller which, so the frame stamp is taken from the node side by wrapping
  `_Node.__init__` and `_Node.merge`. That covers both the born-here and
  merged-into paths without duplicating the merge rule, which would be free to
  drift from the real one. `node_id` survives `finalize()`, so the stamps still
  key the exported objects. This drives the *detected up to this frame* and
  *seen in this frame* modes — the only way to watch a duplicate node split off
  as it happens.
* **The ground-truth cloud** is each scene's own `map.ply`, read straight out
  of the challenge zip and voxel-thinned to 5 cm. It is bare `xyz`: no colour,
  no per-point labels. `loft` is 12.7 M points thinned to 443 k, and the whole
  seven-scene export costs ~20 MB extra.

Everything runs locally: `rsync` the ~130 MB of `viz/data/` to any machine and
`serve_viz.py` needs nothing else. `viz/data/` and `viz/vendor/` are gitignored.

## Finding 1 — our counting error is inflated by the metric, not the pipeline

`cMAE` averages `|gt_c - pred_c|` over every class either side names. That one
number hides four different failures, and only two are duplicates:

| term | meaning | who owns it |
|---|---|---|
| `over` | class exists, we predict too many | deduplication |
| `spurious` | class ground truth does not have | naming |
| `short` | class named, too few found | recall |
| `absent` | class never named at all | vocabulary |

Decomposed over the seven benchmark scenes (per-scene means):

| config | #pred | over | spurious | short | absent | cMAE | floor |
|---|---|---|---|---|---|---|---|
| greedy 0.4 (shipped) | 441 | 99 | 299 | 3 | 32 | 7.27 | 0.59 |
| merge_dist=0.8 | 262 | 48 | 174 | 6 | 32 | 4.44 | 0.65 |
| relative=1.0 | 254 | 63 | 150 | 4 | 32 | 4.22 | 0.61 |
| families | 415 | 87 | 289 | 3 | 35 | 7.09 | 0.65 |
| fam+0.8 | 237 | 39 | 163 | 5 | 37 | 4.28 | 0.74 |
| fam+0.8+n_obs>=3 | 174 | 23 | 119 | 6 | 40 | 3.77 | 0.91 |
| fam+recl0.8+n_obs>=3 | 154 | 19 | 104 | 6 | 40 | 3.42 | 0.93 |

`floor` is `(short + absent) / classes`: what survives if every duplicate and
every spurious node vanished at no cost to what we did find. **It is 0.59, not
the ~4 that had been assumed.** Recall is not what holds cMAE up.

But the dominant term, `spurious`, was mostly an artefact of the measurement.
The ground truth is filtered by `is_structure`; the predictions were not. Score
both sides the same way:

| variant | #pred | over | spurious | short | absent | cMAE | floor |
|---|---|---|---|---|---|---|---|
| as scored today | 441 | 99 | 299 | 3 | 32 | **7.27** | 0.59 |
| structure dropped from predictions too | 248 | 99 | 107 | 3 | 32 | **4.59** | 0.67 |

Of 2095 surplus nodes across seven scenes, **1348 (64%) are structure** —
`floor`×537, `ceiling`×428, `window`×100, `carpet`×90, `door`×75, `wall`×72,
`column`×27 — and **1309 of them carry a label the ground-truth zip does
contain**, stripped only by `is_structure`. These are not detection errors.
**Correcting the asymmetry is a metric fix, not an improvement**, and it is the
first thing to do.

After the fix, the two movable terms are the same size: `over` 99 against
`spurious` 107. Deduplication and naming are equal partners, not a hierarchy.

## Finding 2 — the spurious labels are synonyms, not hallucinations

The non-structure surplus, by label, across seven scenes:

```
lamp 71 · dining table 62 · picture 54 · couch 49 · bench 44 · tv 42
side table 31 · bed 25 · cabinet 25 · wardrobe 24 · tv stand 23 · lantern 23
photo 22 · bookcase 22 · painting 19 · coffee table 18 · wall lamp 16
```

This is almost exactly the `FAMILIES` table already sitting in
`scripts/fusion_sweep.py`: couch/sofa, dining table/table, picture/painting/
photo, bench/chair, lamp/lantern/wall lamp/ceiling lamp. The viewer shows it
directly — a ceiling full of overlapping `lamp` / `wall lamp` / `focus light` /
`lantern` boxes on the same fixtures.

Duplicates inside classes ground truth *does* have (693 total) are far more
concentrated: `chair` 192 (28%), `potted plant` 54, `cabinet` 46, `painting` 34.

## Finding 3 — a global merge threshold cannot fix counting

The class each scene's official numerical question counts, against what each
fusion config produces:

| scene | target | GT | shipped | merge 0.8 | rel 1.0 | fam+0.8+n≥3 | fam+recl |
|---|---|---|---|---|---|---|---|
| arabic_room | sofa | 3 | 18 | 11 | 14 | 9 | 9 |
| chinese_room | chair | 6 | 32 | 15 | 17 | 12 | 10 |
| japanese_room | calligraphy painting | 3 | **0** | 0 | 0 | 0 | 0 |
| livingroom_3 | photo | 2 | **2** | 2 | 2 | **1** | **1** |
| loft | pillow | 11 | 8 | 6 | 8 | **6** | **5** |
| office_1 | computer monitor | 6 | 14 | 7 | 9 | 8 | 7 |
| office_2 | potted plant | 3 | 5 | 4 | 5 | 3 | 2 |

Aggressive merging fixes the over-counted classes (`sofa`, `chair`,
`computer monitor`) and **breaks the ones already under-counted** — `photo`
2 → 1, `pillow` 8 → 5 against a ground truth of 11. One global threshold cannot
serve both. This is direct evidence for a *selective* rule (the size-prior veto
sketched in TASK 24's notes) over a looser uniform one.

`japanese_room`'s `calligraphy painting` is 0 at every setting. No fusion rule
reaches a vocabulary failure.

Note also that all fifteen official numerical questions are *conditional*
counts — "how many X are on Y" — so the whole-scene count is necessary and
nowhere near sufficient.

## Finding 4 — a detection box with no object under it, attributed

`japanese_room` frame 0, 17 lifted detections, 244 nodes built, 185 exported:

| verdict | n | meaning |
|---|---|---|
| kept | 9 (53%) | fused object overlaps the detection |
| moved | 3 (18%) | node survived, but its fused box sits elsewhere |
| structure | 3 (18%) | exists; hidden by the structure filter |
| nms | 2 (12%) | deleted outright by `finalize()` at IoU 0.84 and 0.51 |

The `moved` cases are the interesting ones: `potted plant` node 7 (`n_obs` 22)
has IoU **0.05** with its own frame-0 observation, `bird decoration` node 6
(`n_obs` 27) has 0.04. The fused box is the point-count-weighted mean over 20+
views, so it need not sit on any single one — this is what TASK 24's 0.236 m
lateral centre error looks like on screen.

`bird decoration` also appears twice in that one frame, both merging into the
same node — TASK 24's "30.7% of object-frames have more than one detection".

## Finding 5 — pricing the ground-truth box convention

`object_list.txt` annotates an **oriented** box: `id cx cy cz lx ly lz heading
"label"`. `perception/eval.py:286` builds its AABB as `center ± size/2` with
**the heading ignored**. For a rotated object the box we are scored against is
the object's local box dropped into the world unrotated — neither the true
oriented box nor its world-aligned bound.

Our predictions come from point clouds and are world-axis-aligned by
construction, so the best a perfect system could produce is the world AABB of
the true oriented box. Over 542 non-structure ground-truth objects:

| | |
|---|---|
| heading under 5° (unaffected) | 50.7% |
| **mean IoU a perfect axis-aligned prediction could reach** | **0.783** |
| p25 / p50 / p75 | 0.590 / 0.843 / 0.986 |
| below 0.75 | 42.8% |
| **below 0.50 — can never earn the 2-point tier** | **6.5%** |

Per scene the mean runs 0.735 (`chinese_room`) to 0.850 (`office_2`).

**This is real but not the current bottleneck**: it caps mean IoU at 0.783 and
we are at 0.177. It matters later. The exception is the 6.5% that can never
reach IoU 0.5 however good the perception gets.

The worst-affected are thin elongated objects — `pen` 0.139, `marker` 0.146,
`chopsticks` 0.247 — which overlaps heavily with the sub-0.3 m bin TASK 23
showed cannot reach IoU 0.5 even when cheating. The penalty largely stacks onto
existing failures rather than adding new ones. The exception worth noting is
`loft`'s pillows: four of the fifteen worst, and `loft`'s numerical question is
"How many black pillows are on the sofa?"

The viewer can draw either convention (**"…drawn with heading (true OBB)"**),
so the difference is inspectable per object.

## Corrections to earlier claims

* **"441 nodes against 542 ground-truth objects, so we under-count."** Wrong:
  `fusion_sweep.py` reports `n` as a per-scene mean, not a total. Roughly 3000
  nodes against 542 objects — we over-count about 5×.
* **"Deduplication bottoms out around cMAE 4.0."** Wrong: the floor is 0.59
  (0.67 scored symmetrically), and `fam+recl0.8+n_obs>=3` already reaches 3.42.
* **"Spurious classes dominate."** True as measured, but 64% of that was our
  own asymmetric structure filtering, not the detector.

The viewer reproduced the same asymmetry on screen at first — per-frame
detections were judged by a narrow regex while fused objects used
`vocab.is_structure`, so windows and doors drew a detection box with no object
under it. The manifest now carries one structure verdict per label for both.

## Next

1. **Make the structure filter symmetric in the benchmark.** A metric fix worth
   cMAE 7.27 → 4.59 with no pipeline change. Decide at the same time whether to
   admit openings on both sides: `window` 100 + `door` 75 + `column` 27 +
   `window frame` 12 = 214 nodes per seven scenes would become legitimate, and
   22 of the 75 official questions name one.
2. **The alias merge from TASK 23**, still unimplemented, against `spurious` 107.
3. **Selective deduplication** — size-prior veto plus relative radius — against
   `over` 99, with Finding 3 as the acceptance test: it must not cost `photo`
   or `pillow`.
4. **Vocabulary** for the zero-detection classes such as `calligraphy painting`.

## Files

| file | what |
|---|---|
| `scripts/export_viz.py` | replay one scene into manifest + float32 blob |
| `scripts/serve_viz.py` | static server, fetches three.js on first run |
| `viz/index.html`, `viz/app.js` | the viewer |
| `.gitignore` | ignore `viz/data/`, `viz/vendor/` |

The four analyses above were run from scratch scripts on the box
(`scratch_cmae.py`, `scratch_cmae2.py`, `scratch_frame0.py`,
`scratch_heading.py`), uncommitted, in the same style as TASK 23's and 24's.
441 tests pass, 1 skipped; no library code changed in this task.
