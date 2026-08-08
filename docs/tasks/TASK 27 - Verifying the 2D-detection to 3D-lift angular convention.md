# TASK 27 — Verifying the 2D-detection → 3D-lift angular convention

## Why

Reported symptom: the angle used to lift a 2D detection looks like it does not
match where the detection actually is.

This is a class of bug that cannot be found by looking at outputs. The lift
decides which LiDAR returns belong to a mask by projecting each return into
equirect pixels and testing membership. If the mask's angular convention and
the projection's disagree, the lift attaches the wrong surface to the object
and **nothing downstream complains** — the result is internally consistent,
just wrong. Checking the lift against the projection is circular. It has to be
checked against something independent: ground truth in the map frame.

`perception_benchmark/verify_projection.py` does that. It reads only what is
already on disk — frozen masks (`detections.npz`), poses, and GT from
`viz.json` — so no sidecar and no GPU.

## Verdict

**The angular convention is correct. Two other bugs were found and fixed:**

1. A **temporal** mismatch — the image and the pose used to lift it were ~0.37 s
   apart, up to ~17° of azimuth while turning. Fixed at the source:
   `LatestCache` now timestamp-matches pose and scan to the image.
2. A **static extrinsic** error — the sensor→camera translation was pasted in
   unrotated, displacing the camera 0.1 m forward instead of up. Fixed.

On `arabic_room` the two together take mAP@1 0.31 → 0.39, precision +83%,
centre error −44%, and predicted-node count 329 → 174 (GT 69).

## 1. Round trip — pure geometry, no data

| check | max error |
|---|---|
| equirect pixel → angle → pixel | 2.3e-13 px |
| angle → world dir → angle | 1.3e-14 deg |
| face pixel → world dir → face pixel | 1.1e-13 px |
| **responder projection vs sidecar projection** | **0.000 px** |
| inverse LUT: equirect px → face px → equirect px | 2.7e-05 px |

The fourth row is the one that matters. `perception/geometry.py` (sidecar,
mask side) and `xiao_hei_vln/perception/geometry.py` (responder, lift side) are
two independent implementations of one convention, and the module docstring
warns that they must be kept in step by hand. They agree **exactly** — a
camera-frame point lands on the same equirect pixel through both paths. There
is no sign flip, no 90° face-yaw error, no seam offset.

## 2. Azimuth vs ground truth — 316 frames, 7482 detections

Method: circular-mean longitude of each mask, against the longitude of
same-label GT objects within 8 m, transformed through the **production**
chain (`_rotation_from_quaternion` → `sensor_to_camera_transform` → equirect).
Then a global offset scan.

The scan is what makes this work. A plain mean of nearest-GT residuals cannot
find a large offset — the nearest GT is chosen *after* the offset is applied,
so a wrong pairing always looks small. Sweeping θ over the whole circle and
re-pairing at each step is what would make a 90° or 180° error visible.

```
θ      -180   -135    -90    -45    -10      0    +10    +45    +90   +135
cost   80.3   74.2   53.6   35.9   25.1   22.3   25.1   37.5   56.6   74.7
```

Minimum at **exactly 0°**, symmetric on both sides. Gain from the best offset:
**+0.00°**. Offset regressed against robot yaw: slope −0.007, corr −0.033 — so
no frame confusion either (yaw applied twice would give ≈−1, sign-flipped ≈−2).

The 22.3° floor is *pairing ambiguity*, not error: the scene has 20 wall lamps
and 12 pictures, and "nearest same-label GT" is genuinely uncertain among them.

## 3. Elevation vs ground truth

Run separately, because a vertical bug (flipped `v`, wrong crop) would blur
check 2.

| | |
|---|---|
| best global offset | −1.5° (gain 0.5°) |
| forced-pair median &#124;Δ&#124; | 5.7° · 68% <10° · **100% <20°** |
| residual ∝ 1/range fit | **Δheight +0.053 m** |

The 1/range fit is the discriminator: a camera-height error tilts elevation by
`Δφ ≈ Δh / r`. A mis-signed camera-above-LiDAR offset would read 0.2 m; the
measurement is 0.05 m, and no forced pair exceeds 20°. This is mask centroid
vs box centre, not a convention error.

## 4. The actual finding — image/scan time skew

Everything above tests for a *constant* error. A time skew between the image
and the pose is invisible to all of it: it is zero while the robot is
stationary (two thirds of this capture) and grows with turn rate. So regress
the per-frame azimuth offset on **yaw rate** — the slope is in seconds.

```
skew = -0.3681 s,  corr = -0.629
```

That is a strong correlation, but a per-frame offset estimated from noisy
pairing could produce one spuriously. Falsification: compensate the GT
bearings by `skew × yaw_rate` and re-measure. If the skew is real the
correlation must vanish; if it is an artefact, compensating changes nothing
and the wrong sign is no worse than the right one.

| applied skew | mean azimuth residual | remaining yaw-rate corr |
|---|---|---|
| 0 s | 22.30° | −0.629 |
| −0.2 s | 20.90° | −0.348 |
| **−0.368 s** | **20.13°** | **−0.004** |
| +0.368 s (wrong sign) | 24.81° | −0.850 |

The correlation collapses to zero at the fitted value and **doubles** at the
wrong sign. The effect is real.

The residual only improves by 2.2° because the 22° floor is pairing ambiguity
that no skew correction can touch — the correlation going to zero is the
evidence, not the residual drop.

### What it means

At the turn rates in this capture (up to ~0.8 rad/s) a 0.37 s skew is up to
**17° of azimuth error**, applied to every mask, only while turning. The lift
then selects LiDAR returns from ~17° away from the object — which is exactly
the "the angle for lifting doesn't match the detection" symptom, and exactly
the kind of error that produces boxes fused with the wall behind the object
(the 5.4 m `potted plant` nodes in TASK 26).

The cause is upstream of the geometry: `LatestCache.snapshot()` took whatever
message arrived most recently on each topic with **no timestamp matching**
(noted in TASK 24). The image stream is the slower one, so it was stale
relative to the pose by roughly one tick.

### What was done about it — fix it at the source

The cause is `LatestCache`, so that is where it is fixed. Every message carries
`header.stamp`, but the cache kept only the latest value per slot and paired a
fresh pose with a stale image. It now keeps a short **history** of poses and
registered scans and, at `snapshot()`, returns the entries whose stamps are
**nearest the image's stamp** — the image being the anchor because it is the
laggard. No lag constant, no yaw-rate estimate, no per-machine tuning; it reads
the alignment straight off the timestamps that were always there.

Fallbacks preserve the old behaviour where matching is impossible: no image
(no anchor) → latest; a stream gap wider than 1 s → latest, rather than pairing
the image with an unrelated pose. `tests/test_sync.py` covers the match, both
fallbacks, and that the ring bound does not drop a still-recent pose.

`sensor_scan` and terrain stay latest-only — they are not lift inputs, and
navigation is coarse against a sub-second skew.

This removed the responder/`main` pose-compensation that an earlier pass had
added: with the cache aligned, the lift needs no correction of its own.

The offline **recorder** (`record_navigation.py`) had the same latest-per-topic
pairing in its own `self.latest` dict — it reproduced the skew independently of
`LatestCache`, so captures taken with it baked the bug back in. Since its job is
to reproduce the live ingest, it now mirrors the match: a stamped history of
pose and registered scan, the entry nearest the image stamp written to each
frame, with `image_t` / `pose_t` / `pose_dt_ms` recorded in `meta.json` so the
alignment is auditable per frame. Fresh captures therefore reflect the fixed
stack directly and no longer need the `--image-lag` simulation.

#### Validating it offline

The live fix cannot be replayed on the frozen captures — each viewpoint stored
one already-mispaired pose and a single timestamp, with no stream history to
match against. So `deskew.py` survives as an **offline simulation**:
`replay_score --image-lag S` de-rotates the stored pose by `S × yaw_rate`,
reproducing what the timestamp-matched cache would have handed the lifter. That
is what the scores in §6 measure. Its sign is locked by `tests/test_deskew.py`
(asserts `d(bearing)/d(yaw) = +1` against the production transform, then that
the correction rotates *backwards* on a left turn), and `verify_projection
--skew` drives that same code against ground truth:

| | yaw-rate corr | mean azimuth residual |
|---|---|---|
| no correction | −0.629 | 22.30° |
| simulated match (0.368 s) | **+0.028** | **20.13°** |

The correlation collapsing to zero is the confirmation that the simulation —
and therefore the live matching it stands in for — removes the skew.

## 5. A second bug, found while deriving the sign

`_SENSOR_TO_CAMERA_TRANSLATION` was `(0, 0, 0.1)`. The sim publishes this as a
ROS `static_transform_publisher`, whose translation is in the **parent (sensor)
frame** — it cannot be pasted into a camera-frame constant. With the camera
origin at `c = (0, 0, 0.1)` in sensor coords:

    p_cam = R (p_sensor - c) = R p_sensor - R c
    -R c  = -0.1 * R ẑ = -0.1 * (0, -1, 0) = (0, +0.1, 0)

So the constant is **0.1 m along camera +y (down)** — the LiDAR sits *below*
the camera. The shipped value put the camera 0.1 m **forward** instead, which
tilts elevation by ~`0.1/r` and skews azimuth for anything off-axis.

Measured against ground truth on `arabic_room`:

| | as shipped | corrected |
|---|---|---|
| elevation residual | 2.98° | **2.26°** |
| residual bias (best offset) | −1.50° | **−0.50°** |
| gain from correcting the bias | 0.50° | **0.01°** |
| forced pairs within 5° | 34% | **52%** |
| forced pairs within 10° | 68% | **98%** |

The "gain" row is the one that settles it: after the fix there is essentially
no systematic elevation offset left to remove. Fixed, with
`tests/test_perception_extrinsic.py` stating the geometry physically (where the
LiDAR origin lands in camera coords; a point level with the camera projecting
to the horizon row) rather than restating the constant.

This one is **on by default** — unlike the lag, it is a derivation with one
right answer, not a fitted constant.

## 6. What it buys — `arabic_room`, 316 frames

Scored with `replay_score.py` off the frozen masks, so detection is identical
across every row and only the lift changes. GT = 69 objects.

| config | mAP@1 | P@1 | R@1 | F1@1 | cErr | cMAE | pred |
|---|---|---|---|---|---|---|---|
| baseline (HEAD) | 0.3103 | 0.1003 | 0.4783 | 0.1658 | 0.449 | 14.909 | 329 |
| extrinsic fix only | 0.3304 | 0.1140 | **0.5072** | 0.1862 | 0.373 | 13.818 | 307 |
| **both fixes** | **0.3850** | **0.1839** | 0.4638 | **0.2634** | **0.253** | **7.773** | **174** |

Against baseline: mAP **+24%**, precision **+83%**, F1 **+59%**, centre error
**−44%**, counting MAE **−48%**, and predicted nodes 329 → **174** against 69
real objects — the fragmentation is roughly halved.

The extrinsic fix alone improves *every* metric including recall, which is what
a pure correction should do. The lag correction then trades **3% of recall**
(0.4783 → 0.4638) for the rest. That trade is real and worth stating plainly:
smearing each mask across ~17° of extra bearing spawned nodes, and a few of
them happened to land on GT objects. Those were luck, not detections — the
precision and counting numbers say so.

### The lag sweep is its own confirmation

The offline simulation takes a lag; the live cache does not, so this sweep is a
sanity check on the *simulation*, not a knob anyone tunes in production. The
alignment the cache finds from timestamps corresponds to the physical skew, and
the sweep shows scoring peaks right there:

| simulated lag | mAP@1 | P@1 | F1@1 | cErr | cMAE | pred |
|---|---|---|---|---|---|---|
| 0.0 | 0.3304 | 0.1140 | 0.1862 | 0.373 | 13.818 | 307 |
| 0.184 | 0.3560 | 0.1360 | 0.2132 | 0.334 | 11.227 | 250 |
| **0.368** | **0.3850** | **0.1839** | **0.2634** | **0.253** | **7.773** | **174** |
| 0.552 | 0.3299 | 0.1371 | 0.2145 | 0.309 | 11.136 | 248 |

Half the physical skew and 1.5× it are both clearly worse, on every column —
bearing residuals and scoring, two unrelated measurements, agree that the skew
is ~0.37 s. That is what the live timestamp match recovers automatically.

### Limits of the live fix

Matching handles only what the timestamps expose. It aligns *rotation* through
the pose, which is the dominant error; it does not correct a genuine translation
skew (the scan is map-frame and largely pose-independent, so this is minor).
And its resolution is the pose stream's period — it snaps to the nearest
recorded pose, so a residual up to half an inter-pose interval remains. Both are
well below the ~17° the skew was costing.

## Reproducing

    uv run python perception_benchmark/verify_projection.py --scene arabic_room
    uv run python perception_benchmark/verify_projection.py --skew 0.368      # simulate the cache match
    uv run python perception_benchmark/verify_projection.py --old-extrinsic   # reproduce the extrinsic bug

~90 s per run on 316 frames.

## Related

- TASK 24 — flagged that `LatestCache.snapshot()` did no timestamp sync. This
  measured what it cost and closed it.
- TASK 26 — the oversized fused nodes this helps explain.
- TASK 25 — the seam merge; unaffected, since the seam geometry is verified
  exact by check 1.
