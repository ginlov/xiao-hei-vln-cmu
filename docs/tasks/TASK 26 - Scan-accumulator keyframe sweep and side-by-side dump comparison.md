# TASK 26 — Scan-accumulator keyframe sweep and side-by-side dump comparison

## Why

`ScanAccumulator` keeps the last *k* registered sweeps and lifts detections
against their union. `k` had never been justified — it was 10 because it was
10. Tuning it was impractical: every trial meant re-running the sidecar
(detection ~1 s/frame, ~7 min/scene on `arabic_room`) before anything could be
measured.

## Making the sweep cheap

Detection depends only on **(image, class list, score threshold)**. None of
those is what a keyframe sweep varies, so the masks can be computed once and
replayed. `dump_detections.py` already froze them to `detections.npz` — it did
not need writing, only finding. (A per-run cache was built first and thrown
away; see "Wrong turns".)

`replay_score.py` and `dump_debug.py` gained `--scan-keyframes`,
`--scan-voxel` and `--no-frozen`. With frozen masks present, the sidecar is
never started at all. Live vs frozen produced **byte-identical metrics**, which
is the check that makes the whole shortcut legitimate.

`perception_benchmark/sweep_keyframes.sh` drives it: freeze once, then loop.
`DUMP=1` writes viewer dumps instead of scores.

## The sweep

`DUMP=1 K_VALUES="0 1 2 3 4 5 6 7 8 9 10" sweep_keyframes.sh arabic_room` —
11 dumps, 316 frames each, **~97 min, 3.5 GB**. Detections are constant at 7482
by construction, so every difference is the accumulator.

| k | final nodes | lifts | lift rate |
|---|---|---|---|
| 0 | 214 | 3829 | 51.2% |
| 1 | **194** | 3585 | 47.9% |
| 2 | 252 | 4149 | 55.5% |
| 3 | 276 | 4471 | 59.8% |
| 4 | 284 | 4681 | 62.6% |
| 5 | 300 | 4841 | 64.7% |
| 6 | 316 | 4956 | 66.2% |
| 7 | 315 | 5081 | 67.9% |
| 8 | 329 | 5175 | 69.2% |
| 9 | 323 | 5246 | 70.1% |
| 10 | 327 | 5311 | 71.0% |

**k=0 and k=1 are not adjacent settings.** k=0 disables accumulation and lifts
the raw sweep (~10,619 points); k=1 accumulates one sweep and voxel-downsamples
it to ~5,273. k=1 is therefore the *sparsest* configuration in the sweep, not
the second sparsest, which is why it dips below k=0 on both columns. Read the
curve from k=2 up.

From there the trade is monotonic and unhelpful on its own: more keyframes lift
more detections (51% → 71%) **and** produce more nodes (214 → 327, against 69
GT objects). Denser clouds push marginal detections past `min_inliers`, and
those weak lifts land as new fragments rather than reinforcing existing nodes.
Node count alone cannot say which end is better — that needs scoring, and it
needs eyes.

Cost per k rose 7:04 (k=0) → 10:18 (k=9): the lift scales with the accumulated
cloud, ~5 ms/detection at k=1 against ~76 ms at k=40.

## Comparing the dumps

`viz_app.py` gained a **Compare runs** tab, because the metric table says which
run scores better and not whether the extra nodes are objects or fragments.

- `PERCEPTION_DEBUG_DIR`'s *siblings* are discovered and offered as a sidebar
  "Dump directory" selector, so a sweep is browsable without restarting.
- The compare tab puts N runs side by side at the same viewpoint, sharing the
  sidebar's viewpoint slider and display toggles. Panels can show the
  cumulative 3D graph, the per-viewpoint graph, or the overlay image.
- A summary table covers **every** run (`run_summary` reads scalars only and
  is `persist="disk"` cached — the full table would otherwise hold ~130 MB of
  `viz.json` and re-parse it on each app start).
- A cumulative-node-count chart per run, against the GT line.

A sidebar **class filter** restricts phases 3 and 4 — both 3D panels, the
inspect picker, the lift-input list, the detection table, and the compare
panels — to chosen labels; empty means all. Counts read "1 of 6 nodes" so a
filtered view can never be mistaken for the whole graph. Phases 1–2 are
unfiltered: the overlay is a PNG the dump already baked. The accumulated point
cloud is unfiltered too, and says so — lifted points carry no label once merged.

The **viewpoint navigator** replaces the bare slider: first/prev/next/last
steppers, a "jump to vp #" box, and **◀ new node / new node ▶**, which skip to
the next viewpoint where the cumulative graph *gained* a node of the selected
classes. That last pair is the point of it — `arabic_room` is stationary for
two thirds of its 316 frames, so most viewpoints show nothing new and finding
the ones that do by dragging a slider is hopeless. Filtered to `potted plant`,
54 of 316 frames are growth frames.

The index lives in `session_state`; every button writes it from an `on_click`
callback, the only place Streamlit permits assigning a widget-backed key. It is
clamped whenever the run or scene changes, because a shorter capture would make
the slider raise before anything rendered.

One trap in the lift-input picker: its options are the **original** detection
indices, because `sel` indexes into the npz point arrays (`inmask_det == sel`).
Renumbering a filtered list there would select a different detection's points
and quietly show the wrong cloud. When a filter matches nothing at a viewpoint
the list degrades to all classes with a notice rather than calling `st.stop()`,
which would abandon the rest of the page including the other tabs.

Two details that matter for it being an honest comparison:

**Shared axis ranges.** With `aspectmode="data"` each panel scales to its own
extent, so a run with one stray distant node silently shrinks the room and the
panels stop being comparable by eye. Ranges are computed once from the GT
extent and the robot path — identical across runs — and applied to all panels.

**Camera presets, not sync.** Separate plotly figures cannot share a camera
without custom JS; orbiting one does not move the others. A preset selector
(isometric / top-down / free) puts them all in the same pose instead, with
`uirevision` keyed on the preset so stepping the viewpoint slider does not
throw away an orbit the user set up.

Node ids are **not** comparable between panels — they are per-run counters.
Labels and positions are.

## Animating one class over time

`perception_benchmark/make_video.py` renders the cumulative graph frame by
frame, filtered to a class. The viewer answers "what does the graph look like
at viewpoint N"; this answers "how did it get there", which is the question
fragmentation actually raises — 54 `potted plant` nodes for 5 real plants were
not found at once.

    uv run --with imageio --with imageio-ffmpeg python \
        perception_benchmark/make_video.py \
        --scene arabic_room --labels "potted plant" --run perception_benchmark/debug_k10

Reads a dump directory only (no sidecar, no GPU); ~37 s for 316 frames.
Output goes to `perception_benchmark/videos/` (gitignored).

**MP4, not GIF.** 316 viewpoints is far too long to watch without a scrubber,
a pause and a speed control, and a GIF has none of them. It is also 10× the
file: the same animation is 7.9 MB as a 159-frame GIF and 0.8 MB as a
316-frame h264. `--format gif` remains for pasting somewhere that will not take
a video. There is no ffmpeg on the benchmark box, so the encoder comes from the
`imageio-ffmpeg` wheel, which ships its own binary — hence the `--with` flags,
matching how the viewer already pulls streamlit and plotly.

`viz_app.py` lists any clip for the current scene in an **Animations** expander
and plays MP4s inline with `st.video`, so the animation sits next to the graph
it animates.

The stream is written frame by frame straight off the matplotlib canvas — no
PNG round-trip, no frame buffer held in memory. h264 needs `yuv420p` for
anything to play it in a browser, which needs even dimensions; frames are
cropped by a pixel rather than rescaled, so the plot is never resampled.

Nodes that first appeared within the last `--new-window` viewpoints are drawn
**gold**, everything older fades to thin crimson. Without that the final frames
are an undifferentiated pile of overlapping wireframes and nothing about the
*growth* is visible. Node ids are stable in the cumulative map, so first
appearance is just the earliest viewpoint whose `nodes` list contains the id.

Three rendering details were needed for it to be watchable:

- **No `bbox_inches="tight"`.** It crops to drawn content, so the frame size
  changes as boxes appear — a jittering GIF, and an encoder error for MP4,
  which needs one constant resolution.
- **`set_box_aspect` from the data ranges.** Matplotlib's 3D axes default to a
  cube, which draws 1 m of room height as long as 9 m of floor — every box
  reads as a tall column. `--z-scale` (default 2) exaggerates height on top of
  the true ratio, because a true-aspect room is nearly flat on screen.
- **The frame counter says "54 of 327 nodes", not "54".** With dozens of
  overlapping boxes a filtered frame looks exactly like an unfiltered one —
  it was read as "you are drawing every class" on first review, and the
  denominator is the only thing that says otherwise. The title is a
  `fig.suptitle`, not `ax.set_title`, because a 3D projection overflows its
  axes rect and the pane edges draw *over* an axes-level title (an opaque
  bbox loses too — it lives inside the same axes). Figure artists draw last.
- **Fixed limits for the whole animation**, computed once from GT extent, the
  robot path and the final node set. Per-frame limits make the room breathe as
  nodes appear.

What it shows on `arabic_room`: plants spawn steadily to ~50 nodes by vp_200
and then plateau, and several of the new nodes are **enormous** — the largest
has a 5.4 m diagonal, against a real plant's ~0.8 m. Half of the 54 are
oversized boxes (median diagonal 1.46 m, 90th percentile 3.51 m), so the
fragmentation is not only duplicate plants but mask spill fusing a plant with
whatever is behind it. That is a `nms_gap`/lift problem, not a keyframe one.

## Wrong turns

A `_DetectionCache` was written from scratch before checking whether the repo
already froze masks. It did, in `dump_detections.py`, with two existing
consumers. Deleted.

The first read of the frozen file used `allow_pickle=False` and crashed:
`labels` is `dtype=object`. Both existing consumers already passed
`allow_pickle=True`. Caught by a 4-frame probe run before committing to a
40-minute sweep — worth the two minutes.

## Verification

`streamlit.testing.v1.AppTest` drives the app headless: it renders with no
exceptions, exposes all three tabs, defaults the comparison to the selected run
plus its sweep neighbour, and survives stepping the viewpoint slider (which
re-renders every compare panel). Lint is unchanged from HEAD at 10.

## Open

The scored sweep has not been run — only the dumps. `sweep_keyframes.sh
arabic_room` without `DUMP=1` reuses the same frozen masks and would score all
11 in ~35 min. Until then the *qualitative* review is the only evidence about
which k to pick.
