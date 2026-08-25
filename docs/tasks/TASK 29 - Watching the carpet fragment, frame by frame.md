# TASK 29 — Watching the carpet fragment, frame by frame

## Purpose

Before fixing B6 (large-object fragmentation) we need to *see* it happen, not
just read the final count. The end-of-run number is stark — **2 ground-truth
carpets in `arabic_room` end as 28 separate nodes** (31 node ids ever spawned)
— but a number does not say *when* each spurious node appeared, *where*, or
whether it was a merge the map declined or a lift that landed somewhere wrong.
This task builds a playback that answers those, so the fix is aimed at the real
mechanism.

## What was built

`perception_benchmark/merge_video.py` — a two-panel MP4 animator over any
`dump_debug.py` directory (no sidecar, no GPU):

- **Top panel:** the panorama detection overlay (`vp_NNN.png`) for that
  viewpoint, so the 2D evidence is visible alongside the map.
- **Bottom panel:** a top-down (x-y) accumulation of every lift of the chosen
  class so far, **coloured by the `node_id` it fused into** — the id the
  offline `ObjectMap` already writes into `viz.json`. Same colour = the map
  merged those observations; a physical carpet drawn in six colours *is* the
  fragmentation. Current-frame lifts are ringed, node AABB footprints drawn
  (freshly spawned ones bold), GT footprints dashed, and every newly spawned
  node is joined by a dotted line to its nearest same-class neighbour with the
  gap in metres — a small gap is a merge that should have happened and did not.

Run:

```
uv run --with imageio --with imageio-ffmpeg python \
    perception_benchmark/merge_video.py --scene arabic_room --label carpet \
    --run perception_benchmark/debug
```

Output: `perception_benchmark/videos/arabic_room_debug_carpet_merge.mp4`
(316 frames, ~53 s at 6 fps, MP4 so it scrubs).

## What it shows

- The carpet node count climbs monotonically — 1 → 4 → 10 → 17 → 27 → 28 —
  and never falls: once a fragment spawns, nothing ever merges it away.
- Only two nodes carry the mass (**#409, 108 obs** and **#17, 47 obs**), sitting
  on the two GT footprints; the other ~26 are thin (1–11 obs) fragments
  scattered *across* the same footprints, many with grossly inflated AABBs that
  overlap their neighbours — so the failure is not "lift landed far away", it is
  the map declining to fuse overlapping same-label boxes.
- Fresh nodes routinely spawn a few tens of centimetres from an existing carpet
  node (the dotted-gap annotation), i.e. inside `MERGE_DIST` territory yet not
  merged — the thread to pull on next.

## Scope / honest limits

This visualises the **cross-frame `ObjectMap` fusion** (via `node_id`), which
is the dominant fragmentation source here. The **panorama-seam merge**
(`merge_seam_duplicates`, TASK 25) runs *pre-lift inside a single frame* and is
not recorded in `viz.json`, so it cannot be drawn from the current dump;
visualising it needs a re-dump that logs each frame's union-find groups. Left
as a follow-up.
