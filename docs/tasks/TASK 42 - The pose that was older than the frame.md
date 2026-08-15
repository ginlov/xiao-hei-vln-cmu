# TASK 42 — The pose that was older than the frame

PR #28 (`bd04416`, Long Giang Vu) fixed two things on the perception side. This
task is the audit of whether the drive loop has either.

## The extrinsic: we already have it, and we use it

`_SENSOR_TO_CAMERA_TRANSLATION` was the ROS transform pasted in unrotated,
displacing the camera 0.1 m *forward* instead of 0.1 m *up*; corrected to
`(0, 0.1, 0)`. `bd04416` is an ancestor of this branch, the value in
`src/xiao_hei_vln/perception/geometry.py` is the fixed one, and
`scripts/vlm_approach.py` and `scripts/vlm_locate.py` both call
`sensor_to_camera_transform()` rather than keeping a copy. Nothing to do.

## The timestamp match: the same defect, a much smaller exposure

Their `LatestCache` took latest-per-topic with no timestamp match, so a mask was
lifted against a pose the robot had already turned past — up to 17° of azimuth
while turning.

`scripts/robot_io.py`'s `Capture` never matched stamps either. Each callback
kept the first message on its own topic:

```python
def _on_img(self, m):
    if self.img is None: ...
def _on_pose(self, m):
    if self.pose is not None: return
```

`/state_estimation` publishes at 100-200 Hz and `/camera/image` at ~9.3 Hz, so
the pose landed within milliseconds of subscribing and the frame up to a full
**107 ms** camera period later. Every bearing in the loop is computed by mapping
pixels through that pose.

Three measurements bound what it was worth:

- **The loop captures after a drive returns, not during one.** Of 431 recorded
  drive→capture transitions, the 106 that ended `settled` (the vehicle had not
  moved for 4 s) drifted **exactly 0.000 m**. The 319 that ended `arrived` —
  which returns as soon as the vehicle is inside `ARRIVE_TOL_M`, still coasting
  — drifted a median 0.075 m and at most 0.277 m, over a gap that includes a
  `docker exec` and ROS discovery, so of order a second. Scaled to 107 ms that
  is ~0.008 m.
- **The platform does not snap to a heading after arriving.** We publish
  `theta = 0.0` on `/way_point_with_heading` throughout, so a forced rotation
  at capture time was the thing to rule out. Yaw at capture is spread over the
  whole circle across 585 steps, 23% within 10° of zero — no snap.
- **No systematic skew is visible against ground truth.** Over 54 calls whose
  box is on the right object, in four scenes with a scene graph, the azimuth
  residual between the box's bearing and the true bearing is

  | | |
  |---|---|
  | mean | **−0.11° ± 0.33 (95% CI)** |
  | median | −0.04° |
  | sd | 1.23°, \|residual\| p95 1.68° |

  Statistically indistinguishable from zero, against the 17° their 2 Hz
  responder was losing while driving.

So this is hardening, not a repair, and it is written down that way.

## What changed

A ring buffer keyed on `header.stamp` is what the perception side needed, with
several frames in flight. Here one frame is wanted and only the ordering is
wrong, so the fix is one line: the image callback discards any pose that
arrived before it.

```python
self.img = True
self.pose = None      # take the next one instead
```

The image is the slow topic, so "the first pose after the frame" is within a
*pose* period — 5-10 ms — instead of within a camera period. `done()` then
holds the capture open for that one extra message, well inside its 20 s budget.

`/registered_scan` and `/terrain_map` are deliberately left alone: both arrive
already in the map frame, so a stale one is stale geometry rather than
misregistered geometry, and `noDecayDis` gives the terrain a 1.75 m memory in
any case.

## Testing a module that cannot be imported

`robot_io.py` runs inside `iros2026_system`, the only place with ROS on its
path, so no test had ever imported it. The ROS surface it touches is six names
whose behaviour is irrelevant here, so `tests/test_robot_io_capture.py` stubs
them into `sys.modules` and imports the real module on top — the shipped
callbacks are what run, not a copy of them.

Six tests, including the two failure modes the one-line change could introduce:
that a later pose must not keep clearing itself (or `done()` never fires and
every capture times out), and that clearing the pose must not make the frame
re-writable.

640 tests pass, 6 new.

## Not measured

Whether the residual moves. At −0.11° ± 0.33 there is no room for it to
improve visibly, which is the same reason the change is cheap: it removes a
mechanism whose cost is currently below the noise, before a future change —
capturing while moving, a faster tick — puts it above.
