# TASK 52 — Porting the drive loop into the submission's `ai_module`

The instruction-following stack worked and could not be submitted. It ran the
wrong way round: a laptop drove it, reaching into the *system* container over
`ssh` and `docker exec` to snapshot topics and publish waypoints. A submission
is the opposite shape — a ROS 2 node inside `iros2026_ai_module`, started by the
graders, given one question and left alone.

This task is that inversion, and nothing else. No behaviour of the loop changed.

## What the README actually requires

Three sentences did most of the design work.

> "The system will be relaunched for each language command tested such that
> information collected from previously exploring the scene is not retained."

One question per process. There is no loop over questions, no state to carry,
and no reason for the CLI's `--budget`-from-first-call shape.

> "Timing will begin immediately at system startup."

The ten-minute clock starts before the module has read anything. `execute_plan`'s
CLI starts its clock *after* the decompose call returns, which would have quietly
gifted us the API round trip and the DDS discovery wait — 5–10 s of a 540 s
budget, taken from the end where it is worth most.

> "While more topics may be available from the system, these are the only ones
> allowed to be used during test time."

`/camera/image/compressed` is not on that list and `/camera/image` is. Neither is
`/way_point_reached`, which the loop already did not branch on.

## The seam

The port is small because `Robot` was already an interface: four methods, six
call sites, and a `capture()` that returns exactly what the loop wants —
`(equirect BGR, scan Nx4 in the map frame, terrain Nx4, pose dict)`.

`vlm/robot_node.py` implements the same four methods against live subscriptions.
`approach_loop`, `execute_plan`, `vlm_approach`, `vlm_locate`, `vlm_probe`,
`waypoint_converter_model`, `faces` and `instruction_plan` are byte-identical to
the versions that drove the 0814 runs. A check in `sync_ai_module.sh --check`
asserts that and fails if the two copies drift.

`robot_io.py`'s `Capture` and `Driver` merge into one node, because a persistent
node cannot create and destroy a subscription per call the way a one-shot process
could. Everything they were protecting against is carried over: the pose that
must postdate the image, the turn-in-place stall test, the distance-subsampled
track that a passage constraint is checked against.

One QoS change was forced by the merge. `robot_io` used `qos_profile_sensor_data`
(depth 5) and got away with it because each capture built a fresh node. A node
that lives across a 35 s model call does not: KEEP_LAST depth 5 would hold five
samples and hand back the oldest first, so the "snapshot" would open on a frame
up to half a second stale, and every bearing in the step rides on the image/pose
pairing. depth 1 keeps only the newest, which is what a snapshot means. The other
pipeline measured 1.2 s of exactly this lag on `/camera/image` before changing it.

## Layout, and why it is a directory of loose files

The rule is that only `ai_module/` may change, so the stack has to live there
rather than be imported from the development repo. Rather than rewrite twelve
tuned files' imports, `ai_module/vlm/` mirrors the *shape* of this repo —
`scripts/`, `perception/`, `src/` in the same relative positions — so every
`sys.path.insert(..., parent.parent / "perception")` already in them resolves
unchanged. Zero import edits, and each file stays runnable on its own inside the
container, which is where a failed graded run gets debugged.

The one deliberate deviation is `src/xiao_hei_vln/perception/__init__.py`.
Upstream it re-exports `PerceptionResponder`, which imports the sidecar client
and would drag httpx, pillow and pycocotools into the image to satisfy an import
of `sensor_to_camera_transform`. The vendored copy is an empty namespace stub and
`sync_ai_module.sh` does not overwrite it.

## Package naming

The package is still `dummy_vlm`, the launch file still `dummy_vlm.launch`, the
executable still `dummyVLM` — now a Python entry point installed with
`install(PROGRAMS ... RENAME dummyVLM)`. The README says to integrate a model by
modifying "the system startup script", and that script is not in the repository.
If it invokes `ros2 launch dummy_vlm dummy_vlm.launch`, as the docker README's
own instructions do, we still start. A tidier name risks never being launched.

## Two bugs found by moving the code

**`decompose` wrote its cache unguarded, to a relative path, on the answer path.**
`CACHE = Path("artifacts/decompose_cache.json")` was written immediately after the
API call that produced the entry. Inside the container the working directory is
whatever `ros2 launch` chose and need not be writable — so an unwritable
`artifacts/` would have raised at step 0, after the call was paid for and before
a single waypoint was published, on a cache that can only ever miss on a held-out
scene. Now best-effort, with the path overridable; `challenge_node` points it into
the run directory so the model's raw reply is kept as a record of how the sentence
was split.

**Nothing caught an exception at the top of a run.** A `TypeError` reaching the
top of a live run killed a whole question two legs in during the 0814 sessions.
Scoring is per constraint with partial credit on the trajectory *actually driven*,
so the waypoints already published still count — letting the exception out turns a
partial score into a zero. `main` now logs the traceback whole and idles.

## Image

`python3-opencv` and `python3-scipy` come from apt, not pip, and that is not
style. The base image ships numpy 1.26 from apt; pip resolving either package
pulls a numpy 2.x wheel over it, and the apt-built `cv2` is compiled against
numpy 1.x — it then fails to import with `numpy.core.multiarray failed to
import`, which reads as an opencv problem and is a dependency-resolution one.
`anthropic` is the only pip install and pulls nothing numpy-shaped.

No GPU is needed at any point, which suits the NUC the graders evaluate on.

## What is deliberately not in this task

`/terrain_map_ext` is subscribed and unused — `ConverterModel` still models the
5 m map. `dominant_cluster` still takes the biggest cluster rather than the
nearest credible one (measured at 4–6 m: 0.40 m → 0.16 m). `size_gate` still
judges a feature box against the target's size prior. The Claude classifier is
written but defaults to a stub that routes everything to instruction-following.
Each is a behaviour change that wants its own regression run against a scene we
have logs for, and this task is the inversion only.

## Verifying

`--selftest` on the node captures one frame from every subscribed topic, prints
their shapes, drives two metres ahead and reports whether the vehicle moved — no
key, no model call, no cost. It exists because receive-question through
publish-waypoint through the-vehicle-moved is the only genuinely new path; the
loop above it has been driven for weeks.

The one measurement that must be taken on the box before trusting the port is
`ros2 topic echo /camera/image --field encoding --once`. `_image_to_bgr` refuses
an encoding it does not know rather than guessing, because a wrong channel order
does not crash — it silently shows the model wrong colours, and the questions
name colours.

## Driven, 2026-08-15

That measurement came back **`bgr8`**, 1920x640, on both xiaohei1 and xiaohei2.
So the move off `/camera/image/compressed` is a pure passthrough: the same array
the compressed path produced, minus the JPEG round trip. The risk this section
was written to guard against did not materialise, and the guard stays anyway —
the real robot is a different camera.

Two runs through the graded path itself — `ros2 launch dummy_vlm
dummy_vlm.launch` in the built image, question delivered by `ros2 topic pub
/challenge_question` — on `japanese_room`:

| Question | Result | Calls | Cost | Time |
|---|---|---|---|---|
| "Go near the small table with a vase on it and then to the flowers near the jar." | 1/2 — leg 1 "circling with nothing bound" | 10 | $0.27 | 239 s |
| "Go to the lantern closest to the fan decoration, then take the path near the wardrobe doors to the flowers on the display ledge." | **3/3** — GOTO, PASS, GOTO all satisfied | 9 | $0.24 | 269 s |

Both well inside the 540 s budget. Artefacts are under `runs/aimodule_*`. The
`aim_box` range-agreement gate from TASK 43 fired live in the second run —
*"feature box ignored: it lifts to 1.77 m against the target box's 2.51 m"* —
which is the first time that gate has been observed refusing a box outside a
replay.

`scripts/ai_module.sh` was added alongside `sync_ai_module.sh` to make this
repeatable: `sync` / `build` / `selftest` / `ask "<question>"` / `pull` / `down`
against either box.
