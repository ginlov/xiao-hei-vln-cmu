# Runbook — stopping, starting, and driving the loop

Operating instructions for `scripts/approach_loop.py` against the simulator on
`xiaohei1`. Everything here was run on 2026-08-06; the failure notes are things
that actually happened that day, not hypotheticals.

The split to keep in mind: **the API key never leaves the laptop.** The sim host
runs only ROS. `approach_loop.py` runs locally and reaches the container over
`ssh` + `docker exec`, so the server needs no key, no `uv`, and no checkout of
this branch.

## The three containers

| container | what it is | needed by the loop |
|---|---|---|
| `iros2026_system` | Unity simulation + the organisers' base autonomy | **must be up** |
| `xiao_hei_ai_module` | our own responder | **must be stopped** — it publishes to `/way_point_with_heading` and will fight the loop for control |
| `xiao_hei_perception` | YOLO-World / SAM sidecar | not used; the loop grounds through the VLM |

---

## 1 & 2. Stopping and starting: `scripts/sim.sh`

Run it from the laptop; it does everything over ssh.

```bash
./scripts/sim.sh down                  # everything, sidecar included
./scripts/sim.sh up                    # reuses whatever scene is mounted
./scripts/sim.sh up loft               # a different scene — unpacks the zip if needed
./scripts/sim.sh restart               # down + up: this is how you reset the pose
./scripts/sim.sh status                # containers, whether topics flow, pose, scene
./scripts/sim.sh scenes                # unpacked vs still zipped
```

`up` **blocks until `/terrain_map` is publishing**, so a script can chain
straight into a run rather than sleeping and hoping. Measured on `xiaohei1`:
`down` ~11 s, `up` ~12 s, `restart` ~24 s.

```
$ ./scripts/sim.sh status
  iros2026_system	Up 11 seconds
  simulation: publishing
  pose: x:0.0 y:0.0 z:0.75
  scene: japanese_room
```

Host and default scene come from `XIAO_HEI_SIM_HOST` and `XIAO_HEI_SIM_SCENE`.

### Resetting the robot's pose

There is no reset command — **restarting the scene is the reset**, so
`./scripts/sim.sh restart`. Two runs from different starting poses are not
comparable, so do this between runs.

### Scenes

Three are unpacked; the other fifteen are zipped in the same directory.
`sim.sh up <scene>` unpacks on demand, so `./scripts/sim.sh up loft` just works.
`./scripts/sim.sh scenes` lists both sets.

### What the script is doing, and why each part matters

If you need to run it by hand, or to debug it:

```bash
SCENE=japanese_room
ssh xiaohei1 "export DISPLAY=:0; xhost +local: >/dev/null 2>&1
  cd ~/workspace/chengkai/xiao-hei-vln-cmu
  XIAO_HEI_SCENE_DIR_HOST=/home/ubuntu/workspace/dataset/unity-scene/\$SCENE docker/run dummy up -d
  sleep 3
  docker stop xiao_hei_ai_module
  docker exec -d iros2026_system bash -lc 'cd ~/autonomy_stack_mecanum_wheel_platform && DISPLAY=:0 ./system_simulation.sh > /tmp/sim.log 2>&1'"
```

- **`docker/run dummy up -d` only starts the container — it does not start the
  simulation.** Without the `docker exec` line the container is up and not one
  topic is published.
- **`DISPLAY=:0` and `xhost +local:` are both required.** Unity needs a display;
  without one `/terrain_map_ext` never publishes and the symptom is a stack that
  looks healthy and answers nothing.
- **Do not skip `docker stop xiao_hei_ai_module`.** The loop's preflight catches
  it, but only after a start-up you then have to redo.
- **`docker/run down` leaves the sidecar running** — it takes no responder
  argument, so it carries no `--profile perception`. `docker stop
  xiao_hei_perception` is a separate line.

---

## 3. Run the loop

From the laptop, in the repo root:

```bash
cd ~/Workspace/vln-challenge/xiao-hei-vln-cmu
eval "$(grep -E '^[[:space:]]*export ANTHROPIC_API_KEY=' ~/.zshrc)"

uv run --with anthropic python scripts/approach_loop.py \
  "lantern closest to the fan decoration" \
  --host xiaohei1 --out runs/jp5
```

Two lines that are not optional:

- **`--with anthropic`.** The package is not in `pyproject.toml`; without the
  flag this is a `ModuleNotFoundError` at the first grounding call, after the
  preflight has already passed.
- **The `eval` line.** zsh only sources `~/.zshrc` for interactive shells, so a
  tool-spawned shell has no key. This form pulls that one line and never prints
  the value.

### Flags

| flag | default | what it does |
|---|---|---|
| `--dry-run` | — | ground once, print the waypoint, drive nothing |
| `--max-steps` | 6 | ceiling on grounding calls; each is ≈ $0.0265 |
| `--prompt-version` | `v4-relational` | `v3-occlusion-distance` to compare against the pre-relational prompt |
| `--standoff` | 0.6 | mostly superseded by the converter model choosing a legal point |
| `--backend` | `claude` | `gemini` needs `XIAO_HEI_GEMINI_API_KEY` in `.env`, which does not exist yet |
| `--host` | — | omit to run inside the container instead of over ssh |

### What you get

`runs/<name>/`:

| file | contents |
|---|---|
| `steps.jsonl` | one row per step: `pose`, `reply`, `relation`, `waypoint`, `converter` (the predicted settle point), `drive` |
| `step<N>_face{0..3}.jpg` | the four faces sent to the model that step |
| `step<N>_target.jpg` | the chosen box, cropped — fed to the next call as continuity |

A run ends `ARRIVED` when the vehicle reached the point it asked for, or when
the converter model says no legal point is any closer. `did not arrive` means
it stopped for another reason, and `steps.jsonl` says which.

---

## Probing one phrase without driving

```bash
uv run --with anthropic python scripts/vlm_probe.py \
  snaps/starts/office_1 "potted plant furthest from the projector screen" \
  --scan-from frames_first/office_1_tour
```

**On a comparative phrase — *closest*, *nearest*, *farthest*, *between* — do
not read `box_2d` as the answer.** It is the model's nomination, and the loop
discards it: `resolve_relation` lifts every candidate and the anchor and does
the comparison itself. Judging the pipeline from the probe's JSON means judging
a step that never runs.

`--scan-from` points at a directory that has a pose and a scan — either a
`scripts/snap.sh` directory (`pose.json` + `scan.npy`) or a recorded tour
(`frames_first/<scene>_tour/`). Faces alone, as `scripts/grab_faces.py` writes
them, carry neither. With it, the probe reports what the loop would pick:

```
-> left (heading 270°), confidence 0.6
   box_2d           yaw  +47.68°  pitch +17.79°  [pixels]

!! farthest_from: the box above is the model's nomination, and the loop does not use it.
   It lifts every candidate and measures.
   -> OVERRIDES it: image 0 (front) box [283, 252, 352, 292]
   farthest_from over 4 lifted candidate(s) vs 1 anchor(s): small plant on bookshelf
   shelf=4.75m, plant in blue pot on filing cabinet=5.81m, ... , tall floor plant in
   white pot near the wooden door=10.46m
```

That example is office_1's own q4. The model nominated the filing-cabinet plant
twice across two runs; geometry overrode it both times and landed 0.48 m from
the ground-truth answer, measuring 10.46 m against a true 10.22 m.

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `ssh: connect ... timed out` | `xiaohei1`'s public IP changes on **every** instance restart | update `HostName` in `~/.ssh/config` |
| `ModuleNotFoundError: No module named 'anthropic'` | missing `--with anthropic` | see above |
| `anthropic.AuthenticationError` | the `eval` line was not run in this shell | see above |
| preflight reports `rival_waypoint_publishers` | `xiao_hei_ai_module` is up | `ssh xiaohei1 'docker stop xiao_hei_ai_module'` |
| preflight reports nothing on `/joy` | the local planner discards every waypoint silently | the sim did not fully start; check `docker exec iros2026_system cat /tmp/sim.log` |
| `bridge produced no JSON` | `iros2026_system` is not running | §2 |
| no `/terrain_map`, robot never moves | Unity has no display | `xhost +local:` and `DISPLAY=:0`, then restart the scene |
| GPU rendering dead after an EC2 restart | modeset grabs the Amazon VGA and unloads nvidia | already pinned in `xorg.conf`; if the DCV session is gone, `sudo systemctl restart dcv-autosession` |
| `529 OverloadedError` | API under sustained load | `max_retries` is already 8; raise with `XIAO_HEI_API_MAX_RETRIES` |
| robot wedged, moves 0.02 m and stops | it is against the obstacle inflation | `./scripts/sim.sh restart`; the converter model now predicts this before driving |
| `sim.sh up` times out after 120 s | Unity failed to start | `ssh xiaohei1 'docker exec iros2026_system cat /tmp/sim.log'` |

## Reading the outcome offline

`scripts/waypoint_converter_model.py` needs no sim — give it a captured
`/terrain_map` and it answers where a waypoint would put the robot:

```python
from waypoint_converter_model import ConverterModel
cm = ConverterModel(np.load("loop_terrain.npy"))     # (N, 4) x y z intensity
cm.settle(waypoint_xy, vehicle_xy)                   # where it ends up
cm.legal_points()                                    # everywhere it would stand
```

Predicted within ~0.1 m on three drives. See
[TASK 29](../tasks/TASK%2029%20-%20The%20converter%20is%20not%20a%20clamp,%20and%20we%20can%20predict%20it.md)
and the [drive strategy](../vlm_drive_strategy.md).
