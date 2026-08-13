# Runbook — stopping, starting, and driving the loop

Operating instructions for driving the simulator on `xiaohei1`. Everything here
was run against it; the failure notes are things that actually happened, not
hypotheticals. Sections 1–2 date from 2026-08-06, §3 onward from 2026-08-09.

**Two entry points, and picking the wrong one costs a whole run.**

| script | takes | use it for |
|---|---|---|
| `scripts/execute_plan.py` | a **whole question** | anything from `questions.json` — it decomposes the sentence and walks the clauses in order |
| `scripts/approach_loop.py` | **one object phrase** | a single destination, or debugging one leg in isolation |

`approach_loop.py` treats whatever you hand it as *one object description*. Give
it a three-clause instruction and it will ground the sentence as a single thing,
drive to whichever object it finds most identifiable, and report `ARRIVED` —
having done a third of the question. That has happened; see §3.

**Two places to run them from**, too. Driving from the laptop over `ssh` +
`docker exec` is the default and keeps the API key off the shared box entirely —
the sim host then needs no key, no venv and no checkout. Driving from a terminal
on the box (§3b) needs all three, but survives your laptop closing mid-question.

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

### Choosing the box: `xiaohei1` or `xiaohei2`

**Export it once and everything follows**, including the loop — `sim.sh`,
`drive.sh`, `execute_plan.py` and `approach_loop.py` all read the same
variable, so a scene cannot be restarted on one box and driven on the other:

```bash
export XIAO_HEI_SIM_HOST=xiaohei2
./scripts/sim.sh restart home_building_2
uv run --with anthropic python scripts/execute_plan.py "<question>" --out runs/x
```

`--host` still wins where it is given; it just no longer has to be.

The two boxes keep their scenes in **different places**, so `sim.sh` finds them
rather than assuming:

| | unpacked scenes | zips |
|---|---|---|
| `xiaohei1` | `~/workspace/dataset/unity-scene` | same directory |
| `xiaohei2` | `~/workspace/dataset/unity_scenes_extracted` | `~/workspace/dataset/unity_scenes` |

Override with `XIAO_HEI_SIM_SCENES` / `XIAO_HEI_SIM_ARCHIVE` on a box that
matches neither, and `XIAO_HEI_SIM_REPO` / `XIAO_HEI_SIM_CONTAINER` for the
checkout path and the container name.

**`xiaohei2` is shared** — `~/workspace` has `chengkai`, `long`, `rajath` and
`yuxin` in it, and `xiao_hei_perception` is often running. `sim.sh down`, which
`up` and `restart` both begin with, stops that container. Check `sim.sh status`
and ask before taking the box.

Both hosts' aliases live in `~/.ssh/config`, and **the public IP changes on
every instance restart** — a connection timeout means the `HostName` there is
stale, not that the box is down.

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

Some are unpacked and the rest are zipped; `sim.sh up <scene>` unpacks on demand,
so `./scripts/sim.sh up loft` just works even when only the zip is there. On a
box where the two live in different directories it unpacks across (see the table
above). `./scripts/sim.sh scenes` lists both sets and says which directories it
found.

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

## 3. Drive a whole question

**This is the normal command.** From the laptop, in the repo root:

```bash
cd ~/Workspace/vln-challenge/xiao-hei-vln-cmu
./scripts/sim.sh restart home_building_2          # reset the pose first
eval "$(grep -E '^[[:space:]]*export ANTHROPIC_API_KEY=' ~/.zshrc)"

uv run --with anthropic python scripts/execute_plan.py \
  "Go near the magazine on the ottoman, then go to the potted plant on the dressing table." \
  --host xiaohei1 --out runs/hm2_v6
```

Two lines that are not optional:

- **`--with anthropic`.** The package is not in `pyproject.toml`; without the
  flag this is a `ModuleNotFoundError` at the first grounding call, after the
  preflight has already passed.
- **The `eval` line.** zsh only sources `~/.zshrc` for interactive shells, so a
  tool-spawned shell has no key. This form pulls that one line and never prints
  the value.

### The API key on the laptop

The `eval` line above assumes the key is already in `~/.zshrc`:

```bash
printf '\nexport ANTHROPIC_API_KEY=%s\n' 'sk-ant-...' >> ~/.zshrc
chmod 600 ~/.zshrc
```

(Prefix that with a space, or edit the file, so the key does not land in your
shell history.) On a single-user laptop a dotfile is fine; on the box it is not,
which is why §3b says otherwise.

If you would rather not leave it in plaintext, macOS Keychain works and the
`eval` line becomes a `security` call:

```bash
security add-generic-password -a "$USER" -s anthropic-api-key -w   # prompts, hidden
export ANTHROPIC_API_KEY="$(security find-generic-password -a "$USER" -s anthropic-api-key -w)"
```

Use a different key for the box than for the laptop — the point of two is being
able to revoke one without losing the other.

**Give every run a fresh `--out`.** Re-using a directory overwrites the files
that collide and leaves the ones that do not, so the result is two runs mixed
together with no marker saying which step came from which.

### See the plan without touching the robot

```bash
uv run --with anthropic python scripts/execute_plan.py "<question>" --plan-only
```

Costs one cached model call and no driving. Check the clause order before
spending ten minutes of sim time on it.

### Flags — `execute_plan.py`

| flag | default | what it does |
|---|---|---|
| `--plan-only` | — | decompose and print; touch no robot |
| `--dry-run` | — | ground and compute waypoints, publish nothing |
| `--budget` | 540 | seconds for the **whole question**; README allows 600 |
| `--goto-steps` | 20 | safety cap on grounding calls per destination — the real governor is the leg's share of `--budget` |
| `--model` | `claude-opus-5` | any vision model: `claude-sonnet-5`, `claude-fable-5`, `claude-haiku-4-5-20251001` |
| `--backend` | `claude` | `gemini` reads `XIAO_HEI_GEMINI_API_KEY` (or `GEMINI_API_KEY`); default model `gemini-3.6-flash`. See below — it needs its own `--with` |
| `--prompt-version` | `v6-way-out` | `v5-constraints` is the version the cached replies and the offline scripts are keyed to |
| `--host` | — | omit to run inside the container instead of over ssh |

`--model` does **not** reach the decomposition step, which always uses
`claude-opus-5` — one cached call per question, 3.3 s, 30/30 on drive order.

### Driving on Gemini

```bash
eval "$(grep -E '^[[:space:]]*export (ANTHROPIC|XIAO_HEI_GEMINI)_API_KEY=' ~/.zshrc)"

uv run --with anthropic --with google-genai python scripts/execute_plan.py \
  "<question>" --backend gemini --model gemini-3.1-pro-preview \
  --host xiaohei1 --out runs/x
```

**Both** `--with` flags: decomposition always calls Claude whatever `--backend`
says, so dropping `--with anthropic` fails before the first Gemini call.

Which models are callable is a property of the key's billing tier, not of the
catalogue — `models.list` still advertises `gemini-2.5-flash`, which answers
404 "no longer available to new users". Probed on a Tier 1 key:

| model | note |
|---|---|
| `gemini-3.6-flash` | the default; pinned, thinks unasked |
| `gemini-3.1-pro-preview` | frontier tier; **Free tier gives it a daily quota of zero**, so this one needs billing |
| `gemini-3.1-flash-lite` | no thinking, ~2.7 s a call against ~12 s for Pro |
| `gemini-flash-latest` | resolves to `gemini-3.6-flash` **today** — never use it for an A/B |
| `gemini-2.5-*` | 404 for new keys |

Free → Tier 1 is billing linked in AI Studio (not Cloud Console) and takes
effect immediately. At the measured 6394 input tokens a grounding call, Pro
costs about $0.031 a call, $0.37 a run, $11 for all thirty released questions.

Two things differ from Claude and are handled in code, not by you:

- **`coord_space` is not taken on trust.** Gemini emits 0-1000 normalised
  boxes; `gemini-3.1-pro-preview` sometimes declares them `"pixels"`. Read as
  declared, the box lands on bare wall. `settle_coord_space` corrects it in
  `ground()`. See TASK 40.
- **`XIAO_HEI_GEMINI_MAX_TOKENS`** (default 8192) caps output *including*
  thinking, which the 3.x models spend without being asked — 676 tokens of
  thinking against 251 of answer on one Pro call. Truncation now raises
  instead of arriving as "unparseable reply".

### One destination only

```bash
uv run --with anthropic python scripts/approach_loop.py \
  "lantern closest to the fan decoration" \
  --host xiaohei1 --out runs/jp5
```

Same flags minus the plan ones; `--max-steps` (default 6) replaces
`--goto-steps`. **Do not hand this a multi-clause instruction.** Given
`home_building_1` q5 it ground the whole sentence as one object, drove straight
to the trash can in the last clause, skipped the bedroom and the passage
entirely, and reported `ARRIVED` in 5 calls for $0.13 — a third of the question,
scored as if it were the answer.

### What you get

`runs/<name>/`:

| file | contents |
|---|---|
| `plan.json` | the decomposed clauses and one result row each — written **at the end**, so its absence means the run is still going or died |
| `steps.jsonl` | one row per step: `pose`, `reply`, `relation`, `waypoint`, `converter` (the predicted settle point), `drive` |
| `step<N>_face{0..3}.jpg` | the four faces sent to the model that step |
| `step<N>_target.jpg` | the chosen box, cropped — fed to the next call as continuity |
| `step<N>_{scan,terrain}.npy` | the geometry, so a waypoint can be re-derived after the fact |

A leg ends `arrived` when the vehicle reached the point it asked for, or when
the converter model says no legal point is any closer. **A failed leg does not
end the run** — scoring is per-constraint with partial credit, and the next
destination is still reachable from wherever the robot now stands.

`plan.json`'s `xy` is the **bound target**, not where the robot parked. To answer
"did it actually get there", you need both that and the last `pose` in
`steps.jsonl`.

---

## 3b. Running from a terminal on the box instead

Worth doing when **a run must outlive your laptop**: a question is up to ten
minutes at ~30 s per step, and a dropped ssh leaves the robot parked wherever it
got to and the log half-written.

One-off, on the box:

```bash
ssh xiaohei1
cd ~/workspace/chengkai/vlm-drive     # see "getting the branch there"
./scripts/on_host.sh setup            # builds .venv-drive; needs no root
```

Then it is two exports and the normal commands:

```bash
tmux new -s drive
export XIAO_HEI_SIM_HOST=local        # this machine is the sim box
export ANTHROPIC_API_KEY=sk-ant-...

./scripts/sim.sh restart home_building_2
.venv-drive/bin/python scripts/execute_plan.py "<question>" --out runs/x
# ctrl-b d to detach; `tmux attach -t drive` to come back
```

`./scripts/on_host.sh check` says what is missing if something is.

**Why the venv, and why `setup` is the only part with a script.** The system
python on these boxes has neither pip nor the venv module — `python3 -m venv`
fails asking for `python3-venv`, which needs root — so `uv` is the only way to
install anything, and `.venv-drive/bin/python` is the only interpreter that can
run the loop. Everything else is a plain command.

It installs six things: `numpy`, `opencv-python-headless`, `anthropic`,
`pillow`, `pydantic`, `scipy`. No ROS — `robot_io.py` is copied *into* the
container and run there.

**`XIAO_HEI_SIM_HOST=local`** is what makes `sim.sh`, `drive.sh` and the loop
talk to the local docker instead of ssh-ing somewhere. Forget it and they try to
ssh to `xiaohei1` from `xiaohei1`.

**The key** is a plain export, and on a shared box that is the right default —
it dies with the shell. If you tire of retyping it, put it in a file only you
can read, **not `~/.bashrc`**, which leaks it into every process you start:

```bash
mkdir -p ~/.config/xiao-hei && chmod 700 ~/.config/xiao-hei
( umask 077; printf 'export ANTHROPIC_API_KEY=%s\n' 'sk-ant-...' > ~/.config/xiao-hei/env )
source ~/.config/xiao-hei/env
```

Use a key you can **revoke separately from your laptop's**: `~/workspace` on
these boxes holds four people's directories.

**Getting the branch there.** xiaohei1 can `git fetch` (ssh remote); xiaohei2
cannot (https remote, no credentials — send a `git bundle`). Do **not** switch
the branch of `~/workspace/chengkai/xiao-hei-vln-cmu`; both boxes have other
people's modified files in it. Add a worktree, which touches nothing:

```bash
cd ~/workspace/chengkai/xiao-hei-vln-cmu
git fetch origin feature/vlm-approach-loop
git worktree add ~/workspace/chengkai/vlm-drive origin/feature/vlm-approach-loop
```


## 4. Read a run back

```bash
uv run python scripts/show_run.py runs/hm2_v6              # whole run
uv run python scripts/show_run.py runs/hm2_v6 --leg 1      # one clause
uv run python scripts/show_run.py runs/hm2_v6 --full       # untruncated reasoning
```

Prints the scorecard, then one block per step: position, whether the target was
visible, where it decided to go, and why. `steps.jsonl` carries several hundred
words per row that the executor never reads; this shows the parts that decided
where the robot went.

Lines worth looking for:

| line | means |
|---|---|
| `way out: '...' lifted to (x, y), N m away` | the v6 branch fired — the model boxed an opening and we drove to it rather than along a bearing |
| `(range capped; bearing kept)` | the lift came back past `WAY_MAX_M`; the scanner saw through the opening into the room after next |
| `heading N° reaches ... best drivable is ±D° off it` | no opening was boxed, so it fell back to a bearing |
| `back where it already stood, nothing bound (k/3)` | a loop; not fatal until the third |
| `seen, but the relation is unmeasurable` | the right *kind* of object, not the one the phrase names — still searching |
| `no legal point closer than where we stand` | arrival at the platform's floor, which `obstacleDisThre` sets at roughly 0.9 m from furniture |

## 5. Drive one waypoint by hand

```bash
./scripts/drive.sh 3.54 -2.60        # go there, print the whole driven track
./scripts/drive.sh where             # current pose
```

For settling arguments about whether the stack *can* get somewhere. It prints
the track, not just the endpoint, which is how a passage that was gone *around*
rather than *through* gets caught.

## 6. Decompose a sentence offline

```bash
uv run --with anthropic python scripts/decompose.py "<sentence>"
uv run --with anthropic python scripts/decompose.py --diff        # all 30 official questions
uv run --with anthropic python scripts/decompose.py --json --limit 5
```

`--diff` compares the model's split against the regex fallback on what the
executor actually consumes: clause kinds in order, plus the keep-out count.
Costs no sim time and caches to `artifacts/decompose_cache.json`.

---

## 7. Probing one phrase without driving

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
| `ModuleNotFoundError: No module named 'google'` | missing `--with google-genai` | see "Driving on Gemini" |
| `RuntimeError: Cannot send a request, as the client has been closed` | fixed in TASK 40 — the genai client was a temporary | update the branch |
| gemini 429 `GenerateRequestsPerDayPerProjectPerModel-FreeTier` | Pro on the free tier, whose daily quota is zero | link billing in AI Studio, or use `gemini-3.6-flash` |
| `anthropic.AuthenticationError` | the `eval` line was not run in this shell | see above |
| preflight reports `rival_waypoint_publishers` | `xiao_hei_ai_module` is up | `ssh xiaohei1 'docker stop xiao_hei_ai_module'` |
| preflight reports nothing on `/joy` | the local planner discards every waypoint silently | the sim did not fully start; check `docker exec iros2026_system cat /tmp/sim.log` |
| `bridge produced no JSON` | `iros2026_system` is not running | §2 |
| no `/terrain_map`, robot never moves | Unity has no display | `xhost +local:` and `DISPLAY=:0`, then restart the scene |
| GPU rendering dead after an EC2 restart | modeset grabs the Amazon VGA and unloads nvidia | already pinned in `xorg.conf`; if the DCV session is gone, `sudo systemctl restart dcv-autosession` |
| `529 OverloadedError` | API under sustained load | `max_retries` is already 8; raise with `XIAO_HEI_API_MAX_RETRIES` |
| robot wedged, moves 0.02 m and stops | it is against the obstacle inflation | `./scripts/sim.sh restart`; the converter model now predicts this before driving |
| `sim.sh up` times out after 120 s | Unity failed to start | `ssh xiaohei1 'docker exec iros2026_system cat /tmp/sim.log'` |
| on the box: `ModuleNotFoundError` after `setup` said "done" | the venv is fine but an import reaches further than the dep list | `on_host.sh check` names the missing module; add it to `cmd_setup` |
| on the box: `uv venv` refuses, "already exists" | a failed `python3 -m venv` left a directory with no python in it | `setup` clears it now; by hand, `rm -rf .venv-drive` |
| a run "finished" suspiciously fast, one destination of three | the question went to `approach_loop.py`, which grounds it as one object | use `execute_plan.py`; see §3 |
| `plan.json` missing | it is written at the end — the run is still going, or it died | `ps aux \| grep execute_plan`; `steps.jsonl` is written as it goes |
| steps from two different runs in one directory | `--out` was re-used; colliding names were overwritten and the rest left | fresh `--out` per run |
| a leg reports `arrived` on the wrong object | the model matched the distinguishing feature and dropped the head noun — a clock on a TV sideboard answered "the nightstand with a clock on it" | not fixed; check the `evidence` field, which named the sideboard |
| a search leg never leaves the first room | before TASK 35 this was `reach · cos(Δ)` preferring the corridor to the doorway | fixed; if it recurs, check whether `way out:` appears in the log at all |
| output of a backgrounded run is empty | piping into `tail` buffers until the process exits | drop the pipe, or watch `steps.jsonl` |

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

## Platform parameters, as shipped

These are not ours to choose and not ours to measure. They are read straight
out of the official image, `zhangjicmu/ubuntu24_ros:cmu_vla_challenge_simulation`,
and any number in our code that contradicts one of them is a bug — a comment
reading "the vehicle is ~0.6 m across" is what made TASK 36 first call a
`living_room_1` gap impassable that is merely narrow.

`src/base_autonomy/local_planner/launch/local_planner.launch`:

| | | |
|---|---|---|
| `vehicleLength` | 0.5 m | |
| `vehicleWidth` | **0.5 m** | the only width there is |
| `pathScale` | 0.875 | the path library shrinks to `minPathScale` 0.675 when tight |
| `pointPerPathThre` | 2 | one obstacle point does not reject a path |
| `adjacentRange` | 3.5 m | how far the local map reaches |
| `obstacleHeightThre` | 0.05 m | above the local ground, an obstacle |

`waypoint_converter/launch/waypoint_converter.launch`:

| | | |
|---|---|---|
| `obstacleDisThre` | 0.75 m | where a **waypoint** may be placed, not where the vehicle may drive |
| `waypointXYRadius` | 0.3 m | how near the waypoint counts as arrived |
| `adjDisThre` | 5.0 m | beyond this the converter does not snap |
| `checkTravArea` | true | and it checks against a shipped mesh — see below |
| `trav_area_file_dir` | `mesh/<world>/traversable_area.ply` | |

Because `local_planner` scales its path library and needs two blocking points
to reject a path, the width a gap actually needs sits somewhere above 0.5 m and
depends on the approach — which is why `studio` threads the same gap on three
runs of six. **Do not build a "too narrow, skip it" rule on width alone.**

`traversable_area.ply` ships per scene and is what the converter itself checks,
so it — not a statistic over lidar points — is the authority on whether a gap
is drivable:

```
~/workspace/dataset/unity-scene/<scene>/traversable_area.ply
```

To re-read any of these rather than trusting this table:

```bash
ssh xiaohei1 'docker run --rm --entrypoint bash \
  zhangjicmu/ubuntu24_ros:cmu_vla_challenge_simulation -c \
  "grep -E \"<param\" /home/docker/autonomy_stack_mecanum_wheel_platform/src/\
base_autonomy/local_planner/launch/local_planner.launch"'
```
