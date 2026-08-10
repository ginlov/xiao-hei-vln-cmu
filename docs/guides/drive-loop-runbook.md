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

The `eval` line above assumes the key is already in `~/.zshrc`. Getting it there
is a different question from getting it onto the box (§3b), and the answer is
different too, because **a personal laptop and a shared box are not the same
risk**. On the box, four people have accounts. On your laptop, the threat is
backups, sync, and anything you paste into a terminal that is being recorded.

Three options, weakest at rest to strongest:

**1. `~/.zshrc` — what the `eval` line above expects**

```bash
printf '\nexport ANTHROPIC_API_KEY=%s\n' 'sk-ant-...' >> ~/.zshrc
chmod 600 ~/.zshrc
```

Plaintext, and it goes into **every interactive shell you open**, so anything
you run inherits it. Acceptable on a single-user machine; do not do this on the
box. Note the literal command above puts the key in your shell history — prefix
it with a space (with `HIST_IGNORE_SPACE` set) or edit the file instead.

**2. A separate file, same shape as the box uses**

```bash
mkdir -p ~/.config/xiao-hei && chmod 700 ~/.config/xiao-hei
( umask 077; printf 'export ANTHROPIC_API_KEY=%s\n' 'sk-ant-...' > ~/.config/xiao-hei/env )
chmod 600 ~/.config/xiao-hei/env
```

Then per session, or from `~/.zshrc`:

```bash
source ~/.config/xiao-hei/env
```

Still plaintext, but it is one file you can `rm`, it is outside every repo, and
the same path works on both laptop and box. Swap the `eval` line for
`source ~/.config/xiao-hei/env` if you use this.

**3. macOS Keychain — no plaintext at rest**

```bash
security add-generic-password -a "$USER" -s anthropic-api-key -w    # prompts, hidden
```

Then in place of the `eval` line:

```bash
export ANTHROPIC_API_KEY="$(security find-generic-password -a "$USER" -s anthropic-api-key -w)"
```

The key lives encrypted in the login keychain; the first read after a login may
prompt for permission. This is the only one of the three where the key is not
sitting in a readable file. `security delete-generic-password -s
anthropic-api-key` removes it.

Whichever you pick, **use a different key for the box than for the laptop** — the
point of two keys is being able to revoke one without losing the other.

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
| `--backend` | `claude` | `gemini` reads `XIAO_HEI_GEMINI_API_KEY` (or `GEMINI_API_KEY`) from the environment; default model `gemini-2.5-flash` |
| `--prompt-version` | `v6-way-out` | `v5-constraints` is the version the cached replies and the offline scripts are keyed to |
| `--host` | — | omit to run inside the container instead of over ssh |

`--model` does **not** reach the decomposition step, which always uses
`claude-opus-5` — one cached call per question, 3.3 s, 30/30 on drive order.

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

Everything above drives from the laptop over ssh. The alternative is to sit on
the box itself, which is worth doing when **a run must outlive your laptop**: a
question is up to ten minutes at ~30 s per step, and a dropped connection
leaves the robot parked wherever it got to and the log half-written.

`scripts/on_host.sh` is that path. One-off, on the box:

```bash
ssh xiaohei1
cd ~/workspace/chengkai/vlm-drive          # see "getting the branch there"
./scripts/on_host.sh setup                 # venv + deps, no root needed
./scripts/on_host.sh key                   # store the API key, once
./scripts/on_host.sh check                 # says exactly what is missing
```

Then, per session:

```bash
tmux new -s drive                          # so the run survives the ssh session
./scripts/on_host.sh sim restart home_building_2
./scripts/on_host.sh run "Go near the magazine on the ottoman, then go to the potted plant on the dressing table." --out runs/hm2
# ctrl-b d to detach; `tmux attach -t drive` to come back
```

`on_host.sh one "<phrase>"` is the single-destination equivalent.

### The API key on the box

For the laptop side, see "The API key on the laptop" under §3 — the options
differ because the risks do. Here, two ways, and the trade between them is
persistence:

```bash
./scripts/on_host.sh key         # once: ~/.config/xiao-hei/env, mode 0600
export ANTHROPIC_API_KEY=...     # per shell; dies with it. Wins if both are set.
```

`key` prompts **without echoing** and reads from the tty, so the value never
reaches your shell history, the process list, or a script's stdin. It creates
the file under `umask 077` before writing a byte — a world-readable moment is
still a moment — and puts it at `0600` inside a `0700` directory.

Three things worth being deliberate about:

- **It lives outside the checkout**, at `~/.config/xiao-hei/env`. A key
  committed to a repo that is going to be made public is the one mistake here
  that cannot be undone by deleting the file. Override with `XIAO_HEI_ENV_FILE`.
- **Use a separate key for the box**, one you can revoke without touching your
  laptop's. It is now at rest on a machine you do not solely control: `0600`
  keeps the other accounts in `~/workspace` out, but root and snapshots are
  still root and snapshots.
- **Never `~/.bashrc` or `~/.profile`**, where it leaks into every process you
  start, including anything else anyone runs in your session.

`check` reports which source the key came from and warns if the file's mode has
drifted off `0600`. `rm ~/.config/xiao-hei/env` undoes it.

**What `setup` installs, and why so little.** `numpy`, `opencv-python-headless`,
`anthropic`, `pillow`, `pydantic`, `scipy` — into `.venv-drive`, its own venv,
not the repo's `.venv` (that one belongs to whatever else is set up on the box).
No ROS: `robot_io.py` is copied *into* the container and run there, so `rclpy`
and the message packages are the container's problem. `scipy` is the converter
model's cKDTree; `pillow` and `pydantic` are only there because `vlm_locate`
imports `xiao_hei_vln.perception.geometry` and that package's `__init__` eagerly
pulls in the ROS responder.

It uses **uv**, and not by preference: the system python on these boxes has
neither `pip` nor the `venv` module, and `python3 -m venv` fails asking for
`python3-venv`, which needs root. uv is already at `~/.local/bin/uv` on
xiaohei1. If it is ever missing, `setup` prints how to get it.

> `~/.local/bin` is not on `PATH` for `ssh box 'cmd'` (a non-login shell), so a
> scripted check can report uv missing while it is perfectly present for the
> person typing at the terminal. `on_host.sh` looks in `~/.local/bin` directly.

**Getting the branch there.** xiaohei1 can `git fetch` (ssh remote); xiaohei2
cannot (https remote, no credentials — use a git bundle). Do **not** switch the
branch of `~/workspace/chengkai/xiao-hei-vln-cmu`: on xiaohei1 that checkout has
other people's modified files in it. Add a worktree instead, which touches
nothing that is already there:

```bash
cd ~/workspace/chengkai/xiao-hei-vln-cmu
git fetch origin feature/vlm-approach-loop
git worktree add ~/workspace/chengkai/vlm-drive origin/feature/vlm-approach-loop
```

To update it later: `git -C ~/workspace/chengkai/vlm-drive fetch origin
feature/vlm-approach-loop && git -C ~/workspace/chengkai/vlm-drive reset --hard
origin/feature/vlm-approach-loop`.

**`local` as a host.** `on_host.sh` exports `XIAO_HEI_SIM_HOST=local`, which
makes `sim.sh`, `drive.sh` and the loop all skip ssh and talk to the local
docker. You can set it by hand for the same effect:

```bash
export XIAO_HEI_SIM_HOST=local
./scripts/sim.sh status
./scripts/drive.sh 3.54 -2.60
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
