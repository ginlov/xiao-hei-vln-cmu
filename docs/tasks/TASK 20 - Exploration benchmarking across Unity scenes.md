# TASK 20 — Exploration benchmarking across Unity scenes

Run the frontier explorer across every Unity environment and, per scene,
record how long it took and save a picture of the trajectory — the actual
RViz view from the simulator.

## What we achieved

- **All 18 scenes ready.** Extracted flat into `~/Downloads/unity_env_models/`
  (arabic_room, chinese_room, japanese_room, hotel_room_1/2, livingroom_1–4,
  loft, studio, office_1/2, office_building_1/2(+no_360), home_building_1/2).
- **One-command sweep.** `scripts/exploration_sweep.sh` loops the scenes:
  mount → launch sim → wait for the explorer to finish → tear down, then
  collates `exploration_logs/results.csv` (duration, waypoints, why it
  stopped).
- **RViz screenshot on every run.** The node grabs its RViz window when
  exploration ends, so each run leaves `exploration.log`, `exploration.png`
  (occupancy grid) and `rviz.png` (the sim's view) in
  `exploration_logs/<scene>/`. Best-effort — a missing display just skips it,
  never fails the run.

## Usage

```bash
uv sync --extra exploration

scripts/exploration_sweep.sh chinese_room   # one scene first
scripts/exploration_sweep.sh                # every scene found
```

Tunable by env var: `SCENES_DIR`, `MAX_WAYPOINTS` (default 500), `TIMEOUT`,
`DISPLAY`.

## Gotchas

- **`xhost +local:`** is required or RViz won't start (the containers run as a
  different user than the desktop). The sweep does this itself.
- **A timed-out scene has no images** — the node only writes them on
  completion. Raise `MAX_WAYPOINTS` or `TIMEOUT` and rerun.
- **Logs are written by the container as root.** The sweep clears only the
  stale `exploration.log`/PNGs before each run (never `rm -rf`s the dir), so it
  never deletes the log out from under a running node; `chown` if you need to
  move the output.

## Not yet verified on a live sim

The RViz window-title match (`DEFAULT_MATCH="rviz"` in
`exploration/_capture.py`) and whether the ai_module container can reach the X
server. If `rviz.png` is missing, check the real window titles:

```bash
uv run python -c "from xiao_hei_vln.exploration import list_windows; print(*list_windows(), sep='\n')"
```
