# Adding a New Exploration Strategy

The exploration system is designed around a simple three-method protocol.
Swapping in a new algorithm requires implementing the protocol, registering
one `elif` branch, and selecting it with an env var — no other files need
to change.

## Step 1: Implement the protocol

Create a class that implements `update`, `is_complete`, and `reset`:

```python
# src/xiao_hei_vln/exploration/my_strategy.py
from xiao_hei_vln.messages.inputs import VLMInput
from xiao_hei_vln.messages.outputs import Waypoint


class MyStrategy:
    def __init__(self, max_waypoints: int = 100) -> None:
        self._max_waypoints = max_waypoints
        self._visited: list[Waypoint] = []
        self._done = False

    def update(self, snapshot: VLMInput) -> Waypoint | None:
        """Called once per tick. Return the next waypoint or None."""
        if self._done:
            return None

        pose = snapshot.pose
        if pose is None:
            return None  # wait for odometry

        # Your navigation logic here.
        # Use snapshot.terrain_ext for terrain data,
        # snapshot.pose for robot position.
        wp = self._pick_next_waypoint(pose)

        if len(self._visited) >= self._max_waypoints:
            self._done = True
            return None

        return wp

    def is_complete(self) -> bool:
        return self._done

    def reset(self) -> None:
        self._visited = []
        self._done = False

    def _pick_next_waypoint(self, pose) -> Waypoint | None:
        ...
```

### Optional: plot support

If your strategy maintains a grid and a list of visited waypoints, expose
them so `_maybe_save_png()` can save a debug PNG after exploration ends:

```python
def get_visited_waypoints(self) -> list[Waypoint]:
    return list(self._visited)

def get_grid(self) -> OccupancyGrid:
    return self._grid
```

Without these methods the PNG step is silently skipped — nothing breaks.

## Step 2: Register in the factory

Edit `_build_explorer()` in `src/xiao_hei_vln/app/main.py`.  Add an
`elif` branch for your strategy name:

```python
def _build_explorer(node):
    if _EXPLORATION_MAX_WAYPOINTS <= 0:
        return None

    if _EXPLORATION_STRATEGY == "frontier":
        from xiao_hei_vln.exploration import FrontierExplorer
        explorer = FrontierExplorer(
            max_waypoints=_EXPLORATION_MAX_WAYPOINTS,
            ...
        )
    elif _EXPLORATION_STRATEGY == "my_strategy":          # ← add this
        from xiao_hei_vln.exploration.my_strategy import MyStrategy
        explorer = MyStrategy(max_waypoints=_EXPLORATION_MAX_WAYPOINTS)
    else:
        node.get_logger().error(...)
        return None

    node.get_logger().info(f"Exploration enabled: {type(explorer).__name__} ...")
    return explorer
```

## Step 3: Select at runtime

```bash
# Docker Compose
XIAO_HEI_EXPLORATION_STRATEGY=my_strategy docker compose -f docker/compose_gpu.yml up -d --build

# Or export before compose
export XIAO_HEI_EXPLORATION_STRATEGY=my_strategy
```

The default is `frontier`.  An unrecognised name logs an error and
disables exploration rather than crashing.

## Step 4: Test without ROS

The protocol is pure Python — no ROS install needed for unit tests.
Build a `VLMInput` by hand and drive `update()` directly:

```python
from xiao_hei_vln.messages.inputs import VLMInput
from xiao_hei_vln.messages.common import Stamp
from xiao_hei_vln.messages.sensors import OdomPose
from xiao_hei_vln.exploration.my_strategy import MyStrategy


def test_my_strategy_completes():
    strategy = MyStrategy(max_waypoints=3)

    pose = OdomPose(...)          # build a minimal pose
    snapshot = VLMInput(
        tick_id=0,
        tick_time=Stamp.from_seconds(0.0),
        pose=pose,
    )

    for _ in range(100):
        strategy.update(snapshot)
        if strategy.is_complete():
            break

    assert strategy.is_complete()
```

Run: `uv run pytest tests/ -q`

## Step 5: Run end-to-end

```bash
XIAO_HEI_EXPLORATION_STRATEGY=my_strategy \
XIAO_HEI_EXPLORATION_MAX_WAYPOINTS=50 \
docker compose -f docker/compose_gpu.yml up -d --build

docker logs -f xiao_hei_ai_module
# look for: Exploration enabled: MyStrategy (strategy=my_strategy, ...)
```

Check `exploration_logs/exploration.log` for `WP_SET`, `WP_ADVANCE`,
`WP_SKIP`, and `DONE` events as the run progresses.
