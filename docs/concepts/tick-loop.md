# Tick Loop

The VLM runs on a fixed-frequency timer (default 2 Hz = 500 ms per tick).
This decouples the model's processing rate from the sensor publishing rates
(10-200 Hz).

## Why 2 Hz?

| Constraint | Value |
|---|---|
| Camera topic rate | ~10 Hz |
| Perception sidecar detect+segment | ~200-400 ms per tick |
| Available budget per tick | 500 ms |
| Overhead (snapshot + publish) | ~5 ms |

2 Hz gives the model enough time to complete inference within one tick while
still being responsive to environmental changes.

## LatestCache

`LatestCache` is a thread-safe buffer that stores the most recent message
from each sensor topic:

```python
class LatestCache:
    def snapshot(self, tick_id: int, tick_time: Stamp) -> VLMInput:
        """Atomically capture current state of all slots."""
```

ROS subscriber callbacks run on separate threads and overwrite slots
continuously. The `snapshot()` method takes a lock and copies all current
values into a single `VLMInput`.

## Tick execution flow

```python
def tick():
    # 1. Capture current sensor state
    snapshot = cache.snapshot(tick_id, timestamp)

    # 2. Detect new question → reset responder
    if snapshot.question != last_question:
        responder.reset()
        last_question = snapshot.question

    # 3. Run model inference
    output = responder.respond(snapshot)

    # 4. Publish response (if any)
    if output is not None:
        publisher.publish(output)

    # 5. Check if question is answered
    if responder.is_done():
        cache.clear_question()
        responder.reset()
```

## Configuring tick rate

```bash
XIAO_HEI_VLM_TICK_HZ=2.0   # default
XIAO_HEI_VLM_TICK_HZ=1.0   # slower, for debugging
XIAO_HEI_VLM_TICK_HZ=5.0   # faster, if model is fast enough
```

!!! warning
    Setting tick rate higher than the model can handle will cause ticks to
    overlap. The responder is synchronous — a slow inference blocks the next
    tick.
