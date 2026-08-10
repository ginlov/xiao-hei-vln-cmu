# Tick Loop

The VLM runs on a fixed-frequency timer (default 1 Hz = 1000 ms per tick).
This decouples the model's processing rate from the sensor publishing rates
(10-200 Hz).

## Why 1 Hz?

| Constraint | Value |
|---|---|
| Camera topic rate | ~10 Hz |
| Perception sidecar detect+segment (OWLv2 + SAM 2.1 Large) | ~500-800 ms per tick |
| Available budget per tick | 1000 ms |
| Overhead (snapshot + publish) | ~5 ms |

The stack moved from YOLO-World + SAM-Tiny (~200-400 ms, which fit a 500 ms /
2 Hz tick) to OWLv2 + SAM-Large for detection quality, which no longer fits
500 ms. 1 Hz gives the model enough time to complete inference within one tick.
Task 1/2 answer within a 10-minute budget, not in real time, so the lower rate
costs nothing in responsiveness that matters.

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
XIAO_HEI_VLM_TICK_HZ=1.0   # default
XIAO_HEI_VLM_TICK_HZ=2.0   # faster, only if the detector keeps up
XIAO_HEI_VLM_TICK_HZ=0.5   # slower, for debugging
```

!!! warning
    Setting tick rate higher than the model can handle will cause ticks to
    overlap. The responder is synchronous — a slow inference blocks the next
    tick.
