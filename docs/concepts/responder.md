# Responder Protocol

A responder is any object that implements three methods:

```python
class ResponderProtocol:
    def respond(self, snapshot: VLMInput) -> VLMOutput | None: ...
    def is_done(self) -> bool: ...
    def reset(self) -> None: ...
    def close(self) -> None: ...
```

## Method contract

### `respond(snapshot: VLMInput) -> VLMOutput | None`

Called once per tick (2 Hz). Receives the latest sensor snapshot and returns
either a `VLMOutput` to publish or `None` (skip this tick).

### `is_done() -> bool`

Returns `True` when the responder has produced a final answer for the current
question. The tick loop will then:

1. Clear the question from the cache
2. Call `reset()` to prepare for the next question

### `reset()`

Resets internal state (tick counter, evidence log, done flag). Called when:

- A new question arrives (different text from the previous one)
- The responder reports `is_done()`

### `close()`

Cleanup hook called during shutdown. Used by `QwenResponder` to flush the
VLM tick logger.

## Implementations

### DummyResponder

Deterministic port of the challenge's C++ `dummyVLM.cpp`. Returns fixed
answers without any model inference. No GPU required.

- Numerical: always returns `value=5`
- Object reference: returns a fixed marker position
- Instruction following: returns a single waypoint `(1.0, 0.0)`

### QwenResponder

Production responder using Qwen3.5-VL. Features:

- Multi-tick reasoning for numerical questions (evidence accumulation)
- Timeout safety cap (30 ticks default)
- Optional tick logging for debugging
- Pluggable engine (`HTTPQwenEngine` or `QwenEngine`)

## Lifecycle in the tick loop

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Responding: new question arrives
    Responding --> Responding: respond() returns VLMOutput (not terminal)
    Responding --> Done: is_done() == True
    Done --> Idle: reset() called
```

## Selecting a responder

Set `XIAO_HEI_RESPONDER` environment variable:

```bash
XIAO_HEI_RESPONDER=dummy   # DummyResponder (no GPU)
XIAO_HEI_RESPONDER=qwen    # QwenResponder (requires vLLM)
```
