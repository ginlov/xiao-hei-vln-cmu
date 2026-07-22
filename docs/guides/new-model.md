# New Model Integration

This guide walks through plugging a new VLM into the system. The framework
handles all ROS communication, sensor buffering, and output routing — you only
write the inference logic.

## Step 1: Create a responder

```python
# src/xiao_hei_vln/my_model/responder.py
from xiao_hei_vln.messages import (
    NumericalResponse,
    ObjectReferenceResponse,
    VLMInput,
    VLMOutput,
    Vector3,
    Waypoint,
    WaypointPathResponse,
)
from xiao_hei_vln.messages.question import QuestionType


class MyModelResponder:
    def __init__(self) -> None:
        # Load weights, initialize tokenizer, etc.
        self._done = False

    def respond(self, snapshot: VLMInput) -> VLMOutput | None:
        if snapshot.question is None:
            return None

        match snapshot.question.type:
            case QuestionType.NUMERICAL:
                # Use snapshot.image, snapshot.pose, etc.
                answer = self._count_objects(snapshot)
                self._done = True
                return NumericalResponse(value=answer)

            case QuestionType.OBJECT_REFERENCE:
                pos = self._find_object(snapshot)
                self._done = True
                return ObjectReferenceResponse(
                    label="target",
                    object_id=1,
                    center=pos,
                    size=Vector3(x=0.3, y=0.3, z=0.3),
                )

            case QuestionType.INSTRUCTION_FOLLOWING:
                waypoints = self._plan_path(snapshot)
                self._done = True
                return WaypointPathResponse(waypoints=waypoints)

    def is_done(self) -> bool:
        return self._done

    def reset(self) -> None:
        self._done = False

    def close(self) -> None:
        pass
```

## Step 2: Register in the responder factory

Edit `src/xiao_hei_vln/app/main.py`:

```python
def _build_responder(name: str):
    if name == "dummy":
        from xiao_hei_vln.dummy import DummyResponder
        return DummyResponder(), None
    if name == "scene_gemini":
        # ... existing submission-stack setup ...
        return SceneGeminiResponder(engine, config, scene, ...), logger
    if name == "my_model":
        from xiao_hei_vln.my_model.responder import MyModelResponder
        return MyModelResponder(), None
    raise ValueError(f"Unknown XIAO_HEI_RESPONDER={name!r}")
```

## Step 3: Test without ROS

```python
# tests/test_my_model.py
from xiao_hei_vln.messages import VLMInput, ChallengeQuestion, NumericalResponse
from xiao_hei_vln.my_model.responder import MyModelResponder


def test_numerical_question():
    responder = MyModelResponder()
    snapshot = VLMInput(
        tick_id=0,
        tick_time=Stamp.from_seconds(0),
        question=ChallengeQuestion.from_text("How many chairs", Stamp.from_seconds(0)),
        pose=None,
    )
    output = responder.respond(snapshot)
    assert isinstance(output, NumericalResponse)
    assert responder.is_done()
```

Run: `uv run pytest tests/test_my_model.py -v`

## Step 4: Run in Docker

```bash
# Set environment variable to use your model
export XIAO_HEI_RESPONDER=my_model

# If your model needs additional Python deps, add them to pyproject.toml:
# [project.optional-dependencies]
# my_model = ["transformers>=4.40", ...]

# Rebuild and start. If your model talks to a GPU sidecar (like the
# perception path), add it as a profile-gated service in docker/compose.yml.
XIAO_HEI_RESPONDER=my_model docker/run up -d --build
```

## Tips

- **Multi-tick reasoning**: Don't set `_done = True` immediately. Accumulate
  observations across ticks before committing to an answer. See
  `SceneGeminiResponder` for the explore-then-commit pattern.

- **Use the image**: `snapshot.image` is a 1920x640 panoramic BGR8 frame.
  Convert to your model's expected format (RGB PIL, tensor, etc.).

- **Use the point clouds**: `snapshot.registered_scan.points` is `(N, 4)`
  float32 — useful for spatial reasoning about object positions.

- **Logging**: Pass a `VLMLogger` to record inputs/outputs for debugging.
  See [VLM Logging](vlm-logging.md).
