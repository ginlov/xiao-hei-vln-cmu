"""AnthropicNavEngine request-shaping + tool-parsing, via a fake client."""

from __future__ import annotations

import pytest

from xiao_hei_vln.nav_vlm.config import NavVLMConfig
from xiao_hei_vln.nav_vlm.engine import AnthropicNavEngine, WaypointProposal


class _Block:
    def __init__(self, type, name=None, input=None):
        self.type = type
        self.name = name
        self.input = input


class _Response:
    def __init__(self, content):
        self.content = content


class _Messages:
    def __init__(self, response):
        self._response = response
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _Client:
    def __init__(self, response):
        self.messages = _Messages(response)


def _cfg() -> NavVLMConfig:
    return NavVLMConfig(api_key="sk-test")


def test_parses_tool_use_into_proposal() -> None:
    resp = _Response([
        _Block("text", input=None),  # models can emit text alongside the tool
        _Block("tool_use", name="propose_waypoint",
               input={"done": False, "x": 1.5, "y": -2.0, "heading": 0.3,
                      "rationale": "toward the open doorway"}),
    ])
    eng = AnthropicNavEngine(_cfg(), client=_Client(resp))

    out = eng.propose(user_text="go", panorama_jpg=b"jpg", occupancy_png=b"png")

    assert isinstance(out, WaypointProposal)
    assert (out.done, out.x, out.y, out.heading) == (False, 1.5, -2.0, 0.3)
    assert out.rationale.startswith("toward")


def test_request_forces_tool_and_sends_both_images() -> None:
    resp = _Response([
        _Block("tool_use", name="propose_waypoint",
               input={"done": True, "rationale": "done"}),
    ])
    client = _Client(resp)
    eng = AnthropicNavEngine(_cfg(), client=client)

    eng.propose(user_text="go", panorama_jpg=b"jpg", occupancy_png=b"png")

    kw = client.messages.last_kwargs
    assert kw["tool_choice"] == {"type": "tool", "name": "propose_waypoint"}
    assert kw["model"] == "claude-opus-5"
    image_blocks = [c for c in kw["messages"][0]["content"] if c["type"] == "image"]
    assert len(image_blocks) == 2  # panorama + occupancy


def test_no_panorama_sends_only_occupancy() -> None:
    resp = _Response([
        _Block("tool_use", name="propose_waypoint",
               input={"done": True, "rationale": "done"}),
    ])
    client = _Client(resp)
    eng = AnthropicNavEngine(_cfg(), client=client)

    eng.propose(user_text="go", panorama_jpg=None, occupancy_png=b"png")

    image_blocks = [
        c for c in client.messages.last_kwargs["messages"][0]["content"]
        if c["type"] == "image"
    ]
    assert len(image_blocks) == 1


def test_missing_tool_call_raises() -> None:
    resp = _Response([_Block("text", input=None)])
    eng = AnthropicNavEngine(_cfg(), client=_Client(resp))
    with pytest.raises(ValueError, match="no propose_waypoint"):
        eng.propose(user_text="go", panorama_jpg=None, occupancy_png=b"png")


def test_call_tool_forces_injected_tool_and_returns_input() -> None:
    """The generic call_tool primitive forces an arbitrary tool and parses it."""
    tool = {"name": "answer_object_reference", "input_schema": {"type": "object"}}
    resp = _Response([
        _Block("tool_use", name="answer_object_reference",
               input={"object_id": 7, "rationale": "the red one"}),
    ])
    client = _Client(resp)
    eng = AnthropicNavEngine(_cfg(), client=client)

    out = eng.call_tool(
        system="answer sys", tool=tool, user_text="q",
        images=[(b"png", "image/png")],
    )

    assert out == {"object_id": 7, "rationale": "the red one"}
    kw = client.messages.last_kwargs
    assert kw["tool_choice"] == {"type": "tool", "name": "answer_object_reference"}
    assert kw["system"] == "answer sys"


def test_render_grid_png_smoke() -> None:
    """The occupancy render produces a non-trivial PNG (needs matplotlib)."""
    pytest.importorskip("matplotlib")
    from xiao_hei_vln.exploration._grid import OccupancyGrid
    from xiao_hei_vln.nav_vlm.render import render_grid_png

    grid = OccupancyGrid(resolution=1.0)
    grid._free.update({(0, 0), (1, 0), (2, 0)})
    grid._occupied.update({(2, 1)})
    png = render_grid_png(grid, robot_xy=(0.5, 0.5), trajectory_xy=[(0.5, 0.5), (1.5, 0.5)])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic
