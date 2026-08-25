"""Inference engine for the navigation waypoint proposer.

The explorer depends only on :class:`NavVLMEngineProtocol` — a single
``propose(...)`` method returning a :class:`WaypointProposal`. Anything
satisfying that protocol works, so the concrete backend (Opus 5 today, a
Gemini engine tomorrow) is swappable without touching the explorer.

:class:`AnthropicNavEngine` is the Opus-5 implementation. It forces the model
to answer via the ``propose_waypoint`` tool, so the result is structured, and
imports the ``anthropic`` SDK lazily so this module stays importable without
the ``nav-vlm`` extra. Tests inject a fake client implementing
:class:`AnthropicClientProtocol` — no key and no network needed.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from xiao_hei_vln.nav_vlm.config import NavVLMConfig
from xiao_hei_vln.nav_vlm.prompts import PROPOSE_WAYPOINT_TOOL, SYSTEM_PROMPT

log = logging.getLogger(__name__)

# Direct Messages API endpoint + version, used by the raw-httpx path (the
# portable default, avoiding the SDK's environment-specific transport).
_API_BASE = "https://api.anthropic.com"
_ANTHROPIC_VERSION = "2023-06-01"


@dataclass(frozen=True)
class WaypointProposal:
    """Parsed result of one proposer call.

    ``done`` short-circuits everything else: when true the explorer stops.
    Otherwise ``x``/``y`` are a target in the map frame (``heading`` optional,
    ``None`` = let the caller face the direction of travel).
    """

    done: bool
    x: float | None = None
    y: float | None = None
    heading: float | None = None
    rationale: str = ""
    visible_objects: tuple[str, ...] = ()
    verify_objects: tuple[str, ...] = ()

    @classmethod
    def from_tool_input(cls, data: dict[str, Any]) -> WaypointProposal:
        """Build from the model's ``propose_waypoint`` tool input dict."""
        done = bool(data.get("done", False))
        x = data.get("x")
        y = data.get("y")
        heading = data.get("heading")
        vis = data.get("visible_objects") or []
        verify = data.get("verify_objects") or []
        return cls(
            done=done,
            x=None if x is None else float(x),
            y=None if y is None else float(y),
            heading=None if heading is None else float(heading),
            rationale=str(data.get("rationale", "")),
            visible_objects=tuple(str(v) for v in vis if isinstance(v, str)),
            verify_objects=tuple(str(v) for v in verify if isinstance(v, str)),
        )


class NavVLMEngineProtocol(Protocol):
    """The engine surface :class:`NavVLMExplorer` depends on."""

    def propose(
        self,
        *,
        user_text: str,
        panorama_jpg: bytes | None,
        occupancy_png: bytes,
    ) -> WaypointProposal:
        ...


class ToolCallerProtocol(Protocol):
    """The generic forced-tool surface the object-reference answerer depends on.

    A fake implementing just this returns scripted tool inputs, so the
    answering responder is unit-testable without a key or network.
    """

    def call_tool(
        self,
        *,
        system: str,
        tool: dict[str, Any],
        user_text: str,
        images: list[tuple[bytes, str]],
    ) -> dict[str, Any]:
        ...


# --- Anthropic (Opus 5) backend -------------------------------------------


class _MessagesNamespaceProtocol(Protocol):
    """The ``client.messages`` surface we depend on."""

    def create(self, **kwargs: Any) -> Any: ...


class AnthropicClientProtocol(Protocol):
    """The ``anthropic.Anthropic`` surface we depend on.

    Only ``client.messages.create`` is exercised, so a fake can implement
    just that for tests.
    """

    @property
    def messages(self) -> _MessagesNamespaceProtocol: ...


class AnthropicNavEngine:
    """Opus-5-backed forced-tool-use engine.

    Defaults to the exploration waypoint proposer (``propose_waypoint`` +
    :data:`SYSTEM_PROMPT`), but the system prompt and tool are injectable, so
    the same transport/retry/parse machinery backs the goal-directed Task-1
    navigator and the object-reference answerer (see
    :mod:`xiao_hei_vln.nav_vlm.task1_prompts`). :meth:`call_tool` is the generic
    primitive; :meth:`propose` is the exploration convenience on top of it.
    """

    def __init__(
        self,
        config: NavVLMConfig,
        *,
        client: AnthropicClientProtocol | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        tool: dict[str, Any] | None = None,
    ) -> None:
        self._config = config
        self._system_prompt = system_prompt
        self._tool = tool if tool is not None else PROPOSE_WAYPOINT_TOOL
        # An injected client (tests) uses the SDK-style `.messages.create`
        # surface. Otherwise we POST directly to the Messages API over standard
        # httpx: the `anthropic` SDK build in some environments is bound to a
        # non-portable transport (httpx2/jsfetch) that fails inside containers,
        # whereas a plain httpx POST to api.anthropic.com works everywhere the
        # network does.
        self._client = client
        self._http = None
        if client is None:
            import httpx  # standard httpx (not the SDK's httpx2 fork)

            self._http = httpx.Client(
                base_url=_API_BASE,
                timeout=config.request_timeout_s,
                transport=httpx.HTTPTransport(retries=config.max_retries),
            )
            self._headers = {
                "x-api-key": config.api_key,
                "anthropic-version": _ANTHROPIC_VERSION,
                "content-type": "application/json",
            }

    def call_tool(
        self,
        *,
        system: str,
        tool: dict[str, Any],
        user_text: str,
        images: list[tuple[bytes, str]],
    ) -> dict[str, Any]:
        """One forced-tool multimodal call; return the tool's input dict.

        ``images`` is a list of ``(data, media_type)`` pairs rendered before
        the text block (so the model sees the picture first). Raises on
        transport failure or if the model emitted no matching tool call, so
        callers can log and fall back rather than silently stalling.
        """
        content: list[dict[str, Any]] = [
            _image_block(data, media_type) for data, media_type in images
        ]
        content.append({"type": "text", "text": user_text})

        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "max_tokens": self._config.max_output_tokens,
            "system": system,
            "messages": [{"role": "user", "content": content}],
            "tools": [tool],
            "tool_choice": {"type": "tool", "name": tool["name"]},
        }
        if self._config.temperature is not None:
            # Sent via extra_body (not a create() kwarg) so it works across SDK
            # builds; only sent when explicitly configured, because Opus 5 and
            # other recent models DEPRECATE temperature and 400 on it.
            kwargs["extra_body"] = {"temperature": self._config.temperature}
        if self._config.thinking_budget > 0:
            # Extended thinking is incompatible with forcing a specific tool;
            # fall back to auto tool choice so the request stays valid.
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self._config.thinking_budget,
            }
            kwargs["tool_choice"] = {"type": "auto"}

        if self._client is not None:
            # SDK-style injected client (tests).
            response: Any = self._client.messages.create(**kwargs)
        else:
            # Raw httpx POST. `extra_body` is an SDK concept — merge it into the
            # request JSON directly.
            body = {k: v for k, v in kwargs.items() if k != "extra_body"}
            body.update(kwargs.get("extra_body", {}))
            resp = self._http.post("/v1/messages", headers=self._headers, json=body)
            if resp.status_code >= 400:
                raise ValueError(
                    f"Messages API {resp.status_code}: {resp.text[:300]}"
                )
            response = resp.json()

        tool_input = _extract_tool_input(response, tool["name"])
        if tool_input is None:
            raise ValueError(
                f"model returned no {tool['name']} tool call",
            )
        return tool_input

    def propose(
        self,
        *,
        user_text: str,
        panorama_jpg: bytes | None,
        occupancy_png: bytes,
    ) -> WaypointProposal:
        """One multimodal call; returns the parsed proposal.

        Uses this engine's configured system prompt + tool, so the same method
        serves the exploration proposer and the goal-directed Task-1 navigator
        (both return a :class:`WaypointProposal`, where ``done`` means "stop"
        for exploration and "arrived at the target object" for Task 1).
        """
        images: list[tuple[bytes, str]] = []
        if panorama_jpg is not None:
            images.append((panorama_jpg, "image/jpeg"))
        images.append((occupancy_png, "image/png"))
        tool_input = self.call_tool(
            system=self._system_prompt,
            tool=self._tool,
            user_text=user_text,
            images=images,
        )
        return WaypointProposal.from_tool_input(tool_input)

    def warmup(self) -> None:
        """Minimal call to confirm credentials + model access at boot.

        Uses a small solid PNG (not a 1x1) because the vision API rejects
        sub-minimum images with "Could not process image".
        """
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (200, 200, 200)).save(buf, format="PNG")
        self.propose(
            user_text="Warmup: set done=true.",
            panorama_jpg=None,
            occupancy_png=buf.getvalue(),
        )


# --- helpers ---------------------------------------------------------------


def _image_block(data: bytes, media_type: str) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64.standard_b64encode(data).decode("ascii"),
        },
    }


def _extract_tool_input(response: Any, tool_name: str) -> dict[str, Any] | None:
    """Pull the first matching tool_use block's input from a response.

    Tolerant of both the real SDK's objects (``block.type`` / ``block.input``)
    and plain dicts, so fakes stay simple.
    """
    content = getattr(response, "content", None)
    if content is None and isinstance(response, dict):
        content = response.get("content")
    if not content:
        return None
    for block in content:
        btype = getattr(block, "type", None)
        if btype is None and isinstance(block, dict):
            btype = block.get("type")
        if btype != "tool_use":
            continue
        name = getattr(block, "name", None)
        if name is None and isinstance(block, dict):
            name = block.get("name")
        if name != tool_name:
            continue
        data = getattr(block, "input", None)
        if data is None and isinstance(block, dict):
            data = block.get("input")
        if isinstance(data, dict):
            return data
    return None
