"""Prompt builders for the Qwen3.5 responder.

Two builders are exported:

- `build_user_message(...)` — generic, used for object_reference and
  instruction_following questions.
- `build_numerical_user_message(...)` — specialised for the
  *numerical* question (Phase 3). Lays out a per-tick protocol so the
  model emits either an exploration waypoint OR commits to the final
  integer answer, and accumulates a running tally across ticks via the
  `rationale` field on each `VLMOutput`.

Both builders are pure-Python (no PIL, no vLLM) so they're testable
without a GPU. See `docs/task3_phase3_prompt.md` for the design.
"""

from __future__ import annotations

from xiao_hei_vln.messages import ChallengeQuestion, OdomPose, QuestionType, VLMInput

# --- system prompts --------------------------------------------------------

SYSTEM_PROMPT = """\
You are the vision-language model on board a wheeled robot competing
in the CMU Vision-Language-Navigation Challenge. Each tick you receive
one front-camera image (when available), the robot's current pose in
the global map frame, a running evidence log of what you have already
observed, and one challenge question.

You MUST reply with a single JSON object that conforms to this schema:

  {"kind": "numerical", "value": <int>, "rationale": <str|null>}
    -> use ONLY for "How many ..." questions, once you are confident.

  {"kind": "object_reference", "label": <str>, "object_id": <int>,
   "center": {"x": <float>, "y": <float>, "z": <float>},
   "size":   {"x": <float>, "y": <float>, "z": <float>},
   "heading": <float>, "rationale": <str|null>}
    -> use ONLY for "Find ..." questions; pose values are in the global
       map frame and metres.

  {"kind": "waypoint_path", "waypoints":
     [{"x": <float>, "y": <float>, "heading": <float>}, ...],
   "rationale": <str|null>}
    -> use for instruction-following questions, AND as a navigation
       step when a numerical/object-reference question still requires
       you to explore before you can answer. Waypoints are in the
       global map frame and metres; at least one is required.

Rules:
- Always pick exactly one of the three response shapes.
- Numerical answers must be non-negative integers.
- When proposing waypoints, prefer points that improve coverage of
  whatever the question is about, and stay within ~5 m of the current
  pose so the local terrain map keeps them reachable.
- Do not include any text outside the JSON object.
"""

NUMERICAL_SYSTEM_PROMPT = """\
You are the vision-language model on board a wheeled robot in the CMU
Vision-Language-Navigation Challenge, currently answering a NUMERICAL
question of the form "How many <target> ... ?". You must drive the
robot to viewpoints that give you confident visibility of the target,
then commit to a single integer count.

You will be invoked once every 500 ms with: the current front-camera
image, the robot's pose in the map frame, and an evidence log
containing the rationale you produced on every previous tick of this
question. Use the evidence log as your scratchpad — the only state
that crosses ticks is what you write into `rationale`.

Per-tick protocol:

1. From the current image, estimate `view_count` = how many <target>
   are visible in this frame.
2. From the evidence log, recover your prior `running_total` and the
   set of viewpoints already covered.
3. Decide whether to keep exploring or commit:
   - Explore if any of: the current frame is occluded, you have only
     covered 0-1 distinct viewpoints, or your last two tallies
     disagreed.
   - Commit when you have observed the scene from at least two
     distinct viewpoints AND your last two tallies agree AND the
     current image shows no plausibly new instances of <target>.

You MUST reply with EXACTLY one JSON object:

  - To EXPLORE, emit:
      {"kind": "waypoint_path",
       "waypoints": [{"x": <float>, "y": <float>, "heading": <float>}],
       "rationale": "view_count=<int> running_total=<int> action=explore <free text>"}

  - To COMMIT, emit:
      {"kind": "numerical",
       "value": <int>,
       "rationale": "running_total=<int> action=commit <free text>"}

Waypoint constraints when exploring:
- Coordinates are global map metres. Stay within ~5 m of the current
  pose so the local terrain map can validate reachability.
- Prefer waypoints that face an unexplored sector of the scene; do not
  repeat a viewpoint already listed in the evidence log.
- `heading` is the target yaw in radians (0 = +x axis), facing toward
  the region you want to inspect.

The `rationale` field is REQUIRED on every reply. Always begin it with
the `view_count=... running_total=... action=...` triplet so the next
tick can parse your prior thinking. Keep the free text under 200
characters. Do not output anything outside the JSON object.
"""


def build_system_prompt() -> str:
    return SYSTEM_PROMPT


def build_numerical_system_prompt() -> str:
    return NUMERICAL_SYSTEM_PROMPT


# --- per-tick user messages ------------------------------------------------


def build_user_message(
    snapshot: VLMInput,
    question: ChallengeQuestion,
    evidence_log: list[str],
) -> str:
    """Render the per-tick user message — snapshot summary + the question."""
    parts: list[str] = [f"Question (type={question.type.value}): {question.text}"]
    parts.append(_pose_summary(snapshot.pose))
    parts.append(_terrain_summary(snapshot))
    if evidence_log:
        parts.append("Evidence so far (oldest -> newest):")
        for i, entry in enumerate(evidence_log, start=1):
            parts.append(f"  {i}. {entry}")
    else:
        parts.append("Evidence so far: (none — first tick on this question)")
    parts.append(_response_hint(question.type))
    parts.append("Respond now with the JSON object.")
    return "\n".join(parts)


def build_numerical_user_message(
    snapshot: VLMInput,
    question: ChallengeQuestion,
    evidence_log: list[str],
    *,
    tick_index: int,
    max_ticks: int,
) -> str:
    """Render the per-tick user message for a NUMERICAL question (Phase 3)."""
    parts: list[str] = [
        f"NUMERICAL question: {question.text}",
        _pose_summary(snapshot.pose),
        _terrain_summary(snapshot),
        f"Tick {tick_index} of at most {max_ticks} for this question.",
    ]
    if evidence_log:
        parts.append("Your prior rationale (oldest -> newest):")
        for i, entry in enumerate(evidence_log, start=1):
            parts.append(f"  {i}. {entry}")
        parts.append(
            "Parse the most recent `view_count=...` / `running_total=...` "
            "lines to keep your tally consistent.",
        )
    else:
        parts.append("This is the FIRST tick on this question — initialise your tally.")

    if tick_index >= max_ticks:
        parts.append(
            "BUDGET EXHAUSTED: you MUST commit on this tick. Emit "
            "{\"kind\":\"numerical\",\"value\":<int>,\"rationale\":...}.",
        )
    else:
        remaining = max_ticks - tick_index
        parts.append(
            f"You have {remaining} ticks left. Explore only if you genuinely need "
            "a new viewpoint; otherwise commit.",
        )
    parts.append("Respond now with the JSON object.")
    return "\n".join(parts)


# --- helpers ---------------------------------------------------------------


def _pose_summary(pose: OdomPose | None) -> str:
    if pose is None:
        return "Pose: unknown (no /state_estimation yet)"
    p = pose.position
    return f"Pose: x={p.x:.2f} y={p.y:.2f} z={p.z:.2f} (map frame)"


def _terrain_summary(snapshot: VLMInput) -> str:
    bits: list[str] = []
    if snapshot.terrain_local is not None:
        bits.append(f"terrain_local: {snapshot.terrain_local.points.shape[0]} pts (5 m)")
    if snapshot.terrain_ext is not None:
        bits.append(f"terrain_ext: {snapshot.terrain_ext.points.shape[0]} pts (20 m)")
    if snapshot.registered_scan is not None:
        bits.append(f"lidar: {snapshot.registered_scan.points.shape[0]} pts")
    if not bits:
        return "Sensors: (no lidar/terrain in cache yet)"
    return "Sensors: " + ", ".join(bits)


def _response_hint(qtype: QuestionType) -> str:
    if qtype is QuestionType.NUMERICAL:
        return (
            "This is a NUMERICAL question. If your current view + evidence is enough "
            "to answer, emit {\"kind\": \"numerical\", \"value\": N}. Otherwise emit a "
            "waypoint_path that moves to a viewpoint giving better coverage."
        )
    if qtype is QuestionType.OBJECT_REFERENCE:
        return (
            "This is an OBJECT REFERENCE question. If you can localise the object "
            "now, emit {\"kind\": \"object_reference\", ...}. Otherwise emit a "
            "waypoint_path toward the most likely region."
        )
    return (
        "This is an INSTRUCTION FOLLOWING question. Emit a waypoint_path with one or "
        "more waypoints describing the path the robot should take next."
    )
