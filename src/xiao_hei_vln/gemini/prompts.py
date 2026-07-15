"""Prompt builders for the Gemini responder.

Three system prompts — one per question type — each strict about the
JSON shape Gemini must return. Combined with Gemini's
``response_mime_type='application/json'`` + ``response_schema`` settings
on the engine, this gives near-zero parsing failures.

Per-tick user-message builders inject the question text, pose summary,
trajectory hint, and any free-form context the caller wants.

Pure-Python; no PIL / no SDK imports — testable without the gemini
extra installed.
"""

from __future__ import annotations

from xiao_hei_vln.messages import ChallengeQuestion, OdomPose, QuestionType, VLMInput

# --- system prompts --------------------------------------------------------

SYSTEM_PROMPT_NUMERICAL = """\
You are the vision-language model on board a wheeled robot competing in
the CMU Vision-Language-Navigation Challenge, currently committing the
final answer to a NUMERICAL question of the form "How many <target>...?".

The robot's exploration system has already driven through the room and
will give you, in this prompt:

  - the question text
  - the panorama from the robot's current view
  - a top-down occupancy map (white = free, black = obstacle/wall,
    grey = unobserved) with the robot's trajectory overlaid in blue
  - the robot's current pose (x, y, heading) in the map frame
  - the recent trajectory as a list of (x, y) waypoints

Your task: count the instances of <target> across the observed scene,
and return EXACTLY one JSON object of this shape:

  {"kind": "numerical", "value": <non-negative int>, "rationale": "<≤200 chars>"}

Rules:
  - Use the panorama for visible-instance evidence and the occupancy map
    for spatial coverage. Areas marked grey are UNOBSERVED — be honest
    about uncertainty in the rationale if those areas could plausibly
    contain instances.
  - If you cannot confidently count, return your best estimate as
    ``value`` and explain the uncertainty in ``rationale``. Do NOT
    refuse to answer.
  - ``value`` MUST be a non-negative integer. ``rationale`` MUST be
    present.
  - Do not include any text outside the JSON object.
"""

SYSTEM_PROMPT_OBJECT_REFERENCE = """\
You are the vision-language model on board a wheeled robot in the CMU
Vision-Language-Navigation Challenge, committing the final answer to an
OBJECT-REFERENCE question of the form "Find the <referring expression>".

The exploration system has driven the robot through the room and
provides:

  - the question text (a referring expression)
  - the current panorama
  - a top-down occupancy map with the trajectory overlaid
  - the robot's current pose (x, y) and recent trajectory in the map frame

Your task: identify the unique object the expression refers to and
return its 3D axis-aligned bounding box in the map frame. EXACTLY one
JSON object:

  {"kind": "object_reference",
   "label": "<short noun phrase>",
   "object_id": 0,
   "center": {"x": <float>, "y": <float>, "z": <float>},
   "size":   {"x": <float>, "y": <float>, "z": <float>},
   "heading": <float radians, default 0.0>,
   "rationale": "<≤200 chars>"}

Rules:
  - Coordinates are metres in the map frame; the robot's current pose is
    provided so you can reason relative to it.
  - Estimate ``size`` from typical real-world dimensions of the object
    class if you cannot infer it from the image (e.g., a sofa is roughly
    2.0 × 0.9 × 0.8 m).
  - ``object_id`` is a single integer; if you have no scheme, output 0.
  - Do not include any text outside the JSON object.
"""

SYSTEM_PROMPT_ROUTE = """\
You are the vision-language model on board a wheeled robot in the CMU
Vision-Language-Navigation Challenge, planning a route for an
INSTRUCTION-FOLLOWING question (e.g., "Go to the kitchen counter and
then walk past the sofa").

The exploration system provides:

  - the instruction text
  - the current panorama
  - a top-down occupancy map with the trajectory overlaid
  - the robot's current pose (x, y, heading) and recent trajectory

Your task: output the sequence of waypoints in the map frame that
satisfies the instruction. EXACTLY one JSON object:

  {"kind": "waypoint_path",
   "waypoints": [
     {"x": <float>, "y": <float>, "heading": <float radians>},
     ...
   ],
   "rationale": "<≤200 chars>"}

Rules:
  - Coordinates are metres in the map frame.
  - ``waypoints`` must contain at least one entry. Use 2-6 waypoints for
    multi-step instructions; one waypoint if the instruction is a simple
    "go to X".
  - Waypoints should sit on FREE cells in the occupancy map (white).
    Never plot a waypoint inside a black obstacle cell.
  - ``heading`` is the desired yaw at each waypoint (radians, 0 = +x
    axis). If the instruction does not constrain heading, point toward
    the next waypoint.
  - Do not include any text outside the JSON object.
"""


def build_system_prompt(qtype: QuestionType) -> str:
    """Return the appropriate system prompt for the question type."""
    if qtype is QuestionType.NUMERICAL:
        return SYSTEM_PROMPT_NUMERICAL
    if qtype is QuestionType.OBJECT_REFERENCE:
        return SYSTEM_PROMPT_OBJECT_REFERENCE
    if qtype is QuestionType.INSTRUCTION_FOLLOWING:
        return SYSTEM_PROMPT_ROUTE
    raise ValueError(f"Unsupported question type for Gemini: {qtype!r}")


# --- per-tick user messages ------------------------------------------------


def build_user_message(
    snapshot: VLMInput,
    question: ChallengeQuestion,
    trajectory_xy: list[tuple[float, float]] | None = None,
    *,
    exploration_summary: str | None = None,
) -> str:
    """Render the user message that accompanies the multimodal images.

    ``trajectory_xy`` is the last K (x, y) poses the robot has visited;
    ``exploration_summary`` is an optional free-text status line from
    the exploration phase (e.g., "Phase A frontier exhausted; 95% of
    reachable area observed").
    """
    parts: list[str] = [
        f"Question (type={question.type.value}): {question.text}",
        _pose_summary(snapshot.pose),
        _trajectory_summary(trajectory_xy),
        _terrain_summary(snapshot),
    ]
    if exploration_summary:
        parts.append(f"Exploration status: {exploration_summary}")
    parts.append(_response_hint(question.type))
    parts.append("Respond now with the JSON object — no prose around it.")
    return "\n".join(parts)


# --- helpers ---------------------------------------------------------------


def _pose_summary(pose: OdomPose | None) -> str:
    if pose is None:
        return "Pose: unknown (no /state_estimation yet)"
    p = pose.position
    return f"Pose: x={p.x:.2f} y={p.y:.2f} z={p.z:.2f} (map frame, metres)"


def _trajectory_summary(traj: list[tuple[float, float]] | None) -> str:
    if not traj:
        return "Trajectory: (no recorded waypoints yet)"
    if len(traj) <= 8:
        rendered = ", ".join(f"({x:.1f},{y:.1f})" for x, y in traj)
    else:
        # First 2, last 4 — caller has plenty of context already in the
        # occupancy PNG overlay.
        head = ", ".join(f"({x:.1f},{y:.1f})" for x, y in traj[:2])
        tail = ", ".join(f"({x:.1f},{y:.1f})" for x, y in traj[-4:])
        rendered = f"{head}, ... ({len(traj) - 6} omitted) ..., {tail}"
    return f"Recent trajectory ({len(traj)} samples): {rendered}"


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
            "This is a NUMERICAL question. Commit to a non-negative integer "
            "count. Use the occupancy map's grey regions to flag any uncertainty."
        )
    if qtype is QuestionType.OBJECT_REFERENCE:
        return (
            "This is an OBJECT-REFERENCE question. Localise the unique object "
            "and emit its 3D bbox in the map frame."
        )
    return (
        "This is an INSTRUCTION-FOLLOWING question. Plan a waypoint path through "
        "the FREE region of the occupancy map satisfying the instruction."
    )
