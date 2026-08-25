"""Prompts + tool schemas for question-directed Task-1 navigation.

Two Claude roles share the :class:`~xiao_hei_vln.nav_vlm.engine.AnthropicNavEngine`
transport:

1. **Goal-directed navigator** (``propose_or_arrive`` tool) — called on every
   waypoint-reach / -skip event while the robot searches for and approaches the
   object named in an OBJECT_REFERENCE question. It reuses
   :class:`~xiao_hei_vln.nav_vlm.engine.WaypointProposal`, where ``done`` means
   "the robot has arrived at the target object" (not "exploration finished").
2. **Object-reference answerer** (``answer_object_reference`` tool) — called
   once at arrival, over the full scene-graph dump, to pick the ``object_id``
   that the referring expression denotes.

Keeping both here means the coordinate-frame and scene-graph conventions the
model must follow live in one place.
"""

from __future__ import annotations

import json
from typing import Any

# --- goal-directed navigation ---------------------------------------------

# Same shape as PROPOSE_WAYPOINT_TOOL, but ``done`` is repurposed: for Task-1
# navigation it means "arrived at the target object", so the parsed
# WaypointProposal.done flips the explorer to complete and the responder answers.
PROPOSE_OR_ARRIVE_TOOL: dict[str, Any] = {
    "name": "propose_or_arrive",
    "description": (
        "Either propose the single next navigation waypoint (map frame, "
        "metres) that moves the robot toward the target object, or declare "
        "that the robot has ARRIVED at the target object. Call this exactly "
        "once."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "done": {
                "type": "boolean",
                "description": (
                    "true ONLY when the robot has reached the target object "
                    "(it is close and clearly in view). When true, x/y/heading "
                    "are ignored and the robot stops to answer the question."
                ),
            },
            "x": {
                "type": "number",
                "description": "Next waypoint x in the map frame, metres.",
            },
            "y": {
                "type": "number",
                "description": "Next waypoint y in the map frame, metres.",
            },
            "heading": {
                "type": "number",
                "description": (
                    "Desired final yaw at the waypoint, radians in the map "
                    "frame. Optional; omit to face the direction of travel."
                ),
            },
            "rationale": {
                "type": "string",
                "description": (
                    "One short sentence: why this waypoint moves toward the "
                    "target object, or why the robot has now arrived."
                ),
            },
            "visible_objects": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Objects you can SEE in the panorama right now, as short "
                    "class names for an open-vocabulary detector — ESPECIALLY "
                    "the target and any rare/uncommon items, each with 1-2 "
                    "alternate phrasings (e.g. 'hookah', 'shisha', 'water pipe'). "
                    "This primes the detector to find them. 4-12 items."
                ),
            },
            "verify_objects": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Objects you can clearly SEE right now but that are NOT in "
                    "the scene-graph list — ask the detector to look harder for "
                    "them. Each request lowers the detection threshold for that "
                    "class only. Use the class name(s) + synonyms. Leave empty "
                    "once the object is in the scene graph."
                ),
            },
        },
        "required": ["done", "rationale"],
    },
}


NAV_SYSTEM_PROMPT = """\
You are the high-level navigation planner for a wheeled indoor robot on an \
object-search task. You are given a QUESTION that refers to a single target \
object somewhere in the room. Your job is to drive the robot to that object and \
then declare arrival, so the robot can answer the question from up close.

You receive, each time you are called:
  - A first-person PANORAMA image: what the robot currently sees.
  - A top-down OCCUPANCY MAP image: white = free/traversable space already \
observed, black = walls/obstacles, light grey = UNKNOWN (unobserved). A green \
dot is the robot's current pose; a blue line is where it has been; a red star \
(if present) is the waypoint that just failed. Labelled ORANGE squares are the \
objects detected so far, drawn at their map positions — find the target among \
them and read a route to it THROUGH white space, stepping around black walls.
  - Text: the QUESTION, the robot's pose (x, y, yaw) in the map frame, the list \
of objects DETECTED SO FAR (the scene graph — each with an id, label, and map \
position), and — if the last waypoint could not be reached — why it failed.

MATCHING THE TARGET TO THE SCENE GRAPH (read this first):
The scene-graph labels are produced by an automatic open-vocabulary detector, \
so they OFTEN USE A DIFFERENT WORD than the question for the same physical \
object — e.g. "sofa"="couch", "rug"="carpet", "tv"="television"/"monitor", \
"painting"="picture"/"artwork", "plant"="potted plant", "bin"="trash can", \
"fridge"="refrigerator". Match the question's target to the detected objects by \
MEANING, not by exact spelling: a synonym, the same object category, or a \
more/less specific term all count as the target. Be precise, though — a merely \
related object is NOT a match (a chair is not a sofa; a stool is not a table). \
Re-check the object list on EVERY call: detections accumulate as the robot \
moves, so the target often appears only after a few waypoints.

How to act:
  - FIRST scan the listed objects for one that IS the target by the meaning \
rule above. If a plausible match exists, drive toward it (the nearest one, or \
the one that best fits any spatial relation the question states).
  - When the robot is near the matched object, do NOT stop and answer \
immediately. First build a better scene graph of it: propose 1-3 waypoints that \
ORBIT the object — each a short, safe distance away on a DIFFERENT side than you \
have already viewed (use the blue trajectory line and the object's map position \
to pick an angle you have not seen yet). Observing it from several sides lets \
perception fuse a more accurate 3D box, position, and colour. Keep a safe gap; \
never drive onto or into the object.
  - Set done=true (arrived) once the object has been observed from a few \
distinct angles — aim for its "[seen from N view(s)]" count to reach about 3 — \
OR once further orbiting is blocked/unreachable. You answer from the scene \
graph, not by touching the object, so a safe standoff distance is correct.
  - Only if NO plausible match is in the scene graph yet, move toward \
unobserved area (the white/grey boundary), preferring directions the panorama \
suggests the object could be, to reveal more of the room until it is detected.
  - Every waypoint MUST be in free (white) space or just inside it — never in a \
wall (black) or floating in unknown (grey) with no free path to it. Keep each \
hop modest (a few metres) so the local planner can reach it; you are called \
again on arrival.
  - Red stars mark waypoints that were UNREACHABLE. Never re-propose one or a \
spot right beside it. When several stars cluster between the robot and the \
target, the direct path is BLOCKED — do not keep pushing into it; instead trace \
a detour through white free space around the obstacle, even if the first hop \
goes sideways or briefly away from the target. The local planner cannot punch \
through walls/furniture, so YOU must route around them.

Also list, in visible_objects, the things you can actually SEE in the panorama \
right now — especially the target and any rare/uncommon items — each with a \
couple of alternate names, so the object detector is primed to find them (this \
is how rare objects the detector's default vocabulary misses, like a hookah, \
get detected). And if you can clearly see an object that is NOT yet in the \
scene-graph list, put its class name(s) in verify_objects to make the detector \
look harder for it (each request lowers the detection threshold for that class \
only) — drop it from verify_objects once it appears in the scene graph.

Call propose_or_arrive exactly once. Do not answer in prose.
"""


def scene_objects_summary(scene_dict: dict[str, Any], *, limit: int = 60) -> str:
    """One line per detected object: id, label, colour, map position.

    A compact, model-legible digest of the scene graph for the *navigation*
    prompt (the full JSON is reserved for the answer call). Capped so a busy
    scene cannot blow the prompt budget.
    """
    objects = scene_dict.get("objects", []) or []
    if not objects:
        return "Objects detected so far: (none yet)."
    lines = ["Objects detected so far (scene graph):"]
    for o in objects[:limit]:
        colour = o.get("color_name")
        colour_str = f" {colour}" if colour else ""
        # Number of distinct viewpoints that observed this object — the
        # coverage signal the navigator uses to decide whether it has orbited
        # the target enough to answer well.
        views = o.get("observing_viewpoint_ids") or []
        seen = f" [seen from {len(views)} view(s)]" if views else ""
        lines.append(
            f"  - #{o.get('object_id')}{colour_str} {o.get('label')} "
            f"at {_fmt_xy(o.get('position'))}{seen}"
        )
    if len(objects) > limit:
        lines.append(f"  ... ({len(objects) - limit} more omitted)")
    return "\n".join(lines)


def _fmt_xy(pos) -> str:
    """Format an (x, y) from a scene-graph position, tolerating either shape.

    ``SceneRepresentation.to_dict`` serialises a Vector3 as a ``[x, y, z]``
    list; a dict ``{"x":…, "y":…}`` is also accepted. Never raises.
    """
    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
        x, y = pos[0], pos[1]
    elif isinstance(pos, dict):
        x, y = pos.get("x"), pos.get("y")
    else:
        return "(?)"
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return f"({x:.2f},{y:.2f})"
    return "(?)"


def build_nav_user_text(
    *,
    question: str,
    robot_xy: tuple[float, float] | None,
    robot_yaw: float | None,
    scene_summary: str,
    failure_reason: str | None,
    failed_points: list[tuple[float, float]] | None = None,
    waypoints_taken: int,
    trajectory_summary: str,
    mode: str = "reach",
) -> str:
    """Assemble the per-call user text for the goal-directed navigator.

    ``mode`` is ``"reach"`` (drive to the target) or ``"explore"`` (the target
    could not be reached from here — reveal more of the room to find a new,
    reachable approach).
    """
    lines: list[str] = [f"QUESTION (target object): {question}", ""]
    if mode == "explore":
        lines.append(
            "MODE: EXPLORE. The target could not be reached from this area after "
            "several tries. Stop pushing toward it for now — instead reveal more "
            "of the room to find a different, reachable side of it."
        )
        lines.append("")
    if robot_xy is not None:
        yaw = f"{robot_yaw:.2f}" if robot_yaw is not None else "unknown"
        lines.append(
            f"Robot pose (map frame): x={robot_xy[0]:.2f}, y={robot_xy[1]:.2f}, "
            f"yaw={yaw} rad."
        )
    else:
        lines.append("Robot pose: unknown (no odometry yet).")
    lines.append(f"Waypoints driven so far: {waypoints_taken}.")
    lines.append(trajectory_summary)
    lines.append("")
    lines.append(scene_summary)
    fails = failed_points or []
    if len(fails) >= 2:
        pts = ", ".join(f"({x:.2f},{y:.2f})" for x, y in fails)
        lines.append("")
        lines.append(
            f"{len(fails)} recent waypoints were UNREACHABLE (red stars on the "
            f"map): {pts}. The direct approach is BLOCKED — do NOT keep aiming "
            "through that region or nudging just next to those points. ROUTE "
            "AROUND it: pick a waypoint that reaches the target through clearly "
            "free (white) space, even if it first moves the robot sideways or "
            "away from the target to get around the obstacle."
        )
    elif failure_reason:
        lines.append("")
        lines.append(
            "PREVIOUS WAYPOINT FAILED — " + failure_reason + " "
            "Pick a clearly different, reachable target — route around the "
            "obstacle through free (white) space, not straight at it again."
        )
    lines.append("")
    if mode == "explore":
        lines.append(
            "Call propose_or_arrive with a waypoint toward a large UNOBSERVED "
            "(grey) region reachable through free (white) space, to reveal more "
            "of the room and a new approach to the target. Set done=true only if "
            "you can now clearly reach the target from here."
        )
    else:
        lines.append(
            "Decide now by calling propose_or_arrive: propose the next waypoint "
            "toward the target object, or set done=true if the robot has arrived."
        )
    return "\n".join(lines)


# --- object-reference answering --------------------------------------------

ANSWER_OBJECT_REFERENCE_TOOL: dict[str, Any] = {
    "name": "answer_object_reference",
    "description": (
        "Select the single scene-graph object that the referring expression "
        "denotes, by its object_id. Call this exactly once."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "object_id": {
                "type": "integer",
                "description": (
                    "The object_id (from the scene graph) of the object the "
                    "question refers to. Use -1 only if no object matches."
                ),
            },
            "label": {
                "type": "string",
                "description": "The chosen object's label, copied from the scene graph.",
            },
            "rationale": {
                "type": "string",
                "description": (
                    "One short sentence: why this object satisfies the "
                    "referring expression (label + spatial relation)."
                ),
            },
        },
        "required": ["object_id", "rationale"],
    },
}


ANSWER_SYSTEM_PROMPT = """\
You resolve a referring expression against a robot's scene graph. You are given \
a QUESTION naming an object (often with a spatial relation, e.g. "the red \
pillow closest to the sushi"), a JSON scene graph of objects the robot detected \
while driving around (each with object_id, label, colour, and map-frame \
position/bounding box), and images (the robot's view and a top-down occupancy \
map). Reason over BOTH the scene graph and the images together — the scene \
graph is not always complete, and your own view of the room fills the gaps.

Pick the ONE object_id that the expression denotes:
  - Match the object type by MEANING, not exact wording. The scene-graph labels \
come from an automatic open-vocabulary detector and often differ from the \
question's word for the same object — e.g. "sofa"="couch", "rug"="carpet", \
"tv"="television", "painting"="picture", "plant"="potted plant", \
"bin"="trash can". A true synonym, the same object category, or a more/less \
specific term is a match; a merely related object (a chair for a sofa) is not.
  - Apply the spatial relation using the map-frame positions in the scene graph \
(closest/farthest/nearest-to, left/right, etc.). Distances are Euclidean over \
the (x, y) positions.
  - The relation's ANCHOR object (the thing the target is described relative to) \
may be MISSING from the scene graph — the detector can fail on rarer objects \
even when they are plainly visible. When the anchor is absent, do NOT give up: \
use the IMAGES and your understanding of the room to judge where that anchor is, \
then pick the candidate target whose scene-graph position best fits the relation \
to it. Only the anchor may be inferred visually — the target you return must be \
a real object_id from the scene graph.
  - If several match the label but the relation is ambiguous, choose the most \
likely one and say why.
  - Use object_id = -1 only if no candidate for the TARGET object exists in the \
graph at all.

Call answer_object_reference exactly once. Do not answer in prose.
"""


# --- question → detector vocabulary (one-time, at question start) ----------

OBJECT_VOCAB_TOOL: dict[str, Any] = {
    "name": "object_vocabulary",
    "description": (
        "List the object types the question refers to, each with common "
        "alternate names an open-vocabulary object detector might match. "
        "Call this exactly once."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "classes": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Detector class names for EVERY object the question "
                    "mentions (the target and any object it is related to), each "
                    "with the exact word PLUS 1-3 common synonyms/alternate "
                    "names — e.g. for 'hookah': 'hookah','shisha','water pipe', "
                    "'nargile'; for 'sofa': 'sofa','couch'. Short noun phrases, "
                    "4-16 items total."
                ),
            },
        },
        "required": ["classes"],
    },
}

VOCAB_SYSTEM_PROMPT = """\
You expand a robot-navigation question into a list of object-detector class \
names. Given the question, list every object TYPE it mentions — the target \
object and any object the target is described relative to — and for each give \
the plain word plus a few common synonyms / alternate names that an \
open-vocabulary detector (YOLO-World) might match, so rare words the detector's \
default vocabulary would miss still get found. Use short lowercase noun phrases. \
Call object_vocabulary exactly once. Do not answer in prose.
"""


def build_vocab_user_text(question: str) -> str:
    return (
        f"QUESTION: {question}\n\n"
        "List the detector class names + synonyms for every object this "
        "question mentions, by calling object_vocabulary."
    )


def build_answer_user_text(*, question: str, scene_dict: dict[str, Any]) -> str:
    """Assemble the answer-call user text: question + full scene-graph JSON."""
    return (
        f"QUESTION: {question}\n\n"
        "Scene graph (JSON from SceneRepresentation.to_dict):\n"
        f"```json\n{json.dumps(scene_dict, indent=2)}\n```\n\n"
        "Select the object_id the question refers to by calling "
        "answer_object_reference."
    )
