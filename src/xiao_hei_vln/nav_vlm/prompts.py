"""System prompt + tool schema for the navigation waypoint proposer.

The model is given two images (a first-person panorama and a top-down
occupancy map) and the robot's pose, and must call ``propose_waypoint``
exactly once. Forcing a tool call (rather than free text) is what makes the
output machine-parseable and keeps the coordinate frame unambiguous.
"""

from __future__ import annotations

# The tool the model MUST call. `done` lets it end exploration; otherwise
# (x, y) is a target in the *map* frame and the app snaps it to the nearest
# reachable free cell before publishing, so a slightly-off pick still lands
# somewhere the local planner can drive to.
PROPOSE_WAYPOINT_TOOL: dict = {
    "name": "propose_waypoint",
    "description": (
        "Propose the single next navigation waypoint for the robot, in the "
        "map coordinate frame (metres). Call this exactly once. Set done=true "
        "only when no unexplored, reachable area remains worth visiting."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "done": {
                "type": "boolean",
                "description": (
                    "true if exploration should stop (nothing useful left to "
                    "reach). When true, x/y/heading are ignored."
                ),
            },
            "x": {
                "type": "number",
                "description": "Target x in the map frame, metres.",
            },
            "y": {
                "type": "number",
                "description": "Target y in the map frame, metres.",
            },
            "heading": {
                "type": "number",
                "description": (
                    "Desired final yaw at the target, radians in the map "
                    "frame. Optional; omit to face the direction of travel."
                ),
            },
            "rationale": {
                "type": "string",
                "description": (
                    "One short sentence: why this target advances exploration "
                    "(what it lets the robot see or reach)."
                ),
            },
        },
        "required": ["done", "rationale"],
    },
}


SYSTEM_PROMPT = """\
You are the high-level navigation planner for a wheeled indoor robot. Your \
only job is to choose the SINGLE next waypoint the robot should drive to, so \
that it efficiently explores and understands the room.

You receive, each time you are called:
  - A first-person PANORAMA image: what the robot currently sees.
  - A top-down OCCUPANCY MAP image: white = free/traversable space the robot \
has already observed, black = walls/obstacles, light grey = UNKNOWN \
(unobserved). A green dot is the robot's current pose; a blue line is where \
it has already been; a red star (if present) is the waypoint that just failed.
  - Text: the robot's current pose (x, y, yaw) in the map frame, and — when \
the previous waypoint could not be reached — the reason it failed.

How to choose the next waypoint:
  - Prefer the boundary between white (free) and grey (unknown): that is where \
new area gets revealed. Send the robot toward large unexplored regions.
  - The waypoint MUST be in free (white) space or just inside it — never \
inside a wall (black) or floating in unknown (grey) with no free path to it. \
The robot can only follow paths through white space.
  - Keep each hop modest (roughly within a few metres of the robot) so the \
local planner can actually reach it; you will be called again on arrival.
  - If the previous waypoint FAILED as unreachable, do NOT propose the same \
spot or one right next to it — the robot is blocked there. Pick a different \
direction through confirmed free space.
  - Set done=true only when every reachable frontier has been visited and the \
grey regions left are walled off / unreachable.

Call propose_waypoint exactly once. Do not answer in prose.
"""


def build_user_text(
    *,
    robot_xy: tuple[float, float] | None,
    robot_yaw: float | None,
    failure_reason: str | None,
    visited: int,
    trajectory_summary: str,
) -> str:
    """Assemble the per-call user text block (pose + failure context)."""
    lines: list[str] = []
    if robot_xy is not None:
        yaw = f"{robot_yaw:.2f}" if robot_yaw is not None else "unknown"
        lines.append(
            f"Robot pose (map frame): x={robot_xy[0]:.2f}, y={robot_xy[1]:.2f}, "
            f"yaw={yaw} rad."
        )
    else:
        lines.append("Robot pose: unknown (no odometry yet).")
    lines.append(f"Waypoints reached so far: {visited}.")
    lines.append(trajectory_summary)
    if failure_reason:
        lines.append(
            "PREVIOUS WAYPOINT FAILED — " + failure_reason + " "
            "Pick a clearly different, reachable target."
        )
    lines.append(
        "Choose the next waypoint now by calling propose_waypoint."
    )
    return "\n".join(lines)
