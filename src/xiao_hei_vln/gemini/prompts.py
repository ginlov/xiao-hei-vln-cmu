"""Prompt builders for the marker-based Gemini responder."""

from __future__ import annotations

import json

from xiao_hei_vln.messages import ChallengeQuestion, OdomPose, QuestionType, VLMInput


NUMERICAL_SYSTEM_PROMPT = """\
You are a deterministic numerical reasoning module for the CMU VLN Challenge.
Compute a SINGLE INTEGER answer from ONLY the provided object markers and the
question. Use explicit geometry, not guesses.

Marker fields:
- object_id: stable scene ID
- label: object class
- center: x/y/z bbox center in map metres
- size: x/y/z bbox extents in metres
- heading: yaw in radians
- color: optional RGB estimate, null when unavailable

Spatial predicate conventions:
- ON / on top of: target bottom is close to reference top AND xy centers overlap.
- ABOVE / BELOW: compare z centers or bbox vertical ranges.
- CLOSEST / FURTHEST: compare Euclidean center distances unless the question
  clearly asks for floor distance, then use xy distance.
- BETWEEN: target center lies near the segment between the two reference centers.
- NEAR / BESIDE: use xy distance between centers.

Return valid JSON only:
{"answer": <non-negative int>, "reasoning": "<brief explicit math>"}
"""


OBJECT_REFERENCE_SYSTEM_PROMPT = """\
You are a deterministic object-reference resolver for the CMU VLN Challenge.
Use ONLY the provided object markers and spatial predicates. Select exactly one
marker that satisfies the question's object type, attributes, and spatial
constraints. Compute distances explicitly.

Return valid JSON only:
{"selected_marker_id": <int|null>, "reasoning": "<brief explanation>"}
"""


INSTRUCTION_SYSTEM_PROMPT = """\
You are the mission controller for sequential waypoint execution in the CMU VLN
Challenge. Convert the instruction into the next waypoint path. Use the current
image when available, the robot pose, terrain summary, prior evidence, and any
provided object markers as a scene-graph seed.

Return valid JSON only:
{"waypoints": [{"x": <float>, "y": <float>, "heading": <float>}],
 "reasoning": "<brief explanation>"}
"""


def build_system_prompt(qtype: QuestionType) -> str:
    if qtype is QuestionType.NUMERICAL:
        return NUMERICAL_SYSTEM_PROMPT
    if qtype is QuestionType.OBJECT_REFERENCE:
        return OBJECT_REFERENCE_SYSTEM_PROMPT
    return INSTRUCTION_SYSTEM_PROMPT


def build_user_message(
    snapshot: VLMInput,
    question: ChallengeQuestion,
    markers: list[dict],
    evidence_log: list[str],
) -> str:
    if question.type is QuestionType.NUMERICAL:
        return _numerical_user_message(snapshot, question, markers, evidence_log)
    if question.type is QuestionType.OBJECT_REFERENCE:
        return _object_reference_user_message(snapshot, question, markers, evidence_log)
    return _instruction_user_message(snapshot, question, markers, evidence_log)


def _numerical_user_message(
    snapshot: VLMInput,
    question: ChallengeQuestion,
    markers: list[dict],
    evidence_log: list[str],
) -> str:
    parts = [
        f"QUESTION: {question.text}",
        _pose_summary(snapshot.pose),
        "ALL SCENE MARKERS:",
        json.dumps(markers, indent=2, sort_keys=True),
    ]
    _append_evidence(parts, evidence_log)
    parts.append("Compute the integer count and respond with JSON.")
    return "\n".join(parts)


def _object_reference_user_message(
    snapshot: VLMInput,
    question: ChallengeQuestion,
    markers: list[dict],
    evidence_log: list[str],
) -> str:
    parts = [
        f"QUESTION: {question.text}",
        _pose_summary(snapshot.pose),
        f"SCENE MARKERS (n={len(markers)}):",
        json.dumps(markers, indent=2, sort_keys=True),
    ]
    _append_evidence(parts, evidence_log)
    parts.append("Select exactly one marker ID, or null only if no marker can satisfy the query.")
    return "\n".join(parts)


def _instruction_user_message(
    snapshot: VLMInput,
    question: ChallengeQuestion,
    markers: list[dict],
    evidence_log: list[str],
) -> str:
    parts = [
        f"INSTRUCTION: {question.text}",
        _pose_summary(snapshot.pose),
        _terrain_summary(snapshot),
        "KNOWN OBJECT MARKERS (scene-graph seed, may be incomplete):",
        json.dumps(markers, indent=2, sort_keys=True),
    ]
    _append_evidence(parts, evidence_log)
    parts.append("Return the next waypoint path as JSON.")
    return "\n".join(parts)


def _append_evidence(parts: list[str], evidence_log: list[str]) -> None:
    if not evidence_log:
        parts.append("Evidence so far: (none)")
        return
    parts.append("Evidence so far (oldest -> newest):")
    for i, entry in enumerate(evidence_log, start=1):
        parts.append(f"  {i}. {entry}")


def _pose_summary(pose: OdomPose | None) -> str:
    if pose is None:
        return "Pose: unknown"
    p = pose.position
    return f"Pose: x={p.x:.2f} y={p.y:.2f} z={p.z:.2f} (map frame)"


def _terrain_summary(snapshot: VLMInput) -> str:
    bits: list[str] = []
    if snapshot.terrain_local is not None:
        bits.append(f"terrain_local={snapshot.terrain_local.points.shape[0]} pts")
    if snapshot.terrain_ext is not None:
        bits.append(f"terrain_ext={snapshot.terrain_ext.points.shape[0]} pts")
    if snapshot.registered_scan is not None:
        bits.append(f"registered_scan={snapshot.registered_scan.points.shape[0]} pts")
    return "Sensors: " + (", ".join(bits) if bits else "no terrain/lidar in cache")
