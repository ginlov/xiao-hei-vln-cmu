"""Convert a VLA-3D JSONL entry into a ground-truth VLMOutput.

Numerical entries:   answer is an int  → NumericalResponse(value=answer)
Object-ref entries:  answer is {"object_id": N, "label": "..."} → ObjectReferenceResponse
                     bbox is looked up from the entry's object_list field.
"""

from __future__ import annotations

from xiao_hei_vln.eval_sampler.object_list import parse_object_list
from xiao_hei_vln.messages.outputs import NumericalResponse, ObjectReferenceResponse, VLMOutput


def gt_from_entry(entry: dict) -> VLMOutput | None:
    """Return the ground-truth VLMOutput for a VLA-3D JSONL entry, or None on error."""
    qtype = entry.get("type", "")

    if qtype == "numerical":
        answer = entry.get("answer")
        if not isinstance(answer, int) or isinstance(answer, bool):
            return None
        return NumericalResponse(value=answer)

    if qtype == "object_reference":
        return _object_ref_gt(entry)

    return None


def _object_ref_gt(entry: dict) -> ObjectReferenceResponse | None:
    answer = entry.get("answer", {})
    object_id = answer.get("object_id")
    label = answer.get("label", "")
    if object_id is None:
        return None

    try:
        oid = int(object_id)
    except (TypeError, ValueError):
        return None

    # Bbox comes from the object_list field — the answer dict only has id+label.
    # entry.get("object_list", []) can return None if the field is explicitly null.
    raw = entry.get("object_list")
    object_list = raw if isinstance(raw, list) else []
    lookup = parse_object_list(object_list)
    obj = lookup.get(oid)
    if obj is None:
        return None

    return ObjectReferenceResponse(
        label=label or obj.label,
        object_id=oid,
        center=obj.center,
        size=obj.size,
        heading=obj.heading,
    )
