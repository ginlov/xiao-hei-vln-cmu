"""Compact scene-graph helpers shared by spatial resolver + offline batch.

This is the scoring-side subset of Gemini packaging (size clamps, label
families, reference filtering). It does not depend on live multimodal
prompt builders.
"""

from __future__ import annotations

from typing import Any

from xiao_hei_vln.perception.size_priors import clamp_aabb_size
from xiao_hei_vln.scene import SceneRepresentation

# Large / architectural classes: keep even when AABB is big (windows, sofas…).
_LARGE_OK = {
    "wall", "floor", "ceiling", "room", "column", "door", "door frame",
    "window", "windows", "window frame", "exterior walls",
    "couch", "sofa", "table", "coffee table", "dining table", "desk",
    "shelf", "bookcase", "cabinet", "tv cabinet", "bed", "rug", "carpet",
    "bench", "counter", "kitchen island",
}


def compact_objects(scene: SceneRepresentation) -> list[dict[str, Any]]:
    """Token-lean per-object view with size-prior clamps + near-dup merge."""
    raw: list[tuple[float, dict[str, Any]]] = []
    for o in scene.objects:
        conf = float(getattr(o, "confidence", 1.0) or 1.0)
        label_l = (o.label or "").lower().strip()
        if not label_l or label_l in {"wall", "floor", "ceiling", "room"}:
            continue
        item: dict[str, Any] = {
            "id": o.object_id,
            "label": o.label,
            "center": [
                round(o.position.x, 2),
                round(o.position.y, 2),
                round(o.position.z, 2),
            ],
            "conf": round(conf, 2),
        }
        if o.bbox_min is not None and o.bbox_max is not None:
            sx = max(0.01, o.bbox_max.x - o.bbox_min.x)
            sy = max(0.01, o.bbox_max.y - o.bbox_min.y)
            sz = max(0.01, o.bbox_max.z - o.bbox_min.z)
            if label_l in _LARGE_OK and (sx * sy * sz) > 80.0:
                continue
            size = clamp_aabb_size(label_l, [sx, sy, sz])
            if (
                label_l == "vase"
                and float(o.position.z) < 0.55
                and size[2] < 1.0
                and sz >= 0.55
            ):
                size = [max(size[0], 0.28), max(size[1], 0.28), 2.40]
                item["center"] = [
                    round(o.position.x, 2),
                    round(o.position.y, 2),
                    1.25,
                ]
            item["size"] = [round(v, 2) for v in size]
        color = getattr(o, "color_name", None)
        if color:
            item["color"] = color
        raw.append((conf, item))

    raw.sort(key=lambda t: t[0], reverse=True)
    kept: list[dict[str, Any]] = []
    for _conf, item in raw:
        lab = str(item["label"]).lower()
        cx, cy, _ = item["center"]
        radius = 0.55 if lab in _LARGE_OK else 0.35
        dup = False
        for prev in kept:
            if str(prev["label"]).lower() != lab:
                continue
            px, py, _ = prev["center"]
            if (cx - px) ** 2 + (cy - py) ** 2 <= radius ** 2:
                dup = True
                break
        if not dup:
            kept.append(item)
    return kept


# Morphological / open-vocab expansions from question words → detector labels.
_LABEL_FAMILIES: dict[str, tuple[str, ...]] = {
    "bottle": ("bottle", "beer bottle", "wine bottle"),
    "beer": ("beer bottle", "bottle", "wine bottle"),
    "box": ("box", "round box"),
    "book": ("book", "books"),
    "books": ("book", "books"),
    "window": ("window", "windows"),
    "windows": ("window", "windows"),
    "couch": ("couch", "sofa"),
    "sofa": ("couch", "sofa"),
    "table": ("table", "coffee table", "dining table", "night stand", "nightstand"),
    "shelf": ("shelf", "bookcase"),
    "bookcase": ("bookcase", "shelf"),
    "lamp": ("lamp", "wall lamp", "ceiling lamp", "ceiling light", "focus light", "spot light"),
    "light": ("lamp", "wall lamp", "ceiling lamp", "ceiling light", "focus light", "spot light", "light switch"),
    "focus": ("focus light", "spot light", "ceiling light", "lamp"),
    "picture": ("picture", "painting", "poster", "photo", "drawing", "canvas", "framed record"),
    "framed": ("framed record", "picture", "painting", "poster", "photo"),
    "record": ("framed record", "picture", "painting", "poster"),
    "drawing": ("drawing", "picture", "canvas", "painting", "easel"),
    "easel": ("easel", "canvas", "drawing"),
    "vase": ("vase",),
    "chair": ("chair", "dining chair", "stool"),
    "door": ("door", "door frame", "balcony door", "kitchen door"),
    "plant": ("potted plant",),
    "potted": ("potted plant",),
    "newspaper": ("newspaper", "magazine", "paper"),
    "phone": ("phone",),
    "switch": ("light switch",),
}


def _question_label_vocab(question: str) -> set[str]:
    from xiao_hei_vln.perception.vocab import Vocabulary

    text = question.lower()
    keep: set[str] = set()
    for phrase in Vocabulary(prior=()).current_classes(question):
        keep.add(phrase)
        keep.update(_LABEL_FAMILIES.get(phrase, ()))
        for tok in phrase.split():
            keep.update(_LABEL_FAMILIES.get(tok, ()))
            keep.add(tok)
    for fam_key, members in _LABEL_FAMILIES.items():
        if fam_key in text:
            keep.update(members)
    for phrase in Vocabulary().prior:
        if " " in phrase and phrase in text:
            keep.add(phrase)
            for tok in phrase.split():
                keep.update(_LABEL_FAMILIES.get(tok, ()))
    return {k for k in keep if k and len(k) >= 2}


def filter_objects_for_reference(
    objects: list[dict[str, Any]],
    question: str,
) -> list[dict[str, Any]]:
    """Keep target-like + anchor-like objects for object-reference prompts."""
    wanted = {w.lower() for w in _question_label_vocab(question)}
    if not wanted:
        return objects

    def matches(label: str) -> bool:
        lab = label.lower()
        if lab in wanted:
            return True
        parts = lab.split()
        return any(p in wanted for p in parts) or any(
            w in lab for w in wanted if " " in w
        )

    filtered = [o for o in objects if matches(str(o.get("label") or ""))]
    return filtered if len(filtered) >= 2 else objects
