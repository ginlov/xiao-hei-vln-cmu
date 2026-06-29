"""Generate dataset/challenge_gt.jsonl from questions.json + VLA-3D scene data.

For each of the 45 scoreable challenge questions (numerical + object_reference,
across 15 scenes), this script resolves the ground-truth answer using:
  - VLA-3D scene graph spatial relations (on, above, below, near, between, closest, farthest)
  - Euclidean distance computation as a fallback / cross-check
  - Scene object_list.txt from the challenge scene zips (authoritative IDs + bboxes)

Output format per line (eval-sampler compatible):
  {
    "scene": "arabic_room",
    "type": "object_reference" | "numerical",
    "question": "<exact text from questions.json>",
    "answer": {"object_id": N, "label": "..."} | <int>,
    "object_list": ["id cx cy cz lx ly lz heading \"label\"", ...]
  }

Unresolvable questions are flagged with a stderr warning and skipped.

Usage:
    uv run python dataset_generator/challenge_gt_gen.py
    uv run python dataset_generator/challenge_gt_gen.py --verify   # print answers for review
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from vla3d_loader import VLAObject, VLAScene, load_vla_scene

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE = Path(__file__).parent
ROOT = HERE.parent
CHALLENGE_REPO = Path("/home/leo/Projects/CMU-VLN-Challenge-2026")
CHALLENGE_DATA = Path("/home/leo/Projects/CMU-VLN-Challenge-data/unity_env_models")
VLA3D_ROOT = HERE / "vla-3d" / "Unity"
QUESTIONS_JSON = CHALLENGE_REPO / "questions" / "questions.json"
OUTPUT = ROOT / "dataset" / "challenge_gt.jsonl"

# ---------------------------------------------------------------------------
# Color aliases (VLA-3D color names → human color names)
# ---------------------------------------------------------------------------

COLOR_ALIASES: dict[str, set[str]] = {
    "red":    {"red", "maroon", "crimson", "scarlet", "burgundy"},
    "blue":   {"blue", "navy", "cobalt", "indigo", "teal"},
    "black":  {"black", "dark", "charcoal", "ebony"},
    "white":  {"white", "ivory", "cream", "beige"},
    "green":  {"green", "olive", "forest", "lime"},
    "yellow": {"yellow", "gold", "amber"},
    "purple": {"purple", "violet", "lavender"},
    "orange": {"orange", "coral"},
    "brown":  {"brown", "tan", "khaki"},
    "gray":   {"gray", "grey", "silver"},
}


def _color_matches(obj_colors: tuple, color_name: str) -> bool:
    """Return True if any of obj_colors matches color_name (with aliases)."""
    aliases = COLOR_ALIASES.get(color_name.lower(), {color_name.lower()})
    for c in obj_colors:
        if c and c.lower() not in ("n/a", "_", "") and c.lower() in aliases:
            return True
    return False


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def dist3(a: VLAObject, b: VLAObject) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def dist3_xy(a: VLAObject, b: VLAObject) -> float:
    """2-D (x, y) Euclidean distance — ignores height."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


# ---------------------------------------------------------------------------
# Scene-graph / label helpers
# ---------------------------------------------------------------------------


def _label_match(label: str, query: str) -> bool:
    """Return True if the query string is a substring of label (case-insensitive).

    Directional: the query must appear inside the label — not the other way around.
    This prevents short labels like 'wall' or 'paper' from matching long queries
    like 'wall lamp' or 'paper cup'.
    """
    return query.lower() in label.lower()


def objs(sc: VLAScene, *labels: str) -> list[VLAObject]:
    """Return all objects whose raw_label contains any of the given query strings."""
    result: list[VLAObject] = []
    seen: set[int] = set()
    for o in sc.objects:
        for q in labels:
            if _label_match(o.raw_label, q) and o.id not in seen:
                result.append(o)
                seen.add(o.id)
                break
    return result


def objs_exact(sc: VLAScene, label: str) -> list[VLAObject]:
    """Exact (case-insensitive) label match."""
    lower = label.lower()
    return [o for o in sc.objects if o.raw_label.lower() == lower]


def on_items(sc: VLAScene, container_label: str, item_label: Optional[str] = None) -> list[tuple[VLAObject, VLAObject]]:
    """Return (item, container) pairs where item is ON a container matching container_label."""
    pairs: list[tuple[VLAObject, VLAObject]] = []
    containers = objs(sc, container_label)
    for c in containers:
        for rel_dict in sc.relationships.values():
            for t_id in rel_dict.get("on", {}).get(c.id, []):
                if not isinstance(t_id, int):
                    continue
                item = sc.by_id.get(t_id)
                if item is None:
                    continue
                if item_label is None or _label_match(item.raw_label, item_label):
                    pairs.append((item, c))
    return pairs


def items_above(sc: VLAScene, anchor_label: str, item_label: Optional[str] = None) -> list[tuple[VLAObject, VLAObject]]:
    """Return (item, anchor) pairs where item is ABOVE anchor."""
    pairs: list[tuple[VLAObject, VLAObject]] = []
    anchors = objs(sc, anchor_label)
    for a in anchors:
        for rel_dict in sc.relationships.values():
            for t_id in rel_dict.get("above", {}).get(a.id, []):
                if not isinstance(t_id, int):
                    continue
                item = sc.by_id.get(t_id)
                if item is None:
                    continue
                if item_label is None or _label_match(item.raw_label, item_label):
                    pairs.append((item, a))
    return pairs


def items_below(sc: VLAScene, anchor_label: str, item_label: Optional[str] = None) -> list[tuple[VLAObject, VLAObject]]:
    """Return (item, anchor) pairs where item is BELOW anchor."""
    pairs: list[tuple[VLAObject, VLAObject]] = []
    anchors = objs(sc, anchor_label)
    for a in anchors:
        for rel_dict in sc.relationships.values():
            for t_id in rel_dict.get("below", {}).get(a.id, []):
                if not isinstance(t_id, int):
                    continue
                item = sc.by_id.get(t_id)
                if item is None:
                    continue
                if item_label is None or _label_match(item.raw_label, item_label):
                    pairs.append((item, a))
    return pairs


def near_items(sc: VLAScene, anchor_label: str, item_label: Optional[str] = None) -> list[tuple[VLAObject, VLAObject]]:
    """Return (item, anchor) pairs where item is NEAR anchor."""
    pairs: list[tuple[VLAObject, VLAObject]] = []
    anchors = objs(sc, anchor_label)
    for a in anchors:
        for rel_dict in sc.relationships.values():
            for t_id in rel_dict.get("near", {}).get(a.id, []):
                if not isinstance(t_id, int):
                    continue
                item = sc.by_id.get(t_id)
                if item is None:
                    continue
                if item_label is None or _label_match(item.raw_label, item_label):
                    pairs.append((item, a))
    return pairs


def between_items(sc: VLAScene, item_label: str, anchor1_label: str, anchor2_label: str) -> list[VLAObject]:
    """Return items with item_label that are between anchor1 and anchor2 per scene graph."""
    result: list[VLAObject] = []
    anchor1s = objs(sc, anchor1_label)
    anchor2s = objs(sc, anchor2_label)
    a1_ids = {a.id for a in anchor1s}
    a2_ids = {a.id for a in anchor2s}
    for rel_dict in sc.relationships.values():
        for src_id, pairs in rel_dict.get("between", {}).items():
            item = sc.by_id.get(src_id)
            if item is None or not _label_match(item.raw_label, item_label):
                continue
            for pair in pairs:
                if not isinstance(pair, list) or len(pair) < 2:
                    continue
                if (pair[0] in a1_ids and pair[1] in a2_ids) or (pair[0] in a2_ids and pair[1] in a1_ids):
                    result.append(item)
                    break
    return result


def geo_between(candidates: list[VLAObject], group_a: list[VLAObject], group_b: list[VLAObject]) -> Optional[VLAObject]:
    """Geometric fallback: return candidate closest to the midpoint of the nearest A-B pair."""
    if not candidates or not group_a or not group_b:
        return None
    best, best_score = None, float("inf")
    for c in candidates:
        for a in group_a:
            for b in group_b:
                mid_x = (a.x + b.x) / 2
                mid_y = (a.y + b.y) / 2
                mid_z = (a.z + b.z) / 2
                score = math.sqrt((c.x - mid_x) ** 2 + (c.y - mid_y) ** 2 + (c.z - mid_z) ** 2)
                if score < best_score:
                    best_score, best = score, c
    return best


def closest_to(candidates: list[VLAObject], targets: list[VLAObject]) -> Optional[VLAObject]:
    if not candidates or not targets:
        return None
    return min(candidates, key=lambda c: min(dist3(c, t) for t in targets))


def farthest_from(candidates: list[VLAObject], targets: list[VLAObject]) -> Optional[VLAObject]:
    if not candidates or not targets:
        return None
    return max(candidates, key=lambda c: min(dist3(c, t) for t in targets))


def geo_on_surface(
    sc: VLAScene,
    container_labels: list[str],
    item_labels: list[str],
    xy_thresh: float = 1.0,
    z_extra: float = 0.3,
) -> list[tuple[VLAObject, VLAObject]]:
    """Geometric fallback: items whose center is within xy_thresh of a container and z is
    above (container top - z_extra).  Returns (item, container) pairs."""
    containers: list[VLAObject] = []
    seen_c: set[int] = set()
    for lbl in container_labels:
        for c in objs(sc, lbl):
            if c.id not in seen_c:
                containers.append(c)
                seen_c.add(c.id)

    candidates: list[VLAObject] = []
    seen_i: set[int] = set()
    for lbl in item_labels:
        for o in objs(sc, lbl):
            if o.id not in seen_i:
                candidates.append(o)
                seen_i.add(o.id)

    pairs: list[tuple[VLAObject, VLAObject]] = []
    for c in containers:
        top_z = c.z + c.lz / 2
        for item in candidates:
            if item.id == c.id:
                continue
            if item.z >= top_z - z_extra and dist3_xy(item, c) < xy_thresh:
                pairs.append((item, c))
    return pairs


# ---------------------------------------------------------------------------
# Load challenge scene object_list from zip
# ---------------------------------------------------------------------------


def load_challenge_object_list(scene: str) -> list[str]:
    zip_path = CHALLENGE_DATA / f"{scene}.zip"
    with zipfile.ZipFile(zip_path) as zf:
        return zf.read(f"{scene}/object_list.txt").decode().splitlines()


# ---------------------------------------------------------------------------
# Per-question resolvers
# ---------------------------------------------------------------------------


def _ref(sc: VLAScene, obj: VLAObject) -> dict:
    return {"object_id": obj.id, "label": obj.raw_label}


def resolve_arabic_room(sc: VLAScene) -> list[dict]:
    entries = []

    # numerical: "How many sofas are below a window?"
    sofas = objs_exact(sc, "sofa")
    windows = objs_exact(sc, "window")
    count = 0
    for sofa in sofas:
        for win in windows:
            if win.z > sofa.z and dist3_xy(sofa, win) < 3.0:
                count += 1
                break
    entries.append(("numerical", "How many sofas are below a window?", count))

    # object_reference: "Find the pillow closest to the book on the stool."
    book_on_stool = [item for item, c in on_items(sc, "stool", "book")]
    pillows = objs_exact(sc, "pillow")
    target = closest_to(pillows, book_on_stool)
    entries.append(("object_reference", "Find the pillow closest to the book on the stool.", target))

    # object_reference: "Find the wall lamp that is between a door frame and a window."
    # Scene graph between[] only records floor/ceiling pairs — use geometry.
    wall_lamps = objs_exact(sc, "wall lamp")
    door_frames = objs(sc, "door frame")
    target = geo_between(wall_lamps, door_frames, windows)
    entries.append(("object_reference", "Find the wall lamp that is between a door frame and a window.", target))

    return entries


def resolve_chinese_room(sc: VLAScene) -> list[dict]:
    entries = []

    # numerical: "Count the number of chairs with pillows on them."
    chairs = objs_exact(sc, "chair")
    count = 0
    for chair in chairs:
        has_pillow = False
        for rel_dict in sc.relationships.values():
            for t_id in rel_dict.get("on", {}).get(chair.id, []):
                if isinstance(t_id, int):
                    obj = sc.by_id.get(t_id)
                    if obj and _label_match(obj.raw_label, "pillow"):
                        has_pillow = True
                        break
            if has_pillow:
                break
        if has_pillow:
            count += 1
    entries.append(("numerical", "Count the number of chairs with pillows on them.", count))

    # object_reference: "Find the bowl on the table closest to the folding screen."
    screens = objs(sc, "folding screen")
    tables = objs(sc, "table")
    closest_table = closest_to(tables, screens)
    bowls_on_closest = [item for item, c in on_items(sc, "table", "bowl") if c.id == closest_table.id] if closest_table else []
    if not bowls_on_closest:
        all_bowls_on_tables = [item for item, c in on_items(sc, "table", "bowl")]
        target = closest_to(all_bowls_on_tables, screens)
    else:
        target = bowls_on_closest[0]
    entries.append(("object_reference", "Find the bowl on the table closest to the folding screen.", target))

    # object_reference: "Find the pillow on the chair that is closest to the TV."
    tvs = objs(sc, "tv")
    chairs = objs_exact(sc, "chair")
    closest_chair = closest_to(chairs, tvs)
    pillows_on_closest = [item for item, c in on_items(sc, "chair", "pillow") if c.id == closest_chair.id] if closest_chair else []
    if not pillows_on_closest:
        all_pillows_on_chairs = [item for item, c in on_items(sc, "chair", "pillow")]
        target = closest_to(all_pillows_on_chairs, tvs)
    else:
        target = pillows_on_closest[0]
    entries.append(("object_reference", "Find the pillow on the chair that is closest to the TV.", target))

    return entries


def resolve_home_building_1(sc: VLAScene) -> list[dict]:
    entries = []

    # numerical: "How many pillows are on the sofa under the pictures?"
    pictures = objs(sc, "picture")
    sofas = objs_exact(sc, "sofa")
    sofa_with_pics = None
    best_count = 0
    for sofa in sofas:
        pics_above = [p for p in pictures if p.z > sofa.z and dist3_xy(sofa, p) < 3.0]
        if len(pics_above) > best_count:
            best_count = len(pics_above)
            sofa_with_pics = sofa
    count = 0
    if sofa_with_pics:
        for rel_dict in sc.relationships.values():
            for t_id in rel_dict.get("on", {}).get(sofa_with_pics.id, []):
                if isinstance(t_id, int):
                    obj = sc.by_id.get(t_id)
                    if obj and _label_match(obj.raw_label, "pillow"):
                        count += 1
    entries.append(("numerical", "How many pillows are on the sofa under the pictures?", count))

    # object_reference: "Find the clock on the TV cabinet."
    # Scene graph on[] for tv cabinet is sparse — try scene graph first, then geometry.
    clocks_on_tv = [item for item, c in on_items(sc, "tv cabinet", "clock")]
    if not clocks_on_tv:
        clocks_on_tv = [item for item, c in geo_on_surface(
            sc, ["tv cabinet", "tv stand"], ["clock"], xy_thresh=1.5, z_extra=0.3
        )]
    target = clocks_on_tv[0] if clocks_on_tv else None
    entries.append(("object_reference", "Find the clock on the TV cabinet.", target))

    # object_reference: "Find the bowl closest to the knife rack near the trash can."
    trash_cans = objs(sc, "trash can")
    knife_racks = objs(sc, "knife rack")
    closest_rack = closest_to(knife_racks, trash_cans)
    bowls = objs_exact(sc, "bowl")
    target = closest_to(bowls, [closest_rack]) if closest_rack else closest_to(bowls, knife_racks)
    entries.append(("object_reference", "Find the bowl closest to the knife rack near the trash can.", target))

    return entries


def resolve_home_building_2(sc: VLAScene) -> list[dict]:
    entries = []

    # numerical: "How many red pillows are on the sofa?"
    # VLA-3D uses "maroon" for visually red objects; "red" is mapped via COLOR_ALIASES.
    sofas = objs_exact(sc, "sofa")
    count = 0
    for sofa in sofas:
        for rel_dict in sc.relationships.values():
            for t_id in rel_dict.get("on", {}).get(sofa.id, []):
                if isinstance(t_id, int):
                    obj = sc.by_id.get(t_id)
                    if obj and _label_match(obj.raw_label, "pillow") and _color_matches(obj.colors, "red"):
                        count += 1
    if count == 0:
        # geometric fallback: pillows near sofas
        for sofa in sofas:
            for item, c in geo_on_surface(sc, ["sofa"], ["pillow"], xy_thresh=1.5, z_extra=0.3):
                if c.id == sofa.id and _color_matches(item.colors, "red"):
                    count += 1
    entries.append(("numerical", "How many red pillows are on the sofa?", count))

    # object_reference: "Find the lamp on the nightstand that has the photo on it."
    # Scene graph on[] for nightstands is empty — use geometry.
    # "Lamp" in this scene is labeled "desk light".
    nightstands = objs(sc, "nightstand")
    target = None
    # Find nightstand with a photo on it (geometrically)
    photos = objs(sc, "photo")
    nightstand_with_photo = None
    for ns in nightstands:
        top_z = ns.z + ns.lz / 2
        has_photo = any(p.z >= top_z - 0.3 and dist3_xy(p, ns) < 1.0 for p in photos)
        if has_photo:
            nightstand_with_photo = ns
            break
    if nightstand_with_photo:
        # Find lamp (or desk light) on that nightstand geometrically
        lamp_candidates = objs(sc, "lamp", "desk light", "table lamp")
        top_z = nightstand_with_photo.z + nightstand_with_photo.lz / 2
        lamps_on = [
            o for o in lamp_candidates
            if o.z >= top_z - 0.3 and dist3_xy(o, nightstand_with_photo) < 0.8
        ]
        target = lamps_on[0] if lamps_on else None
    if target is None:
        # Last resort: any lamp geometrically on any nightstand
        lamps_on_ns = [item for item, c in geo_on_surface(
            sc, ["nightstand"], ["lamp", "desk light", "table lamp"], xy_thresh=0.8, z_extra=0.3
        )]
        target = lamps_on_ns[0] if lamps_on_ns else None
    entries.append(("object_reference", "Find the lamp on the nightstand that has the photo on it.", target))

    # object_reference: "Find the speaker on the TV cabinet closest to the potted plant on the TV cabinet."
    # Search both "tv cabinet" and "tv stand" — the scene has both.
    tv_labels = ["tv cabinet", "tv stand"]
    # speakers on TV furniture (scene graph first, then geometry)
    speakers_on_tv = [item for item, c in on_items(sc, "tv cabinet", "speaker")]
    speakers_on_tv += [item for item, c in on_items(sc, "tv stand", "speaker")]
    if not speakers_on_tv:
        speakers_on_tv = [item for item, c in geo_on_surface(
            sc, tv_labels, ["speaker"], xy_thresh=1.5, z_extra=0.3
        )]
    # plants on TV furniture
    plants_on_tv = [item for item, c in on_items(sc, "tv cabinet", "potted plant")]
    plants_on_tv += [item for item, c in on_items(sc, "tv stand", "potted plant")]
    if not plants_on_tv:
        plants_on_tv = [item for item, c in geo_on_surface(
            sc, tv_labels, ["potted plant"], xy_thresh=1.8, z_extra=0.3
        )]
    # If still no plants on TV furniture, use all potted plants near the TV area
    if not plants_on_tv:
        tv_furniture = objs(sc, *tv_labels)
        all_plants = objs(sc, "potted plant")
        plants_on_tv = sorted(all_plants, key=lambda p: min(dist3_xy(p, t) for t in tv_furniture))[:3]
    target = closest_to(speakers_on_tv, plants_on_tv) if plants_on_tv else (speakers_on_tv[0] if speakers_on_tv else None)
    entries.append(("object_reference", "Find the speaker on the TV cabinet closest to the potted plant on the TV cabinet.", target))

    return entries


def resolve_hotel_room_1(sc: VLAScene) -> list[dict]:
    entries = []

    # numerical: "How many pillows are on the bed?"
    count = len([item for item, c in on_items(sc, "bed", "pillow")])
    entries.append(("numerical", "How many pillows are on the bed?", count))

    # object_reference: "Find the bedside table farthest from the window."
    bedside_tables = objs(sc, "bedside table")
    windows = objs(sc, "window")
    target = farthest_from(bedside_tables, windows)
    entries.append(("object_reference", "Find the bedside table farthest from the window.", target))

    # object_reference: "Find the picture above the suitcase furthest from the floor."
    suitcases = objs(sc, "suitcase")
    pics_above = [item for item, c in items_above(sc, "suitcase", "picture")]
    if not pics_above:
        all_pics = objs(sc, "picture")
        pics_above = [p for p in all_pics if any(p.z > s.z and dist3_xy(p, s) < 2.0 for s in suitcases)]
    target = max(pics_above, key=lambda p: p.z) if pics_above else None
    entries.append(("object_reference", "Find the picture above the suitcase furthest from the floor.", target))

    return entries


def resolve_hotel_room_2(sc: VLAScene) -> list[dict]:
    entries = []

    # numerical: "How many pictures are above the bed?"
    beds = objs(sc, "bed")
    pics_above = [item for item, c in items_above(sc, "bed", "picture")]
    if not pics_above:
        all_pics = objs(sc, "picture")
        pics_above = [p for p in all_pics if any(p.z > b.z and dist3_xy(p, b) < 2.5 for b in beds)]
    entries.append(("numerical", "How many pictures are above the bed?", len(pics_above)))

    # object_reference: "Find the flowers near the window."
    flowers_near = [item for item, c in near_items(sc, "window", "flower")]
    if not flowers_near:
        flowers = objs(sc, "flower")
        windows = objs(sc, "window")
        flowers_near = [closest_to(flowers, windows)] if flowers and windows else []
    target = flowers_near[0] if flowers_near else None
    entries.append(("object_reference", "Find the flowers near the window.", target))

    # object_reference: "Find the picture closest to the bench."
    pictures = objs(sc, "picture")
    benches = objs(sc, "bench")
    target = closest_to(pictures, benches)
    entries.append(("object_reference", "Find the picture closest to the bench.", target))

    return entries


def resolve_japanese_room(sc: VLAScene) -> list[dict]:
    entries = []

    # numerical: "How many calligraphy paintings are above the display ledge?"
    ledges = objs(sc, "display ledge")
    cals = objs(sc, "calligraphy", "painting")
    above_items_list = [item for item, c in items_above(sc, "display ledge", None)
                        if _label_match(item.raw_label, "calligraphy") or _label_match(item.raw_label, "painting")]
    if not above_items_list:
        above_items_list = [c for c in cals if any(c.z > l.z and dist3_xy(c, l) < 3.0 for l in ledges)]
    entries.append(("numerical", "How many calligraphy paintings are above the display ledge?", len(above_items_list)))

    # object_reference: "The lantern between the vase and the stone decoration that is closest to the vase."
    lanterns_between = between_items(sc, "lantern", "vase", "stone decoration")
    if not lanterns_between:
        lanterns = objs(sc, "lantern")
        lanterns_between_geo = geo_between(lanterns, objs(sc, "vase"), objs(sc, "stone decoration"))
        lanterns_between = [lanterns_between_geo] if lanterns_between_geo else lanterns
    vases = objs(sc, "vase")
    target = closest_to(lanterns_between, vases)
    entries.append(("object_reference", "The lantern between the vase and the stone decoration that is closest to the vase.", target))

    # object_reference: "The red pillow closest to the sushi."
    # VLA-3D labels red pillows as "maroon" — use COLOR_ALIASES.
    pillows = objs(sc, "pillow")
    red_pillows = [p for p in pillows if _color_matches(p.colors, "red")]
    if not red_pillows:
        red_pillows = pillows  # fallback: all pillows
    sushis = objs(sc, "sushi")
    target = closest_to(red_pillows, sushis)
    entries.append(("object_reference", "The red pillow closest to the sushi.", target))

    return entries


def resolve_livingroom_1(sc: VLAScene) -> list[dict]:
    entries = []

    # numerical: "How many chairs are near the table with a vase on it?"
    tables_with_vase = list({c.id: c for item, c in on_items(sc, "table", "vase")}.values())
    chairs = objs_exact(sc, "chair")
    chairs_near: set[int] = set()
    for t in tables_with_vase:
        for rel_dict in sc.relationships.values():
            for t_id in rel_dict.get("near", {}).get(t.id, []):
                if isinstance(t_id, int):
                    obj = sc.by_id.get(t_id)
                    if obj and _label_match(obj.raw_label, "chair"):
                        chairs_near.add(obj.id)
    if not chairs_near:
        for c in chairs:
            if any(dist3(c, t) < 2.0 for t in tables_with_vase):
                chairs_near.add(c.id)
    entries.append(("numerical", "How many chairs are near the table with a vase on it?", len(chairs_near)))

    # object_reference: "Find the vase on the cabinet below the picture."
    # Scene graph on[] for cabinets is empty — use geometry.
    pictures = objs(sc, "picture")
    cabinets = objs(sc, "cabinet")
    # cabinet that has a picture above it
    cabs_below_pic = [cab for cab in cabinets
                      if any(pic.z > cab.z + 0.3 and dist3_xy(cab, pic) < 2.0 for pic in pictures)]
    if not cabs_below_pic:
        cabs_below_pic = cabinets
    # vases on those cabinets — scene graph first
    vases_on_cab = [item for item, c in on_items(sc, "cabinet", "vase") if c in cabs_below_pic]
    if not vases_on_cab:
        # geometric: vase near/above a cabinet below a picture
        target_cabs = [c.raw_label for c in cabs_below_pic]
        pairs = geo_on_surface(sc, target_cabs, ["vase"], xy_thresh=1.0, z_extra=0.3)
        # filter to cabinets that are below a picture
        vases_on_cab = [item for item, c in pairs if c in cabs_below_pic]
        if not vases_on_cab:
            # broader: any vase near any cabinet below a picture
            vases_all = objs_exact(sc, "vase")
            vases_on_cab = [
                v for v in vases_all
                if any(v.z >= cab.z + cab.lz / 2 - 0.3 and dist3_xy(v, cab) < 1.0 for cab in cabs_below_pic)
            ]
    target = vases_on_cab[0] if vases_on_cab else None
    entries.append(("object_reference", "Find the vase on the cabinet below the picture.", target))

    # object_reference: "Find the pillow on the sofa that is closest to the windows."
    windows = objs(sc, "window")
    sofas = objs_exact(sc, "sofa")
    closest_sofa = closest_to(sofas, windows)
    pillows_on = [item for item, c in on_items(sc, "sofa", "pillow") if c.id == closest_sofa.id] if closest_sofa else []
    if not pillows_on:
        all_pillows = [item for item, c in on_items(sc, "sofa", "pillow")]
        target = closest_to(all_pillows, windows)
    else:
        target = closest_to(pillows_on, windows)
    entries.append(("object_reference", "Find the pillow on the sofa that is closest to the windows.", target))

    return entries


def resolve_livingroom_2(sc: VLAScene) -> list[dict]:
    entries = []

    # numerical: "How many cups are on the coffee table?"
    cups_on_coffee = on_items(sc, "coffee table", "cup")
    entries.append(("numerical", "How many cups are on the coffee table?", len(cups_on_coffee)))

    # object_reference: "Find the stool closest to the shelf near the TV cabinet."
    tv_cabs = objs(sc, "tv cabinet", "TV cabinet", "tv stand")
    shelves = objs(sc, "shelf")
    closest_shelf = closest_to(shelves, tv_cabs)
    stools = objs(sc, "stool")
    target = closest_to(stools, [closest_shelf]) if closest_shelf else closest_to(stools, shelves)
    entries.append(("object_reference", "Find the stool closest to the shelf near the TV cabinet.", target))

    # object_reference: "Find the pillow on the sofa that is closest to the lamp."
    lamps = objs(sc, "lamp")
    sofas = objs_exact(sc, "sofa")
    closest_sofa = closest_to(sofas, lamps)
    pillows_on = [item for item, c in on_items(sc, "sofa", "pillow") if c.id == closest_sofa.id] if closest_sofa else []
    if not pillows_on:
        all_pillows = [item for item, c in on_items(sc, "sofa", "pillow")]
        target = closest_to(all_pillows, lamps)
    else:
        target = closest_to(pillows_on, lamps)
    entries.append(("object_reference", "Find the pillow on the sofa that is closest to the lamp.", target))

    return entries


def resolve_livingroom_3(sc: VLAScene) -> list[dict]:
    entries = []

    # numerical: "How many photos are on the TV cabinet?"
    count = len(on_items(sc, "tv cabinet", "photo"))
    if count == 0:
        # geometric fallback
        pairs = geo_on_surface(sc, ["tv cabinet", "TV cabinet"], ["photo"], xy_thresh=1.0, z_extra=0.3)
        count = len(pairs)
    entries.append(("numerical", "How many photos are on the TV cabinet?", count))

    # object_reference: "Find the potted plant near the books on the cabinet."
    books_on_cab = [item for item, c in on_items(sc, "cabinet", "book")]
    plants = objs(sc, "potted plant")
    if books_on_cab:
        target = closest_to(plants, books_on_cab)
    else:
        cabs = objs(sc, "cabinet")
        target = closest_to(plants, cabs)
    entries.append(("object_reference", "Find the potted plant near the books on the cabinet.", target))

    # object_reference: "Find the vase between the cabinet and the stool."
    vases_between = between_items(sc, "vase", "cabinet", "stool")
    if not vases_between:
        vases = objs_exact(sc, "vase")
        target = geo_between(vases, objs(sc, "cabinet"), objs(sc, "stool"))
    else:
        target = vases_between[0]
    entries.append(("object_reference", "Find the vase between the cabinet and the stool.", target))

    return entries


def resolve_livingroom_4(sc: VLAScene) -> list[dict]:
    entries = []

    # numerical: "How many pillows are on a sofa?"
    count = len(on_items(sc, "sofa", "pillow"))
    entries.append(("numerical", "How many pillows are on a sofa?", count))

    # object_reference: "Find the picture closest to a window."
    pictures = objs(sc, "picture")
    windows = objs(sc, "window")
    target = closest_to(pictures, windows)
    entries.append(("object_reference", "Find the picture closest to a window.", target))

    # object_reference: "Find the fossil decoration closest to the phone."
    fossils = objs(sc, "fossil")
    phones = objs(sc, "phone")
    target = closest_to(fossils, phones)
    entries.append(("object_reference", "Find the fossil decoration closest to the phone.", target))

    return entries


def resolve_loft(sc: VLAScene) -> list[dict]:
    entries = []

    # numerical: "How many black pillows are on the sofa?"
    # VLA-3D colors: all sofa pillows are gray or olive — no black in this scene → answer 0.
    sofas = objs_exact(sc, "sofa")
    count = 0
    for sofa in sofas:
        for rel_dict in sc.relationships.values():
            for t_id in rel_dict.get("on", {}).get(sofa.id, []):
                if isinstance(t_id, int):
                    obj = sc.by_id.get(t_id)
                    if obj and _label_match(obj.raw_label, "pillow") and _color_matches(obj.colors, "black"):
                        count += 1
    entries.append(("numerical", "How many black pillows are on the sofa?", count))

    # object_reference: "The blue chair that is closest to the cup of coffee."
    chairs = objs_exact(sc, "chair")
    blue_chairs = [c for c in chairs if _color_matches(c.colors, "blue")]
    if not blue_chairs:
        blue_chairs = chairs
    cups = objs(sc, "coffee cup", "cup of coffee", "cup")
    target = closest_to(blue_chairs, cups)
    entries.append(("object_reference", "The blue chair that is closest to the cup of coffee.", target))

    # object_reference: "Find the potted plant between a vase and the cabinet with a TV on it."
    tv_cabs = [c for item, c in on_items(sc, "TV cabinet", "tv")]
    if not tv_cabs:
        tv_cabs = [c for item, c in on_items(sc, "cabinet", "tv")]
    if not tv_cabs:
        tv_cabs = objs(sc, "TV cabinet", "tv cabinet")
    plants_between = between_items(sc, "potted plant", "vase", "TV cabinet")
    if not plants_between:
        plants = objs(sc, "potted plant")
        vases = objs(sc, "vase")
        target = geo_between(plants, vases, tv_cabs)
    else:
        target = plants_between[0]
    entries.append(("object_reference", "Find the potted plant between a vase and the cabinet with a TV on it.", target))

    return entries


def resolve_office_1(sc: VLAScene) -> list[dict]:
    entries = []

    # numerical: "How many computer monitors are on the table closest to the map wall decal?"
    # Use exact label for "map wall decal" to avoid matching wall objects.
    wall_decals = objs_exact(sc, "map wall decal")
    tables = objs_exact(sc, "table")
    closest_table = closest_to(tables, wall_decals)
    monitors_on = [item for item, c in on_items(sc, "table", "computer monitor")
                   if closest_table and c.id == closest_table.id]
    entries.append(("numerical", "How many computer monitors are on the table closest to the map wall decal?", len(monitors_on)))

    # object_reference: "Find the potted plant on the file cabinet."
    # Scene graph on[] for file cabinet is empty — use geometry.
    plants_on_fc = [item for item, c in on_items(sc, "file cabinet", "potted plant")]
    if not plants_on_fc:
        plants_on_fc = [item for item, c in geo_on_surface(
            sc, ["file cabinet"], ["potted plant"], xy_thresh=1.2, z_extra=0.3
        )]
    target = plants_on_fc[0] if plants_on_fc else None
    entries.append(("object_reference", "Find the potted plant on the file cabinet.", target))

    # object_reference: "Find the paper cup on the table closest to the projector screen."
    projectors = objs(sc, "projector screen")
    closest_table = closest_to(tables, projectors)
    cups_on = [item for item, c in on_items(sc, "table", "paper cup")
               if closest_table and c.id == closest_table.id]
    if not cups_on:
        all_cups = [item for item, c in on_items(sc, "table", "paper cup")]
        target = closest_to(all_cups, projectors)
    else:
        target = cups_on[0]
    entries.append(("object_reference", "Find the paper cup on the table closest to the projector screen.", target))

    return entries


def resolve_office_2(sc: VLAScene) -> list[dict]:
    entries = []

    # numerical: "How many potted plants are on a table?"
    count = len(on_items(sc, "table", "potted plant"))
    entries.append(("numerical", "How many potted plants are on a table?", count))

    # object_reference: "Find the computer monitor closest to the cabinet with a phone on it."
    # Scene graph on[] for cabinets is empty — use geometry.
    cabinets = objs(sc, "cabinet")
    phones = objs_exact(sc, "phone")
    # Find cabinet(s) with a phone on them geometrically
    cabs_with_phone: list[VLAObject] = []
    for cab in cabinets:
        top_z = cab.z + cab.lz / 2
        if any(ph.z >= top_z - 0.3 and dist3_xy(ph, cab) < 0.8 for ph in phones):
            cabs_with_phone.append(cab)
    # Try scene graph as well
    cabs_with_phone_sg = [c for item, c in on_items(sc, "cabinet", "phone")]
    for c in cabs_with_phone_sg:
        if c.id not in {x.id for x in cabs_with_phone}:
            cabs_with_phone.append(c)
    monitors = objs(sc, "computer monitor")
    target = closest_to(monitors, cabs_with_phone) if cabs_with_phone else None
    entries.append(("object_reference", "Find the computer monitor closest to the cabinet with a phone on it.", target))

    # object_reference: "Find the box on the cabinet that is closest to the whiteboard."
    whiteboards = objs(sc, "whiteboard")
    # Find boxes on cabinets geometrically
    boxes = objs_exact(sc, "box")
    boxes_on_cabs: list[tuple[VLAObject, VLAObject]] = []
    for cab in cabinets:
        top_z = cab.z + cab.lz / 2
        for box in boxes:
            if box.z >= top_z - 0.3 and dist3_xy(box, cab) < 1.0:
                boxes_on_cabs.append((box, cab))
    # Also check scene graph
    for item, c in on_items(sc, "cabinet", "box"):
        if not any(item.id == b.id for b, _ in boxes_on_cabs):
            boxes_on_cabs.append((item, c))
    box_objs = [b for b, _ in boxes_on_cabs]
    target = closest_to(box_objs, whiteboards)
    entries.append(("object_reference", "Find the box on the cabinet that is closest to the whiteboard.", target))

    return entries


def resolve_studio(sc: VLAScene) -> list[dict]:
    entries = []

    # numerical: "How many framed records are above the couch?"
    couches = objs(sc, "couch", "sofa")
    records = objs(sc, "framed record", "record")
    above_couch = [item for item, c in items_above(sc, "couch", None)
                   if _label_match(item.raw_label, "record") or _label_match(item.raw_label, "framed")]
    if not above_couch:
        above_couch = [r for r in records if any(r.z > c.z and dist3_xy(r, c) < 3.0 for c in couches)]
    entries.append(("numerical", "How many framed records are above the couch?", len(above_couch)))

    # object_reference: "Find the vase closest to the guitar."
    vases = objs_exact(sc, "vase")
    guitars = objs(sc, "guitar")
    target = closest_to(vases, guitars)
    entries.append(("object_reference", "Find the vase closest to the guitar.", target))

    # object_reference: "Find the beer bottle furthest from the couch."
    bottles = objs(sc, "beer bottle", "bottle")
    target = farthest_from(bottles, couches)
    entries.append(("object_reference", "Find the beer bottle furthest from the couch.", target))

    return entries


RESOLVERS = {
    "arabic_room": resolve_arabic_room,
    "chinese_room": resolve_chinese_room,
    "home_building_1": resolve_home_building_1,
    "home_building_2": resolve_home_building_2,
    "hotel_room_1": resolve_hotel_room_1,
    "hotel_room_2": resolve_hotel_room_2,
    "japanese_room": resolve_japanese_room,
    "livingroom_1": resolve_livingroom_1,
    "livingroom_2": resolve_livingroom_2,
    "livingroom_3": resolve_livingroom_3,
    "livingroom_4": resolve_livingroom_4,
    "loft": resolve_loft,
    "office_1": resolve_office_1,
    "office_2": resolve_office_2,
    "studio": resolve_studio,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate challenge_gt.jsonl from questions.json + VLA-3D data.")
    parser.add_argument("--verify", action="store_true", help="Print resolved answers for manual review instead of writing JSONL.")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_JSON.read_text())

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    written = errors = 0

    with OUTPUT.open("w") as out_f:
        for scene_entry in questions:
            scene = scene_entry["scene"]
            scene_qs = scene_entry["questions"]

            if scene not in RESOLVERS:
                print(f"[warn] no resolver for scene {scene!r}", file=sys.stderr)
                errors += 1
                continue

            vla_dir = VLA3D_ROOT / scene
            if not vla_dir.exists():
                print(f"[warn] VLA-3D scene not found: {vla_dir}", file=sys.stderr)
                errors += 1
                continue

            sc = load_vla_scene(vla_dir)

            try:
                object_list = load_challenge_object_list(scene)
            except Exception as exc:
                print(f"[warn] could not load object_list for {scene}: {exc}", file=sys.stderr)
                errors += 1
                continue

            entries = RESOLVERS[scene](sc)

            resolved: dict[str, tuple] = {}
            for qtype, question, answer in entries:
                resolved[question] = (qtype, answer)

            for qtype_key, qlist in scene_qs.items():
                if qtype_key == "instruction_following":
                    continue
                for question in qlist:
                    if question not in resolved:
                        print(f"[warn] unresolved: {scene} | {question!r}", file=sys.stderr)
                        errors += 1
                        continue

                    qtype, answer = resolved[question]

                    if answer is None:
                        print(f"[warn] None answer: {scene} | {question!r}", file=sys.stderr)
                        errors += 1
                        continue

                    if args.verify:
                        if isinstance(answer, VLAObject):
                            print(f"  {scene:<20} {qtype:<20} {question[:55]:<55} → id={answer.id} label={answer.raw_label!r}")
                        else:
                            print(f"  {scene:<20} {qtype:<20} {question[:55]:<55} → {answer}")
                        written += 1
                        continue

                    if isinstance(answer, VLAObject):
                        answer_field = {"object_id": answer.id, "label": answer.raw_label}
                    else:
                        answer_field = int(answer)

                    record = {
                        "scene": scene,
                        "type": qtype,
                        "question": question,
                        "answer": answer_field,
                        "object_list": object_list,
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1

    if args.verify:
        print(f"\nResolved {written} questions  (errors/warnings: {errors})")
    else:
        print(f"Written {written} GT entries to {OUTPUT}  (errors/warnings: {errors})")


if __name__ == "__main__":
    main()
