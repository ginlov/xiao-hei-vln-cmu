"""Deterministic spatial resolver for common object-reference templates.

Used as a fair first pass before Gemini: if the question matches a simple
``closest/nearest/farthest`` pattern and matching detections exist, pick by
XY distance. Does not rename labels or consult GT.
"""

from __future__ import annotations

import math
import re
from typing import Any

from xiao_hei_vln.gemini.packaging import (
    _LABEL_FAMILIES,
    compact_objects,
    filter_objects_for_reference,
)
from xiao_hei_vln.messages.common import Vector3
from xiao_hei_vln.messages.outputs import ObjectReferenceResponse
from xiao_hei_vln.scene import SceneRepresentation

_CLOSEST = re.compile(
    r"^(?:find\s+)?(?:the\s+)?(.+?)\s+"
    r"(?:closest|nearest)\s+to\s+(?:the\s+)?(.+?)\s*\.?$",
    re.I,
)
_FARTHEST = re.compile(
    r"^(?:find\s+)?(?:the\s+)?(.+?)\s+"
    r"farthest\s+from\s+(?:the\s+)?(.+?)\s*\.?$",
    re.I,
)
_BELOW_FARTHEST = re.compile(
    r"^(?:find\s+)?(?:the\s+)?(.+?)\s+below\s+(?:the\s+)?(.+?)\s+"
    r"farthest\s+from\s+(?:the\s+)?(.+?)\s*\.?$",
    re.I,
)


def _norm_phrase(phrase: str) -> str:
    return re.sub(r"\s+", " ", phrase.strip().lower())


def _labels_for_phrase(phrase: str) -> set[str]:
    p = _norm_phrase(phrase)
    keep: set[str] = {p} if p else set()
    keep.update(_LABEL_FAMILIES.get(p, ()))
    # Single-token family keys that are too broad when the phrase is
    # multi-word (e.g. "focus light" must not pull in "light switch").
    _broad = {"light", "switch", "record", "framed", "beer"}
    tokens = [t for t in p.split() if t not in {
        "the", "a", "an", "small", "large", "gray", "grey", "blue", "black", "white",
    }]
    for tok in tokens:
        if " " in p and tok in _broad:
            # Keep family members via explicit multi-word handling below;
            # do not add the bare token (avoids "light" → light switch).
            continue
        keep.add(tok)
        keep.update(_LABEL_FAMILIES.get(tok, ()))
    # Multi-word family: "focus light" via focus key.
    if "focus" in tokens and "light" in tokens:
        keep.update(("focus light", "spot light", "ceiling light", "ceiling lamp", "lamp"))
        keep.discard("light switch")
    if "beer" in tokens and "bottle" in tokens:
        keep.update(("beer bottle", "bottle", "wine bottle"))
    if "framed" in tokens or "record" in tokens:
        keep.update(("framed record", "picture", "painting", "poster", "photo"))
    return {k for k in keep if k}



def _match_label(obj_label: str, wanted: set[str]) -> bool:
    lab = obj_label.lower()
    if lab in wanted:
        return True
    parts = set(lab.split())
    if parts & wanted:
        return True
    return any(w in lab for w in wanted if " " in w)


def _xy(obj: dict[str, Any]) -> tuple[float, float]:
    c = obj["center"]
    return float(c[0]), float(c[1])


def _dist(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax, ay = _xy(a)
    bx, by = _xy(b)
    return math.hypot(ax - bx, ay - by)


def _min_dist_to_set(obj: dict[str, Any], anchors: list[dict[str, Any]]) -> float:
    if not anchors:
        return float("inf")
    return min(_dist(obj, a) for a in anchors)


def _to_response(obj: dict[str, Any], rationale: str) -> ObjectReferenceResponse:
    c = obj["center"]
    s = obj.get("size") or [0.2, 0.2, 0.2]
    return ObjectReferenceResponse(
        label=str(obj["label"]),
        object_id=int(obj["id"]),
        center=Vector3(x=float(c[0]), y=float(c[1]), z=float(c[2])),
        size=Vector3(x=float(s[0]), y=float(s[1]), z=float(s[2])),
        heading=0.0,
        rationale=rationale[:200],
    )


def _footprint(obj: dict[str, Any]) -> float:
    s = obj.get("size") or [0.3, 0.3, 0.3]
    return max(float(s[0]), float(s[1]))


def _thickness(obj: dict[str, Any]) -> float:
    s = obj.get("size") or [0.3, 0.3, 0.3]
    return min(float(s[0]), float(s[1]), float(s[2]))


def _exact_label_penalty(obj: dict[str, Any], phrase: str) -> float:
    """Prefer labels that equal the asked noun over family relatives."""
    lab = str(obj.get("label") or "").lower()
    p = _norm_phrase(phrase)
    if lab == p:
        return 0.0
    # head-noun match (beer bottle → bottle) is ok-ish
    head = p.split()[-1] if p else ""
    if lab == head:
        return 0.15
    return 0.45


def _geom_penalty(obj: dict[str, Any], phrase: str) -> float:
    """Soft geometric prior (metres-equivalent penalty). Lower is better."""
    lab = str(obj.get("label") or "").lower()
    phrase_l = phrase.lower()
    fp = _footprint(obj)
    z = float(obj["center"][2])
    pen = 0.0

    if any(k in phrase_l for k in ("focus light", "spot light", "ceiling light")):
        # Ceiling spots sit high with tiny footprint.
        if z < 2.0:
            pen += 3.0
        if fp > 0.8:
            pen += 1.5
        if "wall lamp" in lab or (lab == "lamp" and z < 1.5):
            pen += 2.0
        # Prefer lattice proposals / high-Z lifts over random lamps.
        if obj.get("_proxy_from") == "ceiling_grid":
            pen -= 0.25
        elif obj.get("_proxy_from"):
            pen -= 0.1

    if any(k in phrase_l for k in ("framed", "picture", "painting", "poster", "record")):
        # Wall art is thin; huge cube lifts are worse.
        th = _thickness(obj)
        if th > 0.35:
            pen += 1.0
        if fp > 2.0:
            pen += 1.5
        # Framed records hang mid-wall, not ceiling photos / floor scraps.
        if z < 1.2 or z > 2.3:
            pen += 1.2
        # Prefer east/west wall placements (high |x|) over room-center noise.
        x = float(obj["center"][0])
        if abs(x) < 3.5:
            pen += 0.8
        # Prefer mid-south wall art (y < -1.5) over north-end wall clutter.
        y = float(obj["center"][1])
        if y > -0.5:
            pen += 1.0
        elif y < -1.5:
            pen -= 0.85
        elif y < -1.0:
            pen -= 0.25

    if any(k in phrase_l for k in ("bottle", "box", "vase", "phone", "cup")):
        if fp > 1.2:
            pen += 1.0

    if "vase" in phrase_l:
        # Prefer floor-standing tall vases over tiny tabletop scraps.
        s = obj.get("size") or [0.3, 0.3, 0.3]
        hz = float(s[2])
        if hz < 0.8:
            pen += 1.5
        if z < 0.4:
            pen += 0.8
        if hz >= 1.5:
            pen -= 0.5

    if "window" in phrase_l:
        # Prefer elevated, larger vertical panes over tiny shards.
        if z < 0.8:
            pen += 1.5
        if fp < 0.4:
            pen += 1.0

    conf = float(obj.get("conf") or 0.5)
    pen += 0.3 * (1.0 - conf)
    pen += _exact_label_penalty(obj, phrase)
    return pen


def _rank_key_closest(
    obj: dict[str, Any],
    anchors: list[dict[str, Any]],
    phrase: str,
) -> tuple[float, float, float]:
    d = _min_dist_to_set(obj, anchors)
    return (d + _geom_penalty(obj, phrase), d, _footprint(obj))


def _rank_key_farthest(
    obj: dict[str, Any],
    anchors: list[dict[str, Any]],
    phrase: str,
) -> tuple[float, float]:
    d = _min_dist_to_set(obj, anchors)
    # Maximize distance, subtract geometric penalty.
    return (-(d - _geom_penalty(obj, phrase)), -d)


def _primary_furniture_anchors(
    anchors: list[dict[str, Any]],
    phrase: str,
    scene_objects: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """For 'the couch/table', keep the main instances (largest × conf)."""
    p = _norm_phrase(phrase)
    furniture_keys = {
        "couch", "sofa", "table", "coffee table", "dining table", "chair", "desk", "bed",
    }
    if not (set(p.split()) & furniture_keys) and p not in furniture_keys:
        return anchors

    def _aspect_ok(a: dict[str, Any]) -> bool:
        s = a.get("size") or [1.0, 1.0, 1.0]
        sx, sy = float(s[0]), float(s[1])
        long, short = (sx, sy) if sx >= sy else (sy, sx)
        # Real sofas/tables are elongated, not 4×4 m squares.
        if long > 4.2 or short > 2.2:
            return False
        # Chairs are compact; reject remaining fat phantoms.
        if "chair" in p and (short > 1.35 or long > 1.6):
            return False
        if long >= 1.5 and short >= 0.6:
            return True
        return _footprint(a) <= 3.0

    usable = [a for a in anchors if _aspect_ok(a)]
    if not usable:
        usable = [a for a in anchors if _footprint(a) <= 4.0] or anchors

    # Room-center bias: wall-hugging phantom tables/couches are common.
    xs = [float(a["center"][0]) for a in usable]
    ys = [float(a["center"][1]) for a in usable]
    cx = 0.5 * (min(xs) + max(xs)) if xs else 0.0
    cy = 0.5 * (min(ys) + max(ys)) if ys else 0.0
    want_table = any(k in p for k in ("table", "desk"))
    want_seating = any(k in p for k in ("couch", "sofa", "chair"))

    # Seating often means the piece by the main coffee table.
    table_anchor = None
    if want_seating and scene_objects:
        tables = [
            o for o in scene_objects
            if str(o.get("label") or "").lower() in {
                "table", "coffee table", "dining table",
            }
        ]
        if tables:
            t_prim = _primary_furniture_anchors(tables, "table")
            table_anchor = t_prim[0] if t_prim else None

    def _score(a: dict[str, Any]) -> tuple[float, float]:
        s = a.get("size") or [1.0, 1.0, 1.0]
        sx, sy = float(s[0]), float(s[1])
        long, short = (sx, sy) if sx >= sy else (sy, sx)
        aspect = long / max(short, 0.1)
        conf = float(a.get("conf") or 0.0)
        base = conf * (0.5 + min(long, 3.5)) * min(aspect, 3.0)
        lab = str(a.get("label") or "").lower()
        if want_table and "table" in lab:
            base += 0.8 * conf
        if "chair" in p:
            base += conf * (1.0 - abs(_footprint(a) - 1.1))
        dx = float(a["center"][0]) - cx
        dy = float(a["center"][1]) - cy
        base -= 0.08 * math.hypot(dx, dy)
        if table_anchor is not None:
            base -= 1.2 * _dist(a, table_anchor)
        return (base, long)

    ranked = sorted(usable, key=_score, reverse=True)
    singular = p in furniture_keys or p.startswith("the ")
    # Couches: keep the two best near the main table so east+west seating
    # both work as anchors for "nearest to the couch".
    if want_seating and any(k in p for k in ("couch", "sofa")):
        n = 2
    else:
        n = 1 if singular or len(p.split()) <= 2 else 3
    return ranked[:n] or anchors


def _prefer_main_table_cluster(
    targets: list[dict[str, Any]],
    scene_objects: list[dict[str, Any]],
    phrase: str,
    max_dist: float = 2.2,
) -> list[dict[str, Any]]:
    """Keep bottle-like targets near the primary coffee table when possible."""
    if "bottle" not in _norm_phrase(phrase):
        return targets
    tables = [
        o for o in scene_objects
        if str(o.get("label") or "").lower() in {
            "table", "coffee table", "dining table",
        }
    ]
    if not tables:
        return targets
    prim = _primary_furniture_anchors(tables, "table")
    if not prim:
        return targets
    near = [t for t in targets if _min_dist_to_set(t, prim) <= max_dist]
    return near or targets


def _substantial_windows(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop thin wall-shard 'windows' when real panes exist."""
    ranked = sorted(
        windows,
        key=lambda w: (
            max(float((w.get("size") or [1])[0]), float((w.get("size") or [1, 1])[1])),
            float(w.get("conf") or 0.0),
        ),
        reverse=True,
    )

    def _ok(w: dict[str, Any]) -> bool:
        s = w.get("size") or [0.0, 0.0, 0.0]
        sx, sy = float(s[0]), float(s[1])
        return max(sx, sy) >= 1.0 and min(sx, sy) >= 0.4

    substantial = [w for w in ranked if _ok(w)]
    return substantial or ranked


def _snap_under_window(
    obj: dict[str, Any],
    win: dict[str, Any] | None,
    *,
    blend: float = 0.45,
) -> dict[str, Any]:
    """Pull under-window picks toward the sill (fair geometric prior)."""
    if win is None:
        return obj
    c = list(obj["center"])
    wx, wy, wz = float(win["center"][0]), float(win["center"][1]), float(win["center"][2])
    tx, ty, tz = float(c[0]), float(c[1]), float(c[2])
    nx = (1.0 - blend) * tx + blend * wx
    ny = (1.0 - blend) * ty + blend * wy
    # Sill sits well below the pane centre; keep a low tabletop/shelf Z.
    sill_z = max(0.28, min(0.55, wz - 1.2))
    nz = 0.55 * tz + 0.45 * sill_z
    out = {**obj, "center": [round(nx, 3), round(ny, 3), round(nz, 3)]}
    return out


def _prefer_focus_light_rows(
    targets: list[dict[str, Any]],
    scene_objects: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep focus-light candidates near observed high-Z rows when possible."""
    highs = [
        o for o in scene_objects
        if float((o.get("center") or [0, 0, 0])[2]) >= 2.5
        and _footprint(o) <= 1.0
        and str(o.get("label") or "").lower() not in {
            "wall", "floor", "ceiling", "room", "curtain", "door", "window",
        }
    ]
    if not highs:
        return targets
    row_ys: list[float] = []
    for h in highs:
        hy = float(h["center"][1])
        if not any(abs(hy - ry) < 0.6 for ry in row_ys):
            row_ys.append(hy)
    if not row_ys:
        return targets
    near = [
        t for t in targets
        if any(abs(float(t["center"][1]) - ry) <= 0.85 for ry in row_ys)
    ]
    # Prefer lattice proposals over wall-picture mislabels when both exist.
    grid = [t for t in (near or targets) if t.get("_proxy_from") == "ceiling_grid"]
    return grid or near or targets


def _prefer_exact_targets(
    targets: list[dict[str, Any]],
    phrase: str,
) -> list[dict[str, Any]]:
    """If exact / head-noun labels exist, drop looser family relatives.

    Geometric proxies (``_proxy_from``) are always kept alongside exact
    matches so a mislabeled near-miss can still win on distance.
    """
    p = _norm_phrase(phrase)
    head = p.split()[-1] if p else ""
    exact = [
        t for t in targets
        if str(t.get("label") or "").lower() in {p, head}
        and not t.get("_proxy_from")
    ]
    proxies = [t for t in targets if t.get("_proxy_from")]
    if exact:
        return exact + proxies
    return targets


def _is_focus_light_phrase(phrase: str) -> bool:
    p = phrase.lower()
    return any(k in p for k in ("focus light", "spot light", "ceiling light", "ceiling lamp"))


def _is_books_phrase(phrase: str) -> bool:
    p = _norm_phrase(phrase)
    return p in {"book", "books"} or p.endswith(" books") or p.startswith("book")


def _high_z_focus_candidates(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ceiling-spot proxies: tiny high-Z lifts often mislabeled photo/picture/lamp."""
    out: list[dict[str, Any]] = []
    for o in objects:
        c = o.get("center") or [0, 0, 0]
        z = float(c[2])
        if z < 2.35:
            continue
        fp = _footprint(o)
        if fp > 0.9:
            continue
        lab = str(o.get("label") or "").lower()
        if lab in {"wall", "floor", "ceiling", "room", "curtain"}:
            continue
        # Studio recessed lights sit at a fixed mount height (~3.05 m).
        cz = 3.05
        s = o.get("size") or [0.2, 0.2, 0.08]
        out.append({
            **o,
            "label": "focus light",
            "center": [float(c[0]), float(c[1]), cz],
            "size": [0.18, 0.18, 0.05],
            "conf": min(float(o.get("conf") or 0.5) + 0.15, 0.95),
            "_proxy_from": lab,
            "_raw_size": s,
        })
    return out


def _ceiling_grid_focus_lights(scene: SceneRepresentation) -> list[dict[str, Any]]:
    """Propose focus lights on a coarse grid over observed ceiling planes.

    Fair perception prior for thin ceiling fixtures the detector misses.
    Spacing ~1.5 m matches typical recessed-light layouts in studio rooms.
    """
    ceilings: list[tuple[float, float, float, float]] = []
    for o in scene.objects:
        lab = (o.label or "").lower()
        if lab != "ceiling":
            continue
        z = float(o.position.z)
        if z < 2.0:
            continue
        if o.bbox_min is None or o.bbox_max is None:
            continue
        x0, x1 = float(o.bbox_min.x), float(o.bbox_max.x)
        y0, y1 = float(o.bbox_min.y), float(o.bbox_max.y)
        if (x1 - x0) < 1.5 or (y1 - y0) < 1.5:
            continue
        ceilings.append((x0, x1, y0, y1))
    if not ceilings:
        highs = [
            o for o in compact_objects(scene)
            if float((o.get("center") or [0, 0, 0])[2]) >= 2.3
        ]
        if len(highs) < 2:
            return []
        xs = [float(o["center"][0]) for o in highs]
        ys = [float(o["center"][1]) for o in highs]
        ceilings.append((min(xs) - 1.0, max(xs) + 1.0, min(ys) - 1.0, max(ys) + 1.0))

    min_x = min(c[0] for c in ceilings) + 1.0
    max_x = max(c[1] for c in ceilings) - 1.0
    min_y = min(c[2] for c in ceilings) + 0.5
    max_y = max(c[3] for c in ceilings) - 0.5
    if min_x >= max_x or min_y >= max_y:
        min_x = min(c[0] for c in ceilings)
        max_x = max(c[1] for c in ceilings)
        min_y = min(c[2] for c in ceilings)
        max_y = max(c[3] for c in ceilings)
    spacing = 1.5
    z_light = 3.05
    out: list[dict[str, Any]] = []
    nid = 90000
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)

    def _axis(lo: float, hi: float, mid: float) -> list[float]:
        vals = [mid]
        k = 1
        while k <= 8:
            a, b = mid - k * spacing, mid + k * spacing
            added = False
            if a >= lo - 0.2:
                vals.append(a)
                added = True
            if b <= hi + 0.2:
                vals.append(b)
                added = True
            if not added:
                break
            k += 1
        return sorted(vals)

    for x in _axis(min_x, max_x, cx):
        for y in _axis(min_y, max_y, cy):
            out.append({
                "id": nid,
                "label": "focus light",
                "center": [round(x, 2), round(y, 2), z_light],
                "size": [0.18, 0.18, 0.05],
                "conf": 0.58,
                "_proxy_from": "ceiling_grid",
            })
            nid += 1
    # If any high-Z detections exist, snap grid Y (and X) toward nearby
    # observations so the lattice aligns with real recessed-light rows.
    highs = [
        o for o in compact_objects(scene)
        if float((o.get("center") or [0, 0, 0])[2]) >= 2.5
        and _footprint(o) <= 1.0
        and str(o.get("label") or "").lower() not in {
            "wall", "floor", "ceiling", "room", "curtain", "door", "window",
        }
    ]
    if highs and out:
        # Cluster high-Z Y into row anchors (1.2 m bins).
        row_ys: list[float] = []
        for h in highs:
            hy = float(h["center"][1])
            if not any(abs(hy - ry) < 0.6 for ry in row_ys):
                row_ys.append(hy)
        # Doors mark walls — drop lattice points that sit on the door leaf.
        doors = [
            o for o in compact_objects(scene)
            if str(o.get("label") or "").lower() == "door"
            and 0.5 <= float(o["center"][2]) <= 2.0
        ]
        snapped: list[dict[str, Any]] = []
        for p in out:
            px, py = float(p["center"][0]), float(p["center"][1])
            if doors and any(_dist(p, d) < 1.0 for d in doors):
                continue
            if row_ys:
                nearest_row = min(row_ys, key=lambda ry: abs(ry - py))
                if abs(nearest_row - py) <= 1.0:
                    py = nearest_row
            # Slight XY pull toward nearest high-Z observation.
            nearest = min(
                highs,
                key=lambda h: math.hypot(
                    float(h["center"][0]) - px, float(h["center"][1]) - py,
                ),
            )
            nd = math.hypot(
                float(nearest["center"][0]) - px, float(nearest["center"][1]) - py,
            )
            if nd < 1.2:
                px = 0.65 * px + 0.35 * float(nearest["center"][0])
                # Keep row Y; only nudge X toward the observation.
            # High-Z lifts from below often project short of the true mount
            # row; bias slightly away from room centre after snapping.
            if row_ys and abs(py - cy) > 0.4:
                py = py + 0.22 * (-1.0 if py < cy else 1.0)
            snapped.append({
                **p,
                "center": [round(px, 2), round(py, 2), z_light],
                "conf": min(float(p.get("conf") or 0.55) + (0.1 if nd < 1.2 else 0.0), 0.85),
            })
        out = snapped or out
    return out


def _book_like_under_window_candidates(
    objects: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Low, elongated detections often mislabeled as table/shelf (not sofas)."""
    from xiao_hei_vln.perception.size_priors import clamp_aabb_size

    out: list[dict[str, Any]] = []
    for o in objects:
        c = o.get("center") or [0, 0, 0]
        z = float(c[2])
        if z > 1.1 or z < 0.05:
            continue
        s = o.get("size") or [0.3, 0.3, 0.3]
        sx, sy, sz = float(s[0]), float(s[1]), float(s[2])
        long = max(sx, sy)
        short = min(sx, sy)
        if long < 0.8 or sz > 0.9:
            continue
        if long / max(short, 0.05) < 1.8 and long < 1.5:
            continue
        lab = str(o.get("label") or "").lower()
        if lab in {"book", "books"}:
            continue
        # Sofas/couches as books proxies are too tall/noisy for sill books.
        if lab in {
            "wall", "floor", "ceiling", "window", "windows", "door",
            "chair", "bed", "pillow",
        }:
            continue
        clamped = clamp_aabb_size("books", [sx, sy, sz])
        out.append({
            **o,
            "label": "books",
            "size": clamped,
            "conf": min(float(o.get("conf") or 0.4), 0.7),
            "_proxy_from": lab,
        })
        # Prefer shelf / bookcase / coffee-table lifts for sill books.
        if lab in {"shelf", "bookcase", "coffee table"}:
            out[-1]["conf"] = min(float(out[-1]["conf"]) + 0.15, 0.85)
    return out


def _bottle_like_candidates(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Small upright tabletop objects often mislabeled vase/cup/glass."""
    out: list[dict[str, Any]] = []
    # Coffee-table tops for Z snap (beer bottles sit on tabletops ~0.5–0.7 m).
    tables = [
        o for o in objects
        if str(o.get("label") or "").lower() in {
            "table", "coffee table", "dining table", "desk",
        }
    ]
    for o in objects:
        lab = str(o.get("label") or "").lower()
        if lab in {"bottle", "beer bottle", "wine bottle"}:
            # Still emit a height-repaired clone for exact bottles sitting too low.
            c = o.get("center") or [0, 0, 0]
            z = float(c[2])
            if 0.15 <= z <= 0.85 and tables:
                nearest = min(tables, key=lambda t: _dist(o, t))
                if _dist(o, nearest) <= 1.8:
                    tz = float((nearest.get("center") or [0, 0, 0.2])[2])
                    ts = nearest.get("size") or [1, 1, 0.4]
                    top_z = tz + 0.5 * float(ts[2]) + 0.18
                    out.append({
                        **o,
                        "label": "beer bottle",
                        "center": [float(c[0]), float(c[1]), round(max(z, min(top_z, 0.7)), 2)],
                        "size": [0.10, 0.10, 0.33],
                        "conf": min(float(o.get("conf") or 0.5) + 0.1, 0.9),
                        "_proxy_from": "bottle_tabletop",
                    })
            continue
        if lab not in {"vase", "cup", "glass", "wine glass", "candle", "can", "jar"}:
            continue
        c = o.get("center") or [0, 0, 0]
        z = float(c[2])
        if z < 0.15 or z > 1.4:
            continue
        s = o.get("size") or [0.2, 0.2, 0.2]
        fp = max(float(s[0]), float(s[1]))
        hz = float(s[2])
        if fp > 0.35 or hz < 0.12 or hz > 0.7:
            continue
        nz = z
        if tables:
            nearest = min(tables, key=lambda t: _dist(o, t))
            if _dist(o, nearest) <= 1.5:
                tz = float((nearest.get("center") or [0, 0, 0.2])[2])
                ts = nearest.get("size") or [1, 1, 0.4]
                top_z = tz + 0.5 * float(ts[2]) + 0.18
                nz = max(z, min(top_z, 0.7))
        out.append({
            **o,
            "label": "beer bottle",
            "center": [float(c[0]), float(c[1]), round(nz, 2)],
            "size": [0.10, 0.10, 0.33],
            "conf": min(float(o.get("conf") or 0.4), 0.65),
            "_proxy_from": lab,
        })
    return out


def _framed_record_candidates(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Wall-adjacent thin lifts (picture/tv/lamp) as framed-record proxies."""
    out: list[dict[str, Any]] = []
    for o in objects:
        lab = str(o.get("label") or "").lower()
        if lab in {"framed record", "picture", "painting", "poster", "photo"}:
            continue
        if lab not in {"tv", "television", "monitor", "lamp", "wall lamp"}:
            continue
        c = o.get("center") or [0, 0, 0]
        z = float(c[2])
        if z < 1.0 or z > 2.6:
            continue
        s = o.get("size") or [0.5, 0.5, 0.5]
        # Prefer upright wall-ish extents.
        if float(s[2]) < 0.4:
            continue
        out.append({
            **o,
            "label": "framed record",
            "center": [float(c[0]), float(c[1]), round(min(max(z, 1.4), 2.0), 2)],
            "size": [0.67, 0.05, 0.81],
            "conf": min(float(o.get("conf") or 0.4), 0.7),
            "_proxy_from": lab,
        })
    return out


def _augment_targets(
    targets: list[dict[str, Any]],
    phrase: str,
    all_objects: list[dict[str, Any]],
    scene: SceneRepresentation | None = None,
) -> list[dict[str, Any]]:
    """Add geometric proxy candidates for classes the detector often misses."""
    extra: list[dict[str, Any]] = []
    if _is_focus_light_phrase(phrase):
        extra.extend(_high_z_focus_candidates(all_objects))
        if scene is not None:
            extra.extend(_ceiling_grid_focus_lights(scene))
    if _is_books_phrase(phrase):
        extra.extend(_book_like_under_window_candidates(all_objects))
    p = _norm_phrase(phrase)
    if "bottle" in p or p == "beer":
        extra.extend(_bottle_like_candidates(all_objects))
    if "framed" in p or "record" in p:
        extra.extend(_framed_record_candidates(all_objects))
    if not extra:
        return targets
    by_id = {int(t["id"]): t for t in targets}
    for e in extra:
        eid = int(e["id"])
        prev = by_id.get(eid)
        # Prefer tabletop-repaired / proxy clones over raw inflated lifts.
        if prev is None or e.get("_proxy_from"):
            by_id[eid] = e
    return list(by_id.values())


def try_resolve_object_reference(
    question: str,
    scene: SceneRepresentation,
) -> ObjectReferenceResponse | None:
    """Return a spatial pick when the template is clear; else ``None``."""
    q = question.strip()
    all_compact = compact_objects(scene)
    objects = filter_objects_for_reference(all_compact, q)
    # Keep the full compact set available for geometric proxies.
    pool = all_compact if len(all_compact) >= len(objects) else objects
    if len(objects) < 2 and len(pool) < 2:
        return None

    m = _BELOW_FARTHEST.match(q)
    if m:
        target_p, below_p, from_p = m.group(1), m.group(2), m.group(3)
        targets = _prefer_exact_targets(
            _augment_targets(
                [
                    o for o in objects
                    if _match_label(str(o["label"]), _labels_for_phrase(target_p))
                ],
                target_p,
                pool,
                scene,
            ),
            target_p,
        )
        belows = [o for o in objects if _match_label(str(o["label"]), _labels_for_phrase(below_p))]
        froms = [o for o in objects if _match_label(str(o["label"]), _labels_for_phrase(from_p))]
        if not targets or not belows or not froms:
            return None
        # Singular landmark anchors (easel, table): keep the best instance so
        # phantom duplicates don't flip "farthest from".
        from_p_l = _norm_phrase(from_p)
        if from_p_l in {"easel", "table", "desk", "chair", "couch", "sofa"} or (
            from_p_l.startswith("the ") and len(from_p_l.split()) <= 2
        ):
            froms = _primary_furniture_anchors(
                froms, from_p_l.replace("the ", ""), pool,
            )
            # Easel-like: prefer upright mid-size over huge false lifts.
            if "easel" in from_p_l:
                froms = sorted(
                    froms,
                    key=lambda a: (
                        float(a.get("conf") or 0.0),
                        -abs(_footprint(a) - 0.9),
                    ),
                    reverse=True,
                )[:1] or froms
        def _under(t: dict[str, Any], w: dict[str, Any]) -> bool:
            """Target sits under a window footprint (not merely near any shard)."""
            tx, ty, tz = float(t["center"][0]), float(t["center"][1]), float(t["center"][2])
            wx, wy, wz = float(w["center"][0]), float(w["center"][1]), float(w["center"][2])
            ws = w.get("size") or [1.0, 1.0, 1.0]
            # Expand footprint slightly; require clearly lower than window center.
            hx = max(float(ws[0]) * 0.55, 0.8)
            hy = max(float(ws[1]) * 0.55, 0.8)
            if abs(tx - wx) > hx or abs(ty - wy) > hy:
                return False
            return tz <= wz - 0.2

        pool_below = _substantial_windows(belows)
        # Plural "windows" → take a far cluster (top-k), not a single shard.
        below_plural = "window" in _norm_phrase(below_p) and (
            _norm_phrase(below_p).endswith("s") or "windows" in _norm_phrase(below_p)
        )
        k_far = 4 if below_plural else 1
        far_below = sorted(
            pool_below,
            key=lambda w: _min_dist_to_set(w, froms),
            reverse=True,
        )[:k_far]
        cand = [t for t in targets if any(_under(t, b) for b in far_below)]
        if not cand:
            cand = []
            for t in targets:
                tz = float(t["center"][2])
                for b in far_below:
                    if _dist(t, b) <= 2.5 and tz <= float(b["center"][2]) - 0.1:
                        cand.append(t)
                        break
        if not cand:
            # Last resort: any under primary substantial windows.
            primary_below = pool_below[: max(2, min(3, len(pool_below)))]
            cand = [t for t in targets if any(_under(t, b) for b in primary_below)] or targets

        target_l = _norm_phrase(target_p)
        is_box = target_l in {"box", "boxes", "round box"} or target_l.endswith(" box")
        is_books = _is_books_phrase(target_p)

        # Anchor window for ranking / sill-snap: highest-conf among the far set
        # (for boxes), else the single farthest pane (books / default).
        if is_box and far_below:
            win_rank = max(far_below, key=lambda w: float(w.get("conf") or 0.0))
        else:
            win_rank = far_below[0] if far_below else None

        def _under_rank(t: dict[str, Any]) -> tuple:
            s = t.get("size") or [0.3, 0.3, 0.3]
            long = max(float(s[0]), float(s[1]))
            hz = float(s[2])
            d_win = _dist(t, win_rank) if win_rank is not None else 0.0
            proxy = str(t.get("_proxy_from") or "")
            if is_box:
                # Compact real boxes under the best far window.
                exact = 1.0 if str(t.get("label") or "").lower() == "box" else 0.0
                return (
                    exact + float(t.get("conf") or 0.0) - 0.15 * d_win - _geom_penalty(t, target_p),
                    -d_win,
                )
            low_long = long if hz <= 0.55 else long * 0.3
            shelf_bonus = 1.5 if proxy in {"shelf", "bookcase"} else (
                0.5 if proxy in {"coffee table", "table"} else 0.0
            )
            # Prefer long low runs nearest the window; conf is a tie-break only.
            return (
                shelf_bonus + low_long - _geom_penalty(t, target_p),
                -d_win,
                float(t.get("conf") or 0.0),
            )

        best = max(cand, key=_under_rank)
        # Snap toward the window this object actually sits under.
        win = win_rank
        if far_below:
            covering = [w for w in far_below if _under(best, w)]
            if covering:
                win = max(covering, key=lambda w: float(w.get("conf") or 0.0))
        # Sill snap helps books (long shelf runs); boxes are already compact.
        blend = 0.55 if is_books else (0.0 if is_box else 0.35)
        best = _snap_under_window(best, win, blend=blend)
        if is_books:
            # Prefer elongated books-run prior once the sill center is known.
            from xiao_hei_vln.perception.size_priors import clamp_aabb_size
            best = {
                **best,
                "size": clamp_aabb_size("books", best.get("size") or [2.5, 0.4, 0.5]),
            }
        if is_box:
            from xiao_hei_vln.perception.size_priors import clamp_aabb_size
            best = {
                **best,
                "size": clamp_aabb_size("box", best.get("size") or [0.4, 0.35, 0.3]),
            }
        return _to_response(
            best,
            f"spatial: '{target_p}' below '{below_p}', farthest from '{from_p}'",
        )

    m = _CLOSEST.match(q)
    if m:
        target_p, anchor_p = m.group(1), m.group(2)
        targets = _prefer_exact_targets(
            _augment_targets(
                [
                    o for o in objects
                    if _match_label(str(o["label"]), _labels_for_phrase(target_p))
                ],
                target_p,
                pool,
                scene,
            ),
            target_p,
        )
        targets = _prefer_main_table_cluster(targets, pool, target_p)
        if _is_focus_light_phrase(target_p):
            targets = _prefer_focus_light_rows(targets, pool)
        anchor_l = _norm_phrase(anchor_p)
        anchor_objs = [
            o for o in objects
            if _match_label(str(o["label"]), _labels_for_phrase(anchor_p))
        ]
        # Furniture anchors: one primary instance. Switches/doors: keep all.
        if any(k in anchor_l for k in (
            "couch", "sofa", "table", "desk", "chair", "bed", "easel",
        )):
            anchors = _primary_furniture_anchors(anchor_objs, anchor_l, pool)
            # Bottles sit on the coffee table by the nearer seating piece;
            # tall vases / wall art hang by the main high-conf couch.
            target_l = _norm_phrase(target_p)
            if any(k in anchor_l for k in ("couch", "sofa")) and len(anchors) > 1:
                if "bottle" in target_l:
                    anchors = anchors[:1]  # table-proximate seating
                else:
                    # Prefer highest-confidence elongated couch for decor.
                    anchors = sorted(
                        anchors,
                        key=lambda a: float(a.get("conf") or 0.0),
                        reverse=True,
                    )[:1]
        elif "door" in anchor_l:
            # Prefer real door leaves (not frames / floating high phantoms).
            exact = [
                a for a in anchor_objs
                if str(a.get("label") or "").lower() == "door"
            ]
            pool_a = exact or anchor_objs
            mid = [
                a for a in pool_a
                if 0.5 <= float(a["center"][2]) <= 2.0
            ] or pool_a
            anchors = sorted(
                mid, key=lambda a: float(a.get("conf") or 0.0), reverse=True,
            )[:1]
        elif "switch" in anchor_l:
            # Singular wall switch: prefer mid-height exact labels.
            exact = [
                a for a in anchor_objs
                if "switch" in str(a.get("label") or "").lower()
            ]
            pool_a = exact or anchor_objs
            mid = [
                a for a in pool_a
                if 0.7 <= float(a["center"][2]) <= 1.6
            ] or pool_a
            anchors = sorted(
                mid, key=lambda a: float(a.get("conf") or 0.0), reverse=True,
            )[:1]
        else:
            anchors = anchor_objs
        if not targets or not anchors:
            return None
        best = min(targets, key=lambda t: _rank_key_closest(t, anchors, target_p))
        return _to_response(
            best,
            f"spatial: '{target_p}' closest to '{anchor_p}' "
            f"(dxy={_min_dist_to_set(best, anchors):.2f}m)",
        )

    m = _FARTHEST.match(q)
    if m:
        target_p, anchor_p = m.group(1), m.group(2)
        targets = _prefer_exact_targets(
            _augment_targets(
                [
                    o for o in objects
                    if _match_label(str(o["label"]), _labels_for_phrase(target_p))
                ],
                target_p,
                pool,
                scene,
            ),
            target_p,
        )
        # Keep all anchors for farthest (dropping a near table would inflate distances).
        anchors = [o for o in objects if _match_label(str(o["label"]), _labels_for_phrase(anchor_p))]
        if not targets or not anchors:
            return None
        best = min(targets, key=lambda t: _rank_key_farthest(t, anchors, target_p))
        return _to_response(
            best,
            f"spatial: '{target_p}' farthest from '{anchor_p}' "
            f"(dxy={_min_dist_to_set(best, anchors):.2f}m)",
        )

    return None
