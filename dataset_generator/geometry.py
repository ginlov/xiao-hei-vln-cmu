"""Shared geometry gates for the dataset generators.

`closest`/`near` questions assume the target and anchor are in navigable
proximity. Straight-line xy distance through a wall isn't: "the cabinet closest
to the fire alarm" can pick a cabinet on the far side of a glass partition. This
module flags target->anchor segments that cross a structural wall, so both the
single-layer (`vla3d_ref_to_qa`) and nested (`vla3d_nested_gen`) generators can
apply the same gate.
"""

from __future__ import annotations

import math

from vla3d_loader import VLAObject, VLAScene

# Wall must be at least this tall to count as a divider (so flat "wall decal" /
# "wall lamp" fixtures are excluded; glass partitions are included).
_WALL_MIN_HEIGHT = 1.2
# Keep the crossing away from both segment ends so a wall-mounted anchor
# (picture / light switch / fire alarm, crossing near t=1) or an object backed
# against its own wall (t~0) isn't mistaken for a divider strictly between them.
_WALL_T_LO, _WALL_T_HI = 0.08, 0.92

_SCENE_WALLS: dict[str, list[VLAObject]] = {}


def scene_walls(sc: VLAScene) -> list[VLAObject]:
    """Structural walls (incl. glass partitions — real navigation barriers)."""
    if sc.name not in _SCENE_WALLS:
        ws = []
        for o in sc.objects:
            lbl = o.raw_label.lower()
            if ("wall" in lbl
                    and not any(x in lbl for x in
                                ("decal", "lamp", "clock", "art", "paper",
                                 "mount", "sticker", "switch"))
                    and o.lz >= _WALL_MIN_HEIGHT):
                ws.append(o)
        _SCENE_WALLS[sc.name] = ws
    return _SCENE_WALLS[sc.name]


def _seg_obb_interval(p0, p1, w: VLAObject) -> tuple[float, float] | None:
    """Liang-Barsky clip of segment p0->p1 to wall w's xy OBB. Returns the
    crossing parameter interval (t0, t1) along the segment, or None."""
    ca, sa = math.cos(-w.heading), math.sin(-w.heading)
    hx, hy = w.lx / 2, w.ly / 2

    def loc(p):
        dx, dy = p[0] - w.x, p[1] - w.y
        return (ca * dx - sa * dy, sa * dx + ca * dy)

    a = loc(p0)
    b = loc(p1)
    dx, dy = b[0] - a[0], b[1] - a[1]
    t0, t1 = 0.0, 1.0
    for pdir, q in ((-dx, a[0] + hx), (dx, hx - a[0]),
                    (-dy, a[1] + hy), (dy, hy - a[1])):
        if abs(pdir) < 1e-12:
            if q < 0:
                return None
        else:
            r = q / pdir
            if pdir < 0:
                if r > t1:
                    return None
                t0 = max(t0, r)
            else:
                if r < t0:
                    return None
                t1 = min(t1, r)
    return (t0, t1) if t0 < t1 else None


def wall_between(sc: VLAScene, a: VLAObject, b: VLAObject) -> bool:
    """True if a structural wall separates a and b (their xy segment crosses a
    wall in its middle portion). Both objects must be in the same region."""
    if a is None or b is None:
        return False
    for w in scene_walls(sc):
        if w.region_id != a.region_id:
            continue
        iv = _seg_obb_interval((a.x, a.y), (b.x, b.y), w)
        if iv and iv[0] < _WALL_T_HI and iv[1] > _WALL_T_LO:
            return True
    return False
