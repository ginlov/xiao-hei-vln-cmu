"""Deterministic viewpoint generation for perception benchmarking.

Implements the algorithm in ``VIEWPOINT_ALGORITHM.md``: for a scene, choose a
small set of robot positions (360-deg camera -> position only) such that every
object has geometric line-of-sight from >= k of them, then emit teleport poses
for the capture harness.

Offline eval tooling; NOT part of the challenge submission stack. Reads the
VLA-3D GT data (see the ``vla3d-scene-gt-data`` memory).

    uv run python perception_benchmark/viewgen.py --scene arabic_room
    uv run python perception_benchmark/viewgen.py --all --out perception_benchmark/viewpoints
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from xiao_hei_vln.eval_sampler.object_list import ObjectEntry, parse_object_list

DATA_ROOT = Path(os.environ.get(
    "VLA3D_UNITY_DIR", "/home/long/Projects/dataset/vla-3d/Unity"))

# ── occluder classification ────────────────────────────────────────────────────
# Structural surfaces that block line of sight. floor/ceiling are horizontal, so
# in a top-down 2D grid they would blanket everything -> excluded. Thin wall
# decor (lamp/decal/art) sits ON a wall and is not itself an occluder.
_WALL_KEYS = ("wall", "column", "pillar", "partition")
_WALL_EXCLUDE = ("lamp", "decal", "decor", "art", "picture", "clock")


def is_occluder(label: str) -> bool:
    lo = label.lower()
    if "floor" in lo or "ceiling" in lo:
        return False
    if any(k in lo for k in _WALL_KEYS) and not any(x in lo for x in _WALL_EXCLUDE):
        return True
    return False


# ── data loaders ───────────────────────────────────────────────────────────────

def load_objects(scene: str) -> dict[int, ObjectEntry]:
    lines = (DATA_ROOT / scene / "object_list.txt").read_text().splitlines()
    return parse_object_list(lines)


_PLY_TYPE_BYTES = {"float": 4, "float32": 4, "double": 8, "uchar": 1,
                   "uint8": 1, "int": 4, "int32": 4, "uint": 4}


def _ply_layout(path: Path):
    """Return (n_vertices, data_start_byte, stride_bytes) for a binary PLY."""
    with open(path, "rb") as f:
        n = 0
        stride = 0
        while True:
            line = f.readline()
            s = line.strip()
            if s.startswith(b"element vertex"):
                n = int(s.split()[-1])
            elif s.startswith(b"property"):
                stride += _PLY_TYPE_BYTES[s.split()[1].decode()]
            elif s == b"end_header":
                return n, f.tell(), stride


def load_free_points(scene: str) -> np.ndarray:
    """Parse ``<scene>_free_space_pc_result.ply`` -> (N, 3) xyz (map frame).

    Binary Open3D PLY, 23 bytes/vertex: float x,y,z (12) + uchar rgb (3) +
    int obj_id (4) + int region_id (4)."""
    path = DATA_ROOT / scene / f"{scene}_free_space_pc_result.ply"
    n, start, stride = _ply_layout(path)
    with open(path, "rb") as f:
        f.seek(start)
        buf = f.read(n * stride)
    dt = np.dtype({"names": ["x", "y", "z"], "formats": ["<f4"] * 3,
                   "offsets": [0, 4, 8], "itemsize": stride})
    a = np.frombuffer(buf, dtype=dt, count=n)
    return np.stack([a["x"], a["y"], a["z"]], axis=1).astype(np.float64)


def load_wall_points(scene: str, zband=(0.2, 2.6), max_pts_per_wall=200000) -> np.ndarray:
    """Read only the wall-object points from the dense ``pc_result.ply``.

    A wall object's *bbox* encloses free space (thin shell around a big empty
    box), so filling bboxes solid blocks whole rooms. Instead we pull the actual
    wall surface points — ``object_split.npy`` gives each object's contiguous
    point range, so we seek and read just the wall ranges (not all ~40 M pts)."""
    objs = load_objects(scene)
    wall_ids = {o.object_id for o in objs.values() if is_occluder(o.label)}
    if not wall_ids:
        return np.empty((0, 3))
    split = np.load(DATA_ROOT / scene / f"{scene}_object_split.npy")  # (K, [id, end])
    ends = split[:, 1]
    starts = np.concatenate([[0], ends[:-1]])
    ranges = {int(split[i, 0]): (int(starts[i]), int(ends[i])) for i in range(len(split))}

    path = DATA_ROOT / scene / f"{scene}_pc_result.ply"
    n, data0, stride = _ply_layout(path)
    dt = np.dtype({"names": ["x", "y", "z"], "formats": ["<f4"] * 3,
                   "offsets": [0, 4, 8], "itemsize": stride})
    chunks = []
    with open(path, "rb") as f:
        for wid in wall_ids:
            if wid not in ranges:
                continue
            s, e = ranges[wid]
            cnt = e - s
            if cnt <= 0:
                continue
            f.seek(data0 + s * stride)
            buf = f.read(cnt * stride)
            a = np.frombuffer(buf, dtype=dt, count=cnt)
            xyz = np.stack([a["x"], a["y"], a["z"]], axis=1).astype(np.float64)
            xyz = xyz[(xyz[:, 2] >= zband[0]) & (xyz[:, 2] <= zband[1])]
            if len(xyz) > max_pts_per_wall:                # subsample dense walls
                xyz = xyz[np.random.default_rng(0).choice(len(xyz), max_pts_per_wall, False)]
            chunks.append(xyz)
    return np.vstack(chunks) if chunks else np.empty((0, 3))


# ── occupancy grid ──────────────────────────────────────────────────────────────

@dataclass
class Grid:
    occ: np.ndarray          # (H, W) bool — True = occluder
    origin: np.ndarray       # (2,) world xy of cell (0,0) lower-left
    res: float

    def world_to_cell(self, xy: np.ndarray) -> np.ndarray:
        return np.floor((xy - self.origin) / self.res).astype(np.int64)


def load_floor_xy(scene: str, max_pts=500000) -> np.ndarray:
    """(N,2) xy of the scene's floor-surface points (from ``pc_result.ply``).

    The floor object's *bbox* is an AABB that includes exterior notches, but its
    POINTS trace the true room footprint — so 'is a point over the floor?' is a
    reliable inside-the-room test (free-space membership is not: the free-space
    cloud leaks into exterior areas)."""
    objs = load_objects(scene)
    floor_ids = {o.object_id for o in objs.values() if "floor" in o.label.lower()}
    if not floor_ids:
        return np.empty((0, 2))
    split = np.load(DATA_ROOT / scene / f"{scene}_object_split.npy")
    ends = split[:, 1]
    starts = np.concatenate([[0], ends[:-1]])
    ranges = {int(split[i, 0]): (int(starts[i]), int(ends[i])) for i in range(len(split))}
    path = DATA_ROOT / scene / f"{scene}_pc_result.ply"
    n, data0, stride = _ply_layout(path)
    dt = np.dtype({"names": ["x", "y"], "formats": ["<f4"] * 2,
                   "offsets": [0, 4], "itemsize": stride})
    chunks = []
    with open(path, "rb") as f:
        for fid in floor_ids:
            if fid not in ranges:
                continue
            s, e = ranges[fid]
            f.seek(data0 + s * stride)
            a = np.frombuffer(f.read((e - s) * stride), dtype=dt, count=e - s)
            chunks.append(np.stack([a["x"], a["y"]], axis=1).astype(np.float64))
    xy = np.vstack(chunks) if chunks else np.empty((0, 2))
    if len(xy) > max_pts:
        xy = xy[np.random.default_rng(0).choice(len(xy), max_pts, False)]
    return xy


def over_floor_mask(floor_xy: np.ndarray, pts_xy: np.ndarray,
                    res=0.25, dilate_m=0.6) -> np.ndarray:
    """Bool per pt: is it over the room floor (inside the room)?

    Rasterizes the floor points to a 2D grid and dilates by ``dilate_m`` (to
    tolerate locally-sparse floor sampling and reach floor edges), then tests
    each point's cell. Grid+dilate is far more robust on big scenes than a
    nearest-point distance, which false-flags interior points where the floor
    mesh is sparse (e.g. under furniture)."""
    if len(floor_xy) == 0 or len(pts_xy) == 0:
        return np.ones(len(pts_xy), dtype=bool)   # no floor data -> can't reject
    origin = floor_xy.min(0) - 1.0
    ext = floor_xy.max(0) + 1.0 - origin
    W, H = int(ext[0] / res) + 1, int(ext[1] / res) + 1
    occ = np.zeros((H, W), dtype=bool)
    ci = ((floor_xy - origin) / res).astype(np.int64)
    occ[ci[:, 1], ci[:, 0]] = True
    for _ in range(int(round(dilate_m / res))):    # close gaps + reach edges
        occ[1:, :] |= occ[:-1, :]; occ[:-1, :] |= occ[1:, :]
        occ[:, 1:] |= occ[:, :-1]; occ[:, :-1] |= occ[:, 1:]
    cj = ((pts_xy - origin) / res).astype(np.int64)
    inb = (cj[:, 0] >= 0) & (cj[:, 0] < W) & (cj[:, 1] >= 0) & (cj[:, 1] < H)
    out = np.zeros(len(pts_xy), dtype=bool)
    out[inb] = occ[cj[inb, 1], cj[inb, 0]]
    return out


def build_grid(wall_pts, free_xy, res=0.1, pad=6.0, dilate=1, min_pts_cell=1) -> Grid:
    """Rasterize actual wall-surface points into a 2D occluder grid.

    Cells hit by >= ``min_pts_cell`` wall points become occluders. This traces
    true (thin) wall geometry, unlike filling a wall's enclosing bbox."""
    allxy = np.vstack([free_xy, wall_pts[:, :2]]) if len(wall_pts) else free_xy
    origin = allxy.min(0) - pad
    extent = (allxy.max(0) + pad) - origin
    W, H = int(np.ceil(extent[0] / res)), int(np.ceil(extent[1] / res))
    occ = np.zeros((H, W), dtype=bool)
    if len(wall_pts):
        cx = ((wall_pts[:, 0] - origin[0]) / res).astype(np.int64)
        cy = ((wall_pts[:, 1] - origin[1]) / res).astype(np.int64)
        ok = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
        counts = np.zeros((H, W), dtype=np.int32)
        np.add.at(counts, (cy[ok], cx[ok]), 1)
        occ = counts >= min_pts_cell
    for _ in range(dilate):                     # close thin-wall diagonal gaps
        occ[1:, :] |= occ[:-1, :]; occ[:-1, :] |= occ[1:, :]
        occ[:, 1:] |= occ[:, :-1]; occ[:, :-1] |= occ[:, 1:]
    return Grid(occ, origin, res)


# ── visibility ──────────────────────────────────────────────────────────────────

def _blocked(grid: Grid, p: np.ndarray, t: np.ndarray, stop_short: float) -> bool:
    """True if the segment p->t crosses an occluder cell. The last
    ``stop_short`` m before t are ignored so wall-mounted objects (picture on a
    wall) stay visible from the room side."""
    d = t - p
    dist = math.hypot(d[0], d[1])
    if dist < 1e-6:
        return False
    end = max(0.0, dist - stop_short)
    n = int(end / grid.res) + 1
    ts = np.linspace(0.0, end, n)
    xs = p[0] + (d[0] / dist) * ts
    ys = p[1] + (d[1] / dist) * ts
    cx = ((xs - grid.origin[0]) / grid.res).astype(np.int64)
    cy = ((ys - grid.origin[1]) / grid.res).astype(np.int64)
    H, W = grid.occ.shape
    ok = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
    return bool(grid.occ[cy[ok], cx[ok]].any())


def candidate_positions(grid: Grid, free_xyz: np.ndarray, robot_radius=0.3):
    """Free-space points at least robot_radius from any occluder cell."""
    rad = int(math.ceil(robot_radius / grid.res))
    H, W = grid.occ.shape
    keep = []
    for i, p in enumerate(free_xyz):
        c = grid.world_to_cell(p[:2])
        x0, x1 = max(0, c[0] - rad), min(W, c[0] + rad + 1)
        y0, y1 = max(0, c[1] - rad), min(H, c[1] + rad + 1)
        if not grid.occ[y0:y1, x0:x1].any():
            keep.append(i)
    return free_xyz[keep] if keep else free_xyz[:0]


def visibility(grid, cands, objects, r_min, r_max, stop_short):
    """vis[i] = list of (obj_id, bearing_rad) visible from candidate i."""
    obj_ids = list(objects.keys())
    ocenters = np.array([[objects[k].center.x, objects[k].center.y] for k in obj_ids])
    vis = []
    for p in cands:
        d = ocenters - p[:2]
        dist = np.hypot(d[:, 0], d[:, 1])
        cand = np.where((dist >= r_min) & (dist <= r_max))[0]
        seen = []
        for j in cand:
            t = ocenters[j]
            if not _blocked(grid, p[:2], t, stop_short):
                seen.append((obj_ids[j], math.atan2(d[j, 1], d[j, 0])))
        vis.append(seen)
    return vis, obj_ids


# ── greedy k-set-cover with angular separation ──────────────────────────────────

def _ang_ok(bearing, existing, min_sep):
    return all(abs((bearing - b + math.pi) % (2 * math.pi) - math.pi) >= min_sep
               for b in existing)


def greedy_cover(cands, vis, obj_ids, k=2, min_sep_deg=30.0):
    min_sep = math.radians(min_sep_deg)
    cover_bearings = {o: [] for o in obj_ids}     # object -> chosen bearings
    chosen = []
    remaining = set(range(len(cands)))
    while True:
        best_i, best_gain, best_new = -1, 0, None
        for i in remaining:
            newly = [(o, b) for (o, b) in vis[i]
                     if len(cover_bearings[o]) < k
                     and _ang_ok(b, cover_bearings[o], min_sep)]
            if len(newly) > best_gain:
                best_gain, best_i, best_new = len(newly), i, newly
        if best_i < 0 or best_gain == 0:
            break
        chosen.append((best_i, [o for o, _ in best_new]))
        for o, b in best_new:
            cover_bearings[o].append(b)
        remaining.discard(best_i)
    coverage = {o: len(v) for o, v in cover_bearings.items()}
    return chosen, coverage


# ── patch uncovered objects with close-approach viewpoints ──────────────────────

def patch_uncovered(grid, objects, coverage, free_xyz, r_min, r_max, stop_short):
    """For each 0-coverage object, find the nearest free position with LOS.
    Returns (extra_viewpoints, still_missing)."""
    extra, missing = [], []
    for o, c in coverage.items():
        if c > 0:
            continue
        oc = np.array([objects[o].center.x, objects[o].center.y])
        d = np.hypot(free_xyz[:, 0] - oc[0], free_xyz[:, 1] - oc[1])
        order = np.argsort(d)
        placed = False
        for idx in order:
            p = free_xyz[idx]
            dist = d[idx]
            if dist < r_min or dist > r_max:
                continue
            if not _blocked(grid, p[:2], oc, stop_short):
                extra.append((p, o)); placed = True; break
        if not placed:
            missing.append(o)
    return extra, missing


# ── driver ───────────────────────────────────────────────────────────────────────

def generate(scene: str, *, r_min=0.5, r_max=6.0, k=2, min_sep_deg=30.0,
             res=0.1, robot_radius=0.3, stop_short=0.3, verbose=True) -> dict:
    objects = load_objects(scene)
    free = load_free_points(scene)
    wall_pts = load_wall_points(scene)
    # Keep only free-space points OVER THE ROOM FLOOR — the free-space cloud
    # leaks into exterior areas (notches, courtyards), and we want strictly
    # inside-the-room viewpoints. Falls back to all-free when no floor object.
    floor_xy = load_floor_xy(scene)
    n_free_all = len(free)
    free = free[over_floor_mask(floor_xy, free[:, :2])]
    grid = build_grid(wall_pts, free[:, :2], res=res)
    cands = candidate_positions(grid, free, robot_radius=robot_radius)
    if verbose:
        n_walls = sum(is_occluder(o.label) for o in objects.values())
        print(f"[{scene}] objects={len(objects)} walls={n_walls} "
              f"wall_pts={len(wall_pts)} free={len(free)}/{n_free_all}(in-floor) "
              f"candidates={len(cands)} grid={grid.occ.shape} occ%={grid.occ.mean():.1%}")

    vis, obj_ids = visibility(grid, cands, objects, r_min, r_max, stop_short)
    chosen, coverage = greedy_cover(cands, vis, obj_ids, k=k, min_sep_deg=min_sep_deg)
    extra, missing = patch_uncovered(grid, objects, coverage, free,
                                     r_min, r_max, stop_short)
    for _, o in extra:
        coverage[o] = max(coverage[o], 1)

    # assemble output viewpoints (z from the candidate's own free-space z)
    vps = []
    for vid, (ci, covers) in enumerate(chosen):
        p = cands[ci]
        vps.append({"id": vid, "x": round(float(p[0]), 3), "y": round(float(p[1]), 3),
                    "z": round(float(p[2]), 3), "yaw": 0.0, "covers": sorted(covers)})
    for p, o in extra:
        vps.append({"id": len(vps), "x": round(float(p[0]), 3), "y": round(float(p[1]), 3),
                    "z": round(float(p[2]), 3), "yaw": 0.0, "covers": [o], "patch": True})

    covered = sum(1 for c in coverage.values() if c > 0)
    mv = float(np.mean([c for c in coverage.values()])) if coverage else 0.0
    report = {
        "n_objects": len(objects), "covered": covered,
        "not_observable": sorted(missing),
        "n_viewpoints": len(vps), "patched": len(extra),
        "mean_views_per_object": round(mv, 2),
        "coverage_frac": round(covered / max(len(objects), 1), 3),
    }
    if verbose:
        print(f"[{scene}] viewpoints={len(vps)} covered={covered}/{len(objects)} "
              f"({report['coverage_frac']:.0%}) patched={len(extra)} "
              f"unobservable={len(missing)} mean_views={report['mean_views_per_object']}")
    return {"scene": scene, "frame": "map", "params": {
        "r_min": r_min, "r_max": r_max, "k": k, "min_sep_deg": min_sep_deg,
        "res": res, "robot_radius": robot_radius, "stop_short": stop_short},
        "viewpoints": vps, "report": report}


ALL_SCENES = ["arabic_room", "chinese_room", "home_building_1", "home_building_2",
              "hotel_room_1", "hotel_room_2", "japanese_room", "livingroom_1",
              "livingroom_2", "livingroom_3", "livingroom_4", "loft", "office_1",
              "office_2", "studio"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("perception_benchmark/viewpoints"))
    ap.add_argument("--r-max", type=float, default=6.0)
    ap.add_argument("--k", type=int, default=2)
    args = ap.parse_args()

    scenes = ALL_SCENES if args.all else [args.scene]
    if scenes == [None]:
        ap.error("pass --scene <name> or --all")
    args.out.mkdir(parents=True, exist_ok=True)
    summary = []
    for s in scenes:
        out = generate(s, r_max=args.r_max, k=args.k)
        json.dump(out, open(args.out / f"{s}.json", "w"), indent=2)
        summary.append((s, out["report"]))
    if len(summary) > 1:
        print(f"\n{'scene':18s} {'vps':>4s} {'cov':>10s} {'patch':>5s} {'unobs':>5s} {'mv':>4s}")
        for s, r in summary:
            print(f"{s:18s} {r['n_viewpoints']:4d} "
                  f"{r['covered']:3d}/{r['n_objects']:<3d}({r['coverage_frac']:.0%}) "
                  f"{r['patched']:5d} {len(r['not_observable']):5d} "
                  f"{r['mean_views_per_object']:4.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
