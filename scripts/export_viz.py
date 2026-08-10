#!/usr/bin/env python3
"""Dump one replayed scene into something the browser viewer can draw.

Every number we argue about -- recall, box IoU, duplicate nodes -- is an
average over hundreds of observations, and an average cannot show *why* a box
is wrong. Twice now a picture has overturned a conclusion the aggregates
supported: the equirect overlay is what caught the ``fan decoration`` ground
truth being one box around three fans plus the wall between them. This exports
the same pipeline the benchmark scores, per frame, so the boxes on screen are
the boxes in the table -- not a re-derivation that might drift from it.

Output is a JSON manifest plus one ``float32`` blob of ``xyz`` triples; the
manifest indexes into the blob by point offset so the page fetches geometry as
an ``ArrayBuffer`` and never parses coordinates out of text.

The two halves have very different costs. The perception half needs a recorded
tour of the scene, which costs sim time; the ground-truth half is read straight
out of the scene model and costs nothing. A scene with no recorded tour is
therefore still exported, ground truth only -- that is what the offline scorer
and ``anchor_size.py`` read, and holding it hostage to a tour left half the
official question set unscoreable.

    uv run python scripts/export_viz.py --scene office_2
    uv run python scripts/export_viz.py --scene all --stride 3
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from xiao_hei_vln.eval_sampler.object_list import parse_object_list
from xiao_hei_vln.perception import object_map as om_mod
from xiao_hei_vln.perception import replay
from xiao_hei_vln.perception.lifter import PointLifter
from xiao_hei_vln.perception.object_map import ObjectMap
from xiao_hei_vln.perception.vocab import is_structure

# The scenes the official question set asks about. Seven have a recorded tour;
# the rest export ground truth only until one is recorded.
OFFICIAL_SCENES = (
    "arabic_room", "chinese_room", "home_building_1", "home_building_2",
    "hotel_room_1", "hotel_room_2", "japanese_room", "livingroom_1",
    "livingroom_2", "livingroom_3", "livingroom_4", "loft", "office_1",
    "office_2", "studio",
)
# Where the scene models sit on the two machines this runs on: unpacked under
# the sim host's dataset directory, and as the download batches they arrived in
# on the laptop. A root that does not exist is skipped, so listing both layouts
# costs nothing.
DEFAULT_GT_ROOTS = (
    Path.home() / "workspace/dataset/unity-scene",
    Path.home() / "Workspace/vln-challenge/dataset-official/unity_env_models",
    Path.home() / "Workspace/vln-challenge/dataset-official/unity_env_models 2",
    Path.home() / "Workspace/vln-challenge/dataset-official/unity_env_models 3",
)
# The benchmark's gating, so what is drawn is what is measured.
LIFT = dict(min_move_m=0.15, min_rot_deg=10.0, min_score=0.35, min_inliers=10)


class Blob:
    """Append-only float32 xyz store handing back ``(offset, count)``."""

    def __init__(self) -> None:
        self.chunks: list[np.ndarray] = []
        self.n = 0

    def add(self, pts) -> list[int]:
        # Recorded scans carry a fourth column (intensity); everything drawn
        # here is position only.
        pts = xyz(pts)
        if not len(pts):
            return [self.n, 0]
        self.chunks.append(pts)
        off, self.n = self.n, self.n + len(pts)
        return [off, len(pts)]

    def tobytes(self) -> bytes:
        if not self.chunks:
            return b""
        return np.concatenate(self.chunks, axis=0).astype(np.float32).tobytes()


class Provenance:
    """Record which frames each node was observed in.

    ``ObjectMap.add`` picks an existing node or creates one without telling the
    caller which, so the frame stamp has to be taken from the node side. The
    two entry points -- a node being born, and a node absorbing another
    observation -- cover both paths without duplicating the merge rule here,
    which would be free to drift from the real one. ``node_id`` is stable
    through ``finalize()``, so the map still keys the exported objects.
    """

    def __init__(self) -> None:
        self.frame = 0
        self.seen: dict[int, set[int]] = {}

    def __enter__(self) -> "Provenance":
        node = om_mod._Node
        self._init, self._merge = node.__init__, node.merge

        def __init__(nd, node_id, *a, **k):
            self.seen.setdefault(int(node_id), set()).add(self.frame)
            return self._init(nd, node_id, *a, **k)

        def merge(nd, *a, **k):
            self.seen.setdefault(int(nd.node_id), set()).add(self.frame)
            return self._merge(nd, *a, **k)

        node.__init__, node.merge = __init__, merge
        return self

    def __exit__(self, *exc) -> bool:
        om_mod._Node.__init__, om_mod._Node.merge = self._init, self._merge
        return False


class GtSource:
    """One scene's ground truth, whether it is a zip or an unpacked directory.

    The official models arrive as several download batches -- on the laptop
    they sit in three sibling directories, two of them with a space in the name
    -- while the sim host keeps them unpacked next to the simulator. Both forms
    hold the same ``object_list.txt`` and ``map.ply``, so read either rather
    than maintaining a fourth copy that has to stay in sync with both.
    """

    def __init__(self, path: Path, scene: str) -> None:
        self.path, self.scene = path, scene

    @classmethod
    def find(cls, scene: str, roots: Sequence[Path]) -> "GtSource | None":
        for r in roots:
            if (r / f"{scene}.zip").is_file():
                return cls(r / f"{scene}.zip", scene)
            if (r / scene / "object_list.txt").is_file():
                return cls(r / scene, scene)
        return None

    @contextmanager
    def open(self, member: str):
        """A binary handle on one member. Raises if it is not there."""
        if self.path.suffix == ".zip":
            with zipfile.ZipFile(self.path) as zf, \
                    zf.open(f"{self.scene}/{member}") as fh:
                yield fh
        else:
            with open(self.path / member, "rb") as fh:
                yield fh

    def objects(self) -> dict:
        with self.open("object_list.txt") as fh:
            return parse_object_list([raw.decode().strip() for raw in fh])


_PLY_SIZES = {"char": 1, "uchar": 1, "int8": 1, "uint8": 1,
              "short": 2, "ushort": 2, "int16": 2, "uint16": 2,
              "int": 4, "uint": 4, "int32": 4, "uint32": 4,
              "float": 4, "float32": 4, "double": 8, "float64": 8}


def read_ply_xyz(fh) -> np.ndarray:
    """Vertex positions from a binary-little-endian PLY, ignoring the rest.

    The challenge scenes ship `map.ply` as bare float x/y/z, but reading the
    header rather than assuming that keeps this working if a scene carries
    colour or normals too.
    """
    header = b""
    while b"end_header" not in header:
        chunk = fh.read(8192)
        if not chunk:
            raise ValueError("truncated PLY: no end_header")
        header += chunk
    cut = header.index(b"\n", header.index(b"end_header")) + 1
    lines = header[:cut].decode("ascii", "replace").splitlines()
    rest = header[cut:]

    if not any(ln.startswith("format binary_little_endian") for ln in lines):
        raise ValueError("only binary_little_endian PLY is supported")

    count, props, in_vertex = 0, [], False
    for ln in lines:
        parts = ln.split()
        if parts[:2] == ["element", "vertex"]:
            count, in_vertex = int(parts[2]), True
        elif parts[:1] == ["element"]:
            in_vertex = False
        elif in_vertex and parts[:1] == ["property"] and len(parts) >= 3:
            props.append((parts[1], parts[2]))

    stride = sum(_PLY_SIZES[t] for t, _ in props)
    off, offsets = 0, {}
    for typ, name in props:
        offsets[name] = off
        off += _PLY_SIZES[typ]
    if not all(k in offsets for k in "xyz"):
        raise ValueError("PLY has no x/y/z")

    need = count * stride
    buf = bytearray(rest)
    while len(buf) < need:
        chunk = fh.read(min(1 << 22, need - len(buf)))
        if not chunk:
            break
        buf += chunk
    raw = np.frombuffer(bytes(buf[:need]), dtype=np.uint8).reshape(-1, stride)
    out = np.empty((len(raw), 3), dtype=np.float32)
    for i, k in enumerate("xyz"):
        o = offsets[k]
        out[:, i] = raw[:, o:o + 4].copy().view(np.float32).ravel()
    return out


def xyz(pts) -> np.ndarray:
    """``(N, >=3) -> (N, 3)`` float32, dropping intensity and friends."""
    a = np.asarray(pts, dtype=np.float32)
    if a.size == 0:
        return a.reshape(0, 3)
    if a.ndim == 1:
        a = a.reshape(-1, 3)
    return np.ascontiguousarray(a[:, :3])


def subsample(pts: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    if cap <= 0 or len(pts) <= cap:
        return pts
    return pts[rng.choice(len(pts), cap, replace=False)]


def voxel_thin(pts: np.ndarray, voxel: float, cap: int) -> np.ndarray:
    """One point per occupied voxel -- the accumulated cloud, made drawable."""
    pts = xyz(pts)
    if not len(pts):
        return pts
    keys = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    out = pts[np.sort(idx)]
    if cap and len(out) > cap:
        out = out[np.linspace(0, len(out) - 1, cap).astype(int)]
    return out


def export_scene(scene: str, frames_root: Path, gt_roots: Sequence[Path],
                 out_dir: Path, *, stride: int, max_scan: int, max_det: int,
                 world_voxel: float, max_world: int,
                 gt_voxel: float, max_gt: int, force: bool = False) -> bool:
    frames_dir = frames_root / f"{scene}_tour"
    gt_only = not frames_dir.is_dir()
    if gt_only:
        # The corpus lives on whichever machine recorded the tour, so running
        # this on the laptop finds none of it. Downgrading an existing full
        # export to ground truth only would silently destroy hours of sim time,
        # and the export it replaced is not reproducible here.
        prev = out_dir / f"{scene}.json"
        if prev.is_file() and not force:
            try:
                if json.loads(prev.read_text()).get("frames"):
                    print(f"{scene}: keeping the existing full export "
                          f"(no corpus here; --force to overwrite)",
                          file=sys.stderr)
                    return True
            except (OSError, ValueError):
                pass
        # Otherwise this is not an error: eight of the fifteen official scenes
        # have never been driven. The loop below runs zero times and the
        # ground-truth half carries the export on its own.
        print(f"{scene}: no corpus at {frames_dir} — ground truth only",
              file=sys.stderr)

    frames, detections = [], {}
    if not gt_only:
        frames = replay.load_frames(frames_dir, use_image_pose=True,
                                    min_move_m=LIFT["min_move_m"],
                                    min_rot_deg=LIFT["min_rot_deg"])
        detections = replay.load_detections(frames_dir)
    lifter = PointLifter(min_inliers=LIFT["min_inliers"])
    omap = ObjectMap()
    rng = np.random.default_rng(0)

    blob, out_frames, world = Blob(), [], []
    label_ids: dict[str, int] = {}

    def lid(name: str) -> int:
        return label_ids.setdefault(name, len(label_ids))

    prov = Provenance()
    ctx = prov.__enter__()
    for i, frame in enumerate(frames):
        # Fusion sees every frame; only every `stride`-th is drawable, so the
        # stamp is the frame the viewer will actually be able to stop on.
        ctx.frame = min(i // stride, max(len(frames) - 1, 0) // stride)
        dets = detections.get(frame.tick_id) or []
        scan = frame.scan()
        keep = i % stride == 0
        rec = {"t": int(frame.tick_id),
               "p": [round(float(x), 4) for x in
                     (frame.position.x, frame.position.y, frame.position.z)],
               "dets": []} if keep else None

        for det in dets:
            if float(det["score"]) < LIFT["min_score"]:
                continue
            res = lifter.lift(mask=replay.decode_mask(det["mask_rle"]),
                              scan_points_map=scan,
                              pose_position=frame.position,
                              pose_orientation=frame.orientation)
            if res.position is None:
                continue
            omap.add(det["label"], float(det["score"]), res.inlier_points)
            pts = xyz(res.inlier_points)
            if rec is not None:
                lo, hi = pts.min(axis=0), pts.max(axis=0)
                rec["dets"].append({
                    "l": lid(det["label"]),
                    "s": round(float(det["score"]), 3),
                    "lo": [round(float(x), 4) for x in lo],
                    "hi": [round(float(x), 4) for x in hi],
                    "n": int(len(pts)),
                    "pts": blob.add(subsample(pts, max_det, rng)),
                })

        if keep:
            rec["scan"] = blob.add(subsample(scan, max_scan, rng))
            out_frames.append(rec)
            world.append(subsample(scan, max_scan, rng))
        if (i + 1) % 50 == 0:
            print(f"  {scene}: {i + 1}/{len(frames)} frames", flush=True)

    prov.__exit__()

    # The accumulated cloud, thinned once rather than summed in the browser.
    acc = voxel_thin(np.concatenate(world, axis=0) if world
                     else np.zeros((0, 3), np.float32), world_voxel, max_world)
    world_ref = blob.add(acc)

    last = max(len(out_frames) - 1, 0)
    objects = []
    for nd in omap.export():
        b = nd["bbox_aabb"]
        fs = sorted({min(f, last) for f in prov.seen.get(int(nd["node_id"]), ())})
        objects.append({"l": lid(nd["label"]), "lo": b["min"], "hi": b["max"],
                        "c": nd["center_3d"], "s": nd["score"],
                        "n_obs": nd["n_obs"], "n_pts": nd["n_pts"],
                        "struct": bool(nd["is_structure"]),
                        "f0": fs[0] if fs else 0, "fs": fs})

    gt, gt_cloud, gt_pts = [], [0, 0], None
    src = GtSource.find(scene, gt_roots)
    if src is not None:
        # `heading` is carried through so the viewer can draw the box the scene
        # actually annotates. perception/eval.py builds its AABB as
        # `center +- size/2` with the heading *ignored*, so for any rotated
        # object the box we score against is the object's local box dropped
        # into the world unrotated -- neither the true oriented box nor its
        # world-aligned bound. Worth being able to see.
        for e in src.objects().values():
            c = np.array([e.center.x, e.center.y, e.center.z], dtype=float)
            s = np.array([e.size.x, e.size.y, e.size.z], dtype=float)
            gt.append({"l": lid(e.label), "c": [round(float(x), 4) for x in c],
                       "lo": [round(float(x), 4) for x in (c - s / 2)],
                       "hi": [round(float(x), 4) for x in (c + s / 2)],
                       "sz": [round(float(x), 4) for x in s],
                       "h": round(float(e.heading), 5),
                       "struct": bool(is_structure(e.label))})
        try:
            with src.open("map.ply") as fh:
                pts = read_ply_xyz(fh)
            gt_pts = voxel_thin(pts, gt_voxel, max_gt)
            gt_cloud = blob.add(gt_pts)
            print(f"  {scene}: gt cloud {len(pts)} -> {gt_cloud[1]} points",
                  flush=True)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            print(f"{scene}: no usable map.ply ({exc})", file=sys.stderr)
    else:
        roots = ", ".join(str(r) for r in gt_roots)
        print(f"{scene}: no ground truth under {roots}", file=sys.stderr)
        if gt_only:
            return False  # nothing to export at all

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{scene}.bin").write_bytes(blob.tobytes())
    names = [None] * len(label_ids)
    for k, v in label_ids.items():
        names[v] = k
    # A ground-truth-only export has no lidar cloud, so the scene's own map.ply
    # is the only geometry there is to frame a camera on.
    box = acc if len(acc) else gt_pts
    lo = box.min(axis=0) if box is not None and len(box) else np.zeros(3)
    hi = box.max(axis=0) if box is not None and len(box) else np.ones(3)
    manifest = {
        "scene": scene, "bin": f"{scene}.bin", "stride": stride,
        "gt_only": gt_only, "n_points": blob.n, "labels": names,
        # One structure verdict for every label, so the viewer's hide toggle
        # treats a per-frame detection and its fused object identically. Judging
        # them by different rules is exactly the asymmetry that inflated our
        # counting error, and it would be silly to reproduce it on screen.
        "struct_labels": sorted(v for k, v in label_ids.items() if is_structure(k)),
        "bounds": {"min": [float(x) for x in lo], "max": [float(x) for x in hi]},
        "world": world_ref, "gt_world": gt_cloud, "frames": out_frames,
        "objects": objects, "gt": gt,
    }
    (out_dir / f"{scene}.json").write_text(json.dumps(manifest))
    mb = (blob.n * 12) / 1e6
    tag = "  gt only" if gt_only else ""
    print(f"{scene:16s} {len(out_frames):4d} frames  {len(objects):4d} objects  "
          f"{len(gt):4d} gt  {blob.n:8d} pts ({mb:.1f} MB){tag}", flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="all",
                    help="one scene, or `all` for the official fifteen")
    ap.add_argument("--frames-root", default="frames")
    ap.add_argument("--gt-root", nargs="+", default=[str(p) for p in DEFAULT_GT_ROOTS],
                    help="directories holding <scene>.zip or an unpacked <scene>/")
    ap.add_argument("--out", default="viz/data")
    ap.add_argument("--stride", type=int, default=2,
                    help="keep every Nth frame; fusion still sees all of them")
    ap.add_argument("--max-scan", type=int, default=6000,
                    help="points kept per frame's raw scan")
    ap.add_argument("--max-det", type=int, default=1500,
                    help="points kept per lifted detection")
    ap.add_argument("--world-voxel", type=float, default=0.05)
    ap.add_argument("--max-world", type=int, default=400_000)
    ap.add_argument("--gt-voxel", type=float, default=0.05,
                    help="voxel size for the ground-truth scene cloud")
    ap.add_argument("--max-gt", type=int, default=500_000)
    ap.add_argument("--force", action="store_true",
                    help="let a ground-truth-only export replace a full one")
    args = ap.parse_args()

    scenes = OFFICIAL_SCENES if args.scene == "all" else (args.scene,)
    gt_roots = [Path(p) for p in args.gt_root]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = []
    for s in scenes:
        if not export_scene(s, Path(args.frames_root), gt_roots, out_dir,
                            stride=args.stride, max_scan=args.max_scan,
                            max_det=args.max_det, world_voxel=args.world_voxel,
                            max_world=args.max_world, gt_voxel=args.gt_voxel,
                            max_gt=args.max_gt, force=args.force):
            continue
        ok.append(s)
        # Rewritten per scene, not once at the end: a fifteen-scene export takes
        # minutes and there is no reason to keep the finished ones off the page
        # while the rest run.
        write_index(out_dir)
    print(f"\n{len(ok)} scenes exported; "
          f"{out_dir}/index.json lists everything on disk")
    return 0 if ok else 1


def write_index(out_dir: Path) -> list[str]:
    """List every scene present, not just the ones this run touched.

    Exporting one scene used to rewrite the index to that scene alone, hiding
    the rest of the page's data until someone ran `--scene all` again.
    """
    have = {p.stem for p in out_dir.glob("*.json")} - {"index"}
    known = [s for s in OFFICIAL_SCENES if s in have]
    scenes = known + sorted(have - set(known))
    (out_dir / "index.json").write_text(json.dumps({"scenes": scenes}))
    return scenes


if __name__ == "__main__":
    raise SystemExit(main())
