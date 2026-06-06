"""Parse traversable-area PLY files and object lists from scene zips."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

from xiao_hei_vln.eval_sampler.object_list import ObjectEntry, parse_object_list


def parse_traversable_ply(path: Path) -> np.ndarray:
    """Read an ASCII PLY file and return an (N, 2) array of x-y points."""
    pts: list[list[float]] = []
    with open(path) as f:
        for line in f:
            if line.strip() == "end_header":
                break
        for line in f:
            vals = line.strip().split()
            if len(vals) >= 2:
                pts.append([float(vals[0]), float(vals[1])])
    return np.array(pts, dtype=np.float64)


def parse_traversable_ply_from_zip(zip_path: Path, *, scene_name: str) -> np.ndarray:
    """Extract and parse ``traversable_area.ply`` from a scene zip."""
    ply_member = f"{scene_name}/traversable_area.ply"
    with zipfile.ZipFile(zip_path) as zf, zf.open(ply_member) as f:
        pts: list[list[float]] = []
        for raw in f:
            line = raw.decode().strip()
            if line == "end_header":
                break
        for raw in f:
            vals = raw.decode().strip().split()
            if len(vals) >= 2:
                pts.append([float(vals[0]), float(vals[1])])
    return np.array(pts, dtype=np.float64)


def read_objects_from_zip(
    zip_path: Path, *, scene_name: str
) -> dict[int, ObjectEntry]:
    """Read ``object_list.txt`` from a scene zip via the eval_sampler parser."""
    obj_member = f"{scene_name}/object_list.txt"
    with zipfile.ZipFile(zip_path) as zf:
        try:
            with zf.open(obj_member) as f:
                lines = [raw.decode().strip() for raw in f]
        except KeyError:
            return {}
    return parse_object_list(lines)
