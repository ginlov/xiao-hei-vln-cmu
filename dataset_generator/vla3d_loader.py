"""Load VLA-3D Unity subset into a unified VLAScene representation.

VLA-3D ships 7 files per scene; we use 3:
- <scene>_object_result.csv         : objects w/ bbox + colors + nyu40 + region_id
- <scene>_scene_graph.json          : precomputed relations (on/above/closest/etc)
- <scene>_referential_statements.json: ~120K precomputed ref sentences

Object IDs are 1:1 aligned with the official `object_list.txt`, so a
VLAObject.id can be used as ground-truth answer for the official questions.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

# Where the VLA-3D Unity subset lives. Override with the VLA3D_ROOT env var
# if you keep the data elsewhere. The data is not committed to this repo —
# see dataset_generator/README.md for the download instructions.
DEFAULT_VLA3D_ROOT = Path(
    os.environ.get("VLA3D_ROOT", Path(__file__).parent / "vla-3d" / "Unity")
)

# 8 spatial relations available in scene_graph.json. We expose them all.
RELATION_KEYS = (
    "on", "above", "below", "near", "between", "beside",
    "in", "hanging_on",
    "closest", "second_closest", "third_closest",
    "farthest", "second_farthest", "third_farthest",
)


@dataclass
class VLAObject:
    id: int
    region_id: int                # -1 means unassigned to any room
    raw_label: str                # e.g. "potted plant"
    nyu40_label: str              # NYU40 coarse category
    nyu40_id: int
    # bbox center + axis-aligned extents
    x: float; y: float; z: float
    lx: float; ly: float; lz: float
    heading: float                # bbox orientation about Z
    # top-3 dominant colors (named); "N/A" if absent
    colors: tuple[str, str, str]
    color_percentages: tuple[float, float, float]


@dataclass
class VLARegion:
    region_id: int
    region_name: str              # e.g. "courtyard_1", "bedroom_2"


@dataclass
class VLAScene:
    name: str
    objects: list[VLAObject]
    by_id: dict[int, VLAObject] = field(default_factory=dict)
    regions: dict[int, VLARegion] = field(default_factory=dict)

    # relationships[region_id][rel_name][src_id] -> list[tgt_id]
    # rel_name in RELATION_KEYS
    relationships: dict[int, dict[str, dict[int, list[int]]]] = field(default_factory=dict)

    # Pre-rendered referential statements:
    # ref_statements[region_id] -> list of dicts with keys:
    #   statement, target_id, target_class, distractor_ids,
    #   relation, relation_type, anchors (list of dicts), color_used, size_used
    ref_statements: dict[int, list[dict]] = field(default_factory=dict)

    # Convenience indexes
    def __post_init__(self) -> None:
        if not self.by_id:
            self.by_id = {o.id: o for o in self.objects}

    def label_count(self, label: str, region_id: int | None = None) -> int:
        if region_id is None:
            return sum(1 for o in self.objects if o.raw_label == label)
        return sum(1 for o in self.objects if o.raw_label == label and o.region_id == region_id)

    def objects_with_relation(self, region_id: int, rel: str, src_id: int) -> list[VLAObject]:
        """Return target objects in this relation with src_id, e.g. all books 'on' the table."""
        if region_id not in self.relationships:
            return []
        rel_map = self.relationships[region_id].get(rel, {})
        return [self.by_id[t] for t in rel_map.get(src_id, []) if t in self.by_id]


def _parse_float_or_default(s: str, default: float = 0.0) -> float:
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


def _load_objects(csv_path: Path) -> list[VLAObject]:
    objects: list[VLAObject] = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            colors = (
                row.get("object_color_scheme1", "N/A") or "N/A",
                row.get("object_color_scheme2", "N/A") or "N/A",
                row.get("object_color_scheme3", "N/A") or "N/A",
            )
            pcts = (
                _parse_float_or_default(row.get("object_color_scheme_percentage1", "0")),
                _parse_float_or_default(row.get("object_color_scheme_percentage2", "0")),
                _parse_float_or_default(row.get("object_color_scheme_percentage3", "0")),
            )
            objects.append(VLAObject(
                id=int(row["object_id"]),
                region_id=int(row["region_id"]),
                raw_label=row["raw_label"],
                nyu40_label=row.get("nyu40_label", ""),
                nyu40_id=int(row.get("nyu40_id", "0") or 0),
                x=float(row["object_bbox_cx"]),
                y=float(row["object_bbox_cy"]),
                z=float(row["object_bbox_cz"]),
                lx=float(row["object_bbox_xlength"]),
                ly=float(row["object_bbox_ylength"]),
                lz=float(row["object_bbox_zlength"]),
                heading=float(row["object_bbox_heading"]),
                colors=colors,
                color_percentages=pcts,
            ))
    return objects


def _load_regions(csv_path: Path) -> dict[int, VLARegion]:
    """region_result.csv has columns: region_id, region_label, region_bbox*..."""
    regions: dict[int, VLARegion] = {}
    if not csv_path.exists():
        return regions
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                rid = int(row.get("region_id", ""))
            except ValueError:
                continue
            regions[rid] = VLARegion(
                region_id=rid,
                region_name=row.get("region_label", "") or row.get("region_name", "") or "",
            )
    return regions


def _load_relationships(sg_path: Path) -> dict[int, dict[str, dict[int, list[int]]]]:
    """scene_graph.json -> region_id (int) -> rel_name -> src_id -> [tgt_ids]."""
    out: dict[int, dict[str, dict[int, list[int]]]] = {}
    if not sg_path.exists():
        return out
    sg = json.load(sg_path.open())
    for r_id_str, region in sg.get("regions", {}).items():
        rid = int(r_id_str)
        rel_dict: dict[str, dict[int, list[int]]] = {}
        for rel_name, rel_map in (region.get("relationships", {}) or {}).items():
            cleaned: dict[int, list[int]] = {}
            for src_str, tgts in rel_map.items():
                try:
                    src_id = int(src_str)
                except ValueError:
                    continue
                if not isinstance(tgts, list):
                    continue
                # Most relations: tgts is list[str|int] of object ids.
                # Ternary relations (e.g. 'between'): tgts is list[list[str|int]]
                #   of (a, b) anchor pairs.
                cleaned_list: list = []
                for t in tgts:
                    if isinstance(t, list):
                        inner = [int(x) for x in t if str(x).lstrip("-").isdigit()]
                        if len(inner) >= 2:
                            cleaned_list.append(inner)
                    elif str(t).lstrip("-").isdigit():
                        cleaned_list.append(int(t))
                cleaned[src_id] = cleaned_list
            rel_dict[rel_name] = cleaned
        out[rid] = rel_dict
    return out


def _load_ref_statements(rs_path: Path) -> dict[int, list[dict]]:
    """referential_statements.json -> region_id -> list of normalized stmt dicts."""
    out: dict[int, list[dict]] = defaultdict(list)
    if not rs_path.exists():
        return dict(out)
    rs = json.load(rs_path.open())
    for r_id_str, region_stmts in rs.get("regions", {}).items():
        try:
            rid = int(r_id_str)
        except ValueError:
            continue
        for stmt_text, val_list in region_stmts.items():
            if not isinstance(val_list, list):
                continue  # skip 'region' metadata key
            for v in val_list:
                anchors = []
                for a in v.get("anchors", {}).values():
                    anchors.append({
                        "id": int(a["index"]),
                        "class": a["class"],
                        "color_used": (a.get("color_used") or "").strip(),
                        "size_used": (a.get("size_used") or "").strip(),
                    })
                out[rid].append({
                    "statement": stmt_text,
                    "target_id": int(v["target_index"]),
                    "target_class": v["target_class"],
                    "target_color_used": (v.get("target_color_used") or "").strip(),
                    "target_size_used": (v.get("target_size_used") or "").strip(),
                    "distractor_ids": [int(d) for d in v.get("distractor_ids", [])],
                    "relation": v["relation"],
                    "relation_type": v["relation_type"],
                    "anchors": anchors,
                })
    return dict(out)


def load_vla_scene(scene_dir: Path | str) -> VLAScene:
    scene_dir = Path(scene_dir)
    name = scene_dir.name
    objs = _load_objects(scene_dir / f"{name}_object_result.csv")
    regions = _load_regions(scene_dir / f"{name}_region_result.csv")
    rels = _load_relationships(scene_dir / f"{name}_scene_graph.json")
    refs = _load_ref_statements(scene_dir / f"{name}_referential_statements.json")
    return VLAScene(
        name=name,
        objects=objs,
        regions=regions,
        relationships=rels,
        ref_statements=refs,
    )


def render_object(o: VLAObject) -> str:
    """One-line text record matching the runtime `object_list.txt` schema:

        id  cx cy cz  lx ly lz  heading  "label"

    The label is double-quoted so multi-word labels (e.g. ``potted plant``)
    round-trip through ``xiao_hei_vln.dummy.fixtures.read_object_list``,
    which requires a quoted label.
    """
    return (
        f"{o.id} {o.x:.2f} {o.y:.2f} {o.z:.2f} "
        f"{o.lx:.2f} {o.ly:.2f} {o.lz:.2f} {o.heading:.2f} \"{o.raw_label}\""
    )


def render_object_list(sc: VLAScene, region_ids: set[int] | None = None) -> list[str]:
    """Render every object in `sc` (or only those in `region_ids`).
    Drops objects with region_id < 0 (unassigned)."""
    out: list[str] = []
    for o in sc.objects:
        if o.region_id < 0:
            continue
        if region_ids is not None and o.region_id not in region_ids:
            continue
        out.append(render_object(o))
    return out


def load_all_vla_scenes(root: Path | str = DEFAULT_VLA3D_ROOT) -> dict[str, VLAScene]:
    root = Path(root)
    out: dict[str, VLAScene] = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        try:
            out[d.name] = load_vla_scene(d)
        except FileNotFoundError as e:
            print(f"[skip] {d.name}: {e}")
    return out


if __name__ == "__main__":
    import sys
    from collections import Counter

    scenes = load_all_vla_scenes()
    print(f"Loaded {len(scenes)} VLA-3D scenes\n")

    # Per-scene summary
    print(f"{'Scene':<25} {'#objs':>6} {'#regions':>9} {'#ref':>8} {'#on-pairs':>10}")
    print("-" * 70)
    tot_refs = 0
    tot_on = 0
    for name, sc in scenes.items():
        n_ref = sum(len(v) for v in sc.ref_statements.values())
        n_on = sum(
            len(tgt)
            for r_id, rel_dict in sc.relationships.items()
            for src, tgt in rel_dict.get("on", {}).items()
        )
        tot_refs += n_ref
        tot_on += n_on
        print(f"{name:<25} {len(sc.objects):>6} {len(sc.regions):>9} {n_ref:>8} {n_on:>10}")
    print("-" * 70)
    print(f"{'TOTAL':<25} {'':>6} {'':>9} {tot_refs:>8} {tot_on:>10}")

    if len(sys.argv) > 1 and sys.argv[1] in scenes:
        sc = scenes[sys.argv[1]]
        print(f"\n--- {sc.name} sample ---")
        for o in sc.objects[:5]:
            print(f"  id={o.id:>3} {o.raw_label:<20} region={o.region_id:>2}  "
                  f"colors={o.colors}  nyu40={o.nyu40_label}")
        print(f"\n  3 sample ref statements (region 0):")
        for r in sc.ref_statements.get(0, [])[:3]:
            print(f"    \"{r['statement']}\"  -> target={r['target_id']} ({r['target_class']}), "
                  f"rel={r['relation']}, distractors={len(r['distractor_ids'])}")
