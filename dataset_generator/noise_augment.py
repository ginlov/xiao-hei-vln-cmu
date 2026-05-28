"""Phase 1.5 — perception noise injection for the scene context.

Purpose: at training time the model sees the **pristine** VLA-3D object list,
but at inference time it gets noisy output from a real LiDAR+RGB perception
module. To close the train-inference gap, perturb the scene context every
epoch with three error modes:

  1. drop          — random non-protected objects vanish (false negatives)
  2. label_swap    — labels swapped with confusable neighbours in same nyu40
  3. bbox jitter   — ±0.1 m centre noise, ±0.05 m size noise

The Q&A's `target` and `anchors` are **protected** — they're never dropped or
relabeled and their bbox jitter is also disabled. This guarantees the question
remains answerable under the perturbed scene.

This is a **library** — the training pipeline imports `perturb_scene()` and
calls it once per (epoch × example). It is NOT pre-computed to disk; we want
fresh noise every epoch so the model can't memorize.

CLI mode (`python noise_augment.py`) runs a demo: load loft, pick a Q&A pair,
print the protected and perturbed object lists side-by-side.
"""

from __future__ import annotations

import copy
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from vla3d_loader import DEFAULT_VLA3D_ROOT, VLAObject, VLAScene, load_vla_scene


@dataclass
class NoiseConfig:
    drop_rate: float = 0.10       # P(drop) per non-protected object
    swap_rate: float = 0.05       # P(label swap) per non-protected object
    bbox_pos_std: float = 0.10    # gaussian noise on (x,y,z) centre [m]
    bbox_size_std: float = 0.05   # gaussian noise on (lx,ly,lz)  [m]
    min_bbox_size: float = 0.02   # clamp so size never goes ≤0


# Default confusable label groups. These were hand-picked from the actual
# VLA-3D label vocabulary to mimic real perception confusions. Any pair within
# a group can swap. Override by passing your own `swap_groups` to perturb_scene.
DEFAULT_SWAP_GROUPS: list[set[str]] = [
    {"pillow", "cushion", "sofa pillow"},
    {"chair", "stool"},
    {"sofa", "couch"},
    {"lamp", "wall lamp", "light"},
    {"cup", "glass", "mug", "coffee cup", "wine glass"},
    {"bowl", "plate"},
    {"table", "desk", "coffee table", "dining table", "tea table", "small table", "round table"},
    {"cabinet", "TV cabinet", "shoe rack"},
    {"picture", "painting", "photo", "canvas"},
    {"bed", "mattress"},
    {"nightstand", "bedside table"},
    {"book", "notebook", "newspaper"},
    {"flower", "potted plant", "potted branch", "potted cactus", "potted bamboo"},
    {"vase", "jar"},
    {"door", "door frame"},
    {"window", "curtain"},
    {"box", "container"},
    {"dvd", "phone", "tv remote"},
]


def build_label_to_group(swap_groups: list[set[str]]) -> dict[str, list[str]]:
    """label → list of confusable alternatives (excluding the label itself)."""
    out: dict[str, list[str]] = {}
    for group in swap_groups:
        for label in group:
            out[label] = [other for other in group if other != label]
    return out


def perturb_object(o: VLAObject, rng: random.Random, cfg: NoiseConfig,
                   label_to_group: dict[str, list[str]]) -> VLAObject | None:
    """Apply noise to a single non-protected object. Returns None if dropped."""
    if rng.random() < cfg.drop_rate:
        return None

    new = copy.copy(o)

    # Label swap
    if rng.random() < cfg.swap_rate:
        alternatives = label_to_group.get(o.raw_label, [])
        if alternatives:
            new.raw_label = rng.choice(alternatives)

    # bbox jitter
    new.x = o.x + rng.gauss(0, cfg.bbox_pos_std)
    new.y = o.y + rng.gauss(0, cfg.bbox_pos_std)
    new.z = o.z + rng.gauss(0, cfg.bbox_pos_std)
    new.lx = max(cfg.min_bbox_size, o.lx + rng.gauss(0, cfg.bbox_size_std))
    new.ly = max(cfg.min_bbox_size, o.ly + rng.gauss(0, cfg.bbox_size_std))
    new.lz = max(cfg.min_bbox_size, o.lz + rng.gauss(0, cfg.bbox_size_std))
    return new


def perturb_scene(
    objects: list[VLAObject],
    protected_ids: set[int] | list[int],
    config: NoiseConfig | None = None,
    swap_groups: list[set[str]] | None = None,
    rng: random.Random | None = None,
) -> list[VLAObject]:
    """Return a perturbed copy of `objects`. Objects in `protected_ids` pass
    through unchanged."""
    if config is None: config = NoiseConfig()
    if swap_groups is None: swap_groups = DEFAULT_SWAP_GROUPS
    if rng is None: rng = random.Random()
    protected = set(protected_ids)
    label_to_group = build_label_to_group(swap_groups)

    out: list[VLAObject] = []
    for o in objects:
        if o.id in protected:
            out.append(o)
            continue
        perturbed = perturb_object(o, rng, config, label_to_group)
        if perturbed is not None:
            out.append(perturbed)
    return out


# ── Convenience: extract protected_ids from a Q&A pair ────────────────────────

def protected_ids_for_pair(pair: dict) -> set[int]:
    """Pull target_id + anchors from one of our JSONL Q&A pairs."""
    out: set[int] = set()
    tgt = pair.get("target")
    if isinstance(tgt, int):
        out.add(tgt)
    # `answer` for object_reference is {"object_id": N, "label": ...}
    ans = pair.get("answer")
    if isinstance(ans, dict) and isinstance(ans.get("object_id"), int):
        out.add(ans["object_id"])
    for a in pair.get("anchors", []) or []:
        if isinstance(a, int):
            out.add(a)
    return out


# ── Demo / sanity-check CLI ───────────────────────────────────────────────────

def _format(o: VLAObject) -> str:
    return f"id={o.id:>3} {o.raw_label:<24} ({o.x:+6.2f},{o.y:+6.2f},{o.z:+6.2f})"


def _demo(scene_name: str = "loft", seed: int = 7) -> None:
    HERE = Path(__file__).parent
    scene_dir = DEFAULT_VLA3D_ROOT / scene_name
    sc = load_vla_scene(scene_dir)

    # Pull one nested pair for this scene
    nested = HERE.parent / "dataset" / "vla3d_nested.jsonl"
    pair = None
    if nested.exists():
        for line in nested.open():
            p = json.loads(line)
            if p["scene"] == scene_name:
                pair = p
                break

    protected = protected_ids_for_pair(pair) if pair else set()
    print(f"Scene: {scene_name}  ({len(sc.objects)} objects)")
    if pair:
        print(f"Demo Q: {pair['question']}")
        print(f"Protected IDs: {sorted(protected)}\n")

    rng = random.Random(seed)
    perturbed = perturb_scene(sc.objects, protected, rng=rng)

    print(f"Before: {len(sc.objects)} objects")
    print(f"After:  {len(perturbed)} objects ({len(sc.objects) - len(perturbed)} dropped)\n")

    # Show 5 perturbed non-protected objects side-by-side with original
    print("Sample perturbations (non-protected):")
    pert_by_id = {o.id: o for o in perturbed}
    shown = 0
    for o in sc.objects:
        if o.id in protected:
            continue
        if o.id not in pert_by_id:
            print(f"  DROPPED  {_format(o)}")
            shown += 1
        else:
            p = pert_by_id[o.id]
            if o.raw_label != p.raw_label:
                print(f"  SWAPPED  {_format(o)}  -->  label='{p.raw_label}'")
                shown += 1
            else:
                dx = p.x - o.x; dy = p.y - o.y; dz = p.z - o.z
                if abs(dx) + abs(dy) + abs(dz) > 0.15:
                    print(f"  JITTERED {_format(o)}  -->  Δ=({dx:+.2f},{dy:+.2f},{dz:+.2f})")
                    shown += 1
        if shown >= 8:
            break

    print(f"\nProtected (should be unchanged):")
    for pid in sorted(protected):
        o = sc.by_id[pid]
        p = pert_by_id.get(pid)
        if p is None:
            print(f"  ⚠ protected id {pid} got dropped — bug!")
        else:
            same = (p.x == o.x and p.y == o.y and p.raw_label == o.raw_label)
            print(f"  {'OK' if same else '⚠ BUG'}  {_format(o)}")


if __name__ == "__main__":
    import sys
    scene = sys.argv[1] if len(sys.argv) > 1 else "loft"
    _demo(scene)
