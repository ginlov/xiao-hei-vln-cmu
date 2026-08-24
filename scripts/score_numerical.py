#!/usr/bin/env python3
"""The answer key for the numerical questions, computed rather than typed.

The challenge ships no answers -- only the questions and, for the fifteen
development scenes, the VLA-3D annotation those scenes were built from. So this
derives a key from the annotation, and is a **proxy** exactly the way
`score_if.py` is one, labelled as such everywhere it is quoted.

The split is deliberate and is the whole design:

  * **the reading of the English is declared** -- which noun is the target,
    which is the anchor, which relation joins them, which colour word applies.
    A human has to do that, because there is no key to check it against, and
    pretending a regex did it would hide the judgement rather than remove it.
  * **the count is computed** -- from boxes, by geometry, so no integer in this
    file was typed by a person and none can drift from the annotation.

Every question carries a `reading`, and the report groups by it:

  solid   the annotation answers it and the English admits one reading
  shaky   the annotation answers it under a judgement call that is written down
  open    two readings of the sentence give different integers, and nothing
          here can choose between them

    uv run python scripts/score_numerical.py              # the key
    uv run python scripts/score_numerical.py --why loft   # one question, shown
    uv run python scripts/score_numerical.py --json artifacts/numerical_key.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
CHALLENGE = Path.home() / "Workspace/vln-challenge/CMU-VLN-Challenge-2026"
VLA3D = Path(os.environ.get(
    "XIAO_HEI_VLA3D", Path.home() / "Workspace/vln-challenge/vla-3d/Unity"))

# How much two footprints may miss each other and still count as touching. The
# boxes are axis-aligned around meshes that are not, so a pillow pushed into a
# sofa back overhangs its own footprint by a few centimetres.
PAD_M = 0.10
# "near", for the one question that uses the word. livingroom_1 asks for chairs
# near a table: at any radius from 1.0 m to 2.0 m the answer is the same 8, so
# the vagueness happens not to bite here -- but the threshold is still ours and
# the question is marked `shaky` because of it.
NEAR_M = 1.5
# A named colour dark enough that a person would call it black. VLA-3D snaps
# each object to the nearest CSS colour name, and its `gray` family spans
# darkslategray (47,79,79) through darkgray (169,169,169) -- a range covering
# both "black-ish" and "plainly grey". Rec. 601 luma of darkslategray is 69 and
# of slategray is 125, so the cut sits between them.
BLACK_LUMA = 90
# Colour names that answer to a question's colour word. VLA-3D has 14 schemes;
# these are the only two words the released questions use.
COLOUR_FAMILY = {
    "red": {"red", "maroon"},          # maroon is firebrick (178,34,34)
    "black": {"black"},                # plus the dark end of `gray`, see above
}

# ---------------------------------------------------------------------------
# The declared readings. One entry per scene; the integers come from the code
# below, never from here.
#
#   target        the noun being counted
#   rel           how it must sit relative to the anchor
#   anchor        the noun it sits on/under/near, or None for a bare count
#   pick          which anchor instance the sentence means, when there are many
#   colour        the colour word, if the sentence has one
#   count         "targets" (the usual) or "anchors" (count anchors that have one)
#   reading       solid | shaky | open
# ---------------------------------------------------------------------------
SPECS: dict[str, dict] = {
    "hotel_room_1": dict(
        q="How many pillows are on the bed?",
        target="pillow", rel="on", anchor="bed", reading="solid"),
    "hotel_room_2": dict(
        q="How many pictures are above the bed?",
        target="picture", rel="above", anchor="bed", reading="solid"),
    "japanese_room": dict(
        q="How many calligraphy paintings are above the display ledge?",
        target="calligraphy painting", rel="above", anchor="display ledge",
        reading="solid"),
    "livingroom_2": dict(
        q="How many cups are on the coffee table?",
        target="cup", rel="on", anchor="coffee table", reading="solid"),
    "livingroom_3": dict(
        q="How many photos are on the TV cabinet?",
        target="photo", rel="on", anchor="tv cabinet", reading="solid"),
    "studio": dict(
        q="How many framed records are above the couch?",
        target="framed record", rel="above", anchor="couch", reading="solid"),
    "home_building_1": dict(
        q="How many pillows are on the sofa under the pictures?",
        target="pillow", rel="on", anchor="sofa",
        pick=("under", "picture"), reading="solid"),
    "office_1": dict(
        q="How many computer monitors are on the table closest to the map "
          "wall decal?",
        target="computer monitor", rel="on", anchor="table",
        pick=("closest_to", "map wall decal"), reading="solid"),
    "arabic_room": dict(
        q="How many sofas are below a window?",
        target="sofa", rel="below", anchor="window", reading="solid"),
    "chinese_room": dict(
        q="Count the number of chairs with pillows on them.",
        target="pillow", rel="on", anchor="chair", count="anchors",
        reading="solid"),
    "home_building_2": dict(
        q="How many red pillows are on the sofa?",
        target="pillow", rel="on", anchor="sofa", colour="red",
        reading="solid"),
    "livingroom_1": dict(
        q="How many chairs are near the table with a vase on it?",
        target="chair", rel="near", anchor="table", pick=("with", "vase"),
        reading="shaky",
        note="'near' is our threshold, not theirs; stable over 1.0-2.0 m"),
    "loft": dict(
        q="How many black pillows are on the sofa?",
        target="pillow", rel="on", anchor="sofa", colour="black",
        reading="shaky",
        note="no pillow is labelled black; the two darkslategray (47,79,79) "
             "ones are dark enough that a person would say black, the seven "
             "slategray (112,128,144) ones are not. 'the sofa' also picks one "
             "of five, and this counts across all of them"),
    "livingroom_4": dict(
        q="How many pillows are on a sofa?",
        target="pillow", rel="on", anchor="sofa", reading="open",
        note="'a sofa' -- the total across sofas, or the most on any one"),
    "office_2": dict(
        q="How many potted plants are on a table?",
        target="potted plant", rel="on", anchor="table", reading="open",
        note="'a table' -- the total across tables, or the most on any one"),
}


def load(scene: str) -> list[dict]:
    """Every annotated object in a scene: label, box, dominant colours.

    Read from VLA-3D rather than from `viz/data`, which is derived from it and
    drops the colour columns. Same frame and same objects -- checked object for
    object on `loft`, where the two agree to 1e-4 m.
    """
    f = VLA3D / scene / f"{scene}_object_result.csv"
    if not f.is_file():
        return []
    out = []
    for r in csv.DictReader(f.open()):
        def num(k: str, d: float = 0.0) -> float:
            try:
                return float(r[k])
            except (TypeError, ValueError, KeyError):
                return d
        c = np.array([num("object_bbox_cx"), num("object_bbox_cy"),
                      num("object_bbox_cz")])
        s = np.array([num("object_bbox_xlength"), num("object_bbox_ylength"),
                      num("object_bbox_zlength")])
        cols = []
        for i in (1, 2, 3):
            name = (r.get(f"object_color_scheme{i}") or "").strip()
            if not name or name == "_":
                continue
            rgb = (num(f"object_color_r{i}"), num(f"object_color_g{i}"),
                   num(f"object_color_b{i}"))
            cols.append({"name": name,
                         "pct": num(f"object_color_scheme_percentage{i}"),
                         "rgb": rgb,
                         "luma": 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]})
        out.append({"id": r["object_id"], "label": r["raw_label"].strip().lower(),
                    "c": c, "lo": c - s / 2, "hi": c + s / 2, "sz": s,
                    "colours": cols})
    return out


def of(objs: list[dict], label: str) -> list[dict]:
    return [o for o in objs if o["label"] == label]


def overlaps(a: dict, b: dict, pad: float = PAD_M) -> bool:
    """Do the two footprints touch, seen from above."""
    return bool(a["lo"][0] <= b["hi"][0] + pad and a["hi"][0] >= b["lo"][0] - pad
                and a["lo"][1] <= b["hi"][1] + pad and a["hi"][1] >= b["lo"][1] - pad)


def holds(rel: str, t: dict, a: dict, pad: float = PAD_M) -> bool:
    """Does the relation the sentence states hold between target and anchor?

    `pad` is how far the two footprints may miss each other and still count.
    It is the one free parameter here and `--sens` sweeps it, because a
    question whose integer moves with it is a question we do not actually know
    the answer to.
    """
    if rel == "on":
        # Sits within the anchor's footprint and no lower than its base. Not
        # "above its top": a pillow pushed into a sofa is below the backrest,
        # and a cup on a table is below nothing at all.
        return overlaps(t, a, pad) and t["hi"][2] > a["lo"][2]
    if rel == "above":
        # Hangs over it and clears it vertically. The 0.30 m slack lets a
        # picture whose frame dips behind a headboard still count.
        return overlaps(t, a, pad + 0.7) and t["lo"][2] >= a["hi"][2] - 0.30
    if rel == "below":
        # The anchor must genuinely be overhead, not merely taller: a window
        # on the far wall is above every sofa in the room by height alone.
        return overlaps(t, a, pad + 0.5) and a["lo"][2] >= t["hi"][2]
    if rel == "near":
        return float(np.linalg.norm(t["c"][:2] - a["c"][:2])) <= (
            NEAR_M + max(a["sz"][:2]) / 2)
    raise ValueError(f"unknown relation {rel!r}")


def is_colour(o: dict, want: str) -> bool:
    """Would a person call this object that colour?

    The dominant scheme only. A pillow whose *second* colour is maroon is a
    grey pillow with a pattern, and the question is about the pillow.
    """
    if not o["colours"]:
        return False
    top = o["colours"][0]
    if top["name"] in COLOUR_FAMILY.get(want, set()):
        return True
    # `gray` covers everything from darkslategray to darkgray; only its dark
    # end answers to "black". See BLACK_LUMA.
    return bool(want == "black" and top["name"] == "gray"
                and top["luma"] <= BLACK_LUMA)


def pick_anchors(spec: dict, objs: list[dict]) -> list[dict]:
    """The anchor instances the sentence's relative clause picks out."""
    anchors = of(objs, spec["anchor"])
    pick = spec.get("pick")
    if not pick or not anchors:
        return anchors
    how, noun = pick
    others = of(objs, noun)
    if not others:
        return anchors
    if how == "under":                       # "the sofa under the pictures"
        return [a for a in anchors
                if any(holds("above", o, a) for o in others)] or anchors
    if how == "with":                        # "the table with a vase on it"
        return [a for a in anchors
                if any(holds("on", o, a) for o in others)] or anchors
    if how == "closest_to":                  # "the table closest to the decal"
        ref = others[0]
        return [min(anchors,
                    key=lambda a: float(np.linalg.norm(a["c"][:2] - ref["c"][:2])))]
    raise ValueError(f"unknown pick {how!r}")


def answer(scene: str, spec: dict, objs: list[dict],
           pad: float = PAD_M) -> dict:
    """The count, with everything needed to argue with it."""
    targets = of(objs, spec["target"])
    anchors = pick_anchors(spec, objs)
    colour = spec.get("colour")

    kept, per = [], {id(a): [] for a in anchors}
    for t in targets:
        if colour and not is_colour(t, colour):
            continue
        hit = [a for a in anchors if holds(spec["rel"], t, a, pad)]
        if hit:
            kept.append(t)
            for a in hit:
                per[id(a)].append(t)

    if spec.get("count") == "anchors":
        n = sum(1 for a in anchors if per[id(a)])
        alt = None
    else:
        n = len(kept)                        # each target counted once
        most = max((len(v) for v in per.values()), default=0)
        alt = most if spec["reading"] == "open" and most != n else None

    return {"scene": scene, "question": spec["q"], "answer": n,
            "alt": alt, "reading": spec["reading"], "note": spec.get("note"),
            "n_targets": len(targets), "n_anchors": len(anchors),
            "ids": [t["id"] for t in kept]}


PADS = (0.0, 0.10, 0.30, 0.60)


def spread(scene: str, spec: dict, objs: list[dict]) -> list[int]:
    """The same question answered at every pad in `PADS`."""
    return [answer(scene, spec, objs, p)["answer"] for p in PADS]


def counted_objects(scene: str, pad: float = PAD_M) -> list[dict]:
    """The annotation objects the key counts, with their boxes.

    What `answer` returns as `ids`, resolved back to geometry, so an audit can
    ask "how many of these could the robot see from there" without re-deriving
    the reading of the sentence.
    """
    spec = SPECS.get(scene)
    objs = load(scene)
    if not spec or not objs:
        return []
    ids = set(answer(scene, spec, objs, pad)["ids"])
    if spec.get("count") == "anchors":
        # The question counts anchors, so those are the objects to look for.
        anchors = pick_anchors(spec, objs)
        return [a for a in anchors
                if any(holds(spec["rel"], t, a, pad)
                       for t in of(objs, spec["target"])
                       if not spec.get("colour") or is_colour(t, spec["colour"]))]
    return [o for o in objs if o["id"] in ids]


def key() -> list[dict]:
    """The whole key, in the challenge's own scene order."""
    out = []
    for scene in sorted(SPECS):
        objs = load(scene)
        if not objs:
            out.append({"scene": scene, "question": SPECS[scene]["q"],
                        "answer": None, "reading": "missing",
                        "note": f"no VLA-3D annotation under {VLA3D}"})
            continue
        a = answer(scene, SPECS[scene], objs)
        a["spread"] = spread(scene, SPECS[scene], objs)
        # A question whose integer moves with a threshold of ours is not one
        # we know the answer to, whatever the spec claimed.
        if len(set(a["spread"])) > 1 and a["reading"] == "solid":
            a["reading"] = "shaky"
            a["note"] = ((a["note"] + "; ") if a["note"] else "") + \
                f"answer moves with the footprint pad: {a['spread']} at {list(PADS)} m"
        out.append(a)
    return out


def check_questions() -> list[str]:
    """Every declared question must be the one the challenge actually asks."""
    f = CHALLENGE / "questions/questions.json"
    if not f.is_file():
        return []
    bad = []
    for e in json.loads(f.read_text()):
        want = " ".join(e["questions"]["numerical"][0].split())
        got = " ".join(SPECS.get(e["scene"], {}).get("q", "").split())
        if e["scene"] not in SPECS:
            bad.append(f"{e['scene']}: no spec")
        elif got != want:
            bad.append(f"{e['scene']}: spec says {got!r}, challenge asks {want!r}")
    return bad


def explain(scene: str) -> None:
    """Show the working for one question."""
    spec = SPECS[scene]
    objs = load(scene)
    if not objs:
        print(f"no annotation for {scene} under {VLA3D}")
        return
    anchors = pick_anchors(spec, objs)
    print(f"{scene}: {spec['q']}\n")
    print(f"  anchors: {len(anchors)} x {spec['anchor']}"
          + (f"  (picked by {spec['pick'][0]} {spec['pick'][1]!r} "
             f"from {len(of(objs, spec['anchor']))})" if spec.get("pick") else ""))
    for t in of(objs, spec["target"]):
        hit = [a for a in anchors if holds(spec["rel"], t, a, pad)]
        col = (f"  {t['colours'][0]['name']} "
               f"{t['colours'][0]['pct'] * 100:.0f}% "
               f"rgb{tuple(int(v) for v in t['colours'][0]['rgb'])}"
               if t["colours"] else "  (no colour)")
        ok = "yes" if hit else " no"
        if spec.get("colour"):
            ok += "  colour " + ("yes" if is_colour(t, spec["colour"]) else " no")
        print(f"  {spec['target']:>22} {t['id']:>4}  {spec['rel']} anchor: {ok}{col}")
    a = answer(scene, spec, objs)
    print(f"\n  -> {a['answer']}" + (f"   (other reading: {a['alt']})" if a["alt"] else ""))
    if a["note"]:
        print(f"     {a['note']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--why", metavar="SCENE", help="show the working for one scene")
    ap.add_argument("--json", metavar="PATH", help="write the key as JSON")
    args = ap.parse_args()

    for problem in check_questions():
        print(f"!! {problem}", file=sys.stderr)

    if args.why:
        if args.why not in SPECS:
            print(f"no spec for {args.why!r}; have {', '.join(sorted(SPECS))}")
            return 1
        explain(args.why)
        return 0

    rows = key()
    print(f"{'scene':18s} {'ans':>4} {'alt':>4}  {'reading':8s} {'pad sweep':16s} question")
    for r in rows:
        alt = "" if r.get("alt") is None else str(r["alt"])
        ans = "?" if r["answer"] is None else str(r["answer"])
        sw = r.get("spread")
        sw = "" if not sw else ("stable" if len(set(sw)) == 1
                                else " ".join(str(x) for x in sw))
        print(f"{r['scene']:18s} {ans:>4} {alt:>4}  {r['reading']:8s} {sw:16s} "
              f"{r['question'][:52]}")
    by: dict[str, int] = {}
    for r in rows:
        by[r["reading"]] = by.get(r["reading"], 0) + 1
    print("\n" + "   ".join(f"{k} {v}" for k, v in sorted(by.items())))

    ok = [r["answer"] for r in rows if r["reading"] == "solid"]
    if ok:
        mode = max(set(ok), key=ok.count)
        n = sum(1 for r in rows if r["answer"] == mode)
        print(f"\nbest constant guess: {mode} -> {n}/{len(rows)} "
              f"({100 * n / len(rows):.0f}%). Anything built has to beat that.")

    if args.json:
        p = REPO / args.json
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rows, indent=1, default=str))
        print(f"written: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
