#!/usr/bin/env python3
"""The answer key for the object-reference questions, computed rather than typed.

Same split as `score_numerical.py`, for the same reason:

  * **the reading of the English is declared** -- which noun is the target,
    which relation qualifies it, which object the superlative is measured
    against. A person has to do that; a regex pretending to would hide the
    judgement rather than remove it.
  * **the object is computed** -- by geometry over the VLA-3D boxes, so no
    object id in this file was typed and none can drift from the annotation.

Unlike the numerical key this one has a **built-in check**. The challenge says
of object reference: *"there exists only one correct answer in the scene (the
referred object is unique)"*. So a spec that resolves to zero candidates, or to
more than one, is not an ambiguous question -- it is a misreading, and the
report says so. `--check` exits non-zero when any spec fails to single out
exactly one object, which makes this file self-validating in a way `score_if`
and `score_numerical` are not.

Scoring, from the challenge README:

    IoU >= 0.50  ->  2 points
    IoU >= 0.25  ->  1 point
    otherwise    ->  0

    uv run python scripts/score_reference.py                 # the key
    uv run python scripts/score_reference.py --check         # CI gate
    uv run python scripts/score_reference.py --why studio    # one scene, shown
    uv run python scripts/score_reference.py --sens          # does the reading hold up
    uv run python scripts/score_reference.py --json artifacts/reference_key.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from score_numerical import CHALLENGE, load, of  # noqa: E402

# --------------------------------------------------------------------------
# The two free parameters. `--sens` sweeps both, because a question whose
# answer moves with either is a question we do not actually know.

# How "closest to" measures. `surface` is the gap between the two boxes (0 when
# they touch); `centre` is centre-to-centre. Surface is the better reading of
# the English -- a wide cabinet's centre can be metres from a cup its edge
# nearly touches -- and is the default. `--sens` reports both.
DIST = "surface"

# Footprint slack for the qualifying relations, handed to `score_numerical.holds`.
PAD_M = 0.10

# How much of an object's points its dominant colour scheme must cover before
# the name is worth believing. `loft`'s chairs sit at 0.01 -- see
# `score_numerical.is_colour`.
COLOUR_MIN_PCT = 0.5


def _gap(a: dict, b: dict) -> float:
    """Shortest distance between two axis-aligned boxes; 0 when they overlap."""
    d = np.maximum(0.0, np.maximum(a["lo"] - b["hi"], b["lo"] - a["hi"]))
    return float(np.linalg.norm(d))


def dist(a: dict, b: dict, how: str = DIST) -> float:
    return _gap(a, b) if how == "surface" else float(np.linalg.norm(a["c"] - b["c"]))


# --------------------------------------------------------------------------
# The term language. A term names a set of objects:
#
#   {"label": "pillow"}                              every pillow
#   {"label": "pillow", "colour": "red"}             ... that reads as red
#   {"label": "book", "rel": ("on", {...})}          ... standing in a relation
#   {"label": "table", "pick": ("closest_to", {...})}  ... narrowed to one
#
# `rel` and `pick` nest, which is what the 17 questions carrying two relation
# words need: *"the speaker on the TV cabinet closest to the potted plant on
# the TV cabinet"* is a term with a rel whose pick has a term with a rel.


def resolve(term: dict, objs: list[dict], *, how: str = DIST,
            pad: float = PAD_M) -> list[dict]:
    """Every object the term names, narrowed by its own `pick` if it has one."""
    from score_numerical import holds, is_colour

    # `label` may be a list. The questions name objects in English and the
    # annotation names them in its own vocabulary, and the two do not always
    # agree: *"the lamp on the nightstand"* in `home_building_2` is labelled
    # `desk light`, and that scene's *"TV cabinet"* is a `tv stand`. Listing
    # the alternatives is the declared reading; guessing with a substring
    # match would quietly also match `tv remote` and `kitchen cabinet`.
    want = term["label"]
    labels = [want] if isinstance(want, str) else list(want)
    cands = [o for lab in labels for o in of(objs, lab)]
    if term.get("colour"):
        cands = [o for o in cands
                 if is_colour(o, term["colour"], min_pct=COLOUR_MIN_PCT)]
    if term.get("rel"):
        rel, anchor_term = term["rel"]
        anchors = resolve(anchor_term, objs, how=how, pad=pad)
        if rel.endswith("_inv"):
            # *"the cabinet with a TV on it"* -- the same relation with the
            # arguments swapped. Written as a suffix rather than a separate
            # relation so `holds` stays the single definition of "on".
            base = rel[:-4]
            cands = [o for o in cands
                     if any(holds(base, a, o, pad) for a in anchors)]
        else:
            cands = [o for o in cands
                     if any(holds(rel, o, a, pad) for a in anchors)]
    if term.get("pick"):
        one = pick(cands, term["pick"], objs, how=how, pad=pad)
        cands = [one] if one is not None else []
    return cands


def pick(cands: list[dict], spec, objs: list[dict], *, how: str = DIST,
         pad: float = PAD_M) -> dict | None:
    """The one object a superlative or a betweenness singles out."""
    if not cands:
        return None
    kind = spec[0]
    if kind == "unique":
        return cands[0] if len(cands) == 1 else None
    if kind in ("closest_to", "farthest_from", "near"):
        refs = resolve(spec[1], objs, how=how, pad=pad)
        if not refs:
            return None
        keyed = [(min(dist(c, r, how) for r in refs), c) for c in cands]
        keyed.sort(key=lambda t: t[0])
        return keyed[-1][1] if kind == "farthest_from" else keyed[0][1]
    if kind == "between":
        # Betweenness residual: d(a) + d(b) - d(a,b) is zero exactly on the
        # segment joining them and grows either side of it. Measured
        # centre-to-centre whatever `how` says, because a residual built from
        # surface gaps is not zero on the segment and the comparison stops
        # meaning anything.
        A = resolve(spec[1], objs, how=how, pad=pad)
        B = resolve(spec[2], objs, how=how, pad=pad)
        if not A or not B:
            return None
        best, chosen = float("inf"), None
        for c in cands:
            for a in A:
                for b in B:
                    r = (float(np.linalg.norm(c["c"] - a["c"]))
                         + float(np.linalg.norm(c["c"] - b["c"]))
                         - float(np.linalg.norm(a["c"] - b["c"])))
                    if r < best:
                        best, chosen = r, c
        return chosen
    raise ValueError(f"unknown pick {kind!r}")


# --------------------------------------------------------------------------
# The 30 released questions. `reading` is the declared confidence:
#
#   solid   one reading of the English, and the geometry singles out one object
#   shaky   singles out one object under a judgement call written in `note`
#   open    two readings give different objects; `alt` carries the other one
#
# Question strings are checked against the challenge's own JSON by
# `check_questions`, so a typo here cannot silently answer a question nobody
# asked.

S = SPECS = [
    dict(scene="arabic_room", i=0,
         q="Find the pillow closest to the book on the stool.",
         target={"label": "pillow",
                 "pick": ("closest_to",
                          {"label": "book", "rel": ("on", {"label": "stool"})})},
         reading="solid"),
    dict(scene="arabic_room", i=1,
         q="Find the wall lamp that is between a door frame and a window.",
         target={"label": "wall lamp",
                 "pick": ("between", {"label": "door frame"}, {"label": "window"})},
         reading="solid"),

    dict(scene="chinese_room", i=0,
         q="Find the bowl on the table closest to the folding screen.",
         # Both attachments are grammatical, and the challenge's uniqueness
         # guarantee picks between them: reading it as *the table nearest the
         # screen, then the bowl on it* leaves BOTH bowls, which cannot be
         # right. So 'closest to' modifies the bowl. That reading is primary
         # here and the refuted one is kept as `alt` so the report shows the
         # disagreement rather than hiding the judgement.
         target={"label": "bowl", "rel": ("on", {"label": "table"}),
                 "pick": ("closest_to", {"label": "folding screen"})},
         alt={"label": "bowl",
              "rel": ("on", {"label": "table",
                             "pick": ("closest_to", {"label": "folding screen"})})},
         reading="shaky",
         note="attachment decided by uniqueness: the other reading leaves "
              "both bowls"),
    dict(scene="chinese_room", i=1,
         q="Find the pillow on the chair that is closest to the TV.",
         target={"label": "pillow",
                 "rel": ("on", {"label": "chair",
                                "pick": ("closest_to", {"label": "tv"})})},
         alt={"label": "pillow", "rel": ("on", {"label": "chair"}),
              "pick": ("closest_to", {"label": "tv"})},
         reading="open",
         note="'that is closest' attaches to the chair or to the pillow"),

    dict(scene="home_building_1", i=0,
         q="Find the clock on the TV cabinet.",
         target={"label": "clock", "rel": ("on", {"label": "tv cabinet"})},
         reading="solid"),
    dict(scene="home_building_1", i=1,
         q="Find the bowl closest to the knife rack near the trash can.",
         target={"label": "bowl",
                 "pick": ("closest_to",
                          {"label": "knife rack",
                           "pick": ("near", {"label": "trash can"})})},
         reading="shaky",
         note="only one knife rack exists, so 'near the trash can' is "
              "non-restrictive and the pick cannot use it"),

    dict(scene="home_building_2", i=0,
         q="Find the lamp on the nightstand that has the photo on it.",
         target={"label": ["lamp", "desk light"],
                 "rel": ("on", {"label": "nightstand",
                                "rel": ("on_inv", {"label": "photo"})})},
         reading="shaky",
         note="the only light on the nightstand carrying the photo is "
              "labelled `desk light`, not `lamp`"),
    dict(scene="home_building_2", i=1,
         q="Find the speaker on the TV cabinet closest to the potted plant on "
           "the TV cabinet.",
         target={"label": "speaker",
                 "rel": ("on", {"label": ["tv cabinet", "tv stand"]}),
                 "pick": ("closest_to",
                          {"label": "potted plant",
                           "rel": ("near", {"label": ["tv cabinet", "tv stand"]})})},
         reading="shaky",
         note="the scene's `tv cabinet` carries nothing; the furniture the "
              "speaker sits on is labelled `tv stand`. No potted plant is ON "
              "either, so the plant is read as beside it"),

    dict(scene="hotel_room_1", i=0,
         q="Find the bedside table farthest from the window.",
         target={"label": "bedside table",
                 "pick": ("farthest_from", {"label": "window"})},
         reading="solid"),
    dict(scene="hotel_room_1", i=1,
         q="Find the picture above the suitcase furthest from the floor.",
         # 'furthest from the floor' = highest. Read as qualifying the
         # picture: of the pictures above the suitcase, the highest one.
         target={"label": "picture", "rel": ("above", {"label": "suitcase"}),
                 "pick": ("farthest_from", {"label": "floor"})},
         reading="shaky",
         note="'furthest from the floor' read as height, and as qualifying "
              "the picture rather than the suitcase"),

    dict(scene="hotel_room_2", i=0,
         q="Find the flowers near the window.",
         target={"label": "flowers", "pick": ("near", {"label": "window"})},
         reading="solid"),
    dict(scene="hotel_room_2", i=1,
         q="Find the picture closest to the bench.",
         target={"label": "picture", "pick": ("closest_to", {"label": "bench"})},
         reading="solid"),

    dict(scene="japanese_room", i=0,
         q="The lantern between the vase and the stone decoration that is "
           "closest to the vase.",
         target={"label": "lantern",
                 "pick": ("between", {"label": "vase"},
                          {"label": "zen stone decoration"})},
         reading="shaky",
         note="only one stone decoration, so the trailing 'closest to the "
              "vase' cannot restrict it; betweenness alone decides"),
    dict(scene="japanese_room", i=1,
         q="The red pillow closest to the sushi.",
         target={"label": "pillow", "colour": "red",
                 "pick": ("closest_to", {"label": "sushi"})},
         reading="solid"),

    dict(scene="livingroom_1", i=0,
         q="Find the vase on the cabinet below the picture.",
         target={"label": "vase",
                 "rel": ("on", {"label": ["cabinet", "tv cabinet"],
                                "rel": ("below", {"label": "picture"})})},
         pad=0.40,
         reading="shaky",
         note="vase #23's footprint misses cabinet #2's by 0.35 m in the "
              "annotation, so `on` needs a wider pad here than anywhere else"),
    dict(scene="livingroom_1", i=1,
         q="Find the pillow on the sofa that is closest to the windows.",
         target={"label": "pillow", "rel": ("on", {"label": "sofa"}),
                 "pick": ("closest_to", {"label": "window"})},
         reading="shaky",
         note="one sofa only, so 'that is closest' must qualify the pillow"),

    dict(scene="livingroom_2", i=0,
         q="Find the stool closest to the shelf near the TV cabinet.",
         target={"label": "stool",
                 "pick": ("closest_to",
                          {"label": "shelf",
                           "pick": ("near", {"label": "tv cabinet"})})},
         reading="shaky",
         note="one shelf only; 'near the TV cabinet' is non-restrictive"),
    dict(scene="livingroom_2", i=1,
         q="Find the pillow on the sofa that is closest to the lamp.",
         target={"label": "pillow", "rel": ("on", {"label": "sofa"}),
                 "pick": ("closest_to", {"label": "lamp"})},
         reading="shaky",
         note="one sofa only, so 'that is closest' must qualify the pillow"),

    dict(scene="livingroom_3", i=0,
         q="Find the potted plant near the books on the cabinet.",
         target={"label": "potted plant",
                 "pick": ("closest_to",
                          {"label": "book", "rel": ("on", {"label": "cabinet"})})},
         reading="shaky",
         note="'near' read as the nearest, since several plants are near the "
              "books and the answer must be unique"),
    dict(scene="livingroom_3", i=1,
         q="Find the vase between the cabinet and the stool.",
         target={"label": "vase",
                 "pick": ("between", {"label": "cabinet"}, {"label": "stool"})},
         reading="solid"),

    dict(scene="livingroom_4", i=0,
         q="Find the picture closest to a window.",
         target={"label": "picture", "pick": ("closest_to", {"label": "window"})},
         reading="solid"),
    dict(scene="livingroom_4", i=1,
         q="Find the fossil decoration closest to the phone.",
         target={"label": "fossil decoration",
                 "pick": ("closest_to", {"label": "phone"})},
         reading="solid"),

    dict(scene="loft", i=0,
         q="The blue chair that is closest to the cup of coffee.",
         target={"label": "chair",
                 "pick": ("closest_to", {"label": "coffee cup"})},
         reading="shaky",
         note="all eleven chairs carry a dominant colour scheme covering 1% "
              "of their points, so `blue` cannot be read off the annotation; "
              "the superlative alone decides and the colour is assumed "
              "consistent with it"),
    dict(scene="loft", i=1,
         q="Find the potted plant between a vase and the cabinet with a TV on it.",
         target={"label": "potted plant",
                 "pick": ("between", {"label": "vase"},
                          {"label": ["cabinet", "tv cabinet"],
                           "rel": ("on_inv", {"label": "tv"})})},
         reading="solid"),

    dict(scene="office_1", i=0,
         q="Find the potted plant on the file cabinet.",
         target={"label": "potted plant", "rel": ("on", {"label": "file cabinet"})},
         reading="solid"),
    dict(scene="office_1", i=1,
         q="Find the paper cup on the table closest to the projector screen.",
         target={"label": "paper cup",
                 "rel": ("on", {"label": "table",
                                "pick": ("closest_to",
                                         {"label": "projector screen"})})},
         alt={"label": "paper cup", "rel": ("on", {"label": "table"}),
              "pick": ("closest_to", {"label": "projector screen"})},
         reading="open",
         note="attachment ambiguity, same shape as chinese_room q0"),

    dict(scene="office_2", i=0,
         q="Find the computer monitor closest to the cabinet with a phone on it.",
         target={"label": "computer monitor",
                 "pick": ("closest_to",
                          {"label": "cabinet", "rel": ("on_inv", {"label": "phone"})})},
         reading="solid"),
    dict(scene="office_2", i=1,
         q="Find the box on the cabinet that is closest to the whiteboard.",
         target={"label": "box", "rel": ("on", {"label": "cabinet"}),
                 "pick": ("closest_to", {"label": "whiteboard"})},
         alt={"label": "box",
              "rel": ("on", {"label": "cabinet",
                             "pick": ("closest_to", {"label": "whiteboard"})})},
         reading="open",
         note="'that is closest' attaches to the box or to the cabinet"),

    dict(scene="studio", i=0,
         q="Find the vase closest to the guitar.",
         target={"label": "vase", "pick": ("closest_to", {"label": "guitar"})},
         reading="solid"),
    dict(scene="studio", i=1,
         q="Find the beer bottle furthest from the couch.",
         target={"label": "beer bottle",
                 "pick": ("farthest_from", {"label": "couch"})},
         reading="solid"),
]


# --------------------------------------------------------------------------
# Scoring


def score_box(center, size, gt: dict) -> dict:
    """A predicted box against the ground-truth object: IoU and challenge points."""
    c = np.asarray(center, float)
    s = np.abs(np.asarray(size, float))
    lo, hi = c - s / 2, c + s / 2
    ov = np.maximum(0.0, np.minimum(hi, gt["hi"]) - np.maximum(lo, gt["lo"]))
    inter = float(ov.prod())
    union = float(s.prod()) + float(gt["sz"].prod()) - inter
    iou = inter / union if union > 0 else 0.0
    return {"iou": iou,
            "points": 2 if iou >= 0.50 else (1 if iou >= 0.25 else 0),
            "centre_error_m": float(np.linalg.norm(c - gt["c"])),
            "size_ratio": float(s.prod() / gt["sz"].prod())
            if gt["sz"].prod() > 0 else float("inf")}


# A ground-truth box this many times the median longest side of its own label
# in its own scene is worth looking at before trusting it.
ODD_RATIO = 3.0


def odd_gt(o: dict, objs: list[dict]) -> str:
    """Is the answer's own ground-truth box anomalous for its class?

    The key can only be as good as the annotation, and the annotation is not
    uniformly good. `studio`'s answer to *"the vase closest to the guitar"* is
    a box 2.49 m tall where the scene's other five vases run 0.17-0.69 m -- it
    is either a floor arrangement or a mis-annotation, and either way it is the
    question where our proxy and the organisers' scorer are most likely to
    disagree. Flagging it is not fixing it: nothing here can tell which.
    """
    same = [x for x in objs if x["label"] == o["label"]]
    if len(same) < 3:
        return ""
    longs = sorted(float(max(x["sz"])) for x in same)
    med = longs[len(longs) // 2]
    mine = float(max(o["sz"]))
    if med > 0 and mine / med >= ODD_RATIO:
        return f"box {mine:.2f} m vs {med:.2f} m median for {o['label']}"
    return ""


def answer(spec: dict, *, how: str = DIST, pad: float = PAD_M,
           use_alt: bool = False) -> dict:
    """Resolve one spec to the single object it names."""
    objs = load(spec["scene"])
    if not objs:
        return {"why": "no annotation", "n": 0, "obj": None}
    term = spec["alt"] if use_alt and spec.get("alt") else spec["target"]
    # A spec may override the pad. Exactly one does, and its `note` says why;
    # `--sens` still sweeps the global value over every question, so the
    # override shows up there as a question that does not move when it should.
    cands = resolve(term, objs, how=how, pad=spec.get("pad", pad))
    if len(cands) == 1:
        o = cands[0]
        return {"n": 1, "obj": o, "id": o["id"], "label": o["label"],
                "centre": o["c"].tolist(), "size": o["sz"].tolist(),
                "odd": odd_gt(o, objs), "why": "unique"}
    return {"n": len(cands), "obj": None,
            "ids": [c["id"] for c in cands],
            "why": "no candidate" if not cands else f"{len(cands)} candidates"}


def key(*, how: str = DIST, pad: float = PAD_M) -> list[dict]:
    out = []
    for spec in SPECS:
        got = answer(spec, how=how, pad=pad)
        row = {"scene": spec["scene"], "i": spec["i"], "q": spec["q"],
               "reading": spec["reading"], "note": spec.get("note", ""),
               "resolved": got}
        if spec.get("alt"):
            row["alt"] = answer(spec, how=how, pad=pad, use_alt=True)
            a, b = got.get("id"), row["alt"].get("id")
            row["alt_agrees"] = bool(a and b and a == b)
        out.append(row)
    return out


def check_questions() -> list[str]:
    """Every spec's `q` must be a question the challenge actually asks."""
    f = CHALLENGE / "questions" / "questions.json"
    if not f.is_file():
        return [f"challenge questions not found at {f}"]
    asked = {s["scene"]: s["questions"]["object_reference"]
             for s in json.loads(f.read_text())}
    bad = []
    for spec in SPECS:
        theirs = asked.get(spec["scene"], [])
        if spec["i"] >= len(theirs):
            bad.append(f"{spec['scene']} q{spec['i']}: no such question")
        elif " ".join(theirs[spec["i"]].split()) != " ".join(spec["q"].split()):
            bad.append(f"{spec['scene']} q{spec['i']}:\n"
                       f"    spec:      {spec['q']}\n"
                       f"    challenge: {theirs[spec['i']]}")
    for scene, qs in asked.items():
        have = {s["i"] for s in SPECS if s["scene"] == scene}
        for i in range(len(qs)):
            if i not in have:
                bad.append(f"{scene} q{i}: no spec for {qs[i]!r}")
    return bad


# --------------------------------------------------------------------------


def report(rows: list[dict]) -> None:
    w = max(len(r["scene"]) for r in rows)
    ok = 0
    for r in rows:
        got = r["resolved"]
        mark = "+" if got["n"] == 1 else "!"
        ok += got["n"] == 1
        tail = (f"{got['label']} #{got['id']}" if got["n"] == 1 else got["why"])
        alt = ""
        if "alt_agrees" in r:
            alt = ("  alt agrees" if r["alt_agrees"]
                   else f"  ALT DIFFERS -> {r['alt'].get('id', r['alt']['why'])}")
        print(f" {mark} {r['scene']:<{w}} q{r['i']}  {r['reading']:<6} {tail}{alt}")
        if got.get("odd"):
            print(f"      ODD GROUND TRUTH: {got['odd']}")
    print(f"\n{ok}/{len(rows)} questions resolve to exactly one object")
    for tag in ("solid", "shaky", "open"):
        n = sum(1 for r in rows if r["reading"] == tag)
        good = sum(1 for r in rows if r["reading"] == tag and r["resolved"]["n"] == 1)
        print(f"   {tag:<6} {good}/{n}")
    diff = [r for r in rows if r.get("alt_agrees") is False]
    if diff:
        print(f"\n{len(diff)} question(s) where the two readings disagree:")
        for r in diff:
            print(f"   {r['scene']} q{r['i']}: {r['note']}")


def explain(scene: str, *, how: str = DIST, pad: float = PAD_M) -> None:
    objs = load(scene)
    for spec in [s for s in SPECS if s["scene"] == scene]:
        print(f"\n{spec['q']}\n  reading: {spec['reading']}"
              + (f"\n  note: {spec['note']}" if spec.get("note") else ""))
        for tag, term in (("target", spec["target"]),
                          ("alt", spec.get("alt"))):
            if term is None:
                continue
            cands = resolve(term, objs, how=how, pad=pad)
            print(f"  {tag}: {len(cands)} candidate(s)")
            for c in cands:
                print(f"      #{c['id']:<5} {c['label']:<22}"
                      f" c=({c['c'][0]:+.2f},{c['c'][1]:+.2f},{c['c'][2]:+.2f})"
                      f" sz=({c['sz'][0]:.2f},{c['sz'][1]:.2f},{c['sz'][2]:.2f})")


def sens() -> None:
    """Which answers survive the two free parameters.

    A question that resolves to a *different object* under a different reading
    of "closest" or a different footprint slack is a question we do not know
    the answer to. One that merely stops resolving needs slack to exist and is
    reported separately -- that is a weaker complaint.
    """
    runs = {name: {(r["scene"], r["i"]): r["resolved"].get("id")
                   for r in key(how=how, pad=pad)}
            for name, how, pad in (("surface", "surface", 0.10),
                                   ("centre", "centre", 0.10),
                                   ("pad 0", "surface", 0.00),
                                   ("pad .3", "surface", 0.30))}
    cols = list(runs)
    print(f"{'scene':<17}q  " + "".join(f"{c:>9}" for c in cols))
    flipped = unresolved = 0
    for k in runs["surface"]:
        vals = [runs[c].get(k) for c in cols]
        ids = {v for v in vals if v is not None}
        flag = ""
        if len(ids) > 1:
            flag, flipped = "  <- FLIPS", flipped + 1
        elif None in vals:
            flag, unresolved = "  <- needs slack", unresolved + 1
        print(f"{k[0]:<17}{k[1]}  " + "".join(f"{str(v):>9}" for v in vals) + flag)
    n = len(runs["surface"])
    print(f"\n{flipped}/{n} answers FLIP to a different object  <- these are the "
          f"ones that would make the key wrong")
    print(f"{unresolved}/{n} stop resolving without footprint slack")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--why", metavar="SCENE", help="show one scene's candidates")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 unless every spec singles out one object")
    ap.add_argument("--sens", action="store_true", help="sweep the free parameters")
    ap.add_argument("--json", metavar="PATH", help="write the key")
    ap.add_argument("--dist", choices=("surface", "centre"), default=DIST)
    ap.add_argument("--pad", type=float, default=PAD_M)
    a = ap.parse_args()

    bad = check_questions()
    if bad:
        print("SPEC TEXT DOES NOT MATCH THE CHALLENGE:", file=sys.stderr)
        for b in bad:
            print("  " + b, file=sys.stderr)
        return 2

    if a.why:
        explain(a.why, how=a.dist, pad=a.pad)
        return 0
    if a.sens:
        sens()
        return 0

    rows = key(how=a.dist, pad=a.pad)
    report(rows)
    if a.json:
        p = Path(a.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(
            [{k: v for k, v in r.items() if k != "resolved"}
             | {"resolved": {k: v for k, v in r["resolved"].items() if k != "obj"}}
             for r in rows], indent=1))
        print(f"\nwritten: {p}")
    if a.check:
        return 0 if all(r["resolved"]["n"] == 1 for r in rows) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
