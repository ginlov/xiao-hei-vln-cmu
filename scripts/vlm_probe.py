#!/usr/bin/env python3
"""Ask a VLM to point at one referring expression in the four faces.

This is the single call that `docs/vlm_grounding_test.md` Part A is built from.
Run it by hand on one snapshot to shake out the prompt; the scored 60-pair
sweep reuses `build_prompt` and `ask` unchanged.

The prompt is written to make NOT_VISIBLE cheap to say. Our acceptance bar is
a false-positive rate under 10%, because a miss costs one question while a
confident wrong answer sends the robot elsewhere and corrupts the ordering of
every constraint after it.

    export ANTHROPIC_API_KEY=...
    uv run --with anthropic python scripts/vlm_probe.py snaps/start \
        "the tea table with the elephant figurine on it"

    uv run --with google-genai python scripts/vlm_probe.py snaps/start \
        "the folding screen" --backend gemini
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path

FACES = ["face_0_front.jpg", "face_1_right.jpg",
         "face_2_back.jpg", "face_3_left.jpg"]
HEADINGS = [0, 90, 180, 270]
NAMES = ["front", "right", "back", "left"]
# Pinned, not `gemini-flash-latest`: the alias moved from 3.5 to 3.6 during the
# hour this was written, and an A/B whose model changed halfway measured
# nothing. Verified callable rather than read off `models.list` — the previous
# default, `gemini-2.5-flash`, is still in that catalogue and answers 404 "no
# longer available to new users". `--model gemini-3.1-pro-preview` is the
# frontier tier and needs a billed project; the free tier gives it a daily
# quota of zero.
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"

PROMPT_V3 = """You are looking at what a robot sees from one spot in a room.

The four images are perspective views from the same position, each 100° wide:

  image 0 — heading   0°  (front)
  image 1 — heading  90°  (right)
  image 2 — heading 180°  (back)
  image 3 — heading 270°  (left)

Adjacent images overlap by about 10°, so an object near an edge may appear in
two of them.

TARGET
  "{phrase}"

The target is one of two kinds, and they are judged differently:

(a) A bare object type — "chair", "table", "tv". Report the clearest visible
    instance. Any real instance of that type counts; there is nothing to
    disambiguate, so do not withhold an answer for lack of a qualifier.

(b) A type plus a DISTINGUISHING FEATURE — "the tea table with the elephant
    figurine on it". This room contains several objects of most types, so the
    type alone does not identify the target. Report a match ONLY if you can
    actually see the distinguishing feature. If you can see objects of the
    right type but not the feature on any of them, answer NOT_VISIBLE. That is
    the correct answer, not a failure — a confident wrong match is far more
    costly to us than a miss.

A laser scanner measures the distance to whatever the centre of your box points
at. Three things in a room defeat it, and you can see all three even though you
cannot see the scanner:

  - glass. A window or a glass door returns the glass, not what is behind it.
    An object on the far side is also somewhere the robot cannot drive to.
  - open structure. An open stair, a railing, a wire shelf, a leafy plant: the
    beam passes through the gaps and measures the wall beyond.
  - something in front. Anything between the camera and the target — a desk in
    front of a window, a chair in front of a cabinet — is measured instead.

Report which of these applies. This is the single most useful thing you can
tell us, and it is a question about what you see, not about the scanner.

Also give a rough distance. It is used only to cross-check the scanner, so an
honest wide interval is worth more than a confident narrow one — we need to
catch a scanner reading that is off by a factor of three, not by 20%.

Each image is {size} x {size} pixels.

Reply with JSON only, no prose, no markdown fence:

{{
  "coord_space": "pixels" or "normalized_1000",
      which convention the boxes below are in. State it explicitly — with a
      {size}-pixel image both readings produce plausible-looking numbers, and
      a silent mismatch puts the waypoint on the wrong object.
  "visible": true or false,
  "image_index": 0-3 or null,
  "box_2d": [ymin, xmin, ymax, xmax],
      the TARGET object, in the coord_space you declared
  "feature_box_2d": [ymin, xmin, ymax, xmax] or null,
      the distinguishing feature itself, same convention
  "occlusion": "clear" | "behind_glass" | "see_through" | "partly_occluded",
      what stands between the camera and the target, per the list above
  "same_space": true or false,
      is the target in the same room the camera is in — i.e. could the robot
      drive to it without going through a window, a glass wall, or a balcony
      door. When several instances are visible, prefer one where this is true.
  "distance_m": number,
      your best estimate of camera-to-target distance in metres
  "distance_range_m": [low, high],
      the interval you would actually stand behind
  "confidence": 0.0-1.0,
      probability that this is the object the description refers to
  "evidence": "what you actually see that identifies it",
  "alternates": [
      {{"image_index": n, "box_2d": [...], "why_rejected": "..."}}
  ],
  "explore": {{"heading_deg": 0-359, "why": "..."}}
      where to look next; only meaningful when visible is false
}}"""

# Appended only when driving an approach loop. Kept out of the base prompt so
# the cached replies behind `vlm_sweep.py` stay valid — a prompt edit there
# invalidates every one of them and costs a fresh $1.43 sweep.
#
# Neither field decides anything yet. `target_state` is the qualitative form of
# a question the model answered badly in metres (its `distance_m` is worse than
# a constant), so it is logged and compared against geometry before it is
# allowed near the controller — the same discipline `same_space` failed, having
# returned true on all 52 claimed sightings including chairs behind glass.
APPROACH_HERE = """
  "here": "one short clause naming where the robot is standing and what it
      could search from here — e.g. \\"the dining area left of the staircase;
      table top hidden behind the chair backs\\". This is written into a log of
      places already visited and given back to you on later calls, so write it
      for a reader who cannot see this image."
"""

APPROACH_BLOCK = """

{n} more fields, appended to the JSON above.

{here}  "target_state": "far" | "approaching" | "adjacent",
      how close the camera is to the target, judged only from how the target
      sits in the frame — how much of the view it fills, whether the frame cuts
      it off, whether you can make out surface detail. Do not convert this to
      metres and do not derive it from your distance estimate.
        far          it reads as part of the scene, small in the view
        approaching  it dominates its part of the view but is fully contained
        adjacent     the robot is right at it: it fills much of the frame, or
                     runs past the frame edge, and its surface texture is plain
  "same_object_as_previous": true | false | null,
      null unless a "previous view" image was supplied. When one was, that crop
      is the object this robot was driving toward before it moved, seen from
      further away and from a different angle. Answer true only if the thing
      you boxed is that same physical object, not merely another one like it.
      This room contains several objects of most types and the robot has moved
      since, so a same-type match is not evidence. Say what carries over —
      markings, wear, what sits on it, its position relative to walls and
      neighbours. If you cannot tell, answer false.
"""


# --- v4: comparative relations ------------------------------------------------
#
# v3 has two branches: a bare noun, and a type plus a feature *on* it ("the tea
# table with the elephant figurine on it"). The official question set is not
# shaped like that. Counting the released questions:
#
#                          closest/nearest to   farthest   between   with X on it
#   object_reference            18/30 (60%)      3/30       4/30        2/30
#   instruction_following       14/30 (47%)      5/30      12/30        5/30
#
# The branch v3 was written for is 7-17% of the questions; the dominant form
# has no branch at all. Run live on "the lantern closest to the fan decoration"
# in japanese_room, v3 boxed **the fan decoration** — bearing and elevation
# within 2.6° and 2.2° of the anchor, and 22.5°/34.5° from the lantern. It
# pointed perfectly at the wrong object, because nothing told it which noun was
# the target.
#
# v4 adds a branch that refuses to answer the comparison at all. The model
# reports every candidate and the anchor; we measure and pick. "Closest" is
# arithmetic on positions we can already compute to a median 0.11 m, and it is
# the same division of labour as everywhere else here — the model proposes
# semantics, geometry decides metrics.
_RELATIONAL_BRANCH = """
(c) A type plus a COMPARATIVE RELATION to a different object — "the lantern
    closest to the fan decoration", "the pillow farthest from the lamp", "the
    chair between the table and the window".

    The target is the noun BEFORE the relation word; the object named after it
    is the ANCHOR, and is not the answer. Returning the anchor is the single
    most common way to get this wrong.

    Do NOT decide which candidate wins. Report every visible instance of the
    target type in "candidates", and the object(s) the relation refers to in
    "anchors". We measure the distances between them ourselves — comparing
    positions is arithmetic and we are better at it than you are. Finding and
    naming the objects is not, and that is the part we need from you.

    Still fill "box_2d" and "image_index" with whichever candidate you would
    pick, as a fallback for when the measurement cannot be made.

"""

_RELATIONAL_FIELDS = """  "relation": "closest_to" | "farthest_from" | "between" | null,
      the comparison in the request, if it is of kind (c)
  "target_type": "the head noun of the target, e.g. \\"lantern\\"" or null,
  "candidates": [
      {{"image_index": n, "box_2d": [...], "note": "which one this is"}}
  ],
      every visible instance of the target type; [] if kind (a) or (b)
  "anchors": [
      {{"name": "fan decoration", "image_index": n, "box_2d": [...]}}
  ],
      the object(s) named after the relation word; [] if kind (a) or (b)
"""


_CONSTRAINT_BRANCH = """
PATH CONSTRAINTS. The request may also say something about the *route*, not
about the target. Two kinds, and they are opposites:

  avoid  "avoid the path near the cabinet", "do not go between the sofa and
         the tv" — a region the robot must stay out of. Report the object or
         objects the region is anchored on.

  gate   "take the path between the two tables", "go past the window" — a
         place the robot must pass *through*. Report the object or objects the
         passage is defined by.

Report only what the request actually says. Most requests have neither. If you
are unsure which kind a phrase is, leave both empty rather than guess: sending
the robot around a passage it was told to use is as wrong as driving through a
region it was told to avoid.
"""

_CONSTRAINT_FIELDS = """  "avoid": [
      {{"name": "cabinet", "image_index": n, "box_2d": [...]}}
  ],
      objects anchoring a region to stay out of; [] if the request has none
  "gate": [
      {{"name": "table", "image_index": n, "box_2d": [...]}}
  ],
      objects defining a passage to drive through; [] if the request has none
"""


_WAY_BRANCH = """
WHEN THE TARGET IS NOT VISIBLE, POINT AT THE WAY OUT. A heading alone is not
enough. The robot turns your heading into a bearing and then drives along
whatever the floor allows, and the floor almost always allows the open room
more than it allows a doorway: a door is a metre wide with a wall two metres
behind it, a hallway runs six. So a heading that means "through that door"
arrives as "along the wall beside it", and the robot searches the same room
until its time runs out.

If the way onward is something you can see — a doorway, an archway, an opening,
the mouth of a corridor, the foot of a stair — box it in "way", exactly as you
would box a target. Box the *opening*, not the door leaf and not the room
beyond: the gap the robot drives through, floor to lintel. Then the robot can
be sent to the opening itself rather than in its general direction.

If you can see no opening at all — a blank wall, a room whose exits are all
behind you — leave "way" null and give the heading alone. Do not box a
promising-looking wall.
"""

_WAY_FIELDS = """  "way": {{"name": "doorway to the hall", "image_index": n,
           "box_2d": [...]}} or null,
      the opening to drive to when visible is false; null if none is in sight
"""


def _make_v6() -> str:
    """v6 = v5 plus a boxed way out, by the same surgery so v5 stays exact.

    `explore.heading_deg` was the loop's only way of being told where to go
    when the target is not in sight, and a bearing cannot express "through that
    door". `home_building_1` leg 1 measured the cost: the model named a visible
    doorway on seven of nine calls, the loop walked 20 m for 7 m of net
    displacement, and the leg ended without entering a single room. The
    scoring fix in `explore_direction` stops the bearing being swung onto the
    corridor; this stops it being a bearing at all, so the opening can be
    lifted to a coordinate with the same `_lift_xy` that already places
    targets, anchors and gates.
    """
    # v5 already inserted its branch ahead of this line, so the blank line v4
    # and v3 anchor on is gone; anchor on the single newline v5 leaves.
    a = "\nA laser scanner measures the distance to whatever"
    b = '  "explore": {{"heading_deg": 0-359, "why": "..."}}'
    assert PROMPT_V5.count(a) == 1, "v6 anchor is not unique in v5"
    out = PROMPT_V5.replace(a, "\n" + _WAY_BRANCH + a.lstrip("\n"), 1)
    out = out.replace(b, _WAY_FIELDS + b, 1)
    assert out.count("POINT AT THE WAY OUT") == 1, "way branch not inserted"
    assert out.count('"way": {{') == 1, "way field not inserted"
    return out


def _make_v5() -> str:
    """v5 = v4 plus route constraints, again by surgery so v4 stays exact.

    Only 3 of the 30 instruction-following questions carry a keep-out, but 10
    carry a required passage, and conflating the two turns ten questions into
    guaranteed failures — hence two separate fields rather than one "constraint".
    """
    a = "\n\nA laser scanner measures the distance to whatever"
    b = '  "explore": {{"heading_deg": 0-359, "why": "..."}}'
    out = PROMPT_V4.replace(a, "\n" + _CONSTRAINT_BRANCH + a.lstrip("\n"), 1)
    out = out.replace(b, _CONSTRAINT_FIELDS + b, 1)
    assert out.count("PATH CONSTRAINTS") == 1, "constraint branch not inserted"
    assert out.count('"gate": [') == 1, "constraint fields not inserted"
    return out


def _make_v4() -> str:
    """v4 = v3 plus branch (c) and its fields, by surgery so v3 stays exact.

    Composing from parts would risk drifting v3's bytes, and every one of the
    117 cached replies is keyed to them.
    """
    a = "\n\nA laser scanner measures the distance to whatever"
    b = '  "explore": {{"heading_deg": 0-359, "why": "..."}}'
    out = PROMPT_V3.replace(a, "\n" + _RELATIONAL_BRANCH + a.lstrip("\n"), 1)
    out = out.replace(b, _RELATIONAL_FIELDS + b, 1)
    assert out.count("COMPARATIVE RELATION") == 1, "branch (c) not inserted"
    # `"candidates"` appears in the branch prose as well, so count something
    # that only the schema block has.
    assert out.count('"target_type"') == 1, "relational fields not inserted"
    return out


PROMPT_V4 = _make_v4()
PROMPT_V5 = _make_v5()
PROMPT_V6 = _make_v6()

PROMPTS: dict[str, str] = {
    "v3-occlusion-distance": PROMPT_V3,   # what TASK 26's numbers were measured on
    "v4-relational": PROMPT_V4,
    "v5-constraints": PROMPT_V5,
    "v6-way-out": PROMPT_V6,
}
# v5 stays reachable by name: the offline scripts and the 117 cached replies are
# keyed to it, and a scene that never leaves one room does not need v6.
DEFAULT_PROMPT_VER = "v6-way-out"

# Back-compat for callers that imported the module-level name.
PROMPT = PROMPT_V3


VISITED_BLOCK = """

PLACES ALREADY SEARCHED. The robot wrote these on earlier calls, oldest first.
Do not send it back to one of them unless the request can only be satisfied
there and you say why. Prefer somewhere it has not stood.

{visited}
"""

# The geometric renderings. Same instruction, a tenth of the tokens, and stated
# in the only frame the model and the map share: the headings that number the
# four faces. Map coordinates are the obvious thing to send and the useless one
# — the model reasons over these images and has never seen the frame those
# numbers live in — so `bearing` converts each visited pose into the heading
# the robot would have to turn to in order to drive back to it. `xy` sends the
# raw frame anyway, as the control that shows what that costs.
VISITED_BLOCK_BEARING = """

PLACES ALREADY SEARCHED, given as where they lie from where the robot stands
right now. Headings use the same convention as the four images above: 0° is
image 0 (front), 90° is image 1 (right), 180° is image 2 (back), 270° is image
3 (left). Do not send it back to one of them unless the request can only be
satisfied there and you say why. Prefer somewhere it has not stood.

{visited}
"""

VISITED_BLOCK_XY = """

PLACES ALREADY SEARCHED, as coordinates in the robot's map frame, oldest
first. Do not send it back to one of them unless the request can only be
satisfied there and you say why. Prefer somewhere it has not stood.

{visited}
"""

VISITED_BLOCKS = {"prose": VISITED_BLOCK, "bearing": VISITED_BLOCK_BEARING,
                  "xy": VISITED_BLOCK_XY}


MISSION_BLOCK = """

THE WHOLE INSTRUCTION. The request above is one leg of it, and the rest is here
because a leg often cannot be read alone: "the picture closest to the TV" needs
the TV, which an earlier leg may have named.

  {question}

The plan, in the order the robot drives it:

{plan}

The robot is on step {k}. Steps before it are done and it has stood in those
places. The fields above are about step {k} and nothing else: do not box a
later step's object in "box_2d", and do not let a later step change what you
report as visible.

WITH ONE EXCEPTION, WHICH IS WORTH MORE THAN THE REST OF THIS BLOCK. If, while
looking for step {k}, you happen to see an object a LATER step names, say so in
"sightings". Nothing here is asked twice: the robot arrives at a later step
having forgotten the room, and one sentence written now can save it a search
that costs minutes.

  "sightings": [
      {{"step": n, "what": "blue trash can beside the stainless fridge, on the
        counter run under the window", "image_index": n, "box_2d": [...]}}
  ]

`step` is which numbered step above it belongs to. `what` is written for a
reader who cannot see this image and will arrive from somewhere else, so name
the thing and what it stands next to. `box_2d` is optional and only worth
giving when you are confident which object it is -- it is used to point the
robot in a direction, never to decide it has arrived.

An empty list is the ordinary answer. Report a sighting only when you can
actually see the object, not when you can see the room it is probably in.
{sightings}{done}{keepouts}"""

# Fed back on the leg the sighting was for, and deliberately not merged into
# `VISITED_BLOCK`, which says "do not go back there". A sighting is the
# opposite instruction, and on `home_building_1` the two were the same sentence:
# the model wrote "counter run with window and blue trash can to the right,
# stainless fridge behind" on step 6 of leg 1, and leg 3 -- whose target is "the
# trash can closest to the refridgerator" -- got it back under a heading telling
# it not to return.
SIGHTINGS_BLOCK = """
SEEN EARLIER, AND WORTH GOING BACK FOR. The robot wrote these while working on
an earlier step, when it happened to see what this step is looking for:

{sightings}

Treat this as a lead, not as an answer. It was written from somewhere else and
the robot has moved since; confirm it against what you can see now. If it names
a place you cannot see from here, that is where to head.
"""

# Only rendered once something has actually been banked, so a prompt with an
# empty plan history is byte-for-byte what it was before -- which is what keeps
# the 117 cached replies keyed to v5 valid.
#
# The rule matters most for a passage. The score is on the trajectory and on
# the ORDER of the constraints in it (README §175), so a required passage
# driven and then driven back out of is worse than one driven once: the path
# now reads through, back, through. The robot had no way to know this. On
# `home_building_1` the destination leg after "take the path between the dining
# table and the picture" turned round on its first exploration call in three
# runs of four and drove back out through the same gap, once all the way to
# where the passage had started.
DONE_BLOCK = """
Already driven, in this order, and not to be repeated:

{done}

A constraint the robot has already satisfied is spent. When the current target
is not in sight, the way onward is not back the way it came: prefer an opening
the robot has not used, and treat the passage or doorway it arrived through as
the last resort rather than the obvious choice. Send it back through one only
if the request can only be satisfied on that side and you say so in "why".
"""

# Keep-outs are named here rather than left to the request phrase because they
# hold for the whole run, not for the step: the executor lifts their anchors
# from whichever call happens to see them, so every call has to be willing to
# report them. Without this the `avoid` field only ever gets filled on the leg
# whose own phrase mentions the region, and by then the robot may have already
# driven through it.
KEEPOUT_BLOCK = """
Throughout, the robot must NOT drive through:

{keepouts}

This holds on every step, including this one. Whenever you can see an object
that one of these regions is anchored on, report it under "avoid" -- even if
the request above never mentions it.

(The robot is not steered round a keep-out: see `USE_KEEPOUT` in
`approach_loop`. Naming the region is still worth the two lines it costs --
it is what any later enforcement would be built from, and it is recorded.)
"""


def build_prompt(phrase: str, size: int = 640, *, approach: bool = False,
                 version: str = DEFAULT_PROMPT_VER,
                 visited: list[str] | None = None,
                 visited_kind: str = "prose", ask_here: bool = True,
                 mission: dict | None = None) -> str:
    """The prompt, optionally with the approach fields, a visit log and a plan.

    `visited` is the list of already-searched places, pre-rendered by the
    caller; `visited_kind` says which of `VISITED_BLOCKS` frames it, because
    the heading a bearing list is read under is not the heading prose is read
    under. `ask_here` drops the `here` field from the request entirely, which
    is only sane when nothing is going to feed it back.

    The original design fed back the model's own `here` clauses on the grounds
    that map coordinates are useless to something that reasons over images and
    has never seen the frame. The first half of that has since been measured
    and the second half has not. Over 41 paired calls — the same step, the same
    images, with and without the block, against a same-prompt control arm that
    establishes what stochastic thinking alone does — removing the block
    changes the explore heading more than re-rolling the identical prompt does
    (22/30 steps, sign test p = 0.016). So the model reads it. But on the one
    thing the block asks for, keeping away from places already stood, the
    measured advantage is 0.06 m, and the two samples of the *same* prompt
    differ by 0.06 m too. It moves the answer without moving it anywhere.

    That is what `visited_kind` exists to test. `bearing` states the same
    places in the frame the model can actually act on — the headings that
    number the four faces — at roughly a tenth of the tokens. `xy` sends raw
    map coordinates, which is the reading the original docstring called
    useless, kept as the control that shows whether it was right.

    `mission` is `{question, plan, k, done}`: the sentence the leg was cut out
    of, the whole ordered plan, which step is being asked about, and what has
    already been banked. It is context, not a decision — the model is never
    asked which step to do next, because the progress cursor is the executor's
    and stays monotonic. Reporting on the current step is a judgement the model
    can make from what it sees; deciding that a step is finished is one that it
    demonstrably cannot (`bind_target`). `done` is the executor telling it what
    it decided, which is the opposite direction and safe.
    """
    if version not in PROMPTS:
        raise SystemExit(f"unknown prompt version {version!r}; "
                         f"have {sorted(PROMPTS)}")
    base = PROMPTS[version].format(phrase=phrase, size=size)
    if approach:
        base += APPROACH_BLOCK.format(
            n="Three" if ask_here else "Two",
            here=APPROACH_HERE.lstrip("\n") if ask_here else "")
    if mission:
        keep = mission.get("keepouts") or []
        done = mission.get("done") or []
        seen = mission.get("sightings") or []
        base += MISSION_BLOCK.format(
            question=mission["question"], k=mission["k"],
            plan="\n".join(
                f"  {'->' if i == mission['k'] else '  '} {i}. {line}"
                for i, line in enumerate(mission["plan"], 1)),
            sightings=("" if not seen else SIGHTINGS_BLOCK.format(
                sightings="\n".join(f"  - {x}" for x in seen))),
            done=("" if not done else DONE_BLOCK.format(
                done="\n".join(f"  - {x}" for x in done))),
            keepouts=("" if not keep else KEEPOUT_BLOCK.format(
                keepouts="\n".join(f"  - {x}" for x in keep))))
    if visited:
        if visited_kind not in VISITED_BLOCKS:
            raise SystemExit(f"unknown visited_kind {visited_kind!r}; "
                             f"have {sorted(VISITED_BLOCKS)}")
        base += VISITED_BLOCKS[visited_kind].format(visited="\n".join(
            f"  {i}. {v}" for i, v in enumerate(visited, 1)))
    return base


def to_pixels(box: list[float], space: str | None, size: int) -> list[float]:
    """Box in image pixels, whichever convention the model answered in.

    Claude returns raw pixels; Gemini's trained detection output is normalised
    to 0-1000. On a 640-pixel face both land in the same numeric range, so the
    declared `coord_space` is trusted first and the fallback only guesses when
    it is missing.

    The declaration is only worth trusting once `settle_coord_space` has had a
    look at it — see there for what Gemini 3.1 Pro declares and what it sends.
    """
    if space == "pixels":
        return list(box)
    if space == "normalized_1000":
        return [v / 1000.0 * size for v in box]
    # No declaration: values above the image size can only be 0-1000.
    return [v / 1000.0 * size for v in box] if max(box) > size else list(box)


def every_box(reply: dict) -> list[list[float]]:
    """Every 2-D box anywhere in a reply, wherever the schema puts them."""
    out = [b for b in (reply.get("box_2d"), reply.get("feature_box_2d")) if b]
    for group in ("alternates", "candidates", "anchors", "sightings"):
        for it in reply.get(group) or []:
            if isinstance(it, dict) and it.get("box_2d"):
                out.append(it["box_2d"])
    return [b for b in out if isinstance(b, (list, tuple)) and len(b) == 4]


def settle_coord_space(reply: dict, backend: str, size: int) -> dict:
    """Correct `coord_space` before anything converts a box with it.

    Gemini 3.1 Pro declares `"pixels"` and sends 0-1000. Measured over nine
    calls across three models and three scenes: Pro declared `"pixels"` on two
    of three and `"normalized_1000"` on the third, while `gemini-3.6-flash` and
    `gemini-3.1-flash-lite` declared `"normalized_1000"` every time. In every
    call where the magnitude can decide — a coordinate above the image cannot
    be a pixel — it decided *normalised*, and no call anywhere was shown to be
    in pixels. On `runs/cr_0811_03` step 4 Pro returned
    `feature_box_2d: [506, 388, 885, 559]` under `"pixels"`, and 885 does not
    exist on a 640-pixel face; read as declared, that box lands on bare floor
    two metres right of the plant it describes.

    Magnitude alone cannot carry the fix. The same sweep had Pro declare
    `"pixels"` with a maximum of 494, which is a legal pixel value and a legal
    normalised one, so the reply is undecidable on its own numbers and would
    pass straight through. The backend is what settles it: Gemini's detection
    output is normalised by construction, which is what the comment in
    `to_pixels` already said before anything relied on it.

    For Claude the declaration stands, with one arithmetic override: a box
    outside the image is not a pixel box whatever the reply calls it.
    """
    if not isinstance(reply, dict):
        return reply
    if backend == "gemini":
        reply["coord_space"] = "normalized_1000"
        return reply
    boxes = every_box(reply)
    if boxes and max(max(b) for b in boxes) > size:
        reply["coord_space"] = "normalized_1000"
    return reply


def bearing_deg(box_px: list[float], face_idx: int) -> tuple[float, float]:
    """Egocentric (yaw, pitch) of a box centre, in degrees.

    Yaw is measured the way the robot's map frame does — x forward, y left,
    counter-clockwise positive — so it can be compared against a ground-truth
    `atan2(dy, dx)` without a second convention in the caller.
    """
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "perception"))
    import geometry as G

    ymin, xmin, ymax, xmax = box_px
    d = G.face_pixel_to_world_dir(np.array([(xmin + xmax) / 2]),
                                  np.array([(ymin + ymax) / 2]), face_idx)[0]
    yaw = np.rad2deg(np.arctan2(-d[0], d[2]))            # +y_cam is down
    pitch = -np.rad2deg(np.arcsin(np.clip(d[1] / np.linalg.norm(d), -1, 1)))
    return float(yaw), float(pitch)


def load_faces(d: Path) -> list[bytes]:
    out = []
    for name in FACES:
        p = d / name
        if not p.is_file():
            raise SystemExit(f"missing {p} — run scripts/snap.sh first")
        out.append(p.read_bytes())
    return out


def load_geometry(d: Path) -> tuple[dict, np.ndarray] | None:
    """(pose, scan) for a snapshot, if this directory carries them.

    Two layouts in use: `scripts/snap.sh` writes `pose.json` + `scan.npy`, and
    a recorded tour keeps the pose in `frames.jsonl` beside the scan. Faces on
    their own — `scripts/grab_faces.py` output — carry neither.
    """
    import numpy as np      # kept out of module scope, as elsewhere here

    if (d / "pose.json").is_file() and (d / "scan.npy").is_file():
        return json.loads((d / "pose.json").read_text()), np.load(d / "scan.npy")
    jl = d / "frames.jsonl"
    if jl.is_file():
        rec = json.loads(jl.read_text().splitlines()[0])
        scans = sorted(d.glob("*_registered.npy"))
        if scans and "position" in rec:
            return ({"position": rec["position"],
                     "orientation": rec["orientation"]}, np.load(scans[0]))
    return None


def ask_claude(prompt: str, images: list[bytes], model: str,
               previous: bytes | None = None) -> str:
    import anthropic

    content: list[dict] = []
    if previous is not None:
        # Ahead of the four faces, so "previous view" is established before the
        # model starts reading the current ones.
        content.append({"type": "text", "text": "previous view of the target "
                                                "(from further away):"})
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg",
            "data": base64.b64encode(previous).decode()}})
    for i, raw in enumerate(images):
        # Label each image in-band. Without this the model has to infer the
        # order from arrival, and any reordering downstream would silently
        # remap the headings.
        content.append({"type": "text", "text": f"image {i} ({NAMES[i]}, "
                                                f"heading {HEADINGS[i]}°):"})
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg",
            "data": base64.b64encode(raw).decode()}})
    content.append({"type": "text", "text": prompt})

    # The SDK retries 5xx twice by default, which a sustained 529 outlasts —
    # and a sweep that dies on call 40 of 54 has to be re-invoked by hand even
    # though the cache keeps the earlier work. Backoff is exponential with
    # jitter, so the extra attempts cost nothing when the service is healthy.
    client = anthropic.Anthropic(
        max_retries=int(os.environ.get("XIAO_HEI_API_MAX_RETRIES", "8")))
    # v3 replies measured 460 output tokens, which 2048 covered comfortably.
    # v4 asks the model to enumerate every candidate and anchor with a note
    # each, on top of `alternates` and `evidence`; a five-lantern reply on
    # japanese_room runs past 2048 and comes back with the JSON cut mid-object.
    # `parse` then finds no closing brace and reports "unparseable", which
    # looks like a model failure and is a budget failure.
    #
    # 4096 then stopped covering it either, and not because the answer grew:
    # on `claude-opus-5` thinking is on unless it is switched off — a change
    # from 4.8, where omitting the parameter meant no thinking — and thinking
    # is billed against this same ceiling. Same shape as the Gemini note
    # below. A grounding call that reasons its way across four faces can
    # spend the whole budget before it writes a brace, so the reply is not
    # cut off late, it never starts: `usage.output_tokens` comes back equal
    # to the ceiling. 16000 covers the thinking and the answer, and stays
    # under the SDK's non-streaming HTTP timeout, which is what rules out
    # simply going to the model's 128000.
    msg = client.messages.create(
        model=model,
        max_tokens=int(os.environ.get("XIAO_HEI_CLAUDE_MAX_TOKENS", "16000")),
        messages=[{"role": "user", "content": content}])
    if msg.stop_reason == "max_tokens":
        raise RuntimeError(
            f"reply hit max_tokens ({msg.usage.output_tokens} output tokens, "
            f"thinking counted in) and is truncated — raise "
            f"XIAO_HEI_CLAUDE_MAX_TOKENS rather than treating this as a bad "
            f"reply")
    return "".join(b.text for b in msg.content if b.type == "text")


_GENAI = None       # the one long-lived client; see `ask_gemini`


def ask_gemini(prompt: str, images: list[bytes], model: str,
               previous: bytes | None = None) -> str:
    from google import genai
    from google.genai import types

    key = os.environ.get("XIAO_HEI_GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("set XIAO_HEI_GEMINI_API_KEY (or GEMINI_API_KEY)")
    parts: list = []
    if previous is not None:
        parts.append("previous view of the target (from further away):")
        parts.append(types.Part.from_bytes(data=previous, mime_type="image/jpeg"))
    for i, raw in enumerate(images):
        parts.append(f"image {i} ({NAMES[i]}, heading {HEADINGS[i]}°):")
        parts.append(types.Part.from_bytes(data=raw, mime_type="image/jpeg"))
    parts.append(prompt)
    # Bound to a name, and kept: built inline as
    # `genai.Client(...).models.generate_content(...)` the client is a
    # temporary, and the SDK closes its httpx session when it is collected —
    # which happens before the request goes out. Every call raised
    # `RuntimeError: Cannot send a request, as the client has been closed`, so
    # `--backend gemini` had never once completed a call. Cached because a leg
    # makes ten of these and each new client is a fresh TLS handshake.
    global _GENAI
    if _GENAI is None:
        _GENAI = genai.Client(api_key=key, http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(
                attempts=int(os.environ.get("XIAO_HEI_API_MAX_RETRIES", "8")))))
    r = _GENAI.models.generate_content(
        model=model, contents=parts,
        # Matching `ask_claude`: v4 asks for every candidate and anchor with a
        # note each, and a truncated reply comes back as "unparseable", which
        # reads as a model failure and is a budget failure. Gemini bills its
        # thinking against this same ceiling, and the 3.x models think without
        # being asked — measured 676 thinking tokens against 251 of answer on
        # one `gemini-3.1-pro-preview` grounding call — so the budget has to
        # cover both or a correct answer never gets written.
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=int(os.environ.get(
                "XIAO_HEI_GEMINI_MAX_TOKENS", "8192"))))
    # `r.text` is None when nothing textual survived, which is what truncation
    # mid-thought looks like from here. Say which it was.
    reason = str(getattr(getattr(r, "candidates", [None])[0], "finish_reason",
                         "") or "") if getattr(r, "candidates", None) else ""
    if "MAX_TOKENS" in reason.upper():
        used = getattr(getattr(r, "usage_metadata", None),
                       "thoughts_token_count", None)
        raise RuntimeError(
            f"reply hit max_output_tokens and is truncated (thinking used "
            f"{used} tokens) — raise XIAO_HEI_GEMINI_MAX_TOKENS rather than "
            f"treating this as a bad reply")
    return r.text or ""


def parse(text: str) -> dict | None:
    """Pull the answer out, fence or no fence, one object or several.

    `re.search(r"\\{.*\\}")` used to do this, and it is greedy: it spans from
    the first brace to the last, so a reply containing two JSON objects yields
    one unparseable blob. That is not hypothetical. The approach block asks for
    "three more fields, appended to the JSON above", and on `studio` the model
    read "appended" as a second object — a complete answer with the passage's
    `gate` anchors in one, `here` / `target_state` /
    `same_object_as_previous` in the next. The leg died reporting an
    unparseable reply while holding a perfectly good one.

    So: decode every balanced object and merge them in order. A later object
    only fills a key that is missing or null, so a trailing fragment can add to
    the answer but never overwrite it.
    """
    dec = json.JSONDecoder()
    out: dict = {}
    i = 0
    while i < len(text):
        j = text.find("{", i)
        if j < 0:
            break
        try:
            obj, end = dec.raw_decode(text, j)
        except json.JSONDecodeError:
            i = j + 1
            continue
        i = end
        if isinstance(obj, dict):
            for k, v in obj.items():
                if out.get(k) is None:
                    out[k] = v
    return out or None


def report_relation(d: dict, geo_dir: Path) -> None:
    """Say — loudly — that `box_2d` is not the answer for a comparative phrase.

    On the first live relational run the model answered *the lantern closest to
    the fan decoration* with the fan decoration, so the loop stopped trusting
    the nomination and started measuring the candidates instead. This probe
    prints the reply, which means it prints the one field the loop discards. A
    human reading it forms an impression of a pipeline that does not exist,
    which is exactly how `potted plant furthest from the projector screen`
    looked broken while the loop was answering it correctly.
    """
    rel = d.get("relation")
    if rel not in ("closest_to", "farthest_from", "between"):
        return
    print(f"\n!! {rel}: the box above is the model's nomination, and the loop "
          f"does not use it.\n   It lifts every candidate and measures.",
          file=sys.stderr)

    geo = load_geometry(geo_dir)
    if geo is None:
        print(f"   No pose/scan in {geo_dir} — pass --scan-from DIR (a snap.sh "
              f"directory\n   or a recorded tour) to see what the loop would "
              f"actually pick.", file=sys.stderr)
        return

    # Imported here: vlm_approach imports this module, so a top-level import
    # would be circular.
    from vlm_approach import resolve_relation

    pose, scan = geo
    out = resolve_relation(d, scan, pose, size=640)
    if out is None:
        n_c, n_a = len(d.get("candidates") or []), len(d.get("anchors") or [])
        print(f"   Not measurable ({n_c} candidates, {n_a} anchors, and at "
              f"least two must lift)\n   — the loop would fall back to the "
              f"nomination above.", file=sys.stderr)
        return
    box, i, why = out
    if not out.complete:
        print(f"   {len(out.missed)} candidate(s) the lift could not place were "
              f"left out of this comparison — the winner below is over the "
              f"rest, not over all of them.", file=sys.stderr)
        for az, el, note in out.missed:
            print(f"     missed: {note[:60]} (az {az:+.0f}°, el {el:+.0f}°)",
                  file=sys.stderr)
    same = (i == d.get("image_index")
            and list(box) == list(to_pixels(d["box_2d"], d.get("coord_space"), 640)))
    print(f"   -> {'confirms' if same else 'OVERRIDES'} it: image {i} "
          f"({NAMES[i]}) box {list(box)}\n   {why}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot", help="directory written by scripts/snap.sh")
    ap.add_argument("phrase", help="the referring expression, verbatim")
    ap.add_argument("--backend", choices=["claude", "gemini"], default="claude")
    ap.add_argument("--model", default=None)
    ap.add_argument("--raw", action="store_true", help="print the reply as-is")
    ap.add_argument("--scan-from", default=None, metavar="DIR",
                    help="where to read pose+scan for resolving a comparative "
                         "relation, when the faces live apart from them")
    args = ap.parse_args()

    images = load_faces(Path(args.snapshot))
    prompt = build_prompt(args.phrase)
    model = args.model or ("claude-opus-5" if args.backend == "claude"
                           else DEFAULT_GEMINI_MODEL)
    print(f"backend={args.backend} model={model}\ntarget: {args.phrase!r}\n",
          file=sys.stderr)

    fn = ask_claude if args.backend == "claude" else ask_gemini
    text = fn(prompt, images, model)
    if args.raw:
        print(text)
        return 0

    d = parse(text)
    if d is None:
        print("could not parse JSON; raw reply follows\n", file=sys.stderr)
        print(text)
        return 1
    print(json.dumps(d, indent=2, ensure_ascii=False))

    if d.get("visible"):
        i = d.get("image_index")
        print(f"\n-> {NAMES[i]} (heading {HEADINGS[i]}°), "
              f"confidence {d.get('confidence')}", file=sys.stderr)
        space = d.get("coord_space")
        for key in ("box_2d", "feature_box_2d"):
            if not d.get(key):
                continue
            px = to_pixels(d[key], space, 640)
            yaw, pitch = bearing_deg(px, i)
            print(f"   {key:16s} yaw {yaw:+7.2f}°  pitch {pitch:+6.2f}°  "
                  f"[{space or 'inferred'}]", file=sys.stderr)
        report_relation(d, Path(args.scan_from or args.snapshot))
    else:
        e = d.get("explore") or {}
        print(f"\n-> NOT_VISIBLE; explore heading {e.get('heading_deg')}°",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
