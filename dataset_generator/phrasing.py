"""Shared question-phrasing transforms used to match the official question set.

The CMU-VLN official object_reference questions are not all "Find the X …":
~10% drop the imperative and read "The X …" (e.g. "The red pillow closest to
the sushi."). These helpers apply that variation deterministically.
"""

from __future__ import annotations

import random
import re

# ── Colour vocabulary (shared by ref and num generators) ─────────────────────
# A colour modifier should name what the object actually looks like: keep it
# only when it is the object's DOMINANT colour at >= this fraction. Map VLA-3D's
# technical palette to the basic words the human-authored official set uses
# ("the red pillow", which VLA-3D labels "maroon").
COLOR_DOMINANT_MIN = 0.40
COLOR_BASIC_MAP = {
    "maroon": "red", "navy": "blue", "teal": "blue", "aqua": "blue",
    "olive": "green", "beige": "brown", "tan": "brown", "violet": "purple",
    "gold": "red", "golden": "red", "silver": "gray",
}


def basic_color(word: str) -> str:
    """Map a VLA-3D colour word to the official basic-colour vocabulary."""
    return COLOR_BASIC_MAP.get(word.lower(), word.lower())


def dominant_color_or_none(colors: tuple, percentages: tuple, used: str) -> str | None:
    """If `used` is the object's dominant colour covering >= COLOR_DOMINANT_MIN,
    return its basic-word form; else None (weak / non-dominant modifier)."""
    if not used:
        return None
    cols = [c.lower() for c in colors]
    u = used.lower()
    if not cols or cols[0] != u or percentages[0] < COLOR_DOMINANT_MIN:
        return None
    return basic_color(u)


def to_omit_find(question: str) -> str | None:
    """'Find the X ...' -> 'The X ...'. Returns None if not a 'Find' imperative
    (e.g. numerical 'How many …' questions, which are left untouched)."""
    if question.startswith("Find the "):
        return "The " + question[len("Find the "):]
    # Only "Find the X" → "The X". We deliberately do NOT rewrite "Find a X"
    # to "A X": a leading "A" is routed to instruction_following by the
    # classifier (and the official omit-Find items all start with "The").
    return None


def to_count_phrasing(question: str) -> str | None:
    """'How many X are on the Y?' -> 'Count the number of X on the Y.'
    Mirrors the official "Count the number of chairs with pillows on them."
    Returns None if the question isn't a "How many … are …?" form."""
    m = re.match(r"How many (.+?) are (.+)\?$", question)
    if m is None:
        return None
    return f"Count the number of {m.group(1)} {m.group(2)}."


def apply_count_phrasing(pairs: list[dict], rng: random.Random, frac: float = 0.07) -> int:
    """In-place: rephrase a `frac` fraction of numerical pairs to the
    "Count the number of …" form. Returns the number actually rephrased."""
    num_idx = [i for i, p in enumerate(pairs) if p.get("type") == "numerical"]
    k = int(round(len(num_idx) * frac))
    if k <= 0:
        return 0
    n = 0
    for i in rng.sample(num_idx, min(k, len(num_idx))):
        alt = to_count_phrasing(pairs[i]["question"])
        if alt is not None:
            pairs[i]["question"] = alt
            n += 1
    return n


def apply_omit_find(pairs: list[dict], rng: random.Random, frac: float = 0.10) -> int:
    """In-place: rephrase a `frac` fraction of object_reference pairs to drop
    the leading "Find". Returns the number actually rephrased."""
    ref_idx = [i for i, p in enumerate(pairs)
               if p.get("type") == "object_reference"]
    k = int(round(len(ref_idx) * frac))
    if k <= 0:
        return 0
    chosen = rng.sample(ref_idx, min(k, len(ref_idx)))
    n = 0
    for i in chosen:
        alt = to_omit_find(pairs[i]["question"])
        if alt is not None:
            pairs[i]["question"] = alt
            n += 1
    return n
