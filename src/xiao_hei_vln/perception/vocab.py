"""Open-vocabulary class list management for the perception responder.

YOLOv8x-World v2 accepts a different class list per call — but
re-encoding the prompt embeddings takes ~50 ms on a 4090 for ~50
classes. We keep a stable hybrid list:

* A small **prior** of common indoor objects so the scene graph
  accumulates labels between questions (object referenced *next*
  tick).
* **Question-derived nouns** merged on top, so the detector sees the
  task-specific vocabulary too.

The class list is **dedup-stable** so the sidecar's prompt embedding
cache stays warm: the same set of classes always produces the same
ordered tuple, and the responder only calls ``/reload_classes`` when
the set actually changes.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

log = logging.getLogger(__name__)

# Pinned to the arabic_room ground-truth object labels (from the scene's
# object_list.txt) so the open-vocab class list matches what's actually
# in the scene — used to evaluate detector performance against GT
# without vocabulary mismatch. The "unknown" GT placeholder is dropped.
DEFAULT_PRIOR: tuple[str, ...] = (
    "focus light",
    "pillow",
    "potted plant",
    "wall",
    "window",
    "wall lamp",
    "door",
    "glass",
    "sofa",
    "picture",
    "floor",
    "column",
    "vase",
    "lantern",
    "table",
    "carpet",
    "stool",
    "door frame",
    "ceiling lamp",
    "shoes",
    "exterior walls",
    "arabic jar",
    "book",
    "coffee pot",
    "hookah",
    "hookah wire",
    "tray",
    "ceiling",
)

# Words to drop from a question before treating the rest as candidate
# noun labels. Kept conservative — YOLO-World ignores junk classes
# anyway (they just get low scores), so over-removing is fine but
# under-removing rarely hurts.
_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "from", "for", "with", "by", "and", "or",
    "this", "that", "these", "those", "it", "its", "there", "here",
    "you", "your", "yours", "i", "my", "mine", "we", "our", "ours",
    "how", "what", "which", "where", "when", "who", "whose", "why",
    "many", "much", "some", "any", "more", "most", "less", "least",
    "can", "could", "would", "should", "do", "does", "did", "have",
    "has", "had", "find", "go", "show", "tell", "near", "next",
    "between", "above", "below", "behind", "front", "left", "right",
})


class Vocabulary:
    """Hybrid scene-prior + question-derived class list.

    ``current_classes(question)`` is the only method callers need —
    it returns a stable, deduplicated tuple that the
    :class:`HTTPPerceptionClient` can compare against the last list it
    pushed to the sidecar.
    """

    def __init__(
        self,
        prior: Iterable[str] = DEFAULT_PRIOR,
        *,
        stopwords: frozenset[str] = _STOPWORDS,
    ) -> None:
        self._prior: tuple[str, ...] = tuple(_normalise(p) for p in prior)
        self._stopwords = stopwords

    @property
    def prior(self) -> tuple[str, ...]:
        return self._prior

    def current_classes(self, question_text: str | None) -> tuple[str, ...]:
        """Return the (prior ∪ question-derived) class list for the
        current tick. Order: prior first (preserving its order), then
        question-derived nouns in question order, deduped.
        """
        seen: dict[str, None] = {label: None for label in self._prior}
        if question_text:
            for label in self._extract_nouns(question_text):
                if label not in seen:
                    seen[label] = None
        return tuple(seen.keys())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_nouns(self, text: str) -> list[str]:
        """Pull plausible object labels out of a question.

        v1 heuristic: lowercase, strip punctuation, drop stopwords,
        strip trailing plural ``s`` (so ``"chairs"`` becomes
        ``"chair"`` and matches the prior). Multi-word concepts can be
        added by the caller via ``Vocabulary(prior=...)``; the
        extractor itself emits single words only.
        """
        words = re.findall(r"[a-z]+", text.lower())
        out: list[str] = []
        for w in words:
            if w in self._stopwords or len(w) < 2:
                continue
            singular = _depluralise(w)
            if singular not in out:
                out.append(singular)
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise(label: str) -> str:
    """Lower-case and collapse whitespace. Keeps internal hyphens."""
    return re.sub(r"\s+", " ", label.strip().lower())


def _depluralise(word: str) -> str:
    """Strip a trailing ``s``/``es`` when the result is still ≥3 chars.

    Conservative on purpose — we only handle the common case
    (``chairs`` → ``chair``, ``boxes`` → ``boxe`` is undesirable so we
    leave words ending in -ses/-xes/-zes/-shes/-ches alone). YOLO-World
    forgives the rest.
    """
    if len(word) < 4:
        return word
    if word.endswith(("ses", "xes", "zes", "shes", "ches")):
        return word
    if word.endswith("s"):
        return word[:-1]
    return word
