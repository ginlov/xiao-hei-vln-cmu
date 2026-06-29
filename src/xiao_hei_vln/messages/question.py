"""Inbound `/challenge_question` (std_msgs/String) and its classification.

The challenge has three question categories; the response topic is
determined by which category the question falls into. We start with the
same keyword heuristic as the reference `dummyVLM.cpp` and leave the
classifier swappable for a learned model later.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from xiao_hei_vln.messages.common import Stamp


class QuestionType(StrEnum):
    NUMERICAL = "numerical"
    OBJECT_REFERENCE = "object_reference"
    INSTRUCTION_FOLLOWING = "instruction_following"


def classify_question(text: str) -> QuestionType:
    """Heuristic classifier for the three challenge categories.

    "How many ..." / "Count ..." → numerical; "Find ..." or a bare "The ..."
    noun phrase → object reference; anything else → instruction following.
    ("Count the number of chairs with pillows on them." is an official
    numerical item, so a leading "Count" routes to numerical.)

    The leading-"The" case matters: 3 of the 30 official object_reference
    questions drop the imperative and read "The red pillow closest to the
    sushi." / "The blue chair that is closest to ...". The official
    instruction_following questions never start with "the" — they all begin
    with an action verb (Go / First / Take) — so routing a leading "the" to
    object_reference is unambiguous and avoids misclassifying those items.
    """
    head = text.lstrip().lower()
    if head.startswith("how many") or head.startswith("count "):
        return QuestionType.NUMERICAL
    if head.startswith("find") or head.startswith("the "):
        return QuestionType.OBJECT_REFERENCE
    return QuestionType.INSTRUCTION_FOLLOWING


class ChallengeQuestion(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    type: QuestionType
    received_at: Stamp

    @classmethod
    def from_text(cls, text: str, received_at: Stamp) -> ChallengeQuestion:
        return cls(text=text, type=classify_question(text), received_at=received_at)
