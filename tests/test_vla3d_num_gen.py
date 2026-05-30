"""Regression tests for the numerical-generator's RNG threading.

The bug we guard against: ``emit_refusal`` used to call the module-level
``random.shuffle(...)`` and the module set ``random.seed(42)`` at import
time. That made the byte-identical-jsonl promise depend on import order
and on no other code mutating the global RNG between import and the call.
The fix threads ``rng = random.Random(seed)`` through ``main`` →
``generate_scene`` → ``emit_refusal``; this test pins that behavior.
"""

from __future__ import annotations

import random

from vla3d_loader import VLAScene
from vla3d_num_gen import MAX_REFUSALS_PER_SCENE, emit_refusal


def _empty_scene() -> VLAScene:
    """Scene with no objects → every refusal candidate is 'available'."""
    return VLAScene(name="empty_scene", objects=[])


def _questions(pairs: list[dict]) -> list[str]:
    return [p["question"] for p in pairs]


class TestEmitRefusalDeterminism:
    def test_same_rng_seed_same_output(self) -> None:
        sc = _empty_scene()
        a = emit_refusal(sc, random.Random(42))
        b = emit_refusal(sc, random.Random(42))
        assert _questions(a) == _questions(b)
        assert len(a) == MAX_REFUSALS_PER_SCENE

    def test_global_random_state_does_not_leak_in(self) -> None:
        """The whole point of threading rng: callers must not be able to
        change the output by twiddling the global RNG."""
        sc = _empty_scene()

        random.seed(0)
        baseline = _questions(emit_refusal(sc, random.Random(42)))

        random.seed(12345)  # mutate global state
        random.shuffle(list(range(100)))  # and consume some of it
        after = _questions(emit_refusal(sc, random.Random(42)))

        assert baseline == after

    def test_different_rng_seeds_produce_different_output(self) -> None:
        """Sanity: the rng is actually being consumed (not a no-op).

        Both calls draw from the same 21-candidate pool but emit only the
        first ``MAX_REFUSALS_PER_SCENE`` after shuffling, so different seeds
        can produce both different orderings AND different subsets — we
        only need to assert the result is not byte-identical.
        """
        sc = _empty_scene()
        a = _questions(emit_refusal(sc, random.Random(42)))
        b = _questions(emit_refusal(sc, random.Random(7)))
        assert a != b
