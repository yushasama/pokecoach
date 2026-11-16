from __future__ import annotations

import random
from dataclasses import dataclass

from deepmew.agents import AgentSpec
from deepmew.opponents.mixture import OpponentMixture


@dataclass(frozen=True)
class SampledOpponent:
    index: int
    agent_spec: AgentSpec
    weight: float
    normalized_weight: float
    canonical_label: str
    deck_id: str | None = None
    deck_hash: str | None = None


class MixtureSampler:
    def __init__(self, mixture: OpponentMixture, *, seed: int | None = None):
        self.mixture = mixture.validate()
        self.seed = seed
        self._rng = random.Random(seed)
        self._weights = self.mixture.normalized_weights()

    def sample_one(self) -> SampledOpponent:
        threshold = self._rng.random()
        cumulative = 0.0
        selected = len(self.mixture.entries) - 1
        for index, weight in enumerate(self._weights):
            cumulative += weight
            if threshold <= cumulative:
                selected = index
                break
        entry = self.mixture.entries[selected]
        return SampledOpponent(
            index=selected,
            agent_spec=entry.agent_spec,
            weight=float(entry.weight),
            normalized_weight=self._weights[selected],
            canonical_label=entry.canonical_label,
            deck_id=entry.deck_id,
            deck_hash=entry.deck_hash,
        )

    def sample_sequence(self, count: int) -> list[SampledOpponent]:
        if count < 0:
            raise ValueError("sample count must be non-negative")
        return [self.sample_one() for _ in range(count)]
