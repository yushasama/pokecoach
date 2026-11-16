"""Torch-free game records shipped between farm workers and the trainer."""

from __future__ import annotations

from dataclasses import dataclass

from deepmew_ex.datasets.v2.learning.leaf_tables import TABLES_ATTRIBUTE, LeafTables, leaf_tables
from deepmew_ex.datasets.v2.learning.semantic import Leaf
from deepmew_ex.datasets.v2.learning.trajectory import ActorSequence, Prediction, StageTarget


def tables_of(records: list[list[Leaf]]) -> LeafTables | None:
    """Worker-side encoder input for a batch of new records (``None`` if there are none)."""
    return leaf_tables(tuple(records)) if records else None


@dataclass(frozen=True)
class Decision:
    event_index: int
    stages: tuple[StageTarget, ...]
    logp: float
    value: float
    entropy: float | None


@dataclass(frozen=True)
class Trajectory:
    """One actor seat of one finished native game, compact enough to ship between processes."""
    job: dict
    tables: LeafTables
    record_count: int
    events: tuple[int, ...]
    decisions: tuple[Decision, ...]
    reward: float
    stats: dict

    def sequence(self) -> ActorSequence:
        predictions = tuple(Prediction(f'rl-{index}', decision.event_index, decision.stages, '', True) for index, decision in enumerate(self.decisions))
        # Records are represented by their exact encoder tables; the encoder
        # reads only their count from ``records`` when tables are attached.
        result = ActorSequence('rl-native-game', int(self.job['seat']), 'rl_league', 'train', ((),) * self.record_count, self.events, predictions, (), (), True)
        object.__setattr__(result, TABLES_ATTRIBUTE, self.tables)
        return result
