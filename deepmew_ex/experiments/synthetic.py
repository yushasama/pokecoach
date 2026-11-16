"""Synthetic V1-shaped rows only; never reads or reconstructs replays."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterator, Protocol

import numpy as np

from deepmew_ex.data import Decision, Split


class DatasetAdapter(Protocol):
    def decisions(self, split: Split) -> Iterator[Decision]: ...


class SyntheticDataset:
    version = 'synthetic_v1_ragged_v1'

    def __init__(self, seed: int, train_count: int, validation_count: int) -> None:
        self.seed = seed
        self.counts = {'train': train_count, 'validation': validation_count}
        self.reads: list[str] = []

    def decisions(self, split: Split) -> Iterator[Decision]:
        if split not in self.counts:
            raise ValueError('holdout is sealed and has no S0 reader')
        self.reads.append(split)
        rng = np.random.default_rng(self.seed + (0 if split == 'train' else 100000))
        for index in range(self.counts[split]):
            count = 2 + index % 5
            options = rng.normal(size=(count, 304)).astype(np.float32)
            values = rng.normal(size=144).astype(np.float32)
            target = int(np.argmax(options[:, 0] + values[0] * options[:, 1]))
            decision = Decision(f'{split}-row-{index}', f'{split}-game-{index // 4}', f'{split}-trajectory-{index // 4}', split, options, values, np.array([target], dtype=np.int32), 1, 1, count, float(index % 2), 0.5 + index % 3, 'segmented', index % 4)
            if index == 0:
                decision = replace(decision, actor_loss_weight=0.0, option_features=options[:1], selection=np.array([0], dtype=np.int32), complete_legal_action_count=1)
            elif index == 1:
                decision = replace(decision, actor_loss_weight=0.0)
            elif index == 2:
                decision = replace(decision, max_count=2, selection=np.array([0, 1], dtype=np.int32))
            elif index == 3:
                decision = replace(decision, min_count=0, selection=np.array([], dtype=np.int32))
            yield decision
