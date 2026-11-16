from __future__ import annotations

from dataclasses import replace

from deepmew_ex.datasets.v2.learning.trajectory import actor_sequences
from deepmew_ex.training.batching import temporal_batches
from tests.datasets.v2.learning.test_admission import episode


def test_batch_limits_preserve_every_history_and_remainder() -> None:
    sequences = [sequence for index in range(3) for sequence in actor_sequences(episode(index), 'train')]
    groups = list(temporal_batches(sequences, 4, 10**8, 10**8))
    assert [len(group) for group in groups] == [4, 2]
    assert [row for group in groups for row in group] == sequences
    groups = list(temporal_batches(sequences, 4, 1, 1))
    assert all(len(group) == 1 for group in groups)
    assert [row for group in groups for row in group] == sequences


def test_attention_budget_counts_padded_cells() -> None:
    sequence = actor_sequences(episode(), 'train')[0]
    long = replace(sequence, events=sequence.events * 10)
    groups = list(temporal_batches([sequence, long, sequence], 3, 10**8, len(long.events)**2))
    assert [len(group) for group in groups] == [1, 1, 1]
