"""Deterministic complete-sequence batches; budgets never truncate a history."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from deepmew_ex.datasets.v2.learning.trajectory import ActorSequence


def temporal_batches(sequences: Iterable[ActorSequence], maximum_sequences: int, maximum_leaves: int, maximum_attention_cells: int) -> Iterator[list[ActorSequence]]:
    if min(maximum_sequences, maximum_leaves, maximum_attention_cells) < 1:
        raise ValueError('positive batch budgets required')
    group: list[ActorSequence] = []
    leaves, longest = 0, 0
    for sequence in sequences:
        if not sequence.history_available or not sequence.events:
            raise ValueError('complete temporal histories required')
        size = sum(len(record) for record in sequence.records)
        width = max(longest, len(sequence.events))
        if group and (len(group) == maximum_sequences or leaves + size > maximum_leaves or (len(group) + 1) * width**2 > maximum_attention_cells):
            yield group
            group, leaves, longest = [], 0, 0
        # An oversized single game stays whole. The runtime memory guard may
        # reject it; the batcher never silently discards it or shortens context.
        group.append(sequence)
        leaves += size
        longest = max(longest, len(sequence.events))
    if group:
        yield group
