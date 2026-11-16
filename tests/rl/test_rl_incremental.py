from __future__ import annotations

import random

import pytest
import torch

from deepmew_ex.datasets.v2.learning.semantic import leaves
from deepmew_ex.datasets.v2.learning.trajectory import ActorSequence, actor_sequences
from deepmew_ex.models.temporal import TemporalPolicy
from deepmew_ex.rl.incremental import CachePool, RecordEncoder, tables_of
from tests.datasets.v2.learning.test_admission import episode


def synthetic(seed: int, events: int) -> ActorSequence:
    """A live-shaped prefix: request records, option records (not events) and past actions."""
    rng = random.Random(seed)
    records, order = [], []
    for step in range(events):
        order.append(len(records))
        records.append(list(leaves({'state': {'turn': step, 'hp': [rng.randint(0, 300) for _ in range(rng.randint(1, 5))], 'card': f'card-{rng.randint(0, 40)}'}, 'request': {'kind': rng.choice(['attack', 'play', 'attach'])}})))
        for _ in range(rng.randint(0, 3)):
            records.append(list(leaves({'option': {'card': f'card-{rng.randint(0, 40)}', 'slot': rng.randint(0, 5)}})))
    return ActorSequence('synthetic', 0, 'test', 'evaluation', tuple(records), tuple(order), (), (), (), True)


def model(seed: int = 0) -> TemporalPolicy:
    torch.manual_seed(seed)
    return TemporalPolicy(hidden=32, embedding=8, layers=2, heads=4).eval()


def feed(policy: TemporalPolicy, sequences: list[ActorSequence], seed: int, encoder: RecordEncoder | None = None) -> CachePool:
    """Grow several games' caches together, in random-sized steps like a live farm.

    The pool starts with too few slots/events/records so every growth path runs.
    """
    rng = random.Random(seed)
    pool = CachePool(policy, slots=2, events=4, records=8, encoder=encoder)
    sent_records = [0] * len(sequences)
    sent_events = [0] * len(sequences)
    while any(sent < len(sequence.events) for sent, sequence in zip(sent_events, sequences)):
        new_records, new_events = [], []
        for position, sequence in enumerate(sequences):
            step = min(rng.randint(0, 3), len(sequence.events) - sent_events[position])
            events = list(sequence.events[sent_events[position]:sent_events[position] + step])
            # Records up to the last new event (plus trailing option records) arrive with it.
            upto = len(sequence.records) if sent_events[position] + step == len(sequence.events) else (sequence.events[sent_events[position] + step] if step else sent_records[position])
            new_records.append(list(sequence.records[sent_records[position]:upto]))
            new_events.append(events)
            sent_records[position], sent_events[position] = max(upto, sent_records[position]), sent_events[position] + step
        pool.extend([(position,) for position in range(len(sequences))], [tables_of(records) for records in new_records], new_events)
    return pool


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA'))])
def test_incremental_cache_matches_full_prefix_forward(device: str) -> None:
    policy = model().to(device)
    sequences = [synthetic(1, 17), synthetic(2, 5), synthetic(3, 30), *actor_sequences(episode(), 'train')]
    encoder = RecordEncoder(policy, max_rows=120)  # small caches: exercises the reset path too
    pool = feed(policy, sequences, seed=4, encoder=encoder)
    assert len(pool.record_counts) >= len(sequences)
    assert encoder.stats['resets'] > 0 and encoder.stats['record_misses'] < encoder.stats['records']  # both cache paths ran
    with torch.no_grad():
        for position, sequence in enumerate(sequences):
            records, context = policy.encode(sequence)
            torch.testing.assert_close(pool.game_records((position,)), records, atol=5e-5, rtol=1e-4)
            torch.testing.assert_close(pool.game_context((position,)), context, atol=5e-5, rtol=1e-4)


def test_incremental_cache_matches_padded_training_forward_and_reuses_slots() -> None:
    policy = model(5)
    sequences = [synthetic(7, 12), synthetic(8, 3), synthetic(9, 21)]
    pool = feed(policy, sequences, seed=6)
    with torch.no_grad():
        flat, context, lengths = policy.encode_padded(sequences)
    for index in range(len(sequences)):
        torch.testing.assert_close(pool.game_context((index,)), context[index, :lengths[index]], atol=5e-5, rtol=1e-4)
    pool.release((0,))
    pool.extend([('fresh',)], [tables_of([list(leaves({'a': 1}))])], [[0]])
    assert pool.event_count(('fresh',)) == 1 and pool.game_records(('fresh',)).shape[0] == 1


def test_incremental_rejects_events_beyond_cached_records() -> None:
    pool = CachePool(model())
    with pytest.raises(IndexError):
        pool.extend([('game',)], [tables_of([list(leaves({'a': 1}))])], [[1]])
