from __future__ import annotations

import itertools
from dataclasses import replace

import pytest
import torch

from deepmew_ex.datasets.v2.learning.semantic import leaves
from deepmew_ex.datasets.v2.learning.trajectory import actor_sequences
from deepmew_ex.models.temporal import ByteLeafEncoder, TemporalPolicy, subset_nll
from tests.datasets.v2.learning.test_admission import episode


def test_subset_likelihood_matches_enumeration_and_normalizes() -> None:
    logits = torch.tensor([0.3, -0.9, 1.2], requires_grad=True)
    count_scores = torch.tensor([0.4, 0.1, -0.2])
    choices = [choice for count in range(3) for choice in itertools.combinations(range(3), count)]
    scores = torch.stack([logits[list(choice)].sum() + count_scores[len(choice)] for choice in choices])
    expected = scores.logsumexp(0) - scores
    actual = torch.stack([subset_nll(logits, choice, 0, 2, count_scores) for choice in choices])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close((-actual).exp().sum(), torch.tensor(1.0))
    actual.mean().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_whole_actor_prefix_preserves_optional_multi_and_forced_context() -> None:
    sequences = actor_sequences(episode(), 'train')
    assert [len(row.predictions) for row in sequences] == [2, 2]
    assert sequences[1].predictions[0].stages[0].selected == ()
    assert len(sequences[1].predictions[1].stages[0].selected) == 2
    for sequence in sequences:
        assert len(sequence.preserved_transition_ids) == len(sequence.predictions)
        assert sequence.predictions[-1].event_index > sequence.predictions[0].event_index
        assert 'PRIVATE' not in str(sequence.records)


def test_current_action_future_and_other_actor_cannot_change_earlier_prediction() -> None:
    torch.manual_seed(9)
    model = TemporalPolicy(hidden=16, embedding=4, layers=1, heads=2).eval()
    sequence = actor_sequences(episode(), 'train')[0]
    _, baseline = model.encode(sequence)
    records = list(sequence.records)
    current_index = sequence.predictions[0].event_index
    for index in sequence.events[current_index + 1:]:
        records[index] = list(leaves({'unseen_future': 'future-sentinel'}))
    changed = replace(sequence, records=tuple(records))
    _, modified = model.encode(changed)
    torch.testing.assert_close(baseline[current_index], modified[current_index], atol=1e-6, rtol=1e-6)
    prefix = replace(sequence, events=sequence.events[:current_index + 1])
    _, prefix_value = model.encode(prefix)
    torch.testing.assert_close(baseline[current_index], prefix_value[-1], atol=1e-6, rtol=1e-6)


def test_utf8_encoder_distinguishes_unseen_values_without_vocab() -> None:
    torch.manual_seed(3)
    encoder = ByteLeafEncoder(16, 4)
    records = (list(leaves({'unseen_card': 'new-card-1'})), list(leaves({'unseen_card': 'new-card-2'})))
    output = encoder(records)
    assert not torch.equal(output[0], output[1])
    assert torch.isfinite(output).all()


def test_real_gradient_updates_empty_multi_and_ordered_targets() -> None:
    torch.manual_seed(7)
    model = TemporalPolicy(hidden=16, embedding=4, layers=1, heads=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    sequence = actor_sequences(episode(), 'train')[1]
    before = model.encoder.bytes.weight.detach().clone()
    losses, ids = model.losses(sequence)
    assert len(ids) == 2 and torch.isfinite(losses).all()
    losses.mean().backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
    optimizer.step()
    assert not torch.equal(before, model.encoder.bytes.weight)
    state = torch.randn(16)
    options = torch.randn(3, 16)
    stage = replace(sequence.predictions[1].stages[0], minimum=0, maximum=2, ordered=True)
    choices = [choice for length in range(3) for choice in itertools.permutations(range(3), length)]
    probabilities = [(-model.stage_loss(state, options, replace(stage, selected=choice))).exp() for choice in choices]
    torch.testing.assert_close(torch.stack(probabilities).sum(), torch.tensor(1.0), atol=1e-5, rtol=1e-5)


def test_snapshot_batch_matches_individual_losses_and_drops_no_remainder() -> None:
    model = TemporalPolicy(hidden=16, embedding=4, layers=1, heads=2).eval()
    temporal = actor_sequences(episode(), 'train')[0]
    prediction = replace(temporal.predictions[0], event_index=0)
    snapshot = replace(temporal, events=(temporal.events[0],), predictions=(prediction,), history_available=False)
    loss, ids = model.snapshot_losses([snapshot, snapshot, snapshot])
    expected, _ = model.losses(snapshot)
    torch.testing.assert_close(loss, expected.expand(3), atol=1e-6, rtol=1e-6)
    assert len(ids) == 3
    from deepmew_ex.experiments.v2.full import batches
    assert [len(batch) for batch in batches(iter([snapshot] * 5), 2)] == [2, 2, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware unavailable')
def test_cuda_structured_gradient_update() -> None:
    model = TemporalPolicy(hidden=16, embedding=4, layers=1, heads=2).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    sequence = actor_sequences(episode(), 'train')[1]
    losses, ids = model.losses(sequence)
    losses.mean().backward()
    assert len(ids) == 2 and torch.isfinite(losses).all()
    optimizer.step()
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
