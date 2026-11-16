from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from deepmew_ex.datasets.v2.learning.semantic import leaves
from deepmew_ex.datasets.v2.learning.trajectory import actor_sequences
from deepmew_ex.models.temporal import TemporalPolicy
from tests.datasets.v2.learning.test_admission import episode


def test_temporal_batch_matches_individual_context_losses_and_gradients() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(18)
    model = TemporalPolicy(hidden=16, embedding=4, layers=1, heads=2)
    sequences = list(actor_sequences(episode(), 'train'))
    sequences[0] = replace(sequences[0], events=sequences[0].events[:-1])
    for sequence, (_, actual) in zip(sequences, model.encode_batch(sequences), strict=True):
        _, expected = model.encode(sequence)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    expected_rows = [model.losses(sequence) for sequence in sequences]
    expected = torch.cat([row[0] for row in expected_rows])
    actual, identities = model.batch_losses(sequences)
    assert identities == [identity for row in expected_rows for identity in row[1]]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    expected.mean().backward()
    gradients = {name: value.grad.clone() for name, value in model.named_parameters() if value.grad is not None}
    model.zero_grad(set_to_none=True)
    actual.mean().backward()
    for name, value in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(value.grad, gradients[name], atol=3e-6, rtol=3e-5)


def test_checkpointed_record_chunks_preserve_full_losses_gradients_and_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    import deepmew_ex.models.temporal as temporal
    torch.set_num_threads(2)
    torch.manual_seed(24)
    model = TemporalPolicy(hidden=16, embedding=4, layers=1, heads=2)
    sequences = list(actor_sequences(episode(), 'train'))
    monkeypatch.setattr(temporal, 'ACTIVATION_LEAF_BUDGET', 10**9)
    expected, expected_ids = model.batch_losses(sequences)
    expected.mean().backward()
    gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters() if parameter.grad is not None}
    expected = expected.detach()
    model.zero_grad(set_to_none=True)
    monkeypatch.setattr(temporal, 'ACTIVATION_LEAF_BUDGET', 1)
    actual, actual_ids = model.batch_losses(sequences)
    assert actual_ids == expected_ids
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
    actual.mean().backward()
    for name, parameter in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(parameter.grad, gradients[name], atol=3e-6, rtol=3e-5)
    with torch.no_grad():
        inference, identities = model.batch_losses(sequences)
    assert identities == expected_ids
    torch.testing.assert_close(inference, expected, atol=3e-6, rtol=3e-6)


def test_batch_keeps_games_independent_and_future_hidden() -> None:
    torch.manual_seed(8)
    model = TemporalPolicy(hidden=16, embedding=4, layers=1, heads=2).eval()
    first, other = actor_sequences(episode(), 'train')
    records = list(first.records)
    for index in first.events[1:]:
        records[index] = list(leaves({'future': 'secret'}))
    changed = replace(first, records=tuple(records))
    corrupted_other = replace(other, records=tuple(list(leaves({'other_game': 'unseen'})) for _ in other.records))
    baseline = model.encode_batch([first, other])[0][1][0]
    actual = model.encode_batch([changed, corrupted_other])[0][1][0]
    torch.testing.assert_close(actual, baseline, atol=2e-6, rtol=2e-6)


def test_temporal_batch_rejects_history_free_records() -> None:
    model = TemporalPolicy(hidden=16, embedding=4, layers=1, heads=2)
    sequence = replace(actor_sequences(episode(), 'train')[0], history_available=False)
    with pytest.raises(ValueError, match='causal histories'):
        model.batch_losses([sequence])


def test_ordered_stage_conditions_next_stage_in_the_same_order_as_inference() -> None:
    torch.manual_seed(19)
    model = TemporalPolicy(hidden=16, embedding=4, layers=1, heads=2)
    sequence = actor_sequences(episode(), 'train')[1]
    prediction = sequence.predictions[1]
    first = replace(prediction.stages[0], ordered=True, selected=(1, 0))
    second = replace(first, minimum=1, maximum=1, selected=(0,))
    compound = replace(prediction, stages=(first, second))
    sequence = replace(sequence, predictions=(compound,))
    records, contexts = model.encode(sequence)
    state = contexts[compound.event_index]
    options = records[list(first.options)]
    advanced = model.decoder(options[0], model.decoder(options[1], state))
    expected = model.stage_loss(state, options, first) + model.stage_loss(advanced, options, second)
    individual, _ = model.losses(sequence)
    batched, _ = model.batch_losses([sequence])
    torch.testing.assert_close(individual[0], expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(batched[0], expected, atol=2e-6, rtol=2e-6)
