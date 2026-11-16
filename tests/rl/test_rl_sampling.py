from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
import torch

from deepmew_ex.datasets.v2.learning.trajectory import Prediction, StageInput, StageTarget, actor_sequences
from deepmew_ex.models.temporal import TemporalPolicy, subset_nll
from deepmew_ex.rl.ppo import ValueHead, policy_terms
from deepmew_ex.rl.sampling import draw, sample_subset, subset_logp
from tests.datasets.v2.learning.test_admission import episode
from tests.rl.test_rl_incremental import synthetic


def test_subset_logp_matches_training_nll_and_sampler_matches_distribution() -> None:
    logits = np.array([0.4, -1.1, 0.9, 0.0])
    counts = np.array([0.3, -0.2, 0.5, 0.1])
    minimum, maximum = 1, 3
    subsets = [choice for size in range(minimum, maximum + 1) for choice in itertools.combinations(range(4), size)]
    exact = {choice: subset_logp(logits, counts, choice, minimum, maximum) for choice in subsets}
    for choice, value in exact.items():
        expected = -subset_nll(torch.tensor(logits), choice, minimum, maximum, torch.tensor(counts))
        assert value == pytest.approx(float(expected), abs=1e-9)
    assert sum(np.exp(list(exact.values()))) == pytest.approx(1.0, abs=1e-9)
    rng = np.random.default_rng(0)
    draws = 40000
    seen = Counter(sample_subset(logits, counts, minimum, maximum, rng) for _ in range(draws))
    for choice, value in exact.items():
        probability = np.exp(value)
        assert abs(seen[choice] / draws - probability) < 4 * np.sqrt(probability * (1 - probability) / draws) + 1e-3


def model() -> TemporalPolicy:
    torch.manual_seed(11)
    return TemporalPolicy(hidden=32, embedding=8, layers=1, heads=4).eval()


def decisions(sequence):
    """(event index, StageInput tuple) for every decision of a recorded sequence."""
    return [(prediction.event_index, tuple(StageInput(stage.options, stage.minimum, stage.maximum, stage.ordered, stage.repeated) for stage in prediction.stages)) for prediction in sequence.predictions]


def structured_cases() -> list:
    """Recorded optional/multi decisions plus synthetic ordered, repeated and multi-stage ones."""
    sequence = synthetic(3, 12)
    options = [index for index in range(len(sequence.records)) if index not in sequence.events][:6]
    cases = [(5, (StageInput(tuple(options[:4]), 0, 2, True, False),)),
             (6, (StageInput(tuple(options[:3]), 1, 3, True, True),)),
             (7, (StageInput(tuple(options[:5]), 0, 3, False, False), StageInput(tuple(options[2:6]), 1, 1, False, False))),
             (8, (StageInput(tuple(options[:2]), 2, 2, False, False),)),
             (9, (StageInput(tuple(options[:4]), 1, 1, False, False),))]
    return [(sequence, cases)] + [(recorded, decisions(recorded)) for recorded in actor_sequences(episode(), 'train')]


@pytest.mark.parametrize('greedy', [False, True])
def test_draw_logp_equals_training_likelihood_and_policy_terms(greedy: bool) -> None:
    policy, head = model(), ValueHead(32)
    rng = np.random.default_rng(3)
    for sequence, cases in structured_cases():
        with torch.no_grad():
            records, context = policy.encode(sequence)
        for repeat in range(6):
            predictions, drawn = [], []
            for event_index, stages in cases:
                result = draw(policy, context[event_index], records, stages, rng, greedy=greedy)
                drawn.append(result)
                if greedy:
                    assert result.selected == policy.decode(sequence, event_index, stages)
                targets = tuple(StageTarget(stage.options, selected, stage.minimum, stage.maximum, stage.ordered, stage.repeated) for stage, selected in zip(stages, result.selected, strict=True))
                predictions.append(Prediction(f'p{len(predictions)}', event_index, targets, '', True))
            target_sequence = replace(sequence, predictions=tuple(predictions))
            with torch.no_grad():
                nll, _ = policy.losses(target_sequence)
                logp, entropy, values = policy_terms(policy, head, [target_sequence])
            np.testing.assert_allclose([row.logp for row in drawn], (-nll).numpy(), atol=2e-5)
            np.testing.assert_allclose([row.logp for row in drawn], logp.numpy(), atol=2e-5)
            for row, value in zip(drawn, entropy.numpy(), strict=True):
                assert (row.entropy is None) == bool(np.isnan(value))
                if row.entropy is not None:
                    assert row.entropy == pytest.approx(float(value), abs=2e-5)
            assert values.shape == (len(cases),)
            if greedy:
                break


def test_policy_terms_gradients_are_finite() -> None:
    policy, head = model(), ValueHead(32)
    sequence = actor_sequences(episode(), 'train')[1]
    logp, entropy, values = policy_terms(policy, head, [sequence])
    (logp.sum() + values.sum() + torch.nan_to_num(entropy).sum()).backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in policy.parameters() if parameter.grad is not None)
