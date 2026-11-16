from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from deepmew_ex.evaluation.comparisons import Comparison
from deepmew_ex.experiments.config import ExperimentConfig


def comparison() -> Comparison:
    control = asdict(ExperimentConfig())
    identities = {'dataset': 'a' * 64, 'train_split': 'b' * 64, 'validation_split': 'c' * 64, 'seed': 42, 'budget': [64, 16, 32], 'checkpoint_selection': control['checkpoint_selection'], 'evaluator': control['evaluator'], 'source_population': 'synthetic_only', 'legal_action_handling': control['objective'], 'device': 'cpu'}
    return Comparison('valid-width-contrast', 'control-fixture', 'treatment-fixture-not-run', 'one_factor_ablation', ('hidden_dim',), control, {**control, 'hidden_dim': 32, 'variant': 'snapshot_bc_s0_width32_v1'}, identities, identities.copy())


def test_valid_declared_factor_and_invalid_two_factor() -> None:
    valid = comparison()
    assert valid.validate()['status'] == 'passed'
    invalid = replace(valid, treatment={**valid.treatment, 'learning_rate': 0.0001})
    with pytest.raises(ValueError, match='undeclared'):
        invalid.validate()
    assert replace(invalid, kind='practical_system_comparison', declared_factors=('hidden_dim', 'learning_rate')).validate()['status'] == 'passed'
    with pytest.raises(ValueError, match='one-factor'):
        replace(invalid, declared_factors=('hidden_dim', 'learning_rate')).validate()


@pytest.mark.parametrize('factor', ['dataset', 'train_split', 'seed', 'budget', 'checkpoint_selection', 'evaluator'])
def test_frozen_comparison_controls(factor: str) -> None:
    valid = comparison()
    with pytest.raises(ValueError, match='identities changed'):
        replace(valid, treatment_identities={**valid.frozen_identities, factor: 'changed'}).validate()
