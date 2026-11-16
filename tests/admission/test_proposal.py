from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from deepmew_ex.experiments.config import PROJECT_ROOT, load_config


def test_proposed_s1_is_frozen_and_cannot_execute() -> None:
    path = PROJECT_ROOT / 'experiments/deepmew_ex/s1-proposed.json'
    plan = json.loads(path.read_text())
    receipt = json.loads((PROJECT_ROOT / 'docs/receipts/v1-admission-20261024.json').read_text())
    assert plan['stage'] == 'S1' and plan['execution_authorized'] is False
    assert plan['audit']['manifest_sha256'] == receipt['audit_manifest_sha256']
    assert plan['dataset']['selected_membership'] == receipt['audit']['proposed_membership']
    assert plan['training']['optimizer_updates'] == plan['training']['unique_training_decisions'] == 4096
    assert plan['evaluation']['validation_decisions'] == 1024
    assert plan['resources']['device'] == 'cpu' and plan['resources']['maximum_paid_cost'] == 0
    assert plan['actor']['recurrence'] == 'none'
    with patch('deepmew_ex.experiments.synthetic.SyntheticDataset', side_effect=AssertionError('dataset loaded')):
        with pytest.raises(ValueError, match='proposed S1'):
            load_config(path)
