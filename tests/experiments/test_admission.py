from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from deepmew_ex.experiments.config import ExperimentConfig
from deepmew_ex.experiments.runner import run_s0
from deepmew_ex.registry.store import verify_run


@pytest.mark.parametrize(('field', 'value'), [('dataset', 'v2'), ('dataset', 'real_v1'), ('objective', 'temporal_bc'), ('objective', 'td'), ('objective', 'multi_select'), ('encoder', 'entity_v2'), ('device', 'cuda'), ('updates', 33), ('hidden_dim', 65), ('train_decisions', 65), ('validation_decisions', 17), ('learning_rate', float('nan'))])
def test_reject_before_loading_or_allocation(tmp_path: Path, field: str, value: object) -> None:
    config = replace(ExperimentConfig(output_root=str(tmp_path / 'outputs')), **{field: value})
    with patch('deepmew_ex.experiments.synthetic.SyntheticDataset', side_effect=AssertionError('data loaded')):
        with pytest.raises(ValueError):
            run_s0(config, 'rejected')
    assert not (tmp_path / 'outputs').exists()


def test_training_failure_has_sealed_receipt(tmp_path: Path) -> None:
    config = ExperimentConfig(train_decisions=1, output_root=str(tmp_path / 'outputs'))
    with pytest.raises(ValueError, match='empty eligible train'):
        run_s0(config, 'empty-failure')
    path = config.paths().run(config.family, config.variant, 'empty-failure')
    manifest = verify_run(path)
    assert manifest['status'] == 'failed'
    assert 'failure.json' in manifest['files_sha256']
    assert not list((path / 'checkpoints').iterdir())
