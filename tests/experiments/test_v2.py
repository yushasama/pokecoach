from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path

import numpy as np
import orjson
import pytest
import torch

from deepmew_ex.experiments.config import OutputPaths, load_config
from deepmew_ex.experiments.v2.config import V2ExperimentConfig
from deepmew_ex.experiments.v2.runner import load_semantic_checkpoint, run_v2
from deepmew_ex.registry.store import RunStore, verify_run


@pytest.fixture
def semantic_cache(tmp_path: Path) -> Path:
    store = RunStore.allocate(OutputPaths(tmp_path / 'outputs'), 'deepmew_ex', 'v2_learning_cache', 'fixture')
    store.write('vocabulary.json', {'paths': {'state': 2, 'option': 3}, 'values': {'low': 2, 'high': 3, 'yes': 4, 'no': 5}, 'max_leaves': 1024})
    store.write('config.json', {'window': 2})
    examples = {}
    for split in ('train', 'validation'):
        records = np.array([[2, 2, .1], [2, 3, .2], [3, 4, .3], [3, 5, .4]], dtype=np.float32)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, tokens=records, offsets=np.arange(5, dtype=np.int64))
        name = split + '.npz'
        sha = store.write_bytes(name, buffer.getvalue())
        examples[split] = [{'row_id': f'{split}-{index}', 'game_id': split, 'actor_order': index, 'signature': f'{split}-{index}', 'selection_hash': f'{split}-{index}', 'state_record': index, 'history_records': [0] if index else [], 'option_records': [2, 3], 'target': index, 'cache_file': name, 'cache_sha256': sha} for index in range(2)]
    store.write('examples.json', examples)
    store.seal({'kind': 'v2_learning_cache'})
    return store.path


def test_v2_checkpoint_consumption_and_existing_store(semantic_cache: Path, tmp_path: Path) -> None:
    config = V2ExperimentConfig(str(semantic_cache), 'ordered', 'unit-v2', presentations=8, batch_size=2, evaluate_every=2, hidden_dim=16, embedding_dim=8, max_wall_seconds=60, output_root=str(tmp_path / 'outputs'))
    path = run_v2(config, 'training-unit')
    manifest = verify_run(path)
    assert manifest['run']['status'] == 'completed'
    metrics = orjson.loads((path / 'metrics.json').read_bytes())
    assert metrics['checkpoint_reload_exact'] and metrics['changed_parameter_tensors']
    consumed = orjson.loads((path / 'consumption.json').read_bytes())
    assert consumed['presentation_count'] == 8 and consumed['unique_rows'] == 2
    assert consumed['optimizer_updates'] == 4 and consumed['holdout_fitting_or_selection_reads'] == 0
    model, checkpoint = load_semantic_checkpoint(path / 'checkpoints/best.pt', manifest['artifacts'][0]['sha256'])
    assert checkpoint['history_window'] == 2 and model.history == 'ordered'
    with pytest.raises(ValueError, match='identity'):
        load_semantic_checkpoint(path / 'checkpoints/best.pt', '0' * 64)
    with pytest.raises(FileExistsError):
        run_v2(config, 'training-unit')


def test_s2_configuration_rejects_unsupported_before_data_read(tmp_path: Path) -> None:
    path = tmp_path / 'config.json'
    path.write_bytes(orjson.dumps({'stage': 'S2', 'cache': 'absent', 'history': 'ordered', 'variant': 'unit', 'objective': 'td'}))
    with pytest.raises(ValueError, match='before data loading'):
        load_config(path)
    assert not (tmp_path / 'outputs').exists()


def test_controlled_history_comparison_rejects_optimizer_change(semantic_cache: Path, tmp_path: Path) -> None:
    from deepmew_ex.experiments.v2.comparison import validate_conditions
    config = V2ExperimentConfig(str(semantic_cache), 'snapshot', 'compare-snapshot', presentations=4, batch_size=2, evaluate_every=2, hidden_dim=8, embedding_dim=4, max_wall_seconds=60, output_root=str(tmp_path / 'outputs'))
    paths = {role: run_v2(replace(config, history=role, variant='compare-' + role), 'compare-' + role) for role in ('snapshot', 'null', 'ordered', 'shuffled')}
    assert validate_conditions(tuple(paths.values()))['declared_factor'] == 'history treatment'
    changed = run_v2(replace(config, history='ordered', variant='changed-optimizer', learning_rate=.01), 'changed-optimizer')
    with pytest.raises(ValueError, match='undeclared'):
        validate_conditions((paths['snapshot'], paths['null'], changed, paths['shuffled']))
