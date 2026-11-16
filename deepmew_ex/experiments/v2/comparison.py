"""Validate the declared history treatment without loading any checkpoint tensors."""

from __future__ import annotations

from pathlib import Path

import orjson

from deepmew_ex.datasets.v2.learning.semantic import HISTORY_TREATMENTS
from deepmew_ex.registry.store import canonical, digest, file_hash, verify_run


def validate_conditions(paths: tuple[Path, ...]) -> dict[str, object]:
    conditions = {}
    for path in paths:
        manifest = verify_run(path)
        if manifest['status'] != 'completed' or manifest.get('kind') != 'v2_bc_training':
            raise ValueError('comparison needs completed V2 BC artifacts')
        config = orjson.loads((path / 'config.json').read_bytes())
        role = config['history']
        if role in conditions:
            raise ValueError('duplicate history condition')
        conditions[role] = {'config': config, 'dataset': orjson.loads((path / 'dataset.json').read_bytes()), 'consumption': orjson.loads((path / 'consumption.json').read_bytes()), 'parameters': orjson.loads((path / 'parameters.json').read_bytes()), 'specification': orjson.loads((path / 'model-spec.json').read_bytes()), 'manifest_sha256': file_hash(path / 'manifest.json'), 'checkpoint_sha256': next(artifact['sha256'] for artifact in manifest['artifacts'] if artifact['role'] == 'best'), 'metrics': orjson.loads((path / 'metrics.json').read_bytes())}
    if set(conditions) != set(HISTORY_TREATMENTS):
        raise ValueError('comparison requires rich snapshot, null, ordered and shuffled conditions')
    reference = conditions['null']
    for role, condition in conditions.items():
        stable_config = {key: value for key, value in condition['config'].items() if key not in ('history', 'variant')}
        reference_config = {key: value for key, value in reference['config'].items() if key not in ('history', 'variant')}
        if stable_config != reference_config or condition['dataset'] != reference['dataset'] or condition['consumption'] != reference['consumption']:
            raise ValueError('undeclared data, split, sampling, budget or optimizer difference')
        if role != 'snapshot' and (condition['specification'] != reference['specification'] or condition['parameters']['initial_tensor_sha256'] != reference['parameters']['initial_tensor_sha256']):
            raise ValueError('recurrent control architecture or initialization differs')
    return {'schema': 'deepmew.ex.v2_bc_comparison.v1', 'identity': digest(canonical({role: condition['manifest_sha256'] for role, condition in conditions.items()})), 'declared_factor': 'history treatment', 'memory_ablation': 'ordered vs null; same architecture, initialization, data, targets, updates and selection', 'negative_control': 'shuffled prior requests; same-actor and causal, no future or cross-game records', 'snapshot_comparison': 'practical system comparison; snapshot omits recurrent parameters, so this alone is not a pure memory ablation', 'conditions': {role: {key: value for key, value in condition.items() if key in ('manifest_sha256', 'checkpoint_sha256', 'parameters', 'metrics')} for role, condition in conditions.items()}, 'scope': 'single training seed, frozen validation selection, exploratory league follows separately; E11/E23 confirmation not claimed'}
