from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from deepmew_ex.evaluation.offline import evaluate, load_checkpoint
from deepmew_ex.experiments.synthetic import SyntheticDataset
from deepmew_ex.models.mlp import DeepMewEX
from deepmew_ex.training.bc import admit, check_isolation, encode, set_seed


def test_actual_updates_and_finite_gradients(s0_run: Path) -> None:
    config = json.loads((s0_run / 'config.json').read_text())
    set_seed(config['seed'])
    initial = DeepMewEX(hidden_dim=config['hidden_dim'])
    final, checkpoint = load_checkpoint(s0_run / 'checkpoints/final.pt')
    assert checkpoint['update'] == 32
    assert any(not torch.equal(value, checkpoint['model'][key]) for key, value in initial.state_dict().items())
    row = list(SyntheticDataset(42, 64, 16).decisions('train'))[4]
    options, values = encode(row)
    logits = final(options, values)
    loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([int(row.selection[0])])) * row.actor_loss_weight
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in final.parameters())
    metrics = json.loads((s0_run / 'metrics.json').read_text())
    assert metrics['gradients_finite_every_update'] and metrics['changed_parameter_tensors']


def test_reload_parity_selection_support_and_consumption(s0_run: Path) -> None:
    first, checkpoint = load_checkpoint(s0_run / 'checkpoints/best.pt')
    second, _ = load_checkpoint(s0_run / 'checkpoints/best.pt')
    rows, support = admit(SyntheticDataset(42, 64, 16).decisions('validation'), 'validation')
    assert support.raw == 16 and support.eligible == 12
    for row in rows:
        assert torch.equal(first(*encode(row)), second(*encode(row)))
    assert evaluate(second, rows).report() == checkpoint['validation']
    metrics = json.loads((s0_run / 'metrics.json').read_text())
    maximum = max(entry['validation']['top1'] for entry in metrics['history'])
    earliest = next(entry['update'] for entry in metrics['history'] if entry['validation']['top1'] == maximum)
    assert checkpoint['update'] == earliest == metrics['selected_update']
    assert metrics['support']['train']['eligible'] == 60
    assert metrics['support']['train']['excluded'] == {'forced': 1, 'zero_actor_weight': 1, 'multi_select': 1, 'optional_empty': 1}
    ledger = json.loads((s0_run / 'consumption.json').read_text())
    assert len(ledger['presentations']) == ledger['updates'] == 32
    assert len(ledger['unique_rows']) == 32
    assert ledger['holdout_reads'] == 0 and ledger['dataset_reads'] == ['train', 'validation']
    assert all(row.startswith('train-') for row in ledger['presentations'])


def test_option_permutation_and_semantic_negative_control(s0_run: Path) -> None:
    model, _ = load_checkpoint(s0_run / 'checkpoints/best.pt')
    row = list(SyntheticDataset(42, 64, 16).decisions('validation'))[7]
    permutation = np.arange(len(row.option_features))[::-1].copy()
    remapped_target = int(np.where(permutation == row.selection[0])[0][0])
    permuted = replace(row, option_features=row.option_features[permutation].copy(), selection=np.array([remapped_target], dtype=np.int32))
    original = torch.softmax(model(*encode(row)), dim=0)
    changed_order = torch.softmax(model(*encode(permuted)), dim=0)
    assert torch.allclose(original[permutation.copy()], changed_order, atol=1e-7, rtol=1e-6)
    assert int(permutation[changed_order.argmax()]) == int(original.argmax())
    assert int(permutation[permuted.selection[0]]) == row.selection[0]
    changed_features = row.option_features.copy()
    changed_features[0] = -3 * changed_features[0]
    semantic_change = replace(row, option_features=changed_features)
    assert not torch.allclose(original, torch.softmax(model(*encode(semantic_change)), dim=0), atol=1e-7, rtol=1e-6)


def test_actor_allowlist_ignores_labels_and_metadata(s0_run: Path) -> None:
    model, _ = load_checkpoint(s0_run / 'checkpoints/best.pt')
    row = list(SyntheticDataset(42, 64, 16).decisions('validation'))[5]
    altered = replace(row, terminal_outcome=12345.0, actor_loss_weight=900.0, game_id='oracle-negative-control', selection=np.array([0], dtype=np.int32), continuity='invented_full_trajectory', decision_index=999)
    assert torch.equal(model(*encode(row)), model(*encode(altered)))
    # Encoder reads two named fields only, even if an upstream carrier has oracle extras.
    class Carrier:
        option_features = row.option_features
        value_features = row.value_features
        oracle_hand = 'hidden-A'
    carrier = Carrier()
    before = model(*encode(carrier))
    carrier.oracle_hand = 'hidden-B'
    assert torch.equal(before, model(*encode(carrier)))


@pytest.mark.parametrize('split', ['train', 'validation'])
def test_empty_eligible_sets_are_rejected(split: str) -> None:
    with pytest.raises(ValueError, match='empty eligible'):
        admit([], split)
    with pytest.raises(ValueError, match='empty eligible'):
        admit(list(SyntheticDataset(42, 1, 1).decisions(split)), split)


@pytest.mark.parametrize('kind', ['target', 'nan_options', 'nan_values', 'nan_weight', 'legal_count', 'selection_shape'])
def test_invalid_rows_fail(kind: str) -> None:
    row = list(SyntheticDataset(42, 64, 16).decisions('train'))[4]
    if kind == 'target':
        row = replace(row, selection=np.array([len(row.option_features)], dtype=np.int32))
    elif kind == 'nan_options':
        row.option_features[0, 0] = np.nan
    elif kind == 'nan_values':
        row.value_features[0] = np.inf
    elif kind == 'nan_weight':
        row = replace(row, actor_loss_weight=float('nan'))
    elif kind == 'legal_count':
        row = replace(row, complete_legal_action_count=999)
    else:
        row = replace(row, selection=np.array([[0]], dtype=np.int32))
    with pytest.raises(ValueError):
        admit([row], 'train')


def test_split_isolation_and_sealed_holdout() -> None:
    dataset = SyntheticDataset(42, 64, 16)
    train, validation = list(dataset.decisions('train')), list(dataset.decisions('validation'))
    check_isolation(train, validation)
    for field in ('game_id', 'trajectory_id', 'row_id'):
        contaminated = [replace(validation[0], **{field: getattr(train[0], field)})]
        with pytest.raises(ValueError, match='crosses'):
            check_isolation(train, contaminated)
    with pytest.raises(ValueError, match='holdout'):
        list(dataset.decisions('holdout'))


def test_combinatorial_unsupported_prompts_are_excluded_not_corrupt() -> None:
    row = list(SyntheticDataset(42, 64, 16).decisions('train'))[4]
    multi = replace(row, row_id='multi', max_count=2, complete_legal_action_count=21, selection=np.array([0, 1], dtype=np.int32))
    optional = replace(row, row_id='optional', min_count=0, complete_legal_action_count=len(row.option_features) + 1, selection=np.array([], dtype=np.int32))
    admitted, support = admit([row, multi, optional], 'train')
    assert admitted == [row]
    assert support.raw == 3 and support.excluded == {'multi_select': 1, 'optional_empty': 1}
