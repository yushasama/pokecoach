import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from deepmew_ex.training.bitmap_ledger import BitmapLedger, ExpectedIndex, verify_run_ledger


def sequence(game: str, actor: int, ids: list[str], family: str = 'fam', eligible: bool = True, split: str = 'train', history: bool = True) -> SimpleNamespace:
    predictions = tuple(SimpleNamespace(row_id=value, loss_eligible=eligible) for value in ids)
    return SimpleNamespace(split=split, history_available=history, predictions=predictions, game_id=game, source_family=family, actor=actor, events=(0, 1), preserved_transition_ids=(), unavailable_post_states=())


def build(tmp_path: Path) -> ExpectedIndex:
    rows = [(f'id-{game}-{actor}-{k}', f'g{game}', 'fam' if game < 2 else 'other', game * 2 + actor) for game in range(3) for actor in range(2) for k in range(3)]
    index = ExpectedIndex.build(rows, tmp_path / 'expected-ids.txt.zst')
    index.save(tmp_path)
    (tmp_path / 'expected-membership.json').write_text(json.dumps(index.summary(2)))
    return index


def test_exactly_once_coverage_per_epoch_with_resumable_chain(tmp_path: Path) -> None:
    index = build(tmp_path)
    assert index.summary(2) == {'actor_sequences': 6, 'games': 3, 'unique_eligible_decisions': 18, 'epochs': 2, 'expected_presentations': 36}
    ledger = BitmapLedger(tmp_path, index, 2)
    update = 0
    for epoch in (0, 1):
        for game in range(3):
            group = [sequence(f'g{game}', actor, [f'id-{game}-{actor}-{k}' for k in range(3)], 'fam' if game < 2 else 'other') for actor in range(2)]
            consumed = ledger.admit(epoch, group)
            update += 1
            chain = ledger.record(epoch, update, group, consumed)
            if update == 4:
                checkpoint = chain
    assert ledger.verify_complete() == {'missing': 0, 'unexpected': 0, 'presentations': 36}
    assert ledger.family_counts() == [('raw', 'fam', 12), ('raw', 'other', 6)]
    ledger.close()
    offline = verify_run_ledger(tmp_path)
    assert offline['missing'] == 0 and offline['updates'] == 6 and offline['chain'] == ledger.chain and offline['train_games'] == {'g0', 'g1', 'g2'}
    child = tmp_path / 'child'
    child.mkdir()
    resumed = BitmapLedger(child, ExpectedIndex.load(tmp_path), 2)
    resumed.replay(tmp_path, 4, checkpoint)
    assert resumed.updates == 4 and resumed.presentations() == 24
    with pytest.raises(ValueError, match='committed checkpoint'):
        BitmapLedger(tmp_path / 'child2' if (tmp_path / 'child2').mkdir() is None else tmp_path, ExpectedIndex.load(tmp_path), 2).replay(tmp_path, 4, 'f' * 64)
    resumed.close()


@pytest.mark.parametrize('mutate, message', [
    (lambda: [sequence('g0', 0, ['id-0-0-0', 'not-expected'])], 'preflight'),
    (lambda: [sequence('g1', 0, ['id-0-0-0'])], 'preflight'),
    (lambda: [sequence('g0', 0, ['id-0-0-0'], family='other')], 'preflight'),
    (lambda: [sequence('g0', 0, ['id-0-0-0']), sequence('g0', 1, ['id-0-0-0'])], 'preflight'),
    (lambda: [sequence('g0', 0, ['id-0-0-0']), sequence('g0', 0, ['id-0-0-1'])], 'twice'),
    (lambda: [sequence('g0', 0, ['id-0-0-0'], split='validation')], 'scope'),
    (lambda: [sequence('g0', 0, ['id-0-0-0'], history=False)], 'scope'),
])
def test_admission_rejects_membership_violations(tmp_path: Path, mutate, message: str) -> None:
    ledger = BitmapLedger(tmp_path, build(tmp_path), 2)
    with pytest.raises(ValueError, match=message):
        ledger.admit(0, mutate())
    ledger.close()


def test_double_consumption_missing_updates_and_incomplete_epochs_fail(tmp_path: Path) -> None:
    ledger = BitmapLedger(tmp_path, build(tmp_path), 2)
    group = [sequence('g0', 0, ['id-0-0-0'])]
    ledger.record(0, 1, group, ledger.admit(0, group))
    with pytest.raises(ValueError, match='preflight'):
        ledger.admit(0, [sequence('g0', 1, ['id-0-0-0'])])
    with pytest.raises(ValueError, match='twice'):
        ledger.admit(0, [sequence('g0', 0, ['id-0-0-1'])])
    assert len(ledger.admit(1, group)[0]) == 1
    with pytest.raises(ValueError, match='missing optimizer update'):
        ledger.record(1, 3, group, ledger.admit(1, group))
    with pytest.raises(ValueError, match='missing=35'):
        ledger.verify_complete()
    ledger.close()


def test_duplicate_expected_rows_and_empty_membership_fail(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='duplicate'):
        ExpectedIndex.build([('a', 'g', 'f', 0), ('a', 'g', 'f', 1)])
    with pytest.raises(ValueError, match='empty'):
        ExpectedIndex.build([])
    index = ExpectedIndex.build([(f'x{i}', 'g', 'f', i) for i in range(1000)])
    assert np.array_equal(index.lookup(['x7', 'x999', 'x0']), np.array([7, 999, 0]))
