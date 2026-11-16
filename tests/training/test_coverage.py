from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import replace

import pytest

from deepmew_ex.datasets.v2.learning.trajectory import actor_sequences
from deepmew_ex.training.coverage import admit_update, prepare_expected, verify_consumption
from tests.datasets.v2.learning.test_admission import episode


@pytest.fixture
def ledger() -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(':memory:')
    for name in ('expected', 'consumed'):
        connection.execute(f'CREATE TABLE {name} (epoch INTEGER, phase TEXT, id TEXT, game TEXT, family TEXT, update_id INTEGER, PRIMARY KEY(epoch,phase,id))')
    yield connection
    connection.close()


@pytest.mark.parametrize('change', ['validation', 'history_free', 'duplicate'])
def test_preflight_rejects_unapproved_sequence_membership(ledger: sqlite3.Connection, change: str) -> None:
    sequence = actor_sequences(episode(0), 'train')[0]
    rows = [replace(sequence, split='validation')] if change == 'validation' else [replace(sequence, history_available=False)] if change == 'history_free' else [sequence, sequence]
    with pytest.raises(ValueError):
        prepare_expected(ledger, rows, 1, lambda: None)
    assert ledger.execute('SELECT COUNT(*) FROM expected').fetchone()[0] == 0


def test_update_rejects_unplanned_or_already_consumed_ids(ledger: sqlite3.Connection) -> None:
    sequences = actor_sequences(episode(0), 'train')
    prepare_expected(ledger, sequences, 1, lambda: None)
    sequence = sequences[0]
    ids = [row.row_id for row in sequence.predictions if row.loss_eligible]
    admit_update(ledger, 0, sequence, ids)
    with pytest.raises(ValueError, match='membership differs'):
        admit_update(ledger, 0, sequence, ['unplanned'])
    ledger.execute('INSERT INTO consumed SELECT * FROM expected')
    with pytest.raises(ValueError, match='membership differs'):
        admit_update(ledger, 0, sequence, ids)


def test_independent_membership_detects_missing_epoch_and_wrong_game(ledger: sqlite3.Connection) -> None:
    stats = prepare_expected(ledger, actor_sequences(episode(0), 'train'), 2, lambda: None)
    assert stats['unique_eligible_decisions'] == 4 and stats['expected_presentations'] == 8
    ledger.execute('INSERT INTO consumed SELECT * FROM expected WHERE epoch=0')
    with pytest.raises(ValueError, match='missing=4'):
        verify_consumption(ledger)
    ledger.execute('INSERT INTO consumed SELECT * FROM expected WHERE epoch=1')
    assert verify_consumption(ledger)['presentations'] == 8
    ledger.execute("UPDATE consumed SET game='different-game' WHERE epoch=1")
    with pytest.raises(ValueError, match='missing=4, unexpected=4'):
        verify_consumption(ledger)


def test_resource_stop_rolls_back_incomplete_membership(ledger: sqlite3.Connection) -> None:
    calls = 0

    def check() -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise TimeoutError('budget')

    with pytest.raises(TimeoutError):
        prepare_expected(ledger, actor_sequences(episode(0), 'train'), 1, check)
    assert ledger.execute('SELECT COUNT(*) FROM expected').fetchone()[0] == 0
