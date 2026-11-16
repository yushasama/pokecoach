"""Freeze eligible temporal membership independently of optimizer iteration."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable

from deepmew_ex.datasets.v2.learning.trajectory import ActorSequence


def prepare_expected(connection: sqlite3.Connection, sequences: Iterable[ActorSequence], epochs: int, check_resources: Callable[[], None]) -> dict[str, int]:
    if type(epochs) is not int or epochs < 1:
        raise ValueError('positive epoch count required')
    if connection.execute('SELECT COUNT(*) FROM expected').fetchone()[0]:
        raise ValueError('expected membership must be prepared exactly once')
    count, decisions = 0, 0
    seen: set[tuple[str, int]] = set()
    with connection:
        for sequence in sequences:
            check_resources()
            if sequence.split != 'train' or not sequence.history_available:
                raise ValueError('temporal training requires train-split causal histories')
            identity = (sequence.game_id, sequence.actor)
            if identity in seen:
                raise ValueError('duplicate actor sequence in expected membership')
            seen.add(identity)
            count += 1
            for prediction in sequence.predictions:
                if not prediction.loss_eligible:
                    continue
                connection.executemany('INSERT INTO expected VALUES (?, ?, ?, ?, ?, ?)', ((epoch, 'raw', prediction.row_id, sequence.game_id, sequence.source_family, 0) for epoch in range(epochs)))
                decisions += 1
        if not decisions:
            raise ValueError('empty eligible temporal training membership')
    return {'actor_sequences': count, 'games': len({game for game, _ in seen}), 'unique_eligible_decisions': decisions, 'epochs': epochs, 'expected_presentations': decisions * epochs}


def verify_consumption(connection: sqlite3.Connection) -> dict[str, int]:
    columns = 'epoch,phase,id,game,family'
    missing = connection.execute(f'SELECT COUNT(*) FROM (SELECT {columns} FROM expected EXCEPT SELECT {columns} FROM consumed)').fetchone()[0]
    unexpected = connection.execute(f'SELECT COUNT(*) FROM (SELECT {columns} FROM consumed EXCEPT SELECT {columns} FROM expected)').fetchone()[0]
    if missing or unexpected:
        raise ValueError(f'training coverage mismatch: missing={missing}, unexpected={unexpected}')
    return {'missing': missing, 'unexpected': unexpected, 'presentations': connection.execute('SELECT COUNT(*) FROM consumed').fetchone()[0]}


def admit_update(connection: sqlite3.Connection, epoch: int, sequence: ActorSequence, ids: list[str]) -> None:
    if sequence.split != 'train' or not sequence.history_available:
        raise ValueError('update is outside temporal training scope')
    for identity in ids:
        expected = connection.execute('SELECT game,family FROM expected WHERE epoch=? AND phase=? AND id=?', (epoch, 'raw', identity)).fetchone()
        consumed = connection.execute('SELECT 1 FROM consumed WHERE epoch=? AND phase=? AND id=?', (epoch, 'raw', identity)).fetchone()
        if expected != (sequence.game_id, sequence.source_family) or consumed is not None:
            raise ValueError('update membership differs from independent preflight')
