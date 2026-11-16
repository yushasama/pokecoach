"""Authenticate committed coverage prefixes and exclude live writers on recovery."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import BinaryIO

from deepmew_ex.registry.store import canonical, digest

EMPTY_LEDGER_DIGEST = digest(b'')


class TrainingLease:
    """OS-released lock outside immutable run contents; survives abrupt process exit."""

    def __init__(self, run: Path, require_existing: bool = False) -> None:
        path = run.parent / f'.{run.name}.training-lease'
        if require_existing and not path.is_file():
            raise ValueError('unsealed recovery requires a recorded training lease')
        self.handle: BinaryIO = path.open('a+b')
        if path.stat().st_size == 0:
            self.handle.write(b'\0')
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.handle.close()
            raise ValueError('training writer is still active or its lease is unavailable') from error

    def close(self) -> None:
        if self.handle.closed:
            return
        self.handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


def extend_ledger_digest(connection: sqlite3.Connection, previous: str, update: int) -> str:
    consumed = connection.execute('SELECT epoch,phase,id,game,family,update_id FROM consumed WHERE update_id=? ORDER BY epoch,phase,id', (update,)).fetchall()
    context = connection.execute('SELECT epoch,phase,game,actor,requests,events,post_states,unavailable_post_states,update_id FROM context WHERE update_id=? ORDER BY epoch,phase,game,actor', (update,)).fetchall()
    if not consumed:
        raise ValueError('coverage ledger has a missing optimizer update')
    return digest(canonical({'previous': previous, 'update': update, 'consumed': consumed, 'context': context}))


def verify_ledger_prefix(connection: sqlite3.Connection, updates: int, expected: str) -> None:
    for table in ('consumed', 'context'):
        if connection.execute(f'SELECT COUNT(*) FROM {table} WHERE update_id<1 OR update_id>?', (updates,)).fetchone()[0]:
            raise ValueError('coverage rows are outside the committed update prefix')
    actual = EMPTY_LEDGER_DIGEST
    for update in range(1, updates + 1):
        actual = extend_ledger_digest(connection, actual, update)
    if actual != expected:
        raise ValueError('coverage prefix differs from committed checkpoint identity')
