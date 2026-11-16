"""Game-atomic candidate preparation for complete teacher-recorded source files."""

from __future__ import annotations

import gzip
import os
import sqlite3
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import asdict
from pathlib import Path

import orjson
import zstandard

from deepmew_ex.datasets.v2.contract import binding_hashes
from deepmew_ex.datasets.v2.learning.recorded_actor import recover_recorded_actor_game
from deepmew_ex.datasets.v2.schema import Payload
from deepmew_ex.registry.store import canonical, digest, file_hash


class LockedReferenceIndex:
    """Hash once while Windows denies writes, replacement and deletion.

    Only batch recovery opts into this proof. Other platforms and ordinary
    callers retain prepare_recorded_source's per-call full index hash check.
    """
    def __init__(self, path: Path, sha256: str, check: Callable[[], None]) -> None:
        self.path, self.sha256, self.check = path.resolve(), sha256, check
        self.handle: int | None = None
        self.kernel = None

    def __enter__(self) -> LockedReferenceIndex:
        if os.name != 'nt':
            raise NotImplementedError('write-denying reference index proof requires Windows')
        if self.handle is not None:
            raise ValueError('reference index lock already active')
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateFileW.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.CloseHandle.restype = wintypes.BOOL
        self.check()
        handle = kernel.CreateFileW(str(self.path), 0x80000000, 1, None, 3, 0x80, None)  # GENERIC_READ, FILE_SHARE_READ only, OPEN_EXISTING.
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        self.kernel, self.handle = kernel, handle
        try:
            wal = Path(str(self.path) + '-wal')
            if (wal.exists() and wal.stat().st_size) or file_hash(self.path) != self.sha256:
                raise ValueError('reference index is live or its identity changed')
            self.verify(self.path, self.sha256)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def verify(self, path: Path, sha256: str) -> None:
        self.check()
        if self.handle is None or path.resolve() != self.path or sha256 != self.sha256:
            raise ValueError('inactive or mismatched locked reference index proof')
        wal = Path(str(self.path) + '-wal')
        if wal.exists() and wal.stat().st_size:
            raise ValueError('reference index acquired unexpected WAL content')

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.handle is not None:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def recorded_games(source: Path, check: Callable[[], None]) -> Iterator[tuple[int, list[Payload]]]:
    opener = gzip.open if source.suffix == '.gz' else Path.open
    rows: list[Payload] = []
    first = 1
    with opener(source, 'rb') as handle:
        for line_number, line in enumerate(handle, 1):
            row = orjson.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get('game_id'), str) or type(row.get('game_index')) is not int:
                raise ValueError('recorded source has no exact game coordinate')
            if rows and (row['game_id'], row['game_index']) != (rows[0]['game_id'], rows[0]['game_index']):
                yield first, rows
                rows, first = [], line_number
            rows.append(row)
            if line_number % 1000 == 0:
                check()
        if rows:
            yield first, rows


def recorded_groups(source: Path, index: sqlite3.Connection, staging_root: Path, check: Callable[[], None]) -> Iterator[tuple[int, list[Payload], list[Payload]]]:
    """Group interleaved games by frozen canonical identity, keeping source order.

    Disk staging bounds memory to one complete game. Every source line must have
    its canonical reference; no rows or masks are dropped during grouping.
    """
    with tempfile.TemporaryDirectory(prefix='recorded-game-stage-', dir=staging_root) as directory:
        stage = sqlite3.connect(Path(directory) / 'rows.sqlite3')
        try:
            stage.executescript('CREATE TABLE rows(line INTEGER PRIMARY KEY,game TEXT NOT NULL,source BLOB NOT NULL,frame BLOB NOT NULL);')
            refs = index.execute('SELECT line,payload FROM refs WHERE source=? ORDER BY line', (str(source),))
            pending = refs.fetchone()
            decoder, encoder = zstandard.ZstdDecompressor(), zstandard.ZstdCompressor(level=1)
            opener = gzip.open if source.suffix == '.gz' else Path.open
            with opener(source, 'rb') as handle:
                for line_number, line in enumerate(handle, 1):
                    if pending is None or pending[0] != line_number:
                        raise ValueError('canonical references omit recorded source context')
                    frame = orjson.loads(decoder.decompress(pending[1]))
                    reference = frame['state_before']
                    if reference['source_path'] != str(source) or reference['source_line'] != line_number or not isinstance(frame.get('episode_id'), str):
                        raise ValueError('recorded grouping has inconsistent canonical source coordinates')
                    stage.execute('INSERT INTO rows VALUES (?,?,?,?)', (line_number, frame['episode_id'], encoder.compress(line), pending[1]))
                    pending = refs.fetchone()
                    if line_number % 1000 == 0:
                        stage.commit()
                        check()
            if pending is not None:
                raise ValueError('canonical references extend beyond recorded source')
            stage.commit()
            stage.execute('CREATE INDEX game_source_lines ON rows(game,line)')
            for game, first in stage.execute('SELECT game,MIN(line) FROM rows GROUP BY game ORDER BY MIN(line)'):
                check()
                rows, frames = [], []
                for raw, frame in stage.execute('SELECT source,frame FROM rows WHERE game=? ORDER BY line', (game,)):
                    rows.append(orjson.loads(decoder.decompress(raw)))
                    frames.append(orjson.loads(decoder.decompress(frame)))
                yield first, frames, rows
        finally:
            stage.close()


def prepare_recorded_source(source: Path, source_sha256: str, index_path: Path, index_sha256: str, output: Path, check: Callable[[], None], progress: Callable[[dict[str, object]], None], locked_index: LockedReferenceIndex | None = None) -> dict[str, object]:
    if output.resolve() in {source.resolve(), index_path.resolve()}:
        raise ValueError('recorded candidate output would overwrite an input')
    for path, expected in ((source, source_sha256), (index_path, index_sha256)):
        check()
        if path == index_path and locked_index is not None:
            locked_index.verify(path, expected)
            continue
        if file_hash(path) != expected:
            raise ValueError('recorded recovery input identity changed')
    root = Path(__file__).parents[1]
    modules = ('learning/recorded_cache.py', 'learning/recorded_actor.py', 'learning/actor_events.py', 'learning/queries.py', 'learning/trajectory.py', 'learning/semantic.py', 'learning/admission.py', 'corpus/reader.py', 'conversion.py', 'schema.py', 'contract.py')
    identity = {'source': str(source), 'source_sha256': source_sha256, 'reference_index_sha256': index_sha256, 'code': {name: file_hash(root / name) for name in modules}, 'parser': binding_hashes(), 'version': 2, 'contract': 'recorded_single_actor_selection_history', 'grouping': 'verified canonical episode identity; chronological original source lines within each game'}
    index = sqlite3.connect(index_path.resolve().as_uri() + ('?mode=ro&immutable=1' if locked_index is not None else '?mode=ro'), uri=True)
    connection = sqlite3.connect(output)
    try:
        if index.execute('SELECT sha FROM sources WHERE path=?', (str(source),)).fetchone() != (source_sha256,):
            raise ValueError('recorded source differs from canonical inventory')
        connection.executescript('PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; CREATE TABLE IF NOT EXISTS metadata(identity TEXT NOT NULL); CREATE TABLE IF NOT EXISTS games(game TEXT PRIMARY KEY,first_line INTEGER NOT NULL,requests INTEGER NOT NULL,split TEXT NOT NULL,source_game_id TEXT NOT NULL,source_game_index INTEGER NOT NULL,payload_sha TEXT NOT NULL,post_receipts TEXT NOT NULL); CREATE TABLE IF NOT EXISTS sequences(game TEXT NOT NULL,actor INTEGER NOT NULL,events INTEGER NOT NULL,records INTEGER NOT NULL,eligible INTEGER NOT NULL,payload BLOB NOT NULL,PRIMARY KEY(game,actor)); CREATE TABLE IF NOT EXISTS completion(receipt TEXT NOT NULL);')
        encoded_identity = canonical(identity).decode()
        previous = connection.execute('SELECT identity FROM metadata').fetchone()
        if previous is None:
            connection.execute('INSERT INTO metadata VALUES (?)', (encoded_identity,))
            connection.commit()
        elif previous != (encoded_identity,):
            raise ValueError('recorded recovery code or source identity changed')
        encoder = zstandard.ZstdCompressor(level=3)
        seen: set[str] = set()
        rows_read = 0
        for first, frames, rows in recorded_groups(source, index, output.parent, check):
            check()
            source_key = rows[0]['game_id'], rows[0]['game_index']
            game = frames[0]['episode_id']
            if game in seen:
                raise ValueError('repeated canonical recorded game')
            seen.add(game)
            rows_read += len(rows)
            previous = connection.execute('SELECT first_line,requests,source_game_id,source_game_index,payload_sha,post_receipts,split FROM games WHERE game=?', (game,)).fetchone()
            if previous is not None:
                members = connection.execute('SELECT actor,events,records,eligible,payload FROM sequences WHERE game=?', (game,)).fetchall()
                actual = digest(canonical([*members[0][:-1], digest(members[0][-1]), previous[5], previous[6]])) if len(members) == 1 else None
                if previous[:4] != (first, len(rows), *source_key) or actual != previous[4]:
                    raise ValueError('recorded recovery resume membership changed')
                continue
            recovered = recover_recorded_actor_game(frames, rows)
            sequence = recovered.sequence
            encoded = encoder.compress(orjson.dumps(asdict(sequence)))
            member = (sequence.actor, len(sequence.events), len(sequence.records), sum(row.loss_eligible for row in sequence.predictions), encoded)
            post_receipts = canonical(recovered.post_state_receipts).decode()
            member_sha = digest(canonical([*member[:-1], digest(encoded), post_receipts, sequence.split]))
            with connection:
                connection.execute('INSERT INTO sequences VALUES (?,?,?,?,?,?)', (game, *member))
                connection.execute('INSERT INTO games VALUES (?,?,?,?,?,?,?,?)', (game, first, len(rows), sequence.split, *source_key, member_sha, post_receipts))
            if len(seen) % 100 == 0:
                progress({'games': len(seen), 'requests': rows_read, 'training_admitted': False})
        if connection.execute('SELECT COUNT(*) FROM games').fetchone()[0] != len(seen) or connection.execute('SELECT COUNT(*) FROM sequences').fetchone()[0] != len(seen):
            raise ValueError('recorded source or cache population mismatch')
        check()
        if locked_index is not None:
            locked_index.verify(index_path, index_sha256)
        if file_hash(source) != source_sha256 or connection.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('recorded source mutation or cache integrity failure')
        receipt = {'identity': identity, 'games': len(seen), 'requests': rows_read, 'sequences': len(seen), 'eligible_before_global_admission': connection.execute('SELECT SUM(eligible) FROM sequences').fetchone()[0], 'coverage': 'recorded_single_actor_selection_history; opponent_and_zero_option_prompts_not_proven_complete', 'training_admitted': False}
        encoded_receipt = canonical(receipt).decode()
        with connection:
            previous_receipt = connection.execute('SELECT receipt FROM completion').fetchone()
            if previous_receipt is None:
                connection.execute('INSERT INTO completion VALUES (?)', (encoded_receipt,))
            elif previous_receipt != (encoded_receipt,):
                raise ValueError('recorded completion changed')
        connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    finally:
        connection.close()
        index.close()
    return {**receipt, 'cache_sha256': file_hash(output)}
