from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest
import zstandard

from deepmew_ex.datasets.v2.learning import prepared
from deepmew_ex.datasets.v2.learning.prepared import CandidateCache, finalize_membership, prepare_membership, publish_prepared
from deepmew_ex.datasets.v2.learning.prepared_recovery import audit_ingested
from deepmew_ex.datasets.v2.learning.recovery_cache import ReferenceShard
from deepmew_ex.datasets.v2.learning.trajectory import ActorSequence
from deepmew_ex.registry.store import RunStore, canonical, file_hash, verify_run
from tests.datasets.v2.corpus.conftest import corpus_inventory
from tests.datasets.v2.learning.test_factory_cache import factory_shard
from tests.datasets.v2.learning.test_prepared import renamed, sample
from tests.datasets.v2.learning.test_prepared_publication import factory, publication
from tests.datasets.v2.learning.test_recovery import reference_game


def cache_for(path: Path, sequences: list[ActorSequence], teacher: bool = False) -> CandidateCache:
    identity = {'fixture': True}
    encoder = zstandard.ZstdCompressor(level=3)
    with sqlite3.connect(path) as connection:
        connection.executescript('CREATE TABLE metadata(identity TEXT); CREATE TABLE completion(receipt TEXT); CREATE TABLE games(game TEXT PRIMARY KEY); CREATE TABLE sequences(game TEXT,actor INTEGER,payload BLOB);')
        connection.execute('INSERT INTO metadata VALUES (?)', (canonical(identity).decode(),))
        for sequence in sequences:
            connection.execute('INSERT OR IGNORE INTO games VALUES (?)', (sequence.game_id,))
            connection.execute('INSERT INTO sequences VALUES (?,?,?)', (sequence.game_id, sequence.actor, encoder.compress(orjson.dumps(asdict(sequence)))))
        connection.execute('INSERT INTO completion VALUES (?)', (canonical({'identity': identity, 'sequences': len(sequences), 'requests': sum(len(row.predictions) for row in sequences)}).decode(),))
    connection.close()
    return CandidateCache(path, file_hash(path), 'recorded' if teacher else 'reference')


def staged(path: Path, sequences: list[ActorSequence]) -> sqlite3.Connection:
    connection = sqlite3.connect(path)

    def stop() -> None:
        raise InterruptedError('ingestion committed')

    with pytest.raises(InterruptedError, match='ingestion committed'):
        prepare_membership(connection, sequences, lambda: None, stop)
    return connection


def table_rows(connection: sqlite3.Connection) -> dict[str, list[tuple[object, ...]]]:
    return {name: connection.execute(f'SELECT * FROM {name} ORDER BY 1,2').fetchall() for name in ('sequences', 'predictions', 'reservations')}


def test_legacy_lookup_layout_recovery_preserves_every_admission_row(tmp_path: Path) -> None:
    sequences = [renamed(sample(0), 'a'), renamed(sample(0, 'holdout'), 'b'), renamed(sample(1), 'c')]
    cache = cache_for(tmp_path / 'cache.sqlite3', sequences)
    connection = staged(tmp_path / 'interrupted.sqlite3', sequences)
    connection.executescript('DROP TABLE reservations; CREATE TABLE reservations(signature TEXT PRIMARY KEY,rank INTEGER NOT NULL);')
    connection.execute('BEGIN EXCLUSIVE')
    coverage = audit_ingested(connection, (cache,), lambda: None)
    connection.commit()
    actual = finalize_membership(connection, lambda: None)
    reference = sqlite3.connect(tmp_path / 'reference.sqlite3')
    expected = prepare_membership(reference, sequences, lambda: None)
    assert actual == expected
    assert table_rows(connection) == table_rows(reference)
    assert coverage[0]['requests'] == sum(len(row.predictions) for row in sequences)
    plan = connection.execute('EXPLAIN QUERY PLAN SELECT rank FROM reservations WHERE signature=?', ('a' * 64,)).fetchall()
    assert any('USING PRIMARY KEY' in row[3] for row in plan)
    assert 'WITHOUT ROWID' in connection.execute("SELECT sql FROM sqlite_master WHERE name='reservations'").fetchone()[0]
    connection.close()
    reference.close()


def test_hard_exit_during_global_update_rolls_back_and_recovers_exactly(tmp_path: Path) -> None:
    sequences = [sample(0)]
    cache = cache_for(tmp_path / 'cache.sqlite3', sequences)
    path = tmp_path / 'crashed.sqlite3'
    script = '''import os,sqlite3,sys
from deepmew_ex.datasets.v2.learning.prepared import prepare_membership
from tests.datasets.v2.learning.test_prepared import sample
class Crash(sqlite3.Connection):
 def execute(self, sql, parameters=()):
  result=super().execute(sql,parameters)
  if sql.startswith('UPDATE predictions SET reason='): os._exit(93)
  return result
c=sqlite3.connect(sys.argv[1],factory=Crash)
prepare_membership(c,[sample(0)],lambda:None)
'''
    result = subprocess.run([sys.executable, '-B', '-c', script, str(path)], capture_output=True, timeout=30)
    assert result.returncode == 93, result.stderr.decode()
    connection = sqlite3.connect(path)
    connection.execute('BEGIN EXCLUSIVE')
    audit_ingested(connection, (cache,), lambda: None)
    connection.commit()
    result = finalize_membership(connection, lambda: None)
    reference = sqlite3.connect(tmp_path / 'reference.sqlite3')
    assert result == prepare_membership(reference, sequences, lambda: None)
    assert table_rows(connection) == table_rows(reference)
    connection.close()
    reference.close()


@pytest.mark.parametrize('defect', ['missing_actor', 'source_mask', 'split', 'payload', 'signature', 'cache_changed', 'already_admitted'])
def test_recovery_rejects_identity_or_admission_drift(tmp_path: Path, defect: str) -> None:
    sequences = [renamed(sample(0), 'a'), renamed(sample(1), 'b')]
    cache = cache_for(tmp_path / 'cache.sqlite3', sequences)
    connection = staged(tmp_path / 'interrupted.sqlite3', sequences)
    if defect == 'missing_actor':
        connection.execute('DELETE FROM sequences WHERE ordinal=1')
    elif defect == 'source_mask':
        connection.execute('UPDATE predictions SET source_eligible=1-source_eligible WHERE sequence_id=0')
    elif defect == 'split':
        connection.execute("UPDATE sequences SET split='holdout',rank=3 WHERE ordinal=0")
    elif defect == 'payload':
        connection.execute('UPDATE sequences SET payload=? WHERE ordinal=0', (zstandard.ZstdCompressor().compress(b'{}'),))
    elif defect == 'signature':
        connection.execute('UPDATE predictions SET signature=? WHERE sequence_id=0', ('0' * 64,))
    elif defect == 'cache_changed':
        with cache.path.open('ab') as handle:
            handle.write(b'changed')
    else:
        connection.execute("UPDATE predictions SET reason='admitted',eligible=1")
    connection.commit()
    connection.execute('BEGIN EXCLUSIVE')
    with pytest.raises(ValueError):
        audit_ingested(connection, (cache,), lambda: None)
    connection.rollback()
    connection.close()


def test_recovery_keeps_incomplete_teacher_masks_and_original_counts(tmp_path: Path) -> None:
    source = [sample(0)]
    cache = cache_for(tmp_path / 'teacher.sqlite3', source, True)
    masked = [replace(row, predictions=tuple(replace(prediction, loss_eligible=False) for prediction in row.predictions)) for row in source]
    connection = staged(tmp_path / 'interrupted.sqlite3', masked)
    connection.execute('BEGIN EXCLUSIVE')
    coverage = audit_ingested(connection, (cache,), lambda: None)
    connection.commit()
    assert coverage[0]['source_eligible'] == coverage[0]['withheld_incomplete_history'] == len(source[0].predictions)
    assert coverage[0]['fitting'] is False
    assert finalize_membership(connection, lambda: None)['reasons'] == {'source_mask': len(source[0].predictions)}
    connection.close()


def test_native_cache_array_recovers_exact_actor_payload(tmp_path: Path) -> None:
    sequences = [sample(0)]
    path = tmp_path / 'native.sqlite3'
    identity = {'fixture': 'native'}
    with sqlite3.connect(path) as source:
        source.executescript('CREATE TABLE metadata(identity TEXT); CREATE TABLE completion(receipt TEXT); CREATE TABLE sources(sha TEXT,status TEXT,payload BLOB);')
        source.execute('INSERT INTO metadata VALUES (?)', (canonical(identity).decode(),))
        source.execute('INSERT INTO completion VALUES (?)', (canonical({'identity': identity, 'candidate_requests': len(sequences[0].predictions)}).decode(),))
        source.execute('INSERT INTO sources VALUES (?,?,?)', ('a' * 64, 'candidate', zstandard.ZstdCompressor(level=3).compress(orjson.dumps([asdict(row) for row in sequences]))))
    source.close()
    connection = staged(tmp_path / 'interrupted.sqlite3', sequences)
    connection.execute('BEGIN EXCLUSIVE')
    result = audit_ingested(connection, (CandidateCache(path, file_hash(path), 'raw'),), lambda: None)
    assert result[0]['actor_sequences'] == 1 and result[0]['fitting'] is True
    connection.rollback()
    connection.close()


def test_resource_stop_interrupts_long_sql_and_preserves_ingestion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sequences = [renamed(sample(0), f'game-{index:04}') for index in range(1000)]
    connection = staged(tmp_path / 'interrupted.sqlite3', sequences)
    updating = False
    clock = 0.0

    def trace(sql: str) -> None:
        nonlocal updating
        if sql.startswith('UPDATE predictions SET reason='):
            updating = True

    def monotonic() -> float:
        nonlocal clock
        clock += 2
        return clock

    def check() -> None:
        if updating:
            raise TimeoutError('approved resource deadline')

    connection.set_trace_callback(trace)
    monkeypatch.setattr(prepared, 'time', SimpleNamespace(monotonic=monotonic))
    with pytest.raises(TimeoutError, match='approved resource deadline'):
        finalize_membership(connection, check)
    assert not connection.in_transaction
    assert connection.execute('SELECT COUNT(*) FROM sequences').fetchone()[0] == len(sequences)
    assert connection.execute("SELECT COUNT(*) FROM predictions WHERE reason!='pending' OR eligible!=0").fetchone()[0] == 0
    assert connection.execute('SELECT COUNT(*) FROM reservations').fetchone()[0] == 0
    connection.close()


def test_production_publication_recovers_verified_ingestion_and_seals(factory_shard: ReferenceShard, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = factory(factory_shard, tmp_path / 'cache.sqlite3')
    store, inventory, sha = publication(tmp_path, (cache,))

    def interrupt(connection: sqlite3.Connection, check: object) -> dict[str, object]:
        raise InterruptedError('after ingestion')

    with monkeypatch.context() as patch:
        patch.setattr(prepared, 'finalize_membership', interrupt)
        with pytest.raises(InterruptedError):
            publish_prepared(store, (cache,), inventory, sha, lambda: None)
        with pytest.raises(InterruptedError):
            publish_prepared(store, (cache,), inventory, sha, lambda: None, file_hash(Path(prepared.__file__)))
    prior = file_hash(Path(prepared.__file__))
    result = publish_prepared(store, (cache,), inventory, sha, lambda: None, prior)
    assert canonical(verify_run(store.path)) == canonical(result)
    assert result['source_coverage']['sources'] == json.loads((store.path / 'ingestion.json').read_bytes())['sources']
    assert result['training_admitted'] and result['optimizer_updates'] == 0
    assert (store.path / 'ingestion-recovery.json').exists()
    assert len(list(store.path.glob('ingestion-recovery*.json'))) == 2


def test_production_recovery_refuses_live_sql_writer(factory_shard: ReferenceShard, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = factory(factory_shard, tmp_path / 'cache.sqlite3')
    store, inventory, sha = publication(tmp_path, (cache,))

    def interrupt(connection: sqlite3.Connection, check: object) -> dict[str, object]:
        raise InterruptedError('after ingestion')

    with monkeypatch.context() as patch:
        patch.setattr(prepared, 'finalize_membership', interrupt)
        with pytest.raises(InterruptedError):
            publish_prepared(store, (cache,), inventory, sha, lambda: None)
    writer = sqlite3.connect(store.path / 'membership.sqlite3')
    writer.execute('BEGIN EXCLUSIVE')
    try:
        with pytest.raises(sqlite3.OperationalError, match='locked'):
            publish_prepared(store, (cache,), inventory, sha, lambda: None, file_hash(Path(prepared.__file__)))
        assert not (store.path / 'ingestion-recovery.json').exists()
        assert not (store.path / 'manifest.json').exists()
    finally:
        writer.rollback()
        writer.close()
