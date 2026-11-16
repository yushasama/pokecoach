from __future__ import annotations

import sqlite3
from pathlib import Path

import orjson
import pytest
import zstandard

from deepmew_ex.datasets.v2.learning.recovery_cache import ReferenceShard, prepare_reference_shard
from deepmew_ex.datasets.v2.schema import Payload
from deepmew_ex.registry.store import canonical, digest, file_hash
from tests.datasets.v2.corpus.conftest import corpus_inventory
from tests.datasets.v2.learning.test_recovery import reference_game


@pytest.fixture
def shard_input(reference_game: tuple[list[Payload], list[Payload], Payload], tmp_path: Path) -> tuple[ReferenceShard, Path, str]:
    frames, rows, summary = reference_game
    source, summaries, index = (tmp_path / name for name in ('source.jsonl', 'summaries.jsonl', 'index.sqlite3'))
    source.write_bytes(b''.join(orjson.dumps(row) + b'\n' for row in rows))
    summaries.write_bytes(orjson.dumps(summary) + b'\n')
    compressor = zstandard.ZstdCompressor()
    with sqlite3.connect(index) as connection:
        connection.executescript('CREATE TABLE sources(path TEXT,sha TEXT); CREATE TABLE refs(source TEXT,game TEXT,global_order INTEGER,payload BLOB);')
        connection.execute('INSERT INTO sources VALUES (?,?)', (str(source), file_hash(source)))
        for frame in frames:
            frame['state_before']['source_path'] = str(source)
            frame['state_before']['source_sha256'] = file_hash(source)
            connection.execute('INSERT INTO refs VALUES (?,?,?,?)', (str(source), 'game', frame['global_order'], compressor.compress(orjson.dumps(frame))))
    return ReferenceShard(source, file_hash(source), summaries, file_hash(summaries), 2, 1), index, file_hash(index)


def test_streamed_candidate_cache_preserves_full_history_and_masks(shard_input: tuple[ReferenceShard, Path, str], tmp_path: Path) -> None:
    shard, index, sha = shard_input
    output = tmp_path / 'cache.sqlite3'
    receipt = prepare_reference_shard(shard, index, sha, output, lambda: None, lambda row: None)
    assert receipt['games'] == receipt['sequences'] == receipt['eligible_before_global_admission'] == 1
    assert receipt['requests'] == receipt['missing_immediate_post_states'] == 2
    assert not receipt['training_admitted']
    with sqlite3.connect(output) as connection:
        row = connection.execute('SELECT payload FROM sequences').fetchone()[0]
    sequence = orjson.loads(zstandard.ZstdDecompressor().decompress(row))
    assert len(sequence['events']) == 4
    assert [row['loss_eligible'] for row in sequence['predictions']] == [True, False]
    assert prepare_reference_shard(shard, index, sha, output, lambda: None, lambda row: None) == receipt


def test_interrupted_candidate_recovery_resumes_without_duplicate_games(shard_input: tuple[ReferenceShard, Path, str], tmp_path: Path) -> None:
    shard, index, sha = shard_input
    output = tmp_path / 'cache.sqlite3'

    def interrupt() -> None:
        if not output.exists():
            return
        with sqlite3.connect(output) as connection:
            committed = connection.execute('SELECT COUNT(*) FROM games').fetchone()[0]
        if committed:
            raise InterruptedError('after game commit')

    with pytest.raises(InterruptedError, match='game commit'):
        prepare_reference_shard(shard, index, sha, output, interrupt, lambda row: None)
    receipt = prepare_reference_shard(shard, index, sha, output, lambda: None, lambda row: None)
    assert receipt['games'] == receipt['sequences'] == 1


@pytest.mark.parametrize('target', ['source', 'index'])
def test_changed_input_is_rejected_before_output_allocation(shard_input: tuple[ReferenceShard, Path, str], tmp_path: Path, target: str) -> None:
    shard, index, sha = shard_input
    (shard.decisions if target == 'source' else index).write_bytes(b'changed')
    output = tmp_path / 'never.sqlite3'
    with pytest.raises(ValueError, match='hash mismatch'):
        prepare_reference_shard(shard, index, sha, output, lambda: None, lambda row: None)
    assert not output.exists()


def test_incomplete_reference_history_is_not_cached(shard_input: tuple[ReferenceShard, Path, str], tmp_path: Path) -> None:
    shard, index, _ = shard_input
    with sqlite3.connect(index) as connection:
        connection.execute('DELETE FROM refs WHERE global_order=1')
    output = tmp_path / 'cache.sqlite3'
    with pytest.raises(ValueError, match='complete source game'):
        prepare_reference_shard(shard, index, file_hash(index), output, lambda: None, lambda row: None)
    with sqlite3.connect(output) as connection:
        assert connection.execute('SELECT COUNT(*) FROM games').fetchone()[0] == 0


@pytest.mark.parametrize('mutation', ["DELETE FROM sequences", "UPDATE sequences SET payload=X'00'", 'UPDATE sequences SET eligible=0'])
def test_resume_rejects_missing_or_altered_sequences(shard_input: tuple[ReferenceShard, Path, str], tmp_path: Path, mutation: str) -> None:
    shard, index, sha = shard_input
    output = tmp_path / 'cache.sqlite3'
    prepare_reference_shard(shard, index, sha, output, lambda: None, lambda row: None)
    with sqlite3.connect(output) as connection:
        connection.execute('DELETE FROM completion')
        connection.execute(mutation)
    with pytest.raises(ValueError, match='payload identity changed'):
        prepare_reference_shard(shard, index, sha, output, lambda: None, lambda row: None)


def test_resume_binds_conversion_dependency(shard_input: tuple[ReferenceShard, Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import deepmew_ex.datasets.v2.learning.recovery_cache as cache
    shard, index, sha = shard_input
    output = tmp_path / 'cache.sqlite3'
    prepare_reference_shard(shard, index, sha, output, lambda: None, lambda row: None)
    original = cache.file_hash
    monkeypatch.setattr(cache, 'file_hash', lambda path: '0' * 64 if path.name == 'conversion.py' else original(path))
    with pytest.raises(ValueError, match='preparation identity changed'):
        prepare_reference_shard(shard, index, sha, output, lambda: None, lambda row: None)


def test_verified_v3_histories_are_enriched_without_repeating_semantic_recovery(shard_input: tuple[ReferenceShard, Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import deepmew_ex.datasets.v2.learning.recovery_cache as cache
    shard, index, sha = shard_input
    parent, child = tmp_path / 'v3.sqlite3', tmp_path / 'v4.sqlite3'
    prepare_reference_shard(shard, index, sha, parent, lambda: None, lambda row: None)
    with sqlite3.connect(parent) as connection:
        actor, events, records, eligible, encoded = connection.execute('SELECT actor,events,records,eligible,payload FROM sequences').fetchone()
        expected = orjson.loads(zstandard.ZstdDecompressor().decompress(encoded))
        previous = orjson.loads(zstandard.ZstdDecompressor().decompress(encoded))
        previous['records'] = [[leaf for leaf in record if not leaf[0].startswith('/actor_queries/')] for record in previous['records']]
        encoded = zstandard.ZstdCompressor(level=3).compress(orjson.dumps(previous))
        connection.execute('UPDATE sequences SET payload=?', (encoded,))
        connection.execute('UPDATE games SET sequence_identity=?', (canonical([(actor, events, records, eligible, digest(encoded))]).decode(),))
        identity = orjson.loads(connection.execute('SELECT identity FROM metadata').fetchone()[0])
        identity['version'] = 3
        identity.pop('query_enrichment_parent')
        identity['code'].pop('learning/queries.py')
        connection.execute('UPDATE metadata SET identity=?', (canonical(identity).decode(),))
        connection.execute('DELETE FROM completion')
    connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    connection.close()
    parent_sha = file_hash(parent)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError('verified parent must not repeat expensive semantic recovery')

    monkeypatch.setattr(cache, 'recover_reference_game', forbidden)
    receipt = prepare_reference_shard(shard, index, sha, child, lambda: None, lambda row: None, parent, parent_sha)
    assert receipt['games_enriched_from_verified_v3'] == 1
    assert file_hash(parent) == parent_sha
    with sqlite3.connect(child) as connection:
        actual = orjson.loads(zstandard.ZstdDecompressor().decompress(connection.execute('SELECT payload FROM sequences').fetchone()[0]))
    assert actual == expected


@pytest.mark.parametrize('target', ['source', 'summary', 'index', 'parent'])
def test_candidate_output_cannot_overwrite_input(shard_input: tuple[ReferenceShard, Path, str], tmp_path: Path, target: str) -> None:
    shard, index, sha = shard_input
    parent = tmp_path / 'parent.sqlite3'
    parent.write_bytes(b'preserved parent')
    output = {'source': shard.decisions, 'summary': shard.summaries, 'index': index, 'parent': parent}[target]
    before = output.read_bytes()
    with pytest.raises(ValueError, match='overwrite a recovery input'):
        prepare_reference_shard(shard, index, sha, output, lambda: None, lambda row: None, parent, file_hash(parent))
    assert output.read_bytes() == before


def test_query_enrichment_rejects_uncheckpointed_parent(shard_input: tuple[ReferenceShard, Path, str], tmp_path: Path) -> None:
    shard, index, sha = shard_input
    parent, child = tmp_path / 'parent.sqlite3', tmp_path / 'child.sqlite3'
    parent.write_bytes(b'parent')
    parent.with_name(parent.name + '-wal').write_bytes(b'uncheckpointed writes')
    with pytest.raises(ValueError, match='uncheckpointed writes'):
        prepare_reference_shard(shard, index, sha, child, lambda: None, lambda row: None, parent, file_hash(parent))
    assert not child.exists()


def test_query_enrichment_rejects_unchanged_dependency_drift(shard_input: tuple[ReferenceShard, Path, str], tmp_path: Path) -> None:
    shard, index, sha = shard_input
    parent, child = tmp_path / 'parent.sqlite3', tmp_path / 'child.sqlite3'
    prepare_reference_shard(shard, index, sha, parent, lambda: None, lambda row: None)
    with sqlite3.connect(parent) as connection:
        identity = orjson.loads(connection.execute('SELECT identity FROM metadata').fetchone()[0])
        identity['version'] = 3
        identity['code'].pop('learning/queries.py')
        identity['code']['learning/actor_events.py'] = '0' * 64
        connection.execute('UPDATE metadata SET identity=?', (canonical(identity).decode(),))
    connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    connection.close()
    before = file_hash(parent)
    with pytest.raises(ValueError, match='unchanged semantic dependency drift'):
        prepare_reference_shard(shard, index, sha, child, lambda: None, lambda row: None, parent, before)
    assert file_hash(parent) == before
