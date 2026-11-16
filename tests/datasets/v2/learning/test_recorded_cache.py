from __future__ import annotations

import copy
import os
import sqlite3
from pathlib import Path

import orjson
import pytest
import zstandard

from deepmew_ex.datasets.v2.learning.recorded_cache import LockedReferenceIndex, prepare_recorded_source, recorded_games
from deepmew_ex.datasets.v2.learning.prepared import cached_sequences
from deepmew_ex.datasets.v2.schema import Payload
from deepmew_ex.registry.store import file_hash
from tests.datasets.v2.learning.test_recorded_actor import recorded
from tests.datasets.v2.learning.test_recovery import reference_game
from tests.datasets.v2.corpus.conftest import corpus_inventory


@pytest.fixture
def source_input(recorded: tuple[list[Payload], list[Payload]], tmp_path: Path) -> tuple[Path, str, Path, str]:
    frames, rows = recorded
    source, index = tmp_path / 'source.jsonl', tmp_path / 'index.sqlite3'
    source.write_bytes(b''.join(orjson.dumps(row) + b'\n' for row in rows))
    sha = file_hash(source)
    with sqlite3.connect(index) as connection:
        connection.executescript('CREATE TABLE sources(path TEXT,sha TEXT); CREATE TABLE refs(source TEXT,line INTEGER,payload BLOB);')
        connection.execute('INSERT INTO sources VALUES (?,?)', (str(source), sha))
        for frame in frames:
            frame['state_before'].update({'source_path': str(source), 'source_sha256': sha})
            connection.execute('INSERT INTO refs VALUES (?,?,?)', (str(source), frame['state_before']['source_line'], zstandard.ZstdCompressor().compress(orjson.dumps(frame))))
    connection.close()
    return source, sha, index, file_hash(index)


def test_recorded_cache_complete_membership_and_idempotent_resume(source_input: tuple[Path, str, Path, str], tmp_path: Path) -> None:
    output = tmp_path / 'candidate.sqlite3'
    receipt = prepare_recorded_source(*source_input, output, lambda: None, lambda value: None)
    assert receipt['requests'] == 2 and receipt['games'] == receipt['sequences'] == receipt['eligible_before_global_admission'] == 1
    assert not receipt['training_admitted']
    assert prepare_recorded_source(*source_input, output, lambda: None, lambda value: None) == receipt


@pytest.mark.parametrize('mutation', ["DELETE FROM sequences", 'UPDATE sequences SET eligible=99', "UPDATE games SET post_receipts='changed'", "UPDATE sequences SET payload=X'00'"])
def test_recorded_resume_rejects_unknown_modified_effects(source_input: tuple[Path, str, Path, str], tmp_path: Path, mutation: str) -> None:
    output = tmp_path / 'candidate.sqlite3'
    prepare_recorded_source(*source_input, output, lambda: None, lambda value: None)
    with sqlite3.connect(output) as connection:
        connection.execute(mutation)
    with pytest.raises(ValueError, match='resume membership changed'):
        prepare_recorded_source(*source_input, output, lambda: None, lambda value: None)


def test_reused_native_label_is_separated_by_exact_game_index(tmp_path: Path) -> None:
    source = tmp_path / 'reused.jsonl'
    source.write_bytes(b'\n'.join(orjson.dumps({'game_id': 'reused', 'game_index': index}) for index in (0, 0, 1, 1)))
    groups = list(recorded_games(source, lambda: None))
    assert [(first, len(rows), rows[0]['game_index']) for first, rows in groups] == [(1, 2, 0), (3, 2, 1)]


@pytest.mark.skipif(os.name != 'nt', reason='Windows write-sharing enforcement')
def test_locked_index_hashes_once_and_denies_writes_through_two_sources(source_input: tuple[Path, str, Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import deepmew_ex.datasets.v2.learning.recorded_cache as module
    source, source_sha, index, index_sha = source_input
    calls: list[Path] = []
    original = module.file_hash
    def tracked(path: Path) -> str:
        calls.append(path.resolve())
        return original(path)
    monkeypatch.setattr(module, 'file_hash', tracked)
    with LockedReferenceIndex(index, index_sha, lambda: None) as locked:
        with pytest.raises(PermissionError):
            index.open('r+b')
        with pytest.raises(PermissionError):
            index.rename(tmp_path / 'moved.sqlite3')
        for ordinal in range(2):
            result = prepare_recorded_source(source, source_sha, index, index_sha, tmp_path / f'candidate-{ordinal}.sqlite3', lambda: None, lambda row: None, locked)
            assert result['games'] == 1 and result['requests'] == 2
        assert calls.count(index.resolve()) == 1
    with index.open('r+b') as handle:
        assert handle.read(16) == b'SQLite format 3\x00'
    with pytest.raises(ValueError, match='inactive'):
        locked.verify(index, index_sha)


@pytest.mark.skipif(os.name != 'nt', reason='Windows write-sharing enforcement')
def test_locked_index_rejects_wrong_identity_and_releases_failed_lock(source_input: tuple[Path, str, Path, str], tmp_path: Path) -> None:
    _, _, index, index_sha = source_input
    with pytest.raises(ValueError, match='identity'):
        with LockedReferenceIndex(index, '0' * 64, lambda: None):
            pytest.fail('changed index accepted')
    with LockedReferenceIndex(index, index_sha, lambda: None) as locked:
        with pytest.raises(ValueError, match='mismatched'):
            locked.verify(tmp_path / 'other.sqlite3', index_sha)
        with pytest.raises(ValueError, match='mismatched'):
            locked.verify(index, '0' * 64)
        wal = Path(str(index) + '-wal')
        wal.write_bytes(b'unexpected')
        with pytest.raises(ValueError, match='WAL'):
            locked.verify(index, index_sha)
        wal.unlink()


@pytest.mark.parametrize('defect', [None, 'missing_reference', 'wrong_source_line', 'reordered_decision'])
def test_interleaved_canonical_games_preserve_order_masks_and_splits(recorded: tuple[list[Payload], list[Payload]], tmp_path: Path, defect: str | None) -> None:
    frames, rows = recorded
    other_frames, other_rows = copy.deepcopy(frames), copy.deepcopy(rows)
    for frame, row in zip(other_frames, other_rows, strict=True):
        row['game_index'] += 1
        row['seed'] += 1
        frame['episode_id'] += ':second'
        frame['frame_id'] += ':second'
        frame['native_game_binding']['source_game_index'] = row['game_index']
        frame['native_game_binding']['canonical_native_game_id'] += ':second'
        frame['effective_split'] = 'holdout'
    pairs = [(frames[0], rows[0]), (other_frames[0], other_rows[0]), (frames[1], rows[1]), (other_frames[1], other_rows[1])]
    if defect == 'reordered_decision':
        pairs[0][1]['decision_index'] = 1
    source, index = tmp_path / 'interleaved.jsonl', tmp_path / 'index.sqlite3'
    source.write_bytes(b''.join(orjson.dumps(row) + b'\n' for _, row in pairs))
    source_sha = file_hash(source)
    connection = sqlite3.connect(index)
    connection.executescript('CREATE TABLE sources(path TEXT,sha TEXT); CREATE TABLE refs(source TEXT,line INTEGER,payload BLOB);')
    connection.execute('INSERT INTO sources VALUES (?,?)', (str(source), source_sha))
    for line, (frame, _) in enumerate(pairs, 1):
        if defect == 'missing_reference' and line == 2:
            continue
        frame['state_before'].update(source_path=str(source), source_sha256=source_sha, source_line=99 if defect == 'wrong_source_line' and line == 1 else line)
        connection.execute('INSERT INTO refs VALUES (?,?,?)', (str(source), line, zstandard.ZstdCompressor().compress(orjson.dumps(frame))))
    connection.commit(); connection.close()
    output = tmp_path / 'candidate.sqlite3'
    if defect:
        with pytest.raises(ValueError):
            prepare_recorded_source(source, source_sha, index, file_hash(index), output, lambda: None, lambda row: None)
        return
    result = prepare_recorded_source(source, source_sha, index, file_hash(index), output, lambda: None, lambda row: None)
    assert result['games'] == 2 and result['requests'] == 4 and result['eligible_before_global_admission'] == 2
    sequences = list(cached_sequences(output, result['cache_sha256'], 'recorded', lambda: None))
    assert {s.split for s in sequences} == {frames[0]['effective_split'], 'holdout'}
    assert all([p.loss_eligible for p in s.predictions] == [True, False] for s in sequences)
    assert {tuple(p.row_id for p in s.predictions) for s in sequences} == {tuple(f['frame_id'] for f in frames), tuple(f['frame_id'] for f in other_frames)}
    assert not list(tmp_path.glob('recorded-game-stage-*'))
