from __future__ import annotations

import gzip
import sqlite3
from pathlib import Path

import orjson
import pytest

from deepmew_ex.datasets.v2.learning.factory_cache import factory_manifest_shards, prepare_factory_shard
from deepmew_ex.datasets.v2.learning.recovery_cache import ReferenceShard
from deepmew_ex.datasets.v2.schema import Payload
from deepmew_ex.registry.store import file_hash
from tests.datasets.v2.corpus.conftest import corpus_inventory
from tests.datasets.v2.learning.test_recovery import reference_game


@pytest.fixture
def factory_shard(reference_game: tuple[list[Payload], list[Payload], Payload], tmp_path: Path) -> ReferenceShard:
    frames, rows, summary = reference_game
    for frame, row in zip(frames, rows, strict=True):
        row['legal_set'] = frame['legal_actions']
    source, summaries = tmp_path / 'source.jsonl', tmp_path / 'summaries.jsonl'
    source.write_bytes(b''.join(orjson.dumps(row) + b'\n' for row in rows))
    summaries.write_bytes(orjson.dumps(summary) + b'\n')
    return ReferenceShard(source, file_hash(source), summaries, file_hash(summaries), len(rows), 1)


def test_factory_cache_preserves_explicit_reserved_split(factory_shard: ReferenceShard, tmp_path: Path) -> None:
    output = tmp_path / 'cache.sqlite3'
    receipt = prepare_factory_shard(factory_shard, {(700000, 0): 'test'}, 'a' * 64, output, lambda: None, lambda row: None)
    assert receipt['games'] == 1 and receipt['requests'] == 2 and not receipt['training_admitted']
    with sqlite3.connect(output) as connection:
        assert connection.execute('SELECT split FROM games').fetchone() == ('test',)
    connection.close()
    assert prepare_factory_shard(factory_shard, {(700000, 0): 'test'}, 'a' * 64, output, lambda: None, lambda row: None) == receipt


def test_unresolved_lineage_fails_before_output_allocation(factory_shard: ReferenceShard, tmp_path: Path) -> None:
    output = tmp_path / 'absent.sqlite3'
    with pytest.raises(ValueError, match='unresolved lineage'):
        prepare_factory_shard(factory_shard, {}, 'a' * 64, output, lambda: None, lambda row: None)
    assert not output.exists()


@pytest.mark.parametrize('mutation', ["UPDATE sequences SET payload=X'00'", "DELETE FROM sequences", "UPDATE games SET split='train'"])
def test_factory_resume_rejects_committed_membership_changes(factory_shard: ReferenceShard, tmp_path: Path, mutation: str) -> None:
    output = tmp_path / 'cache.sqlite3'
    prepare_factory_shard(factory_shard, {(700000, 0): 'test'}, 'a' * 64, output, lambda: None, lambda row: None)
    with sqlite3.connect(output) as connection:
        connection.execute(mutation)
    connection.close()
    with pytest.raises(ValueError, match='committed membership'):
        prepare_factory_shard(factory_shard, {(700000, 0): 'test'}, 'a' * 64, output, lambda: None, lambda row: None)


def test_factory_resume_does_not_recover_already_committed_games(factory_shard: ReferenceShard, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import deepmew_ex.datasets.v2.learning.factory_cache as cache
    output = tmp_path / 'cache.sqlite3'

    def interrupt() -> None:
        if not output.exists():
            return
        connection = sqlite3.connect(output)
        count = connection.execute('SELECT COUNT(*) FROM games').fetchone()[0]
        connection.close()
        if count:
            raise InterruptedError('complete game preserved')

    with pytest.raises(InterruptedError):
        prepare_factory_shard(factory_shard, {(700000, 0): 'test'}, 'a' * 64, output, interrupt, lambda row: None)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError('committed game should be reused')

    monkeypatch.setattr(cache, 'recover_factory_source_game', forbidden)
    assert prepare_factory_shard(factory_shard, {(700000, 0): 'test'}, 'a' * 64, output, lambda: None, lambda row: None)['games'] == 1


def test_factory_resume_rejects_new_split_policy(factory_shard: ReferenceShard, tmp_path: Path) -> None:
    output = tmp_path / 'cache.sqlite3'
    prepare_factory_shard(factory_shard, {(700000, 0): 'test'}, 'a' * 64, output, lambda: None, lambda row: None)
    with pytest.raises(ValueError, match='lineage identity changed'):
        prepare_factory_shard(factory_shard, {(700000, 0): 'train'}, 'a' * 64, output, lambda: None, lambda row: None)


def original_manifest(root: Path) -> dict[str, object]:
    shards = root / 'shards'
    shards.mkdir()
    hashes = {}
    for ordinal, requests in enumerate((2, 3)):
        path = shards / f'games_shard_{ordinal:03d}.jsonl.gz'
        with gzip.open(path, 'wb') as handle:
            handle.write(orjson.dumps({'game_id': f'game-{ordinal}', 'seed': 100 + ordinal, 'game_index': ordinal, 'decisions': requests}) + b'\n')
        hashes[f'.research\\old\\shards\\{path.name}'] = 'sha256:' + file_hash(path)
        hashes[f'.research\\old\\shards\\decisions_shard_{ordinal:03d}.jsonl.gz'] = 'sha256:' + str(ordinal) * 64
    return {'schema_version': 'deepmew.refgames_factory.manifest.v0', 'status': 'complete', 'workers': 2, 'target_games': 2, 'summary': {'shard_hashes': hashes, 'target_games': 2, 'completed_games': 2, 'game_rows': 2, 'decision_rows': 5}}


def test_original_manifest_uses_hashes_and_verified_summaries_without_new_schema(tmp_path: Path) -> None:
    manifest = original_manifest(tmp_path)
    path = tmp_path / 'manifest.json'
    path.write_bytes(orjson.dumps(manifest))
    result = factory_manifest_shards(tmp_path, file_hash(path), lambda: None)
    assert [(row.games, row.decision_rows) for row in result] == [(1, 2), (1, 3)]
    assert result[1].decisions_sha256 == '1' * 64
    assert not result[0].decisions.exists()  # Body validation belongs to recovery, not this metadata adapter.


@pytest.mark.parametrize('defect', ['manifest_hash', 'summary_hash', 'count', 'duplicate_basename', 'missing_shard', 'truncated_summary', 'duplicate_game'])
def test_original_manifest_rejects_identity_population_and_container_errors(tmp_path: Path, defect: str) -> None:
    manifest = original_manifest(tmp_path)
    path = tmp_path / 'manifest.json'
    hashes = manifest['summary']['shard_hashes']
    summary_path = tmp_path / 'shards/games_shard_001.jsonl.gz'
    if defect == 'count':
        manifest['summary']['decision_rows'] = 99
    elif defect == 'duplicate_basename':
        hashes['different/games_shard_000.jsonl.gz'] = 'a' * 64
    elif defect == 'missing_shard':
        del hashes['.research\\old\\shards\\decisions_shard_001.jsonl.gz']
    elif defect in ('summary_hash', 'truncated_summary'):
        summary_path.write_bytes(summary_path.read_bytes()[:-8])
        if defect == 'truncated_summary':
            hashes['.research\\old\\shards\\games_shard_001.jsonl.gz'] = file_hash(summary_path)
    elif defect == 'duplicate_game':
        with gzip.open(summary_path, 'wb') as handle:
            handle.write(orjson.dumps({'game_id': 'game-0', 'seed': 101, 'game_index': 1, 'decisions': 3}) + b'\n')
        hashes['.research\\old\\shards\\games_shard_001.jsonl.gz'] = file_hash(summary_path)
    path.write_bytes(orjson.dumps(manifest))
    with pytest.raises((ValueError, EOFError)):
        factory_manifest_shards(tmp_path, 'f' * 64 if defect == 'manifest_hash' else file_hash(path), lambda: None)
