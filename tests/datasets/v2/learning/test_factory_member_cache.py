from __future__ import annotations

import gzip
import sqlite3
from dataclasses import replace
from pathlib import Path

import orjson
import pytest

from deepmew_ex.datasets.v2.learning.factory_cache import prepare_factory_shard
from deepmew_ex.datasets.v2.learning.factory_member_cache import prepare_factory_members
from deepmew_ex.datasets.v2.learning.prepared import cached_sequences
from deepmew_ex.datasets.v2.learning.recovery_cache import ReferenceShard
from deepmew_ex.registry.store import file_hash
from tests.datasets.v2.learning.test_factory_cache import corpus_inventory, factory_shard, reference_game
from tests.datasets.v2.learning.test_factory_members import interrupted


def member_shard(shard: ReferenceShard, tmp_path: Path, data: bytes) -> ReferenceShard:
    path = tmp_path / 'members.gz'
    path.write_bytes(data)
    return replace(shard, decisions=path, decisions_sha256=file_hash(path))


def test_interrupted_write_then_valid_retry_recovers_one_whole_game(factory_shard: ReferenceShard, tmp_path: Path) -> None:
    raw = factory_shard.decisions.read_bytes()
    shard = member_shard(factory_shard, tmp_path, interrupted(raw[:len(raw) // 2]) + gzip.compress(raw))
    cache = tmp_path / 'candidate.sqlite3'
    result = prepare_factory_members(shard, {(700000, 0): 'test'}, 'a' * 64, cache, lambda: None, lambda row: None)
    assert result['games'] == 1 and result['requests'] == 2
    assert result['member_statuses'] == {'rejected_gzip_span': 1, 'candidate': 1}
    assert result['source_bytes_accounted'] == shard.decisions.stat().st_size
    assert not result['summary_games_without_complete_recovery'] and not result['training_admitted']
    sequences = list(cached_sequences(cache, result['cache_sha256'], 'factory_members', lambda: None))
    assert all(s.split == 'test' for s in sequences)
    assert all(p.row_id.startswith('factory-member-frame:') for s in sequences for p in s.predictions)
    assert all(s.unavailable_post_states == tuple('unavailable-post-state:' + p.row_id for p in s.predictions) for s in sequences)
    assert prepare_factory_members(shard, {(700000, 0): 'test'}, 'a' * 64, cache, lambda: None, lambda row: None) == result


def test_duplicate_complete_members_do_not_duplicate_decisions(factory_shard: ReferenceShard, tmp_path: Path) -> None:
    raw = gzip.compress(factory_shard.decisions.read_bytes())
    shard = member_shard(factory_shard, tmp_path, raw + raw)
    result = prepare_factory_members(shard, {(700000, 0): 'train'}, 'a' * 64, tmp_path / 'cache.sqlite3', lambda: None, lambda row: None)
    assert result['games'] == 1 and result['requests'] == 2
    assert result['member_statuses'] == {'candidate': 1, 'duplicate_verified_game': 1}


def test_crc_valid_incomplete_game_is_preserved_but_not_recovered(factory_shard: ReferenceShard, tmp_path: Path) -> None:
    shard = member_shard(factory_shard, tmp_path, gzip.compress(factory_shard.decisions.read_bytes().splitlines(keepends=True)[0]))
    result = prepare_factory_members(shard, {(700000, 0): 'train'}, 'a' * 64, tmp_path / 'cache.sqlite3', lambda: None, lambda row: None)
    assert result['games'] == 0 and result['requests'] == 0
    assert len(result['summary_games_without_complete_recovery']) == 1 and result['member_statuses'] == {'incomplete_game': 1}


def test_conflicting_complete_retries_require_investigation(factory_shard: ReferenceShard, tmp_path: Path) -> None:
    raw = factory_shard.decisions.read_bytes()
    rows = [orjson.loads(line) for line in raw.splitlines()]
    rows[0]['source_annotation'] = 'different'
    other = b''.join(orjson.dumps(row) + b'\n' for row in rows)
    shard = member_shard(factory_shard, tmp_path, gzip.compress(raw) + gzip.compress(other))
    with pytest.raises(ValueError, match='conflicting source content'):
        prepare_factory_members(shard, {(700000, 0): 'train'}, 'a' * 64, tmp_path / 'cache.sqlite3', lambda: None, lambda row: None)


def test_verified_parent_features_are_reused_without_reconversion(factory_shard: ReferenceShard, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import deepmew_ex.datasets.v2.learning.factory_member_cache as module
    raw = factory_shard.decisions.read_bytes()
    second = [{**orjson.loads(line), 'game_id': 'second-game', 'game_index': 1, 'seed': 700001} for line in raw.splitlines()]
    first_summary = orjson.loads(factory_shard.summaries.read_bytes())
    factory_shard.summaries.write_bytes(orjson.dumps(first_summary) + b'\n' + orjson.dumps({**first_summary, 'game_id': 'second-game', 'game_index': 1, 'seed': 700001}) + b'\n')
    shard = member_shard(replace(factory_shard, summaries_sha256=file_hash(factory_shard.summaries), games=2, decision_rows=4), tmp_path, gzip.compress(raw) + gzip.compress(b''.join(orjson.dumps(row) + b'\n' for row in second)))
    splits = {(700000, 0): 'test', (700001, 1): 'test'}
    parent = tmp_path / 'parent.sqlite3'
    prior = prepare_factory_shard(shard, splits, 'a' * 64, parent, lambda: None, lambda row: None)
    original = list(cached_sequences(parent, prior['cache_sha256'], 'factory', lambda: None))
    fresh_path = tmp_path / 'fresh.sqlite3'
    fresh_receipt = prepare_factory_members(shard, splits, 'a' * 64, fresh_path, lambda: None, lambda row: None)
    fresh = list(cached_sequences(fresh_path, fresh_receipt['cache_sha256'], 'factory_members', lambda: None))
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError('must reuse already verified parent features')
    monkeypatch.setattr(module, 'recover_factory_source_game', forbidden)
    path = tmp_path / 'cache.sqlite3'
    result = prepare_factory_members(shard, splits, 'a' * 64, path, lambda: None, lambda row: None, parent, prior['cache_sha256'])
    current = list(cached_sequences(path, result['cache_sha256'], 'factory_members', lambda: None))
    assert result['reused_parent_games'] == 2
    assert [(s.records, s.events, s.split) for s in original] == [(s.records, s.events, s.split) for s in current]
    assert current == fresh


@pytest.mark.parametrize('mutation', ["UPDATE sequences SET payload=X'00'", "UPDATE games SET split='train'", "UPDATE members SET end=end-1"])
def test_changed_membership_or_byte_partition_cannot_resume(factory_shard: ReferenceShard, tmp_path: Path, mutation: str) -> None:
    shard = member_shard(factory_shard, tmp_path, gzip.compress(factory_shard.decisions.read_bytes()))
    cache = tmp_path / 'cache.sqlite3'
    prepare_factory_members(shard, {(700000, 0): 'test'}, 'a' * 64, cache, lambda: None, lambda row: None)
    connection = sqlite3.connect(cache)
    connection.execute(mutation); connection.commit(); connection.close()
    with pytest.raises(ValueError, match='committed features|byte partition'):
        prepare_factory_members(shard, {(700000, 0): 'test'}, 'a' * 64, cache, lambda: None, lambda row: None)


def test_unresolved_lineage_and_changed_source_fail_before_output(factory_shard: ReferenceShard, tmp_path: Path) -> None:
    shard = member_shard(factory_shard, tmp_path, gzip.compress(factory_shard.decisions.read_bytes()))
    output = tmp_path / 'absent.sqlite3'
    with pytest.raises(ValueError, match='lineage'):
        prepare_factory_members(shard, {}, 'a' * 64, output, lambda: None, lambda row: None)
    assert not output.exists()
    with pytest.raises(ValueError, match='source identity'):
        prepare_factory_members(replace(shard, decisions_sha256='0' * 64), {(700000, 0): 'train'}, 'a' * 64, output, lambda: None, lambda row: None)
    assert not output.exists()
