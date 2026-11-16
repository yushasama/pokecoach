from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from deepmew_ex.datasets.v2.learning.factory_cache import prepare_factory_shard
from deepmew_ex.datasets.v2.learning.prepared import cached_sequences
from deepmew_ex.datasets.v2.learning.recovery_cache import ReferenceShard
from deepmew_ex.registry.store import file_hash
from tests.datasets.v2.learning.test_factory_cache import corpus_inventory, factory_shard, reference_game


def complete(factory_shard: ReferenceShard, output: Path) -> str:
    return prepare_factory_shard(factory_shard, {(700000, 0): 'test'}, 'a' * 64, output, lambda: None, lambda row: None)['cache_sha256']


def test_completed_candidate_stream_preserves_all_context_and_original_split(factory_shard: ReferenceShard, tmp_path: Path) -> None:
    output = tmp_path / 'cache.sqlite3'
    sha = complete(factory_shard, output)
    sequences = list(cached_sequences(output, sha, 'factory', lambda: None))
    assert sum(len(sequence.predictions) for sequence in sequences) == factory_shard.decision_rows
    assert all(sequence.split == 'test' and sequence.history_available for sequence in sequences)


@pytest.mark.parametrize('defect', ['hash', 'incomplete', 'orphan', 'wrong_producer', 'live_writer'])
def test_unfinished_or_inconsistent_candidate_cannot_reach_membership(factory_shard: ReferenceShard, tmp_path: Path, defect: str) -> None:
    output = tmp_path / 'cache.sqlite3'
    sha = complete(factory_shard, output)
    if defect == 'hash':
        sha = '0' * 64
    elif defect == 'live_writer':
        Path(str(output) + '-wal').write_bytes(b'active')
    elif defect != 'wrong_producer':
        connection = sqlite3.connect(output)
        connection.execute('DELETE FROM completion' if defect == 'incomplete' else 'DELETE FROM games')
        connection.commit()
        connection.close()
        sha = file_hash(output)
    with pytest.raises(ValueError, match='identity|completed|population|contract'):
        list(cached_sequences(output, sha, 'recorded' if defect == 'wrong_producer' else 'factory', lambda: None))
