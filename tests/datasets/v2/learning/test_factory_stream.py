from __future__ import annotations

import gzip
import json
import zlib
from dataclasses import replace
from pathlib import Path

import pytest

from deepmew_ex.datasets.v2.learning.factory_cache import prepare_factory_shard
from deepmew_ex.datasets.v2.learning.factory_source import SourceStreamAudit, factory_rows
from deepmew_ex.datasets.v2.learning.recovery_cache import ReferenceShard
from deepmew_ex.registry.store import file_hash
from tests.datasets.v2.learning.test_factory_cache import corpus_inventory, factory_shard, reference_game


def interrupted_gzip(data: bytes) -> bytes:
    encoder = zlib.compressobj(wbits=31)
    return encoder.compress(data) + encoder.flush(zlib.Z_SYNC_FLUSH)


@pytest.mark.parametrize('complete', [False, True])
def test_complete_json_rows_survive_missing_gzip_footer(tmp_path: Path, complete: bool) -> None:
    path = tmp_path / 'source.gz'
    payload = b'{"value":1}\n{"value":2}'
    path.write_bytes(gzip.compress(payload) if complete else interrupted_gzip(payload))
    audit = SourceStreamAudit()
    assert list(factory_rows(path, audit, lambda: None, True)) == [(1, {'value': 1}), (2, {'value': 2})]
    assert audit.container_complete == complete and audit.incomplete_tail_bytes == 0
    if not complete:
        with pytest.raises(EOFError):
            list(factory_rows(path, SourceStreamAudit(), lambda: None))


def test_partial_json_tail_is_audited_and_never_invented(tmp_path: Path) -> None:
    path = tmp_path / 'source.gz'
    path.write_bytes(interrupted_gzip(b'{"value":1}\n{"value":'))
    audit = SourceStreamAudit()
    assert list(factory_rows(path, audit, lambda: None, True)) == [(1, {'value': 1})]
    assert audit.incomplete_tail_bytes == 9 and audit.incomplete_tail_sha256


@pytest.mark.parametrize('defect', ['checksum', 'interior_json', 'closed_invalid_tail', 'nonfinite_json'])
def test_salvage_does_not_ignore_corrupt_compression_or_interior_records(tmp_path: Path, defect: str) -> None:
    path = tmp_path / 'source.gz'
    if defect == 'checksum':
        data = bytearray(gzip.compress(b'{"value":1}\n'))
        data[-8] ^= 1
        path.write_bytes(data)
    elif defect == 'interior_json':
        path.write_bytes(interrupted_gzip(b'{broken}\n{"value":1}\n'))
    elif defect == 'nonfinite_json':
        path.write_bytes(interrupted_gzip(b'{"value":NaN}\n'))
    else:
        path.write_bytes(gzip.compress(b'{"value":'))
    with pytest.raises((ValueError, zlib.error)):
        list(factory_rows(path, SourceStreamAudit(), lambda: None, True))


def test_concatenated_gzip_members_preserve_record_order(tmp_path: Path) -> None:
    path = tmp_path / 'source.gz'
    path.write_bytes(gzip.compress(b'{"value":1}\n') + gzip.compress(b'{"value":2}\n'))
    audit = SourceStreamAudit()
    assert list(factory_rows(path, audit, lambda: None)) == [(1, {'value': 1}), (2, {'value': 2})]
    assert audit.container_complete


@pytest.mark.parametrize('tail', ['footer_only', 'partial_game', 'missing_summary'])
def test_factory_prefix_cache_emits_only_complete_source_supported_games(factory_shard: ReferenceShard, tmp_path: Path, tail: str) -> None:
    decisions, summaries = tmp_path / 'decisions.gz', tmp_path / 'summaries.gz'
    raw = factory_shard.decisions.read_bytes()
    rows = raw.splitlines(keepends=True)
    if tail == 'partial_game':
        raw = rows[0] + rows[1][:20]
    elif tail == 'missing_summary':
        extras = [{**json.loads(row), 'game_id': 'unrecorded-summary', 'game_index': 1, 'seed': 700001} for row in rows]
        raw += b''.join(json.dumps(row).encode() + b'\n' for row in extras)
    decisions.write_bytes(interrupted_gzip(raw))
    summaries.write_bytes(interrupted_gzip(factory_shard.summaries.read_bytes()))
    shard = replace(factory_shard, decisions=decisions, decisions_sha256=file_hash(decisions), summaries=summaries, summaries_sha256=file_hash(summaries))
    output = tmp_path / 'cache.sqlite3'
    receipt = prepare_factory_shard(shard, {(700000, 0): 'test'}, 'a' * 64, output, lambda: None, lambda row: None, allow_truncated=True)
    assert not receipt['source_streams']['decisions']['container_complete']
    assert not receipt['source_streams']['summaries']['container_complete']
    assert receipt['games'] == (0 if tail == 'partial_game' else 1)
    assert receipt['requests'] == (0 if tail == 'partial_game' else factory_shard.decision_rows)
    assert len(receipt['incomplete_games']) == (0 if tail == 'footer_only' else 1)
    assert not receipt['training_admitted']
    assert prepare_factory_shard(shard, {(700000, 0): 'test'}, 'a' * 64, output, lambda: None, lambda row: None, allow_truncated=True) == receipt
