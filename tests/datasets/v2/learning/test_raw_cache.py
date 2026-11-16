from __future__ import annotations

import sqlite3
from pathlib import Path

import orjson
import pytest
import zstandard

from deepmew_ex.datasets.v2.learning.raw_cache import RawSource, freeze_raw_sources, prepare_raw_sources, raw_reference_paths
from deepmew_ex.datasets.v2.learning.prepared import cached_sequences
from deepmew_ex.registry.store import RunStore, canonical, digest, file_hash
from tests.datasets.v2.learning.test_legacy_contract import legacy_replay
from tests.datasets.v2.learning.test_raw_masks import mask_inputs


@pytest.fixture
def raw_inputs(tmp_path: Path) -> tuple[tuple[RawSource, ...], Path]:
    replay = legacy_replay()
    source, manifest = tmp_path / 'replay.json', tmp_path / 'manifest.json'
    raw = canonical(replay)
    source.write_bytes(raw)
    manifest.write_bytes(b'{}')
    return (RawSource(source, digest(raw), replay['id'], '1.30.1', 'validation'),), manifest


def test_raw_candidate_cache_preserves_version_source_split_and_context(raw_inputs: tuple[tuple[RawSource, ...], Path], tmp_path: Path) -> None:
    sources, manifest = raw_inputs
    output = tmp_path / 'cache.sqlite3'
    receipt = prepare_raw_sources(sources, manifest, file_hash(manifest), output, lambda: None, lambda row: None)
    assert receipt['source_counts'] == {'candidate': 1} and receipt['candidate_requests'] == 4
    assert not receipt['training_admitted']
    with pytest.raises(ValueError, match='historical masks remain unreconciled'):
        list(cached_sequences(output, receipt['cache_sha256'], 'raw', lambda: None))
    with sqlite3.connect(output) as connection:
        episode, payload = connection.execute('SELECT episode,payload FROM sources').fetchone()
    connection.close()
    decoder = zstandard.ZstdDecompressor()
    assert orjson.loads(decoder.decompress(episode))['module_version'] == '1.30.1'
    sequences = orjson.loads(decoder.decompress(payload))
    assert all(sequence['split'] == 'validation' for sequence in sequences)
    assert sum(len(sequence['predictions']) for sequence in sequences) == 4
    assert all(sequence['history_available'] for sequence in sequences)
    assert prepare_raw_sources(sources, manifest, file_hash(manifest), output, lambda: None, lambda row: None) == receipt


def test_raw_recovery_resume_reuses_committed_source(raw_inputs: tuple[tuple[RawSource, ...], Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import deepmew_ex.datasets.v2.learning.raw_cache as cache
    sources, manifest = raw_inputs
    output = tmp_path / 'cache.sqlite3'

    def interrupt() -> None:
        if not output.exists():
            return
        connection = sqlite3.connect(output)
        count = connection.execute('SELECT COUNT(*) FROM sources').fetchone()[0]
        connection.close()
        if count:
            raise InterruptedError('after committed source')

    with pytest.raises(InterruptedError):
        prepare_raw_sources(sources, manifest, file_hash(manifest), output, interrupt, lambda row: None)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError('committed replay must not be reconverted')

    monkeypatch.setattr(cache, 'convert_episode', forbidden)
    assert prepare_raw_sources(sources, manifest, file_hash(manifest), output, lambda: None, lambda row: None)['source_counts'] == {'candidate': 1}


@pytest.mark.parametrize('defect', ['payload', 'split', 'source', 'mask_receipt'])
def test_raw_resume_rejects_identity_changes(raw_inputs: tuple[tuple[RawSource, ...], Path], tmp_path: Path, defect: str) -> None:
    sources, manifest = raw_inputs
    output = tmp_path / 'cache.sqlite3'
    prepare_raw_sources(sources, manifest, file_hash(manifest), output, lambda: None, lambda row: None)
    if defect == 'source':
        sources[0].path.write_bytes(b'changed')
    else:
        with sqlite3.connect(output) as connection:
            connection.execute({'payload': "UPDATE sources SET payload=X'00'", 'split': "UPDATE sources SET split='train'", 'mask_receipt': "UPDATE sources SET mask_receipt='{}'"}[defect])
        connection.close()
    with pytest.raises(ValueError, match='identity changed|membership changed'):
        prepare_raw_sources(sources, manifest, file_hash(manifest), output, lambda: None, lambda row: None)


@pytest.mark.parametrize('defect,status', [('specification', 'unsupported'), ('action', 'rejected')])
def test_raw_unsupported_and_rejected_sources_remain_explicit(raw_inputs: tuple[tuple[RawSource, ...], Path], tmp_path: Path, defect: str, status: str) -> None:
    sources, manifest = raw_inputs
    replay = legacy_replay()
    if defect == 'specification':
        replay['specification']['action']['type'] = 'unknown'
    else:
        replay['steps'][1][0]['action'] = [999]
    raw = canonical(replay)
    sources[0].path.write_bytes(raw)
    source = RawSource(sources[0].path, digest(raw), replay['id'], '1.30.1', 'validation')
    receipt = prepare_raw_sources((source,), manifest, file_hash(manifest), tmp_path / 'rejections.sqlite3', lambda: None, lambda row: None)
    assert receipt['source_counts'] == {status: 1} and receipt['candidate_requests'] == 0
    assert len(receipt['reasons']) == 1 and not receipt['training_admitted']


@pytest.mark.parametrize('historical_status', ['quarantined', 'admitted_actor_visible_episode'])
def test_raw_source_plan_preserves_inventory_population_and_reserved_split(raw_inputs: tuple[tuple[RawSource, ...], Path], tmp_path: Path, historical_status: str) -> None:
    sources, _ = raw_inputs
    source = sources[0]
    corpus = tmp_path / 'historical'
    corpus.mkdir()
    entry = {'path': str(source.path), 'sha256': source.sha256, 'byte_size': source.path.stat().st_size, 'status': 'available'}
    result = {'kind': 'raw', 'source': entry, 'status': historical_status}
    if historical_status == 'admitted_actor_visible_episode':
        child = corpus / 'raw'
        child.mkdir()
        episode = {'source_native_game_id': source.native_game_id, 'module_version': source.module_version, 'source_sha256': source.sha256}
        RunStore(child).write_bytes('episode.json.zst', zstandard.ZstdCompressor().compress(canonical(episode)))
        RunStore(child).seal({'kind': 'v2_raw_episode'})
        result.update({'path': str(child), 'manifest_sha256': file_hash(child / 'manifest.json'), 'native_game_id': source.native_game_id, 'effective_split': 'holdout'})
    store = RunStore(corpus)
    store.write('inventory.json', {'replays': [entry, {**entry, 'path': 'duplicate-path', 'alias_of': str(source.path)}], 'raw_splits': {source.native_game_id: 'holdout'}})
    store.write('port.json', {'results': [result], 'raw_native_game_conflicts': {}, 'raw_quarantine_reasons': {'unsupported_source_contract': 1} if historical_status == 'quarantined' else {}})
    store.seal({'kind': 'v2_full_corpus_port'})
    plan, receipt = freeze_raw_sources(corpus, lambda: None, lambda row: None)
    assert plan == (RawSource(source.path, source.sha256, source.native_game_id, source.module_version, 'holdout'),)
    assert receipt['sources'] == 1 and receipt['path_aliases'] == 1 and not receipt['training_admitted']
    if historical_status == 'quarantined':
        source.path.write_bytes(b'changed')
        with pytest.raises(ValueError, match='source changed'):
            freeze_raw_sources(corpus, lambda: None, lambda row: None)
    else:
        (child / 'episode.json.zst').write_bytes(b'changed')
        with pytest.raises(ValueError, match='episode changed'):
            freeze_raw_sources(corpus, lambda: None, lambda row: None)


def test_reference_paths_use_verified_catalog_and_keep_unrepresented_sources(tmp_path: Path) -> None:
    corpus, catalog = tmp_path / 'corpus', tmp_path / 'catalog'
    corpus.mkdir()
    catalog.mkdir()
    source_sha = 'a' * 64
    metadata = {'source_id': 'kaggle_ingest', 'status': 'completed', 'source_fingerprint': [{'path': 'arbitrary-name.jsonl', 'sha256': source_sha}, {'path': 'unrepresented.jsonl', 'sha256': source_sha}]}
    (catalog / 'unit.json').write_bytes(canonical(metadata))
    (catalog / 'catalog.json').write_bytes(canonical({'included_units': [{'source_id': 'kaggle_ingest', 'manifest_path': 'unit.json', 'manifest_sha256': file_hash(catalog / 'unit.json')}]}))
    store = RunStore(corpus)
    store.write('inventory.json', {'catalog': {'path': str(catalog / 'catalog.json'), 'sha256': file_hash(catalog / 'catalog.json')}})
    store.seal({'kind': 'v2_full_corpus_port'})
    connection = sqlite3.connect(':memory:')
    connection.executescript('CREATE TABLE sources(path PRIMARY KEY,sha); CREATE TABLE refs(source,game); CREATE INDEX source_game ON refs(source,game);')
    connection.execute('INSERT INTO sources VALUES (?,?)', ('arbitrary-name.jsonl', source_sha))
    connection.executemany('INSERT INTO refs VALUES (?,?)', [('arbitrary-name.jsonl', 'actual-native')] * 3)
    try:
        mapping, missing = raw_reference_paths(corpus, connection, lambda: None)
        assert mapping == {'actual-native': ('arbitrary-name.jsonl',)}
        assert missing == ({'path': 'unrepresented.jsonl', 'sha256': source_sha},)
        connection.execute("UPDATE sources SET sha='different'")
        with pytest.raises(ValueError, match='source binding'):
            raw_reference_paths(corpus, connection, lambda: None)
    finally:
        connection.close()


@pytest.mark.parametrize('broken_join', [False, True])
def test_raw_cache_applies_historical_masks_and_never_hides_join_corruption(mask_inputs: tuple, tmp_path: Path, broken_join: bool) -> None:
    episode, raw, sequences, references, source_bytes = mask_inputs
    source_path = tmp_path / 'actions.jsonl'
    source_path.write_bytes(source_bytes['table.jsonl'])
    references[0]['target_mask']['source_target_eligible'] = False
    if broken_join:
        references[0]['chosen_action']['indices'] = [999]
    for reference in references:
        reference['state_before']['source_path'] = str(source_path)
    corpus, catalog = tmp_path / 'corpus', tmp_path / 'catalog'
    corpus.mkdir()
    catalog.mkdir()
    (catalog / 'unit.json').write_bytes(canonical({'source_id': 'kaggle_ingest', 'status': 'completed', 'source_fingerprint': [{'path': str(source_path), 'sha256': file_hash(source_path)}]}))
    (catalog / 'catalog.json').write_bytes(canonical({'included_units': [{'source_id': 'kaggle_ingest', 'manifest_path': 'unit.json', 'manifest_sha256': file_hash(catalog / 'unit.json')}]}))
    store = RunStore(corpus)
    store.write('inventory.json', {'catalog': {'path': str(catalog / 'catalog.json'), 'sha256': file_hash(catalog / 'catalog.json')}})
    store.seal({'kind': 'v2_full_corpus_port'})
    index_path = tmp_path / 'refs.sqlite3'
    connection = sqlite3.connect(index_path)
    connection.executescript('CREATE TABLE sources(path PRIMARY KEY,sha); CREATE TABLE refs(id,source,game,family,payload); CREATE INDEX source_game ON refs(source,game);')
    connection.execute('INSERT INTO sources VALUES (?,?)', (str(source_path), file_hash(source_path)))
    encoder = zstandard.ZstdCompressor()
    connection.executemany('INSERT INTO refs VALUES (?,?,?,?,?)', [(reference['frame_id'], str(source_path), episode.source_native_game_id, 'kaggle_ingest', encoder.compress(canonical(reference))) for reference in references])
    connection.commit()
    connection.close()
    replay_path = tmp_path / 'replay.json'
    replay_path.write_bytes(raw)
    source = RawSource(replay_path, digest(raw), episode.source_native_game_id, episode.module_version, 'train')
    output = tmp_path / 'cache.sqlite3'
    args = ((source,), corpus / 'manifest.json', file_hash(corpus / 'manifest.json'), output, lambda: None, lambda row: None, index_path, file_hash(index_path))
    if broken_join:
        with pytest.raises(ValueError, match='historical selected action'):
            prepare_raw_sources(*args)
        connection = sqlite3.connect(output)
        assert connection.execute('SELECT COUNT(*) FROM sources').fetchone()[0] == 0
        assert connection.execute('SELECT COUNT(*) FROM completion').fetchone()[0] == 0
        connection.close()
        return
    receipt = prepare_raw_sources(*args)
    assert receipt['historical_masks']['status'] == 'applied'
    streamed = list(cached_sequences(output, receipt['cache_sha256'], 'raw', lambda: None))
    assert len(streamed) == len(sequences) and not streamed[0].predictions[0].loss_eligible
    connection = sqlite3.connect(output)
    payload, masks = connection.execute('SELECT payload,mask_receipt FROM sources').fetchone()
    connection.close()
    restored = orjson.loads(zstandard.ZstdDecompressor().decompress(payload))
    assert not restored[0]['predictions'][0]['loss_eligible']
    assert orjson.loads(masks)['matched_native_actions'] == len(episode.frames)
    assert restored[0]['records'] == orjson.loads(canonical(sequences[0].records))
