from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from deepmew_ex.datasets.v2.learning.factory_cache import prepare_factory_shard
from deepmew_ex.datasets.v2.learning.history import QUERY_INPUT_CONTRACT
from deepmew_ex.datasets.v2.learning.prepared import CandidateCache, PreparedCorpus, publish_prepared
from deepmew_ex.datasets.v2.learning.recorded_cache import prepare_recorded_source
from deepmew_ex.datasets.v2.learning.recovery_cache import ReferenceShard
from deepmew_ex.registry.store import RunStore, canonical, file_hash
from tests.datasets.v2.corpus.conftest import corpus_inventory
from tests.datasets.v2.learning.test_factory_cache import factory_shard
from tests.datasets.v2.learning.test_recorded_actor import recorded
from tests.datasets.v2.learning.test_recorded_cache import source_input
from tests.datasets.v2.learning.test_recovery import reference_game


def publication(tmp_path: Path, caches: tuple[CandidateCache, ...]) -> tuple[RunStore, Path, str]:
    evidence = tmp_path / 'coverage-evidence.json'
    evidence.write_bytes(canonical({'scope': 'fixture source universe', 'caches': len(caches)}))
    original = tmp_path / 'original'
    original.mkdir()
    RunStore(original).write('port.json', {'catalog_rows': 2 * len(caches)})
    RunStore(original).seal({'kind': 'v2_full_corpus_port'})
    plan = {'candidate_caches': [{'path': str(cache.path.resolve()), 'sha256': cache.sha256, 'kind': cache.kind} for cache in caches], 'source_reconciliation_complete': True, 'input_contract': QUERY_INPUT_CONTRACT, 'coverage_evidence': [{'path': str(evidence), 'sha256': file_hash(evidence)}], 'original_corpus': {'manifest': str(original / 'manifest.json'), 'sha256': file_hash(original / 'manifest.json')}}
    inventory = tmp_path / 'inventory.json'
    inventory.write_bytes(canonical(plan))
    output = tmp_path / 'published'
    output.mkdir()
    return RunStore(output), inventory, file_hash(inventory)


def factory(factory_shard: ReferenceShard, path: Path) -> CandidateCache:
    receipt = prepare_factory_shard(factory_shard, {(700000, 0): 'train'}, 'a' * 64, path, lambda: None, lambda row: None)
    return CandidateCache(path, receipt['cache_sha256'], 'factory')


def test_actual_candidate_cache_publishes_and_trainer_reader_preserves_full_history(factory_shard: ReferenceShard, tmp_path: Path) -> None:
    cache = factory(factory_shard, tmp_path / 'factory.sqlite3')
    store, inventory, sha = publication(tmp_path, (cache,))
    result = publish_prepared(store, (cache,), inventory, sha, lambda: None)
    assert result['training_admitted'] and result['optimizer_updates'] == 0
    reader = PreparedCorpus(store.path)
    try:
        sequences = list(reader.sequences('train'))
        assert sum(len(sequence.predictions) for sequence in sequences) == factory_shard.decision_rows
        assert all(len(sequence.events) == 2 * len(sequence.predictions) for sequence in sequences)
        coverage = json.loads((store.path / 'coverage.json').read_bytes())
        shape = coverage['sources'][0]['shapes_by_original_split']['train']
        assert shape['events'] == sum(len(sequence.events) for sequence in sequences)
        assert shape['max_leaves'] == max(sum(map(len, sequence.records)) for sequence in sequences)
    finally:
        reader.close()


def test_incomplete_teacher_histories_are_retained_but_cannot_enter_training_loss(source_input: tuple[Path, str, Path, str], tmp_path: Path) -> None:
    path = tmp_path / 'teacher.sqlite3'
    receipt = prepare_recorded_source(*source_input, path, lambda: None, lambda row: None)
    cache = CandidateCache(path, receipt['cache_sha256'], 'recorded')
    store, inventory, sha = publication(tmp_path, (cache,))
    with pytest.raises(ValueError, match='no eligible training'):
        publish_prepared(store, (cache,), inventory, sha, lambda: None)
    assert not (store.path / 'manifest.json').exists()
    with sqlite3.connect(store.path / 'membership.sqlite3') as connection:
        assert connection.execute('SELECT COUNT(*),SUM(eligible) FROM predictions').fetchone() == (receipt['requests'], 0)


@pytest.mark.parametrize('defect', ['omitted_cache', 'duplicate_cache', 'changed_inventory', 'unreconciled', 'missing_evidence', 'changed_evidence', 'unfinished_cache', 'original_manifest', 'original_report'])
def test_publication_refuses_incomplete_or_changed_evidence(factory_shard: ReferenceShard, tmp_path: Path, defect: str) -> None:
    cache = factory(factory_shard, tmp_path / 'factory.sqlite3')
    caches = (cache,)
    store, inventory, sha = publication(tmp_path, caches)
    plan = json.loads(inventory.read_bytes())
    if defect == 'omitted_cache':
        caches = ()
    elif defect == 'duplicate_cache':
        caches = (cache, cache)
    elif defect == 'changed_inventory':
        inventory.write_bytes(b'changed')
    elif defect in ('unreconciled', 'missing_evidence'):
        plan['source_reconciliation_complete' if defect == 'unreconciled' else 'coverage_evidence'] = False if defect == 'unreconciled' else []
        inventory.write_bytes(canonical(plan))
        sha = file_hash(inventory)
    elif defect == 'changed_evidence':
        Path(plan['coverage_evidence'][0]['path']).write_bytes(b'changed')
    elif defect in ('original_manifest', 'original_report'):
        (tmp_path / 'original' / ('manifest.json' if defect == 'original_manifest' else 'port.json')).write_bytes(b'changed')
    else:
        with sqlite3.connect(cache.path) as connection:
            connection.execute('DELETE FROM completion')
        caches = (CandidateCache(cache.path, file_hash(cache.path), cache.kind),)
        plan['candidate_caches'][0]['sha256'] = caches[0].sha256
        inventory.write_bytes(canonical(plan))
        sha = file_hash(inventory)
    with pytest.raises(ValueError):
        publish_prepared(store, caches, inventory, sha, lambda: None)
    assert not (store.path / 'manifest.json').exists()


def test_recovery_publication_production_trainer_and_checkpoint_reload(factory_shard: ReferenceShard, tmp_path: Path) -> None:
    from deepmew_ex.experiments.v2.full import FullTrainingConfig, train
    from deepmew_ex.evaluation.league.temporal import load_full_prefix_checkpoint
    training = factory(factory_shard, tmp_path / 'factory.sqlite3')
    rows = [json.loads(line) for line in factory_shard.decisions.read_bytes().splitlines()]
    summary = json.loads(factory_shard.summaries.read_bytes())
    for row in rows:
        row.update({'game_id':'validation-game','game_index':1,'seed':700001})
        row['raw_observation']['current']['turn'] += 1
    summary.update({'game_id':'validation-game','game_index':1,'seed':700001})
    decisions, summaries = tmp_path / 'validation.jsonl', tmp_path / 'validation-summaries.jsonl'
    decisions.write_bytes(b''.join(json.dumps(row).encode() + b'\n' for row in rows))
    summaries.write_bytes(json.dumps(summary).encode() + b'\n')
    shard = ReferenceShard(decisions, file_hash(decisions), summaries, file_hash(summaries), len(rows), 1)
    path = tmp_path / 'validation.sqlite3'
    receipt = prepare_factory_shard(shard, {(700001, 1):'validation'}, 'a' * 64, path, lambda: None, lambda row: None)
    validation = CandidateCache(path, receipt['cache_sha256'], 'factory')
    caches = (training, validation)
    store, inventory, sha = publication(tmp_path, caches)
    publish_prepared(store, caches, inventory, sha, lambda: None)
    config = FullTrainingConfig(str(tmp_path / 'original'), str(tmp_path / 'outputs'), device='cpu', hidden=16, embedding=4, layers=1, heads=2, epochs=2, temporal_batch_size=2, checkpoint_updates=2, minimum_free_bytes=1, max_wall_seconds=120, prepared_corpus=str(store.path))
    result = train(config, 'published-source-two-epochs')
    expected = json.loads((result / 'expected-membership.json').read_bytes())
    assert expected['expected_presentations'] == 2 * expected['unique_eligible_decisions'] > 0
    with sqlite3.connect((result / 'consumption.sqlite3').as_uri() + '?mode=ro&immutable=1', uri=True) as ledger:
        assert ledger.execute('SELECT COUNT(*) FROM consumed').fetchone()[0] == expected['expected_presentations']
    ledger.close()
    selected = json.loads((result / 'selection.json').read_bytes())['selected']
    model, evidence = load_full_prefix_checkpoint(Path(selected['checkpoint']), selected['sha256'], 'cpu')
    assert evidence['input_contract'] == QUERY_INPUT_CONTRACT
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_interrupted_publication_remains_unsealed_and_keeps_original_cache(factory_shard: ReferenceShard, tmp_path: Path) -> None:
    cache = factory(factory_shard, tmp_path / 'factory.sqlite3')
    store, inventory, sha = publication(tmp_path, (cache,))
    def stop_after_membership() -> None:
        path = store.path / 'membership.sqlite3'
        if path.exists():
            with sqlite3.connect(path) as connection:
                if connection.execute("SELECT 1 FROM sqlite_master WHERE name='predictions'").fetchone() and connection.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]:
                    raise InterruptedError('cooperative stop')
    with pytest.raises(InterruptedError):
        publish_prepared(store, (cache,), inventory, sha, stop_after_membership)
    assert file_hash(cache.path) == cache.sha256
    assert not (store.path / 'manifest.json').exists()
