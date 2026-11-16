from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import orjson
import pytest

from deepmew_ex.datasets.v2.corpus.config import CorpusConfig
from deepmew_ex.datasets.v2.corpus.reader import reference_rows, resolve_references, resolve_row
from deepmew_ex.datasets.v2.corpus.records import action_envelope, native_game_binding
from deepmew_ex.datasets.v2.corpus.runner import run_corpus
from deepmew_ex.datasets.v2.corpus.workers import canonical_part, initialize, raw_episode
from deepmew_ex.experiments.config import OutputPaths
from deepmew_ex.registry.store import file_hash, verify_run
from ..conftest import synthetic_replay


def test_enriched_envelope_preserves_identity_and_optional_metadata() -> None:
    legal = [{'index': 0, 'semantic_key': 'YES'}]
    chosen = {'indices': [0], 'semantics': legal, 'complete_action_id': json.dumps({'schema': 'deepmew.complete_legal_option_selection.v0', 'indices': [0], 'semantics': legal})}
    assert action_envelope(legal, chosen)['indices'] == [0]
    assert action_envelope(legal, {'indices': [], 'semantics': [], 'allows_empty_choice': True})['allows_empty_choice'] is True


@pytest.mark.parametrize('failure', ['indices', 'semantics', 'complete', 'optional', 'unknown', 'duplicate'])
def test_invalid_envelopes_fail_closed(failure: str) -> None:
    chosen = {'indices': [0], 'semantics': [{'index': 0, 'semantic_key': 'YES'}]}
    if failure == 'indices': chosen['indices'] = [99]
    elif failure == 'semantics': chosen['semantics'][0]['semantic_key'] = 'NO'
    elif failure == 'complete': chosen['complete_action_id'] = json.dumps({'schema': 'other', 'indices': [0], 'semantics': chosen['semantics']})
    elif failure == 'optional': chosen = {'indices': [], 'allows_empty_choice': False}
    elif failure == 'unknown': chosen['opaque_hidden_state'] = 'PRIVATE'
    else: chosen['indices'] = [0, 0]
    with pytest.raises(ValueError):
        action_envelope([{'index': 0, 'semantic_key': 'YES'}], chosen)


def test_native_game_disambiguation_is_exact() -> None:
    assert native_game_binding('r21_staryu', 'g:game_index=7', 7)['raw_game_id'] == 'g'
    with pytest.raises(ValueError, match='disambiguation_conflict'):
        native_game_binding('r21_staryu', 'g:game_index=7', 8)


def test_resolution_batch_is_bounded_before_source_read() -> None:
    with pytest.raises(ValueError, match='batch must'):
        list(resolve_references([], max_references=0))
    frame = {'schema': 'deepmew.ex.semantic_reference.v2', 'source_schema': 'deepmew.refgames_factory.decision.v0', 'state_before': {'source_path': 'missing.jsonl.gz', 'source_sha256': '0' * 64, 'source_line': 1}}
    with pytest.raises(ValueError, match='batch exceeded'):
        list(resolve_references([frame, frame], max_references=1))


def test_full_port_round_trip_masks_source_splits_and_lazy_resolution(corpus_inventory: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    inventory = orjson.loads(corpus_inventory.read_bytes())
    source = Path(inventory['sources'][0]['path'])
    before = file_hash(source)
    config = CorpusConfig(corpus_inventory, 'test-corpus', OutputPaths(tmp_path / 'outputs'), workers=1, min_free_bytes=0)
    path = run_corpus(config)
    assert file_hash(source) == before
    report = orjson.loads((path / 'port.json').read_bytes())
    assert report['catalog_rows'] == 2 and report['totals']['semantic_references'] == 2
    frames = list(reference_rows(path, 'unified_r2'))
    assert [row['target_mask']['source_target_eligible'] for row in frames] == [True, False]
    assert all(row['source_split'] == 'train' and row['effective_split'] == 'validation' for row in frames)
    assert all(not row['capabilities']['snapshot_bc'] and not row['capabilities']['td_transition'] for row in frames)
    snapshots = list(resolve_references(frames))
    assert len(snapshots) == 2 and snapshots[0].state.actor_private['hand'] == [{'id': 123}]
    assert b'PRIVATE-SENTINEL' not in snapshots[0].state.state_id.encode()
    assert not snapshots[0].td_transition
    from deepmew_ex.cli import main
    main(['dataset', 'report', '--audit-path', str(path)])
    assert json.loads(capsys.readouterr().out)['result']['catalog_rows'] == 2
    with pytest.raises(FileExistsError): run_corpus(config)
    with pytest.raises(ValueError, match='explicit snapshot resolution'): list(reference_rows(tmp_path / 'missing', objective='td_transition'))


def test_snapshot_negative_controls(corpus_inventory: Path, tmp_path: Path) -> None:
    inventory = orjson.loads(corpus_inventory.read_bytes())
    root = tmp_path / 'worker'
    root.mkdir()
    initialize(str(corpus_inventory), str(root))
    result = canonical_part(inventory['parts'][0])
    import io,zstandard
    with (Path(result['path']) / 'frames.jsonl.zst').open('rb') as handle:
        with zstandard.ZstdDecompressor().stream_reader(handle) as stream:
            frame = orjson.loads(next(io.BufferedReader(stream)))
    row = orjson.loads(Path(inventory['sources'][0]['path']).read_bytes().splitlines()[0])
    corrupt = copy.deepcopy(row)
    corrupt['raw_observation']['current']['players'][1]['hand'] = [{'id': 666}]
    with pytest.raises(ValueError, match='content hash'): resolve_row(frame, corrupt)
    wrong_game = copy.deepcopy(row)
    wrong_game['game_id'] = 'different-game'
    with pytest.raises(ValueError, match='native game identity'): resolve_row(frame, wrong_game)
    wrong_index = copy.deepcopy(row)
    wrong_index['game_index'] = 99
    with pytest.raises(ValueError, match='native game identity'): resolve_row(frame, wrong_index)
    stamped = copy.deepcopy(row)
    stamped['information_mode'] = 'privileged_simulator_teacher'
    with pytest.raises(ValueError, match='not deployable'): resolve_row(frame, stamped)
    frame['source_schema'] = 'deepmew.replay_action_row.v1'
    with pytest.raises(ValueError, match='unsupported reference before source read'): list(resolve_references([frame]))


def test_corrupt_part_is_sealed_failed(corpus_inventory: Path, tmp_path: Path) -> None:
    inventory = orjson.loads(corpus_inventory.read_bytes())
    root = tmp_path / 'worker'
    root.mkdir()
    initialize(str(corpus_inventory), str(root))
    part = inventory['parts'][0]
    part['target_mask']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='hash_or_size'): canonical_part(part)
    assert verify_run(root / 'canonical' / part['part_id'])['status'] == 'failed'


@pytest.mark.parametrize('invalid', [False, True])
def test_raw_episode_conversion_and_quarantine(corpus_inventory: Path, tmp_path: Path, invalid: bool) -> None:
    data = synthetic_replay()
    if invalid:
        data['module_version'] = 'unsupported'
    source = tmp_path / 'raw.json'
    source.write_text(json.dumps(data))
    root = tmp_path / 'worker'
    root.mkdir()
    initialize(str(corpus_inventory), str(root))
    result = raw_episode({'path': str(source), 'sha256': file_hash(source), 'byte_size': source.stat().st_size}, 64 * 1024**2, 10000)
    assert result['frames'] == (0 if invalid else 4)
    if invalid:
        assert result['reason'] == 'unsupported_source_contract' and result['candidate_request_events'] == 4
    assert verify_run(Path(result['path']))['status'] == 'completed'


def test_uncataloged_snapshot_port_preserves_unknown_admission(corpus_inventory: Path, tmp_path: Path) -> None:
    from deepmew_ex.datasets.v2.corpus.snapshots import port_snapshot_sources
    inventory = json.loads(corpus_inventory.read_bytes())
    source_rows = [json.loads(line) for line in Path(inventory['sources'][0]['path']).read_bytes().splitlines()]
    part = inventory['parts'][0]
    with gzip.open(part['files']['part_0000.transitions.jsonl.gz']['path'], 'rb') as handle:
        transitions = [json.loads(line) for line in handle]
    source = tmp_path / 'filtered' / 'shards' / 'decisions.jsonl.gz'
    source.parent.mkdir(parents=True)
    with gzip.open(source, 'wb') as handle:
        for row, transition in zip(source_rows, transitions):
            row['legal_set'] = transition['legal_action_set']
            row['outcome'] = {'won': True, 'lost': False, 'draw': False}
            handle.write(json.dumps(row).encode() + b'\n')
    before = file_hash(source)
    path = port_snapshot_sources([source], OutputPaths(tmp_path / 'outputs'), 'supplement')
    refs = list(reference_rows(path))
    assert len(refs) == 2 and all(row['target_mask']['source_target_eligible'] is None for row in refs)
    assert all(row['effective_split'] == 'quarantine_pending_dedup_and_split_admission' for row in refs)
    assert len(list(resolve_references(refs))) == 2 and file_hash(source) == before
