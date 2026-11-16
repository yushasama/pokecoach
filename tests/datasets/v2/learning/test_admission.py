from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import orjson
import pytest

from deepmew_ex.datasets.v2.contract import load_contract
from deepmew_ex.datasets.v2.conversion import convert_episode
from deepmew_ex.datasets.v2.corpus.reader import CorpusEpisode
from deepmew_ex.datasets.v2.learning.admission import AdmissionConfig, admit_replays
from deepmew_ex.datasets.v2.learning.semantic import Vocabulary, action_options, actor_state, input_signature, leaves, request_context, request_logs
from deepmew_ex.datasets.v2.schema import Episode
from deepmew_ex.experiments.config import OutputPaths
from deepmew_ex.registry.store import canonical, verify_run
from ..conftest import source_row, synthetic_replay


def episode(index: int = 0) -> Episode:
    raw = synthetic_replay(f'learning-{index}')
    for step in raw['steps']:
        for agent in step:
            agent['observation']['current']['turn'] += index * 100
    data = canonical(raw)
    return convert_episode(data, source_row(data, Path('unused'), raw['id']), load_contract())


def test_semantic_actor_inputs_and_permutation_signature() -> None:
    ep = episode()
    frame = ep.frames[0]
    state = actor_state(next(item for item in ep.states if item.state_id == frame.state_before_id))
    options, target, reason = action_options(frame)
    assert reason is None and target == 1
    encoded = list(leaves({'state': state, 'request': request_context(frame)}))
    assert 'PRIVATE' not in str(encoded) and 'result' not in state['board']
    vocab = Vocabulary.build([encoded, list(leaves(options[0]))])
    assert vocab.encode(encoded).shape == (len(encoded), 3)
    assert input_signature(state, request_context(frame), options) == input_signature(state, request_context(frame), options[::-1])
    assert len(request_logs(ep)) == len(ep.frames)


def test_public_history_excludes_inactive_or_future_snapshots() -> None:
    ep = episode()
    wrong = next(event for event in ep.public_events if event.actor_seat != ep.frames[0].actor_seat)
    changed = replace(ep.frames[0], public_event_ids=(wrong.event_id,))
    with pytest.raises(ValueError, match='nondeployable'):
        request_logs(replace(ep, frames=(changed, *ep.frames[1:])))


def test_causal_cache_records_separate_target_from_inputs() -> None:
    from deepmew_ex.datasets.v2.learning.cache import game_records
    ep = episode()
    selected = [{'frame_id': frame.frame_id, 'row_id': frame.frame_id, 'actor': frame.actor_seat, 'split': 'train'} for frame in ep.frames if frame.actor_seat == 0]
    cache = game_records(ep, selected, 8)
    assert [len(row['history_records']) for row in cache.examples] == [0, 1]
    assert cache.examples[0]['target'] == 1
    current = cache.records[cache.examples[1]['state_record']]
    prior = cache.records[cache.examples[1]['history_records'][0]]
    assert all('past_action' not in path for path, _, _ in current)
    assert any('past_action' in path for path, _, _ in prior)
    assert 'PRIVATE' not in str(cache.records)


def test_admission_membership_counts_and_immutability(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import deepmew_ex.datasets.v2.learning.admission as owner
    episodes = [CorpusEpisode(episode(index), split) for index, split in enumerate(('train', 'validation', 'holdout'))]
    corpus = tmp_path / 'corpus'
    corpus.mkdir()
    (corpus / 'manifest.json').write_bytes(b'{}')
    (corpus / 'port.json').write_bytes(orjson.dumps({'catalog_rows': 9000000, 'results': [{'native_game_id': row.episode.source_native_game_id, 'path': str(tmp_path / str(index)), 'status': 'admitted_actor_visible_episode'} for index, row in enumerate(episodes)]}))
    monkeypatch.setattr(owner, 'raw_episodes', lambda path: iter(episodes))
    config = AdmissionConfig(corpus, OutputPaths(tmp_path / 'outputs'), 'unit', 2, 2)
    result = admit_replays(config)
    manifest = verify_run(result)
    membership = orjson.loads((result / 'membership.json').read_bytes())
    assert manifest['status'] == 'completed' and len(membership['train']) == len(membership['validation']) == 2
    report = orjson.loads((result / 'admission.json').read_bytes())
    assert report['counts']['train']['raw_decisions'] == 4
    assert report['counts']['train']['excluded:optional_empty'] == 2
    assert report['selected']['train']['games'] == 1
    with pytest.raises(FileExistsError):
        admit_replays(config)


def test_unsupported_objective_and_empty_membership_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = AdmissionConfig(tmp_path / 'missing', OutputPaths(tmp_path / 'outputs'), 'bad', objective='td')
    with pytest.raises(ValueError, match='unsupported objective'):
        admit_replays(config)
    assert not config.paths.root.exists()
    import deepmew_ex.datasets.v2.learning.admission as owner
    corpus = tmp_path / 'corpus'
    corpus.mkdir()
    (corpus / 'manifest.json').write_bytes(b'{}')
    (corpus / 'port.json').write_bytes(orjson.dumps({'catalog_rows': 0, 'results': []}))
    monkeypatch.setattr(owner, 'raw_episodes', lambda path: iter([]))
    with pytest.raises(ValueError, match='empty eligible'):
        admit_replays(AdmissionConfig(corpus, config.paths, 'empty'))
    assert verify_run(config.paths.run('deepmew_ex', 'v2_admission', 'empty'))['status'] == 'failed'
