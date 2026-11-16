from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pytest

from deepmew_ex.datasets.v2.contract import load_contract
from deepmew_ex.datasets.v2.conversion import convert_episode
from deepmew_ex.datasets.v2.reader import trajectories, validate_episode
from deepmew_ex.datasets.v2.schema import Episode, Payload
from deepmew_ex.registry.store import canonical
from .conftest import source_row, synthetic_replay


def converted(data: Payload | None = None) -> Episode:
    data = data or synthetic_replay()
    raw = canonical(data)
    return convert_episode(raw, source_row(raw, Path('source.json'), data['id'], data['module_version']), load_contract())


@pytest.mark.parametrize('version', ['1.30.2', '1.32.0'])
def test_exact_semantics_and_round_trip(version: str) -> None:
    data = synthetic_replay(version=version)
    episode = converted(data)
    validate_episode(episode)
    assert len(episode.frames) == 4
    assert episode.to_bytes() == converted(data).to_bytes() == Episode.from_bytes(episode.to_bytes()).to_bytes()
    assert episode.frames[0].chosen_action['stages'][0]['source_selected_values'] == [1]
    assert episode.frames[1].chosen_action['stages'][0]['source_selected_values'] == []
    assert episode.frames[3].chosen_action['stages'][0]['source_selected_values'] == [0, 1]
    assert episode.frames[0].request_step == 0 and episode.frames[0].action_step == 1
    assert episode.frames[0].action_result.next_same_actor_frame_id == episode.frames[2].frame_id
    assert episode.frames[0].v1_link['status'] == 'unavailable'
    assert not episode.capabilities['td_transition']


def test_private_negative_controls_and_unknown_events() -> None:
    episode = converted()
    raw = episode.to_bytes()
    assert b'PRIVATE-' not in raw
    assert all(event.public_payload.keys() <= {'type', 'playerIndex'} for event in episode.public_events)
    assert any(event.kind == 'unknown' and event.public_payload['type'] == 987 for event in episode.public_events)
    for state in episode.states:
        assert state.actor_private['hand'] == [{'id': 100 + state.actor_seat}]
        assert all('hand' not in player for player in state.shared_public['current']['players'])
    data = synthetic_replay()
    data['steps'][0][0]['observation']['current']['players'][1]['hand'] = [{'cardId': 666}]
    with pytest.raises(ValueError, match='privileged'):
        converted(data)


@pytest.mark.parametrize('surface', ['deck', 'prize', 'unknown_extension', 'nested_extension', 'public_payload'])
def test_hidden_and_unclassified_source_extensions_fail(surface: str) -> None:
    data = synthetic_replay()
    observation = data['steps'][0][0]['observation']
    if surface == 'unknown_extension':
        observation['futureHiddenPayload'] = {'secret': 123}
    elif surface == 'nested_extension':
        observation['current']['players'][0]['hand'][0]['hiddenPayload'] = 123
    elif surface == 'public_payload':
        observation['logs'][0]['playerIndex'] = {'hiddenPayload': 123}
    else:
        observation['current']['players'][0][surface] = [123]
    with pytest.raises(ValueError):
        converted(data)


def test_causal_history_reset_and_future_invariance() -> None:
    episode = converted()
    views = list(trajectories(episode))
    assert [len(view.history) for view in views] == [0, 0, 1, 1]
    assert [view.reset for view in views] == [True, True, False, False]
    for view in views:
        assert all(item.state_before.actor_seat == view.actor_seat and item.state_before.observation_step < view.current_state.observation_step for item in view.history)
        assert all(item.actor_seat == view.actor_seat and item.observation_step <= view.current_state.observation_step for item in view.public_history)
        assert not {'terminal_outcome', 'action_result', 'state_after'} & set(asdict(view))
    changed = synthetic_replay()
    changed['steps'][4][1]['observation']['current']['turn'] = 999
    changed['steps'][4][1]['observation']['logs'].append({'type': 999, 'playerIndex': 1})
    future_views = list(trajectories(converted(changed)))
    # Episode IDs bind the complete source file; compare actual current/prior inputs.
    assert [asdict(view.current_state) for view in views] == [asdict(view.current_state) for view in future_views]
    assert list(trajectories(episode, 'snapshot_bc'))[2].history == ()


@pytest.mark.parametrize('failure', ['seat', 'choice', 'duplicate', 'gap', 'version', 'seed', 'safety', 'incomplete', 'terminal_reward', 'terminal_status', 'early_reward', 'request_gap', 'unknown_order'])
def test_corrupt_contracts_fail(failure: str) -> None:
    data = synthetic_replay()
    if failure == 'seat':
        data['steps'][0][0]['observation']['current']['yourIndex'] = 1
    elif failure == 'choice':
        data['steps'][1][0]['action'] = [99]
    elif failure == 'duplicate':
        data['steps'][4][1]['action'] = [0, 0]
    elif failure == 'gap':
        data['steps'][2] = data['steps'][2][:1]
    elif failure == 'version':
        data['module_version'] = '9.9.9'
    elif failure == 'seed':
        data['configuration']['seed'] = 40009
    elif failure == 'safety':
        data['steps'][2][0]['status'] = 'ERROR'
    elif failure == 'terminal_reward':
        data['steps'][4][0]['reward'] = 1
    elif failure == 'terminal_status':
        data['steps'][4][0]['status'] = 'ACTIVE'
    elif failure == 'early_reward':
        data['steps'][2][0]['reward'] = 1
    elif failure == 'request_gap':
        data['steps'][2][0]['observation']['select'] = None
    elif failure == 'unknown_order':
        data['steps'][1][1]['observation']['select']['context'] = 10
    else:
        data['statuses'] = ['ACTIVE', 'ACTIVE']
    with pytest.raises(ValueError):
        converted(data)


def test_stale_legal_source_is_rejected() -> None:
    episode = converted()
    frames = list(episode.frames)
    frames[0] = replace(frames[0], legal_actions=frames[1].legal_actions)
    with pytest.raises(ValueError):
        validate_episode(replace(episode, frames=tuple(frames)))


def test_effect_chain_requests_remain_distinct() -> None:
    data = synthetic_replay()
    for step, seat in ((0, 0), (1, 1), (2, 0)):
        data['steps'][step][seat]['observation']['select']['effect'] = {'id': 1121, 'serial': 75, 'playerIndex': 0}
    episode = converted(data)
    assert len({frame.frame_id for frame in episode.frames[:3]}) == 3
    assert all(frame.legal_actions.source_select['effect']['id'] == 1121 for frame in episode.frames[:3])
    assert [frame.actor_seat for frame in episode.frames[:3]] == [0, 1, 0]
