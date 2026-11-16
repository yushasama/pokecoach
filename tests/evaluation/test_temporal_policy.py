from __future__ import annotations

import os
import random
from pathlib import Path

import pytest

from deepmew_ex.datasets.v2.contract import load_contract
from deepmew_ex.datasets.v2.conversion import convert_episode
from deepmew_ex.datasets.v2.learning.trajectory import actor_sequences
from deepmew_ex.evaluation.league.temporal import FullPrefixHistory, FullPrefixLivePolicy, load_full_prefix_checkpoint
from deepmew_ex.evaluation.league.temporal import ActorNotifyingSeat
from deepmew_ex.evaluation.league.policies import IndependentSeat
from deepmew_ex.datasets.v2.learning.trajectory import ActorSequence, StageInput
from deepmew_ex.models.temporal import TemporalPolicy
from deepmew_ex.registry.store import canonical
from tests.datasets.v2.conftest import source_row, synthetic_replay


CHECKPOINT_SHA256 = '5ee59fbc608b2f81adacc7b043ff7573c4ecb689d0ca60ce7016e408ca89e15f'


def consecutive_replay() -> tuple[dict[str, object], object]:
    replay = synthetic_replay('live-prefix-parity')
    active = (0, 0, 1, 0)
    actions = ((0, [1]), (0, [0]), (1, []), (0, [0]))
    for step, agents in enumerate(replay['steps']):
        for seat, agent in enumerate(agents):
            agent['status'] = 'DONE' if step == len(replay['steps']) - 1 else 'ACTIVE' if seat == active[step] else 'INACTIVE'
            agent['action'] = []
    for step, (seat, selected) in enumerate(actions, 1):
        replay['steps'][step][seat]['action'] = selected
    raw = canonical(replay)
    episode = convert_episode(raw, source_row(raw, Path('unused'), replay['id']), load_contract())
    return replay, episode


def test_live_prefix_exactly_matches_offline_sequence_and_direct_effects() -> None:
    replay, episode = consecutive_replay()
    expected = next(sequence for sequence in actor_sequences(episode, 'evaluation') if sequence.actor == 0)
    history = FullPrefixHistory()
    for action_step in range(1, len(replay['steps'])):
        previous = replay['steps'][action_step - 1]
        actor = next(seat for seat, agent in enumerate(previous) if agent['status'] == 'ACTIVE')
        history.observe_actor(actor)
        if actor != 0:
            continue
        decision = history.prepare(previous[actor]['observation'])
        history.commit(decision, [replay['steps'][action_step][actor]['action']])
    assert tuple(history.records) == expected.records
    assert tuple(history.events) == expected.events
    assert sum(any('/observed_action_result/' in path for path, _, _ in record) for record in history.records) == 1
    assert 'PRIVATE' not in str(history.records)


def test_live_policy_selects_reconstructable_action_and_resets_per_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    replay = synthetic_replay('live-policy')
    observation = replay['steps'][0][0]['observation']
    model = TemporalPolicy(hidden=16, embedding=4, layers=1, heads=2).eval()
    policy = FullPrefixLivePolicy(model)
    policy.observe_actor(0)
    selected = policy.select(observation, random.Random(7))
    assert policy.counts['model_choices'] == 1 and policy.history.actor_order == 1
    contract = load_contract()
    projected = contract.project_actor(observation)
    legal = contract.build_legal('test-state', projected['select'])
    chosen = contract.reconstruct_chosen(legal, [selected])
    assert chosen.complete and not chosen.failures
    reset = FullPrefixLivePolicy(model)
    assert reset.history.actor_order == 0 and not reset.history.records

    forced = synthetic_replay('forced')['steps'][0][0]['observation']
    forced['select']['option'] = forced['select']['option'][:1]
    monkeypatch.setattr('deepmew_ex.evaluation.league.policies.greedy_fallback', lambda value: [0])
    fallback = FullPrefixLivePolicy(model)
    fallback.observe_actor(0)
    assert fallback.select(forced, random.Random(7)) == [0]
    assert fallback.counts['fallback:forced'] == 1 and fallback.history.actor_order == 1


def test_turn_notifications_isolate_other_seat_and_reject_seat_drift() -> None:
    replay = synthetic_replay('isolation')
    history = FullPrefixHistory()
    history.observe_actor(1)
    history.observe_actor(0)
    decision = history.prepare(replay['steps'][0][0]['observation'])
    history.commit(decision, [[1]])
    changed = synthetic_replay('isolation-changed')['steps'][1][1]['observation']
    history.observe_actor(1)
    with pytest.raises(ValueError, match='seat changed'):
        history.prepare(changed)


def test_production_seat_wrapper_preserves_typed_query_history_in_new_mode() -> None:
    observation = synthetic_replay('native-query-wrapper')['steps'][0][0]['observation']
    observation['logs'] = []
    observation['current']['looking'] = [None, {'id': 90, 'serial': 91, 'playerIndex': 0}]
    observation['select'].update({'type': 1, 'context': 7, 'deck': [{'id': 12, 'serial': 34, 'playerIndex': 0}, {'id': 56, 'serial': 78, 'playerIndex': 0}], 'option': [{'type': 3, 'area': 1, 'index': index, 'playerIndex': 0} for index in (0, 1)]})
    direct = FullPrefixHistory(public_event_mode=True)
    direct.observe(observation)
    decision = direct.prepare(observation)
    direct.commit(decision, [[1]])

    class FixedModel:
        def decode(self, sequence: ActorSequence, event_index: int, stages: tuple[StageInput, ...]) -> tuple[tuple[int, ...], ...]:
            return ((1,),)

    policy = FullPrefixLivePolicy(FixedModel(), public_event_mode=True)
    wrapped = ActorNotifyingSeat(IndependentSeat(policy, 9), (policy,))
    assert wrapped.select(observation, random.Random(10)) == [1]
    assert policy.history.records == direct.records and policy.history.events == direct.events
    assert any(path == '/actor_queries/deck/1/id' and value == 'int:56' for record in policy.history.records for path, value, _ in record)
    assert 'PRIVATE' not in str(policy.history.records)


def test_observed_queries_cannot_be_attached_to_a_different_request() -> None:
    import copy
    observation = synthetic_replay('query-source-drift')['steps'][0][0]['observation']
    observation['select']['deck'] = observation['current']['looking'] = None
    history = FullPrefixHistory(public_event_mode=True)
    history.observe(observation)
    changed = copy.deepcopy(observation)
    changed['current']['turn'] += 1
    with pytest.raises(ValueError, match='observed source request'):
        history.prepare(changed)


@pytest.mark.skipif('DEEPMEW_TEMPORAL_CHECKPOINT' not in os.environ, reason='explicit recovered checkpoint path required')
def test_recovered_checkpoint_identity_and_frozen_parameter_count() -> None:
    checkpoint = Path(os.environ['DEEPMEW_TEMPORAL_CHECKPOINT'])
    model, evidence = load_full_prefix_checkpoint(checkpoint, CHECKPOINT_SHA256, 'cpu')
    assert evidence['sha256'] == CHECKPOINT_SHA256
    assert evidence['parameters'] == 628805
    assert evidence['updates'] == 4160 and evidence['presentations'] == 258881
    assert evidence['optimizer_state_present'] and evidence['rng_state_present']
    assert not any(parameter.requires_grad for parameter in model.parameters())
