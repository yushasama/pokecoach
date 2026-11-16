"""Streaming golden reader with explicit, causal same-actor history views."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from deepmew_ex.datasets.v2.contract import load_contract
from deepmew_ex.datasets.v2.conversion import game_split
from deepmew_ex.datasets.v2.schema import Episode, HistoryStep, OBJECTIVES, TrainingTrajectory, identity, payload
from deepmew_ex.registry.store import verify_run


def validate_episode(episode: Episode) -> None:
    contract = load_contract()
    if episode.split != game_split(episode.source_native_game_id) or episode.completion_status != 'complete_after_deck_setup':
        raise ValueError('invalid split or completion boundary')
    states = {state.state_id: state for state in episode.states}
    events = {event.event_id: event for event in episode.public_events}
    if len(states) != len(episode.states) or len(events) != len(episode.public_events):
        raise ValueError('duplicate content identity')
    for state in episode.states:
        fields = {key: value for key, value in asdict(state).items() if key not in ('state_id', 'privileged_teacher')}
        if identity('state2', fields) != state.state_id or state.privileged_teacher is not None:
            raise ValueError('state hash or privileged namespace violation')
        public = json.loads(json.dumps(state.shared_public))
        if 'select' in public or any('hand' in player for player in public['current']['players']):
            raise ValueError('private data in public namespace')
        public['current']['players'][state.actor_seat]['hand'] = state.actor_private['hand']
        contract.assert_visible(public, expected_mode='actor_visible')
    for event in episode.public_events:
        projected = contract.public_logs([event.public_payload])
        if projected != [event.public_payload]:
            raise ValueError('private public-event extension')
        fields = {'episode_id': episode.episode_id, 'seat': event.actor_seat, 'observation_step': event.observation_step, 'log_index': event.log_index, 'payload': event.public_payload}
        if identity('event2', fields) != event.event_id:
            raise ValueError('event identity mismatch')
    seen, orders, previous = set(), [0, 0], (-1, -1)
    for index, frame in enumerate(episode.frames):
        if frame.frame_id in seen or frame.episode_id != episode.episode_id or frame.global_order != index or frame.actor_seat not in (0, 1):
            raise ValueError('frame identity/order violation')
        seen.add(frame.frame_id)
        seat = frame.actor_seat
        if frame.actor_order != orders[seat] or frame.control_owner != seat or frame.request_step + 1 != frame.action_step:
            raise ValueError('actor history or request/action gap')
        if (frame.action_step, seat) <= previous:
            raise ValueError('noncausal global order')
        previous = frame.action_step, seat
        orders[seat] += 1
        state, after = states[frame.state_before_id], states[frame.action_result.state_after_id]
        if (state.actor_seat, state.observation_step) != (seat, frame.request_step) or (after.actor_seat, after.observation_step) != (seat, frame.action_step):
            raise ValueError('pre/post source coordinate mismatch')
        legal = contract.build_legal(state.state_id, frame.legal_actions.source_select)
        chosen = contract.reconstruct_chosen(legal, [stage['source_selected_values'] for stage in frame.chosen_action['stages']])
        if payload(asdict(legal)) != frame.legal_actions.canonical_contract or payload(asdict(chosen)) != frame.chosen_action or not chosen.complete or legal.failures:
            raise ValueError('legal/chosen reconstruction mismatch')
        for event_id in frame.public_event_ids:
            event = events[event_id]
            if event.actor_seat != seat or event.observation_step > frame.request_step:
                raise ValueError('future or cross-seat public history')
        expected_next = episode.frames[index + 1].frame_id if index + 1 < len(episode.frames) else None
        next_actor = next((item.frame_id for item in episode.frames[index + 1:] if item.actor_seat == seat), None)
        if frame.action_result.next_event_frame_id != expected_next or frame.action_result.next_same_actor_frame_id != next_actor or frame.action_result.td_transition:
            raise ValueError('result/successor link mismatch')
        proof = frame.source_proof
        if proof['source_sha256'] != episode.source_sha256 or proof['request'] != {'step': frame.request_step, 'seat': seat, 'field': 'observation'} or proof['action'] != {'step': frame.action_step, 'seat': seat, 'field': 'action'} or proof['result'] != {'step': frame.action_step, 'seat': seat, 'field': 'observation'}:
            raise ValueError('source proof mismatch')
        if frame.capability_receipt_id != episode.capabilities['receipt_id'] or frame.v1_link['status'] != 'unavailable':
            raise ValueError('unproven capability or V1 join')


def iter_episodes(path: Path, objective: str = 'snapshot_bc', split: str | None = None) -> Iterator[Episode]:
    # Gate before touching the dataset or allocating anything.
    if objective not in OBJECTIVES:
        raise ValueError(f'unsupported V2 objective: {objective}; TD/teacher/response proof is absent')
    if split is not None and split not in ('golden_development', 'golden_validation'):
        raise ValueError('invalid golden split')
    manifest = verify_run(path)
    if manifest['status'] != 'completed' or manifest['kind'] != 'v2_golden_conversion':
        raise ValueError('not a completed semantic golden conversion')
    report = json.loads((path / 'conversion.json').read_text(encoding='utf-8'))
    games = set()
    for item in report['episodes']:
        relative = Path(item['path'])
        if relative.is_absolute() or len(relative.parts) != 1:
            raise ValueError('episode path traversal')
        episode = Episode.from_bytes((path / relative).read_bytes())
        validate_episode(episode)
        if episode.source_native_game_id in games:
            raise ValueError('duplicate game across partitions')
        games.add(episode.source_native_game_id)
        if not episode.capabilities.get(objective):
            raise ValueError('episode capability does not support requested objective')
        if split is None or episode.split == split:
            yield episode


def trajectories(episode: Episode, objective: str = 'temporal_bc') -> Iterator[TrainingTrajectory]:
    if objective not in OBJECTIVES or not episode.capabilities.get(objective):
        raise ValueError('unsupported objective/capability')
    validate_episode(episode)
    states = {state.state_id: state for state in episode.states}
    history: list[list[HistoryStep]] = [[], []]
    for frame in episode.frames:
        seat = frame.actor_seat
        public = tuple(event for event in episode.public_events if event.actor_seat == seat and event.observation_step <= frame.request_step)
        prior = tuple(history[seat]) if objective == 'temporal_bc' else ()
        yield TrainingTrajectory(episode.episode_id, episode.split, seat, not history[seat], prior, states[frame.state_before_id], frame.legal_actions, public if objective == 'temporal_bc' else (), frame.chosen_action, bool(frame.eligibility['actor_loss']), frame.capability_receipt_id)
        history[seat].append(HistoryStep(frame.frame_id, states[frame.state_before_id], frame.legal_actions, frame.chosen_action))
