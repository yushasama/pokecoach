"""Restore source-supported event identities from complete native replay archives."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace

from deepmew_ex.datasets.v2.contract import load_contract
from deepmew_ex.datasets.v2.conversion import state_view
from deepmew_ex.datasets.v2.learning.admission import SPLIT_PRIORITY
from deepmew_ex.datasets.v2.learning.history import FullPrefixHistory
from deepmew_ex.datasets.v2.learning.semantic import actor_state, input_signature, relative_players, request_context
from deepmew_ex.datasets.v2.learning.trajectory import ActorSequence, Prediction, StageTarget
from deepmew_ex.datasets.v2.reader import validate_episode
from deepmew_ex.datasets.v2.schema import Episode, Payload
from deepmew_ex.registry.store import digest


def recover_raw_histories(episode: Episode, raw: bytes, split: str) -> tuple[ActorSequence, ...]:
    if digest(raw) != episode.source_sha256 or split not in SPLIT_PRIORITY:
        raise ValueError('raw recovery source identity or split mismatch')
    validate_episode(episode)
    replay = json.loads(raw)
    if replay.get('id') != episode.source_native_game_id:
        raise ValueError('raw recovery native game identity mismatch')
    request_steps = [frame.request_step for frame in episode.frames]
    if not request_steps or request_steps != list(range(request_steps[0], request_steps[-1] + 1)):
        raise ValueError('raw recovery has unrecorded global requests')
    histories = [FullPrefixHistory(public_event_mode=True) for _ in (0, 1)]
    predictions: dict[int, list[Prediction]] = {0: [], 1: []}
    transitions: dict[int, list[str]] = {0: [], 1: []}
    unavailable: dict[int, list[str]] = {0: [], 1: []}
    contract = load_contract()
    by_id = {frame.frame_id: frame for frame in episode.frames}
    for frame in episode.frames:
        seat = frame.actor_seat
        agents = replay['steps'][frame.request_step]
        if [index for index, agent in enumerate(agents) if agent.get('status') == 'ACTIVE'] != [seat]:
            raise ValueError('raw request observer differs from admitted frame')
        observation = agents[seat]['observation']
        state = state_view(observation, seat, frame.request_step, contract)
        if state.state_id != frame.state_before_id or contract.project_actor(observation)['select'] != frame.legal_actions.source_select:
            raise ValueError('raw observation differs from admitted canonical state')
        selected = replay['steps'][frame.action_step][seat]['action']
        if [selected] != [stage['source_selected_values'] for stage in frame.chosen_action['stages']]:
            raise ValueError('raw selected action differs from admitted source action')
        for history in histories:
            history.observe(observation)
        history = histories[seat]
        decision = history.prepare(observation)
        if decision.fallback_reason not in (None, 'forced'):
            raise ValueError('raw structured action contract cannot be represented exactly')
        stages = []
        for stage, legal, target in zip(decision.stages, decision.contract['stages'], frame.chosen_action['stages'], strict=True):
            positions = [option['position'] for option in legal['options']]
            stages.append(StageTarget(stage.options, tuple(positions.index(value) for value in target['source_selected_values']), stage.minimum, stage.maximum, stage.ordered, stage.repeated))
        options = [relative_players(option['payload'], seat) for stage in decision.contract['stages'] for option in stage['options']]
        signature = input_signature(actor_state(state), request_context(frame), options)
        predictions[seat].append(Prediction(frame.frame_id, decision.event_index, tuple(stages), signature, decision.fallback_reason != 'forced'))
        history.commit(decision, [selected])
        transitions[seat].append(frame.action_result.state_after_id)
        next_frame = by_id.get(frame.action_result.next_same_actor_frame_id)
        if next_frame is None or frame.action_result.state_after_id != next_frame.state_before_id:
            unavailable[seat].append(frame.action_result.state_after_id)
    return tuple(ActorSequence(episode.source_native_game_id, seat, episode.source_family, split, tuple(history.records), tuple(history.events), tuple(predictions[seat]), tuple(transitions[seat]), tuple(unavailable[seat]), True) for seat, history in enumerate(histories) if predictions[seat])


def inherit_raw_masks(episode: Episode, raw: bytes, sequences: tuple[ActorSequence, ...], references: Sequence[Payload], source_bytes: Mapping[str, bytes]) -> tuple[tuple[ActorSequence, ...], dict[str, object]]:
    """Join legacy post-action rows to exact native actions, never to future inputs.

    The caller verifies the reference-index identity. This function verifies the
    referenced source bytes itself and preserves every false historical mask.
    Unmatched references fail; a native replay target absent from the historical
    table remains a new coordinate subject to global admission.
    """
    if digest(raw) != episode.source_sha256:
        raise ValueError('raw mask source identity mismatch')
    replay = json.loads(raw)
    if replay.get('id') != episode.source_native_game_id or replay.get('configuration', {}).get('seed') != episode.seed:
        raise ValueError('raw mask native identity mismatch')
    frames = {(frame.action_step, frame.actor_seat): frame for frame in episode.frames}
    frame_ids = {frame.frame_id for frame in episode.frames}
    if len(frames) != len(episode.frames) or {row.row_id for sequence in sequences for row in sequence.predictions} != frame_ids or sum(len(sequence.predictions) for sequence in sequences) != len(frame_ids):
        raise ValueError('raw mask candidate coordinates differ from episode')
    if any(sequence.game_id != episode.source_native_game_id or sequence.split not in SPLIT_PRIORITY or tuple(row.row_id for row in sequence.predictions) != tuple(frame.frame_id for frame in episode.frames if frame.actor_seat == sequence.actor) for sequence in sequences):
        raise ValueError('raw mask candidate game, split or actor mismatch')
    split = max((sequence.split for sequence in sequences), key=SPLIT_PRIORITY.__getitem__)
    sources: dict[str, tuple[str, list[Payload]]] = {}
    masks: dict[str, bool] = {}
    bindings: list[dict[str, object]] = []
    seen: set[str] = set()
    for reference in references:
        if reference['source_schema'] != 'deepmew.replay_action_row.v1' or reference['native_game_binding']['raw_game_id'] != episode.source_native_game_id or reference['frame_id'] in seen:
            raise ValueError('raw mask reference contract, identity or duplication mismatch')
        seen.add(reference['frame_id'])
        coordinate = reference['state_before']
        path, expected = coordinate['source_path'], coordinate['source_sha256']
        if path not in sources:
            encoded = source_bytes[path]
            if digest(encoded) != expected:
                raise ValueError('raw mask historical source hash mismatch')
            sources[path] = expected, [json.loads(line) for line in encoded.splitlines()]
        if sources[path][0] != expected or type(coordinate['source_line']) is not int or not 1 <= coordinate['source_line'] <= len(sources[path][1]):
            raise ValueError('raw mask historical source coordinate mismatch')
        row = sources[path][1][coordinate['source_line'] - 1]
        step, seat = row.get('step'), row.get('player_id')
        if type(step) is not int or type(seat) is not int or (step, seat) not in frames:
            raise ValueError('raw mask row does not identify an exact replay action')
        frame = frames[step, seat]
        if row.get('schema_version') != reference['source_schema'] or row.get('game_id') != episode.source_native_game_id or row.get('seed') != episode.seed or reference['global_order'] != step or reference['actor_seat'] != seat:
            raise ValueError('raw mask historical action identity mismatch')
        # The historical converter labeled this field raw_observation even when
        # the original action-table field was state_json. Verify its exact bytes.
        observation = row.get('state_json')
        observed_sha = digest(json.dumps(observation, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode())
        if coordinate['source_field'] not in ('raw_observation', 'state_json') or observed_sha != coordinate['content_sha256'] or observation != replay['steps'][step][seat]['observation']:
            raise ValueError('raw mask historical post-observation mismatch')
        if row.get('chosen_indices') != replay['steps'][step][seat]['action'] or reference['chosen_action']['indices'] != row['chosen_indices'] or reference['legal_actions'] != row.get('legal_options_json'):
            raise ValueError('raw mask historical selected action mismatch')
        if frame.request_step != step - 1 or replay['steps'][step - 1][seat]['status'] != 'ACTIVE' or [row['chosen_indices']] != [stage['source_selected_values'] for stage in frame.chosen_action['stages']]:
            raise ValueError('raw mask prior-request action alignment mismatch')
        eligible, reserved = reference['target_mask']['source_target_eligible'], reference['effective_split']
        if type(eligible) is not bool or reserved not in SPLIT_PRIORITY:
            raise ValueError('raw mask historical eligibility or split is unknown')
        masks[frame.frame_id] = masks.get(frame.frame_id, True) and eligible
        split = max((split, reserved), key=SPLIT_PRIORITY.__getitem__)
        bindings.append({'reference_id': reference['frame_id'], 'raw_frame_id': frame.frame_id, 'source_sha256': expected, 'source_line': coordinate['source_line'], 'source_eligible': eligible})
    restored = tuple(replace(sequence, split=split, predictions=tuple(replace(row, loss_eligible=row.loss_eligible and masks.get(row.row_id, True)) for row in sequence.predictions)) for sequence in sequences)
    return restored, {'references': len(references), 'matched_native_actions': len(masks), 'masked_native_actions': sum(not value for value in masks.values()), 'effective_split': split, 'bindings': bindings, 'input_records_changed': False, 'training_admitted': False}
