"""Complete actor request prefixes and structured targets, without sample caps."""

from __future__ import annotations

from dataclasses import dataclass
import math

from deepmew_ex.datasets.v2.learning.semantic import Leaf, actor_state, frame_inputs, input_signature, leaves, relative_players, request_context, request_logs, selected_payload
from deepmew_ex.datasets.v2.schema import Episode, Payload


@dataclass(frozen=True)
class StageInput:
    options: tuple[int, ...]
    minimum: int
    maximum: int
    ordered: bool
    repeated: bool


@dataclass(frozen=True)
class StageTarget:
    options: tuple[int, ...]
    selected: tuple[int, ...]
    minimum: int
    maximum: int
    ordered: bool
    repeated: bool


@dataclass(frozen=True)
class Prediction:
    row_id: str
    event_index: int
    stages: tuple[StageTarget, ...]
    signature: str
    loss_eligible: bool


@dataclass(frozen=True)
class ActorSequence:
    game_id: str
    actor: int
    source_family: str
    split: str
    records: tuple[list[Leaf], ...]
    events: tuple[int, ...]
    predictions: tuple[Prediction, ...]
    preserved_transition_ids: tuple[str, ...]
    unavailable_post_states: tuple[str, ...]
    history_available: bool = True


def sequence_from_payload(value: Payload) -> ActorSequence:
    """Restore a prepared semantic sequence without coercing identities or indices."""
    expected = set(ActorSequence.__dataclass_fields__)
    if set(value) != expected or type(value['actor']) is not int or value['actor'] not in (0, 1) or type(value['history_available']) is not bool:
        raise ValueError('invalid prepared actor sequence')
    if any(not isinstance(value[key], str) or not value[key] for key in ('game_id', 'source_family', 'split')):
        raise ValueError('missing prepared sequence identity')
    records = []
    for record in value['records']:
        if not record or any(len(leaf) != 3 or not isinstance(leaf[0], str) or not isinstance(leaf[1], str) or type(leaf[2]) not in (int, float) or not math.isfinite(leaf[2]) for leaf in record):
            raise ValueError('invalid prepared semantic leaves')
        records.append([tuple(leaf) for leaf in record])
    events = tuple(value['events'])
    if not events or any(type(index) is not int or not 0 <= index < len(records) for index in events) or any(first >= second for first, second in zip(events, events[1:])):
        raise ValueError('invalid prepared causal event order')
    predictions = []
    for row in value['predictions']:
        if not isinstance(row['row_id'], str) or not isinstance(row['signature'], str) or type(row['loss_eligible']) is not bool or type(row['event_index']) is not int or not 0 <= row['event_index'] < len(events):
            raise ValueError('invalid prepared prediction')
        stages = []
        for stage in row['stages']:
            if set(stage) != set(StageTarget.__dataclass_fields__) or type(stage['ordered']) is not bool or type(stage['repeated']) is not bool:
                raise ValueError('invalid prepared action stage')
            options, selected = tuple(stage['options']), tuple(stage['selected'])
            if any(type(index) is not int or not 0 <= index < len(records) for index in options) or any(type(index) is not int or not 0 <= index < len(options) for index in selected):
                raise ValueError('invalid prepared action coordinate')
            minimum, maximum = stage['minimum'], stage['maximum']
            if type(minimum) is not int or type(maximum) is not int or not 0 <= minimum <= len(selected) <= maximum or not stage['repeated'] and (maximum > len(options) or len(set(selected)) != len(selected)):
                raise ValueError('invalid prepared selection bounds')
            stages.append(StageTarget(options, selected, minimum, maximum, stage['ordered'], stage['repeated']))
        if not stages:
            raise ValueError('prepared decision has no action contract')
        predictions.append(Prediction(row['row_id'], row['event_index'], tuple(stages), row['signature'], row['loss_eligible']))
    if len({row.row_id for row in predictions}) != len(predictions) or any(first.event_index >= second.event_index for first, second in zip(predictions, predictions[1:])):
        raise ValueError('duplicate or reordered prepared predictions')
    transitions, unavailable = tuple(value['preserved_transition_ids']), tuple(value['unavailable_post_states'])
    if any(not isinstance(identity, str) for identity in (*transitions, *unavailable)):
        raise ValueError('invalid prepared transition identity')
    return ActorSequence(value['game_id'], value['actor'], value['source_family'], value['split'], tuple(records), events, tuple(predictions), transitions, unavailable, value['history_available'])


def structured_inputs(contract: Payload, seat: int, records: list[list[Leaf]]) -> tuple[StageInput, ...]:
    if contract.get('failures'):
        raise ValueError('invalid structured action contract')
    result = []
    for stage in contract['stages']:
        if stage['conditional_options'] or stage['ordering_significance'] == 'unknown':
            raise ValueError('unsupported conditional or unknown-order action')
        ids = [option['option_id'] for option in stage['options']]
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate legal option identities')
        minimum, maximum = stage['minimum_count'], stage['maximum_count']
        repeated = stage['repetition_allowed']
        if repeated and stage['ordering_significance'] != 'ordered':
            raise ValueError('unordered repetition needs a separate exact likelihood')
        positions = []
        for option in stage['options']:
            positions.append(len(records))
            records.append(list(leaves(relative_players(option['payload'], seat))))
        result.append(StageInput(tuple(positions), minimum, maximum, stage['ordering_significance'] == 'ordered', repeated))
    return tuple(result)


def structured_target(contract: Payload, chosen: Payload, seat: int, records: list[list[Leaf]]) -> tuple[StageTarget, ...]:
    if chosen.get('failures') or not chosen.get('complete'):
        raise ValueError('invalid structured action contract')
    inputs = structured_inputs(contract, seat, records)
    if len(inputs) != len(chosen['stages']):
        raise ValueError('structured stage count mismatch')
    result = []
    for stage, source, target in zip(inputs, contract['stages'], chosen['stages'], strict=True):
        ids = [option['option_id'] for option in source['options']]
        selected = tuple(ids.index(identity) for identity in target['selected_option_ids'])
        if not stage.minimum <= len(selected) <= stage.maximum or not stage.repeated and len(selected) != len(set(selected)):
            raise ValueError('illegal structured target')
        result.append(StageTarget(stage.options, selected, stage.minimum, stage.maximum, stage.ordered, stage.repeated))
    return tuple(result)


def actor_sequences(episode: Episode, split: str, excluded_signatures: set[str] | None = None) -> tuple[ActorSequence, ...]:
    states = {state.state_id: state for state in episode.states}
    logs = request_logs(episode)
    excluded = excluded_signatures or set()
    result = []
    for actor in (0, 1):
        frames = [frame for frame in episode.frames if frame.actor_seat == actor]
        if not frames:
            continue
        records, events, predictions, transitions, unavailable = [], [], [], [], []
        previous = None
        for frame in frames:
            # A post-action observation is actor input only if it is actually a
            # request observation. Other recorded post states remain audit data.
            if previous is not None:
                after = states[previous.action_result.state_after_id]
                transitions.append(after.state_id)
                if after.state_id == frame.state_before_id:
                    events.append(len(records))
                    records.append(list(leaves({'observed_action_result': actor_state(after)})))
                else:
                    unavailable.append(after.state_id)
            current = frame_inputs(states[frame.state_before_id], frame, logs[frame.frame_id])
            events.append(len(records))
            records.append(list(leaves(current)))
            stages = structured_target(frame.legal_actions.canonical_contract, frame.chosen_action, actor, records)
            options = [relative_players(option['payload'], actor) for stage in frame.legal_actions.canonical_contract['stages'] for option in stage['options']]
            signature = input_signature(actor_state(states[frame.state_before_id]), request_context(frame), options)
            eligible = frame.legal_actions.canonical_contract['complete_action_count'] != 1 and signature not in excluded
            predictions.append(Prediction(frame.frame_id, len(events) - 1, stages, signature, eligible))
            events.append(len(records))
            records.append(list(leaves({'past_action': selected_payload(frame)})))
            previous = frame
        transitions.append(frames[-1].action_result.state_after_id)
        unavailable.append(frames[-1].action_result.state_after_id)
        result.append(ActorSequence(episode.source_native_game_id, actor, episode.source_family, split, tuple(records), tuple(events), tuple(predictions), tuple(transitions), tuple(unavailable)))
    return tuple(result)
