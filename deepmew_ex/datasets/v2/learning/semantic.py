"""Lossless semantic leaf records and a train-owned categorical vocabulary."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from deepmew_ex.datasets.v2.schema import DecisionFrame, Episode, GameState, Json, Payload
from deepmew_ex.registry.store import canonical, digest

Leaf = tuple[str, str, float]
HISTORY_TREATMENTS = ('snapshot', 'null', 'ordered', 'shuffled')


def relative_players(value: Json, seat: int) -> Json:
    if isinstance(value, dict):
        return {key: int(item != seat) if key == 'playerIndex' and type(item) is int and item in (0, 1) else relative_players(item, seat) for key, item in value.items()}
    if isinstance(value, list):
        return [relative_players(item, seat) for item in value]
    return value


def leaves(value: Json, path: str = '') -> Iterator[Leaf]:
    if isinstance(value, dict) and value:
        for key in sorted(value):
            yield from leaves(value[key], f'{path}/{key}')
        return
    if isinstance(value, list) and value:
        for index, item in enumerate(value):
            yield from leaves(item, f'{path}/{index}')
        return
    kind = type(value).__name__
    number = math.copysign(math.log1p(abs(value)), value) / 10 if type(value) in (int, float) else 0.0
    if not math.isfinite(number):
        raise ValueError('nonfinite_semantic_number')
    yield path, f'{kind}:{value}', number


def actor_state(state: GameState) -> Payload:
    # Timing/identities/outcomes are not actor features. Card/zone relationships
    # remain exact leaves, with the current actor consistently in player slot0.
    import copy
    current = copy.deepcopy(state.shared_public['current'])
    seat = state.actor_seat
    current['players'] = [current['players'][seat], current['players'][1 - seat]]
    current['yourIndex'] = 0
    if current.get('firstPlayer') in (0, 1):
        current['firstPlayer'] = int(current['firstPlayer'] != seat)
    current.pop('result', None)
    return {'board': current, 'own_hand': state.actor_private['hand']}


def request_context(frame: DecisionFrame) -> Payload:
    return relative_players({key: value for key, value in frame.legal_actions.source_select.items() if key != 'option'}, frame.actor_seat)


def selection_options(contract: Payload, seat: int) -> tuple[list[Payload], list[str], str | None]:
    stages = contract['stages']
    if contract['complete_action_count'] == 1:
        return [], [], 'forced'
    if len(stages) != 1:
        return [], [], 'multiple_selection_stages'
    stage = stages[0]
    if stage['minimum_count'] == 0:
        return [], [], 'optional_empty'
    if stage['minimum_count'] != 1 or stage['maximum_count'] != 1:
        return [], [], 'multi_select'
    options = stage['options']
    ids = [item['option_id'] for item in options]
    return [relative_players(item['payload'], seat) for item in options], ids, None


def action_options(frame: DecisionFrame) -> tuple[list[Payload], int | None, str | None]:
    options, ids, reason = selection_options(frame.legal_actions.canonical_contract, frame.actor_seat)
    if reason:
        return [], None, reason
    chosen = frame.chosen_action['stages'][0]['selected_option_ids']
    if len(ids) < 2 or len(chosen) != 1 or chosen[0] not in ids or len(set(ids)) != len(ids):
        raise ValueError('invalid_single_selection_crosswalk')
    return options, ids.index(chosen[0]), None


def selected_payload(frame: DecisionFrame) -> Json:
    return action_payload(frame.legal_actions.canonical_contract, frame.chosen_action, frame.actor_seat)


def action_payload(contract: Payload, chosen_action: Payload, seat: int) -> Json:
    # Past compound/forced actions remain context, even when they bear no loss.
    result = []
    for stage, chosen in zip(contract['stages'], chosen_action['stages'], strict=True):
        mapping = {item['option_id']: item['payload'] for item in stage['options']}
        result.append([relative_players(mapping[identity], seat) for identity in chosen['selected_option_ids']])
    return result


def request_logs(episode: Episode) -> dict[str, list[Payload]]:
    events = {event.event_id: event for event in episode.public_events}
    result = {}
    for frame in episode.frames:
        batch = [events[identity] for identity in frame.public_event_ids]
        if any(event.actor_seat != frame.actor_seat or event.observation_step != frame.request_step for event in batch):
            raise ValueError('nondeployable_public_snapshot')
        result[frame.frame_id] = [relative_players(event.public_payload, frame.actor_seat) for event in batch]
    return result


def frame_inputs(state: GameState, frame: DecisionFrame, logs: list[Payload]) -> Payload:
    return {'state': actor_state(state), 'request': request_context(frame), 'public_log_snapshot': logs}


def input_signature(state: Payload, context: Payload, options: list[Payload]) -> str:
    # Independent of target, source IDs, clock and option permutation.
    legal = sorted(canonical(option).decode() for option in options)
    return digest(canonical({'state': state, 'request': context, 'options': legal}))


@dataclass(frozen=True)
class Vocabulary:
    paths: dict[str, int]
    values: dict[str, int]
    max_leaves: int = 1024

    @classmethod
    def build(cls, records: list[list[Leaf]], max_leaves: int = 1024) -> Vocabulary:
        paths, values = set(), set()
        for row in records:
            if not row or len(row) > max_leaves:
                raise ValueError('semantic_leaf_budget_exceeded')
            for path, value, _ in row:
                paths.add(path)
                values.add(value)
        return cls({value: index + 2 for index, value in enumerate(sorted(paths))}, {value: index + 2 for index, value in enumerate(sorted(values))}, max_leaves)

    def encode(self, record: list[Leaf], counts: Counter[str] | None = None) -> np.ndarray:
        if not record or len(record) > self.max_leaves:
            raise ValueError('semantic_leaf_budget_exceeded')
        result = np.empty((len(record), 3), dtype=np.float32)
        for index, (path, value, numeric) in enumerate(record):
            result[index] = self.paths.get(path, 1), self.values.get(value, 1), numeric
            if counts is not None:
                counts['leaves'] += 1
                counts['unknown_path'] += path not in self.paths
                counts['unknown_value'] += value not in self.values
        return result
