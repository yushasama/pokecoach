"""Recover teacher-only archives without claiming their missing opponent timeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from deepmew_ex.datasets.v2.contract import load_contract
from deepmew_ex.datasets.v2.corpus.reader import resolve_row
from deepmew_ex.datasets.v2.learning.actor_events import actor_events
from deepmew_ex.datasets.v2.learning.queries import actor_queries
from deepmew_ex.datasets.v2.learning.admission import SPLIT_PRIORITY
from deepmew_ex.datasets.v2.learning.semantic import action_payload, actor_state, input_signature, leaves, relative_players
from deepmew_ex.datasets.v2.learning.trajectory import ActorSequence, Prediction, structured_target
from deepmew_ex.datasets.v2.schema import Payload, payload


@dataclass(frozen=True)
class RecordedActorGame:
    sequence: ActorSequence
    recorded_requests: int
    post_state_receipts: tuple[Payload, ...]
    coverage: str = 'recorded_single_actor_selection_history; opponent_and_zero_option_prompts_not_proven_complete'
    training_admitted: bool = False


def observation_hash(observation: Payload) -> str:
    return hashlib.sha256(json.dumps(observation, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def recover_recorded_actor_game(frames: list[Payload], rows: list[Payload]) -> RecordedActorGame:
    """Caller must verify the entire source archive and complete game grouping.

    Dense teacher indices do not prove a dense two-player engine timeline. Exact
    post-observation evidence stays bound to its original observer and source;
    another player's private post-state never becomes this actor's input.
    """
    if not rows or len(frames) != len(rows):
        raise ValueError('incomplete recorded actor references')
    by_order = {frame['global_order']: frame for frame in frames}
    if len(by_order) != len(rows) or set(by_order) != set(range(len(rows))):
        raise ValueError('missing or repeated recorded actor coordinate')
    first = rows[0]
    schema, seat = first.get('schema_version'), first.get('seat')
    if schema not in ('deepmew.refgame_decision_row.rebuilt.v1', 'deepmew.refgame_decision_row.rebuilt.v2') or type(seat) is not int or seat not in (0, 1):
        raise ValueError('unsupported recorded actor contract')
    if not isinstance(first.get('game_id'), str) or any(type(first.get(key)) is not int for key in ('game_index', 'seed')):
        raise ValueError('missing recorded game identity')
    native = frames[0]['native_game_binding']['canonical_native_game_id']
    # Different generation runs reuse native labels (confirmed R18 examples).
    # The canonical source-unit-qualified identity distinguishes those games.
    game_id = frames[0]['episode_id']
    family = frames[0]['source_family']
    if any(frame['effective_split'] not in SPLIT_PRIORITY for frame in frames):
        raise ValueError('unknown recorded historical split')
    split = max((frame['effective_split'] for frame in frames), key=SPLIT_PRIORITY.__getitem__)
    contract = load_contract()
    records, events, predictions, transitions, unavailable, post_receipts = [], [], [], [], [], []
    result: int | None = None
    for index, row in enumerate(rows):
        frame = by_order[index]
        if any(row.get(key) != first.get(key) for key in ('schema_version', 'seat', 'game_id', 'game_index', 'seed')) or frame['source_schema'] != schema or frame['source_family'] != family or frame['episode_id'] != game_id or frame['native_game_binding']['canonical_native_game_id'] != native:
            raise ValueError('mixed recorded game identity or observer')
        if type(row.get('decision_index')) is not int or row['decision_index'] != index or frame['actor_order'] != index:
            raise ValueError('recorded actor order is incomplete')
        outcome = row.get('outcome')
        if not isinstance(outcome, dict) or type(outcome.get('result')) is not int or outcome['result'] not in (0, 1, 2) or outcome.get('crashed') is not False or outcome.get('timed_out') is not False:
            raise ValueError('recorded game lacks safe terminal evidence')
        if result is not None and result != outcome['result']:
            raise ValueError('recorded terminal results disagree')
        result = outcome['result']
        if frame['terminal_target'] != (.5 if result == 2 else float(result == seat)):
            raise ValueError('recorded actor outcome mismatch')
        resolved = resolve_row(frame, row)
        observation = row[frame['state_before']['source_field']]
        projected = payload(contract.project_actor(observation))
        context = relative_players({key: value for key, value in projected['select'].items() if key != 'option'}, seat)
        logs = relative_players(actor_events(observation.get('logs', []), seat, seat), seat)
        current = {'state': actor_state(resolved.state), 'request': context, 'public_log_snapshot': logs, 'actor_queries': relative_players(actor_queries(observation, seat), seat)}
        events.append(len(records))
        records.append(list(leaves(current)))
        stages = structured_target(resolved.legal_actions, resolved.target_action, seat, records)
        options = [relative_players(option['payload'], seat) for stage in resolved.legal_actions['stages'] for option in stage['options']]
        eligible = resolved.source_target_eligible is True and resolved.legal_actions['complete_action_count'] != 1
        predictions.append(Prediction(frame['frame_id'], len(events) - 1, stages, input_signature(current['state'], context, options), eligible))
        events.append(len(records))
        records.append(list(leaves({'past_action': action_payload(resolved.legal_actions, resolved.target_action, seat)})))
        after = row.get('post_observation')
        reference = frame['action_result']['state_after_reference']
        if after is None:
            if reference is not None:
                raise ValueError('declared post-state is absent')
            unavailable.append('unavailable-post-state:' + frame['frame_id'])
            continue
        if schema != 'deepmew.refgame_decision_row.rebuilt.v2' or not isinstance(after, dict) or not isinstance(after.get('current'), dict):
            raise ValueError('unsupported recorded post-state')
        post_actor = after['current'].get('yourIndex')
        if type(post_actor) is not int or post_actor not in (0, 1) or row.get('post_actor_seat') != post_actor:
            raise ValueError('recorded post-state observer mismatch')
        sha = observation_hash(after)
        if reference is not None and (reference.get('source_field') != 'post_observation' or reference.get('source_line') != frame['state_before']['source_line'] or reference.get('source_path') != frame['state_before']['source_path'] or reference.get('content_sha256') != sha or post_actor != seat):
            raise ValueError('canonical post-state identity mismatch')
        transitions.append('recorded-post:sha256:' + sha)
        post_receipts.append({'frame_id': frame['frame_id'], 'source_path': frame['state_before']['source_path'], 'source_sha256': frame['state_before']['source_sha256'], 'source_line': frame['state_before']['source_line'], 'source_field': 'post_observation', 'content_sha256': sha, 'observer': post_actor, 'learner_input': False, 'reason': 'exact source transition preserved for audit; only actual actor request states feed this objective'})
        if post_actor != seat:
            unavailable.append('other-observer-post-state:' + frame['frame_id'])
    sequence = ActorSequence(game_id, seat, family, split, tuple(records), tuple(events), tuple(predictions), tuple(transitions), tuple(unavailable), True)
    return RecordedActorGame(sequence, len(rows), tuple(post_receipts))
