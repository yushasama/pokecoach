"""Recover recorded reference-game history without inventing intermediate effects.

The caller must verify source/summary files and cross-format split/deduplication
before admission. Per-row observation identities and actions are checked here.
This adapter proves recorded-request continuity, not every native engine event.
"""

from __future__ import annotations

from dataclasses import dataclass

from deepmew_ex.datasets.v2.contract import load_contract
from deepmew_ex.datasets.v2.corpus.reader import ResolvedSnapshot, resolve_row
from deepmew_ex.datasets.v2.learning.admission import SPLIT_PRIORITY
from deepmew_ex.datasets.v2.learning.actor_events import actor_events, observer_card_identities
from deepmew_ex.datasets.v2.learning.queries import actor_queries
from deepmew_ex.datasets.v2.learning.semantic import action_payload, actor_state, input_signature, leaves, relative_players
from deepmew_ex.datasets.v2.learning.trajectory import ActorSequence, Prediction, structured_target
from deepmew_ex.datasets.v2.schema import Payload, payload


@dataclass(frozen=True)
class RecoveredGame:
    game_id: str
    sequences: tuple[ActorSequence, ...]
    recorded_requests: int
    missing_immediate_post_states: int
    terminal_result: int
    effective_split: str
    public_events: int
    event_identity_fields_retained: int = 0
    event_identity_fields_withheld: int = 0
    coverage: str = 'complete_recorded_request_log; intermediate_engine_events_not_proven'
    admission: str = 'candidate_requires_source_hash_split_and_cross_format_duplicate_verification'


def recover_reference_game(frames: list[Payload], rows: list[Payload], summary: Payload, excluded_signatures: frozenset[str] = frozenset()) -> RecoveredGame:
    if summary.get('schema_version') != 'deepmew.refgames_factory.game.v0':
        raise ValueError('unsupported game-summary contract')
    count, result = summary.get('decisions'), summary.get('result')
    if type(count) is not int or count < 1 or len(rows) != count or len(frames) != count:
        raise ValueError('incomplete recorded request log')
    if type(result) is not int or result not in (0, 1, 2) or summary.get('crashed') is not False or summary.get('timed_out') is not False or summary.get('error'):
        raise ValueError('unsafe or incomplete terminal summary')
    game = summary.get('game_id')
    if not isinstance(game, str) or not game or type(summary.get('game_index')) is not int or type(summary.get('seed')) is not int:
        raise ValueError('missing game identity')
    if len({frame['frame_id'] for frame in frames}) != count:
        raise ValueError('duplicate source coordinates')
    # Canonical storage groups each game's rows by actor, while source rows use
    # global request order. Join on verified coordinates, never physical order.
    if any(type(frame['global_order']) is not int for frame in frames):
        raise ValueError('invalid global request coordinate')
    by_order = {frame['global_order']: frame for frame in frames}
    if len(by_order) != count or set(by_order) != set(range(count)):
        raise ValueError('missing or repeated reference coordinate')
    if any(frame['effective_split'] not in SPLIT_PRIORITY for frame in frames):
        raise ValueError('unknown historical split')
    split = max((frame['effective_split'] for frame in frames), key=SPLIT_PRIORITY.__getitem__)
    families = {frame['source_family'] for frame in frames}
    if len(families) != 1:
        raise ValueError('mixed source families inside game')
    contract = load_contract()
    prepared: dict[int, list[tuple[Payload, Payload, ResolvedSnapshot, list[Payload]]]] = {0: [], 1: []}
    raw_prefix: list[Payload] = []
    pending_public: dict[int, list[tuple[int, list[Payload]]]] = {0: [], 1: []}
    public_count = 0
    retained, withheld = 0, 0
    for index, row in enumerate(rows):
        frame = by_order[index]
        if row.get('schema_version') != 'deepmew.refgames_factory.decision.v0' or frame['source_schema'] != row['schema_version']:
            raise ValueError('unsupported reference decision contract')
        if any(row.get(key) != summary[key] for key in ('game_id', 'game_index', 'seed')) or frame['native_game_binding']['raw_game_id'] != game:
            raise ValueError('mixed game identity or seed')
        if type(row.get('decision_index')) is not int or row['decision_index'] != index or frame['global_order'] != index:
            raise ValueError('missing, repeated or reordered request')
        seat = row.get('seat')
        if type(seat) is not int or seat not in (0, 1) or frame['actor_seat'] != seat or frame['actor_order'] != len(prepared[seat]):
            raise ValueError('actor order or identity mismatch')
        outcome = row.get('outcome')
        if not isinstance(outcome, dict) or outcome.get('result') != result or outcome.get('crashed') is not False or outcome.get('timed_out') is not False:
            raise ValueError('decision terminal evidence mismatch')
        expected_outcome = .5 if result == 2 else float(result == seat)
        if frame['terminal_target'] != expected_outcome:
            raise ValueError('actor-relative outcome mismatch')
        if frame['action_result']['state_after_reference'] is not None:
            raise ValueError('explicit post-state requires its own verified resolver')
        resolved = resolve_row(frame, row)
        observation = row[frame['state_before']['source_field']]
        logs = observation.get('logs', [])
        if not isinstance(logs, list) or any(not isinstance(event, dict) for event in logs):
            raise ValueError('malformed recorded event log')
        raw_prefix.extend(logs)
        if 'ordered_public_reveal_stream' in row:
            stream = row['ordered_public_reveal_stream']
            if not isinstance(stream, list) or any(not isinstance(event, dict) or type(event.get('event_index')) is not int or event['event_index'] != position for position, event in enumerate(stream)):
                raise ValueError('invalid cumulative event coordinates')
            if [{key: value for key, value in event.items() if key != 'event_index'} for event in stream] != raw_prefix:
                raise ValueError('cumulative event stream differs from recorded causal prefix')
        public_count += len(logs)
        for target in (0, 1):
            pending_public[target].append((seat, logs))
        # A witness comes from this request, never a later observation or the
        # other player's private view. Earlier prefix records stay immutable.
        witness = observer_card_identities(observation['current'], seat)
        public = []
        for origin, batch in pending_public[seat]:
            projected = actor_events(batch, seat, origin, witness)
            for before, after in zip(batch, projected, strict=True):
                identity_fields = {key for key, value in before.items() if (key.startswith('cardId') or key.startswith('serial')) and value is not None}
                retained += len(identity_fields & after.keys())
                withheld += len(identity_fields - after.keys())
            public.extend(projected)
        prepared[seat].append((frame, row, resolved, public))
        pending_public[seat] = []
    sequences = []
    for seat, decisions in prepared.items():
        if not decisions:
            continue
        records, events, predictions, gaps = [], [], [], []
        for frame, row, resolved, public in decisions:
            observation = row[frame['state_before']['source_field']]
            projected = payload(contract.project_actor(observation))
            context = relative_players({key: value for key, value in projected['select'].items() if key != 'option'}, seat)
            logs = [relative_players(event, seat) for event in public]
            current = {'state': actor_state(resolved.state), 'request': context, 'public_log_snapshot': logs, 'actor_queries': relative_players(actor_queries(observation, seat), seat)}
            events.append(len(records))
            records.append(list(leaves(current)))
            stages = structured_target(resolved.legal_actions, resolved.target_action, seat, records)
            options = [relative_players(option['payload'], seat) for stage in resolved.legal_actions['stages'] for option in stage['options']]
            signature = input_signature(current['state'], context, options)
            eligible = resolved.source_target_eligible is True and resolved.legal_actions['complete_action_count'] != 1 and signature not in excluded_signatures
            predictions.append(Prediction(frame['frame_id'], len(events) - 1, stages, signature, eligible))
            events.append(len(records))
            records.append(list(leaves({'past_action': action_payload(resolved.legal_actions, resolved.target_action, seat)})))
            gaps.append('unavailable-post-state:' + frame['frame_id'])
        sequences.append(ActorSequence(game, seat, next(iter(families)), split, tuple(records), tuple(events), tuple(predictions), (), tuple(gaps), True))
    return RecoveredGame(game, tuple(sequences), count, count, result, split, public_count, retained, withheld)
