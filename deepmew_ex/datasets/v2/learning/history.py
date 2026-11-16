"""Shared actor-visible history construction for recovered data and live play."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from deepmew_ex.datasets.v2.contract import load_contract
from deepmew_ex.datasets.v2.conversion import state_view
from deepmew_ex.datasets.v2.learning.actor_events import actor_events, observer_card_identities
from deepmew_ex.datasets.v2.learning.queries import actor_queries
from deepmew_ex.datasets.v2.learning.semantic import action_payload, actor_state, leaves, relative_players
from deepmew_ex.datasets.v2.learning.trajectory import ActorSequence, StageInput, structured_inputs
from deepmew_ex.datasets.v2.schema import Payload, payload


LEGACY_INPUT_CONTRACT = 'actor_request_action_prefix_v1'
QUERY_INPUT_CONTRACT = 'recorded_public_events_actor_queries_v1'
INPUT_CONTRACTS = (LEGACY_INPUT_CONTRACT, QUERY_INPUT_CONTRACT)


@dataclass(frozen=True)
class LiveDecision:
    sequence: ActorSequence
    event_index: int
    stages: tuple[StageInput, ...]
    legal: object
    contract: Payload
    fallback_reason: str | None


class FullPrefixHistory:
    def __init__(self, public_event_mode: bool = False) -> None:
        self.records: list[list[tuple[str, str, float]]] = []
        self.events: list[int] = []
        self.actor_seat: int | None = None
        self.actor_order = 0
        self.last_global_actor: int | None = None
        self.direct_continuation = False
        self.pending = False
        self.public_event_mode = public_event_mode
        self.pending_public: list[tuple[int, list[Payload]]] = []
        self.public_observation_ready = False
        self.pending_queries: tuple[int, Payload] | None = None
        self.observed_request_sha: str | None = None

    def observe(self, observation: Payload) -> None:
        current = observation.get('current')
        if not isinstance(current, dict) or type(current.get('yourIndex')) is not int:
            raise ValueError('native observation has no actor identity')
        self.observe_actor(current['yourIndex'])
        if self.public_event_mode:
            contract = load_contract()
            self.pending_queries = (current['yourIndex'], actor_queries(observation, current['yourIndex']))
            self.observed_request_sha = contract.sha_json(contract.project_actor(observation))
            logs = observation.get('logs', [])
            if not isinstance(logs, list) or any(not isinstance(event, dict) for event in logs):
                raise ValueError('malformed live public event log')
            self.pending_public.append((current['yourIndex'], [payload(event) for event in logs]))
            self.public_observation_ready = True

    def observe_actor(self, actor: int) -> None:
        if actor not in (0, 1):
            raise ValueError('native actor index is outside two-seat contract')
        if self.pending:
            raise ValueError('new native prompt arrived before prior action commit')
        self.direct_continuation = self.last_global_actor == actor
        self.last_global_actor = actor
        self.public_observation_ready = False
        self.pending_queries = None
        self.observed_request_sha = None

    def prepare(self, observation: Payload) -> LiveDecision:
        if self.pending:
            raise ValueError('live temporal decision already pending')
        if self.public_event_mode and not self.public_observation_ready:
            raise ValueError('public history requires full observation notification')
        contract = load_contract()
        projected = payload(contract.project_actor(observation))
        seat = projected['current']['yourIndex']
        if self.public_event_mode and (self.pending_queries is None or self.pending_queries[0] != seat or self.observed_request_sha != contract.sha_json(projected)):
            raise ValueError('typed query view does not match the observed source request')
        if type(seat) is not int or seat not in (0, 1) or self.last_global_actor != seat:
            raise ValueError('actor-turn notification does not match visible request')
        if self.actor_seat is not None and self.actor_seat != seat:
            raise ValueError('live actor seat changed within game')
        self.actor_seat = seat
        state = state_view(observation, seat, self.actor_order, contract)
        if self.actor_order and self.direct_continuation and not self.public_event_mode:
            self.events.append(len(self.records))
            self.records.append(list(leaves({'observed_action_result': actor_state(state)})))
        source_select = payload(projected['select'])
        visible_logs = contract.public_logs(observation.get('logs', []))
        if self.public_event_mode:
            witness = observer_card_identities(projected['current'], seat)
            visible_logs = [event for origin, batch in self.pending_public for event in actor_events(batch, seat, origin, witness)]
        logs = [relative_players(event, seat) for event in visible_logs]
        context = relative_players({key: value for key, value in source_select.items() if key != 'option'}, seat)
        current = {'state': actor_state(state), 'request': context, 'public_log_snapshot': logs}
        if self.public_event_mode:
            current['actor_queries'] = relative_players(self.pending_queries[1], seat)
        self.events.append(len(self.records))
        self.records.append(list(leaves(current)))
        event_index = len(self.events) - 1
        legal = contract.build_legal(state.state_id, source_select)
        if legal.failures:
            raise ValueError('native legal contract unsupported')
        canonical_contract = payload(asdict(legal))
        fallback_reason = 'forced' if legal.complete_action_count == 1 else None
        stages: tuple[StageInput, ...] = ()
        if fallback_reason is None or self.public_event_mode:
            candidate_records = list(self.records)
            try:
                stages = structured_inputs(canonical_contract, seat, candidate_records)
            except ValueError:
                fallback_reason = 'unsupported_structured_action'
            else:
                self.records = candidate_records
        sequence = ActorSequence('live-native-game', seat, 'native_cabt', 'evaluation', tuple(self.records), tuple(self.events), (), (), (), True)
        self.pending = True
        self.pending_public = []
        self.public_observation_ready = False
        self.pending_queries = None
        self.observed_request_sha = None
        return LiveDecision(sequence, event_index, stages, legal, canonical_contract, fallback_reason)

    def commit(self, decision: LiveDecision, selected: list[list[int]]) -> None:
        if not self.pending:
            raise ValueError('no live temporal decision is pending')
        contract = load_contract()
        chosen = contract.reconstruct_chosen(decision.legal, selected)
        if not chosen.complete or chosen.failures:
            raise ValueError('live choice contract reconstruction failed')
        if self.actor_seat is None:
            raise ValueError('live actor seat is unavailable')
        self.events.append(len(self.records))
        self.records.append(list(leaves({'past_action': action_payload(decision.contract, payload(asdict(chosen)), self.actor_seat)})))
        self.actor_order += 1
        self.pending = False


