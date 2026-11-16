"""V2 golden objects keep semantic payloads separate from optional encodings."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Literal, TypeAlias

from deepmew_ex.registry.store import canonical, digest

Json: TypeAlias = None | bool | int | float | str | list['Json'] | dict[str, 'Json']
Payload: TypeAlias = dict[str, Json]
Objective = Literal['snapshot_bc', 'temporal_bc']
SCHEMA = 'deepmew.ex.semantic_golden.v2'
OBJECTIVES = ('snapshot_bc', 'temporal_bc')


def checked_json(value: object) -> Json:
    if isinstance(value, Enum):
        return checked_json(value.value)
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, (tuple, list)):
        return [checked_json(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: checked_json(item) for key, item in value.items()}
    raise ValueError('invalid JSON value or non-finite number')


def payload(value: object) -> Payload:
    result = checked_json(value)
    if not isinstance(result, dict):
        raise ValueError('expected JSON object')
    return result


def identity(kind: str, value: object) -> str:
    return f'{kind}:sha256:{digest(canonical(value))}'


@dataclass(frozen=True)
class GameState:
    state_id: str
    shared_public: Payload
    actor_private: Payload
    actor_seat: int
    observation_step: int
    # Simulator/teacher inputs are never serialized, including hashes of their contents.
    privileged_teacher: None = None


@dataclass(frozen=True)
class PublicEvent:
    event_id: str
    actor_seat: int
    observation_step: int
    log_index: int
    kind: str
    public_payload: Payload
    visibility: str = 'conservative_public_projection'
    confidence: str = 'derived'


@dataclass(frozen=True)
class LegalActionSet:
    legal_action_set_id: str
    source_select: Payload
    canonical_contract: Payload


@dataclass(frozen=True)
class ActionResult:
    state_after_id: str
    resulting_step: int
    next_event_frame_id: str | None
    next_same_actor_frame_id: str | None
    immediate_reward: None = None
    td_transition: bool = False


@dataclass(frozen=True)
class DecisionFrame:
    frame_id: str
    episode_id: str
    global_order: int
    actor_order: int
    actor_seat: int
    control_owner: int
    request_step: int
    action_step: int
    state_before_id: str
    legal_actions: LegalActionSet
    chosen_action: Payload
    public_event_ids: tuple[str, ...]
    action_result: ActionResult
    source_proof: Payload
    capability_receipt_id: str
    eligibility: Payload
    v1_link: Payload


@dataclass(frozen=True)
class Episode:
    schema: str
    episode_id: str
    source_family: str
    source_schema_version: str
    source_sha256: str
    source_native_game_id: str
    source_asset_id: str
    module_version: str
    split: str
    seed: int
    terminal_outcome: tuple[float, float]
    completion_status: str
    capabilities: Payload
    states: tuple[GameState, ...]
    public_events: tuple[PublicEvent, ...]
    frames: tuple[DecisionFrame, ...]

    def to_bytes(self) -> bytes:
        return canonical(asdict(self))

    @classmethod
    def from_bytes(cls, raw: bytes) -> Episode:
        data = payload(json.loads(raw))
        if data['schema'] != SCHEMA:
            raise ValueError('unsupported V2 schema')
        states = tuple(GameState(**item) for item in data.pop('states'))
        events = tuple(PublicEvent(**item) for item in data.pop('public_events'))
        frames = []
        for item in data.pop('frames'):
            item['legal_actions'] = LegalActionSet(**item['legal_actions'])
            item['action_result'] = ActionResult(**item['action_result'])
            item['public_event_ids'] = tuple(item['public_event_ids'])
            frames.append(DecisionFrame(**item))
        data['terminal_outcome'] = tuple(data['terminal_outcome'])
        return cls(**data, states=states, public_events=events, frames=tuple(frames))


@dataclass(frozen=True)
class HistoryStep:
    frame_id: str
    state_before: GameState
    legal_actions: LegalActionSet
    chosen_action: Payload


@dataclass(frozen=True)
class TrainingTrajectory:
    """Causal actor inputs: prior actions, current state/legal set and separate target."""

    episode_id: str
    split: str
    actor_seat: int
    reset: bool
    history: tuple[HistoryStep, ...]
    current_state: GameState
    current_legal_actions: LegalActionSet
    public_history: tuple[PublicEvent, ...]
    target_action: Payload
    loss_eligible: bool
    capability_receipt_id: str
