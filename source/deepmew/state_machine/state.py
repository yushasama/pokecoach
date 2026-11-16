"""State-machine enums and dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TurnState(str, Enum):
    FALLBACK = "FALLBACK"
    NO_ATTACKER_READY = "NO_ATTACKER_READY"
    ATTACKER_NEAR_READY = "ATTACKER_NEAR_READY"
    ATTACK_READY = "ATTACK_READY"


@dataclass(frozen=True)
class PokemonReadiness:
    area: str
    index: int
    card_id: int
    attached_energy: int
    best_shortfall: int | None
    best_damage: int | None
    can_attack_by_energy: bool


@dataclass(frozen=True)
class AttachmentEffect:
    known_target: bool
    improves_readiness: bool
    enables_attack: bool
    target_is_active: bool
    target_is_closest: bool
    before_shortfall: int | None
    after_shortfall: int | None


@dataclass(frozen=True)
class BoardPlan:
    state: TurnState
    energy_attached: bool
    legal_attack_available: bool
    legal_attach_available: bool
    legal_play_available: bool
    bench_count: int
    active_shortfall: int | None
    best_shortfall: int | None
    closest_area: str | None
    closest_index: int | None
    readiness: tuple[PokemonReadiness, ...]
