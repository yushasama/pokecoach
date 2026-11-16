from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DeckCard:
    card_id: int
    count: int
    name: str
    card_type: str
    basic: bool
    stage1: bool
    stage2: bool
    ex: bool
    mega_ex: bool
    ace_spec: bool
    hp: int
    evolves_from: str | None
    attacks: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class DeckSummary:
    path: str
    total_cards: int
    unique_cards: int
    pokemon_count: int
    trainer_count: int
    energy_count: int
    basic_count: int
    evolution_count: int
    ex_count: int
    mega_ex_count: int
    ace_spec_count: int
    cards: tuple[DeckCard, ...]
    main_attackers: tuple[DeckCard, ...]


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    engine_accepted: bool | None
    engine_error: str | None
