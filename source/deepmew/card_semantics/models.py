from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttackSemantics:
    attack_id: int
    effective_damage: int
    reason: str
    assumptions: tuple[str, ...]
