"""Observation to board-state transition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from cg.api import OptionType

from deepmew.state_machine.readiness import _own_player, _own_pokemon
from deepmew.state_machine.state import BoardPlan, TurnState


@dataclass(frozen=True)
class TurnStateFacts:
    legal_attack: bool
    active_shortfall: int | None
    best_shortfall: int | None
    has_readiness: bool


@dataclass(frozen=True)
class TurnStateRule:
    name: str
    guard: Callable[[TurnStateFacts], bool]
    state: TurnState


def is_attack_ready(facts: TurnStateFacts) -> bool:
    return facts.legal_attack or facts.active_shortfall == 0


def is_attacker_near_ready(facts: TurnStateFacts) -> bool:
    return facts.best_shortfall is not None and facts.best_shortfall <= 1


def has_visible_attacker(facts: TurnStateFacts) -> bool:
    return facts.has_readiness


def always(_: TurnStateFacts) -> bool:
    return True


TURN_STATE_RULES: tuple[TurnStateRule, ...] = (
    TurnStateRule("attack-ready", is_attack_ready, TurnState.ATTACK_READY),
    TurnStateRule("attacker-near-ready", is_attacker_near_ready, TurnState.ATTACKER_NEAR_READY),
    TurnStateRule("no-attacker-ready", has_visible_attacker, TurnState.NO_ATTACKER_READY),
    TurnStateRule("fallback", always, TurnState.FALLBACK),
)


def classify_turn_state(facts: TurnStateFacts) -> TurnState:
    for rule in TURN_STATE_RULES:
        if rule.guard(facts):
            return rule.state
    raise AssertionError("unreachable turn-state rule table")


def analyze_board(obs: dict[str, Any]) -> BoardPlan:
    select = obs.get("select")
    current = obs.get("current")
    if select is None or current is None:
        return BoardPlan(TurnState.FALLBACK, False, False, False, False, 0, None, None, None, None, ())

    options = select.get("option") or []
    legal_attack = any(option.get("type") == OptionType.ATTACK.value for option in options)
    legal_attach = any(option.get("type") == OptionType.ATTACH.value for option in options)
    legal_play = any(option.get("type") == OptionType.PLAY.value for option in options)
    player = _own_player(current) or {}
    bench_count = len(player.get("bench") or [])
    readiness = tuple(_own_pokemon(current))
    active = next((item for item in readiness if item.area == "ACTIVE"), None)
    known = [item for item in readiness if item.best_shortfall is not None]
    closest = min(known, key=lambda item: (item.best_shortfall or 99, 0 if item.area == "ACTIVE" else 1), default=None)
    active_shortfall = active.best_shortfall if active is not None else None
    best_shortfall = closest.best_shortfall if closest is not None else None

    state = classify_turn_state(
        TurnStateFacts(
            legal_attack=legal_attack,
            active_shortfall=active_shortfall,
            best_shortfall=best_shortfall,
            has_readiness=bool(readiness),
        )
    )

    return BoardPlan(
        state=state,
        energy_attached=bool(current.get("energyAttached")),
        legal_attack_available=legal_attack,
        legal_attach_available=legal_attach,
        legal_play_available=legal_play,
        bench_count=bench_count,
        active_shortfall=active_shortfall,
        best_shortfall=best_shortfall,
        closest_area=closest.area if closest is not None else None,
        closest_index=closest.index if closest is not None else None,
        readiness=readiness,
    )
