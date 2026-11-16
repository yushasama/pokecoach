from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable

from cg.api import AreaType, OptionType, SelectContext, SelectType
from deepmew.deck_profiles import (
    CURRENT_WATER_MEGA_ABOMASNOW,
    KYOGRE_CARD_ID,
    MEGA_ABOMASNOW_CARD_ID,
    SNOVER_CARD_ID,
)
from deepmew.options import resolve_option_identity
from deepmew.policy.attacks import attack_pressure
from deepmew.state_machine import BoardPlan


class EvolvePriorityState(str, Enum):
    NO_OPINION = "NO_OPINION"
    NOT_MAIN_PHASE = "NOT_MAIN_PHASE"
    NO_IMPORTANT_EVOLVE = "NO_IMPORTANT_EVOLVE"
    PRESERVE_KO_OR_PRIZE_ATTACK = "PRESERVE_KO_OR_PRIZE_ATTACK"
    PRESERVE_TARGET_CREATING_PLAY = "PRESERVE_TARGET_CREATING_PLAY"
    DELAY_NON_TARGET_PLAY_FOR_EVOLVE = "DELAY_NON_TARGET_PLAY_FOR_EVOLVE"


class EvolvePriorityAction(str, Enum):
    NO_OPINION = "NO_OPINION"
    DELAY_NON_TARGET_PLAY = "DELAY_NON_TARGET_PLAY"


@dataclass(frozen=True)
class EvolvePriorityFacts:
    select_type: str | None
    context: str | None
    important_evolve_indices: tuple[int, ...]
    non_target_play_indices: tuple[int, ...]
    target_creating_play_indices: tuple[int, ...]
    ko_or_prize_attack_available: bool
    weak_attack_available: bool
    bench_count: int
    lone_active: bool


@dataclass(frozen=True)
class EvolvePriorityDecision:
    state: EvolvePriorityState
    action: EvolvePriorityAction
    play_score_delta: int = 0
    reason: str = ""


@dataclass(frozen=True)
class EvolvePriorityRule:
    name: str
    guard: Callable[[EvolvePriorityFacts], bool]
    state: EvolvePriorityState
    action: EvolvePriorityAction
    play_score_delta: int = 0

    def to_decision(self) -> EvolvePriorityDecision:
        return EvolvePriorityDecision(
            state=self.state,
            action=self.action,
            play_score_delta=self.play_score_delta,
            reason=self.name,
        )


def _enum_name(enum_cls: Any, value: Any) -> str | None:
    if value is None:
        return None
    try:
        return enum_cls(value).name
    except (TypeError, ValueError):
        return f"UNKNOWN_{value}"


def _option_type(option: dict[str, Any]) -> str | None:
    return _enum_name(OptionType, option.get("type"))


def _area_name(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return AreaType(value).name
    except (TypeError, ValueError):
        return str(value).upper()


def _option_index(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _card_id(value: Any) -> int | None:
    if not isinstance(value, dict):
        return None
    raw = value.get("id", value.get("cardId", value.get("cardID")))
    return _option_index(raw)


def _own_player(obs: dict[str, Any]) -> dict[str, Any]:
    current = obs.get("current") or {}
    players = current.get("players") or []
    your_index = _option_index(current.get("yourIndex")) or 0
    if not isinstance(players, list) or your_index < 0 or your_index >= len(players):
        return {}
    player = players[your_index]
    return player if isinstance(player, dict) else {}


def _readiness_for_target(plan: BoardPlan | None, option: dict[str, Any]) -> Any | None:
    if plan is None:
        return None
    area = _area_name(option.get("inPlayArea"))
    index = _option_index(option.get("inPlayIndex")) or 0
    return next(
        (
            item
            for item in getattr(plan, "readiness", ()) or ()
            if item.area == area and item.index == index
        ),
        None,
    )


def is_important_mega_evolve(
    obs: dict[str, Any],
    option: dict[str, Any],
    plan: BoardPlan | None,
) -> bool:
    if _option_type(option) != "EVOLVE":
        return False
    identity = resolve_option_identity(obs, option)
    if identity.card_id == MEGA_ABOMASNOW_CARD_ID and CURRENT_WATER_MEGA_ABOMASNOW.is_mega_line(
        identity.target_card_id
    ):
        return True
    target = _readiness_for_target(plan, option)
    return bool(
        identity.card_id == MEGA_ABOMASNOW_CARD_ID
        and target is not None
        and CURRENT_WATER_MEGA_ABOMASNOW.is_mega_line(target.card_id)
    )


def is_target_creating_play(obs: dict[str, Any], option: dict[str, Any]) -> bool:
    if _option_type(option) != "PLAY":
        return False
    identity = resolve_option_identity(obs, option)
    card_id = identity.card_id
    if card_id is None:
        source_index = _option_index(option.get("index"))
        hand = _own_player(obs).get("hand") or []
        if source_index is not None and isinstance(hand, list) and 0 <= source_index < len(hand):
            card_id = _card_id(hand[source_index])
    return card_id in {SNOVER_CARD_ID, KYOGRE_CARD_ID}


def is_non_target_play(obs: dict[str, Any], option: dict[str, Any]) -> bool:
    return _option_type(option) == "PLAY" and not is_target_creating_play(obs, option)


def extract_evolve_priority_facts(
    obs: dict[str, Any],
    select: dict[str, Any] | None,
    options: list[dict[str, Any]],
    plan: BoardPlan | None,
    *,
    use_card_semantics: bool = True,
) -> EvolvePriorityFacts:
    select = select or {}
    options = options or []
    attack_pressures = [
        attack_pressure(option, obs, use_card_semantics)
        for option in options
        if _option_type(option) == "ATTACK"
    ]
    player = _own_player(obs)
    active = player.get("active") or []

    return EvolvePriorityFacts(
        select_type=_enum_name(SelectType, select.get("type")),
        context=_enum_name(SelectContext, select.get("context")),
        important_evolve_indices=tuple(
            index for index, option in enumerate(options) if is_important_mega_evolve(obs, option, plan)
        ),
        non_target_play_indices=tuple(
            index for index, option in enumerate(options) if is_non_target_play(obs, option)
        ),
        target_creating_play_indices=tuple(
            index for index, option in enumerate(options) if is_target_creating_play(obs, option)
        ),
        ko_or_prize_attack_available=any(pressure in {"ko", "meaningful"} for pressure in attack_pressures),
        weak_attack_available="weak" in attack_pressures,
        bench_count=int(getattr(plan, "bench_count", 0) or 0),
        lone_active=bool(active) and int(getattr(plan, "bench_count", 0) or 0) == 0,
    )


def is_not_main_phase(facts: EvolvePriorityFacts) -> bool:
    return facts.select_type != "MAIN" or facts.context != "MAIN"


def has_no_important_evolve(facts: EvolvePriorityFacts) -> bool:
    return not facts.important_evolve_indices


def has_ko_or_prize_attack(facts: EvolvePriorityFacts) -> bool:
    return facts.ko_or_prize_attack_available


def has_non_target_play_to_delay(facts: EvolvePriorityFacts) -> bool:
    return bool(facts.non_target_play_indices)


def always(_: EvolvePriorityFacts) -> bool:
    return True


EVOLVE_PRIORITY_RULES: tuple[EvolvePriorityRule, ...] = (
    EvolvePriorityRule(
        name="not-main-phase",
        guard=is_not_main_phase,
        state=EvolvePriorityState.NOT_MAIN_PHASE,
        action=EvolvePriorityAction.NO_OPINION,
    ),
    EvolvePriorityRule(
        name="no-important-evolve",
        guard=has_no_important_evolve,
        state=EvolvePriorityState.NO_IMPORTANT_EVOLVE,
        action=EvolvePriorityAction.NO_OPINION,
    ),
    EvolvePriorityRule(
        name="preserve-ko-or-prize-attack",
        guard=has_ko_or_prize_attack,
        state=EvolvePriorityState.PRESERVE_KO_OR_PRIZE_ATTACK,
        action=EvolvePriorityAction.NO_OPINION,
    ),
    EvolvePriorityRule(
        name="important-evolve-before-non-target-play",
        guard=has_non_target_play_to_delay,
        state=EvolvePriorityState.DELAY_NON_TARGET_PLAY_FOR_EVOLVE,
        action=EvolvePriorityAction.DELAY_NON_TARGET_PLAY,
        play_score_delta=-28,
    ),
    EvolvePriorityRule(
        name="no-opinion",
        guard=always,
        state=EvolvePriorityState.NO_OPINION,
        action=EvolvePriorityAction.NO_OPINION,
    ),
)


def classify_evolve_priority(facts: EvolvePriorityFacts) -> EvolvePriorityDecision:
    for rule in EVOLVE_PRIORITY_RULES:
        if rule.guard(facts):
            return rule.to_decision()
    raise AssertionError("unreachable evolve-priority rule table")


def apply_evolve_priority_adjustment(
    candidates: list[Any],
    decision: EvolvePriorityDecision,
    facts: EvolvePriorityFacts,
) -> list[Any]:
    if decision.action == EvolvePriorityAction.NO_OPINION:
        return candidates
    if decision.action == EvolvePriorityAction.DELAY_NON_TARGET_PLAY:
        return penalize_non_target_play_candidates(candidates, decision, facts)
    raise ValueError(f"unsupported evolve priority action: {decision.action}")


def penalize_non_target_play_candidates(
    candidates: list[Any],
    decision: EvolvePriorityDecision,
    facts: EvolvePriorityFacts,
) -> list[Any]:
    non_target_indices = set(facts.non_target_play_indices)
    adjusted = []
    for candidate in candidates:
        if candidate.index in non_target_indices and candidate.option_type == "PLAY":
            adjusted.append(
                replace(
                    candidate,
                    score=candidate.score + decision.play_score_delta,
                    reasons=tuple(candidate.reasons)
                    + (
                        f"evolve-priority-state={decision.state.value}",
                        decision.reason,
                        f"delay-non-target-play={decision.play_score_delta}",
                    ),
                )
            )
        else:
            adjusted.append(candidate)
    return adjusted
