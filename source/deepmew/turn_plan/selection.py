from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from cg.api import AreaType, OptionType, SelectContext, SelectType

from deepmew.deck_profiles import (
    BASIC_WATER_ENERGY_ID,
    CYRANO_CARD_ID,
    KYOGRE_CARD_ID,
    LILLIES_DETERMINATION_CARD_ID,
    MAXIMUM_BELT_CARD_ID,
    MEGA_ABOMASNOW_CARD_ID,
    MEGA_SIGNAL_CARD_ID,
    SNOVER_CARD_ID,
    WAITRESS_CARD_ID,
)
from deepmew.options.identity import card_name_for_id
from deepmew.policy.enums import enum_name
from deepmew.policy.selection import guard_selection, selection_bounds
from deepmew.state_machine import analyze_board


PLAN_CARD_PRIORITY = (
    MEGA_ABOMASNOW_CARD_ID,
    SNOVER_CARD_ID,
    BASIC_WATER_ENERGY_ID,
    KYOGRE_CARD_ID,
    MEGA_SIGNAL_CARD_ID,
    WAITRESS_CARD_ID,
    LILLIES_DETERMINATION_CARD_ID,
    CYRANO_CARD_ID,
    MAXIMUM_BELT_CARD_ID,
)
SETUP_POKEMON_PRIORITY = (SNOVER_CARD_ID, KYOGRE_CARD_ID)
ACTIVE_REPLACEMENT_PRIORITY = (MEGA_ABOMASNOW_CARD_ID, SNOVER_CARD_ID, KYOGRE_CARD_ID)
EVOLVE_TARGET_PRIORITY = (SNOVER_CARD_ID, MEGA_ABOMASNOW_CARD_ID)
ATTACH_TARGET_PRIORITY = (MEGA_ABOMASNOW_CARD_ID, SNOVER_CARD_ID, KYOGRE_CARD_ID)
VISIBLE_PLAN_CARD_IDS = frozenset(PLAN_CARD_PRIORITY)
ATTACH_CONTEXTS = {"ATTACH_TO", "ATTACH_FROM", "TO_HAND_ENERGY"}
EVOLVE_CONTEXTS = {"EVOLVE", "EVOLVES_FROM", "EVOLVES_TO"}


class SelectionPlanState(str, Enum):
    NO_OPINION = "NO_OPINION"
    PLAN_AWARE_ATTACH_TARGET = "PLAN_AWARE_ATTACH_TARGET"
    PLAN_AWARE_EVOLVE_TARGET = "PLAN_AWARE_EVOLVE_TARGET"
    PLAN_AWARE_CARD_PICK = "PLAN_AWARE_CARD_PICK"
    PLAN_AWARE_IS_FIRST = "PLAN_AWARE_IS_FIRST"
    FALLBACK = "FALLBACK"


@dataclass(frozen=True)
class SelectionOptionFact:
    index: int
    option_type: str
    area: str | None
    source_index: int | None
    player_index: int | None
    own_option: bool
    visible: bool
    card_id: int | None
    card_name: str | None
    source_zone: str | None
    hidden_reason: str
    attached_energy: int | None = None
    best_shortfall: int | None = None
    can_attack_by_energy: bool = False


@dataclass(frozen=True)
class SelectionFacts:
    select_type: str | None
    context: str | None
    option_count: int
    active_card_id: int | None
    active_card_name: str | None
    prior_turn_intent: str | None
    legal_option_identities: tuple[SelectionOptionFact, ...]
    important_evolve_targets: tuple[int, ...]
    attach_plan_targets: tuple[int, ...]
    visible_plan_card_indices: tuple[int, ...]
    fallback_reason: str = ""


@dataclass(frozen=True)
class SelectionRule:
    name: str
    guard: Callable[[SelectionFacts], bool]
    state: SelectionPlanState
    selector: Callable[[dict[str, Any], SelectionFacts], list[int] | None]


@dataclass(frozen=True)
class SelectionPlanDecision:
    state: SelectionPlanState
    rule_name: str
    selected_indices: tuple[int, ...]
    fallback: bool
    reason: str
    facts: SelectionFacts


def _current(obs: dict[str, Any]) -> dict[str, Any]:
    current = obs.get("current") if isinstance(obs, dict) else None
    return current if isinstance(current, dict) else {}


def _select(obs: dict[str, Any]) -> dict[str, Any]:
    select = obs.get("select") if isinstance(obs, dict) else None
    return select if isinstance(select, dict) else {}


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _your_index(current: dict[str, Any]) -> int:
    return _int_or_none(current.get("yourIndex")) or 0


def _players(current: dict[str, Any]) -> list[Any]:
    players = current.get("players") or []
    return players if isinstance(players, list) else []


def _player(current: dict[str, Any], player_index: int | None) -> dict[str, Any] | None:
    players = _players(current)
    if player_index is None or player_index < 0 or player_index >= len(players):
        return None
    player = players[player_index]
    return player if isinstance(player, dict) else None


def _own_player(current: dict[str, Any]) -> dict[str, Any] | None:
    return _player(current, _your_index(current))


def _area_name(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return AreaType(value).name
    except (TypeError, ValueError):
        return str(value).upper()


def _card_id(item: Any) -> int | None:
    if not isinstance(item, dict):
        return None
    return _int_or_none(item.get("id", item.get("cardId", item.get("cardID"))))


def _zone_for_area(player: dict[str, Any] | None, current: dict[str, Any], area: str | None) -> tuple[list[Any] | None, str | None]:
    if area == "LOOKING":
        looking = current.get("looking")
        return (looking if isinstance(looking, list) else None), "looking"
    if player is None:
        return None, None
    zone_name = {
        "HAND": "hand",
        "DISCARD": "discard",
        "PRIZE": "prize",
        "ACTIVE": "active",
        "BENCH": "bench",
        "STADIUM": "stadium",
        "ENERGY": "energyCards",
        "TOOL": "tools",
        "PRE_EVOLUTION": "preEvolution",
    }.get(area)
    if zone_name is None:
        return None, None
    zone = player.get(zone_name)
    return (zone if isinstance(zone, list) else None), zone_name


def _visible_card_from_option(obs: dict[str, Any], option: dict[str, Any]) -> tuple[int | None, str | None, str]:
    current = _current(obs)
    area = _area_name(option.get("area"))
    source_index = _int_or_none(option.get("index"))
    if area == "DECK":
        return None, None, "hidden-deck"
    player_index = _int_or_none(option.get("playerIndex"))
    if player_index is None:
        player_index = _your_index(current)
    own_option = player_index == _your_index(current)
    if area in {"HAND", "PRIZE", "DECK"} and not own_option:
        return None, None, "hidden-opponent-zone"
    zone, zone_name = _zone_for_area(_player(current, player_index), current, area)
    if zone is None or source_index is None:
        return None, zone_name, "missing-zone-or-index"
    if source_index < 0 or source_index >= len(zone):
        return None, zone_name, "index-out-of-range"
    card_id = _card_id(zone[source_index])
    if card_id is None:
        return None, zone_name, "card-hidden-or-unknown"
    return card_id, zone_name, "visible"


def _active_card_id(obs: dict[str, Any]) -> int | None:
    current = _current(obs)
    player = _own_player(current)
    active = (player or {}).get("active") or []
    if not active:
        return None
    return _card_id(active[0])


def _readiness_by_slot(obs: dict[str, Any]) -> dict[tuple[str, int], Any]:
    plan = analyze_board(obs)
    return {(item.area, item.index): item for item in plan.readiness}


def _option_fact(
    obs: dict[str, Any],
    option: dict[str, Any],
    index: int,
    readiness_by_slot: dict[tuple[str, int], Any],
) -> SelectionOptionFact:
    current = _current(obs)
    option_type = enum_name(OptionType, option.get("type"))
    area = _area_name(option.get("area"))
    source_index = _int_or_none(option.get("index"))
    player_index = _int_or_none(option.get("playerIndex"))
    if player_index is None:
        player_index = _your_index(current)
    own_option = player_index == _your_index(current)
    card_id, source_zone, hidden_reason = _visible_card_from_option(obs, option)
    readiness = readiness_by_slot.get((area or "", source_index if source_index is not None else -1)) if own_option else None
    return SelectionOptionFact(
        index=index,
        option_type=option_type,
        area=area,
        source_index=source_index,
        player_index=player_index,
        own_option=own_option,
        visible=card_id is not None,
        card_id=card_id,
        card_name=card_name_for_id(card_id),
        source_zone=source_zone,
        hidden_reason=hidden_reason,
        attached_energy=getattr(readiness, "attached_energy", None),
        best_shortfall=getattr(readiness, "best_shortfall", None),
        can_attack_by_energy=bool(getattr(readiness, "can_attack_by_energy", False)),
    )


def extract_selection_facts(obs: dict[str, Any], prior_intent: str | None = None) -> SelectionFacts:
    select = _select(obs)
    options = select.get("option") or []
    readiness_by_slot = _readiness_by_slot(obs)
    identities = tuple(
        _option_fact(obs, option, index, readiness_by_slot)
        for index, option in enumerate(options)
        if isinstance(option, dict)
    )
    active_card_id = _active_card_id(obs)
    context = enum_name(SelectContext, select.get("context"))
    important_evolve_targets = tuple(
        fact.index
        for fact in identities
        if fact.own_option
        and fact.card_id in EVOLVE_TARGET_PRIORITY
        and fact.visible
        and context in EVOLVE_CONTEXTS
    )
    attach_plan_targets = tuple(
        fact.index
        for fact in identities
        if fact.own_option
        and fact.area in {"ACTIVE", "BENCH"}
        and fact.visible
        and fact.card_id in ATTACH_TARGET_PRIORITY
        and context in ATTACH_CONTEXTS
    )
    visible_plan_card_indices = tuple(
        fact.index
        for fact in identities
        if fact.own_option and fact.visible and fact.card_id in VISIBLE_PLAN_CARD_IDS
    )
    return SelectionFacts(
        select_type=enum_name(SelectType, select.get("type")),
        context=context,
        option_count=len(options),
        active_card_id=active_card_id,
        active_card_name=card_name_for_id(active_card_id),
        prior_turn_intent=prior_intent,
        legal_option_identities=identities,
        important_evolve_targets=important_evolve_targets,
        attach_plan_targets=attach_plan_targets,
        visible_plan_card_indices=visible_plan_card_indices,
        fallback_reason="no-visible-options" if not options else "",
    )


def _by_index(facts: SelectionFacts) -> dict[int, SelectionOptionFact]:
    return {fact.index: fact for fact in facts.legal_option_identities}


def _priority_map(priority: tuple[int, ...]) -> dict[int, int]:
    return {card_id: rank for rank, card_id in enumerate(priority)}


def _selection_from_ranked_indices(obs: dict[str, Any], ranked_indices: list[int]) -> list[int] | None:
    select = _select(obs)
    min_count, max_count = selection_bounds(select, len(select.get("option") or []))
    if max_count == 0:
        return [] if min_count == 0 else None
    if len(ranked_indices) < min_count:
        return None
    return ranked_indices[:max_count]


def _rank_by_card_priority(
    facts: SelectionFacts,
    indices: tuple[int, ...] | list[int],
    priority: tuple[int, ...],
) -> list[int]:
    by_index = _by_index(facts)
    ranks = _priority_map(priority)
    return sorted(indices, key=lambda index: (ranks.get(by_index[index].card_id or -1, 999), index))


def _is_first_context(facts: SelectionFacts) -> bool:
    return facts.select_type == "YES_NO" and facts.context == "IS_FIRST"


def _select_go_second(obs: dict[str, Any], facts: SelectionFacts) -> list[int] | None:
    by_index = _by_index(facts)
    for index, fact in by_index.items():
        if fact.option_type == "NO":
            return [index]
    return None


def _has_visible_basic_water_attach_pick(facts: SelectionFacts) -> bool:
    return facts.context in ATTACH_CONTEXTS and any(
        fact.own_option and fact.visible and fact.card_id == BASIC_WATER_ENERGY_ID
        for fact in facts.legal_option_identities
    )


def _select_basic_water(obs: dict[str, Any], facts: SelectionFacts) -> list[int] | None:
    indices = [
        fact.index
        for fact in facts.legal_option_identities
        if fact.own_option and fact.visible and fact.card_id == BASIC_WATER_ENERGY_ID
    ]
    return _selection_from_ranked_indices(obs, sorted(indices))


def _has_attach_plan_target(facts: SelectionFacts) -> bool:
    return facts.context in ATTACH_CONTEXTS and bool(facts.attach_plan_targets)


def _select_attach_plan_target(obs: dict[str, Any], facts: SelectionFacts) -> list[int] | None:
    by_index = _by_index(facts)
    ranks = _priority_map(ATTACH_TARGET_PRIORITY)
    ranked = sorted(
        facts.attach_plan_targets,
        key=lambda index: (
            ranks.get(by_index[index].card_id or -1, 999),
            0 if by_index[index].can_attack_by_energy else 1,
            by_index[index].best_shortfall if by_index[index].best_shortfall is not None else 99,
            index,
        ),
    )
    return _selection_from_ranked_indices(obs, ranked)


def _has_evolve_target(facts: SelectionFacts) -> bool:
    return facts.context in EVOLVE_CONTEXTS and bool(facts.important_evolve_targets)


def _select_evolve_target(obs: dict[str, Any], facts: SelectionFacts) -> list[int] | None:
    ranked = _rank_by_card_priority(facts, facts.important_evolve_targets, EVOLVE_TARGET_PRIORITY)
    return _selection_from_ranked_indices(obs, ranked)


def _has_setup_active_pick(facts: SelectionFacts) -> bool:
    return facts.context == "SETUP_ACTIVE_POKEMON" and any(
        fact.own_option and fact.visible and fact.card_id in SETUP_POKEMON_PRIORITY
        for fact in facts.legal_option_identities
    )


def _select_setup_pokemon(obs: dict[str, Any], facts: SelectionFacts) -> list[int] | None:
    indices = [
        fact.index
        for fact in facts.legal_option_identities
        if fact.own_option and fact.visible and fact.card_id in SETUP_POKEMON_PRIORITY
    ]
    ranked = _rank_by_card_priority(facts, indices, SETUP_POKEMON_PRIORITY)
    return _selection_from_ranked_indices(obs, ranked)


def _has_setup_bench_pick(facts: SelectionFacts) -> bool:
    return facts.context == "SETUP_BENCH_POKEMON" and any(
        fact.own_option and fact.visible and fact.card_id in SETUP_POKEMON_PRIORITY
        for fact in facts.legal_option_identities
    )


def _has_to_active_pick(facts: SelectionFacts) -> bool:
    return facts.context == "TO_ACTIVE" and any(
        fact.own_option and fact.visible and fact.card_id in ACTIVE_REPLACEMENT_PRIORITY
        for fact in facts.legal_option_identities
    )


def _select_to_active(obs: dict[str, Any], facts: SelectionFacts) -> list[int] | None:
    by_index = _by_index(facts)
    ranks = _priority_map(ACTIVE_REPLACEMENT_PRIORITY)
    indices = [
        fact.index
        for fact in facts.legal_option_identities
        if fact.own_option and fact.visible and fact.card_id in ACTIVE_REPLACEMENT_PRIORITY
    ]
    ranked = sorted(
        indices,
        key=lambda index: (
            0 if by_index[index].can_attack_by_energy else 1,
            by_index[index].best_shortfall if by_index[index].best_shortfall is not None else 99,
            ranks.get(by_index[index].card_id or -1, 999),
            index,
        ),
    )
    return _selection_from_ranked_indices(obs, ranked)


def _has_visible_plan_card_pick(facts: SelectionFacts) -> bool:
    return facts.context == "TO_HAND" and bool(facts.visible_plan_card_indices)


def _select_visible_plan_card(obs: dict[str, Any], facts: SelectionFacts) -> list[int] | None:
    ranked = _rank_by_card_priority(facts, facts.visible_plan_card_indices, PLAN_CARD_PRIORITY)
    return _selection_from_ranked_indices(obs, ranked)


def _always(_: SelectionFacts) -> bool:
    return True


def _no_selection(_: dict[str, Any], __: SelectionFacts) -> list[int] | None:
    return None


SELECTION_RULES: tuple[SelectionRule, ...] = (
    SelectionRule("is-first-go-second", _is_first_context, SelectionPlanState.PLAN_AWARE_IS_FIRST, _select_go_second),
    SelectionRule("evolve-mega-line-target", _has_evolve_target, SelectionPlanState.PLAN_AWARE_EVOLVE_TARGET, _select_evolve_target),
    SelectionRule("attach-plan-target", _has_attach_plan_target, SelectionPlanState.PLAN_AWARE_ATTACH_TARGET, _select_attach_plan_target),
    SelectionRule("attach-visible-basic-water", _has_visible_basic_water_attach_pick, SelectionPlanState.PLAN_AWARE_CARD_PICK, _select_basic_water),
    SelectionRule("setup-active-plan-pokemon", _has_setup_active_pick, SelectionPlanState.PLAN_AWARE_CARD_PICK, _select_setup_pokemon),
    SelectionRule("setup-bench-plan-pokemon", _has_setup_bench_pick, SelectionPlanState.PLAN_AWARE_CARD_PICK, _select_setup_pokemon),
    SelectionRule("to-active-plan-pokemon", _has_to_active_pick, SelectionPlanState.PLAN_AWARE_CARD_PICK, _select_to_active),
    SelectionRule("visible-plan-card-to-hand", _has_visible_plan_card_pick, SelectionPlanState.PLAN_AWARE_CARD_PICK, _select_visible_plan_card),
    SelectionRule("fallback", _always, SelectionPlanState.FALLBACK, _no_selection),
)


def plan_selection_action(obs: dict[str, Any], prior_intent: str | None = None) -> SelectionPlanDecision:
    facts = extract_selection_facts(obs, prior_intent)
    for rule in SELECTION_RULES:
        if not rule.guard(facts):
            continue
        selected = rule.selector(obs, facts)
        if selected is None:
            reason = facts.fallback_reason or "no-safe-plan-selection"
            return SelectionPlanDecision(rule.state, rule.name, (), True, reason, facts)
        guarded = guard_selection(selected, _select(obs))
        if guarded != selected:
            return SelectionPlanDecision(rule.state, rule.name, tuple(guarded), True, "planned-selection-failed-guard", facts)
        return SelectionPlanDecision(rule.state, rule.name, tuple(guarded), False, "planned-selection-action", facts)
    raise AssertionError("unreachable selection rule table")


def choose_plan_selection_action(obs: dict[str, Any], prior_intent: str | None = None) -> list[int] | None:
    decision = plan_selection_action(obs, prior_intent)
    if decision.fallback:
        return None
    return list(decision.selected_indices)


def selection_option_summaries(facts: SelectionFacts) -> list[dict[str, Any]]:
    return [
        {
            "index": fact.index,
            "option_type": fact.option_type,
            "area": fact.area,
            "source_index": fact.source_index,
            "player_index": fact.player_index,
            "own_option": fact.own_option,
            "visible": fact.visible,
            "card_id": fact.card_id,
            "card_name": fact.card_name,
            "source_zone": fact.source_zone,
            "hidden_reason": fact.hidden_reason,
            "attached_energy": fact.attached_energy,
            "best_shortfall": fact.best_shortfall,
            "can_attack_by_energy": fact.can_attack_by_energy,
        }
        for fact in facts.legal_option_identities
    ]
