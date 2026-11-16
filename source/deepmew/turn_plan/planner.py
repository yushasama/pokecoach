from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from cg.api import OptionType, SelectContext, SelectType

from deepmew.deck_profiles import (
    CYRANO_CARD_ID,
    CURRENT_WATER_MEGA_ABOMASNOW,
    KYOGRE_CARD_ID,
    LILLIES_DETERMINATION_CARD_ID,
    MEGA_ABOMASNOW_CARD_ID,
    MEGA_SIGNAL_CARD_ID,
    SNOVER_CARD_ID,
    WAITRESS_CARD_ID,
)
from deepmew.options.identity import card_name_for_id, resolve_option_identity
from deepmew.policy.attacks import attack_damage_for_option, attack_pressure
from deepmew.policy.enums import enum_name
from deepmew.policy.selection import guard_selection
from deepmew.policy_rules.evolve_priority import is_important_mega_evolve
from deepmew.state_machine import analyze_board, attachment_effect


SETUP_SEARCH_PLAY_PRIORITY = (
    MEGA_SIGNAL_CARD_ID,
    WAITRESS_CARD_ID,
    LILLIES_DETERMINATION_CARD_ID,
    CYRANO_CARD_ID,
)
TARGET_CREATING_PLAY_PRIORITY = (SNOVER_CARD_ID, KYOGRE_CARD_ID)


class TurnIntent(str, Enum):
    TAKE_KO = "TAKE_KO"
    APPLY_PRIZE_PRESSURE = "APPLY_PRIZE_PRESSURE"
    DRAW_OR_SEARCH_FOR_PLAN = "DRAW_OR_SEARCH_FOR_PLAN"
    CREATE_ATTACH_TARGET = "CREATE_ATTACH_TARGET"
    EVOLVE_PRIMARY_LINE = "EVOLVE_PRIMARY_LINE"
    ATTACH_TO_PLAN_TARGET = "ATTACH_TO_PLAN_TARGET"
    ATTACK_MEANINGFULLY = "ATTACK_MEANINGFULLY"
    FALLBACK = "FALLBACK"


@dataclass(frozen=True)
class LegalOptionFact:
    index: int
    option_type: str
    card_id: int | None = None
    card_name: str | None = None
    attack_id: int | None = None
    attack_pressure: str | None = None
    attack_damage: int | None = None
    improves_plan: bool = False
    enables_attack: bool = False
    target_is_closest: bool = False
    important_mega_evolve: bool = False
    target_creating_play: bool = False
    setup_search_play: bool = False


@dataclass(frozen=True)
class TurnPlanFacts:
    select_type: str | None
    context: str | None
    board_state: str
    active_card_id: int | None
    active_card_name: str | None
    bench_count: int
    energy_attached: bool
    primary_mega_in_play: bool
    primary_mega_active: bool
    primary_plan_online: bool
    primary_plan_incomplete: bool
    weak_attack_only: bool
    meaningful_attack_available: bool
    ko_or_prize_attack_available: bool
    legal_options: tuple[LegalOptionFact, ...]
    ko_attack_indices: tuple[int, ...]
    prize_pressure_attack_indices: tuple[int, ...]
    meaningful_attack_indices: tuple[int, ...]
    weak_attack_indices: tuple[int, ...]
    setup_search_play_indices: tuple[int, ...]
    target_creating_play_indices: tuple[int, ...]
    important_evolve_indices: tuple[int, ...]
    attach_to_plan_indices: tuple[int, ...]


@dataclass(frozen=True)
class TurnPlanRule:
    name: str
    guard: Callable[[TurnPlanFacts], bool]
    intent: TurnIntent


@dataclass(frozen=True)
class TurnPlanDecision:
    intent: TurnIntent
    rule_name: str
    selected_indices: tuple[int, ...]
    fallback: bool
    reason: str
    facts: TurnPlanFacts


def _active_card_id(obs: dict[str, Any]) -> int | None:
    current = obs.get("current") or {}
    players = current.get("players") or []
    your_index = int(current.get("yourIndex") or 0)
    if your_index >= len(players):
        return None
    active = (players[your_index] or {}).get("active") or []
    if not active:
        return None
    try:
        return int(active[0].get("id"))
    except (AttributeError, TypeError, ValueError):
        return None


def _own_pokemon_card_ids(obs: dict[str, Any]) -> frozenset[int]:
    current = obs.get("current") or {}
    players = current.get("players") or []
    your_index = int(current.get("yourIndex") or 0)
    if your_index >= len(players):
        return frozenset()
    player = players[your_index] or {}
    ids: set[int] = set()
    for slot in (player.get("active") or []) + (player.get("bench") or []):
        if not isinstance(slot, dict):
            continue
        try:
            ids.add(int(slot.get("id")))
        except (TypeError, ValueError):
            continue
    return frozenset(ids)


def _option_fact(obs: dict[str, Any], option: dict[str, Any], index: int, use_card_semantics: bool) -> LegalOptionFact:
    option_type = enum_name(OptionType, option.get("type"))
    identity = resolve_option_identity(obs, option)
    plan = analyze_board(obs)
    pressure = None
    damage = None
    attack_id = identity.attack_id
    improves_plan = False
    enables_attack = False
    target_is_closest = False
    important_evolve = False
    target_creating = False
    setup_search = False

    if option_type == "ATTACK":
        pressure = attack_pressure(option, obs, use_card_semantics)
        damage = attack_damage_for_option(option, obs, use_card_semantics)
    elif option_type == "ATTACH":
        effect = attachment_effect(plan, option)
        improves_plan = effect.improves_readiness
        enables_attack = effect.enables_attack
        target_is_closest = effect.target_is_closest
    elif option_type == "EVOLVE":
        important_evolve = is_important_mega_evolve(obs, option, plan)
    elif option_type == "PLAY":
        card_id = identity.card_id
        target_creating = card_id in TARGET_CREATING_PLAY_PRIORITY
        setup_search = card_id in SETUP_SEARCH_PLAY_PRIORITY or CURRENT_WATER_MEGA_ABOMASNOW.is_setup_card(card_id)

    return LegalOptionFact(
        index=index,
        option_type=option_type,
        card_id=identity.card_id,
        card_name=identity.card_name,
        attack_id=attack_id,
        attack_pressure=pressure,
        attack_damage=damage,
        improves_plan=improves_plan,
        enables_attack=enables_attack,
        target_is_closest=target_is_closest,
        important_mega_evolve=important_evolve,
        target_creating_play=target_creating,
        setup_search_play=setup_search,
    )


def extract_turn_plan_facts(obs: dict[str, Any], *, use_card_semantics: bool = True) -> TurnPlanFacts:
    select = obs.get("select") or {}
    options = select.get("option") or []
    plan = analyze_board(obs)
    option_facts = tuple(_option_fact(obs, option, index, use_card_semantics) for index, option in enumerate(options))
    active_card_id = _active_card_id(obs)
    ko = tuple(fact.index for fact in option_facts if fact.option_type == "ATTACK" and fact.attack_pressure == "ko")
    prize = tuple(
        fact.index
        for fact in option_facts
        if fact.option_type == "ATTACK" and fact.attack_pressure in {"ko", "meaningful"}
    )
    meaningful = tuple(fact.index for fact in option_facts if fact.option_type == "ATTACK" and fact.attack_pressure == "meaningful")
    weak = tuple(fact.index for fact in option_facts if fact.option_type == "ATTACK" and fact.attack_pressure == "weak")
    setup = tuple(fact.index for fact in option_facts if fact.setup_search_play)
    target = tuple(fact.index for fact in option_facts if fact.target_creating_play)
    evolve = tuple(fact.index for fact in option_facts if fact.important_mega_evolve)
    attach = tuple(fact.index for fact in option_facts if fact.option_type == "ATTACH" and fact.improves_plan)

    own_card_ids = _own_pokemon_card_ids(obs)
    primary_mega_in_play = MEGA_ABOMASNOW_CARD_ID in own_card_ids
    primary_mega_active = active_card_id == MEGA_ABOMASNOW_CARD_ID
    ko_or_prize_attack_available = bool(prize)
    meaningful_attack_available = bool(meaningful)
    weak_attack_only = bool(weak) and not ko_or_prize_attack_available
    # The real deck plan is online only when the evolved Mega Abomasnow ex is the
    # active attacker and can apply genuine KO/meaningful pressure. A base Snover
    # poking with a weak attack is base-form tempo, not a completed Mega plan.
    primary_plan_online = primary_mega_active and ko_or_prize_attack_available
    primary_plan_incomplete = not primary_plan_online

    return TurnPlanFacts(
        select_type=enum_name(SelectType, select.get("type")),
        context=enum_name(SelectContext, select.get("context")),
        board_state=plan.state.value,
        active_card_id=active_card_id,
        active_card_name=card_name_for_id(active_card_id),
        bench_count=plan.bench_count,
        energy_attached=plan.energy_attached,
        primary_mega_in_play=primary_mega_in_play,
        primary_mega_active=primary_mega_active,
        primary_plan_online=primary_plan_online,
        primary_plan_incomplete=primary_plan_incomplete,
        weak_attack_only=weak_attack_only,
        meaningful_attack_available=meaningful_attack_available,
        ko_or_prize_attack_available=ko_or_prize_attack_available,
        legal_options=option_facts,
        ko_attack_indices=ko,
        prize_pressure_attack_indices=prize,
        meaningful_attack_indices=meaningful,
        weak_attack_indices=weak,
        setup_search_play_indices=setup,
        target_creating_play_indices=target,
        important_evolve_indices=evolve,
        attach_to_plan_indices=attach,
    )


def _is_main(facts: TurnPlanFacts) -> bool:
    return facts.select_type == "MAIN" and facts.context == "MAIN"


def has_ko_attack(facts: TurnPlanFacts) -> bool:
    return _is_main(facts) and bool(facts.ko_attack_indices)


def has_prize_pressure_attack(facts: TurnPlanFacts) -> bool:
    return _is_main(facts) and bool(facts.prize_pressure_attack_indices)


def has_setup_search_for_incomplete_plan(facts: TurnPlanFacts) -> bool:
    return _is_main(facts) and facts.primary_plan_incomplete and bool(facts.setup_search_play_indices)


def needs_target_creating_body(facts: TurnPlanFacts) -> bool:
    return _is_main(facts) and facts.bench_count == 0 and bool(facts.target_creating_play_indices)


def has_important_evolve_ready(facts: TurnPlanFacts) -> bool:
    return _is_main(facts) and bool(facts.important_evolve_indices)


def has_attach_to_plan_target(facts: TurnPlanFacts) -> bool:
    return _is_main(facts) and not facts.energy_attached and bool(facts.attach_to_plan_indices)


def has_meaningful_attack_without_development(facts: TurnPlanFacts) -> bool:
    return _is_main(facts) and bool(facts.meaningful_attack_indices)


def always(_: TurnPlanFacts) -> bool:
    return True


TURN_PLAN_RULES: tuple[TurnPlanRule, ...] = (
    TurnPlanRule("take-visible-ko", has_ko_attack, TurnIntent.TAKE_KO),
    TurnPlanRule("apply-prize-pressure", has_prize_pressure_attack, TurnIntent.APPLY_PRIZE_PRESSURE),
    TurnPlanRule("draw-or-search-for-incomplete-plan", has_setup_search_for_incomplete_plan, TurnIntent.DRAW_OR_SEARCH_FOR_PLAN),
    TurnPlanRule("create-attach-target", needs_target_creating_body, TurnIntent.CREATE_ATTACH_TARGET),
    TurnPlanRule("evolve-primary-mega-line", has_important_evolve_ready, TurnIntent.EVOLVE_PRIMARY_LINE),
    TurnPlanRule("attach-to-plan-target", has_attach_to_plan_target, TurnIntent.ATTACH_TO_PLAN_TARGET),
    TurnPlanRule("attack-meaningfully", has_meaningful_attack_without_development, TurnIntent.ATTACK_MEANINGFULLY),
    TurnPlanRule("fallback", always, TurnIntent.FALLBACK),
)


def choose_turn_intent(facts: TurnPlanFacts) -> tuple[TurnIntent, str]:
    for rule in TURN_PLAN_RULES:
        if rule.guard(facts):
            return rule.intent, rule.name
    raise AssertionError("unreachable turn-plan rule table")


def _facts_by_index(facts: TurnPlanFacts) -> dict[int, LegalOptionFact]:
    return {fact.index: fact for fact in facts.legal_options}


def _best_attack_index(facts: TurnPlanFacts, indices: tuple[int, ...]) -> int | None:
    by_index = _facts_by_index(facts)
    return max(indices, key=lambda index: (by_index[index].attack_damage or -1, -index), default=None)


def _best_by_card_priority(facts: TurnPlanFacts, indices: tuple[int, ...], priority: tuple[int, ...]) -> int | None:
    by_index = _facts_by_index(facts)
    priority_map = {card_id: rank for rank, card_id in enumerate(priority)}
    return min(indices, key=lambda index: (priority_map.get(by_index[index].card_id, 999), index), default=None)


def _best_attach_index(facts: TurnPlanFacts) -> int | None:
    by_index = _facts_by_index(facts)
    return max(
        facts.attach_to_plan_indices,
        key=lambda index: (
            1 if by_index[index].enables_attack else 0,
            1 if by_index[index].target_is_closest else 0,
            -index,
        ),
        default=None,
    )


def choose_action_for_intent(facts: TurnPlanFacts, intent: TurnIntent) -> int | None:
    if intent == TurnIntent.TAKE_KO:
        return _best_attack_index(facts, facts.ko_attack_indices)
    if intent == TurnIntent.APPLY_PRIZE_PRESSURE:
        return _best_attack_index(facts, facts.prize_pressure_attack_indices)
    if intent == TurnIntent.DRAW_OR_SEARCH_FOR_PLAN:
        return _best_by_card_priority(facts, facts.setup_search_play_indices, SETUP_SEARCH_PLAY_PRIORITY)
    if intent == TurnIntent.CREATE_ATTACH_TARGET:
        return _best_by_card_priority(facts, facts.target_creating_play_indices, TARGET_CREATING_PLAY_PRIORITY)
    if intent == TurnIntent.EVOLVE_PRIMARY_LINE:
        return facts.important_evolve_indices[0] if facts.important_evolve_indices else None
    if intent == TurnIntent.ATTACH_TO_PLAN_TARGET:
        return _best_attach_index(facts)
    if intent == TurnIntent.ATTACK_MEANINGFULLY:
        return _best_attack_index(facts, facts.meaningful_attack_indices)
    return None


def plan_first_action(obs: dict[str, Any], *, use_card_semantics: bool = True) -> TurnPlanDecision:
    facts = extract_turn_plan_facts(obs, use_card_semantics=use_card_semantics)
    intent, rule_name = choose_turn_intent(facts)
    index = choose_action_for_intent(facts, intent)
    select = obs.get("select") or {}
    if index is None:
        return TurnPlanDecision(intent, rule_name, (), True, "no-confident-plan-action", facts)
    guarded = guard_selection([index], select)
    if guarded != [index]:
        return TurnPlanDecision(intent, rule_name, tuple(guarded), True, "planned-action-failed-selection-guard", facts)
    return TurnPlanDecision(intent, rule_name, (index,), False, "planned-first-action", facts)
