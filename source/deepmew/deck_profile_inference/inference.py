"""Aggregate v2 function tags over a 60-card decklist into a deck profile.

Everything here is count-weighted: a 4-of line contributes 4× the strength of a
one-of tech, and primary ``reason_tags`` are only emitted when a plan clears a
copy-strength threshold, so a single tech card can never define the deck. Deck
archetype labels are carried through as metadata and are never read as inference
logic. The three kinds of bench pressure (ordinary attack damage, attack-based
damage counters, Ability-based damage counters) and the kinds of protection
(counter-placement prevention, attack-based bench protection, self-bench Tera
immunity, Ability prevention, damage wall/reduction) are kept strictly separate.
"""

from __future__ import annotations

from typing import Callable

from deepmew.deck_profile_inference.schemas import (
    CardRef,
    DeckList,
    DeckProfile,
    ProfileConfig,
    STAGE_1,
    STAGE_2,
    STAGE_BASIC,
    SUBTYPE_BASIC_ENERGY,
    SUBTYPE_ITEM,
    SUBTYPE_SPECIAL_ENERGY,
    SUBTYPE_STADIUM,
    SUBTYPE_TOOL,
    V2Card,
)

Resolved = list[tuple[V2Card, int]]
Predicate = Callable[[V2Card], bool]


def _resolve(deck: DeckList, v2map: dict[str, V2Card]) -> tuple[Resolved, int, int]:
    resolved: Resolved = []
    resolved_copies = 0
    unresolved_copies = 0
    for card_id, count in deck.card_counts.items():
        card = v2map.get(str(card_id))
        if card is None:
            unresolved_copies += count
            continue
        resolved.append((card, count))
        resolved_copies += count
    return resolved, resolved_copies, unresolved_copies


def _collect(resolved: Resolved, pred: Predicate, note: str, config: ProfileConfig) -> list[CardRef]:
    refs = [
        CardRef(
            card_id=card.card_id,
            name=card.name,
            count=count,
            is_core=count >= config.core_copy_threshold,
            note=note,
        )
        for card, count in resolved
        if pred(card)
    ]
    by_id = {card.card_id: card for card, _ in resolved}
    refs.sort(key=lambda r: (-r.count, -by_id[r.card_id].max_damage, r.name))
    return refs


def _strength(refs: list[CardRef]) -> int:
    return sum(r.count for r in refs)


def _has_rule(card: V2Card, *rule_ids: str) -> bool:
    return any(rid in card.rule_ids for rid in rule_ids)


# --- protection classification (the v1 conflation bug, fixed) ----------------- #


def _protection_plan(resolved: Resolved, config: ProfileConfig) -> dict[str, list[CardRef]]:
    plan: dict[str, list[CardRef]] = {}

    def add(kind: str, pred: Predicate) -> None:
        refs = _collect(resolved, pred, kind, config)
        if refs:
            plan[kind] = refs

    add("counter_placement_prevention", lambda c: _has_rule(c, "prevents_counter_placement", "protects_bench_from_counters"))
    add("attack_bench_protection", lambda c: _has_rule(c, "protects_own_bench"))
    add("self_bench_protection", lambda c: c.has("self_bench_protection"))
    add("ability_prevention", lambda c: c.has("ability_lock"))
    add("damage_wall", lambda c: c.has("wall_tank") or c.has("attack_prevention"))
    add("damage_reduction", lambda c: c.has("damage_reduction"))
    return plan


# --- dependencies / choke points ---------------------------------------------- #


def _stage_copies(resolved: Resolved, stage_type: str) -> int:
    return sum(count for card, count in resolved if card.stage_type == stage_type)


def _dependencies(resolved: Resolved, config: ProfileConfig) -> dict[str, int]:
    def copies(pred: Predicate) -> int:
        return sum(count for card, count in resolved if pred(card))

    stage2 = _stage_copies(resolved, STAGE_2)
    stage1 = _stage_copies(resolved, STAGE_1)
    damage_counter = copies(lambda c: c.has("bench_damage_counters") or c.has("active_damage_counters"))
    deps = {
        "stadium_dependency": copies(lambda c: c.subtype == SUBTYPE_STADIUM),
        "tool_dependency": copies(lambda c: c.subtype == SUBTYPE_TOOL),
        "special_energy_dependency": copies(lambda c: c.subtype == SUBTYPE_SPECIAL_ENERGY),
        "stage2_dependency": stage2,
        "stage1_dependency": stage1,
        "ability_engine_dependency": copies(lambda c: c.has("ability_engine")),
        "bench_fragility": copies(
            lambda c: c.is_pokemon
            and c.is_function_card
            and not c.has("attacker")
            and (c.hp is None or c.hp <= config.bench_fragility_hp_max)
        ),
        "item_reliance": copies(lambda c: c.subtype == SUBTYPE_ITEM),
        "hand_size_plan": copies(lambda c: c.has("hand_size_scaling")),
        "damage_counter_plan": damage_counter,
        "bench_attack_damage_plan": copies(lambda c: c.has("bench_attack_damage")),
        "self_ko_liability": copies(lambda c: c.has("self_ko")),
    }
    return {k: v for k, v in deps.items() if v > 0}


def _choke_points(deps: dict[str, int], config: ProfileConfig) -> list[str]:
    chokes: list[str] = []
    if deps.get("stage2_dependency", 0) >= 2:
        chokes.append("stage2_evolution_setup_reliance")
    if deps.get("ability_engine_dependency", 0) >= 1:
        chokes.append("ability_engine_reliance")
    if deps.get("special_energy_dependency", 0) >= 1:
        chokes.append("special_energy_reliance")
    if deps.get("item_reliance", 0) >= config.high_item_reliance:
        chokes.append("high_item_reliance")
    if deps.get("self_ko_liability", 0) >= 1:
        chokes.append("self_ko_liability")
    if deps.get("hand_size_plan", 0) >= 1:
        chokes.append("hand_size_dependent")
    if deps.get("bench_fragility", 0) >= 2:
        chokes.append("bench_setup_fragility")
    if deps.get("stadium_dependency", 0) >= 2:
        chokes.append("stadium_reliance")
    return chokes


# --- reason tags (generalized deck identity) ---------------------------------- #


def _reason_tags(
    profile: DeckProfile,
    resolved: Resolved,
    main_attackers: list[CardRef],
    config: ProfileConfig,
) -> list[str]:
    tags: list[str] = []
    gate = config.min_reason_strength

    def strength(pred: Predicate) -> int:
        return sum(count for card, count in resolved if pred(card))

    stage2 = profile.evolution_plan.get("stage2_pokemon_count", 0)
    stage1 = profile.evolution_plan.get("stage1_pokemon_count", 0)
    if stage2 >= 2:
        tags.append("stage2_evolution_deck")
    elif stage1 >= 2:
        tags.append("stage1_evolution_deck")
    elif main_attackers:
        tags.append("basic_focused_deck")

    if strength(lambda c: c.has("bench_attack_damage")) >= gate:
        tags.append("ordinary_bench_attack_spread")
    if strength(lambda c: c.has("bench_damage_counters")) >= gate:
        tags.append("bench_damage_counter_pressure")
    if strength(lambda c: c.has("ability_damage_counters")) >= gate:
        tags.append("ability_damage_counter_engine")
    if strength(lambda c: c.has("active_damage_counters")) >= gate:
        tags.append("active_damage_counter_burst")
    if strength(lambda c: c.has("hand_size_scaling")) >= gate:
        tags.append("hand_size_scaling_plan")
    if strength(lambda c: c.has("gust_opponent") or c.has("switch_opponent")) >= gate:
        tags.append("gust_switch_disruption")
    if strength(lambda c: c.has("retreat_lock") or c.has("item_lock") or c.has("ability_lock") or c.has("attack_prevention")) >= gate:
        tags.append("lock_control")
    if any(_strength(refs) >= gate for refs in profile.protection_plan.values()):
        tags.append("protection_support")
    if strength(lambda c: c.has("status_condition")) >= gate:
        tags.append("status_pressure")
    if strength(lambda c: c.has("hand_disruption") or c.has("deck_disruption")) >= gate:
        tags.append("hand_deck_disruption")
    if strength(lambda c: c.has("energy_acceleration")) >= gate:
        tags.append("energy_acceleration_engine")

    # Single-prize tempo: the deck attacks with non-ex Pokémon lines.
    core_attackers = [r for r in main_attackers if r.is_core]
    if core_attackers:
        by_id = {card.card_id: card for card, _ in resolved}
        if all(not by_id[r.card_id].is_ex for r in core_attackers):
            tags.append("single_prize_tempo")
    return tags


# --- confidence / review ------------------------------------------------------ #
#
# Confidence and review are judged against the *identity-defining* plan cards
# (attackers, finishers, bench/counter/lock/protection/status/gust), not every
# staple. A profile is uncertain when its central plan rests on v2-ambiguous
# parses; it is flagged for review only when an actually-ambiguous (v2 low/medium,
# needs_manual_review) card is load-bearing, so the flag stays discriminating.


def _special_plan_refs(profile: DeckProfile) -> list[CardRef]:
    """Control / counter / protection cards — the hard-to-parse plan pieces."""
    refs: list[CardRef] = []
    refs += profile.damage_counter_plan + profile.ability_damage_counter_plan
    refs += profile.switch_gust_plan + profile.lock_control_plan + profile.status_plan
    for bucket in profile.bench_pressure.values():
        refs += bucket
    for bucket in profile.protection_plan.values():
        refs += bucket
    return refs


def _identity_refs(profile: DeckProfile) -> list[CardRef]:
    return profile.main_attackers + profile.finishers + profile.setup_attackers + _special_plan_refs(profile)


def _finalize_confidence_review(
    profile: DeckProfile,
    resolved: Resolved,
    resolved_copies: int,
    unresolved_copies: int,
    config: ProfileConfig,
) -> tuple[str, float, list[str]]:
    total = resolved_copies + unresolved_copies
    resolved_share = resolved_copies / total if total else 0.0
    by_id = {card.card_id: (card, count) for card, count in resolved}

    function_weight = sum(count for card, count in resolved if card.is_function_card)
    review_weight = sum(count for card, count in resolved if card.is_function_card and card.needs_manual_review)
    review_share = review_weight / function_weight if function_weight else 0.0

    identity_ids = {r.card_id for r in _identity_refs(profile)}
    lb_weight = sum(count for cid, (card, count) in by_id.items() if cid in identity_ids)
    lb_low = sum(count for cid, (card, count) in by_id.items() if cid in identity_ids and card.confidence == "low")
    lb_lowmed = sum(
        count for cid, (card, count) in by_id.items() if cid in identity_ids and card.confidence in ("low", "medium")
    )

    if resolved_share < 0.90 or (not profile.main_attackers and not profile.reason_tags):
        confidence = "low"
    elif lb_weight and lb_low / lb_weight > 0.10:
        confidence = "medium"
    elif lb_weight and lb_lowmed / lb_weight > 0.5:
        confidence = "medium"
    else:
        confidence = "high"

    # Flag review only when a *control/counter/protection* plan card (the tricky
    # parses) is load-bearing and genuinely ambiguous — not merely because a main
    # attacker also carries a medium secondary tag.
    special_ids = {r.card_id for r in _special_plan_refs(profile)}
    ambiguous = sorted(
        (
            (card, count)
            for cid, (card, count) in by_id.items()
            if cid in special_ids
            and count >= config.core_copy_threshold
            and card.needs_manual_review
            and card.confidence in ("low", "medium")
        ),
        key=lambda pair: (-pair[1], pair[0].name),
    )

    reasons: list[str] = []
    if unresolved_copies > 0:
        reasons.append(f"{unresolved_copies} card copies unresolved in v2")
    for card, count in ambiguous[:5]:
        reasons.append(f"plan rests on review-flagged '{card.name}' x{count} (v2 {card.confidence})")
    if not profile.main_attackers:
        reasons.append("no main attacker identified")
    non_shape = [t for t in profile.reason_tags if not t.endswith("_deck")]
    if len(non_shape) >= 4:
        reasons.append("multiple competing plans (audit primary identity)")
    return confidence, review_share, reasons


# --- top-level ---------------------------------------------------------------- #


def infer_profile(
    deck: DeckList,
    v2map: dict[str, V2Card],
    config: ProfileConfig | None = None,
    deck_id: str = "",
) -> DeckProfile:
    config = config or ProfileConfig()
    resolved, resolved_copies, unresolved_copies = _resolve(deck, v2map)

    by_id = {card.card_id: card for card, _ in resolved}
    attacker_refs = _collect(resolved, lambda c: c.is_pokemon and c.has("attacker"), "attacker", config)
    main_attackers = [r for r in attacker_refs if r.is_core]
    if not main_attackers and attacker_refs:
        main_attackers = attacker_refs[:1]  # fallback: identify the top attacker
    # Among the real lines (count>=2), surface the actual damage threats first;
    # the count>=2 gate already keeps one-of techs out of main_attackers.
    main_attackers.sort(key=lambda r: (-by_id[r.card_id].max_damage, -r.count, r.name))

    bench_pressure = {
        "ordinary_bench_attack_damage": _collect(resolved, lambda c: c.has("bench_attack_damage"), "bench_attack_damage", config),
        "bench_damage_counters": _collect(resolved, lambda c: c.has("bench_damage_counters"), "bench_damage_counters", config),
        "ability_damage_counters": _collect(resolved, lambda c: c.has("ability_damage_counters"), "ability_damage_counters", config),
    }
    bench_pressure = {k: v for k, v in bench_pressure.items() if v}

    profile = DeckProfile(
        deck_id=deck_id or deck.deck_hash,
        deck_hash=deck.deck_hash,
        archetype_label=deck.archetype,
        split=deck.split,
        card_count=deck.total_cards,
        profile_confidence="low",
        main_attackers=main_attackers,
        setup_attackers=_collect(resolved, lambda c: c.has("setup_attacker"), "setup_attacker", config),
        finishers=_collect(resolved, lambda c: c.is_pokemon and c.has("finisher"), "finisher", config),
        draw_engine=_collect(resolved, lambda c: c.has("draw"), "draw", config),
        search_engine=_collect(resolved, lambda c: c.has("search") or c.has("evolution_search"), "search", config),
        damage_counter_plan=_collect(
            resolved, lambda c: c.has("bench_damage_counters") or c.has("active_damage_counters"), "attack_damage_counters", config
        ),
        ability_damage_counter_plan=_collect(resolved, lambda c: c.has("ability_damage_counters"), "ability_damage_counters", config),
        bench_pressure=bench_pressure,
        switch_gust_plan=_collect(
            resolved, lambda c: c.has("gust_opponent") or c.has("switch_opponent") or c.has("switch_self"), "switch_gust", config
        ),
        lock_control_plan=_collect(
            resolved,
            lambda c: c.has("retreat_lock") or c.has("item_lock") or c.has("tool_lock") or c.has("ability_lock") or c.has("attack_prevention"),
            "lock_control",
            config,
        ),
        protection_plan=_protection_plan(resolved, config),
        healing_recovery_plan=_collect(
            resolved,
            lambda c: c.has("healing") or c.has("pokemon_recovery") or c.has("discard_recovery") or c.has("pokemon_pickup") or c.has("energy_recovery"),
            "healing_recovery",
            config,
        ),
        disruption_plan=_collect(resolved, lambda c: c.has("hand_disruption") or c.has("deck_disruption"), "disruption", config),
        status_plan=_collect(resolved, lambda c: c.has("status_condition"), "status", config),
    )

    profile.energy_plan = _energy_plan(resolved, config)
    profile.evolution_plan = _evolution_plan(resolved, config)
    profile.dependencies = _dependencies(resolved, config)
    profile.choke_points = _choke_points(profile.dependencies, config)
    profile.reason_tags = _reason_tags(profile, resolved, main_attackers, config)
    confidence, review_share, review_reasons = _finalize_confidence_review(
        profile, resolved, resolved_copies, unresolved_copies, config
    )
    profile.profile_confidence = confidence
    profile.flagged_review_share = review_share
    profile.manual_review_reasons = review_reasons
    profile.resolved_card_share = resolved_copies / (resolved_copies + unresolved_copies) if (resolved_copies + unresolved_copies) else 0.0
    return profile


def _energy_plan(resolved: Resolved, config: ProfileConfig) -> dict:
    basic_types = sorted(
        {card.provided_energy for card, _ in resolved if card.subtype == SUBTYPE_BASIC_ENERGY and card.provided_energy}
    )
    return {
        "basic_energy_count": sum(count for card, count in resolved if card.subtype == SUBTYPE_BASIC_ENERGY),
        "basic_energy_types": basic_types,
        "special_energy": _collect(resolved, lambda c: c.subtype == SUBTYPE_SPECIAL_ENERGY, "special_energy", config),
        "acceleration": _collect(resolved, lambda c: c.has("energy_acceleration"), "energy_acceleration", config),
        "recovery": _collect(resolved, lambda c: c.has("energy_recovery"), "energy_recovery", config),
        "reposition": _collect(resolved, lambda c: c.has("energy_reposition"), "energy_reposition", config),
    }


def _evolution_plan(resolved: Resolved, config: ProfileConfig) -> dict:
    stage2 = _stage_copies(resolved, STAGE_2)
    stage1 = _stage_copies(resolved, STAGE_1)
    basic = _stage_copies(resolved, STAGE_BASIC)
    max_stage = "stage2" if stage2 else "stage1" if stage1 else "basic"
    return {
        "basic_pokemon_count": basic,
        "stage1_pokemon_count": stage1,
        "stage2_pokemon_count": stage2,
        "max_stage": max_stage,
        "rare_candy": _collect(resolved, lambda c: c.has("evolution_skip"), "evolution_skip", config),
        "evolution_search": _collect(resolved, lambda c: c.has("evolution_search"), "evolution_search", config),
    }


def infer_all(
    decks: list[DeckList], v2map: dict[str, V2Card], config: ProfileConfig | None = None
) -> list[DeckProfile]:
    config = config or ProfileConfig()
    ordered = sorted(decks, key=lambda d: d.deck_hash)
    return [infer_profile(deck, v2map, config, deck_id=f"deck_{i:04d}") for i, deck in enumerate(ordered)]
