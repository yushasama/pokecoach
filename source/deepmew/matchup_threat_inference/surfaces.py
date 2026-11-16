"""Derive a deck's threat surface and coverage surface from card-function evidence.

A *threat surface* is what a deck presents (offensive pressure) and exposes
(structural liabilities); a *coverage surface* is what it can answer with. Both are
derived from the resolved ``(V2Card, copies)`` list (so every v2 tag/rule_id is in
reach) plus the already-inferred DeckProfile (for count-weighted dependencies). No
archetype/deck name is ever read. Copy counts drive strength; one-of techs are
recorded but flagged ``reliable=False`` so they cannot masquerade as full coverage.
"""

from __future__ import annotations

from typing import Callable

from deepmew.deck_profile_inference.schemas import (
    CardRef,
    DeckList,
    DeckProfile,
    SUBTYPE_ITEM,
    SUBTYPE_STADIUM,
    SUBTYPE_SUPPORTER,
    SUBTYPE_TOOL,
    V2Card,
)
from deepmew.matchup_threat_inference.schemas import (
    CoverageItem,
    MatchupConfig,
    OFFENSIVE_THREATS,
    STRUCTURAL_THREATS,
    ThreatItem,
    weakest,
)

Resolved = list[tuple[V2Card, int]]
Predicate = Callable[[V2Card], bool]


def resolve(deck: DeckList, v2map: dict[str, V2Card]) -> Resolved:
    out: Resolved = []
    for card_id, count in deck.card_counts.items():
        card = v2map.get(str(card_id))
        if card is not None:
            out.append((card, count))
    return out


def _refs(resolved: Resolved, pred: Predicate, note: str, config: MatchupConfig) -> list[CardRef]:
    by_id = {card.card_id: card for card, _ in resolved}
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
    refs.sort(key=lambda r: (-r.count, -by_id[r.card_id].max_damage, r.name))
    return refs


def _strength(refs: list[CardRef]) -> int:
    return sum(r.count for r in refs)


def _has_rule(card: V2Card, *rule_ids: str) -> bool:
    return any(rid in card.rule_ids for rid in rule_ids)


def _confidence_and_review(
    resolved: Resolved, refs: list[CardRef], config: MatchupConfig
) -> tuple[str, bool, list[str]]:
    """Confidence = weakest v2 confidence among load-bearing evidence; flag review
    when a load-bearing card is review-flagged and genuinely ambiguous (low/medium)."""
    by_id = {card.card_id: card for card, _ in resolved}
    # Load-bearing = the reliable (core) copies if any, else all cited cards.
    core = [r for r in refs if r.is_core]
    load_bearing = core or refs
    confidence = "high"
    review_flagged = False
    reasons: list[str] = []
    for ref in load_bearing:
        card = by_id.get(ref.card_id)
        if card is None:
            continue
        confidence = weakest(confidence, card.confidence)
        if card.needs_manual_review and card.confidence in ("low", "medium"):
            review_flagged = True
            reasons.append(f"evidence '{card.name}' x{ref.count} review-flagged (v2 {card.confidence})")
    return confidence, review_flagged, reasons[:3]


# --- offensive threat predicates ---------------------------------------------- #

_OFFENSIVE_PREDICATES: dict[str, Predicate] = {
    "bench_attack_damage_threat": lambda c: c.has("bench_attack_damage"),
    "bench_damage_counter_threat": lambda c: c.has("bench_damage_counters"),
    "ability_damage_counter_threat": lambda c: c.has("ability_damage_counters"),
    "active_damage_counter_threat": lambda c: c.has("active_damage_counters"),
    "hand_size_scaling_threat": lambda c: c.has("hand_size_scaling"),
    "ability_engine_threat": lambda c: c.has("ability_engine"),
    "lock_control_threat": lambda c: (
        c.has("retreat_lock") or c.has("item_lock") or c.has("tool_lock")
        or c.has("ability_lock") or c.has("attack_prevention")
    ),
    "gust_switch_threat": lambda c: c.has("gust_opponent") or c.has("switch_opponent"),
    "status_threat": lambda c: (
        c.has("status_condition") or c.has("poison") or c.has("burn")
        or c.has("paralysis") or c.has("confusion") or c.has("sleep")
    ),
}


def _build_offensive(resolved: Resolved, config: MatchupConfig) -> list[ThreatItem]:
    items: list[ThreatItem] = []
    for category in OFFENSIVE_THREATS:
        refs = _refs(resolved, _OFFENSIVE_PREDICATES[category], category, config)
        if not refs:
            continue
        strength = _strength(refs)
        confidence, flagged, reasons = _confidence_and_review(resolved, refs, config)
        items.append(
            ThreatItem(
                category=category,
                kind="offensive",
                strength=strength,
                is_primary=strength >= config.min_threat_strength,
                confidence=confidence,
                evidence=refs,
                review_flagged=flagged,
                review_reasons=reasons,
            )
        )
    return items


# --- structural / liability threats (count-weighted dependencies) ------------- #


def _structural_item(
    category: str,
    resolved: Resolved,
    pred: Predicate,
    dep_copies: int,
    threshold: int,
    config: MatchupConfig,
) -> ThreatItem | None:
    if dep_copies < threshold:
        return None
    refs = _refs(resolved, pred, category, config)
    confidence, flagged, reasons = _confidence_and_review(resolved, refs, config)
    return ThreatItem(
        category=category,
        kind="structural",
        strength=dep_copies,
        is_primary=dep_copies >= threshold,
        confidence=confidence,
        evidence=refs[:8],
        review_flagged=flagged,
        review_reasons=reasons,
    )


def _build_structural(resolved: Resolved, profile: DeckProfile, config: MatchupConfig) -> list[ThreatItem]:
    deps = profile.dependencies
    candidates = [
        _structural_item(
            "tool_dependency_threat", resolved,
            lambda c: c.subtype == SUBTYPE_TOOL,
            deps.get("tool_dependency", 0), config.tool_dependency_min, config,
        ),
        _structural_item(
            "stadium_dependency_threat", resolved,
            lambda c: c.subtype == SUBTYPE_STADIUM,
            deps.get("stadium_dependency", 0), config.stadium_dependency_min, config,
        ),
        _structural_item(
            "special_energy_dependency_threat", resolved,
            lambda c: c.subtype == "special_energy",
            deps.get("special_energy_dependency", 0), config.special_energy_min, config,
        ),
        _structural_item(
            "self_ko_prize_liability", resolved,
            lambda c: c.has("self_ko"),
            deps.get("self_ko_liability", 0), config.self_ko_min, config,
        ),
        _structural_item(
            "fragile_bench_liability", resolved,
            lambda c: c.is_pokemon and c.is_function_card and not c.has("attacker")
            and (c.hp is None or c.hp <= 90),
            deps.get("bench_fragility", 0), config.bench_fragility_min, config,
        ),
    ]
    return [c for c in candidates if c is not None]


def build_threat_surface(
    resolved: Resolved, profile: DeckProfile, config: MatchupConfig
) -> tuple[list[ThreatItem], list[ThreatItem]]:
    presents = sorted(
        _build_offensive(resolved, config),
        key=lambda t: (-t.strength, t.category),
    )
    exposes = sorted(
        _build_structural(resolved, profile, config),
        key=lambda t: (-t.strength, t.category),
    )
    return presents, exposes


# --- coverage derivation ------------------------------------------------------ #
#
# Each coverage category lists a predicate over v2 cards and whether it is
# ``tag_backed`` (a clean v2 tag/rule_id) or ``heuristic`` (no clean v2 tag, so the
# detection is best-effort and explicitly down-weighted — never silently trusted).

_COVERAGE_RULES: dict[str, tuple[Predicate, str]] = {
    # Tera-style own-bench damage immunity is the only clean "protects bench from
    # ordinary attack damage" signal. Counter-placement prevention is NOT here.
    "bench_attack_damage_protection": (lambda c: c.has("self_bench_protection"), "tag_backed"),
    # Battle Cage family: prevents damage *counters* from being placed.
    "damage_counter_protection": (
        lambda c: _has_rule(c, "prevents_counter_placement", "protects_bench_from_counters"),
        "tag_backed",
    ),
    # Rabsca family: prevents *attack effects* on the bench (not Ability effects).
    "attack_effect_protection": (
        lambda c: _has_rule(c, "protects_own_bench")
        or (c.has("bench_protection") and not _has_rule(c, "prevents_counter_placement", "protects_bench_from_counters")),
        "tag_backed",
    ),
    "ability_prevention": (lambda c: c.has("ability_lock"), "tag_backed"),
    "hand_disruption": (lambda c: c.has("hand_disruption") or c.has("deck_disruption"), "tag_backed"),
    # No clean v2 tool-removal tag (v2 mislabels Tool Scrapper as tool_dependency);
    # detect Item/Supporter cards that reference a Tool, flag heuristic.
    "tool_removal": (
        lambda c: c.subtype in (SUBTYPE_ITEM, SUBTYPE_SUPPORTER) and _has_rule(c, "references_tool"),
        "heuristic",
    ),
    # Playing any Stadium discards the opponent's Stadium — stadium "bump" capability.
    "stadium_bump": (lambda c: c.subtype == SUBTYPE_STADIUM, "heuristic"),
    "energy_denial": (lambda c: c.has("energy_denial") or _has_rule(c, "discards_opponent_energy"), "tag_backed"),
    "switching_outs": (lambda c: c.has("switch_self"), "tag_backed"),
    "healing_recovery": (
        lambda c: c.has("healing") or c.has("pokemon_recovery") or c.has("discard_recovery")
        or c.has("pokemon_pickup") or c.has("energy_recovery"),
        "tag_backed",
    ),
    "gust_pressure": (lambda c: c.has("gust_opponent"), "tag_backed"),
}


def build_coverage_surface(resolved: Resolved, config: MatchupConfig) -> list[CoverageItem]:
    items: list[CoverageItem] = []
    for category, (pred, source) in _COVERAGE_RULES.items():
        refs = _refs(resolved, pred, category, config)
        if not refs:
            continue
        strength = _strength(refs)
        reliable = strength >= config.core_copy_threshold
        confidence, flagged, reasons = _confidence_and_review(resolved, refs, config)
        if source == "heuristic":
            # Heuristic detection is never trusted at high confidence.
            confidence = weakest(confidence, "medium")
        items.append(
            CoverageItem(
                category=category,
                strength=strength,
                reliable=reliable,
                confidence=confidence,
                source=source,
                evidence=refs,
                review_flagged=flagged,
                review_reasons=reasons,
            )
        )
    items.sort(key=lambda c: (-c.strength, c.category))
    return items
