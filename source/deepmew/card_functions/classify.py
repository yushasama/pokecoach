"""Finalize parse events into a :class:`ClassifiedCard`.

Pipeline per card: ``scan_card`` (events) -> derive ordered tags ->
``vanilla_or_low_signal`` fallback -> confidence (weakest event) ->
``needs_manual_review`` + ``review_reasons`` -> ``manual_audit_priority``.

Output is deterministic: tags follow :data:`TAG_ORDER`, events keep scan order,
and ``classify_all`` sorts cards by numeric ``card_id``.
"""

from __future__ import annotations

from deepmew.card_functions.rules import scan_card
from deepmew.card_functions.schemas import (
    CARD_TYPE_POKEMON,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    CardInput,
    ClassifiedCard,
    ClassifierConfig,
    ParseEvent,
    PriorityHints,
    RISK_CLASSES,
    SECTION_ABILITY,
    SECTION_ATTACK,
    SECTION_TERA,
    order_tags,
    weakest_confidence,
)

# Tags that, on their own, do not make a card "meaningful" for the
# vanilla/low-signal fallback.
_BACKGROUND_TAGS = frozenset(
    {"attacker", "finisher", "vanilla_or_low_signal", "needs_manual_review"}
)


def _source_texts(card: CardInput) -> dict[str, str]:
    ability = " | ".join(f"{m.name}: {m.effect}".strip(": ") for m in card.ability_texts)
    attack = " | ".join(
        f"{m.name} [{m.cost}] {'' if m.damage is None else m.damage}: {m.effect}".strip(": ")
        for m in card.attack_texts
    )
    tera = " | ".join(m.effect for m in card.tera_texts)
    return {
        "ability_text": ability,
        "attack_text": attack,
        "tera_text": tera,
        "trainer_text": card.trainer_text,
        "energy_text": card.energy_text,
        "move_names": card.move_names,
        "damage_values": card.damage_values,
        "full_text": card.full_text,
    }


def _present_risk_classes(tags: set[str]) -> list[str]:
    return [name for name, tagset in RISK_CLASSES.items() if tags & tagset]


def _review(
    card: CardInput, tags: set[str], events: list[ParseEvent], confidence: str, hints: PriorityHints
) -> list[str]:
    reasons: list[str] = list(_present_risk_classes(tags))
    if any(e.confidence == CONFIDENCE_LOW for e in events) or confidence == CONFIDENCE_LOW:
        reasons.append("broad_fallback_rule")
    if card.card_type == CARD_TYPE_POKEMON and card.ability_texts and card.attack_texts:
        reasons.append("ability_and_attack")
    # Being a meta staple only warrants review if the card actually has a function
    # worth auditing; basic Energy / plain vanilla beatsticks stay clean low-signal.
    if hints.is_top_meta(card.name) and "vanilla_or_low_signal" not in tags:
        reasons.append("top_meta_core")
    # Stable, de-duplicated order.
    seen: dict[str, None] = {}
    for reason in reasons:
        seen.setdefault(reason, None)
    return list(seen)


def _audit_priority(
    card: CardInput, tags: set[str], confidence: str, reasons: list[str], hints: PriorityHints
) -> int:
    priority = 2 * len(_present_risk_classes(tags))
    if hints.is_load_bearing(card.name):
        priority += 5
    elif hints.is_top_meta(card.name):
        priority += 3
    frequency = hints.frequency(card.name)
    if frequency >= 6:
        priority += 3
    elif frequency > 0:
        priority += 1
    if confidence == CONFIDENCE_LOW:
        priority += 3
    elif confidence == CONFIDENCE_MEDIUM:
        priority += 1
    if "ability_and_attack" in reasons:
        priority += 2
    if "self_ko" in tags:
        priority += 2
    return priority


def classify_card(
    card: CardInput,
    hints: PriorityHints | None = None,
    config: ClassifierConfig | None = None,
) -> ClassifiedCard:
    hints = hints or PriorityHints()
    config = config or ClassifierConfig()

    events = scan_card(card)
    tag_set: set[str] = {e.tag for e in events}

    meaningful = tag_set - _BACKGROUND_TAGS
    if not meaningful:
        tag_set.add("vanilla_or_low_signal")
        events.append(
            ParseEvent("card", "no_special_function", "vanilla_or_low_signal", "high", "no notable effect text")
        )

    confidence = weakest_confidence(e.confidence for e in events)
    reasons = _review(card, tag_set, events, confidence, hints)
    needs_review = bool(reasons)
    if needs_review:
        tag_set.add("needs_manual_review")
    priority = _audit_priority(card, tag_set, confidence, reasons, hints)

    sources = _source_texts(card)
    return ClassifiedCard(
        card_id=card.card_id,
        name=card.name,
        card_type=card.card_type,
        stage=card.stage_type,
        hp=card.hp,
        tags=tuple(order_tags(tag_set)),
        parse_events=tuple(events),
        confidence=confidence,
        needs_manual_review=needs_review,
        review_reasons=tuple(reasons),
        manual_audit_priority=priority,
        rule=card.rule,
        subtype=card.subtype,
        provided_energy=card.provided_energy,
        **sources,
    )


def _card_sort_key(card: CardInput) -> tuple[int, str]:
    try:
        return (int(card.card_id), "")
    except (TypeError, ValueError):
        return (1 << 31, str(card.card_id))


def classify_all(
    cards: list[CardInput],
    hints: PriorityHints | None = None,
    config: ClassifierConfig | None = None,
) -> list[ClassifiedCard]:
    ordered = sorted(cards, key=_card_sort_key)
    return [classify_card(card, hints, config) for card in ordered]
