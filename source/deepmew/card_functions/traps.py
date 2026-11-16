"""Canonical semantic-trap cases shared by the report and the regression tests.

Each case encodes the *intended* classification of a card that v1 got wrong (or
that is easy to get wrong). Cases are located by name plus optional disambiguating
text so the right variant is picked (e.g. the hand-size Alakazam, not the
energy-scaling one). Keeping these here means the report and tests cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deepmew.card_functions.schemas import ClassifiedCard, _norm_name


@dataclass(frozen=True)
class TrapCase:
    key: str
    name_contains: str
    must_include: tuple[str, ...] = ()
    must_exclude: tuple[str, ...] = ()
    text_contains: str | None = None
    require_review: bool = False
    # tag -> required source section of at least one parse event producing it.
    require_source_section: dict[str, str] = field(default_factory=dict)


TRAP_CASES: tuple[TrapCase, ...] = (
    TrapCase(
        key="boss_orders",
        name_contains="Boss's Orders",
        must_include=("gust_opponent",),
        must_exclude=("bench_attack_damage", "bench_damage_counters"),
    ),
    TrapCase(
        key="rare_candy",
        name_contains="Rare Candy",
        must_include=("evolution_skip", "evolution_enabler"),
        must_exclude=("item_lock", "attack_prevention", "effect_prevention"),
    ),
    TrapCase(
        key="scoop_up_cyclone",
        name_contains="Scoop Up Cyclone",
        must_include=("pokemon_pickup", "pokemon_recovery"),
        must_exclude=("search",),
    ),
    TrapCase(
        key="energy_switch",
        name_contains="Energy Switch",
        must_include=("energy_reposition",),
        must_exclude=("energy_acceleration",),
    ),
    TrapCase(
        key="rabsca",
        name_contains="Rabsca",
        must_include=("bench_protection",),
        must_exclude=("finisher",),
        require_review=True,
    ),
    TrapCase(
        key="dragapult_ex",
        name_contains="Dragapult ex",
        must_include=("bench_damage_counters", "self_bench_protection"),
        must_exclude=("wall_tank",),
    ),
    TrapCase(
        key="battle_cage",
        name_contains="Battle Cage",
        must_include=("effect_prevention", "bench_protection"),
        must_exclude=("bench_attack_damage", "attack_prevention"),
    ),
    TrapCase(
        key="mega_starmie_ex",
        name_contains="Mega Starmie ex",
        must_include=("bench_attack_damage",),
        must_exclude=("bench_damage_counters",),
    ),
    TrapCase(
        key="dusknoir",
        name_contains="Dusknoir",
        must_include=("ability_damage_counters", "self_ko"),
        require_source_section={"ability_damage_counters": "ability", "self_ko": "ability"},
    ),
    TrapCase(
        key="dusclops",
        name_contains="Dusclops",
        must_include=("ability_damage_counters", "self_ko"),
        require_source_section={"ability_damage_counters": "ability", "self_ko": "ability"},
    ),
    TrapCase(
        key="alakazam_hand_size",
        name_contains="Alakazam",
        text_contains="for each card in your hand",
        must_include=("hand_size_scaling", "active_damage_counters"),
        require_source_section={"hand_size_scaling": "attack", "active_damage_counters": "attack"},
    ),
    TrapCase(
        key="hops_trevenant",
        name_contains="Trevenant",
        must_include=("retreat_lock", "damage_modifier", "attacker"),
        require_review=True,
    ),
)


def find_trap_card(cards: list[ClassifiedCard], case: TrapCase) -> ClassifiedCard | None:
    """Locate the classified card for ``case`` by name (+ optional text)."""
    name_needle = _norm_name(case.name_contains)
    text_needle = case.text_contains.lower() if case.text_contains else None
    for card in cards:
        if name_needle not in _norm_name(card.name):
            continue
        if text_needle is not None and text_needle not in card.full_text.lower():
            continue
        return card
    return None


def evaluate_case(card: ClassifiedCard | None, case: TrapCase) -> dict[str, object]:
    """Return a pass/fail record for ``case`` against ``card``."""
    failures: list[str] = []
    if card is None:
        return {"key": case.key, "card_id": None, "name": None, "passed": False, "failures": ["card not found"]}

    tags = set(card.tags)
    for tag in case.must_include:
        if tag not in tags:
            failures.append(f"missing {tag}")
    for tag in case.must_exclude:
        if tag in tags:
            failures.append(f"unexpected {tag}")
    if case.require_review and not card.needs_manual_review:
        failures.append("expected needs_manual_review")
    for tag, section in case.require_source_section.items():
        sources = {e.section for e in card.parse_events if e.tag == tag}
        if section not in sources:
            failures.append(f"{tag} not sourced from {section} (got {sorted(sources)})")

    return {
        "key": case.key,
        "card_id": card.card_id,
        "name": card.name,
        "tags": list(card.tags),
        "passed": not failures,
        "failures": failures,
    }


def evaluate_all(cards: list[ClassifiedCard]) -> list[dict[str, object]]:
    return [evaluate_case(find_trap_card(cards, case), case) for case in TRAP_CASES]
