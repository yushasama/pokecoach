"""Schemas and tag vocabulary for CardFunctionClassifier v2.

Offline/training-only. This package parses ``EN_Card_Data.csv`` (the 2,022-row,
1,267-card-id reference pool) into card *function tags*. It is the first stage of
the intended pipeline:

    card text -> function tags -> deck profile -> matchup threats
                 -> action scores -> reason tags

This module only owns the first arrow (``card text -> function tags``). It must
never be imported by the submission runtime (``main.py``) or the battle engine
(``cg``): it reads a data file that does not exist at evaluation time and would
violate INVARIANT #7 (no external I/O during evaluation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

ARTIFACT_VERSION = "deepmew.card_functions.v2"

# --- Card types (derived from the Stage/Type column) --------------------------

CARD_TYPE_POKEMON = "pokemon"
CARD_TYPE_TRAINER = "trainer"
CARD_TYPE_ENERGY = "energy"

# Trainer / energy sub-types.
SUBTYPE_ITEM = "item"
SUBTYPE_SUPPORTER = "supporter"
SUBTYPE_STADIUM = "stadium"
SUBTYPE_TOOL = "pokemon_tool"
SUBTYPE_BASIC_ENERGY = "basic_energy"
SUBTYPE_SPECIAL_ENERGY = "special_energy"

# Move "kinds" (sections) a card's text can be split into.
SECTION_ABILITY = "ability"
SECTION_ATTACK = "attack"
SECTION_TERA = "tera"
SECTION_ITEM = "item"
SECTION_SUPPORTER = "supporter"
SECTION_STADIUM = "stadium"
SECTION_TOOL = "pokemon_tool"
SECTION_ENERGY = "energy"

# --- Confidence ordering ------------------------------------------------------

CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"
_CONFIDENCE_RANK = {CONFIDENCE_HIGH: 2, CONFIDENCE_MEDIUM: 1, CONFIDENCE_LOW: 0}


def weakest_confidence(values: Iterable[str]) -> str:
    """Return the lowest-confidence value (``low`` < ``medium`` < ``high``)."""
    ranked = [v for v in values if v in _CONFIDENCE_RANK]
    if not ranked:
        return CONFIDENCE_HIGH
    return min(ranked, key=lambda v: _CONFIDENCE_RANK[v])


# --- Function tags ------------------------------------------------------------
#
# The canonical tag vocabulary. The ordering here is also the stable display
# order used everywhere tags are serialized.

TAG_ORDER: tuple[str, ...] = (
    "attacker",
    "setup_attacker",
    "finisher",
    "draw",
    "search",
    "gust_opponent",
    "switch_self",
    "switch_opponent",
    "retreat_lock",
    "item_lock",
    "tool_lock",
    "ability_lock",
    "attack_prevention",
    "effect_prevention",
    "bench_attack_damage",
    "bench_damage_counters",
    "active_damage_counters",
    "ability_damage_counters",
    "damage_modifier",
    "damage_reduction",
    "healing",
    "pokemon_recovery",
    "discard_recovery",
    "pokemon_pickup",
    "energy_acceleration",
    "energy_recovery",
    "energy_denial",
    "energy_reposition",
    "special_energy_dependency",
    "evolution_search",
    "evolution_skip",
    "evolution_enabler",
    "wall_tank",
    "bench_protection",
    "self_bench_protection",
    "hand_disruption",
    "deck_disruption",
    "hand_size_scaling",
    "prize_count_scaling",
    "discard_scaling",
    "stadium_dependency",
    "tool_dependency",
    "ability_engine",
    "status_condition",
    "poison",
    "burn",
    "paralysis",
    "confusion",
    "sleep",
    "coin_flip_variance",
    "self_damage",
    "self_discard",
    "self_ko",
    "vanilla_or_low_signal",
    "needs_manual_review",
)
TAG_SET = frozenset(TAG_ORDER)
_TAG_INDEX = {tag: i for i, tag in enumerate(TAG_ORDER)}


def order_tags(tags: Iterable[str]) -> list[str]:
    """Return unique tags in canonical :data:`TAG_ORDER`."""
    unique = {t for t in tags if t in TAG_SET}
    return sorted(unique, key=lambda t: _TAG_INDEX[t])


# Semantic "risk classes" that flag a card for manual review and bump its
# manual-audit priority. Each maps to the tags that belong to it.
RISK_CLASSES: dict[str, frozenset[str]] = {
    "prevention": frozenset(
        {
            "attack_prevention",
            "effect_prevention",
            "damage_reduction",
            "wall_tank",
            "bench_protection",
            "self_bench_protection",
        }
    ),
    "damage_counters": frozenset(
        {"bench_damage_counters", "active_damage_counters", "ability_damage_counters"}
    ),
    "bench": frozenset(
        {
            "bench_attack_damage",
            "bench_damage_counters",
            "bench_protection",
            "self_bench_protection",
        }
    ),
    "switch_gust": frozenset({"gust_opponent", "switch_opponent", "switch_self"}),
    "lock": frozenset({"retreat_lock", "item_lock", "tool_lock", "ability_lock"}),
    "ability_prevention": frozenset({"ability_lock"}),
    "energy_movement": frozenset(
        {"energy_acceleration", "energy_recovery", "energy_denial", "energy_reposition"}
    ),
    "status": frozenset(
        {"status_condition", "poison", "burn", "paralysis", "confusion", "sleep"}
    ),
    "self_ko": frozenset({"self_ko"}),
}


@dataclass(frozen=True)
class MoveText:
    """One parsed unit of a Pokémon card: an Ability, attack, or Tera/rule box."""

    kind: str  # SECTION_ABILITY | SECTION_ATTACK | SECTION_TERA
    name: str
    cost: str
    damage: int | None
    effect: str


@dataclass(frozen=True)
class CardInput:
    """A single card (one ``card_id``), assembled from one or more CSV rows."""

    card_id: str
    name: str
    card_type: str
    stage_type: str  # raw Stage/Type value, e.g. "Stage 2 Pokémon", "Item"
    subtype: str | None  # item/supporter/stadium/pokemon_tool/basic/special energy
    rule: str | None  # "ACE SPEC" | "Pokémon ex" | "Mega Pokémon ex" | None
    hp: int | None
    provided_energy: str  # Type column for energy cards (e.g. "{W}")
    previous_stage: str
    moves: tuple[MoveText, ...] = ()  # Pokémon Abilities/attacks/Tera
    trainer_text: str = ""  # Item/Supporter/Stadium/Tool effect text
    energy_text: str = ""  # Energy effect text

    @property
    def ability_texts(self) -> list[MoveText]:
        return [m for m in self.moves if m.kind == SECTION_ABILITY]

    @property
    def attack_texts(self) -> list[MoveText]:
        return [m for m in self.moves if m.kind == SECTION_ATTACK]

    @property
    def tera_texts(self) -> list[MoveText]:
        return [m for m in self.moves if m.kind == SECTION_TERA]

    @property
    def move_names(self) -> str:
        return " | ".join(m.name for m in self.moves if m.name)

    @property
    def damage_values(self) -> str:
        return " | ".join(str(m.damage) for m in self.moves if m.damage is not None)

    @property
    def max_printed_damage(self) -> int:
        damages = [m.damage for m in self.moves if m.damage is not None]
        return max(damages) if damages else 0

    @property
    def full_text(self) -> str:
        parts: list[str] = []
        for m in self.moves:
            label = m.name or m.kind
            if m.effect:
                parts.append(f"{label}: {m.effect}")
        if self.trainer_text:
            parts.append(self.trainer_text)
        if self.energy_text:
            parts.append(self.energy_text)
        return "\n".join(parts)


@dataclass(frozen=True)
class ParseEvent:
    """A structured, inspectable observation produced before final tags.

    Each event explains *why* a tag was assigned: which section of card text
    matched, which rule fired, the resulting tag, a confidence, and a snippet.
    """

    section: str
    rule_id: str
    tag: str
    confidence: str
    snippet: str

    def as_compact(self) -> str:
        return f"{self.section}|{self.rule_id}|{self.tag}|{self.confidence}"


@dataclass(frozen=True)
class ClassifiedCard:
    """Final classification output for one card."""

    card_id: str
    name: str
    card_type: str
    stage: str
    hp: int | None
    tags: tuple[str, ...]
    parse_events: tuple[ParseEvent, ...]
    confidence: str
    needs_manual_review: bool
    review_reasons: tuple[str, ...]
    manual_audit_priority: int
    # Source text fields kept so a human can inspect why tags were assigned.
    rule: str | None = None
    subtype: str | None = None
    provided_energy: str = ""
    ability_text: str = ""
    attack_text: str = ""
    tera_text: str = ""
    trainer_text: str = ""
    energy_text: str = ""
    move_names: str = ""
    damage_values: str = ""
    full_text: str = ""

    def has_tag(self, tag: str) -> bool:
        return tag in self.tags


@dataclass
class ClassifierConfig:
    """Tunable thresholds for the classifier (kept explicit, no hardcoding)."""

    finisher_min_damage: int = 180
    high_frequency_decks: int = 6  # appears in >= N decks -> high-frequency hint
    output_csv_name: str = "all_cards_function_tags_v2.csv"
    report_name: str = "card_function_classifier_v2_report.md"


# Output column order for the all-cards CSV export.
CSV_COLUMNS: tuple[str, ...] = (
    "card_id",
    "name",
    "card_type",
    "subtype",
    "stage",
    "hp",
    "rule",
    "provided_energy",
    "tags",
    "confidence",
    "needs_manual_review",
    "review_reasons",
    "manual_audit_priority",
    "parse_events",
    "ability_text",
    "attack_text",
    "tera_text",
    "trainer_text",
    "energy_text",
    "move_names",
    "damage_values",
    "full_text",
)


@dataclass(frozen=True)
class PriorityHints:
    """Handoff-derived hints used only to prioritise manual audit order."""

    load_bearing: frozenset[str] = field(default_factory=frozenset)  # counter-matrix
    top_meta: frozenset[str] = field(default_factory=frozenset)  # any top-meta deck
    deck_frequency: dict[str, int] = field(default_factory=dict)  # name -> #decks

    def is_load_bearing(self, name: str) -> bool:
        return _norm_name(name) in self.load_bearing

    def is_top_meta(self, name: str) -> bool:
        return _norm_name(name) in self.top_meta

    def frequency(self, name: str) -> int:
        return self.deck_frequency.get(_norm_name(name), 0)


def _norm_name(name: str) -> str:
    return " ".join(str(name).replace("’", "'").lower().split())
