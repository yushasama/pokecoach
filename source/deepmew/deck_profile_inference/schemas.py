"""Schemas for DeckProfileInference v0 (``function tags -> deck profile``).

Offline/training-only. This package aggregates CardFunctionClassifier v2 tags over
mined 60-card decklists into generalized, count-weighted deck profiles. It is the
second arrow of the pipeline:

    card text -> function tags -> deck profile -> matchup threats
                 -> action scores -> reason tags

It owns only ``function tags -> deck profile``. It computes **no** action scores,
trains **no** labels, scripts **no** deck names, and claims **no** matchup win
rates. Archetype labels are carried through as metadata only and are never used as
inference logic. It must never be imported by the submission runtime (`main.py`)
or the battle engine (`cg`) (INVARIANT #7/#8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ARTIFACT_VERSION = "deepmew.deck_profile_inference.v0"
PROFILE_KIND = "deck_profile_inference"

# --- Card-type / stage constants (mirror the v2 stage_type values) ------------

STAGE_BASIC = "Basic Pokémon"
STAGE_1 = "Stage 1 Pokémon"
STAGE_2 = "Stage 2 Pokémon"
SUBTYPE_ITEM = "item"
SUBTYPE_SUPPORTER = "supporter"
SUBTYPE_STADIUM = "stadium"
SUBTYPE_TOOL = "pokemon_tool"
SUBTYPE_BASIC_ENERGY = "basic_energy"
SUBTYPE_SPECIAL_ENERGY = "special_energy"

RULE_EX = "Pokémon ex"
RULE_MEGA_EX = "Mega Pokémon ex"
_EX_RULES = frozenset({RULE_EX, RULE_MEGA_EX})

# Tags treated as "background" — they do not, by themselves, make a card a
# function-bearing card for confidence / review accounting.
BACKGROUND_TAGS = frozenset({"attacker", "finisher", "vanilla_or_low_signal", "needs_manual_review"})

# Generalized dependency / choke-point vocabulary (stable schema keys).
DEPENDENCY_KEYS: tuple[str, ...] = (
    "stadium_dependency",
    "tool_dependency",
    "special_energy_dependency",
    "stage2_dependency",
    "stage1_dependency",
    "ability_engine_dependency",
    "bench_fragility",
    "item_reliance",
    "hand_size_plan",
    "damage_counter_plan",
    "bench_attack_damage_plan",
    "self_ko_liability",
)


@dataclass(frozen=True)
class V2Card:
    """One row of the CardFunctionClassifier v2 CSV, parsed for aggregation."""

    card_id: str
    name: str
    card_type: str
    stage_type: str
    subtype: str | None
    rule: str | None
    hp: int | None
    provided_energy: str
    tags: frozenset[str]
    rule_ids: frozenset[str]
    confidence: str
    needs_manual_review: bool
    max_damage: int

    def has(self, tag: str) -> bool:
        return tag in self.tags

    @property
    def is_pokemon(self) -> bool:
        return self.card_type == "pokemon"

    @property
    def is_ex(self) -> bool:
        return self.rule in _EX_RULES

    @property
    def is_function_card(self) -> bool:
        return bool(self.tags - BACKGROUND_TAGS)


@dataclass(frozen=True)
class CardRef:
    """A card cited as evidence inside a profile plan, with its copy count."""

    card_id: str
    name: str
    count: int
    is_core: bool  # count >= core_copy_threshold (a real line, not a one-of tech)
    note: str = ""  # which tag/role this reference supports

    def as_text(self) -> str:
        return f"{self.name} x{self.count}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "card_id": self.card_id,
            "name": self.name,
            "count": self.count,
            "is_core": self.is_core,
            "note": self.note,
        }


@dataclass
class DeckProfile:
    """A generalized, count-weighted profile inferred from one decklist."""

    deck_id: str
    deck_hash: str
    archetype_label: str  # carried through only; never an inference input
    split: str
    card_count: int
    profile_confidence: str

    main_attackers: list[CardRef] = field(default_factory=list)
    setup_attackers: list[CardRef] = field(default_factory=list)
    finishers: list[CardRef] = field(default_factory=list)
    draw_engine: list[CardRef] = field(default_factory=list)
    search_engine: list[CardRef] = field(default_factory=list)

    energy_plan: dict[str, Any] = field(default_factory=dict)
    evolution_plan: dict[str, Any] = field(default_factory=dict)
    # Bench-pressure is deliberately split three ways (the v1 conflation bug).
    bench_pressure: dict[str, list[CardRef]] = field(default_factory=dict)
    damage_counter_plan: list[CardRef] = field(default_factory=list)
    ability_damage_counter_plan: list[CardRef] = field(default_factory=list)

    switch_gust_plan: list[CardRef] = field(default_factory=list)
    lock_control_plan: list[CardRef] = field(default_factory=list)
    protection_plan: dict[str, list[CardRef]] = field(default_factory=dict)
    healing_recovery_plan: list[CardRef] = field(default_factory=list)
    disruption_plan: list[CardRef] = field(default_factory=list)
    status_plan: list[CardRef] = field(default_factory=list)

    dependencies: dict[str, int] = field(default_factory=dict)
    choke_points: list[str] = field(default_factory=list)
    manual_review_reasons: list[str] = field(default_factory=list)
    reason_tags: list[str] = field(default_factory=list)

    resolved_card_share: float = 0.0
    flagged_review_share: float = 0.0

    def to_json_dict(self) -> dict[str, Any]:
        def refs(items: list[CardRef]) -> list[dict[str, Any]]:
            return [r.to_dict() for r in items]

        return {
            "deck_id": self.deck_id,
            "deck_hash": self.deck_hash,
            "archetype_label": self.archetype_label,
            "split": self.split,
            "card_count": self.card_count,
            "profile_confidence": self.profile_confidence,
            "main_attackers": refs(self.main_attackers),
            "setup_attackers": refs(self.setup_attackers),
            "finishers": refs(self.finishers),
            "draw_engine": refs(self.draw_engine),
            "search_engine": refs(self.search_engine),
            "energy_plan": _plan_json(self.energy_plan),
            "evolution_plan": _plan_json(self.evolution_plan),
            "bench_pressure": {k: refs(v) for k, v in self.bench_pressure.items()},
            "damage_counter_plan": refs(self.damage_counter_plan),
            "ability_damage_counter_plan": refs(self.ability_damage_counter_plan),
            "switch_gust_plan": refs(self.switch_gust_plan),
            "lock_control_plan": refs(self.lock_control_plan),
            "protection_plan": {k: refs(v) for k, v in self.protection_plan.items()},
            "healing_recovery_plan": refs(self.healing_recovery_plan),
            "disruption_plan": refs(self.disruption_plan),
            "status_plan": refs(self.status_plan),
            "dependencies": self.dependencies,
            "choke_points": self.choke_points,
            "manual_review_reasons": self.manual_review_reasons,
            "reason_tags": self.reason_tags,
            "resolved_card_share": round(self.resolved_card_share, 4),
            "flagged_review_share": round(self.flagged_review_share, 4),
        }


def _plan_json(plan: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in plan.items():
        if isinstance(value, list) and value and isinstance(value[0], CardRef):
            out[key] = [r.to_dict() for r in value]
        else:
            out[key] = value
    return out


@dataclass
class ProfileConfig:
    """Tunable thresholds. Kept explicit — no hardcoded card names anywhere."""

    core_copy_threshold: int = 2  # count >= this = a real line, not a one-of tech
    min_reason_strength: int = 2  # weighted copies needed to emit a primary reason tag
    high_item_reliance: int = 12  # item copies above which item lock is a real choke
    bench_fragility_hp_max: int = 90  # support Pokémon at/below this HP count as fragile
    low_confidence_share_medium: float = 0.15  # share of low-conf function weight -> medium
    output_csv_name: str = "deck_profiles_v0.csv"
    output_jsonl_name: str = "deck_profiles_v0.jsonl"
    report_name: str = "deck_profile_inference_v0_report.md"


@dataclass(frozen=True)
class DeckList:
    """A full mined decklist: ``card_id -> copies`` plus carry-through metadata."""

    deck_hash: str
    archetype: str
    split: str
    card_counts: dict[str, int]
    is_canonical: bool = True

    @property
    def total_cards(self) -> int:
        return sum(self.card_counts.values())
