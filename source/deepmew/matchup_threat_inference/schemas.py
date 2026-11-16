"""Schemas + vocabulary for MatchupThreatInference v0 (``deck profile -> matchup threats``).

Offline/training-only. Stage 3 of:

    card text -> function tags -> deck profile -> matchup threats
                 -> action scores -> reason tags

This package owns only ``deck profile -> matchup threats``. A *matchup threat* is a
structured hypothesis — evidence (cited cards), a confidence, and reason tags — about
what a deck **presents** (offensive pressure), what it **exposes** (structural
liabilities), and what it can **answer** (coverage). It is NOT an action score and
NOT a win-rate claim. There are deliberately **no** win-rate / EV fields anywhere.

The semantic separations the v1 pipeline conflated are preserved exactly:
- ordinary bench *attack* damage  vs  attack-based bench *damage counters*
  vs  Ability-based damage counters (three distinct threats / answers);
- counter-placement *prevention* (Battle Cage) does NOT answer ordinary bench
  attack damage; attack-effect bench protection (Rabsca) does NOT answer
  Ability-placed counters; Ability *prevention* (Psyduck/Damp) DOES.

Archetype labels are carried through as display metadata only and are never read as
inference logic (proven by a test). This package must never be imported by the
submission runtime (``main.py``) or the engine (``cg``) — INVARIANT #7/#8.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from deepmew.deck_profile_inference.schemas import CardRef

ARTIFACT_VERSION = "deepmew.matchup_threat_inference.v0"
INFERENCE_KIND = "matchup_threat_inference"

# --- Threat categories --------------------------------------------------------
# Offensive threats: pressure a deck applies to an opponent.
OFFENSIVE_THREATS: tuple[str, ...] = (
    "bench_attack_damage_threat",
    "bench_damage_counter_threat",
    "ability_damage_counter_threat",
    "active_damage_counter_threat",
    "hand_size_scaling_threat",
    "ability_engine_threat",
    "lock_control_threat",
    "gust_switch_threat",
    "status_threat",
)
# Structural / liability threats: the deck's own exploitable dependencies.
STRUCTURAL_THREATS: tuple[str, ...] = (
    "tool_dependency_threat",
    "stadium_dependency_threat",
    "special_energy_dependency_threat",
    "self_ko_prize_liability",
    "fragile_bench_liability",
)
THREAT_CATEGORIES: tuple[str, ...] = OFFENSIVE_THREATS + STRUCTURAL_THREATS
_THREAT_INDEX = {t: i for i, t in enumerate(THREAT_CATEGORIES)}

# --- Coverage categories: answers a deck carries ------------------------------
COVERAGE_CATEGORIES: tuple[str, ...] = (
    "bench_attack_damage_protection",
    "damage_counter_protection",
    "attack_effect_protection",
    "ability_prevention",
    "hand_disruption",
    "tool_removal",
    "stadium_bump",
    "energy_denial",
    "switching_outs",
    "healing_recovery",
    "gust_pressure",
)
_COVERAGE_INDEX = {c: i for i, c in enumerate(COVERAGE_CATEGORIES)}

# --- Coverage-gap categories: missing/insufficient answers given a threat -----
COVERAGE_GAP_CATEGORIES: tuple[str, ...] = (
    "no_bench_attack_damage_answer",
    "no_damage_counter_answer",
    "no_ability_counter_answer",
    "battle_cage_does_not_answer_ordinary_bench_damage",
    "rabsca_does_not_answer_ability_counters",
    "no_hand_size_answer",
    "no_tool_dependency_answer",
    "no_stadium_dependency_answer",
    "no_special_energy_answer",
    "fragile_bench_into_spread",
    "setup_deck_into_lock",
)

# --- Answer matrix: which coverage answers which threat -----------------------
# Each threat maps to coverage that FULLY answers it, coverage that PARTIALLY
# mitigates it, and the generic coverage-gap tag emitted when no full answer
# exists. This is the load-bearing semantic contract; it is intentionally
# explicit (no card names, no archetype names) so it is auditable and testable.
#
# Note the deliberate exclusions that encode the v1 fixes:
#  * bench_attack_damage_threat is NOT answered by damage_counter_protection
#    (Battle Cage) nor by attack_effect_protection (Rabsca);
#  * ability_damage_counter_threat is NOT answered by attack_effect_protection
#    (Rabsca) — only ability_prevention is a full answer (damage_counter_
#    protection is partial; it stops the counters landing regardless of source).


@dataclass(frozen=True)
class AnswerRule:
    full: frozenset[str]
    partial: frozenset[str]
    gap: str | None  # the coverage-gap tag when there is no full answer


def _rule(full: tuple[str, ...], partial: tuple[str, ...], gap: str | None) -> AnswerRule:
    return AnswerRule(full=frozenset(full), partial=frozenset(partial), gap=gap)


ANSWER_MATRIX: dict[str, AnswerRule] = {
    "bench_attack_damage_threat": _rule(
        ("bench_attack_damage_protection",),
        ("healing_recovery", "gust_pressure"),
        "no_bench_attack_damage_answer",
    ),
    "bench_damage_counter_threat": _rule(
        ("damage_counter_protection", "attack_effect_protection"),
        ("healing_recovery", "bench_attack_damage_protection"),
        "no_damage_counter_answer",
    ),
    "ability_damage_counter_threat": _rule(
        ("ability_prevention",),
        ("damage_counter_protection", "healing_recovery"),
        "no_ability_counter_answer",
    ),
    "active_damage_counter_threat": _rule(
        ("healing_recovery", "ability_prevention"),
        ("switching_outs",),
        None,
    ),
    "hand_size_scaling_threat": _rule(
        ("hand_disruption",),
        (),
        "no_hand_size_answer",
    ),
    "ability_engine_threat": _rule(
        ("ability_prevention",),
        (),
        None,
    ),
    "lock_control_threat": _rule(
        ("switching_outs",),
        ("gust_pressure", "healing_recovery"),
        None,
    ),
    "gust_switch_threat": _rule(
        (),
        ("switching_outs",),
        None,
    ),
    "status_threat": _rule(
        ("switching_outs", "healing_recovery"),
        (),
        None,
    ),
    "tool_dependency_threat": _rule(
        ("tool_removal",),
        (),
        "no_tool_dependency_answer",
    ),
    "stadium_dependency_threat": _rule(
        ("stadium_bump",),
        (),
        "no_stadium_dependency_answer",
    ),
    "special_energy_dependency_threat": _rule(
        ("energy_denial",),
        (),
        "no_special_energy_answer",
    ),
    # The opponent's own liabilities (not "answered" by the subject; they are
    # opportunities handled via the compound fragile_bench_into_spread gap).
    "self_ko_prize_liability": _rule((), (), None),
    "fragile_bench_liability": _rule((), (), None),
}


def threat_sort_key(category: str) -> int:
    return _THREAT_INDEX.get(category, len(THREAT_CATEGORIES))


def coverage_sort_key(category: str) -> int:
    return _COVERAGE_INDEX.get(category, len(COVERAGE_CATEGORIES))


# --- Confidence ---------------------------------------------------------------
_CONFIDENCE_RANK = {"high": 2, "medium": 1, "low": 0}
_RANK_CONFIDENCE = {2: "high", 1: "medium", 0: "low"}


def weakest(*values: str) -> str:
    ranked = [v for v in values if v in _CONFIDENCE_RANK]
    if not ranked:
        return "high"
    return min(ranked, key=lambda v: _CONFIDENCE_RANK[v])


def lower_one(confidence: str) -> str:
    return _RANK_CONFIDENCE[max(0, _CONFIDENCE_RANK.get(confidence, 2) - 1)]


# --- Data model ---------------------------------------------------------------


@dataclass
class ThreatItem:
    """One threat a deck presents/exposes, with its card-function evidence."""

    category: str
    kind: str  # "offensive" | "structural"
    strength: int  # weighted copies of supporting cards
    is_primary: bool  # strength clears the primary threshold
    confidence: str
    evidence: list[CardRef] = field(default_factory=list)
    review_flagged: bool = False
    review_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "kind": self.kind,
            "strength": self.strength,
            "is_primary": self.is_primary,
            "confidence": self.confidence,
            "evidence": [r.to_dict() for r in self.evidence],
            "review_flagged": self.review_flagged,
            "review_reasons": list(self.review_reasons),
        }


@dataclass
class CoverageItem:
    """One answer a deck carries, with evidence and a reliability flag."""

    category: str
    strength: int
    reliable: bool  # strength >= core_copy_threshold (not just a one-of tech)
    confidence: str
    source: str  # "tag_backed" | "heuristic"
    evidence: list[CardRef] = field(default_factory=list)
    review_flagged: bool = False
    review_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "strength": self.strength,
            "reliable": self.reliable,
            "confidence": self.confidence,
            "source": self.source,
            "evidence": [r.to_dict() for r in self.evidence],
            "review_flagged": self.review_flagged,
            "review_reasons": list(self.review_reasons),
        }


@dataclass
class DeckThreatProfile:
    """A deck's full matchup-threat surface (presents + exposes + coverage)."""

    deck_id: str
    deck_hash: str
    archetype_label: str  # display metadata only; never inference logic
    split: str
    profile_confidence: str  # carried from DeckProfileInference v0
    threat_confidence: str

    presents: list[ThreatItem] = field(default_factory=list)  # offensive
    exposes: list[ThreatItem] = field(default_factory=list)  # structural liabilities
    coverage: list[CoverageItem] = field(default_factory=list)

    reason_tags: list[str] = field(default_factory=list)
    manual_review_reasons: list[str] = field(default_factory=list)

    def all_threats(self) -> list[ThreatItem]:
        return self.presents + self.exposes

    def coverage_map(self) -> dict[str, CoverageItem]:
        return {c.category: c for c in self.coverage}

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "deck_id": self.deck_id,
            "deck_hash": self.deck_hash,
            "archetype_label": self.archetype_label,
            "split": self.split,
            "profile_confidence": self.profile_confidence,
            "threat_confidence": self.threat_confidence,
            "presents": [t.to_dict() for t in self.presents],
            "exposes": [t.to_dict() for t in self.exposes],
            "coverage": [c.to_dict() for c in self.coverage],
            "reason_tags": list(self.reason_tags),
            "manual_review_reasons": list(self.manual_review_reasons),
        }


@dataclass
class AnsweredThreat:
    """How a subject's coverage answers one opponent threat in a matchup."""

    threat_category: str
    status: str  # "full" | "partial" | "none"
    answered_by: list[str] = field(default_factory=list)  # coverage categories used

    def to_dict(self) -> dict[str, Any]:
        return {
            "threat_category": self.threat_category,
            "status": self.status,
            "answered_by": list(self.answered_by),
        }


@dataclass
class MatchupPairing:
    """Subject-vs-opponent coverage-gap analysis. No win-rate fields, by design."""

    subject_deck_id: str
    opponent_deck_id: str
    subject_label: str  # display only
    opponent_label: str  # display only
    opponent_threats: list[str] = field(default_factory=list)
    answered: list[AnsweredThreat] = field(default_factory=list)
    coverage_gaps: list[str] = field(default_factory=list)
    matchup_confidence: str = "high"
    reason_tags: list[str] = field(default_factory=list)
    manual_review_reasons: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "subject_deck_id": self.subject_deck_id,
            "opponent_deck_id": self.opponent_deck_id,
            "subject_label": self.subject_label,
            "opponent_label": self.opponent_label,
            "opponent_threats": list(self.opponent_threats),
            "answered": [a.to_dict() for a in self.answered],
            "coverage_gaps": list(self.coverage_gaps),
            "matchup_confidence": self.matchup_confidence,
            "reason_tags": list(self.reason_tags),
            "manual_review_reasons": list(self.manual_review_reasons),
        }


@dataclass
class MatchupConfig:
    """Tunable thresholds. Explicit — no hardcoded card names anywhere."""

    core_copy_threshold: int = 2  # coverage/threat reliability cutoff (a real line)
    min_threat_strength: int = 2  # weighted copies for a threat to be "primary"
    tool_dependency_min: int = 2
    stadium_dependency_min: int = 2
    special_energy_min: int = 2
    bench_fragility_min: int = 2
    ability_engine_min: int = 1
    self_ko_min: int = 1
    hand_size_min: int = 2
    setup_stage2_min: int = 2  # stage2 copies that make a deck "setup-reliant"
    output_csv_name: str = "matchup_threats_v0.csv"
    output_jsonl_name: str = "matchup_threats_v0.jsonl"
    pairings_jsonl_name: str = "matchup_pairings_v0.jsonl"
    report_name: str = "matchup_threat_inference_v0_report.md"
