"""Schemas + reason-tag vocabulary for StrategicActionScorer v0.

Offline/training-only. Stage 4 of:

    card text -> function tags -> deck profile -> matchup threats
                 -> action scores -> reason tags

This package owns ``matchup threats -> action scores + reason tags``. It is a
*transparent* scorer: it ranks candidate action records against a matchup context
(subject + opponent threat surfaces from MatchupThreatInference v0) using only
card-function / deck-profile / matchup-threat evidence, and explains every score with
reason tags + a confidence + a per-component breakdown.

Hard limits (enforced by tests):
- It changes **no** runtime policy behavior — `main.py` / `cg` never import it.
- It trains **no** labels, runs **no** BC / RL.
- It emits **no** win-rate / EV fields.
- It uses **no** archetype/deck names as logic (only threat/coverage categories).

A real engine action/state stream is NOT assumed. A "candidate action" is an abstract
record (its card-function role + a matchup context); the build script materialises
candidates from each deck's own answer/attacker/engine cards — it does not fabricate
live game state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from deepmew.matchup_threat_inference.schemas import (
    ANSWER_MATRIX,
    DeckThreatProfile,
    weakest,
)

ARTIFACT_VERSION = "deepmew.strategic_action_scoring.v0"
SCORING_KIND = "strategic_action_scoring"

# --- Reason tags (the 15 required, stable display order) ---------------------
ANSWER_REASON_TAGS: tuple[str, ...] = (
    "ANSWER_BENCH_ATTACK_DAMAGE",
    "ANSWER_BENCH_DAMAGE_COUNTERS",
    "ANSWER_ABILITY_COUNTERS",
    "HAND_RESET_HAND_SIZE_PLAN",
    "REMOVE_LOAD_BEARING_TOOL",
    "BUMP_LOAD_BEARING_STADIUM",
    "DENY_SPECIAL_ENERGY",
    "PROTECT_FRAGILE_BENCH",
    "ADVANCE_MAIN_ATTACKER",
    "SET_UP_ENGINE",
)
BAD_REASON_TAGS: tuple[str, ...] = (
    "BAD_OVERBENCH_INTO_SPREAD",
    "BAD_WRONG_PROTECTION_TYPE",
    "BAD_LOW_VALUE_TECH_TIMING",
    "BAD_DISCARD_CORE_PIECE",
    "BAD_RESOURCE_TIMING",
)
REASON_TAGS: tuple[str, ...] = ANSWER_REASON_TAGS + BAD_REASON_TAGS
_REASON_INDEX = {t: i for i, t in enumerate(REASON_TAGS)}


def reason_sort_key(tag: str) -> int:
    return _REASON_INDEX.get(tag, len(REASON_TAGS))


# --- Threat -> positive answer-tag mapping ------------------------------------
# Which reason tag is earned when a candidate's coverage answers a present
# opponent threat. (Threats with no dedicated answer tag are intentionally absent.)
THREAT_ANSWER_TAG: dict[str, str] = {
    "bench_attack_damage_threat": "ANSWER_BENCH_ATTACK_DAMAGE",
    "bench_damage_counter_threat": "ANSWER_BENCH_DAMAGE_COUNTERS",
    "ability_damage_counter_threat": "ANSWER_ABILITY_COUNTERS",
    "hand_size_scaling_threat": "HAND_RESET_HAND_SIZE_PLAN",
    "tool_dependency_threat": "REMOVE_LOAD_BEARING_TOOL",
    "stadium_dependency_threat": "BUMP_LOAD_BEARING_STADIUM",
    "special_energy_dependency_threat": "DENY_SPECIAL_ENERGY",
}

# Coverage categories that are "protection" (bringing the wrong one = a mistake).
PROTECTION_COVERAGE: frozenset[str] = frozenset(
    {"bench_attack_damage_protection", "damage_counter_protection",
     "attack_effect_protection", "ability_prevention"}
)
# Present bench/counter threats that protection is meant to answer.
BENCH_THREATS: frozenset[str] = frozenset(
    {"bench_attack_damage_threat", "bench_damage_counter_threat", "ability_damage_counter_threat"}
)
# Spread threats that punish a fragile bench / over-benching.
SPREAD_THREATS: frozenset[str] = frozenset(
    {"bench_attack_damage_threat", "bench_damage_counter_threat"}
)
# Structural threats whose answer only counts when the dependency is load-bearing.
STRUCTURAL_ANSWER_THREATS: frozenset[str] = frozenset(
    {"tool_dependency_threat", "stadium_dependency_threat", "special_energy_dependency_threat"}
)

# Action kinds (abstract; not engine OptionType values).
KIND_PLAY_ANSWER = "play_answer"
KIND_ATTACK_ADVANCE = "attack_advance"
KIND_SETUP_ENGINE = "setup_engine"
KIND_BENCH_PLAY = "bench_play"
KIND_RESOURCE_SPEND = "resource_spend"


@dataclass
class ScorerConfig:
    """Transparent, explicit weights. No hardcoded card names anywhere."""

    core_copy_threshold: int = 2
    load_bearing_min: int = 2  # structural-threat strength needed to justify an answer

    base_play_answer: float = 0.0
    base_attack_advance: float = 1.0
    base_setup_engine: float = 0.8
    base_bench_play: float = 0.3
    base_resource_spend: float = 0.0

    answer_full_core: float = 3.0
    answer_full_tech: float = 1.2
    answer_partial: float = 1.0
    protect_fragile_bonus: float = 1.5
    advance_main_bonus: float = 1.5
    setup_engine_bonus: float = 1.2

    wrong_type_penalty: float = -2.0
    low_value_tech_penalty: float = -1.5
    overbench_penalty: float = -2.0
    discard_core_penalty: float = -3.0
    resource_timing_penalty: float = -1.5

    # Build-side bounds (keep the dataset inspectable).
    max_opponents: int = 6
    max_candidates_per_deck: int = 14

    output_csv_name: str = "strategic_action_scores_v0.csv"
    output_jsonl_name: str = "strategic_action_scores_v0.jsonl"
    report_name: str = "strategic_action_scoring_v0_report.md"


@dataclass(frozen=True)
class ActionCandidate:
    """An abstract candidate action: its card-function role, not live game state."""

    candidate_id: str
    action_kind: str
    card_id: str
    card_name: str
    copies: int
    is_core: bool
    provides_coverage: tuple[str, ...] = ()
    advances_main_attacker: bool = False
    sets_up_engine: bool = False
    benches_fragile: bool = False
    discards_core_piece: bool = False
    resource_mistimed: bool = False
    card_confidence: str = "high"

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "action_kind": self.action_kind,
            "card_id": self.card_id,
            "card_name": self.card_name,
            "copies": self.copies,
            "is_core": self.is_core,
            "provides_coverage": list(self.provides_coverage),
            "advances_main_attacker": self.advances_main_attacker,
            "sets_up_engine": self.sets_up_engine,
            "benches_fragile": self.benches_fragile,
            "discards_core_piece": self.discards_core_piece,
            "resource_mistimed": self.resource_mistimed,
            "card_confidence": self.card_confidence,
        }


@dataclass(frozen=True)
class MatchupContext:
    """Subject + opponent threat surfaces from MatchupThreatInference v0."""

    subject: DeckThreatProfile
    opponent: DeckThreatProfile

    def opponent_primary_threats(self) -> dict[str, Any]:
        return {t.category: t for t in self.opponent.all_threats() if t.is_primary}

    def subject_exposes(self) -> set[str]:
        return {t.category for t in self.subject.exposes if t.is_primary}


@dataclass
class ScoreComponent:
    reason: str
    delta: float

    def to_dict(self) -> dict[str, Any]:
        return {"reason": self.reason, "delta": round(self.delta, 4)}


@dataclass
class ActionScore:
    """Transparent score for one candidate in one matchup. NO win-rate fields."""

    candidate_id: str
    card_id: str
    card_name: str
    action_kind: str
    subject_deck_id: str
    opponent_deck_id: str
    score: float
    reason_tags: list[str] = field(default_factory=list)
    components: list[ScoreComponent] = field(default_factory=list)
    confidence: str = "high"

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "card_id": self.card_id,
            "card_name": self.card_name,
            "action_kind": self.action_kind,
            "subject_deck_id": self.subject_deck_id,
            "opponent_deck_id": self.opponent_deck_id,
            "score": round(self.score, 4),
            "reason_tags": list(self.reason_tags),
            "components": [c.to_dict() for c in self.components],
            "confidence": self.confidence,
        }


__all__ = [
    "ARTIFACT_VERSION",
    "REASON_TAGS",
    "ANSWER_REASON_TAGS",
    "BAD_REASON_TAGS",
    "THREAT_ANSWER_TAG",
    "PROTECTION_COVERAGE",
    "BENCH_THREATS",
    "SPREAD_THREATS",
    "STRUCTURAL_ANSWER_THREATS",
    "ANSWER_MATRIX",
    "ScorerConfig",
    "ActionCandidate",
    "MatchupContext",
    "ScoreComponent",
    "ActionScore",
    "reason_sort_key",
    "weakest",
    "KIND_PLAY_ANSWER",
    "KIND_ATTACK_ADVANCE",
    "KIND_SETUP_ENGINE",
    "KIND_BENCH_PLAY",
    "KIND_RESOURCE_SPEND",
]
