"""Typed domain objects and config for OpponentBeliefState v0.

Card IDs are canonical strings; names are display-only enrichment. Every field
used by the belief *update* logic is public/revealed. ``ground_truth_*`` style
fields never appear here - the evaluation module joins ground truth separately so
the update path cannot read it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CardId = str

ARTIFACT_VERSION = "deepmew.opponent_belief.v0"
BELIEF_KIND = "opponent_belief"

# --- Hypothesis hierarchy levels ----------------------------------------------
LEVEL_KNOWN_HASH = "known_hash"
LEVEL_UNKNOWN_VARIANT = "unknown_within_archetype"
LEVEL_GLOBAL_UNKNOWN = "global_unknown"

UNKNOWN_ARCHETYPE = "UNKNOWN"
UNKNOWN_VARIANT = "unknown_variant"
GLOBAL_UNKNOWN_ID = "GLOBAL_UNKNOWN"

# --- Public evidence event types (no private/future channels) -----------------
EVENT_CARD_REVEALED = "CARD_REVEALED"
EVENT_CARD_PLAYED = "CARD_PLAYED"
EVENT_POKEMON_BENCHED = "POKEMON_BENCHED"
EVENT_POKEMON_EVOLVED = "POKEMON_EVOLVED"
EVENT_ENERGY_ATTACHED = "ENERGY_ATTACHED"
EVENT_TOOL_ATTACHED = "TOOL_ATTACHED"
EVENT_STADIUM_PLAYED = "STADIUM_PLAYED"
EVENT_ABILITY_USED = "ABILITY_USED"
EVENT_ATTACK_USED = "ATTACK_USED"
EVENT_CARD_DISCARDED_PUBLIC = "CARD_DISCARDED_PUBLIC"
EVENT_SEARCH_RESULT_REVEALED = "SEARCH_RESULT_REVEALED"
EVENT_PUBLIC_RECOVERY = "PUBLIC_RECOVERY"

PUBLIC_EVENT_TYPES = frozenset(
    {
        EVENT_CARD_REVEALED,
        EVENT_CARD_PLAYED,
        EVENT_POKEMON_BENCHED,
        EVENT_POKEMON_EVOLVED,
        EVENT_ENERGY_ATTACHED,
        EVENT_TOOL_ATTACHED,
        EVENT_STADIUM_PLAYED,
        EVENT_ABILITY_USED,
        EVENT_ATTACK_USED,
        EVENT_CARD_DISCARDED_PUBLIC,
        EVENT_SEARCH_RESULT_REVEALED,
        EVENT_PUBLIC_RECOVERY,
    }
)

# Belief status vocabulary (configurable thresholds, not canon).
STATUS_AMBIGUOUS = "ambiguous"
STATUS_LIKELY = "likely"
STATUS_CONCENTRATED = "concentrated"
STATUS_EXACT_HASH_CANDIDATE = "exact_hash_candidate"

# Belief-update ablation modes (Workstream D1).
BELIEF_MODES = ("B0", "B1", "B2", "B3", "B4", "B5")


@dataclass(frozen=True)
class EvidenceEvent:
    """One public/revealed event. Only public fields - this is the leakage boundary.

    ``card_id`` is the canonical id of the revealed card. ``count`` is the number of
    *public copies seen so far* of this card (monotonic within a game). Anything not
    publicly observable (opponent hand, prizes, deck list) is intentionally absent.
    """

    event_id: str
    sequence_index: int
    turn: int
    actor: str  # "opponent" - the player we are modeling
    event_type: str
    card_id: CardId
    card_name: str = ""
    count: int = 1
    source_card_id: CardId | None = None
    target_zone: str | None = None
    reliability: float = 1.0
    evidence_scope: str = "public"
    availability_public: bool = False  # True only when legal availability was public

    def is_public(self) -> bool:
        return self.evidence_scope == "public" and self.event_type in PUBLIC_EVENT_TYPES


@dataclass(frozen=True)
class EvidenceContribution:
    """Per-hypothesis decomposition of one update step (full transparency)."""

    hypothesis_id: str
    posterior_before: float
    diagnosticity: float
    chain_bonus: float
    coverage: float
    contradiction: float
    log_score_after: float


@dataclass
class UpdateTrace:
    """The transparent record of one evidence event applied to the whole belief."""

    sequence_index: int
    event: EvidenceEvent
    diagnosticity_total: float
    chain_total: float
    coverage_total: float
    contradiction_total: float
    unknown_mass_before: float
    unknown_mass_after: float
    top_before: str
    top_after: str
    top_probability_after: float
    applied: bool = True
    skip_reason: str = ""
    reliability_applied: float = 1.0
    contributions: list[EvidenceContribution] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_index": self.sequence_index,
            "applied": self.applied,
            "skip_reason": self.skip_reason,
            "reliability_applied": round(self.reliability_applied, 6),
            "event": {
                "event_id": self.event.event_id,
                "turn": self.event.turn,
                "event_type": self.event.event_type,
                "card_id": self.event.card_id,
                "card_name": self.event.card_name,
                "count": self.event.count,
                "evidence_scope": self.event.evidence_scope,
            },
            "diagnosticity_total": round(self.diagnosticity_total, 6),
            "chain_total": round(self.chain_total, 6),
            "coverage_total": round(self.coverage_total, 6),
            "contradiction_total": round(self.contradiction_total, 6),
            "unknown_mass_before": round(self.unknown_mass_before, 6),
            "unknown_mass_after": round(self.unknown_mass_after, 6),
            "top_before": self.top_before,
            "top_after": self.top_after,
            "top_probability_after": round(self.top_probability_after, 6),
            "contributions": [
                {
                    "hypothesis_id": c.hypothesis_id,
                    "posterior_before": round(c.posterior_before, 6),
                    "diagnosticity": round(c.diagnosticity, 6),
                    "chain_bonus": round(c.chain_bonus, 6),
                    "coverage": round(c.coverage, 6),
                    "contradiction": round(c.contradiction, 6),
                    "log_score_after": round(c.log_score_after, 6),
                }
                for c in self.contributions
            ],
        }


@dataclass
class BeliefSnapshot:
    """A normalized hierarchical belief snapshot at one point in the trace."""

    sequence_index: int
    archetype_posterior: dict[str, float]
    variant_posterior: dict[str, float]
    known_hash_posterior: dict[str, float]
    structural_playstyle_posterior: dict[str, float]
    unknown_mass: float
    unknown_within_archetype_mass: float
    global_unknown_mass: float
    top_hypothesis: str
    top_archetype: str
    top_probability: float
    runner_up_probability: float
    margin: float
    normalized_entropy: float
    credible_set: list[str]
    credible_set_mass: float
    belief_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_index": self.sequence_index,
            "archetype_posterior": _round_map(self.archetype_posterior),
            "variant_posterior": _round_map(self.variant_posterior),
            "known_hash_posterior": _round_map(self.known_hash_posterior),
            "structural_playstyle_posterior": _round_map(self.structural_playstyle_posterior),
            "unknown_mass": round(self.unknown_mass, 6),
            "unknown_within_archetype_mass": round(self.unknown_within_archetype_mass, 6),
            "global_unknown_mass": round(self.global_unknown_mass, 6),
            "top_hypothesis": self.top_hypothesis,
            "top_archetype": self.top_archetype,
            "top_probability": round(self.top_probability, 6),
            "runner_up_probability": round(self.runner_up_probability, 6),
            "margin": round(self.margin, 6),
            "normalized_entropy": round(self.normalized_entropy, 6),
            "credible_set": list(self.credible_set),
            "credible_set_mass": round(self.credible_set_mass, 6),
            "belief_status": self.belief_status,
        }


def _round_map(values: dict[str, float]) -> dict[str, float]:
    return {k: round(v, 6) for k, v in sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))}


@dataclass
class BeliefConfig:
    """All weights and caps are explicit, configured, and reported.

    ``mode`` selects the ablation; the individual flags are derived from the mode in
    :func:`for_mode` but may also be set directly.
    """

    mode: str = "B5"
    use_diagnosticity: bool = True
    use_chain: bool = True
    use_contradiction: bool = True
    use_unknown_floor: bool = True
    hard_top1: bool = False  # B0
    uniform_over_matches: bool = False  # B1

    # Diagnosticity (log-lift) caps and role factors.
    max_card_log_contribution: float = 2.0
    lift_floor: float = 1e-6
    role_factor_terminal: float = 1.0
    role_factor_line: float = 0.4
    role_factor_support: float = 0.6
    role_factor_energy: float = 0.3
    role_factor_tech: float = 0.7
    role_factor_staple: float = 0.05
    role_factor_splash: float = 0.1

    # Chain / package combination evidence (correlation-safe, submodular).
    chain_gamma: float = 0.6
    chain_completion_bonus: float = 0.5
    chain_member_cap: float = 1.6  # cap on the submodular increment term per component

    # Hash-level multiset compatibility / contradiction.
    hash_coverage_weight: float = 0.35
    contradiction_penalty: float = 1.1
    overcopy_penalty: float = 1.4

    # Update tempering + UNKNOWN handling.
    temperature: float = 0.7  # < 1 slows lock-in (B5); 1.0 for un-tempered ablations
    unknown_floor: float = 0.04
    global_unknown_prior: float = 0.10

    # Status thresholds (configurable, not canon).
    concentrated_top_prob: float = 0.6
    concentrated_max_entropy: float = 0.55
    likely_top_prob: float = 0.4
    exact_hash_min_prob: float = 0.5
    exact_hash_min_margin: float = 0.2
    credible_mass: float = 0.8

    @classmethod
    def for_mode(cls, mode: str, **overrides: Any) -> "BeliefConfig":
        mode = mode.upper()
        if mode not in BELIEF_MODES:
            raise ValueError(f"unknown belief mode {mode!r}; expected one of {BELIEF_MODES}")
        cfg = cls(mode=mode)
        cfg.hard_top1 = mode == "B0"
        cfg.uniform_over_matches = mode == "B1"
        cfg.use_diagnosticity = mode in ("B2", "B3", "B4", "B5")
        cfg.use_unknown_floor = mode in ("B3", "B4", "B5")
        cfg.use_chain = mode in ("B4", "B5")
        cfg.use_contradiction = mode in ("B4", "B5")
        cfg.temperature = 0.7 if mode == "B5" else 1.0
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "use_diagnosticity": self.use_diagnosticity,
            "use_chain": self.use_chain,
            "use_contradiction": self.use_contradiction,
            "use_unknown_floor": self.use_unknown_floor,
            "hard_top1": self.hard_top1,
            "uniform_over_matches": self.uniform_over_matches,
            "max_card_log_contribution": self.max_card_log_contribution,
            "role_factors": {
                "terminal": self.role_factor_terminal,
                "line": self.role_factor_line,
                "support": self.role_factor_support,
                "energy": self.role_factor_energy,
                "tech": self.role_factor_tech,
                "staple": self.role_factor_staple,
                "splash": self.role_factor_splash,
            },
            "chain_gamma": self.chain_gamma,
            "chain_completion_bonus": self.chain_completion_bonus,
            "chain_member_cap": self.chain_member_cap,
            "hash_coverage_weight": self.hash_coverage_weight,
            "contradiction_penalty": self.contradiction_penalty,
            "overcopy_penalty": self.overcopy_penalty,
            "temperature": self.temperature,
            "unknown_floor": self.unknown_floor,
            "global_unknown_prior": self.global_unknown_prior,
            "credible_mass": self.credible_mass,
        }
