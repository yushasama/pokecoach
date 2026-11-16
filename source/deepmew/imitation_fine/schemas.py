"""K4.1 fine-grained imitation schema: vocabularies, example rows, and config.

K4 broad action-category behavior cloning was an honest negative: archetype labels
did not improve which *category* the player chose, because select context already
dominates that. K4.1 labels replay decisions at finer granularity -- chosen option
identity, card_id, target, plus programmatic intent pseudo-labels (goal_of_deck,
goal_of_play, punishment, risk justification, last-stand/comeback) -- so that
archetype/deck identity can pay off where it actually lives: which card, which
target, which option advances the deck's plan while avoiding punishment.

All labels are DETERMINISTIC pseudo-labels derived from existing replay rows,
legal options, card metadata, K2A archetypes, board-state snapshots, and outcomes.
They are not hand annotations, and weak evidence is allowed to emit ``unknown`` /
low confidence rather than a confident guess. This layer is offline/training-only
and is never imported by the submission/eval runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ARTIFACT_VERSION = "deepmew.imitation_fine.v1"
IMITATION_FINE_KIND = "imitation_fine"

OUTPUT_FILES = {
    "examples": "imitation_fine_examples.jsonl",
    "label_summary": "imitation_label_summary.csv",
    "goal_of_deck_summary": "goal_of_deck_summary.csv",
    "goal_of_play_summary": "goal_of_play_summary.csv",
    "punishment_summary": "punishment_summary.csv",
    "risk_summary": "risk_justification_summary.csv",
    "last_stand_summary": "last_stand_summary.csv",
    "teacher_summary": "teacher_filter_summary.csv",
    "report": "imitation_fine_report.md",
}

UNKNOWN = "unknown"

# Broad, archetype/package-derived deck-plan vocabulary (intentionally coarse).
GOAL_OF_DECK_VOCAB: tuple[str, ...] = (
    "prize_race",
    "setup_sweeper",
    "tempo_pressure",
    "control_stall",
    "resource_denial",
    "disruption",
    "toolbox_flex",
    "engine_setup",
    "spread_damage",
    UNKNOWN,
)

# Fixed per-decision intent vocabulary.
GOAL_OF_PLAY_VOCAB: tuple[str, ...] = (
    "attacker_readiness",
    "board_development",
    "prize_conversion",
    "damage_setup",
    "consistency_find_piece",
    "resource_preservation",
    "deny_punish",
    "reduce_punishability",
    "disruption",
    "stall_extend_game",
    "lethal_or_forced_win",
    "forced_defense",
    UNKNOWN,
)

# Punishment is decomposed, never a single mush penalty. A choice is "punished"
# when it leads into a branch where future edge becomes unrecoverable or very hard
# to regain against competent opposition -- not merely "took damage next turn".
PUNISHMENT_VOCAB: tuple[str, ...] = (
    "prize_punish",
    "tempo_lock_punish",
    "resource_punish",
    "board_liability_punish",
    "deck_plan_punish",
    "clock_or_sequence_punish",
    "information_punish",
)

# Risk justification. Doctrine: do NOT learn blanket risk aversion. Risk is bad
# only when it sacrifices position without buying comeback equity; risk is correct
# when passive play loses anyway.
RISK_VOCAB: tuple[str, ...] = (
    "risk_justified_by_lethal",
    "risk_justified_by_prize_race",
    "risk_justified_by_forced_tempo",
    "risk_justified_by_setup_payoff",
    "risk_justified_by_denying_opponent_plan",
    "risk_justified_by_only_out",
    "risk_justified_by_prize_swing",
    "risk_justified_by_tempo_flip",
    "risk_justified_by_disruption_window",
    "risk_unjustified",
)

# Last-stand / comeback equity flags.
LAST_STAND_FLAGS: tuple[str, ...] = (
    "passive_line_loses",
    "last_stand_triggered",
    "comeback_equity_line",
    "slow_bleed_risk",
)

# The strategic decision context: the MAIN turn phase (SelectContext.MAIN == 0).
MAIN_CONTEXT_ID = 0


@dataclass
class FineConfig:
    """Thresholds for the deterministic labelers and BC tables."""

    # Board/last-stand thresholds (prizes are *remaining*; lower == winning).
    behind_prize_margin: int = 1          # behind when my_remaining - opp_remaining >= this
    last_stand_prize_margin: int = 2      # decisively behind
    late_turn: int = 6                    # turn at/after which slow bleed bites
    low_hp_fraction: float = 0.4          # active HP fraction considered exposed
    low_deck_count: int = 6               # deck-out danger threshold
    low_hand_count: int = 2               # resource-starvation threshold

    # Multi-prize liability: ex/Mega cards give up 2-3 prizes when KO'd.
    liability_min_max_hp: int = 200       # high-HP terminal proxy for a multi-prize ex

    # Confidence bands.
    confidence_high: float = 0.66
    confidence_low: float = 0.33

    # BC table backoff: a conditioned cell must reach this before it is trusted.
    min_cell_count: int = 8
    # Filtered-BC trust gate.
    teacher_min_confidence: float = 0.5
    teacher_min_label_confidence: float = 0.4

    # K1.7 rank-aware teacher weighting (DEC-037). Rank coverage is partial (~45%),
    # so rank scaling is applied only when a pilot rank is known and the
    # high-rank-only filter is OFF by default (missing-rank rows stay usable).
    use_rank_teacher_weight: bool = True
    high_rank_filter_enabled: bool = False
    max_teacher_rank_for_filtered: int | None = 50


@dataclass
class BoardSnapshot:
    """Engine-free per-decision board read from raw ``state_json`` (acting seat)."""

    turn: int
    my_prizes_remaining: int
    opp_prizes_remaining: int
    my_active_id: int | None
    my_active_hp: int
    my_active_max_hp: int
    my_active_energy: int
    my_bench_count: int
    opp_active_id: int | None
    opp_active_hp: int
    opp_active_max_hp: int
    opp_bench_count: int
    my_hand_count: int
    my_deck_count: int
    valid: bool = True

    @property
    def prize_deficit(self) -> int:
        """Positive == behind (I have more prizes left to take than the opponent)."""
        return self.my_prizes_remaining - self.opp_prizes_remaining

    @property
    def my_active_hp_fraction(self) -> float:
        if self.my_active_max_hp <= 0:
            return 1.0
        return max(0.0, min(1.0, self.my_active_hp / self.my_active_max_hp))

    @property
    def opp_active_hp_fraction(self) -> float:
        if self.opp_active_max_hp <= 0:
            return 1.0
        return max(0.0, min(1.0, self.opp_active_hp / self.opp_active_max_hp))


@dataclass
class FineExample:
    """One fine-grained, deterministically labeled imitation row."""

    episode_id: str
    seat: int
    turn: int
    step: int
    deck_hash: str
    archetype: str
    variant_label: str
    archetype_confidence: float
    classification_status: str
    opponent_archetype: str
    split: str
    deck_zoo_bucket: str

    select_context_id: int
    select_context: str
    option_type: str
    chosen_indices: list[int]
    chosen_option_identity: str
    chosen_card_id: int | None
    chosen_card_name: str
    chosen_target_area: str
    chosen_target_slot: int | None
    chosen_target_card_id: int | None
    chosen_target_card_name: str

    goal_of_deck: str
    goal_of_play: str
    punishment_labels: list[str]
    risk_justification: str
    passive_line_loses: bool
    last_stand_triggered: bool
    comeback_equity_line: bool
    slow_bleed_risk: bool
    comeback_equity_score: float

    result: int
    won: bool
    is_draw: bool
    result_reason: str
    teacher_weight: float
    label_confidence: float
    missing_fields: list[str]
    source_run_id: str = ""

    # K1.7 rank metadata (leaderboard standing at scout time; partial coverage ~45%).
    # Defaults are the rank-missing/neutral state so back-compat and rank-absent
    # builds work unchanged.
    pilot_rank: int | None = None
    pilot_rank_bucket: str = "rank_missing"
    pilot_rank_weight: float = 1.0       # K5-lite teacher rank factor (neutral 1.0 when missing)
    pilot_rank_missing: bool = True
    pilot_team_id: str = ""
    opponent_rank: int | None = None
    opponent_rank_bucket: str = "rank_missing"
    opponent_rank_missing: bool = True

    def fine_target_key(self) -> str:
        """Stable fine-grained imitation target for BC tables.

        PLAY/EVOLVE resolve to the source card; ATTACH to the target area; other
        categories fall back to the bare option type. This is the label the BC
        scorer learns to reproduce, finer than K4's bare option category.
        """
        if self.option_type in ("PLAY", "EVOLVE") and self.chosen_card_id is not None:
            return f"{self.option_type}:{self.chosen_card_id}"
        if self.option_type == "ATTACH" and self.chosen_target_area:
            return f"ATTACH:{self.chosen_target_area}"
        return self.option_type

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id, "seat": self.seat, "turn": self.turn, "step": self.step,
            "deck_hash": self.deck_hash, "archetype": self.archetype, "variant_label": self.variant_label,
            "archetype_confidence": self.archetype_confidence, "classification_status": self.classification_status,
            "opponent_archetype": self.opponent_archetype, "split": self.split,
            "deck_zoo_bucket": self.deck_zoo_bucket,
            "select_context_id": self.select_context_id, "select_context": self.select_context,
            "option_type": self.option_type, "chosen_indices": self.chosen_indices,
            "chosen_option_identity": self.chosen_option_identity, "chosen_card_id": self.chosen_card_id,
            "chosen_card_name": self.chosen_card_name, "chosen_target_area": self.chosen_target_area,
            "chosen_target_slot": self.chosen_target_slot, "chosen_target_card_id": self.chosen_target_card_id,
            "chosen_target_card_name": self.chosen_target_card_name,
            "fine_target_key": self.fine_target_key(),
            "goal_of_deck": self.goal_of_deck, "goal_of_play": self.goal_of_play,
            "punishment_labels": self.punishment_labels, "risk_justification": self.risk_justification,
            "passive_line_loses": self.passive_line_loses, "last_stand_triggered": self.last_stand_triggered,
            "comeback_equity_line": self.comeback_equity_line, "slow_bleed_risk": self.slow_bleed_risk,
            "comeback_equity_score": self.comeback_equity_score,
            "result": self.result, "won": self.won, "is_draw": self.is_draw, "result_reason": self.result_reason,
            "teacher_weight": self.teacher_weight, "label_confidence": self.label_confidence,
            "missing_fields": self.missing_fields, "source_run_id": self.source_run_id,
            "pilot_rank": self.pilot_rank, "pilot_rank_bucket": self.pilot_rank_bucket,
            "pilot_rank_weight": self.pilot_rank_weight, "pilot_rank_missing": self.pilot_rank_missing,
            "pilot_team_id": self.pilot_team_id,
            "opponent_rank": self.opponent_rank, "opponent_rank_bucket": self.opponent_rank_bucket,
            "opponent_rank_missing": self.opponent_rank_missing,
        }
