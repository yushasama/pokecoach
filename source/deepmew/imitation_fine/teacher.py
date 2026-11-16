"""Teacher filtering and reward weighting for K5-lite behavior cloning.

Filtered BC trains only on *trusted* rows (won, confident archetype, decent label
confidence). Reward-weighted BC keeps more rows but scales each by a transparent,
multiplicative weight built from outcome, teacher trust, deck-goal progress,
punishment, and justified last-stand risk. Both are pure functions over a
``FineExample`` so they are deterministic and unit-testable.
"""

from __future__ import annotations

from deepmew.imitation_fine.schemas import FineConfig, FineExample
from deepmew.rank_meta.schemas import teacher_rank_factor

# Which goal_of_play advances which goal_of_deck. Membership grants the
# goal-progress reward bonus; absence is neutral, never a penalty.
GOAL_PROGRESS: dict[str, frozenset[str]] = {
    "prize_race": frozenset({"prize_conversion", "lethal_or_forced_win", "damage_setup", "attacker_readiness"}),
    "setup_sweeper": frozenset({"attacker_readiness", "board_development", "damage_setup", "prize_conversion"}),
    "tempo_pressure": frozenset({"attacker_readiness", "prize_conversion", "deny_punish", "damage_setup"}),
    "control_stall": frozenset({"stall_extend_game", "resource_preservation", "reduce_punishability", "disruption", "deny_punish"}),
    "resource_denial": frozenset({"disruption", "deny_punish", "resource_preservation"}),
    "disruption": frozenset({"disruption", "deny_punish", "reduce_punishability"}),
    "toolbox_flex": frozenset({"board_development", "consistency_find_piece", "deny_punish", "reduce_punishability"}),
    "engine_setup": frozenset({"consistency_find_piece", "board_development", "attacker_readiness"}),
    "spread_damage": frozenset({"damage_setup", "board_development", "prize_conversion"}),
}

WEIGHT_MIN = 0.1
WEIGHT_MAX = 3.0


def goal_progress(goal_of_deck: str, goal_of_play: str) -> bool:
    """True when this play advances the deck's plan."""
    return goal_of_play in GOAL_PROGRESS.get(goal_of_deck, frozenset())


def is_trusted_teacher(example: FineExample, cfg: FineConfig) -> bool:
    """Filtered-BC gate: trust only winning, confident, decently-labeled rows.

    With ``cfg.high_rank_filter_enabled`` (default off) the gate additionally
    requires a known pilot rank at/under ``cfg.max_teacher_rank_for_filtered`` -- an
    opt-in high-rank-only experiment. By default rank is NOT required, because rank
    coverage is only ~45% and the default must not discard all rank-missing rows.
    """
    if not example.won:
        return False
    if example.classification_status != "confident":
        return False
    if example.archetype_confidence < cfg.teacher_min_confidence:
        return False
    if example.label_confidence < cfg.teacher_min_label_confidence:
        return False
    if cfg.high_rank_filter_enabled:
        if example.pilot_rank_missing or example.pilot_rank is None:
            return False
        if cfg.max_teacher_rank_for_filtered is not None and example.pilot_rank > cfg.max_teacher_rank_for_filtered:
            return False
    return True


def reward_weight(example: FineExample, cfg: FineConfig | None = None) -> float:
    """Transparent multiplicative reward weight for reward-weighted BC."""
    cfg = cfg or FineConfig()
    weight = 1.0

    # Outcome.
    if example.is_draw:
        weight *= 0.8
    elif example.won:
        weight *= 1.5
    else:
        weight *= 0.6

    # Teacher trust.
    if example.classification_status == "confident" and example.archetype_confidence >= 0.66:
        weight *= 1.3
    elif example.classification_status in ("unknown", "low_confidence"):
        weight *= 0.7

    # Deck-goal progress.
    if goal_progress(example.goal_of_deck, example.goal_of_play):
        weight *= 1.2

    # Label confidence scales 0.6..1.0.
    weight *= 0.6 + 0.4 * max(0.0, min(1.0, example.label_confidence))

    risk_justified = example.risk_justification.startswith("risk_justified")
    last_stand = example.last_stand_triggered

    # Punishment downweights -- unless the risk was justified or it is a forced
    # last stand. Never penalize a correct comeback line just because it is risky.
    if example.punishment_labels and not (risk_justified or last_stand):
        weight *= 0.5

    # Reward correct last-stand comeback risk.
    if last_stand and risk_justified:
        weight *= 1.15

    # K1.7 rank-aware trust: scale by the pilot's leaderboard rank when known.
    # Neutral (1.0) for rank-missing rows, so partial coverage never drops them.
    if cfg.use_rank_teacher_weight and not example.pilot_rank_missing:
        weight *= teacher_rank_factor(example.pilot_rank_bucket)

    return round(max(WEIGHT_MIN, min(WEIGHT_MAX, weight)), 4)
