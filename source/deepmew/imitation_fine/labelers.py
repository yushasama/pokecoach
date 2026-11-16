"""Deterministic pseudo-labelers for K4.1 intent / punishment / risk / last-stand.

Every label is derived from a single observed decision (option type + resolved
identity + role), the engine-free board snapshot, the deck's archetype goal, and
the final result. These are observational, same-frame pseudo-labels -- NOT proven
full-game counterfactuals (Bible §8 / CLAUDE.md §3). Weak evidence emits
``unknown`` / low confidence rather than a confident guess.

Doctrine baked in here:
- Punishment is decomposed and means *loss of future agency* (a branch where edge
  becomes unrecoverable), not "took damage next turn". It is a forward board-risk
  surface, deliberately NOT keyed off the final loss, to avoid outcome leakage.
- Risk is contextual, never blanket aversion: a committal line while *behind and
  passive-losing* earns a justification label; the same line while ahead/stable is
  ``risk_unjustified``.
- Last stand: when passive play slow-bleeds to a near-certain loss, the objective
  flips from preserving position to maximizing comeback equity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deepmew.imitation_fine import card_roles as roles
from deepmew.imitation_fine.schemas import UNKNOWN, BoardSnapshot, FineConfig

# Passive / low-commitment option categories used by several heuristics.
_PASSIVE_OPTIONS = {"END", "ABILITY"}


@dataclass
class LastStandResult:
    passive_line_loses: bool
    last_stand_triggered: bool
    comeback_equity_line: bool
    slow_bleed_risk: bool
    comeback_equity_score: float


@dataclass
class LabelResult:
    goal_of_play: str
    goal_confidence: float
    risk_justification: str
    punishment_labels: list[str] = field(default_factory=list)
    last_stand: LastStandResult | None = None
    label_confidence: float = 0.0


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def derive_goal_of_play(
    option_type: str,
    role: str,
    target_area: str,
    board: BoardSnapshot,
    cfg: FineConfig,
) -> tuple[str, float]:
    """Map one decision to its fixed-vocabulary intent + a confidence."""
    if option_type == "ATTACK":
        if board.opp_prizes_remaining <= 2:
            return "lethal_or_forced_win", 0.6
        if board.opp_active_hp_fraction <= cfg.low_hp_fraction:
            return "prize_conversion", 0.6
        return "damage_setup", 0.5

    if option_type == "ATTACH":
        if target_area == "BENCH":
            return "board_development", 0.6
        if target_area == "ACTIVE":
            return "attacker_readiness", 0.6
        return "attacker_readiness", 0.4

    if option_type == "PLAY":
        mapping = {
            roles.POKEMON_BASIC: ("board_development", 0.7),
            roles.POKEMON_EVO: ("attacker_readiness", 0.6),
            roles.SUPPORTER_GUST: ("deny_punish", 0.7),
            roles.SUPPORTER_DISRUPT: ("disruption", 0.7),
            roles.SUPPORTER_DRAW: ("consistency_find_piece", 0.7),
            roles.ITEM_CONSISTENCY: ("consistency_find_piece", 0.6),
            roles.ENERGY: ("attacker_readiness", 0.4),
            roles.TOOL: ("reduce_punishability", 0.4),
        }
        return mapping.get(role, ("board_development", 0.3))

    if option_type == "EVOLVE":
        return "attacker_readiness", 0.6

    if option_type == "RETREAT":
        if board.my_active_hp_fraction <= cfg.low_hp_fraction:
            return "forced_defense", 0.6
        return "reduce_punishability", 0.5

    if option_type == "ABILITY":
        return "consistency_find_piece", 0.35

    if option_type == "END":
        if board.prize_deficit >= cfg.behind_prize_margin and board.turn >= cfg.late_turn:
            return "stall_extend_game", 0.4
        return "resource_preservation", 0.45

    return UNKNOWN, 0.2


def derive_last_stand(board: BoardSnapshot, cfg: FineConfig) -> LastStandResult:
    """Decide whether passive play is losing and how recoverable the spot is."""
    behind = board.prize_deficit >= cfg.behind_prize_margin
    decisively_behind = board.prize_deficit >= cfg.last_stand_prize_margin
    late = board.turn >= cfg.late_turn
    weak_board = board.my_active_hp_fraction <= cfg.low_hp_fraction or board.my_bench_count == 0

    passive_line_loses = bool(behind and late and weak_board)
    last_stand_triggered = bool(decisively_behind and late and weak_board)
    slow_bleed_risk = bool(
        behind and (board.my_deck_count <= cfg.low_deck_count or board.my_hand_count <= cfg.low_hand_count or weak_board)
    )

    # Comeback equity: how recoverable the position still is. Start neutral,
    # penalize the prize deficit, reward remaining resources (bench, hand, board
    # energy, live deck). Bounded to [0, 1].
    resources = (
        min(board.my_bench_count, 3) * 0.1
        + min(board.my_hand_count, 4) * 0.05
        + min(board.my_active_energy, 3) * 0.07
        + (0.1 if board.my_deck_count > cfg.low_deck_count else 0.0)
    )
    score = _clamp(0.5 - 0.1 * max(0, board.prize_deficit) + resources)
    comeback_equity_line = bool(last_stand_triggered and score >= 0.4)
    return LastStandResult(
        passive_line_loses=passive_line_loses,
        last_stand_triggered=last_stand_triggered,
        comeback_equity_line=comeback_equity_line,
        slow_bleed_risk=slow_bleed_risk,
        comeback_equity_score=round(score, 4),
    )


def _is_committal(option_type: str, target_area: str) -> bool:
    if option_type in ("ATTACK", "EVOLVE"):
        return True
    return option_type == "ATTACH" and target_area == "ACTIVE"


def derive_risk(
    option_type: str,
    target_area: str,
    board: BoardSnapshot,
    last_stand: LastStandResult,
    cfg: FineConfig,
) -> tuple[str, float]:
    """Contextual risk justification. Empty string == no notable risk taken."""
    committal = _is_committal(option_type, target_area)
    exposed = board.my_bench_count == 0 or board.my_active_hp_fraction <= cfg.low_hp_fraction
    risky = committal and exposed
    if not risky:
        return "", 0.0

    if last_stand.last_stand_triggered or last_stand.passive_line_loses:
        if option_type == "ATTACK" and board.opp_prizes_remaining <= 2:
            return "risk_justified_by_lethal", 0.6
        if option_type == "ATTACK":
            return "risk_justified_by_prize_swing", 0.55
        return "risk_justified_by_only_out", 0.55

    if board.prize_deficit >= cfg.behind_prize_margin:
        return "risk_justified_by_forced_tempo", 0.45

    # Committal + exposed while ahead/stable: unnecessary risk.
    return "risk_unjustified", 0.5


def derive_punishment(
    option_type: str,
    board: BoardSnapshot,
    risk_label: str,
    last_stand: LastStandResult,
    cfg: FineConfig,
) -> tuple[list[str], float]:
    """Decomposed forward punishment surface (observational, outcome-independent)."""
    labels: list[str] = []
    risk_justified = risk_label.startswith("risk_justified")
    late = board.turn >= cfg.late_turn

    liability = board.my_active_max_hp >= cfg.liability_min_max_hp
    if (
        liability
        and board.my_active_hp_fraction <= cfg.low_hp_fraction
        and board.my_bench_count == 0
        and not risk_justified
    ):
        labels.append("board_liability_punish")

    if (
        board.opp_prizes_remaining <= 2
        and option_type in _PASSIVE_OPTIONS
        and not last_stand.last_stand_triggered
    ):
        labels.append("prize_punish")

    if board.my_active_energy == 0 and board.my_bench_count == 0 and late:
        labels.append("tempo_lock_punish")

    if board.my_hand_count <= cfg.low_hand_count and board.my_deck_count <= cfg.low_deck_count:
        labels.append("resource_punish")

    if board.my_deck_count <= cfg.low_deck_count and board.prize_deficit >= cfg.behind_prize_margin:
        labels.append("deck_plan_punish")

    if (
        option_type == "END"
        and board.prize_deficit >= cfg.behind_prize_margin
        and late
        and not last_stand.last_stand_triggered
    ):
        labels.append("clock_or_sequence_punish")

    # information_punish needs hidden-info reasoning not available single-frame; it
    # stays in the vocabulary but is intentionally never asserted here.
    confidence = 0.4 if labels else 0.0
    # De-dup while preserving order.
    seen: set[str] = set()
    deduped = [lbl for lbl in labels if not (lbl in seen or seen.add(lbl))]
    return deduped, confidence


def label_decision(
    option_type: str,
    card_id: int | None,
    target_area: str,
    board: BoardSnapshot,
    cfg: FineConfig,
) -> LabelResult:
    """Run all labelers for one decision and combine into a LabelResult."""
    role = roles.card_role(card_id) if option_type in ("PLAY", "EVOLVE") else roles.UNKNOWN_ROLE
    goal, goal_conf = derive_goal_of_play(option_type, role, target_area, board, cfg)
    last_stand = derive_last_stand(board, cfg)
    risk_label, _risk_conf = derive_risk(option_type, target_area, board, last_stand, cfg)
    punishment, _pun_conf = derive_punishment(option_type, board, risk_label, last_stand, cfg)
    label_confidence = round(goal_conf * (1.0 if board.valid else 0.5), 4)
    return LabelResult(
        goal_of_play=goal,
        goal_confidence=goal_conf,
        risk_justification=risk_label,
        punishment_labels=punishment,
        last_stand=last_stand,
        label_confidence=label_confidence,
    )
