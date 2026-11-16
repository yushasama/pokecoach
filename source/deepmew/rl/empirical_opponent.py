"""Offline empirical opponent behavior and local-strength summaries.

This module is analysis-only. It consumes normalized actor-visible action rows
and completed pairwise game rows; it imports no learner, environment, or
runtime policy surface.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


BEHAVIOR_SCHEMA_VERSION = "deepmew.empirical_opponent_behavior.v1"
FROZEN_PANEL_BT_INDEX_SCHEMA_VERSION = "deepmew.frozen_panel_bt_index.v1"
BACKOFF_LEVELS = (
    "exact_policy_checkpoint_deck",
    "deck_strength_bucket",
    "strength_bucket",
    "deck",
    "global",
)


class EmpiricalOpponentError(ValueError):
    """Input rows or an estimator configuration are not auditable."""


def _text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EmpiricalOpponentError(f"{name} must be a non-empty string")

    return value.strip()


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EmpiricalOpponentError(f"{name} must be a positive integer")

    return value


def _positive_finite(value: Any, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise EmpiricalOpponentError(f"{name} must be finite and positive")

    return number


@dataclass(frozen=True)
class OpponentBehaviorRow:
    actor_visible_state_key: str
    action_family: str
    policy_checkpoint_id: str
    opponent_deck_hash: str
    strength_bucket: str
    count: int = 1

    def __post_init__(self) -> None:
        for field_name in (
            "actor_visible_state_key",
            "action_family",
            "policy_checkpoint_id",
            "opponent_deck_hash",
            "strength_bucket",
        ):
            object.__setattr__(self, field_name, _text(getattr(self, field_name), name=field_name))
        object.__setattr__(self, "count", _positive_int(self.count, name="count"))

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "OpponentBehaviorRow":
        return cls(
            actor_visible_state_key=row.get("actor_visible_state_key"),
            action_family=row.get("action_family"),
            policy_checkpoint_id=row.get("policy_checkpoint_id"),
            opponent_deck_hash=row.get("opponent_deck_hash"),
            strength_bucket=row.get("strength_bucket"),
            count=row.get("count", 1),
        )


@dataclass(frozen=True)
class BehaviorEstimate:
    probabilities: Mapping[str, float]
    chosen_level: str
    support: int
    candidate_support: int
    effective_support: float
    backoff_support: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BEHAVIOR_SCHEMA_VERSION,
            "probabilities": dict(self.probabilities),
            "chosen_level": self.chosen_level,
            "support": self.support,
            "candidate_support": self.candidate_support,
            "effective_support": self.effective_support,
            "backoff_support": dict(self.backoff_support),
        }


class EmpiricalOpponentBehavior:
    """Count-table behavior estimate with the mission's fixed backoff order."""

    def __init__(
        self,
        *,
        min_support: int,
        prior_strength: float,
        global_alpha: float,
        action_families: Sequence[str],
        cells: Mapping[str, Mapping[tuple[str, ...], Counter[str]]],
        global_counts: Counter[str],
    ) -> None:
        self.min_support = _positive_int(min_support, name="min_support")
        self.prior_strength = _positive_finite(prior_strength, name="prior_strength")
        self.global_alpha = _positive_finite(global_alpha, name="global_alpha")
        self.action_families = tuple(sorted({_text(value, name="action_family") for value in action_families}))
        if not self.action_families:
            raise EmpiricalOpponentError("at least one action family is required")
        self.cells = {level: dict(cells.get(level, {})) for level in BACKOFF_LEVELS[:-1]}
        self.global_counts = Counter(global_counts)

    @classmethod
    def fit(
        cls,
        rows: Iterable[OpponentBehaviorRow | Mapping[str, Any]],
        *,
        min_support: int = 8,
        prior_strength: float = 2.0,
        global_alpha: float = 0.5,
    ) -> "EmpiricalOpponentBehavior":
        cells: dict[str, defaultdict[tuple[str, ...], Counter[str]]] = {
            level: defaultdict(Counter) for level in BACKOFF_LEVELS[:-1]
        }
        global_counts: Counter[str] = Counter()
        families: set[str] = set()
        observed = 0
        for value in rows:
            row = value if isinstance(value, OpponentBehaviorRow) else OpponentBehaviorRow.from_mapping(value)
            keys = {
                "exact_policy_checkpoint_deck": (
                    row.actor_visible_state_key,
                    row.policy_checkpoint_id,
                    row.opponent_deck_hash,
                ),
                "deck_strength_bucket": (
                    row.actor_visible_state_key,
                    row.opponent_deck_hash,
                    row.strength_bucket,
                ),
                "strength_bucket": (row.actor_visible_state_key, row.strength_bucket),
                "deck": (row.actor_visible_state_key, row.opponent_deck_hash),
            }
            for level, key in keys.items():
                cells[level][key][row.action_family] += row.count
            global_counts[row.action_family] += row.count
            families.add(row.action_family)
            observed += row.count
        if observed == 0:
            raise EmpiricalOpponentError("behavior rows must not be empty")

        return cls(
            min_support=min_support,
            prior_strength=prior_strength,
            global_alpha=global_alpha,
            action_families=sorted(families),
            cells=cells,
            global_counts=global_counts,
        )

    def estimate(
        self,
        *,
        actor_visible_state_key: str,
        policy_checkpoint_id: str,
        opponent_deck_hash: str,
        strength_bucket: str,
        candidate_action_families: Sequence[str] | None = None,
    ) -> BehaviorEstimate:
        state_key = _text(actor_visible_state_key, name="actor_visible_state_key")
        policy_id = _text(policy_checkpoint_id, name="policy_checkpoint_id")
        deck_hash = _text(opponent_deck_hash, name="opponent_deck_hash")
        bucket = _text(strength_bucket, name="strength_bucket")
        if candidate_action_families is None:
            candidates = self.action_families
        else:
            candidates = tuple(
                sorted({_text(value, name="candidate_action_family") for value in candidate_action_families})
            )
        if not candidates:
            raise EmpiricalOpponentError("candidate_action_families must not be empty")
        queries = {
            "exact_policy_checkpoint_deck": (state_key, policy_id, deck_hash),
            "deck_strength_bucket": (state_key, deck_hash, bucket),
            "strength_bucket": (state_key, bucket),
            "deck": (state_key, deck_hash),
        }
        support: dict[str, int] = {}
        selected_level = "global"
        selected_counts = self.global_counts
        for level in BACKOFF_LEVELS[:-1]:
            counts = self.cells[level].get(queries[level], Counter())
            support[level] = sum(counts.values())
            if selected_level == "global" and support[level] >= self.min_support:
                selected_level = level
                selected_counts = counts
        support["global"] = sum(self.global_counts.values())
        global_candidate_support = sum(self.global_counts[family] for family in candidates)
        global_denominator = global_candidate_support + self.global_alpha * len(candidates)
        global_prior = {
            family: (self.global_counts[family] + self.global_alpha) / global_denominator
            for family in candidates
        }
        if selected_level == "global":
            probabilities = global_prior
            candidate_support = global_candidate_support
            effective_support = global_denominator
        else:
            candidate_support = sum(selected_counts[family] for family in candidates)
            denominator = candidate_support + self.prior_strength
            probabilities = {
                family: (
                    selected_counts[family] + self.prior_strength * global_prior[family]
                )
                / denominator
                for family in candidates
            }
            effective_support = denominator

        return BehaviorEstimate(
            probabilities=dict(sorted(probabilities.items())),
            chosen_level=selected_level,
            support=support[selected_level],
            candidate_support=candidate_support,
            effective_support=effective_support,
            backoff_support=support,
        )


@dataclass(frozen=True, order=True)
class PolicyDeckPair:
    policy_checkpoint_id: str
    deck_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "policy_checkpoint_id",
            _text(self.policy_checkpoint_id, name="policy_checkpoint_id"),
        )
        object.__setattr__(self, "deck_hash", _text(self.deck_hash, name="deck_hash"))

    @property
    def pair_id(self) -> str:
        payload = json.dumps(
            [self.policy_checkpoint_id, self.deck_hash],
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")

        return "policy-deck:" + hashlib.sha256(payload).hexdigest()[:24]

    def to_dict(self) -> dict[str, str]:
        return {
            "pair_id": self.pair_id,
            "policy_checkpoint_id": self.policy_checkpoint_id,
            "deck_hash": self.deck_hash,
        }


@dataclass(frozen=True)
class PairwiseGameRow:
    left: PolicyDeckPair
    right: PolicyDeckPair
    left_score: float
    count: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.left, PolicyDeckPair) or not isinstance(self.right, PolicyDeckPair):
            raise EmpiricalOpponentError("left and right must be PolicyDeckPair values")
        if self.left == self.right:
            raise EmpiricalOpponentError("pairwise game rows must contain two distinct policy-deck pairs")
        score = float(self.left_score)
        if score not in {0.0, 0.5, 1.0}:
            raise EmpiricalOpponentError("left_score must be 0, 0.5, or 1")
        object.__setattr__(self, "left_score", score)
        object.__setattr__(self, "count", _positive_int(self.count, name="count"))

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "PairwiseGameRow":
        outcome = row.get("outcome")
        if "left_score" in row:
            score = row["left_score"]
        elif outcome in {"win", "draw", "loss"}:
            score = {"win": 1.0, "draw": 0.5, "loss": 0.0}[outcome]
        else:
            raise EmpiricalOpponentError("pairwise row needs left_score or win/draw/loss outcome")

        return cls(
            left=PolicyDeckPair(
                policy_checkpoint_id=row.get("left_policy_checkpoint_id"),
                deck_hash=row.get("left_deck_hash"),
            ),
            right=PolicyDeckPair(
                policy_checkpoint_id=row.get("right_policy_checkpoint_id"),
                deck_hash=row.get("right_deck_hash"),
            ),
            left_score=score,
            count=row.get("count", 1),
        )


@dataclass
class _MatchupCounts:
    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        return self.wins + 0.5 * self.draws


def _logistic(value: float) -> float:
    clipped = min(35.0, max(-35.0, value))

    return 1.0 / (1.0 + math.exp(-clipped))


def _wilson_interval(successes: float, total: int, z: float) -> list[float]:
    if total <= 0:
        return [0.0, 1.0]
    proportion = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        (proportion * (1.0 - proportion) + z2 / (4.0 * total)) / total
    ) / denominator

    return [max(0.0, center - radius), min(1.0, center + radius)]


def _aggregate_games(
    rows: Iterable[PairwiseGameRow | Mapping[str, Any]],
) -> tuple[
    list[PolicyDeckPair],
    dict[tuple[PolicyDeckPair, PolicyDeckPair], _MatchupCounts],
]:
    matchups: dict[tuple[PolicyDeckPair, PolicyDeckPair], _MatchupCounts] = defaultdict(_MatchupCounts)
    pairs: set[PolicyDeckPair] = set()
    for value in rows:
        row = value if isinstance(value, PairwiseGameRow) else PairwiseGameRow.from_mapping(value)
        pairs.update((row.left, row.right))
        first, second = sorted((row.left, row.right), key=lambda pair: pair.pair_id)
        score = row.left_score if row.left == first else 1.0 - row.left_score
        counts = matchups[(first, second)]
        if score == 1.0:
            counts.wins += row.count
        elif score == 0.5:
            counts.draws += row.count
        else:
            counts.losses += row.count
    if len(pairs) < 2 or not matchups:
        raise EmpiricalOpponentError("pairwise ratings require at least two connected pairs")

    return sorted(pairs, key=lambda pair: pair.pair_id), dict(matchups)


def _require_connected(
    pairs: Sequence[PolicyDeckPair],
    matchups: Mapping[tuple[PolicyDeckPair, PolicyDeckPair], _MatchupCounts],
) -> None:
    neighbors: dict[PolicyDeckPair, set[PolicyDeckPair]] = defaultdict(set)
    for left, right in matchups:
        neighbors[left].add(right)
        neighbors[right].add(left)
    reached = {pairs[0]}
    pending = [pairs[0]]
    while pending:
        current = pending.pop()
        for neighbor in neighbors[current] - reached:
            reached.add(neighbor)
            pending.append(neighbor)
    if reached != set(pairs):
        raise EmpiricalOpponentError("pairwise rating graph must be connected")


def fit_frozen_panel_bt_index(
    rows: Iterable[PairwiseGameRow | Mapping[str, Any]],
    *,
    strength_bucket_by_pair_id: Mapping[str, str] | None = None,
    baseline_index: float = 1000.0,
    index_scale: float = 400.0,
    ridge: float = 0.01,
    confidence_z: float = 1.959963984540054,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Fit a connected, panel-relative Bradley-Terry competence index."""

    pairs, matchups = _aggregate_games(rows)
    _require_connected(pairs, matchups)
    baseline = float(baseline_index)
    scale = _positive_finite(index_scale, name="index_scale")
    regularization = _positive_finite(ridge, name="ridge")
    z_value = _positive_finite(confidence_z, name="confidence_z")
    iterations = _positive_int(max_iterations, name="max_iterations")
    convergence_tolerance = _positive_finite(tolerance, name="tolerance")
    if not math.isfinite(baseline):
        raise EmpiricalOpponentError("baseline_index must be finite")
    index = {pair: position for position, pair in enumerate(pairs)}
    theta = np.zeros(len(pairs), dtype=np.float64)

    def information_and_gradient(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        information = np.eye(len(pairs), dtype=np.float64) * regularization
        gradient = -regularization * values
        for (left, right), counts in matchups.items():
            left_index = index[left]
            right_index = index[right]
            probability = _logistic(float(values[left_index] - values[right_index]))
            residual = counts.score - counts.games * probability
            gradient[left_index] += residual
            gradient[right_index] -= residual
            weight = counts.games * probability * (1.0 - probability)
            information[left_index, left_index] += weight
            information[right_index, right_index] += weight
            information[left_index, right_index] -= weight
            information[right_index, left_index] -= weight

        return information, gradient

    converged = False
    for iteration in range(1, iterations + 1):
        information, gradient = information_and_gradient(theta)
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError as exc:
            raise EmpiricalOpponentError("pairwise rating information matrix is singular") from exc
        theta += step
        theta -= theta.mean()
        if float(np.max(np.abs(step))) <= convergence_tolerance:
            converged = True
            break
    if not converged:
        raise EmpiricalOpponentError("pairwise rating fit did not converge")
    information, _gradient = information_and_gradient(theta)
    covariance = np.linalg.inv(information)
    centering = np.eye(len(pairs)) - np.ones((len(pairs), len(pairs))) / len(pairs)
    centered_covariance = centering @ covariance @ centering.T
    index_per_logit = scale / math.log(10.0)
    indices = baseline + theta * index_per_logit
    standard_errors = np.sqrt(np.maximum(0.0, np.diag(centered_covariance))) * index_per_logit
    games_by_pair: Counter[PolicyDeckPair] = Counter()
    for (left, right), counts in matchups.items():
        games_by_pair[left] += counts.games
        games_by_pair[right] += counts.games
    buckets = strength_bucket_by_pair_id or {}
    index_rows = []
    for pair, index_value, standard_error in zip(pairs, indices, standard_errors, strict=True):
        bucket = buckets.get(pair.pair_id, "unassigned")
        bucket = _text(bucket, name=f"strength bucket for {pair.pair_id}")
        index_rows.append({
            **pair.to_dict(),
            "pairwise_game_count": games_by_pair[pair],
            "frozen_panel_bt_index": float(index_value),
            "confidence_interval_95": [
                float(index_value - z_value * standard_error),
                float(index_value + z_value * standard_error),
            ],
            "standard_error": float(standard_error),
            "strength_bucket": bucket,
        })
    index_rows.sort(key=lambda row: (-row["frozen_panel_bt_index"], row["pair_id"]))
    matchup_rows = []
    for (first, second), counts in sorted(
        matchups.items(),
        key=lambda item: (item[0][0].pair_id, item[0][1].pair_id),
    ):
        for left, right, wins, losses in (
            (first, second, counts.wins, counts.losses),
            (second, first, counts.losses, counts.wins),
        ):
            decided = wins + losses
            matchup_rows.append({
                "left": left.to_dict(),
                "right": right.to_dict(),
                "pairwise_game_count": counts.games,
                "wins": wins,
                "draws": counts.draws,
                "losses": losses,
                "decided_win_rate": wins / decided if decided else None,
                "point_rate": (wins + 0.5 * counts.draws) / counts.games,
                "point_rate_wilson95": _wilson_interval(
                    wins + 0.5 * counts.draws,
                    counts.games,
                    z_value,
                ),
                "direct_ordered_payoff": (wins - losses) / counts.games,
            })

    return {
        "schema_version": FROZEN_PANEL_BT_INDEX_SCHEMA_VERSION,
        "method": {
            "name": "ridge_bradley_terry_frozen_panel_index",
            "draw_score_per_side": 0.5,
            "baseline_index": baseline,
            "index_scale": scale,
            "ridge": regularization,
            "iterations": iteration,
            "confidence_interval": "normal_95_from_centered_regularized_observed_information",
            "strength_bucket_source": "caller_supplied_independent_panel_or_unassigned",
            "external_ladder_rating_equivalence": False,
            "kaggle_score_prediction": False,
        },
        "indices": index_rows,
        "matchups": matchup_rows,
    }
