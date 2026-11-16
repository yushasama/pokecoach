"""Pure research-selection gate for Q-guided policy candidates.

This module consumes only aggregate metrics and explicit audit receipts. It
does not load teacher shards, checkpoints, simulator state, or runtime policy
code. Missing instrumentation is therefore visible as ``not_instrumented``
rather than being silently converted to zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class CheckStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_INSTRUMENTED = "not_instrumented"


class ResearchVerdict(str, Enum):
    SELECTED = "selected_research_candidate"
    REJECTED = "rejected"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class EffectEstimate:
    """Candidate-minus-reference effect with a predeclared confidence interval."""

    mean: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        values = (self.mean, self.lower, self.upper)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise TypeError("effect estimate must be numeric")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("effect estimate must be finite")
        if not self.lower <= self.mean <= self.upper:
            raise ValueError("effect confidence interval must contain its mean")


@dataclass(frozen=True)
class SafetyEvidence:
    illegal_actions: int | None
    crashes: int | None
    timeouts: int | None

    def __post_init__(self) -> None:
        for name in ("illegal_actions", "crashes", "timeouts"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")


@dataclass(frozen=True)
class AuditEvidence:
    leakage: CheckStatus
    game_disjoint: CheckStatus
    signature_disjoint: CheckStatus
    source_disjoint: CheckStatus
    identities_bound: CheckStatus
    equal_budget: CheckStatus

    def __post_init__(self) -> None:
        for name in (
            "leakage",
            "game_disjoint",
            "signature_disjoint",
            "source_disjoint",
            "identities_bound",
            "equal_budget",
        ):
            if not isinstance(getattr(self, name), CheckStatus):
                raise TypeError(f"{name} must be a CheckStatus")


@dataclass(frozen=True)
class OfflineEvidence:
    heldout_rows: int
    heldout_groups: int
    mean_teacher_regret_delta: EffectEstimate

    def __post_init__(self) -> None:
        for name in ("heldout_rows", "heldout_groups"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class GameplayEvidence:
    games_per_participant: int
    win_rate_delta: EffectEstimate
    severe_regression_cells: int

    def __post_init__(self) -> None:
        for name in ("games_per_participant", "severe_regression_cells"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class AcceptanceThresholds:
    minimum_heldout_rows: int = 32
    minimum_heldout_groups: int = 16
    minimum_games_per_participant: int = 256

    def __post_init__(self) -> None:
        for name in (
            "minimum_heldout_rows",
            "minimum_heldout_groups",
            "minimum_games_per_participant",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class AcceptanceDecision:
    verdict: ResearchVerdict
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class QMetricSummary:
    rows: int
    actions: int
    q_mean_squared_error: float
    gaussian_nll: float
    interval_95_coverage: float
    q_head_ranking_accuracy: float
    q_head_mean_teacher_regret: float


@dataclass(frozen=True)
class PolicyMetricSummary:
    rows: int
    candidate_ranking_accuracy: float
    reference_ranking_accuracy: float
    candidate_mean_teacher_regret: float
    reference_mean_teacher_regret: float
    mean_teacher_regret_delta: float


def decide_research_candidate(
    *,
    safety: SafetyEvidence,
    audit: AuditEvidence,
    offline: OfflineEvidence,
    gameplay: GameplayEvidence,
    thresholds: AcceptanceThresholds = AcceptanceThresholds(),
) -> AcceptanceDecision:
    """Return a tri-state research verdict without authorizing promotion."""

    failures: list[str] = []
    missing: list[str] = []
    for name in ("illegal_actions", "crashes", "timeouts"):
        value = getattr(safety, name)
        if value is None:
            missing.append(f"{name} is not instrumented")
        elif value > 0:
            failures.append(f"{name}={value}")
    for name in (
        "leakage",
        "game_disjoint",
        "signature_disjoint",
        "source_disjoint",
        "identities_bound",
        "equal_budget",
    ):
        status = getattr(audit, name)
        if status is CheckStatus.FAILED:
            failures.append(f"{name} check failed")
        elif status is CheckStatus.NOT_INSTRUMENTED:
            missing.append(f"{name} check is not instrumented")
    if gameplay.severe_regression_cells > 0:
        failures.append(f"severe_regression_cells={gameplay.severe_regression_cells}")
    if offline.mean_teacher_regret_delta.lower > 0.0:
        failures.append("heldout teacher regret is clearly worse than the reference")
    if gameplay.win_rate_delta.upper < 0.0:
        failures.append("equal-budget gameplay is clearly worse than the reference")
    if failures:
        return AcceptanceDecision(ResearchVerdict.REJECTED, tuple(failures))
    if offline.heldout_rows < thresholds.minimum_heldout_rows:
        missing.append(
            f"heldout_rows={offline.heldout_rows} < {thresholds.minimum_heldout_rows}"
        )
    if offline.heldout_groups < thresholds.minimum_heldout_groups:
        missing.append(
            f"heldout_groups={offline.heldout_groups} < {thresholds.minimum_heldout_groups}"
        )
    if gameplay.games_per_participant < thresholds.minimum_games_per_participant:
        missing.append(
            "games_per_participant="
            f"{gameplay.games_per_participant} < {thresholds.minimum_games_per_participant}"
        )
    if offline.mean_teacher_regret_delta.upper >= 0.0:
        missing.append("heldout teacher-regret improvement is not statistically resolved")
    if gameplay.win_rate_delta.lower <= 0.0:
        missing.append("equal-budget gameplay improvement is not statistically resolved")
    if missing:
        return AcceptanceDecision(ResearchVerdict.AMBIGUOUS, tuple(missing))

    return AcceptanceDecision(
        ResearchVerdict.SELECTED,
        (
            "all safety, leakage, identity, split, support, offline, and equal-budget gameplay gates passed",
        ),
    )


def score_q_predictions(
    teacher_means: Sequence[Sequence[float]],
    predicted_means: Sequence[Sequence[float]],
    predicted_variances: Sequence[Sequence[float]],
    teacher_variances: Sequence[Sequence[float]] | None = None,
    *,
    variance_floor: float = 1e-6,
) -> QMetricSummary:
    """Measure Q-head quality separately from exported-policy ranking."""

    if not math.isfinite(variance_floor) or variance_floor <= 0.0:
        raise ValueError("variance_floor must be finite and positive")
    rows = _aligned_rows(teacher_means, predicted_means, predicted_variances)
    if teacher_variances is None:
        teacher_variances = tuple(tuple(0.0 for _ in row) for row in teacher_means)
    _aligned_rows(teacher_means, teacher_variances)
    squared_errors: list[float] = []
    nll_values: list[float] = []
    covered = 0
    correct = 0
    regrets: list[float] = []
    actions = 0
    for targets, predictions, prediction_variance, target_variance in zip(
        teacher_means,
        predicted_means,
        predicted_variances,
        teacher_variances,
        strict=True,
    ):
        target_values = _finite_row(targets, "teacher means", lower=0.0, upper=1.0)
        prediction_values = _finite_row(predictions, "predicted means")
        prediction_var_values = _finite_row(prediction_variance, "predicted variances", lower=0.0)
        target_var_values = _finite_row(target_variance, "teacher variances", lower=0.0)
        target_best = _best_indices(target_values)
        predicted_best = _deterministic_best(prediction_values)
        correct += int(predicted_best in target_best)
        regrets.append(max(target_values) - target_values[predicted_best])
        for target, prediction, predicted_var, target_var in zip(
            target_values,
            prediction_values,
            prediction_var_values,
            target_var_values,
            strict=True,
        ):
            error = prediction - target
            total_variance = max(predicted_var + target_var, variance_floor)
            squared_errors.append(error * error)
            nll_values.append(0.5 * (math.log(2.0 * math.pi * total_variance) + error * error / total_variance))
            covered += int(abs(error) <= 1.96 * math.sqrt(total_variance))
            actions += 1

    return QMetricSummary(
        rows=rows,
        actions=actions,
        q_mean_squared_error=_mean(squared_errors),
        gaussian_nll=_mean(nll_values),
        interval_95_coverage=covered / actions,
        q_head_ranking_accuracy=correct / rows,
        q_head_mean_teacher_regret=_mean(regrets),
    )


def score_policy_rankings(
    teacher_means: Sequence[Sequence[float]],
    candidate_scores: Sequence[Sequence[float]],
    reference_scores: Sequence[Sequence[float]],
) -> PolicyMetricSummary:
    """Compare candidate and reference choices against identical teacher rows."""

    rows = _aligned_rows(teacher_means, candidate_scores, reference_scores)
    candidate_correct = 0
    reference_correct = 0
    candidate_regrets: list[float] = []
    reference_regrets: list[float] = []
    for targets, candidate, reference in zip(
        teacher_means,
        candidate_scores,
        reference_scores,
        strict=True,
    ):
        target_values = _finite_row(targets, "teacher means", lower=0.0, upper=1.0)
        candidate_values = _finite_row(candidate, "candidate scores")
        reference_values = _finite_row(reference, "reference scores")
        target_best = _best_indices(target_values)
        candidate_best = _deterministic_best(candidate_values)
        reference_best = _deterministic_best(reference_values)
        candidate_correct += int(candidate_best in target_best)
        reference_correct += int(reference_best in target_best)
        candidate_regrets.append(max(target_values) - target_values[candidate_best])
        reference_regrets.append(max(target_values) - target_values[reference_best])
    candidate_mean_regret = _mean(candidate_regrets)
    reference_mean_regret = _mean(reference_regrets)

    return PolicyMetricSummary(
        rows=rows,
        candidate_ranking_accuracy=candidate_correct / rows,
        reference_ranking_accuracy=reference_correct / rows,
        candidate_mean_teacher_regret=candidate_mean_regret,
        reference_mean_teacher_regret=reference_mean_regret,
        mean_teacher_regret_delta=candidate_mean_regret - reference_mean_regret,
    )


def _aligned_rows(reference: Sequence[Sequence[float]], *others: Sequence[Sequence[float]]) -> int:
    rows = len(reference)
    if rows == 0 or any(len(value) != rows for value in others):
        raise ValueError("metric inputs must contain the same non-zero row count")
    for row_index, reference_row in enumerate(reference):
        if len(reference_row) == 0:
            raise ValueError("metric rows must contain at least one action")
        if any(len(value[row_index]) != len(reference_row) for value in others):
            raise ValueError("metric rows must have aligned action counts")

    return rows


def _finite_row(
    values: Sequence[float],
    name: str,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> tuple[float, ...]:
    if any(isinstance(value, bool) for value in values):
        raise TypeError(f"{name} must be numeric and not boolean")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must be finite")
    if lower is not None and any(value < lower for value in result):
        raise ValueError(f"{name} must be at least {lower}")
    if upper is not None and any(value > upper for value in result):
        raise ValueError(f"{name} must be at most {upper}")

    return result


def _best_indices(values: Sequence[float]) -> frozenset[int]:
    best = max(values)

    return frozenset(index for index, value in enumerate(values) if value == best)


def _deterministic_best(values: Sequence[float]) -> int:
    return min(range(len(values)), key=lambda index: (-values[index], index))


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)
