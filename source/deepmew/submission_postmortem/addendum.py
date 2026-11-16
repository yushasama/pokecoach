"""Statistical addendum for an existing submission postmortem run.

The analyses in this module consume the already-sanitized episode and decision
signals emitted by :mod:`deepmew.submission_postmortem.pipeline`.  They do not
read raw observations, hidden zones, legal alternatives, critics, or Q-values.
Every effect is retrospective and observational unless explicitly described as
a sampling calculation.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ADDENDUM_SCHEMA_VERSION = "deepmew.submission_postmortem.statistical_addendum.v0"
DEFAULT_ALPHA = 0.05
DEFAULT_BOOTSTRAP_SAMPLES = 4_000
DEFAULT_PERMUTATION_SAMPLES = 40_000
DEFAULT_RANDOM_SEED = 54788723
# Four equivalent games is the primary, explicit beta-binomial sensitivity
# assumption. It is not estimated from this sparse cohort because marginal
# likelihood collapses toward complete pooling when most matchup groups are
# singletons.
DEFAULT_MATCHUP_PRIOR_STRENGTH = 4.0
DEFAULT_RIDGE_PENALTY = 2.0
NORMAL_95 = 1.959963984540054
FLOAT_EPSILON = 1e-12


@dataclass(frozen=True)
class AddendumArtifacts:
    """Materialized tables and the combined machine-readable payload."""

    payload: Mapping[str, Any]
    tables: Mapping[str, Sequence[Mapping[str, Any]]]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    output = float(value)

    return output if math.isfinite(output) else None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        output = int(value)
    except (TypeError, ValueError):
        return None

    return output


def _episode_key(value: Mapping[str, Any]) -> tuple[int, int, str]:
    ordinal = _integer(value.get("ordinal"))
    episode_id = _integer(value.get("episode_id"))

    return (
        0 if ordinal is not None else 1,
        ordinal if ordinal is not None else episode_id if episode_id is not None else 0,
        str(value.get("episode_id") or ""),
    )


def _decisive_summaries(
    episode_summaries: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return sorted(
        (
            row
            for row in episode_summaries
            if not row.get("excluded_from_opponent_attribution")
            and row.get("result") in {"win", "loss"}
        ),
        key=_episode_key,
    )


def _matchup_name(summary: Mapping[str, Any]) -> str:
    deck = _mapping(summary.get("opponent_deck"))

    return str(
        summary.get("opponent_label")
        or deck.get("stable_hash")
        or summary.get("opponent_team")
        or "unknown"
    )


def wilson_interval(successes: int, trials: int, *, alpha: float = DEFAULT_ALPHA) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""

    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("successes and trials must define a valid binomial count")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if trials == 0:
        return (0.0, 1.0)
    if alpha != DEFAULT_ALPHA:
        raise ValueError("only the preregistered 95% interval is supported")
    z = NORMAL_95
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    ) / denominator

    return (max(0.0, center - half_width), min(1.0, center + half_width))


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    max_iterations = 300
    convergence = 3e-14
    floor = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    result = d
    for iteration in range(1, max_iterations + 1):
        doubled = 2 * iteration
        coefficient = iteration * (b - iteration) * x / ((qam + doubled) * (a + doubled))
        d = 1.0 + coefficient * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + coefficient / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        result *= d * c
        coefficient = -(a + iteration) * (qab + iteration) * x / (
            (a + doubled) * (qap + doubled)
        )
        d = 1.0 + coefficient * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + coefficient / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= convergence:
            break

    return result


def regularized_beta_cdf(x: float, a: float, b: float) -> float:
    """Evaluate the regularized incomplete beta function without SciPy."""

    if a <= 0.0 or b <= 0.0:
        raise ValueError("beta shape parameters must be positive")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        value = front * _beta_continued_fraction(a, b, x) / a
    else:
        value = 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b

    return min(1.0, max(0.0, value))


def beta_quantile(probability: float, a: float, b: float) -> float:
    """Invert a beta CDF by stable bisection."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    if probability == 0.0:
        return 0.0
    if probability == 1.0:
        return 1.0
    low = 0.0
    high = 1.0
    for _ in range(100):
        midpoint = (low + high) / 2.0
        if regularized_beta_cdf(midpoint, a, b) < probability:
            low = midpoint
        else:
            high = midpoint

    return (low + high) / 2.0


def clopper_pearson_interval(
    successes: int,
    trials: int,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[float, float]:
    """Return an exact two-sided Clopper-Pearson binomial interval."""

    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("successes and trials must define a valid binomial count")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if trials == 0:
        return (0.0, 1.0)
    lower = 0.0 if successes == 0 else beta_quantile(alpha / 2.0, successes, trials - successes + 1)
    upper = 1.0 if successes == trials else beta_quantile(1.0 - alpha / 2.0, successes + 1, trials - successes)

    return (lower, upper)


def _performance_elo(win_rate: float) -> float | None:
    if not 0.0 < win_rate < 1.0:
        return None

    return 400.0 * math.log10(win_rate / (1.0 - win_rate))


def _binomial_probability(successes: int, trials: int, probability: float) -> float:
    return math.comb(trials, successes) * probability ** successes * (1.0 - probability) ** (trials - successes)


def exact_binomial_two_sided_p(successes: int, trials: int, *, null_probability: float) -> float:
    observed = _binomial_probability(successes, trials, null_probability)
    probability = sum(
        _binomial_probability(value, trials, null_probability)
        for value in range(trials + 1)
        if _binomial_probability(value, trials, null_probability) <= observed + 1e-15
    )

    return min(1.0, probability)


def rating_stability(wins: int, losses: int) -> dict[str, Any]:
    """Quantify finite-sample uncertainty in a decisive win/loss record."""

    games = wins + losses
    if games <= 0:
        raise ValueError("rating stability requires at least one decisive game")
    win_rate = wins / games
    wilson_low, wilson_high = wilson_interval(wins, games)
    exact_low, exact_high = clopper_pearson_interval(wins, games)
    posterior_a = wins + 0.5
    posterior_b = losses + 0.5
    thresholds = (0.45, 0.50, 0.55, 0.60)
    exceedance = [
        {
            "threshold": threshold,
            "posterior_probability_above": 1.0 - regularized_beta_cdf(threshold, posterior_a, posterior_b),
            "prior": "Jeffreys Beta(0.5, 0.5)",
        }
        for threshold in thresholds
    ]
    plug_in_standard_error = math.sqrt(win_rate * (1.0 - win_rate) / games)
    null_standard_error = math.sqrt(0.25 / games)
    elo = _performance_elo(win_rate)

    return {
        "schema_version": ADDENDUM_SCHEMA_VERSION,
        "decisive_games": games,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "wilson_95": {"low": wilson_low, "high": wilson_high},
        "clopper_pearson_95": {"low": exact_low, "high": exact_high},
        "jeffreys_posterior_95": {
            "low": beta_quantile(DEFAULT_ALPHA / 2.0, posterior_a, posterior_b),
            "high": beta_quantile(1.0 - DEFAULT_ALPHA / 2.0, posterior_a, posterior_b),
            "prior": "Beta(0.5, 0.5)",
        },
        "performance_elo_delta_vs_average_field": elo,
        "performance_elo_95_from_wilson": {
            "low": _performance_elo(wilson_low),
            "high": _performance_elo(wilson_high),
        },
        "expected_binomial_variance": {
            "win_count_plug_in": games * win_rate * (1.0 - win_rate),
            "win_count_at_50_percent": games * 0.25,
            "win_rate_standard_error_plug_in": plug_in_standard_error,
            "win_rate_standard_error_at_50_percent": null_standard_error,
        },
        "posterior_threshold_probabilities": exceedance,
        "test_against_50_percent": {
            "exact_two_sided_p": exact_binomial_two_sided_p(wins, games, null_probability=0.5),
            "wilson_contains_50_percent": wilson_low <= 0.5 <= wilson_high,
            "verdict": "statistically_unresolved_not_equivalent",
            "interpretation": (
                "This sample does not distinguish the observed win rate from 50%; "
                "failure to reject is not proof that the rates are equivalent."
            ),
        },
        "elo_scope": (
            "field-relative performance delta implied by a stationary binary win rate; "
            "not Kaggle's historical ladder rating and not opponent-strength adjusted"
        ),
    }


def _window_counts(outcomes: Sequence[int], end: int, window_size: int) -> tuple[int, int] | None:
    if end + 1 < window_size:
        return None
    values = outcomes[end - window_size + 1:end + 1]

    return (sum(values), len(values))


def _rolling_fields(outcomes: Sequence[int], end: int, window_size: int) -> dict[str, Any]:
    prefix = f"rolling_{window_size}"
    counts = _window_counts(outcomes, end, window_size)
    if counts is None:
        return {
            f"{prefix}_games": None,
            f"{prefix}_wins": None,
            f"{prefix}_win_rate": None,
            f"{prefix}_wilson_low": None,
            f"{prefix}_wilson_high": None,
        }
    wins, games = counts
    low, high = wilson_interval(wins, games)

    return {
        f"{prefix}_games": games,
        f"{prefix}_wins": wins,
        f"{prefix}_win_rate": wins / games,
        f"{prefix}_wilson_low": low,
        f"{prefix}_wilson_high": high,
    }


def _maximum_split_statistic(values: np.ndarray, minimum_segment: int) -> tuple[float, int]:
    total = float(values.sum())
    cumulative = np.cumsum(values)
    best_value = -1.0
    best_split = minimum_segment
    for split in range(minimum_segment, len(values) - minimum_segment + 1):
        left = float(cumulative[split - 1]) / split
        right = (total - float(cumulative[split - 1])) / (len(values) - split)
        difference = abs(left - right)
        if difference > best_value:
            best_value = difference
            best_split = split

    return best_value, best_split


def _phase_indices(length: int) -> list[np.ndarray]:
    groups = [[], [], []]
    for index in range(length):
        groups[min(2, (index * 3) // max(1, length))].append(index)

    return [np.asarray(group, dtype=np.int64) for group in groups]


def _binomial_log_likelihood(wins: float, games: int, probability: float) -> float:
    if games <= 0:
        return 0.0
    if probability <= 0.0:
        return 0.0 if wins == 0.0 else -math.inf
    if probability >= 1.0:
        return 0.0 if wins == games else -math.inf

    return wins * math.log(probability) + (games - wins) * math.log1p(-probability)


def _best_low_interval(
    values: np.ndarray,
    *,
    minimum_length: int = 5,
    maximum_length: int = 20,
) -> tuple[float, int, int]:
    total_wins = float(values.sum())
    total_games = len(values)
    pooled_rate = total_wins / total_games
    pooled_log_likelihood = _binomial_log_likelihood(total_wins, total_games, pooled_rate)
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    best_statistic = 0.0
    best_start = 0
    best_end = min(total_games, minimum_length)
    for start in range(total_games):
        for end in range(start + minimum_length, min(total_games, start + maximum_length) + 1):
            inside_games = end - start
            outside_games = total_games - inside_games
            if outside_games <= 0:
                continue
            inside_wins = float(cumulative[end] - cumulative[start])
            outside_wins = total_wins - inside_wins
            inside_rate = inside_wins / inside_games
            outside_rate = outside_wins / outside_games
            if inside_rate >= outside_rate:
                continue
            alternative = (
                _binomial_log_likelihood(inside_wins, inside_games, inside_rate)
                + _binomial_log_likelihood(outside_wins, outside_games, outside_rate)
            )
            statistic = 2.0 * (alternative - pooled_log_likelihood)
            if statistic > best_statistic:
                best_statistic = statistic
                best_start = start
                best_end = end

    return best_statistic, best_start, best_end


def rolling_performance(
    episode_summaries: Sequence[Mapping[str, Any]],
    *,
    permutation_samples: int = DEFAULT_PERMUTATION_SAMPLES,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build chronological windows and selection-adjusted trend diagnostics."""

    if permutation_samples <= 0:
        raise ValueError("permutation_samples must be positive")
    ordered = _decisive_summaries(episode_summaries)
    outcomes = [1 if row.get("result") == "win" else 0 for row in ordered]
    rows: list[dict[str, Any]] = []
    cumulative_wins = 0
    for index, (summary, outcome) in enumerate(zip(ordered, outcomes)):
        cumulative_wins += outcome
        cumulative_games = index + 1
        cumulative_low, cumulative_high = wilson_interval(cumulative_wins, cumulative_games)
        row = {
            "episode_id": summary.get("episode_id"),
            "ordinal": summary.get("ordinal"),
            "result": summary.get("result"),
            "opponent_deck": _matchup_name(summary),
            "matchup": _matchup_name(summary),
            "first_second": summary.get("first_second_proxy"),
            "game_length_turns": summary.get("turn_count_proxy"),
            "cumulative_games": cumulative_games,
            "cumulative_wins": cumulative_wins,
            "cumulative_win_rate": cumulative_wins / cumulative_games,
            "cumulative_wilson_low": cumulative_low,
            "cumulative_wilson_high": cumulative_high,
            "provenance": "posthoc_deck_metadata_and_actor_visible_trajectory",
        }
        for window_size in (5, 10, 20):
            row.update(_rolling_fields(outcomes, index, window_size))
        rows.append(row)

    if len(outcomes) < 10:
        return rows, {
            "status": "insufficient_games",
            "significant_change_detected": False,
            "interpretation": "At least 10 decisive games are required for the preregistered scan.",
        }
    values = np.asarray(outcomes, dtype=np.float64)
    minimum_segment = max(5, len(values) // 6)
    observed_statistic, split = _maximum_split_statistic(values, minimum_segment)
    low_interval_statistic, low_interval_start, low_interval_end = _best_low_interval(values)
    rng = np.random.default_rng(random_seed)
    exceedances = 0
    low_interval_exceedances = 0
    phase_groups = _phase_indices(len(values))
    observed_phase_rates = [float(values[group].mean()) for group in phase_groups]
    observed_minimum_phase = min(observed_phase_rates)
    minimum_phase_exceedances = 0
    for _ in range(permutation_samples):
        permuted = rng.permutation(values)
        statistic, _split = _maximum_split_statistic(permuted, minimum_segment)
        exceedances += statistic >= observed_statistic - FLOAT_EPSILON
        permuted_low_statistic, _start, _end = _best_low_interval(permuted)
        low_interval_exceedances += permuted_low_statistic >= low_interval_statistic - FLOAT_EPSILON
        permuted_minimum = min(float(permuted[group].mean()) for group in phase_groups)
        minimum_phase_exceedances += permuted_minimum <= observed_minimum_phase + FLOAT_EPSILON
    scan_p = (exceedances + 1) / (permutation_samples + 1)
    phase_p = (minimum_phase_exceedances + 1) / (permutation_samples + 1)
    low_interval_p = (low_interval_exceedances + 1) / (permutation_samples + 1)
    phase_names = ("early", "middle", "late")
    phase_rows = []
    for name, group, rate in zip(phase_names, phase_groups, observed_phase_rates):
        wins = int(values[group].sum())
        low, high = wilson_interval(wins, len(group))
        phase_rows.append({
            "phase": name,
            "games": len(group),
            "wins": wins,
            "losses": len(group) - wins,
            "win_rate": rate,
            "wilson_low": low,
            "wilson_high": high,
        })

    return rows, {
        "status": "complete",
        "permutation_samples": permutation_samples,
        "random_seed": random_seed,
        "minimum_segment_games": minimum_segment,
        "maximum_change_point": {
            "split_after_game": split,
            "absolute_win_rate_difference": observed_statistic,
            "familywise_permutation_p": scan_p,
            "monte_carlo_standard_error": math.sqrt(scan_p * (1.0 - scan_p) / (permutation_samples + 1)),
        },
        "selected_low_interval": {
            "start_game": low_interval_start + 1,
            "end_game": low_interval_end,
            "games": low_interval_end - low_interval_start,
            "wins": int(values[low_interval_start:low_interval_end].sum()),
            "losses": int((low_interval_end - low_interval_start) - values[low_interval_start:low_interval_end].sum()),
            "episode_ids": [
                str(ordered[index].get("episode_id"))
                for index in range(low_interval_start, low_interval_end)
            ],
            "one_sided_likelihood_ratio": low_interval_statistic,
            "scan_adjusted_permutation_p": low_interval_p,
            "monte_carlo_standard_error": math.sqrt(
                low_interval_p * (1.0 - low_interval_p) / (permutation_samples + 1)
            ),
        },
        "chronological_thirds": phase_rows,
        "weakest_third_selection_adjusted_p": phase_p,
        "weakest_third_monte_carlo_standard_error": math.sqrt(
            phase_p * (1.0 - phase_p) / (permutation_samples + 1)
        ),
        "significant_change_detected": scan_p < DEFAULT_ALPHA,
        "collapse_exceeds_random_variance": low_interval_p < DEFAULT_ALPHA,
        "interpretation": (
            "Permutation tests condition on the observed total wins and adjust for selecting "
            "the largest permanent split, lowest contiguous interval, or weakest chronological "
            "third. Exchangeability excludes opponent-mix and initiative drift, so neither is adjusted."
        ),
    }


def exposure_normalized_matchups(
    episode_summaries: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Normalize matchup losses for exposure and shrink small groups."""

    decisive = _decisive_summaries(episode_summaries)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for summary in decisive:
        grouped[_matchup_name(summary)].append(summary)
    total_losses = sum(row.get("result") == "loss" for row in decisive)
    total_games = len(decisive)
    global_loss_rate = total_losses / total_games if total_games else 0.0
    prior_strength = DEFAULT_MATCHUP_PRIOR_STRENGTH
    prior_a = max(FLOAT_EPSILON, global_loss_rate * prior_strength)
    prior_b = max(FLOAT_EPSILON, (1.0 - global_loss_rate) * prior_strength)
    rows: list[dict[str, Any]] = []
    for matchup, values in grouped.items():
        losses = sum(row.get("result") == "loss" for row in values)
        wins = sum(row.get("result") == "win" for row in values)
        games = wins + losses
        loss_rate = losses / games
        expected_losses = games * global_loss_rate
        wilson_low, wilson_high = wilson_interval(losses, games)
        posterior_a = prior_a + losses
        posterior_b = prior_b + wins
        posterior_mean = posterior_a / (posterior_a + posterior_b)
        sensitivity = []
        for strength in (2.0, 4.0, 8.0):
            sensitivity_a = losses + global_loss_rate * strength
            sensitivity_b = wins + (1.0 - global_loss_rate) * strength
            sensitivity.append({
                "prior_strength": strength,
                "posterior_mean_loss_rate": sensitivity_a / (sensitivity_a + sensitivity_b),
                "credible_low": beta_quantile(DEFAULT_ALPHA / 2.0, sensitivity_a, sensitivity_b),
                "credible_high": beta_quantile(1.0 - DEFAULT_ALPHA / 2.0, sensitivity_a, sensitivity_b),
            })
        rows.append({
            "matchup": matchup,
            "games": games,
            "wins": wins,
            "observed_losses": losses,
            "expected_losses_from_exposure": expected_losses,
            "observed_minus_expected_losses": losses - expected_losses,
            "loss_ratio": losses / expected_losses if expected_losses > 0.0 else None,
            "observed_loss_rate": loss_rate,
            "wilson_loss_rate_low": wilson_low,
            "wilson_loss_rate_high": wilson_high,
            "bayesian_shrunk_loss_rate": posterior_mean,
            "bayesian_credible_low": beta_quantile(DEFAULT_ALPHA / 2.0, posterior_a, posterior_b),
            "bayesian_credible_high": beta_quantile(1.0 - DEFAULT_ALPHA / 2.0, posterior_a, posterior_b),
            "posterior_probability_worse_than_field": 1.0 - regularized_beta_cdf(
                global_loss_rate,
                posterior_a,
                posterior_b,
            ),
            "shrinkage_sensitivity": sensitivity,
            "loss_replay_ids": [str(row.get("episode_id")) for row in values if row.get("result") == "loss"],
            "provenance": "posthoc_deck_metadata_and_observed_outcome",
        })
    rows.sort(
        key=lambda row: (
            -float(row["bayesian_shrunk_loss_rate"]),
            -int(row["games"]),
            str(row["matchup"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["posterior_rank"] = rank

    return rows, {
        "global_loss_rate": global_loss_rate,
        "global_losses": total_losses,
        "global_games": total_games,
        "bayesian_shrinkage_prior": {
            "family": "Beta",
            "mean": global_loss_rate,
            "mean_source": "plug_in_pooled_observed_loss_rate",
            "strength_equivalent_games": prior_strength,
            "alpha": prior_a,
            "beta": prior_b,
            "status": "plug_in_empirical_bayes_mean_fixed_strength_sensitivity",
            "credible_interval_scope": (
                "conditional on the pooled plug-in mean and fixed prior strength; "
                "global-rate uncertainty is not propagated"
            ),
        },
        "ranking_status": "posterior_point_estimate_only_intervals_overlap",
        "interpretation": (
            "Ranks use a beta-binomial empirical-Bayes posterior mean. Overlapping credible "
            "intervals mean the data do not establish a unique true ordering."
        ),
    }


def _fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    row_one = a + b
    row_two = c + d
    column_one = a + c
    total = row_one + row_two
    lower = max(0, column_one - row_two)
    upper = min(row_one, column_one)

    def probability(value: int) -> float:
        return math.comb(row_one, value) * math.comb(row_two, column_one - value) / math.comb(total, column_one)

    observed = probability(a)

    return min(1.0, sum(probability(value) for value in range(lower, upper + 1) if probability(value) <= observed + 1e-15))


def first_player_effect(episode_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Quantify the observed loss association for going second versus first."""

    decisive = _decisive_summaries(episode_summaries)
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in decisive:
        position = str(row.get("first_second_proxy") or "unavailable")
        groups[position].append(row)
    first = groups.get("first", [])
    second = groups.get("second", [])
    first_losses = sum(row.get("result") == "loss" for row in first)
    first_wins = sum(row.get("result") == "win" for row in first)
    second_losses = sum(row.get("result") == "loss" for row in second)
    second_wins = sum(row.get("result") == "win" for row in second)
    if not first or not second:
        return {"status": "not_identifiable", "reason": "both first and second groups require evidence"}
    risk_first = first_losses / len(first)
    risk_second = second_losses / len(second)
    first_interval = wilson_interval(first_losses, len(first))
    second_interval = wilson_interval(second_losses, len(second))
    corrected = any(value == 0 for value in (first_losses, first_wins, second_losses, second_wins))
    a, b, c, d = (float(second_losses), float(second_wins), float(first_losses), float(first_wins))
    if corrected:
        a, b, c, d = (value + 0.5 for value in (a, b, c, d))
    odds_ratio = (a * d) / (b * c)
    odds_se = math.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
    relative_risk = risk_second / risk_first if risk_first > 0.0 else None
    if second_losses > 0 and first_losses > 0 and relative_risk is not None:
        risk_se = math.sqrt(
            1.0 / second_losses
            - 1.0 / len(second)
            + 1.0 / first_losses
            - 1.0 / len(first)
        )
        relative_risk_interval = (
            math.exp(math.log(relative_risk) - NORMAL_95 * risk_se),
            math.exp(math.log(relative_risk) + NORMAL_95 * risk_se),
        )
    else:
        relative_risk_interval = (None, None)
    descriptive_excess_losses = len(second) * (risk_second - risk_first)
    risk_difference = risk_second - risk_first
    risk_difference_se = math.sqrt(
        risk_second * (1.0 - risk_second) / len(second)
        + risk_first * (1.0 - risk_first) / len(first)
    )

    return {
        "status": "complete",
        "outcome": "loss",
        "reference": "going_first",
        "going_first": {
            "games": len(first),
            "wins": first_wins,
            "losses": first_losses,
            "loss_risk": risk_first,
            "wilson_95": {"low": first_interval[0], "high": first_interval[1]},
        },
        "going_second": {
            "games": len(second),
            "wins": second_wins,
            "losses": second_losses,
            "loss_risk": risk_second,
            "wilson_95": {"low": second_interval[0], "high": second_interval[1]},
        },
        "risk_difference": risk_difference,
        "risk_difference_95": {
            "low": max(-1.0, risk_difference - NORMAL_95 * risk_difference_se),
            "high": min(1.0, risk_difference + NORMAL_95 * risk_difference_se),
            "method": "unpooled_normal_approximation",
        },
        "relative_risk": relative_risk,
        "relative_risk_95": {"low": relative_risk_interval[0], "high": relative_risk_interval[1]},
        "relative_risk_interval_method": "Katz_log_approximation",
        "odds_ratio": odds_ratio,
        "odds_ratio_95": {
            "low": math.exp(math.log(odds_ratio) - NORMAL_95 * odds_se),
            "high": math.exp(math.log(odds_ratio) + NORMAL_95 * odds_se),
        },
        "odds_ratio_interval_method": "Wald_log_approximation",
        "haldane_anscombe_correction_used": corrected,
        "fisher_exact_two_sided_p": _fisher_exact_two_sided(second_losses, second_wins, first_losses, first_wins),
        "descriptive_excess_losses_vs_first_rate": descriptive_excess_losses,
        "alone_explains_record": None,
        "alone_explains_record_status": "not_established_and_not_causally_identifiable",
        "interpretation": (
            "Position is associated with outcome but is confounded with matchup and chronology. "
            "The interval includes weak or null effects, so going second alone is not established "
            "as an explanation of the record."
        ),
        "provenance": "actor_visible_first_player_and_observed_outcome",
    }


def _bootstrap_mean_interval(
    values: Sequence[float],
    *,
    random_seed: int,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    finite_values = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite_values.size == 0:
        return {"n": 0, "mean": None, "low": None, "high": None}
    if finite_values.size == 1:
        value = float(finite_values[0])

        return {"n": 1, "mean": value, "low": value, "high": value}
    rng = np.random.default_rng(random_seed)
    indices = rng.integers(0, finite_values.size, size=(samples, finite_values.size))
    means = finite_values[indices].mean(axis=1)

    return {
        "n": int(finite_values.size),
        "mean": float(finite_values.mean()),
        "low": float(np.quantile(means, DEFAULT_ALPHA / 2.0)),
        "high": float(np.quantile(means, 1.0 - DEFAULT_ALPHA / 2.0)),
        "method": "episode_bootstrap_percentile",
        "bootstrap_samples": samples,
    }


def _actor_turn_snapshots(
    episode_id: str,
    actor_seat: int,
    decision_rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    rows = [
        row
        for row in decision_rows
        if row.get("role") == "actor"
        and str(row.get("episode_id")) == episode_id
        and _integer(row.get("seat")) == actor_seat
        and (_integer(row.get("turn")) or 0) > 0
        and _mapping(row.get("state_snapshot")).get("available") is True
    ]
    rows.sort(key=lambda row: (_integer(row.get("request_step")) or 0, str(row.get("decision_id") or "")))
    last_by_turn: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        turn = _integer(row.get("turn"))
        if turn is not None:
            last_by_turn[turn] = row

    return [last_by_turn[turn] for turn in sorted(last_by_turn)]


def _negative_metrics(snapshot: Mapping[str, Any]) -> tuple[str, ...]:
    output = []
    for key in ("prize_lead", "board_delta", "energy_delta"):
        value = _finite(snapshot.get(key))
        if value is not None and value < 0.0:
            output.append(key)

    return tuple(output)


def _first_negative_turn(rows: Sequence[Mapping[str, Any]], metric: str) -> int | None:
    for row in rows:
        snapshot = _mapping(row.get("state_snapshot"))
        value = _finite(snapshot.get(metric))
        if value is not None and value < 0.0:
            return _integer(row.get("turn"))

    return None


def _persistent_disadvantage_turn(rows: Sequence[Mapping[str, Any]]) -> int | None:
    predicates = []
    for row in rows:
        snapshot = _mapping(row.get("state_snapshot"))
        prize = _finite(snapshot.get("prize_lead"))
        board = _finite(snapshot.get("board_delta"))
        energy = _finite(snapshot.get("energy_delta"))
        predicates.append(
            prize is not None
            and prize < 0.0
            and (
                (board is not None and board < 0.0)
                or (energy is not None and energy < 0.0)
            )
        )
    for index, predicate in enumerate(predicates):
        if predicate and all(predicates[index:]):
            return _integer(rows[index].get("turn"))

    return None


def _turn_phase(turn: int | None) -> str:
    if turn is None:
        return "unavailable"
    if turn <= 4:
        return "early"
    if turn <= 8:
        return "middle"

    return "late"


def loss_timeline(
    episode_summaries: Sequence[Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
    *,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Measure observable loss clocks while refusing irreversibility claims."""

    losses = [row for row in _decisive_summaries(episode_summaries) if row.get("result") == "loss"]
    output: list[dict[str, Any]] = []
    for summary in losses:
        episode_id = str(summary.get("episode_id"))
        actor_seat = _integer(summary.get("actor_seat"))
        if actor_seat is None:
            continue
        rows = _actor_turn_snapshots(episode_id, actor_seat, decision_rows)
        first_by_metric = {
            metric: _first_negative_turn(rows, metric)
            for metric in ("prize_lead", "board_delta", "energy_delta")
        }
        candidates = [value for value in first_by_metric.values() if value is not None]
        first_observable = min(candidates) if candidates else None
        first_evidence: tuple[str, ...] = ()
        if first_observable is not None:
            first_row = next((row for row in rows if _integer(row.get("turn")) == first_observable), None)
            first_evidence = _negative_metrics(_mapping(_mapping(first_row).get("state_snapshot")))
        persistent = _persistent_disadvantage_turn(rows)
        terminal_snapshot = _mapping(summary.get("terminal_state_snapshot"))
        terminal_turn = _integer(terminal_snapshot.get("turn")) or _integer(summary.get("turn_count_proxy"))
        proxy_turn = persistent if persistent is not None else terminal_turn
        output.append({
            "episode_id": episode_id,
            "matchup": _matchup_name(summary),
            "first_second": summary.get("first_second_proxy"),
            "terminal_failure_class": summary.get("terminal_failure_class"),
            "first_observable_disadvantage_turn": first_observable,
            "first_observable_disadvantage_metrics": list(first_evidence),
            "first_prize_deficit_turn": first_by_metric["prize_lead"],
            "first_board_deficit_turn": first_by_metric["board_delta"],
            "first_energy_deficit_turn": first_by_metric["energy_delta"],
            "first_persistent_observable_disadvantage_turn": persistent,
            "persistent_proxy_definition": (
                "from this actor-observed turn onward every retained turn ends with prize_lead<0 "
                "and either board_delta<0 or energy_delta<0"
            ),
            "first_irreversible_disadvantage": None,
            "first_irreversible_disadvantage_status": "not_identifiable_from_realized_trajectory",
            "effectively_lost_state": None,
            "effectively_lost_state_status": "not_identifiable_without_counterfactual_value_or_branching",
            "terminal_loss_turn": terminal_turn,
            "turns_first_observable_to_terminal": terminal_turn - first_observable
            if terminal_turn is not None and first_observable is not None
            else None,
            "turns_persistent_proxy_to_terminal": terminal_turn - persistent
            if terminal_turn is not None and persistent is not None
            else None,
            "collapse_timing_proxy": _turn_phase(proxy_turn),
            "representative_turn": proxy_turn,
            "provenance": "actor_visible_hindsight_proxy",
        })
    observable_gaps = [
        float(row["turns_first_observable_to_terminal"])
        for row in output
        if _finite(row.get("turns_first_observable_to_terminal")) is not None
    ]
    persistent_gaps = [
        float(row["turns_persistent_proxy_to_terminal"])
        for row in output
        if _finite(row.get("turns_persistent_proxy_to_terminal")) is not None
    ]
    phase_counts = Counter(str(row.get("collapse_timing_proxy")) for row in output)
    phase_rows = []
    for phase in ("early", "middle", "late", "unavailable"):
        count = phase_counts[phase]
        if count == 0:
            continue
        low, high = wilson_interval(count, len(output))
        phase_rows.append({
            "phase": phase,
            "losses": count,
            "loss_share": count / len(output),
            "wilson_low": low,
            "wilson_high": high,
        })

    return output, {
        "losses": len(output),
        "first_irreversible_disadvantage_status": "not_identifiable",
        "recoverable_to_effectively_lost_turns_status": "not_identifiable",
        "first_observable_to_terminal": _bootstrap_mean_interval(
            observable_gaps,
            random_seed=random_seed + 101,
        ),
        "persistent_proxy_to_terminal": _bootstrap_mean_interval(
            persistent_gaps,
            random_seed=random_seed + 102,
        ),
        "persistent_proxy_available_games": len(persistent_gaps),
        "collapse_timing_proxy": phase_rows,
        "interpretation": (
            "Observable deficits and hindsight persistence are timing descriptors only. "
            "A realized replay cannot prove recoverability, irreversibility, or a winning alternative."
        ),
    }


def _histogram_rows(
    values: Sequence[float],
    edges: Sequence[float],
    *,
    metric: str,
) -> list[dict[str, Any]]:
    finite_values = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    rows = []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        if index == len(edges) - 2:
            count = int(np.sum((finite_values >= low) & (finite_values <= high)))
        else:
            count = int(np.sum((finite_values >= low) & (finite_values < high)))
        rows.append({
            "metric": metric,
            "bin_low": low,
            "bin_high": high,
            "count": count,
            "share": count / len(finite_values) if len(finite_values) else None,
        })

    return rows


def _decision_phase(row: Mapping[str, Any]) -> str:
    turn = _integer(row.get("turn"))
    if turn is None:
        return "unavailable"
    if turn <= 0:
        return "setup"

    return _turn_phase(turn)


def confidence_diagnostics(
    episode_summaries: Sequence[Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
    *,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Describe action-selection confidence without treating it as P(win)."""

    decisive = _decisive_summaries(episode_summaries)
    summary_by_key = {
        (str(row.get("episode_id")), _integer(row.get("actor_seat"))): row
        for row in decisive
    }
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    probabilities: list[float] = []
    entropies: list[float] = []
    for row in decision_rows:
        key = (str(row.get("episode_id")), _integer(row.get("seat")))
        if row.get("role") != "actor" or key not in summary_by_key or key[1] is None:
            continue
        rating = _mapping(row.get("rating"))
        probability = _finite(rating.get("chosen_probability"))
        entropy = _finite(rating.get("entropy"))
        if probability is None and entropy is None:
            continue
        grouped[(key[0], int(key[1]))].append(row)
        if probability is not None:
            probabilities.append(probability)
        if entropy is not None:
            entropies.append(entropy)
    game_rows: list[dict[str, Any]] = []
    phase_game_rows: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        summary = summary_by_key[key]
        game_probabilities = [
            value
            for value in (_finite(_mapping(row.get("rating")).get("chosen_probability")) for row in rows)
            if value is not None
        ]
        game_entropies = [
            value
            for value in (_finite(_mapping(row.get("rating")).get("entropy")) for row in rows)
            if value is not None
        ]
        game_rows.append({
            "episode_id": key[0],
            "result": summary.get("result"),
            "matchup": _matchup_name(summary),
            "first_second": summary.get("first_second_proxy"),
            "decisions": len(rows),
            "mean_chosen_probability": sum(game_probabilities) / len(game_probabilities) if game_probabilities else None,
            "mean_entropy": sum(game_entropies) / len(game_entropies) if game_entropies else None,
            "provenance": "exact_runtime_actor_policy_rescore",
        })
        phase_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            phase_groups[_decision_phase(row)].append(row)
        for phase, phase_rows in phase_groups.items():
            phase_probabilities = [
                value
                for value in (
                    _finite(_mapping(row.get("rating")).get("chosen_probability"))
                    for row in phase_rows
                )
                if value is not None
            ]
            phase_entropies = [
                value
                for value in (
                    _finite(_mapping(row.get("rating")).get("entropy"))
                    for row in phase_rows
                )
                if value is not None
            ]
            phase_game_rows.append({
                "episode_id": key[0],
                "result": summary.get("result"),
                "matchup": _matchup_name(summary),
                "phase": phase,
                "mean_chosen_probability": sum(phase_probabilities) / len(phase_probabilities)
                if phase_probabilities
                else None,
                "mean_entropy": sum(phase_entropies) / len(phase_entropies) if phase_entropies else None,
            })

    def grouped_rows(values: Sequence[Mapping[str, Any]], key_name: str) -> list[dict[str, Any]]:
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in values:
            groups[str(row.get(key_name) or "unavailable")].append(row)
        output = []
        for index, (name, members) in enumerate(sorted(groups.items())):
            confidence_values = [
                float(value)
                for value in (_finite(row.get("mean_chosen_probability")) for row in members)
                if value is not None
            ]
            entropy_values = [
                float(value)
                for value in (_finite(row.get("mean_entropy")) for row in members)
                if value is not None
            ]
            wins = sum(row.get("result") == "win" for row in members)
            decisive_count = sum(row.get("result") in {"win", "loss"} for row in members)
            win_interval = wilson_interval(wins, decisive_count) if decisive_count else (0.0, 1.0)
            output.append({
                key_name: name,
                "games": len(members),
                "mean_chosen_probability": _bootstrap_mean_interval(
                    confidence_values,
                    random_seed=random_seed + 200 + index,
                ),
                "mean_entropy": _bootstrap_mean_interval(
                    entropy_values,
                    random_seed=random_seed + 300 + index,
                ),
                "observed_game_win_rate": wins / decisive_count if decisive_count else None,
                "observed_game_win_rate_wilson_low": win_interval[0],
                "observed_game_win_rate_wilson_high": win_interval[1],
            })

        return output

    by_outcome = grouped_rows(game_rows, "result")
    by_matchup = grouped_rows(game_rows, "matchup")
    by_phase = grouped_rows(phase_game_rows, "phase")
    histogram_tables = {
        "confidence_histogram": _histogram_rows(
            probabilities,
            (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0),
            metric="chosen_probability",
        ),
        "entropy_histogram": _histogram_rows(
            entropies,
            (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0),
            metric="entropy",
        ),
        "confidence_games": game_rows,
        "confidence_by_outcome": by_outcome,
        "confidence_by_matchup": by_matchup,
        "confidence_by_phase": by_phase,
    }

    return {
        "probability_calibration_status": "not_identifiable",
        "probability_calibration_reason": (
            "chosen_probability is the submitted actor's action-selection probability, not a "
            "predicted probability of winning; ECE, Brier calibration, and a global/local "
            "win-calibration verdict would be category errors"
        ),
        "decision_count_with_confidence": len(probabilities),
        "decision_count_with_entropy": len(entropies),
        "episode_is_inference_unit": True,
        "by_outcome": by_outcome,
        "by_matchup": by_matchup,
        "by_phase": by_phase,
        "interpretation": (
            "The tables test descriptive action-confidence associations with episode-level "
            "bootstrap intervals. They do not establish overconfidence or a blunder."
        ),
    }, histogram_tables


def _minimum_prize_lead_by_episode(
    decision_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], float]:
    output: dict[tuple[str, int], float] = {}
    for row in decision_rows:
        if row.get("role") != "actor":
            continue
        seat = _integer(row.get("seat"))
        if seat is None:
            continue
        snapshot = _mapping(row.get("state_snapshot"))
        prize_lead = _finite(snapshot.get("prize_lead"))
        if prize_lead is None:
            continue
        key = (str(row.get("episode_id")), seat)
        output[key] = min(output.get(key, prize_lead), prize_lead)

    return output


def _game_feature_rows(
    episode_summaries: Sequence[Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    decisive = _decisive_summaries(episode_summaries)
    minimum_prize_lead = _minimum_prize_lead_by_episode(decision_rows)
    rows = []
    for summary in decisive:
        turn_count = _finite(summary.get("turn_count_proxy"))
        turn_denominator = max(1.0, turn_count or 1.0)
        actor_counts = _mapping(_mapping(summary.get("action_counts")).get("actor"))
        tempo = _mapping(summary.get("tempo"))
        tag_evidence = _mapping(summary.get("tag_evidence"))
        no_evolution_value = _mapping(tag_evidence.get("no_focal_evolution")).get("value")
        key = (str(summary.get("episode_id")), _integer(summary.get("actor_seat")) or 0)
        minimum_lead = minimum_prize_lead.get(key)
        if no_evolution_value is True:
            focal_evolution_realized = 0.0
        elif no_evolution_value is False:
            focal_evolution_realized = 1.0
        else:
            focal_evolution_realized = None
        rows.append({
            "episode_id": str(summary.get("episode_id")),
            "loss": 1.0 if summary.get("result") == "loss" else 0.0,
            "matchup": _matchup_name(summary),
            "going_second": 1.0 if summary.get("first_second_proxy") == "second" else 0.0,
            "game_length": turn_count,
            "first_attack_turn": _finite(tempo.get("first_attack_turn")),
            "attachment_rate": (_finite(actor_counts.get("ATTACH")) or 0.0) / turn_denominator,
            "attack_density": (_finite(actor_counts.get("ATTACK")) or 0.0) / turn_denominator,
            "plays_per_turn": (_finite(actor_counts.get("PLAY")) or 0.0) / turn_denominator,
            "maximum_prize_deficit": max(0.0, -minimum_lead) if minimum_lead is not None else None,
            "focal_evolution_realized": focal_evolution_realized,
        })

    return rows


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)

    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_ridge_logistic(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    penalty: float,
    max_iterations: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    coefficients = np.zeros(features.shape[1], dtype=np.float64)
    penalty_diagonal = np.ones(features.shape[1], dtype=np.float64)
    penalty_diagonal[0] = 0.0
    for _ in range(max_iterations):
        probabilities = _sigmoid(features @ coefficients)
        weights = np.maximum(probabilities * (1.0 - probabilities), 1e-8)
        gradient = features.T @ (probabilities - targets) + penalty * penalty_diagonal * coefficients
        hessian = (features.T * weights) @ features + penalty * np.diag(penalty_diagonal)
        step = np.linalg.solve(hessian, gradient)
        coefficients -= step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    probabilities = _sigmoid(features @ coefficients)
    weights = np.maximum(probabilities * (1.0 - probabilities), 1e-8)
    hessian = (features.T * weights) @ features + penalty * np.diag(penalty_diagonal)

    return coefficients, hessian


def _auc(targets: np.ndarray, probabilities: np.ndarray) -> float | None:
    positives = np.where(targets == 1.0)[0]
    negatives = np.where(targets == 0.0)[0]
    if len(positives) == 0 or len(negatives) == 0:
        return None
    favorable = 0.0
    for positive in positives:
        favorable += float(np.sum(probabilities[positive] > probabilities[negatives]))
        favorable += 0.5 * float(np.sum(probabilities[positive] == probabilities[negatives]))

    return favorable / (len(positives) * len(negatives))


def _prediction_metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    clipped = np.clip(probabilities, 1e-9, 1.0 - 1e-9)

    return {
        "log_loss": float(-np.mean(targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped))),
        "brier_score": float(np.mean((probabilities - targets) ** 2)),
        "roc_auc": _auc(targets, probabilities),
    }


def _leave_one_out_predictions(
    design: np.ndarray,
    targets: np.ndarray,
    *,
    penalty: float,
) -> np.ndarray:
    predictions = np.empty(len(targets), dtype=np.float64)
    for held_out in range(len(targets)):
        mask = np.ones(len(targets), dtype=bool)
        mask[held_out] = False
        fold_coefficients, _fold_hessian = _fit_ridge_logistic(
            design[mask],
            targets[mask],
            penalty=penalty,
        )
        predictions[held_out] = float(_sigmoid(design[held_out:held_out + 1] @ fold_coefficients)[0])

    return predictions


def _row_log_loss(targets: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-9, 1.0 - 1e-9)

    return -(targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped))


def observational_loss_model(
    episode_summaries: Sequence[Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
    *,
    penalty: float = DEFAULT_RIDGE_PENALTY,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit a small-N ridge logistic retrospective association model."""

    game_rows = _game_feature_rows(episode_summaries, decision_rows)
    if len(game_rows) < 10:
        return ({"status": "insufficient_games"}, [], game_rows)
    matchup_counts = Counter(str(row["matchup"]) for row in game_rows)
    supported_matchups = sorted(name for name, count in matchup_counts.items() if count >= 3)
    continuous_names = (
        "game_length",
        "first_attack_turn",
        "attachment_rate",
        "attack_density",
        "plays_per_turn",
        "maximum_prize_deficit",
    )
    binary_names = ("going_second", "focal_evolution_realized")
    values_by_name: dict[str, np.ndarray] = {}
    scaling: dict[str, dict[str, Any]] = {}
    feature_names: list[str] = []
    for name in continuous_names:
        raw = np.asarray([
            np.nan if _finite(row.get(name)) is None else float(row[name])
            for row in game_rows
        ], dtype=np.float64)
        available = raw[np.isfinite(raw)]
        median = float(np.median(available)) if available.size else 0.0
        filled = np.where(np.isfinite(raw), raw, median)
        mean = float(filled.mean())
        standard_deviation = float(filled.std(ddof=0))
        if standard_deviation <= FLOAT_EPSILON:
            standard_deviation = 1.0
        values_by_name[name] = (filled - mean) / standard_deviation
        feature_names.append(name)
        scaling[name] = {
            "mean": mean,
            "standard_deviation": standard_deviation,
            "median_imputation": median,
            "available_games": int(available.size),
            "scope": "full_trajectory_posthoc" if name != "first_attack_turn" else "realized_timing",
            "effect_scale": "one_observed_standard_deviation",
        }
        missing = ~np.isfinite(raw)
        if bool(missing.any()):
            missing_name = f"{name}_missing"
            values_by_name[missing_name] = missing.astype(np.float64)
            feature_names.append(missing_name)
            scaling[missing_name] = {
                "mean": float(missing.mean()),
                "standard_deviation": None,
                "median_imputation": None,
                "available_games": len(game_rows),
                "scope": "evidence_availability",
                "effect_scale": "missing_vs_available",
            }
    for name in binary_names:
        raw = np.asarray([
            np.nan if _finite(row.get(name)) is None else float(row[name])
            for row in game_rows
        ], dtype=np.float64)
        available = raw[np.isfinite(raw)]
        median = float(np.median(available)) if available.size else 0.0
        values_by_name[name] = np.where(np.isfinite(raw), raw, median)
        feature_names.append(name)
        scaling[name] = {
            "mean": float(values_by_name[name].mean()),
            "standard_deviation": None,
            "median_imputation": median,
            "available_games": int(available.size),
            "scope": "pregame" if name == "going_second" else "full_trajectory_posthoc",
            "effect_scale": "one_vs_zero",
        }
        missing = ~np.isfinite(raw)
        if bool(missing.any()):
            missing_name = f"{name}_missing"
            values_by_name[missing_name] = missing.astype(np.float64)
            feature_names.append(missing_name)
            scaling[missing_name] = {
                "mean": float(missing.mean()),
                "standard_deviation": None,
                "median_imputation": None,
                "available_games": len(game_rows),
                "scope": "evidence_availability",
                "effect_scale": "missing_vs_available",
            }
    matchup_categories = [
        str(row["matchup"]) if str(row["matchup"]) in supported_matchups else "__other__"
        for row in game_rows
    ]
    matchup_category_counts = Counter(matchup_categories)
    if "__other__" in matchup_category_counts:
        reference_matchup = "__other__"
    elif supported_matchups:
        reference_matchup = max(supported_matchups, key=lambda name: (matchup_counts[name], name))
    else:
        reference_matchup = None
    for matchup in sorted(matchup_category_counts):
        if matchup == reference_matchup:
            continue
        name = f"matchup::{matchup}"
        values_by_name[name] = np.asarray([1.0 if category == matchup else 0.0 for category in matchup_categories])
        feature_names.append(name)
        scaling[name] = {
            "mean": float(values_by_name[name].mean()),
            "standard_deviation": None,
            "median_imputation": None,
            "available_games": int(matchup_category_counts[matchup]),
            "scope": "posthoc_matchup_metadata",
            "effect_scale": f"indicator_vs_{reference_matchup}",
        }
    design = np.column_stack([
        np.ones(len(game_rows), dtype=np.float64),
        *(values_by_name[name] for name in feature_names),
    ])
    targets = np.asarray([float(row["loss"]) for row in game_rows], dtype=np.float64)
    if len(np.unique(targets)) < 2:
        return ({
            "status": "not_identifiable_single_outcome",
            "games": len(game_rows),
            "loss_events": int(targets.sum()),
        }, [], game_rows)
    coefficients, _hessian = _fit_ridge_logistic(design, targets, penalty=penalty)
    rng = np.random.default_rng(random_seed + 401)
    coefficient_samples = []
    for _ in range(DEFAULT_BOOTSTRAP_SAMPLES):
        sample = rng.integers(0, len(game_rows), size=len(game_rows))
        if len(np.unique(targets[sample])) < 2:
            continue
        sample_coefficients, _sample_hessian = _fit_ridge_logistic(
            design[sample],
            targets[sample],
            penalty=penalty,
        )
        coefficient_samples.append(sample_coefficients)
    bootstrap_coefficients = np.asarray(coefficient_samples, dtype=np.float64)
    predictions = _leave_one_out_predictions(design, targets, penalty=penalty)
    full_row_log_loss = _row_log_loss(targets, predictions)
    effect_rows = []
    for index, name in enumerate(feature_names, start=1):
        coefficient = float(coefficients[index])
        samples = bootstrap_coefficients[:, index]
        standard_error = float(samples.std(ddof=1))
        low = float(np.quantile(samples, DEFAULT_ALPHA / 2.0))
        high = float(np.quantile(samples, 1.0 - DEFAULT_ALPHA / 2.0))
        reduced_design = np.delete(design, index, axis=1)
        reduced_predictions = _leave_one_out_predictions(reduced_design, targets, penalty=penalty)
        row_importance = _row_log_loss(targets, reduced_predictions) - full_row_log_loss
        importance_indices = rng.integers(0, len(game_rows), size=(DEFAULT_BOOTSTRAP_SAMPLES, len(game_rows)))
        importance_samples = row_importance[importance_indices].mean(axis=1)
        importance = float(row_importance.mean())
        effect_rows.append({
            "feature": name,
            "scope": scaling[name]["scope"],
            "effect_scale": scaling[name]["effect_scale"],
            "standardized_log_odds_effect": coefficient,
            "episode_bootstrap_standard_error": standard_error,
            "episode_bootstrap_ci_low": low,
            "episode_bootstrap_ci_high": high,
            "standardized_odds_ratio": math.exp(max(-30.0, min(30.0, coefficient))),
            "odds_ratio_ci_low": math.exp(max(-30.0, min(30.0, low))),
            "odds_ratio_ci_high": math.exp(max(-30.0, min(30.0, high))),
            "leave_one_episode_out_drop_log_loss_importance": importance,
            "importance_bootstrap_low": float(np.quantile(importance_samples, DEFAULT_ALPHA / 2.0)),
            "importance_bootstrap_high": float(np.quantile(importance_samples, 1.0 - DEFAULT_ALPHA / 2.0)),
            "inference_status": "exploratory_episode_bootstrap_fixed_feature_schema",
        })
    effect_rows.sort(
        key=lambda row: (
            -float(row.get("leave_one_episode_out_drop_log_loss_importance") or 0.0),
            str(row["feature"]),
        )
    )
    for rank, row in enumerate(effect_rows, start=1):
        row["importance_rank"] = rank

    metrics = _prediction_metrics(targets, predictions)
    bootstrap_metrics: dict[str, list[float]] = defaultdict(list)
    for _ in range(DEFAULT_BOOTSTRAP_SAMPLES):
        sample = rng.integers(0, len(game_rows), size=len(game_rows))
        sample_metrics = _prediction_metrics(targets[sample], predictions[sample])
        for name, value in sample_metrics.items():
            if value is not None and math.isfinite(float(value)):
                bootstrap_metrics[name].append(float(value))
    metric_intervals = {}
    for name, value in metrics.items():
        samples = bootstrap_metrics.get(name, [])
        metric_intervals[name] = {
            "estimate": value,
            "fixed_prediction_resample_low": float(np.quantile(samples, DEFAULT_ALPHA / 2.0)) if samples else None,
            "fixed_prediction_resample_high": float(np.quantile(samples, 1.0 - DEFAULT_ALPHA / 2.0)) if samples else None,
            "uncertainty_status": "naive_episode_resample_of_fixed_overlapping_LOEO_predictions",
        }
    prediction_rows = [
        {
            "episode_id": row["episode_id"],
            "observed_loss": int(target),
            "leave_one_episode_out_loss_probability": float(probability),
        }
        for row, target, probability in zip(game_rows, targets, predictions)
    ]

    return {
        "status": "exploratory_low_power",
        "model_kind": "ridge_logistic_retrospective_association",
        "games": len(game_rows),
        "loss_events": int(targets.sum()),
        "feature_count": len(feature_names),
        "ridge_penalty": penalty,
        "reference_matchup": reference_matchup,
        "matchup_category_counts": dict(sorted(matchup_category_counts.items())),
        "supported_matchup_minimum_games": 3,
        "feature_scaling": scaling,
        "leave_one_episode_out_metrics": metric_intervals,
        "performance_uncertainty_status": (
            "point estimates use overlapping leave-one-episode-out fits; fixed-prediction resampling "
            "does not capture full training-procedure uncertainty"
        ),
        "effects": effect_rows,
        "causal_status": "not_causal",
        "hazard_status": "not_a_time_to_event_hazard_model",
        "interpretation": (
            "This model ranks retrospective associations. Several variables summarize the full "
            "trajectory and are outcome-adjacent. Coefficient uncertainty uses an episode bootstrap "
            "conditional on a fixed feature schema; drop-log-loss importance uses overlapping "
            "leave-one-episode-out fits. Both remain exploratory and unstable at 41 games."
        ),
    }, prediction_rows, game_rows


def _gower_matrix(values: np.ndarray) -> np.ndarray:
    differences = np.abs(values[:, None, :] - values[None, :, :])

    return differences.mean(axis=2)


def _initial_medoids(distances: np.ndarray, cluster_count: int) -> list[int]:
    medoids = [int(np.argmax(distances.sum(axis=1)))]
    while len(medoids) < cluster_count:
        nearest = np.min(distances[:, medoids], axis=1)
        nearest[medoids] = -1.0
        medoids.append(int(np.argmax(nearest)))

    return medoids


def _fit_kmedoids(distances: np.ndarray, cluster_count: int) -> tuple[np.ndarray, list[int]]:
    medoids = _initial_medoids(distances, cluster_count)
    labels = np.zeros(len(distances), dtype=np.int64)
    for _ in range(100):
        labels = np.argmin(distances[:, medoids], axis=1)
        updated = []
        for cluster in range(cluster_count):
            members = np.where(labels == cluster)[0]
            if len(members) == 0:
                nearest = np.min(distances[:, medoids], axis=1)
                nearest[medoids] = -1.0
                updated.append(int(np.argmax(nearest)))
                continue
            within = distances[np.ix_(members, members)]
            updated.append(int(members[int(np.argmin(within.sum(axis=1)))]))
        if updated == medoids:
            break
        medoids = updated
    labels = np.argmin(distances[:, medoids], axis=1)

    return labels, medoids


def _silhouette_values(distances: np.ndarray, labels: np.ndarray) -> np.ndarray:
    output = np.zeros(len(labels), dtype=np.float64)
    unique = sorted(set(int(value) for value in labels))
    for index in range(len(labels)):
        own = labels[index]
        own_members = np.where(labels == own)[0]
        own_members = own_members[own_members != index]
        if len(own_members) == 0:
            output[index] = 0.0
            continue
        within = float(distances[index, own_members].mean())
        other_distances = [
            float(distances[index, np.where(labels == cluster)[0]].mean())
            for cluster in unique
            if cluster != own
        ]
        nearest_other = min(other_distances)
        denominator = max(within, nearest_other)
        output[index] = (nearest_other - within) / denominator if denominator > 0.0 else 0.0

    return output


def _cluster_stability_leave_one_out(
    distances: np.ndarray,
    base_labels: np.ndarray,
    cluster_count: int,
) -> np.ndarray:
    matches = np.zeros((len(base_labels), len(base_labels)), dtype=np.float64)
    evidence = np.zeros_like(matches)
    for omitted in range(len(base_labels)):
        retained = np.asarray([index for index in range(len(base_labels)) if index != omitted], dtype=np.int64)
        reduced_labels, _medoids = _fit_kmedoids(distances[np.ix_(retained, retained)], cluster_count)
        for left_position, left in enumerate(retained):
            for right_position in range(left_position + 1, len(retained)):
                right = retained[right_position]
                base_same = base_labels[left] == base_labels[right]
                reduced_same = reduced_labels[left_position] == reduced_labels[right_position]
                evidence[left, right] += 1.0
                evidence[right, left] += 1.0
                if base_same == reduced_same:
                    matches[left, right] += 1.0
                    matches[right, left] += 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        stability = np.divide(matches, evidence, out=np.zeros_like(matches), where=evidence > 0.0)

    return stability


def failure_clustering(
    episode_summaries: Sequence[Mapping[str, Any]],
    game_feature_rows: Sequence[Mapping[str, Any]],
    loss_timeline_rows: Sequence[Mapping[str, Any]],
    confidence_game_rows: Sequence[Mapping[str, Any]],
    *,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Cluster loss phenotypes with deterministic Gower-distance medoids."""

    summaries = {
        str(row.get("episode_id")): row
        for row in _decisive_summaries(episode_summaries)
        if row.get("result") == "loss"
    }
    timeline = {str(row.get("episode_id")): row for row in loss_timeline_rows}
    confidence = {str(row.get("episode_id")): row for row in confidence_game_rows}
    features_by_episode = {str(row.get("episode_id")): row for row in game_feature_rows if row.get("loss") == 1.0}
    episode_ids = sorted(summaries, key=lambda value: _episode_key(summaries[value]))
    if len(episode_ids) < 6:
        return ({"status": "insufficient_losses"}, [], [])
    feature_names = (
        "going_second",
        "game_length",
        "first_attack_turn",
        "attachment_rate",
        "attack_density",
        "plays_per_turn",
        "maximum_prize_deficit",
        "focal_evolution_realized",
        "prizes_taken",
        "maximum_prize_lead",
        "terminal_board_delta",
        "terminal_energy_delta",
        "zero_prize",
        "never_led",
        "boardout",
        "deckout",
        "mean_chosen_probability",
    )
    raw_rows = []
    severity_values = []
    for episode_id in episode_ids:
        summary = summaries[episode_id]
        game = _mapping(features_by_episode.get(episode_id))
        rating = _mapping(confidence.get(episode_id))
        tags = _mapping(summary.get("tag_evidence"))
        terminal = _mapping(summary.get("terminal_state_snapshot"))
        prizes = _mapping(summary.get("prizes"))
        starting_prizes = _finite(prizes.get("starting_prizes")) or 0.0
        prizes_taken = _finite(prizes.get("taken")) or 0.0
        severity = 1.0 - prizes_taken / starting_prizes if starting_prizes > 0.0 else None
        severity_values.append(severity)
        raw_rows.append({
            "going_second": game.get("going_second"),
            "game_length": game.get("game_length"),
            "first_attack_turn": game.get("first_attack_turn"),
            "attachment_rate": game.get("attachment_rate"),
            "attack_density": game.get("attack_density"),
            "plays_per_turn": game.get("plays_per_turn"),
            "maximum_prize_deficit": game.get("maximum_prize_deficit"),
            "focal_evolution_realized": game.get("focal_evolution_realized"),
            "prizes_taken": prizes_taken,
            "maximum_prize_lead": _finite(_mapping(summary.get("tempo")).get("maximum_prize_lead")),
            "terminal_board_delta": _finite(terminal.get("board_delta")),
            "terminal_energy_delta": _finite(terminal.get("energy_delta")),
            "zero_prize": 1.0 if _mapping(tags.get("zero_prize")).get("value") is True else 0.0,
            "never_led": 1.0 if _mapping(tags.get("never_led")).get("value") is True else 0.0,
            "boardout": 1.0 if _mapping(tags.get("boardout")).get("value") is True else 0.0,
            "deckout": 1.0 if _mapping(tags.get("deckout")).get("value") is True else 0.0,
            "mean_chosen_probability": rating.get("mean_chosen_probability"),
        })
    matrix = np.empty((len(raw_rows), len(feature_names)), dtype=np.float64)
    scaling = {}
    for column, name in enumerate(feature_names):
        raw = np.asarray([
            np.nan if _finite(row.get(name)) is None else float(row[name])
            for row in raw_rows
        ], dtype=np.float64)
        finite_values = raw[np.isfinite(raw)]
        median = float(np.median(finite_values)) if finite_values.size else 0.0
        filled = np.where(np.isfinite(raw), raw, median)
        minimum = float(filled.min())
        maximum = float(filled.max())
        feature_range = maximum - minimum
        matrix[:, column] = (filled - minimum) / feature_range if feature_range > FLOAT_EPSILON else 0.0
        scaling[name] = {
            "minimum": minimum,
            "maximum": maximum,
            "median_imputation": median,
            "available_losses": int(finite_values.size),
        }
    distances = _gower_matrix(matrix)
    candidates = []
    for cluster_count in range(2, min(5, len(episode_ids) // 3) + 1):
        labels, medoids = _fit_kmedoids(distances, cluster_count)
        sizes = Counter(int(value) for value in labels)
        if min(sizes.values()) < 3:
            continue
        silhouettes = _silhouette_values(distances, labels)
        candidates.append((float(silhouettes.mean()), cluster_count, labels, medoids, silhouettes))
    if not candidates:
        return ({
            "status": "no_stable_partition",
            "losses": len(episode_ids),
            "reason": "no k=2..5 partition retained at least three losses per cluster",
        }, [], [])
    mean_silhouette, cluster_count, labels, medoids, silhouettes = max(candidates, key=lambda value: (value[0], -value[1]))
    stability = _cluster_stability_leave_one_out(distances, labels, cluster_count)
    global_feature_mean = matrix.mean(axis=0)
    cluster_rows = []
    membership_rows = []
    for cluster in range(cluster_count):
        members = np.where(labels == cluster)[0]
        cluster_mean = matrix[members].mean(axis=0)
        deviations = sorted(
            (
                (abs(float(cluster_mean[index] - global_feature_mean[index])), name, float(cluster_mean[index] - global_feature_mean[index]))
                for index, name in enumerate(feature_names)
            ),
            reverse=True,
        )
        label_parts = [
            f"{'high' if direction >= 0.0 else 'low'}_{name}"
            for _magnitude, name, direction in deviations[:2]
        ]
        member_ids = [episode_ids[index] for index in members]
        medoid_index = medoids[cluster]
        ordered_representatives = sorted(members, key=lambda index: (distances[index, medoid_index], episode_ids[index]))
        representative_ids = [episode_ids[index] for index in ordered_representatives[:3]]
        within_pairs = [
            stability[left, right]
            for position, left in enumerate(members)
            for right in members[position + 1:]
        ]
        cluster_confidence = float(np.mean(within_pairs)) if within_pairs else None
        cluster_severity = [
            float(severity_values[index])
            for index in members
            if severity_values[index] is not None
        ]
        frequency_low, frequency_high = wilson_interval(len(members), len(episode_ids))
        severity_interval = _bootstrap_mean_interval(
            cluster_severity,
            random_seed=random_seed + 500 + cluster,
        )
        representative_turns = [
            _mapping(timeline.get(episode_id)).get("representative_turn")
            for episode_id in representative_ids
        ]
        cluster_rows.append({
            "cluster_id": cluster,
            "label": " + ".join(label_parts),
            "losses": len(members),
            "loss_frequency": len(members) / len(episode_ids),
            "frequency_wilson_low": frequency_low,
            "frequency_wilson_high": frequency_high,
            "mean_silhouette": float(silhouettes[members].mean()),
            "leave_one_episode_out_coassignment_confidence": cluster_confidence,
            "severity_no_prize_progress_index": severity_interval,
            "representative_replay_ids": representative_ids,
            "representative_turns": representative_turns,
            "matchup_counts": dict(sorted(Counter(_matchup_name(summaries[episode_id]) for episode_id in member_ids).items())),
            "confidence_status": "exploratory_resampling_proxy",
        })
        for index in members:
            membership_rows.append({
                "episode_id": episode_ids[index],
                "cluster_id": cluster,
                "silhouette": float(silhouettes[index]),
                "distance_to_medoid": float(distances[index, medoid_index]),
                "representative_turn": _mapping(timeline.get(episode_ids[index])).get("representative_turn"),
                "matchup": _matchup_name(summaries[episode_ids[index]]),
            })
    cluster_rows.sort(key=lambda row: (-int(row["losses"]), int(row["cluster_id"])))

    return {
        "status": "exploratory" if mean_silhouette >= 0.15 else "low_separation",
        "method": "Gower_distance_k_medoids",
        "selection": "highest_mean_silhouette_with_minimum_cluster_size_3",
        "losses": len(episode_ids),
        "cluster_count": cluster_count,
        "mean_silhouette": mean_silhouette,
        "feature_scaling": scaling,
        "confidence_definition": "leave-one-episode-out pairwise co-assignment agreement",
        "severity_definition": "one minus prizes taken divided by starting prizes",
        "clusters": cluster_rows,
        "interpretation": (
            "Clusters are repeated observed phenotypes, not causes or policy modes. With only "
            "22 losses, membership and labels remain exploratory."
        ),
    }, cluster_rows, membership_rows


def _signal_evidence_rows(
    episode_summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    decisive = _decisive_summaries(episode_summaries)
    signal_names = sorted({
        str(name)
        for summary in decisive
        for name in _mapping(summary.get("tag_evidence"))
    })
    rows = []
    for signal in signal_names:
        present = [
            summary
            for summary in decisive
            if _mapping(_mapping(summary.get("tag_evidence")).get(signal)).get("value") is True
        ]
        losses = [summary for summary in present if summary.get("result") == "loss"]
        wins = [summary for summary in present if summary.get("result") == "win"]
        if not losses:
            continue
        low, high = wilson_interval(len(losses), len(present))
        rows.append({
            "signal": signal,
            "games_present": len(present),
            "losses_present": len(losses),
            "wins_present": len(wins),
            "loss_rate": len(losses) / len(present),
            "loss_rate_wilson_low": low,
            "loss_rate_wilson_high": high,
            "loss_replay_ids": [str(summary.get("episode_id")) for summary in losses],
        })

    return rows


def _planning_confidence(support: int, interval_width: float) -> str:
    if support >= 8 and interval_width <= 0.5:
        return "high_descriptive_low_causal"
    if support >= 4:
        return "moderate_descriptive_low_causal"

    return "low_descriptive_low_causal"


def engineering_priorities(
    episode_summaries: Sequence[Mapping[str, Any]],
    matchup_rows: Sequence[Mapping[str, Any]],
    initiative: Mapping[str, Any],
    loss_timeline_rows: Sequence[Mapping[str, Any]],
    cluster_rows: Sequence[Mapping[str, Any]],
    confidence_payload: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate evidence-bound tasks without fabricating expected Elo gains."""

    decisive = _decisive_summaries(episode_summaries)
    losses = [row for row in decisive if row.get("result") == "loss"]
    global_loss_rate = len(losses) / len(decisive)
    candidates: list[dict[str, Any]] = []

    def add_candidate(
        *,
        issue: str,
        action: str,
        replay_ids: Sequence[str],
        support_games: int,
        observed_losses: int,
        interval_low: float,
        interval_high: float,
        cost_low: float,
        cost_high: float,
        evidence: str,
    ) -> None:
        unique_ids = list(dict.fromkeys(str(value) for value in replay_ids if value))
        cost_midpoint = (cost_low + cost_high) / 2.0
        expected_from_global = support_games * global_loss_rate
        observational_excess = max(0.0, observed_losses - expected_from_global)
        priority_score = (observational_excess + 0.25 * math.sqrt(max(1, observed_losses))) / max(0.25, cost_midpoint)
        candidates.append({
            "issue": issue,
            "engineering_task": action,
            "supporting_evidence": evidence,
            "replay_ids": unique_ids,
            "observed_loss_exposure": observed_losses,
            "support_games": support_games,
            "loss_rate_interval_low": interval_low,
            "loss_rate_interval_high": interval_high,
            "expected_impact": (
                f"Observed opportunity is bounded above by {observed_losses} associated losses; "
                "recoverable losses and realized Elo are unknown."
            ),
            "estimated_implementation_cost_weeks_low": cost_low,
            "estimated_implementation_cost_weeks_high": cost_high,
            "cost_status": "planning_judgment_not_measured",
            "confidence": _planning_confidence(support_games, interval_high - interval_low),
            "expected_elo_gain_per_engineering_week": None,
            "expected_elo_gain_status": "not_identifiable_from_observational_replays",
            "observational_opportunity_per_planning_week": priority_score,
        })

    for signal in _signal_evidence_rows(decisive):
        add_candidate(
            issue=f"Loss signal: {signal['signal']}",
            action=(
                f"Build an actor-visible regression slice and targeted evaluation for `{signal['signal']}` "
                "before changing policy behavior."
            ),
            replay_ids=signal["loss_replay_ids"],
            support_games=int(signal["games_present"]),
            observed_losses=int(signal["losses_present"]),
            interval_low=float(signal["loss_rate_wilson_low"]),
            interval_high=float(signal["loss_rate_wilson_high"]),
            cost_low=0.5,
            cost_high=1.0,
            evidence=(
                f"{signal['losses_present']}/{signal['games_present']} observed games with the signal were losses; "
                f"95% Wilson loss-rate interval [{signal['loss_rate_wilson_low']:.3f}, "
                f"{signal['loss_rate_wilson_high']:.3f}]."
            ),
        )
    for matchup in matchup_rows:
        observed_losses = int(matchup.get("observed_losses") or 0)
        if observed_losses == 0:
            continue
        add_candidate(
            issue=f"Matchup exposure: {matchup['matchup']}",
            action=(
                f"Collect a balanced first/second evaluation slice against `{matchup['matchup']}` and "
                "audit setup/resource trajectories before any matchup-specific policy edit."
            ),
            replay_ids=matchup.get("loss_replay_ids") or (),
            support_games=int(matchup["games"]),
            observed_losses=observed_losses,
            interval_low=float(matchup["wilson_loss_rate_low"]),
            interval_high=float(matchup["wilson_loss_rate_high"]),
            cost_low=0.75,
            cost_high=1.5,
            evidence=(
                f"Observed {observed_losses}/{matchup['games']} losses; exposure-expected "
                f"{float(matchup['expected_losses_from_exposure']):.2f}; shrunk loss rate "
                f"{float(matchup['bayesian_shrunk_loss_rate']):.3f} with 95% credible interval "
                f"[{float(matchup['bayesian_credible_low']):.3f}, {float(matchup['bayesian_credible_high']):.3f}]."
            ),
        )
    for cluster in cluster_rows:
        replay_ids = cluster.get("representative_replay_ids") or ()
        losses_in_cluster = int(cluster.get("losses") or 0)
        add_candidate(
            issue=f"Repeated failure phenotype: {cluster.get('label')}",
            action=(
                f"Turn cluster `{cluster.get('cluster_id')}` medoids into deterministic replay regressions "
                "and test one narrowly scoped hypothesis at a time."
            ),
            replay_ids=replay_ids,
            support_games=losses_in_cluster,
            observed_losses=losses_in_cluster,
            interval_low=float(cluster.get("frequency_wilson_low") or 0.0),
            interval_high=float(cluster.get("frequency_wilson_high") or 1.0),
            cost_low=0.5,
            cost_high=1.0,
            evidence=(
                f"Cluster contains {losses_in_cluster}/{len(losses)} losses; frequency 95% Wilson "
                f"[{float(cluster.get('frequency_wilson_low') or 0.0):.3f}, "
                f"{float(cluster.get('frequency_wilson_high') or 1.0):.3f}], mean silhouette "
                f"{float(cluster.get('mean_silhouette') or 0.0):.3f}."
            ),
        )

    all_loss_ids = [str(row.get("episode_id")) for row in losses]
    second_loss_ids = [
        str(row.get("episode_id"))
        for row in losses
        if row.get("first_second_proxy") == "second"
    ]
    unclassified_ids = [
        str(row.get("episode_id"))
        for row in losses
        if row.get("terminal_failure_class") == "loss_unclassified"
    ]
    persistent_ids = [
        str(row.get("episode_id"))
        for row in loss_timeline_rows
        if row.get("first_persistent_observable_disadvantage_turn") is not None
    ]
    generic_candidates = [
        (
            "Initiative-conditioned setup",
            "Add balanced first/second evaluation blocks and a going-second setup trace before training changes.",
            second_loss_ids,
            int(_mapping(initiative.get("going_second")).get("games") or len(second_loss_ids)),
            len(second_loss_ids),
            _finite(_mapping(initiative.get("going_second")).get("wilson_95", {}).get("low")) or 0.0,
            _finite(_mapping(initiative.get("going_second")).get("wilson_95", {}).get("high")) or 1.0,
            0.5,
            1.0,
            (
                f"Going-second loss relative risk {float(initiative.get('relative_risk') or 0.0):.3f}; "
                f"95% interval [{float(_mapping(initiative.get('relative_risk_95')).get('low') or 0.0):.3f}, "
                f"{float(_mapping(initiative.get('relative_risk_95')).get('high') or 0.0):.3f}]."
            ),
        ),
        (
            "Terminal cause coverage",
            "Instrument actor-visible terminal reason coverage before interpreting unclassified losses.",
            unclassified_ids,
            len(losses),
            len(unclassified_ids),
            *wilson_interval(len(unclassified_ids), len(losses)),
            0.25,
            0.75,
            f"{len(unclassified_ids)}/{len(losses)} losses have no provable terminal subclass.",
        ),
        (
            "Recoverability instrumentation gap",
            "Add prospective actor-visible risk instrumentation; do not infer irreversible turns from replay hindsight.",
            persistent_ids or all_loss_ids[:5],
            len(losses),
            len(persistent_ids),
            *wilson_interval(len(persistent_ids), len(losses)),
            1.0,
            2.0,
            f"A sustained observable-deficit proxy exists in {len(persistent_ids)}/{len(losses)} losses; true irreversibility remains unidentifiable.",
        ),
        (
            "Action confidence lacks an outcome target",
            "If outcome calibration is needed, design a separately validated actor-visible win-risk target; keep action probability distinct.",
            all_loss_ids[:8],
            len(decisive),
            len(losses),
            *wilson_interval(len(losses), len(decisive)),
            1.0,
            2.0,
            str(confidence_payload.get("probability_calibration_reason")),
        ),
        (
            "Sparse matchup evaluation",
            "Collect a preregistered, matchup-stratified panel with equal initiative exposure before ranking policy changes.",
            all_loss_ids[:10],
            len(decisive),
            len(losses),
            *wilson_interval(len(losses), len(decisive)),
            0.75,
            1.5,
            f"Only {len(decisive)} decisive games cover {len({_matchup_name(row) for row in decisive})} shells.",
        ),
        (
            "Full-trajectory model leakage risk",
            "Add fixed-turn landmark features and cross-fitted prediction before using the observational loss model for prioritization.",
            all_loss_ids[:8],
            len(decisive),
            len(losses),
            *wilson_interval(len(losses), len(decisive)),
            1.0,
            2.0,
            "Game length, final prize deficit, total action rates, and eventual evolution are downstream trajectory summaries.",
        ),
        (
            "Safety regression receipt",
            "Keep every candidate experiment gated on zero illegal actions, crashes, and timeouts with exact actor-binding checks.",
            all_loss_ids[:5],
            len(decisive),
            len(losses),
            *wilson_interval(len(losses), len(decisive)),
            0.25,
            0.5,
            "Current suite is safety-clean and exact actor reconstruction matched every scored decision.",
        ),
    ]
    for issue, action, replay_ids, support, observed, low, high, cost_low, cost_high, evidence in generic_candidates:
        add_candidate(
            issue=issue,
            action=action,
            replay_ids=replay_ids,
            support_games=support,
            observed_losses=observed,
            interval_low=float(low),
            interval_high=float(high),
            cost_low=cost_low,
            cost_high=cost_high,
            evidence=evidence,
        )
    deduplicated: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        deduplicated.setdefault(str(candidate["issue"]), candidate)
    ordered = sorted(
        deduplicated.values(),
        key=lambda row: (
            -float(row["observational_opportunity_per_planning_week"]),
            -int(row["observed_loss_exposure"]),
            str(row["issue"]),
        ),
    )[:20]
    for rank, row in enumerate(ordered, start=1):
        row["priority_rank"] = rank

    return ordered, {
        "requested_sort_metric_status": "expected_elo_gain_per_engineering_week_not_identifiable",
        "actual_sort_metric": "observational_opportunity_per_planning_week",
        "actual_sort_metric_warning": (
            "This is a triage score from excess observed loss exposure and planning-cost midpoints; "
            "it is not expected Elo, intervention efficacy, or a causal estimate."
        ),
        "task_count": len(ordered),
    }


def _expected_wilson_interval(probability: float, trials: int) -> tuple[float, float]:
    z = NORMAL_95
    denominator = 1.0 + z * z / trials
    center = (probability + z * z / (2.0 * trials)) / denominator
    half_width = z * math.sqrt(
        probability * (1.0 - probability) / trials + z * z / (4.0 * trials * trials)
    ) / denominator

    return (max(0.0, center - half_width), min(1.0, center + half_width))


def data_value_projection(wins: int, losses: int) -> dict[str, Any]:
    games = wins + losses
    if games <= 0:
        raise ValueError("data-value projection requires at least one decisive game")
    win_rate = wins / games
    current_standard_error = math.sqrt(win_rate * (1.0 - win_rate) / games)
    rows = []
    for additional_games in (500, 1_000, 5_000, 10_000):
        total_games = games + additional_games
        standard_error = math.sqrt(win_rate * (1.0 - win_rate) / total_games)
        low, high = _expected_wilson_interval(win_rate, total_games)
        elo = _performance_elo(win_rate)
        elo_low = _performance_elo(low)
        elo_high = _performance_elo(high)
        rows.append({
            "additional_games": additional_games,
            "projected_total_games": total_games,
            "expected_standard_error": standard_error,
            "standard_error_reduction_fraction": 1.0 - standard_error / current_standard_error
            if current_standard_error > 0.0
            else None,
            "expected_wilson_low": low,
            "expected_wilson_high": high,
            "expected_wilson_width": high - low,
            "expected_performance_elo_half_width": max(elo - elo_low, elo_high - elo)
            if elo is not None and elo_low is not None and elo_high is not None
            else None,
        })

    return {
        "current_games": games,
        "current_win_rate": win_rate,
        "current_standard_error": current_standard_error,
        "projection_kind": (
            "plug_in_expected_precision_conditional_on_true_p_equal_to_the_current_observed_rate; "
            "not posterior-predictive and not a guaranteed realized interval"
        ),
        "projections": rows,
        "assumptions": [
            "future games are independent Bernoulli trials",
            "true win rate and opponent mixture remain stationary",
            "initiative and matchup exposure do not drift",
        ],
        "data_vs_policy_value_status": "not_numerically_identifiable_without_policy_gain_and_cost_priors",
        "decision_judgment": (
            "A designed, stratified, instrumented evaluation is higher value before another blind "
            "policy iteration. More uncontrolled ladder games narrow mixture-average win-rate "
            "uncertainty but do not identify causes."
        ),
    }


def _decision_memo(
    rating: Mapping[str, Any],
    trend: Mapping[str, Any],
    matchups: Sequence[Mapping[str, Any]],
    initiative: Mapping[str, Any],
    confidence: Mapping[str, Any],
    loss_model: Mapping[str, Any],
    clusters: Mapping[str, Any],
    data_value: Mapping[str, Any],
) -> dict[str, Any]:
    repeated = [row for row in matchups if int(row.get("games") or 0) >= 4]
    repeated.sort(
        key=lambda row: (
            -float(row.get("posterior_probability_worse_than_field") or 0.0),
            -int(row.get("games") or 0),
        )
    )
    repeated_name = str(repeated[0].get("matchup")) if repeated else "no repeated matchup"

    return {
        "what_we_know_with_high_confidence": [
            (
                f"The observed record is {rating['wins']}W/{rating['losses']}L, but its 95% "
                f"Wilson interval is [{rating['wilson_95']['low']:.3f}, "
                f"{rating['wilson_95']['high']:.3f}]."
            ),
            (
                f"The exact test against 50% is p={rating['test_against_50_percent']['exact_two_sided_p']:.3f}; "
                "the sample does not resolve 46% versus 50%."
            ),
            "Chosen-action confidence is not a win-probability forecast, so outcome calibration is not identifiable.",
            "Historical replay cannot identify an irreversible decision or counterfactual winning action.",
        ],
        "what_we_suspect": [
            (
                f"The highest-evidence repeated matchup concern is `{repeated_name}`, but its "
                "posterior interval overlaps other shells."
            ),
            (
                f"Going second has observed loss RR {float(initiative.get('relative_risk') or 0.0):.2f}, "
                f"95% CI [{float(_mapping(initiative.get('relative_risk_95')).get('low') or 0.0):.2f}, "
                f"{float(_mapping(initiative.get('relative_risk_95')).get('high') or 0.0):.2f}]; "
                "this is a hypothesis, not an established cause."
            ),
            (
                "The middle collapse is descriptively real, but the selection-adjusted permutation "
                f"test is p={float(trend.get('weakest_third_selection_adjusted_p') or 0.0):.3f}."
            ),
        ],
        "what_remains_unknowable_from_this_corpus": [
            "Absolute Kaggle Elo or the location of a historical 700-Elo peak without rating history.",
            "Causal deck, card, initiative, or action effects from matchmaking-selected trajectories.",
            "Counterfactual Q-values, blunder labels, recoverability, or expected Elo gain per engineering week.",
            "Whether confidence is globally or locally miscalibrated as win probability.",
        ],
        "fix_before_retraining": [
            "Improve actor-visible terminal-cause and fixed-turn risk instrumentation.",
            "Run balanced first/second and matchup-stratified evaluation with preregistered slices.",
            "Turn representative failure-cluster replays into deterministic regression fixtures.",
            "Keep action preference distinct from any separately validated outcome-risk target.",
        ],
        "do_not_change": [
            "Do not mutate default v2.4 or protected submission/runtime surfaces from this postmortem.",
            "Do not add card-identity-specific heuristics from sparse observational counts.",
            "Do not import a mismatched critic, fabricate Q-values, or label confident actions as blunders.",
            "Do not interpret the exploratory loss model or clusters causally.",
        ],
        "bottleneck_verdict": {
            "primary": "instrumentation",
            "secondary": "evaluation_data",
            "policy": "plausible_hypotheses_not_isolated",
            "reason": (
                "Terminal/recoverability/value surfaces are missing, while 41 games leave broad "
                "rating and matchup intervals. Additional designed evaluation should precede blind iteration."
            ),
        },
        "model_status": loss_model.get("status"),
        "cluster_status": clusters.get("status"),
        "confidence_status": confidence.get("probability_calibration_status"),
        "data_value_status": data_value.get("data_vs_policy_value_status"),
    }


def build_statistical_addendum(
    episode_summaries: Sequence[Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
    *,
    random_seed: int = DEFAULT_RANDOM_SEED,
    permutation_samples: int = DEFAULT_PERMUTATION_SAMPLES,
) -> AddendumArtifacts:
    """Build all additive phases from already-sanitized base-run outputs."""

    decisive = _decisive_summaries(episode_summaries)
    wins = sum(row.get("result") == "win" for row in decisive)
    losses = sum(row.get("result") == "loss" for row in decisive)
    rating = rating_stability(wins, losses)
    rolling_rows, trend = rolling_performance(
        decisive,
        permutation_samples=permutation_samples,
        random_seed=random_seed,
    )
    matchup_rows, matchup_metadata = exposure_normalized_matchups(decisive)
    initiative = first_player_effect(decisive)
    loss_timeline_rows, loss_timeline_metadata = loss_timeline(
        decisive,
        decision_rows,
        random_seed=random_seed,
    )
    confidence, confidence_tables = confidence_diagnostics(
        decisive,
        decision_rows,
        random_seed=random_seed,
    )
    loss_model, model_prediction_rows, game_feature_rows = observational_loss_model(
        decisive,
        decision_rows,
        random_seed=random_seed,
    )
    clusters, cluster_rows, cluster_membership_rows = failure_clustering(
        decisive,
        game_feature_rows,
        loss_timeline_rows,
        confidence_tables["confidence_games"],
        random_seed=random_seed,
    )
    priorities, priority_metadata = engineering_priorities(
        decisive,
        matchup_rows,
        initiative,
        loss_timeline_rows,
        cluster_rows,
        confidence,
    )
    data_value = data_value_projection(wins, losses)
    memo = _decision_memo(
        rating,
        trend,
        matchup_rows,
        initiative,
        confidence,
        loss_model,
        clusters,
        data_value,
    )
    payload = {
        "schema_version": ADDENDUM_SCHEMA_VERSION,
        "source_contract": {
            "episode_sample_unit": "one attributable decisive game",
            "decisive_games": len(decisive),
            "actor_decisions_are_not_independent_samples": True,
            "chronology": "episode_id_order_proxy",
            "raw_replays_read": False,
            "raw_observations_read": False,
            "hidden_information_read": False,
            "counterfactual_values_available": False,
        },
        "phase_a_rating_stability": rating,
        "phase_b_rolling_performance": trend,
        "phase_c_exposure_normalization": {
            **matchup_metadata,
            "matchups": matchup_rows,
        },
        "phase_d_first_player_effect": initiative,
        "phase_e_loss_timeline": {
            **loss_timeline_metadata,
            "losses": loss_timeline_rows,
        },
        "phase_f_confidence_diagnostics": confidence,
        "phase_g_observational_loss_model": loss_model,
        "phase_h_failure_clustering": clusters,
        "phase_i_engineering_priority": {
            **priority_metadata,
            "tasks": priorities,
        },
        "phase_j_data_vs_policy": data_value,
        "phase_k_decision_memo": memo,
        "global_interpretation_boundary": (
            "All findings are observational or sampling projections. No causal effects, Q-values, "
            "counterfactual blunders, irreversible states, or realized engineering Elo gains are claimed."
        ),
    }
    tables: dict[str, Sequence[Mapping[str, Any]]] = {
        "rolling_performance.csv": rolling_rows,
        "matchup_normalized.csv": matchup_rows,
        "loss_timeline.csv": loss_timeline_rows,
        "confidence_histogram.csv": confidence_tables["confidence_histogram"],
        "entropy_histogram.csv": confidence_tables["entropy_histogram"],
        "confidence_games.csv": confidence_tables["confidence_games"],
        "confidence_by_outcome.csv": confidence_tables["confidence_by_outcome"],
        "confidence_by_matchup.csv": confidence_tables["confidence_by_matchup"],
        "confidence_by_phase.csv": confidence_tables["confidence_by_phase"],
        "loss_model_effects.csv": loss_model.get("effects") or (),
        "loss_model_predictions.csv": model_prediction_rows,
        "failure_clusters.csv": cluster_rows,
        "failure_cluster_members.csv": cluster_membership_rows,
        "engineering_priorities.csv": priorities,
        "data_value.csv": data_value["projections"],
    }

    return AddendumArtifacts(payload=payload, tables=tables)


def _format_percent(value: Any) -> str:
    number = _finite(value)

    return "n/a" if number is None else f"{number:.1%}"


def _format_number(value: Any, digits: int = 3) -> str:
    number = _finite(value)

    return "n/a" if number is None else f"{number:.{digits}f}"


def _markdown_table(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[tuple[str, str]],
    *,
    limit: int | None = None,
) -> list[str]:
    selected = list(rows if limit is None else rows[:limit])
    if not selected:
        return ["No supported rows."]
    output = [
        "| " + " | ".join(label for _key, label in columns) + " |",
        "|" + "|".join("---" for _key, _label in columns) + "|",
    ]
    for row in selected:
        values = []
        for key, _label in columns:
            value = row.get(key)
            if isinstance(value, float):
                rendered = f"{value:.3f}"
            elif isinstance(value, (list, dict)):
                rendered = str(value)
            else:
                rendered = "n/a" if value is None else str(value)
            values.append(rendered.replace("|", "/").replace("\n", " "))
        output.append("| " + " | ".join(values) + " |")

    return output


def render_statistical_addendum(payload: Mapping[str, Any]) -> str:
    """Render the structured addendum without introducing new claims."""

    rating = _mapping(payload.get("phase_a_rating_stability"))
    trend = _mapping(payload.get("phase_b_rolling_performance"))
    matchup_payload = _mapping(payload.get("phase_c_exposure_normalization"))
    matchups = list(matchup_payload.get("matchups") or ())
    initiative = _mapping(payload.get("phase_d_first_player_effect"))
    timeline = _mapping(payload.get("phase_e_loss_timeline"))
    confidence = _mapping(payload.get("phase_f_confidence_diagnostics"))
    model = _mapping(payload.get("phase_g_observational_loss_model"))
    clusters = _mapping(payload.get("phase_h_failure_clustering"))
    priorities = _mapping(payload.get("phase_i_engineering_priority"))
    data_value = _mapping(payload.get("phase_j_data_vs_policy"))
    memo = _mapping(payload.get("phase_k_decision_memo"))
    wilson = _mapping(rating.get("wilson_95"))
    exact = _mapping(rating.get("clopper_pearson_95"))
    posterior = _mapping(rating.get("jeffreys_posterior_95"))
    lines = [
        "# Statistical Addendum to the Submission Postmortem",
        "",
        "This extends the authoritative report; it does not replace or revise any base finding.",
        "The inferential unit is one attributable decisive game (41 games), never one actor decision.",
        "Chronology is numeric episode-ID order because the offline catalog has no creation timestamps.",
        "",
        "## Phase A - Rating stability",
        "",
        f"Observed record: **{rating.get('wins')}W-{rating.get('losses')}L**; win rate "
        f"{_format_percent(rating.get('win_rate'))}.",
        "",
        f"- Wilson 95% CI: [{_format_percent(wilson.get('low'))}, {_format_percent(wilson.get('high'))}].",
        f"- Clopper-Pearson 95% CI: [{_format_percent(exact.get('low'))}, {_format_percent(exact.get('high'))}].",
        f"- Jeffreys posterior 95% credible interval: [{_format_percent(posterior.get('low'))}, {_format_percent(posterior.get('high'))}].",
        f"- Field-relative performance Elo: {_format_number(rating.get('performance_elo_delta_vs_average_field'), 1)}; "
        f"Wilson-transformed interval [{_format_number(_mapping(rating.get('performance_elo_95_from_wilson')).get('low'), 1)}, "
        f"{_format_number(_mapping(rating.get('performance_elo_95_from_wilson')).get('high'), 1)}].",
        f"- Exact two-sided p versus 50%: {_format_number(_mapping(rating.get('test_against_50_percent')).get('exact_two_sided_p'))}.",
        "",
        "Jeffreys-posterior threshold probabilities:",
        "",
    ]
    lines.extend(_markdown_table(
        list(rating.get("posterior_threshold_probabilities") or ()),
        (("threshold", "true win rate above"), ("posterior_probability_above", "posterior probability")),
    ))
    lines.extend([
        "",
        "Verdict: **46% and 50% are statistically unresolved, not proven equivalent.** Absolute Kaggle Elo is unavailable.",
        "",
        "## Phase B - Rolling performance and trend",
        "",
        f"Maximum permanent split difference: {_format_percent(_mapping(trend.get('maximum_change_point')).get('absolute_win_rate_difference'))}; "
        f"look-elsewhere-adjusted permutation p={_format_number(_mapping(trend.get('maximum_change_point')).get('familywise_permutation_p'))}.",
        f"Worst selected low interval: games {_mapping(trend.get('selected_low_interval')).get('start_game')}-"
        f"{_mapping(trend.get('selected_low_interval')).get('end_game')} "
        f"({_mapping(trend.get('selected_low_interval')).get('wins')}W/"
        f"{_mapping(trend.get('selected_low_interval')).get('losses')}L), scan-adjusted p="
        f"{_format_number(_mapping(trend.get('selected_low_interval')).get('scan_adjusted_permutation_p'))}.",
        f"Weakest-third selection-adjusted permutation p={_format_number(trend.get('weakest_third_selection_adjusted_p'))}.",
        "",
        "The collapse is descriptively real, but this sample does not show it to be larger than constant-rate finite-sample variation. Opponent/initiative drift remains a confound.",
        "",
        "## Phase C - Exposure-normalized matchups",
        "",
    ])
    lines.extend(_markdown_table(
        matchups,
        (
            ("matchup", "matchup"),
            ("games", "games"),
            ("observed_losses", "losses"),
            ("expected_losses_from_exposure", "expected"),
            ("loss_ratio", "O/E"),
            ("bayesian_shrunk_loss_rate", "shrunk loss rate"),
            ("bayesian_credible_low", "CrI low"),
            ("bayesian_credible_high", "CrI high"),
        ),
        limit=12,
    ))
    lines.extend([
        "",
        "The posterior point ranking is not a proven ordering: most intervals overlap and opponent strength, exact deck, and initiative are confounded.",
        "",
        "## Phase D - First-player effect",
        "",
        f"Going-second loss RR: {_format_number(initiative.get('relative_risk'))} "
        f"(Katz approximate 95% CI [{_format_number(_mapping(initiative.get('relative_risk_95')).get('low'))}, "
        f"{_format_number(_mapping(initiative.get('relative_risk_95')).get('high'))}]); "
        f"OR {_format_number(initiative.get('odds_ratio'))} "
        f"(Wald-log approximate 95% CI [{_format_number(_mapping(initiative.get('odds_ratio_95')).get('low'))}, "
        f"{_format_number(_mapping(initiative.get('odds_ratio_95')).get('high'))}]); "
        f"Fisher p={_format_number(initiative.get('fisher_exact_two_sided_p'))}.",
        "",
        "Going second alone does not statistically explain the record; the interval includes weak or null effects and matchup/chronology are confounded.",
        "",
        "## Phase E - Loss timeline",
        "",
        "A true first irreversible disadvantage and recoverable-to-lost interval are `not_identifiable` from a single realized trajectory.",
        f"The observable first-deficit-to-terminal proxy averages "
        f"{_format_number(_mapping(timeline.get('first_observable_to_terminal')).get('mean'), 2)} turns "
        f"(episode-bootstrap 95% CI [{_format_number(_mapping(timeline.get('first_observable_to_terminal')).get('low'), 2)}, "
        f"{_format_number(_mapping(timeline.get('first_observable_to_terminal')).get('high'), 2)}]).",
        f"The stricter persistent-deficit proxy is available in {timeline.get('persistent_proxy_available_games')}/{timeline.get('losses')} losses.",
        "",
        "## Phase F - Confidence diagnostics",
        "",
        f"Calibration verdict: `{confidence.get('probability_calibration_status')}`. {confidence.get('probability_calibration_reason')}",
        "",
        "Confidence and entropy histograms plus episode-bootstrap outcome/matchup/phase tables are emitted as artifacts. They are descriptive action-preference associations, not win calibration.",
        "",
        "## Phase G - Observational loss model",
        "",
        f"Model status: `{model.get('status')}`; {model.get('games')} games, {model.get('loss_events')} losses, "
        f"{model.get('feature_count')} penalized features.",
        "",
    ])
    lines.extend(_markdown_table(
        list(model.get("effects") or ()),
        (
            ("importance_rank", "rank"),
            ("feature", "feature"),
            ("standardized_log_odds_effect", "log OR"),
            ("episode_bootstrap_standard_error", "bootstrap SE"),
            ("episode_bootstrap_ci_low", "CI low"),
            ("episode_bootstrap_ci_high", "CI high"),
            ("leave_one_episode_out_drop_log_loss_importance", "drop-log-loss"),
        ),
        limit=12,
    ))
    lines.extend([
        "",
        "This is a retrospective ridge association model, not a causal or time-to-event hazard model. Full-trajectory features are outcome-adjacent and the intervals are exploratory.",
        "",
        "## Phase H - Failure clustering",
        "",
        f"Status: `{clusters.get('status')}`; method `{clusters.get('method')}`; "
        f"k={clusters.get('cluster_count')}; mean silhouette {_format_number(clusters.get('mean_silhouette'))}.",
        "",
    ])
    lines.extend(_markdown_table(
        list(clusters.get("clusters") or ()),
        (
            ("cluster_id", "cluster"),
            ("label", "phenotype"),
            ("losses", "losses"),
            ("frequency_wilson_low", "freq CI low"),
            ("frequency_wilson_high", "freq CI high"),
            ("leave_one_episode_out_coassignment_confidence", "stability"),
            ("representative_replay_ids", "representative replays"),
            ("representative_turns", "turns"),
        ),
    ))
    lines.extend([
        "",
        "Clusters are exploratory observed phenotypes, not causes or policy modes.",
        "",
        "## Phase I - Engineering priority",
        "",
        f"Expected Elo gain per engineering week is `{priorities.get('requested_sort_metric_status')}`. "
        "The table is therefore sorted by a transparent observational-opportunity/cost triage score, not expected Elo.",
        "",
    ])
    lines.extend(_markdown_table(
        list(priorities.get("tasks") or ()),
        (
            ("priority_rank", "rank"),
            ("issue", "issue"),
            ("replay_ids", "replays"),
            ("estimated_implementation_cost_weeks_low", "weeks low"),
            ("estimated_implementation_cost_weeks_high", "weeks high"),
            ("confidence", "confidence"),
            ("observational_opportunity_per_planning_week", "triage score"),
        ),
        limit=20,
    ))
    lines.extend([
        "",
        "## Phase J - Data versus policy",
        "",
    ])
    lines.extend(_markdown_table(
        list(data_value.get("projections") or ()),
        (
            ("additional_games", "additional games"),
            ("projected_total_games", "total"),
            ("expected_standard_error", "expected SE"),
            ("standard_error_reduction_fraction", "SE removed"),
            ("expected_wilson_low", "Wilson low"),
            ("expected_wilson_high", "Wilson high"),
            ("expected_performance_elo_half_width", "Elo half-width"),
        ),
    ))
    lines.extend([
        "",
        str(data_value.get("decision_judgment")),
        "",
        "## Phase K - Decision memo",
        "",
        "### 1. What we know with high confidence",
        "",
    ])
    lines.extend(f"- {value}" for value in memo.get("what_we_know_with_high_confidence") or ())
    lines.extend(["", "### 2. What we suspect", ""])
    lines.extend(f"- **Hypothesis:** {value}" for value in memo.get("what_we_suspect") or ())
    lines.extend(["", "### 3. What remains unknowable", ""])
    lines.extend(f"- {value}" for value in memo.get("what_remains_unknowable_from_this_corpus") or ())
    lines.extend(["", "### 4. What should be fixed before retraining", ""])
    lines.extend(f"- {value}" for value in memo.get("fix_before_retraining") or ())
    lines.extend(["", "### 5. What should not be changed", ""])
    lines.extend(f"- {value}" for value in memo.get("do_not_change") or ())
    lines.extend([
        "",
        "### 6. Bottleneck",
        "",
        f"Primary: **{_mapping(memo.get('bottleneck_verdict')).get('primary')}**; "
        f"secondary: **{_mapping(memo.get('bottleneck_verdict')).get('secondary')}**. "
        f"{_mapping(memo.get('bottleneck_verdict')).get('reason')}",
        "",
        "## Interpretation boundary",
        "",
        str(payload.get("global_interpretation_boundary")),
    ])

    return "\n".join(lines) + "\n"


__all__ = [
    "ADDENDUM_SCHEMA_VERSION",
    "AddendumArtifacts",
    "beta_quantile",
    "build_statistical_addendum",
    "clopper_pearson_interval",
    "confidence_diagnostics",
    "data_value_projection",
    "engineering_priorities",
    "exposure_normalized_matchups",
    "failure_clustering",
    "first_player_effect",
    "loss_timeline",
    "observational_loss_model",
    "rating_stability",
    "regularized_beta_cdf",
    "render_statistical_addendum",
    "rolling_performance",
    "wilson_interval",
]
