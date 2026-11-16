"""Compare two or more RL run summaries.

Deliberately simple for v0: it reports numeric metric deltas of each run versus a
baseline (the first summary). It does NOT claim statistical significance unless
the sample size clears a conservative bar, and it carries explicit
confidence-interval placeholders marked unavailable. This is the seed of the
"every module must prove itself through ablation" loop (DEC-040): a module is
kept only if its run beats the control on the agreed metrics by more than noise.
"""

from __future__ import annotations

from typing import Any

# Below this many decided games, do not even discuss significance.
MIN_GAMES_FOR_SIGNIFICANCE = 200

_COMPARABLE_METRICS = (
    "win_rate",
    "mean_reward",
    "illegal_attempt_rate",
    "crash_rate",
    "timeout_rate",
    "avg_game_length",
    "steps_per_sec",
    "games_per_sec",
)


def _runtime(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics") or summary
    return metrics.get("runtime_metrics", {}) if isinstance(metrics, dict) else {}


def _label(summary: dict[str, Any]) -> str:
    return str(summary.get("experiment_id") or summary.get("label") or "run")


def _completed(summary: dict[str, Any]) -> int:
    runtime = _runtime(summary)
    return int(runtime.get("completed", 0))


def compare_runs(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Return baseline + per-run metric deltas. ``summaries[0]`` is the baseline."""
    if not summaries:
        raise ValueError("compare_runs needs at least one summary")
    baseline = summaries[0]
    base_runtime = _runtime(baseline)
    base_completed = _completed(baseline)

    comparisons: list[dict[str, Any]] = []
    for summary in summaries[1:]:
        runtime = _runtime(summary)
        completed = _completed(summary)
        deltas: dict[str, Any] = {}
        for metric in _COMPARABLE_METRICS:
            base_value = base_runtime.get(metric)
            value = runtime.get(metric)
            if isinstance(base_value, (int, float)) and isinstance(value, (int, float)):
                deltas[metric] = {
                    "baseline": base_value,
                    "candidate": value,
                    "delta": round(value - base_value, 6),
                }
        enough = min(base_completed, completed) >= MIN_GAMES_FOR_SIGNIFICANCE
        comparisons.append(
            {
                "candidate": _label(summary),
                "deltas": deltas,
                "sample_size": {"baseline_completed": base_completed, "candidate_completed": completed},
                "significance": {
                    "claimed": False,
                    "enough_games_for_discussion": enough,
                    "min_games_threshold": MIN_GAMES_FOR_SIGNIFICANCE,
                    "confidence_interval": "unavailable",
                    "note": (
                        "deltas are descriptive only; no significance is claimed. "
                        "A promotion decision requires a paired arena/ladder gate with CIs."
                    ),
                },
            }
        )

    return {
        "baseline": _label(baseline),
        "comparisons": comparisons,
        "metrics_compared": list(_COMPARABLE_METRICS),
    }
