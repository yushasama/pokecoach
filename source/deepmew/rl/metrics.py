"""First-class run metrics for the RL ablation ladder.

Experiments are *measured*, not just run (DEC-040: a module earns inclusion only
by improving measured play strength / sample efficiency / robustness / failure
rate across controlled ablations). This module defines the metric vocabulary and
computes a summary from a run result dict.

The smoke backend can only fill the runtime/throughput metrics; the
training-specific metrics are reserved as explicit placeholders so a PPO /
PufferLib / SB3 backend slots into the same schema without changing the report
format. Placeholders are emitted as ``None`` with an ``unavailable`` marker, not
silently omitted.
"""

from __future__ import annotations

from typing import Any

# Metrics the smoke / eval / learning backends can compute directly.
RUNTIME_METRICS: tuple[str, ...] = (
    "win_rate",
    "per_opponent_win_rate",
    "mean_reward",
    "decision_steps",
    "episodes",
    "completed",
    "illegal_attempt_rate",
    "crash_rate",
    "timeout_rate",
    "avg_game_length",
    "steps_per_sec",
    "games_per_sec",
    "env_fps",
)

# Reserved for a gradient backend (PPO/Puffer/SB3/CleanRL). The smoke backend
# emits these as None so the schema is stable across backends.
TRAINING_METRIC_PLACEHOLDERS: tuple[str, ...] = (
    "update_count",
    "rollout_steps",
    "policy_loss",
    "value_loss",
    "entropy",
    "approximate_kl",
    "clip_fraction",
    "explained_variance",
    "gradient_norm",
    "parameter_norm",
    "parameter_delta_norm",
)

ALL_METRICS: tuple[str, ...] = RUNTIME_METRICS + TRAINING_METRIC_PLACEHOLDERS


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def per_opponent_win_rate(per_opponent: dict[str, dict[str, int]]) -> dict[str, float]:
    rates: dict[str, float] = {}
    for label, bucket in per_opponent.items():
        completed = bucket.get("wins", 0) + bucket.get("losses", 0) + bucket.get("draws", 0)
        rates[label] = round(_safe_div(bucket.get("wins", 0), completed), 6)
    return rates


def summarize(result: dict[str, Any], *, elapsed_seconds: float) -> dict[str, Any]:
    """Compute the metrics summary from a `RLRunResult.to_dict()` payload."""
    episodes = int(result.get("episodes", 0))
    decision_steps = int(result.get("decision_steps", 0))
    completed = int(result.get("completed", result.get("wins", 0) + result.get("losses", 0) + result.get("draws", 0)))
    wins = int(result.get("wins", 0))
    crashes = int(result.get("crashes", 0))
    timeouts = int(result.get("timeouts", 0))
    illegal = int(result.get("illegal_attempts", 0))
    per_opponent = result.get("per_opponent", {}) or {}

    runtime = {
        "win_rate": round(_safe_div(wins, completed), 6),
        "per_opponent_win_rate": per_opponent_win_rate(per_opponent),
        "mean_reward": float(result.get("mean_reward", 0.0)),
        "decision_steps": decision_steps,
        "episodes": episodes,
        "completed": completed,
        "illegal_attempt_rate": round(_safe_div(illegal, decision_steps), 6),
        "crash_rate": round(_safe_div(crashes, episodes), 6),
        "timeout_rate": round(_safe_div(timeouts, episodes), 6),
        "avg_game_length": round(_safe_div(decision_steps, episodes), 4),
        "steps_per_sec": round(_safe_div(decision_steps, elapsed_seconds), 4),
        "games_per_sec": round(_safe_div(episodes, elapsed_seconds), 4),
        "env_fps": round(_safe_div(decision_steps, elapsed_seconds), 4),
    }
    training_metrics = result.get("training_metrics", {}) or {}
    placeholders = {}
    for name in TRAINING_METRIC_PLACEHOLDERS:
        if name in training_metrics:
            placeholders[name] = {"value": training_metrics[name], "status": "available"}
        else:
            placeholders[name] = {"value": None, "status": "unavailable_in_this_backend"}
    return {
        "elapsed_seconds": round(float(elapsed_seconds), 4),
        "runtime_metrics": runtime,
        "training_metrics": training_metrics,
        "training_metric_placeholders": placeholders,
        "metric_names": list(ALL_METRICS),
        "note": (
            "avg_game_length counts the RL seat's decisions per episode; "
            "throughput metrics reflect only the backend that produced this run "
            "and are not a claim about a different CPU/GPU/vectorized backend "
            "(measure first - see training_cost_fermi.md)."
        ),
    }
