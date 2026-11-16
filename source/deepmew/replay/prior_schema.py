"""Versioned replay-prior artifact schema for DeepMew (offline only).

`deepmew.replay_prior.v1` is a tiny, transparent count-table prior keyed on
`select_context + option_type`, built offline from `deepmew.replay_decision_row.v1`
tables. The target statistic is the smoothed win advantage of an option type over
its select-context baseline.

This is NOT a value model and NOT online behavior. The win label is the
whole-game `agent_win` outcome attributed to every decision in the game (crude
credit assignment), used only as a weak count-table signal. No per-option score
is invented. The artifact is never loaded by the eval/runtime submission path.
"""

from __future__ import annotations

import math
from typing import Any


SCHEMA_VERSION = "deepmew.replay_prior.v1"
KEY_FIELDS: tuple[str, ...] = ("select_context", "option_type")
TARGET = "smoothed_win_advantage"

ENTRY_FIELDS: tuple[str, ...] = (
    "select_context",
    "option_type",
    "count",
    "wins",
    "win_rate",
    "context_count",
    "context_wins",
    "context_win_rate",
    "smoothed_rate",
    "advantage",
)

_TOP_FIELDS: tuple[str, ...] = (
    "schema_version",
    "key_fields",
    "target",
    "smoothing",
    "min_count",
    "source",
    "baseline",
    "entries",
)


class ReplayPriorSchemaError(ValueError):
    """Raised when a replay-prior artifact violates the versioned schema."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _is_rate(value: Any) -> bool:
    return _is_number(value) and -1e-9 <= float(value) <= 1.0 + 1e-9


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayPriorSchemaError(message)


def validate_prior_entry(entry: dict[str, Any], *, index: int | None = None, min_count: int = 0) -> dict[str, Any]:
    where = "" if index is None else f"entry {index}: "
    _require(isinstance(entry, dict), f"{where}entry must be a JSON object")
    missing = [field for field in ENTRY_FIELDS if field not in entry]
    _require(not missing, f"{where}missing entry fields: {', '.join(sorted(missing))}")

    for field in ("select_context", "option_type"):
        _require(isinstance(entry[field], str) and bool(entry[field].strip()), f"{where}{field} must be a non-empty string")
    for field in ("count", "wins", "context_count", "context_wins"):
        _require(_is_int(entry[field]) and entry[field] >= 0, f"{where}{field} must be a non-negative integer")
    _require(entry["wins"] <= entry["count"], f"{where}wins cannot exceed count")
    _require(entry["context_wins"] <= entry["context_count"], f"{where}context_wins cannot exceed context_count")
    _require(entry["context_count"] > 0, f"{where}context_count must be positive")
    _require(entry["count"] >= min_count, f"{where}count {entry['count']} is below min_count {min_count}")
    for field in ("win_rate", "context_win_rate", "smoothed_rate"):
        _require(_is_rate(entry[field]), f"{where}{field} must be a probability in [0, 1]")
    _require(_is_number(entry["advantage"]) and -1.0 - 1e-9 <= float(entry["advantage"]) <= 1.0 + 1e-9,
             f"{where}advantage must be a number in [-1, 1]")
    return entry


def validate_prior_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    _require(isinstance(artifact, dict), "prior artifact must be a JSON object")
    missing = [field for field in _TOP_FIELDS if field not in artifact]
    _require(not missing, f"missing required fields: {', '.join(sorted(missing))}")

    _require(artifact["schema_version"] == SCHEMA_VERSION,
             f"unsupported schema_version {artifact['schema_version']!r}; expected {SCHEMA_VERSION!r}")
    _require(list(artifact["key_fields"]) == list(KEY_FIELDS), f"key_fields must be {list(KEY_FIELDS)}")
    _require(artifact["target"] == TARGET, f"target must be {TARGET!r}")

    smoothing = artifact["smoothing"]
    _require(isinstance(smoothing, dict) and _is_number(smoothing.get("alpha")) and float(smoothing.get("alpha")) >= 0,
             "smoothing.alpha must be a non-negative number")
    _require(isinstance(smoothing.get("baseline"), str) and bool(smoothing["baseline"].strip()),
             "smoothing.baseline must be a non-empty string")

    min_count = artifact["min_count"]
    _require(_is_int(min_count) and min_count >= 0, "min_count must be a non-negative integer")

    source = artifact["source"]
    _require(isinstance(source, dict), "source must be a JSON object")
    for field in ("row_schema", "source_run_ids", "source_paths", "agent_filter"):
        _require(field in source, f"source.{field} is required")
    _require(isinstance(source["source_run_ids"], list), "source.source_run_ids must be a list")
    _require(isinstance(source["source_paths"], list), "source.source_paths must be a list")
    _require(source["agent_filter"] is None or isinstance(source["agent_filter"], str),
             "source.agent_filter must be a string or null")

    baseline = artifact["baseline"]
    _require(isinstance(baseline, dict), "baseline must be a JSON object")
    _require("global_win_rate" in baseline and _is_rate(baseline["global_win_rate"]),
             "baseline.global_win_rate must be a probability in [0, 1]")
    _require(isinstance(baseline.get("per_context"), dict), "baseline.per_context must be a JSON object")

    _require(isinstance(artifact["entries"], list), "entries must be a list")
    for index, entry in enumerate(artifact["entries"]):
        validate_prior_entry(entry, index=index, min_count=min_count)
    return artifact
