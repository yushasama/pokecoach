"""Versioned per-decision replay row schema for DeepMew.

A decision row is one `deepmew.decision_frame.v1` `DecisionFrame` flattened into a
normalized, queryable record. It carries the frame's provenance and identity, the
acting seat, derived agent/opponent labels and deck hash, the select context, a
compact legal/chosen action summary, the chosen resolution status, and the
back-filled per-game outcome. A downstream consumer recovers all of this from one
JSONL table without opening `frames.jsonl`.

This is trace-grain, not mixture-grain: existing trace runs do not carry mixture
id / `opponent_entry_index`, so those fields are intentionally absent. Per-option
policy scores are unavailable in the trace and are intentionally not invented.
"""

from __future__ import annotations

import math
from typing import Any


SCHEMA_VERSION = "deepmew.replay_decision_row.v1"

PROVENANCE_FIELDS: tuple[str, ...] = (
    "schema_version",
    "source_kind",
    "source_run_id",
    "source_path",
)

# Copied verbatim from the DecisionFrame.
FRAME_IDENTITY_FIELDS: tuple[str, ...] = (
    "trace_run_id",
    "game_id",
    "frame_id",
    "decision_index",
    "seed",
    "seat",
    "turn",
    "agent_spec",
    "agent_label",
    "policy_version",
    "deck_path",
)

# Derived at ingest without replaying games.
DERIVED_FIELDS: tuple[str, ...] = (
    "deck_id",
    "deck_hash",
    "opponent_label",
)

DECISION_CONTEXT_FIELDS: tuple[str, ...] = (
    "select_type",
    "select_context",
    "min_count",
    "max_count",
    "legal_count",
    "legal_option_types",
)

CHOSEN_FIELDS: tuple[str, ...] = (
    "chosen_indices",
    "chosen_option_types",
    "chosen_card_ids",
    "chosen_card_names",
    "chosen_attack_ids",
    "chosen_resolved",
)

OUTCOME_FIELDS: tuple[str, ...] = (
    "final_result",
    "final_result_reason",
    "agent_win",
    "draw",
    "crashed",
    "timed_out",
)

DECISION_ROW_FIELDS: tuple[str, ...] = (
    PROVENANCE_FIELDS
    + FRAME_IDENTITY_FIELDS
    + DERIVED_FIELDS
    + DECISION_CONTEXT_FIELDS
    + CHOSEN_FIELDS
    + OUTCOME_FIELDS
)

# Fields whose value may legitimately be ``None``. Everything else must be present
# and non-null. ``seat``/``turn``/``seed`` are nullable (engine may omit them);
# deck/opponent derivations are nullable when the deck file or other seat is
# unavailable; ``final_result``/``final_result_reason`` are nullable for
# crashed/timed-out games.
NULLABLE_FIELDS: frozenset[str] = frozenset(
    {
        "seed",
        "seat",
        "turn",
        "policy_version",
        "deck_path",
        "deck_id",
        "deck_hash",
        "opponent_label",
        "final_result",
        "final_result_reason",
    }
)

_BOOL_FIELDS: tuple[str, ...] = ("chosen_resolved", "agent_win", "draw", "crashed", "timed_out")
_LIST_FIELDS: tuple[str, ...] = (
    "legal_option_types",
    "chosen_indices",
    "chosen_option_types",
    "chosen_card_ids",
    "chosen_card_names",
    "chosen_attack_ids",
)
_NON_NEG_INT_FIELDS: tuple[str, ...] = ("decision_index", "min_count", "max_count", "legal_count")
_NON_EMPTY_STR_FIELDS: tuple[str, ...] = (
    "trace_run_id",
    "game_id",
    "frame_id",
    "agent_spec",
    "agent_label",
    "select_type",
    "select_context",
)

_DRAW_RESULT = 2

# Option types that carry a card/attack/target identity worth resolving. Other
# select types (YES/NO/NUMBER/...) have nothing to resolve.
CARD_BEARING_OPTION_TYPES: frozenset[str] = frozenset({"PLAY", "EVOLVE", "ATTACH", "ATTACK"})


class ReplayDecisionSchemaError(ValueError):
    """Raised when a decision row violates the versioned schema."""


def _is_real_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _where(index: int | None) -> str:
    return "" if index is None else f"row {index}: "


def _require_present(row: dict[str, Any], fields: tuple[str, ...], index: int | None) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise ReplayDecisionSchemaError(f"{_where(index)}missing required fields: {', '.join(sorted(missing))}")


def _require_non_null(row: dict[str, Any], index: int | None) -> None:
    null_fields = [
        field
        for field in DECISION_ROW_FIELDS
        if field not in NULLABLE_FIELDS and row.get(field) is None
    ]
    if null_fields:
        raise ReplayDecisionSchemaError(
            f"{_where(index)}fields must not be null: {', '.join(sorted(null_fields))}"
        )


def validate_decision_row(row: dict[str, Any], *, index: int | None = None) -> dict[str, Any]:
    """Validate a fully-formed decision row against the versioned schema."""
    if not isinstance(row, dict):
        raise ReplayDecisionSchemaError(f"{_where(index)}decision row must be a JSON object")
    _require_present(row, DECISION_ROW_FIELDS, index)
    _require_non_null(row, index)
    where = _where(index)

    if row["schema_version"] != SCHEMA_VERSION:
        raise ReplayDecisionSchemaError(
            f"{where}unsupported schema_version {row['schema_version']!r}; expected {SCHEMA_VERSION!r}"
        )
    for field in ("source_kind", "source_run_id", "source_path"):
        if not isinstance(row[field], str) or not row[field].strip():
            raise ReplayDecisionSchemaError(f"{where}{field} must be a non-empty string")
    for field in _NON_EMPTY_STR_FIELDS:
        if not isinstance(row[field], str) or not row[field].strip():
            raise ReplayDecisionSchemaError(f"{where}{field} must be a non-empty string")

    for field in _NON_NEG_INT_FIELDS:
        if not _is_int(row[field]) or row[field] < 0:
            raise ReplayDecisionSchemaError(f"{where}{field} must be a non-negative integer, got {row[field]!r}")

    if row["seat"] is not None and (row["seat"] not in (0, 1) or _is_real_bool(row["seat"])):
        raise ReplayDecisionSchemaError(f"{where}seat must be 0, 1, or null, got {row['seat']!r}")
    if row["turn"] is not None and (not _is_int(row["turn"]) or row["turn"] < 0):
        raise ReplayDecisionSchemaError(f"{where}turn must be a non-negative integer or null, got {row['turn']!r}")
    if row["seed"] is not None and not _is_int(row["seed"]):
        raise ReplayDecisionSchemaError(f"{where}seed must be an integer or null, got {row['seed']!r}")

    for field in _LIST_FIELDS:
        if not isinstance(row[field], list):
            raise ReplayDecisionSchemaError(f"{where}{field} must be a list, got {type(row[field]).__name__}")

    for field in _BOOL_FIELDS:
        if not _is_real_bool(row[field]):
            raise ReplayDecisionSchemaError(f"{where}{field} must be a boolean, got {row[field]!r}")

    # Chosen indices must be within the declared legal range.
    legal_count = row["legal_count"]
    for chosen in row["chosen_indices"]:
        if not _is_int(chosen) or chosen < 0 or chosen >= legal_count:
            raise ReplayDecisionSchemaError(
                f"{where}chosen index {chosen!r} out of legal range [0, {legal_count})"
            )

    result = row["final_result"]
    if result is not None and (not _is_int(result) or result not in (0, 1, _DRAW_RESULT)):
        raise ReplayDecisionSchemaError(f"{where}final_result must be null, 0, 1, or {_DRAW_RESULT}, got {result!r}")
    if row["final_result_reason"] is not None and not isinstance(row["final_result_reason"], str):
        raise ReplayDecisionSchemaError(f"{where}final_result_reason must be a string or null")

    _validate_outcome_coherence(row, index)
    return row


def _validate_outcome_coherence(row: dict[str, Any], index: int | None) -> None:
    where = _where(index)
    result = row["final_result"]
    seat = row["seat"]
    expected_draw = result == _DRAW_RESULT
    if row["draw"] != expected_draw:
        raise ReplayDecisionSchemaError(f"{where}draw={row['draw']} is inconsistent with final_result={result}")

    failed = row["crashed"] or row["timed_out"]
    no_clean_winner = failed or result is None or expected_draw
    if no_clean_winner:
        if row["agent_win"]:
            raise ReplayDecisionSchemaError(
                f"{where}agent_win must be False for draw/crash/timeout/no-result rows"
            )
        return

    expected_win = seat is not None and result == seat
    if row["agent_win"] != expected_win:
        raise ReplayDecisionSchemaError(
            f"{where}agent_win={row['agent_win']} is inconsistent with final_result={result} seat={seat}"
        )
