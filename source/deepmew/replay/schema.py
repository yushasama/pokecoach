"""Versioned per-game replay row schema for DeepMew.

A replay row is one completed (or clearly crashed/timed-out) game, carrying the
self-contained identity already emitted by the V3.7.1 opponent-mixture row
contract (``results.json.matchups`` / ``matchups.csv``) plus a small provenance
header. The intent is that a downstream consumer can read a single JSONL table
and recover every row's mixture, policy, deck, seed, sampled opponent entry,
seat, result, and failure state without opening sidecar files.

This is the per-game layer. A later per-decision-frame layer can reuse the same
provenance header and bump ``SCHEMA_VERSION``.
"""

from __future__ import annotations

import math
from typing import Any


SCHEMA_VERSION = "deepmew.replay_row.v1"

# Provenance prepended by ingest. These are not part of the matchup row
# contract; they record where the row came from so the table is self-describing.
PROVENANCE_FIELDS: tuple[str, ...] = (
    "schema_version",
    "source_kind",
    "source_run_id",
    "source_path",
)

# Identity / result fields carried verbatim from the opponent-mixture row
# contract. Order matches the V3.7.1 ``matchups.csv`` header so CSV round-trips
# stay readable.
MATCHUP_IDENTITY_FIELDS: tuple[str, ...] = (
    "game_index",
    "mixture_id",
    "mixture_name",
    "mixture_version",
    "run_seed",
    "agent_spec",
    "agent_canonical_label",
    "agent_policy_id",
    "agent_policy_options",
    "opponent_spec",
    "opponent_canonical_label",
    "opponent_policy_id",
    "opponent_policy_options",
    "opponent_entry_index",
    "opponent_weight",
    "opponent_normalized_weight",
    "agent_deck_id",
    "agent_deck_hash",
    "opponent_deck_id",
    "opponent_deck_hash",
    "agent_seat",
    "result",
    "agent_win",
    "opponent_win",
    "draw",
    "crashed",
    "timed_out",
    "turns",
    "elapsed_seconds",
    "error",
)

REPLAY_ROW_FIELDS: tuple[str, ...] = PROVENANCE_FIELDS + MATCHUP_IDENTITY_FIELDS

# Fields whose value may legitimately be ``None``. Everything else in
# MATCHUP_IDENTITY_FIELDS (and every provenance field) must be present and
# non-null. ``result`` is nullable because a crashed/timed-out game may not have
# a winning seat; ``run_seed`` is nullable because seedless runs are allowed.
NULLABLE_FIELDS: frozenset[str] = frozenset(
    {"result", "error", "run_seed", "turns", "elapsed_seconds"}
)

_BOOL_FIELDS: tuple[str, ...] = (
    "agent_win",
    "opponent_win",
    "draw",
    "crashed",
    "timed_out",
)

_DRAW_RESULT = 2


class ReplaySchemaError(ValueError):
    """Raised when a replay row violates the versioned schema."""


def _is_real_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _where(index: int | None) -> str:
    return "" if index is None else f"row {index}: "


def _require_present(row: dict[str, Any], fields: tuple[str, ...], index: int | None) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise ReplaySchemaError(f"{_where(index)}missing required fields: {', '.join(sorted(missing))}")


def _require_non_null(row: dict[str, Any], fields: tuple[str, ...], index: int | None) -> None:
    null_fields = [field for field in fields if field not in NULLABLE_FIELDS and row.get(field) is None]
    if null_fields:
        raise ReplaySchemaError(f"{_where(index)}fields must not be null: {', '.join(sorted(null_fields))}")


def _validate_identity(row: dict[str, Any], index: int | None) -> None:
    _require_present(row, MATCHUP_IDENTITY_FIELDS, index)
    _require_non_null(row, MATCHUP_IDENTITY_FIELDS, index)

    where = _where(index)

    # Mixture identity.
    for field in ("mixture_id", "mixture_version", "mixture_name"):
        if not isinstance(row[field], str) or not row[field].strip():
            raise ReplaySchemaError(f"{where}{field} must be a non-empty string")

    # Deck hashes are the join key for later value-model work.
    for field in ("agent_deck_hash", "opponent_deck_hash"):
        if not isinstance(row[field], str) or not row[field].strip():
            raise ReplaySchemaError(f"{where}{field} must be a non-empty deck hash")

    # Sampled opponent identity.
    if not _is_int(row["opponent_entry_index"]) or row["opponent_entry_index"] < 0:
        raise ReplaySchemaError(
            f"{where}opponent_entry_index must be a non-negative integer, got {row['opponent_entry_index']!r}"
        )

    if not _is_int(row["game_index"]) or row["game_index"] < 1:
        raise ReplaySchemaError(f"{where}game_index must be a positive integer, got {row['game_index']!r}")

    if row["agent_seat"] not in (0, 1) or _is_real_bool(row["agent_seat"]):
        raise ReplaySchemaError(f"{where}agent_seat must be 0 or 1, got {row['agent_seat']!r}")

    for field in ("opponent_weight", "opponent_normalized_weight"):
        if not _is_number(row[field]):
            raise ReplaySchemaError(f"{where}{field} must be a finite number, got {row[field]!r}")

    for field in ("agent_policy_options", "opponent_policy_options"):
        if not isinstance(row[field], list):
            raise ReplaySchemaError(f"{where}{field} must be a list, got {type(row[field]).__name__}")

    # Boolean state fields must be real booleans, not ints or strings.
    for field in _BOOL_FIELDS:
        if not _is_real_bool(row[field]):
            raise ReplaySchemaError(f"{where}{field} must be a boolean, got {row[field]!r}")

    # result is a seat code (0/1) or {_DRAW_RESULT} for a draw, or None.
    result = row["result"]
    if result is not None and (not _is_int(result) or result not in (0, 1, _DRAW_RESULT)):
        raise ReplaySchemaError(f"{where}result must be null, 0, 1, or {_DRAW_RESULT}, got {result!r}")

    if row["turns"] is not None and (not _is_int(row["turns"]) or row["turns"] < 0):
        raise ReplaySchemaError(f"{where}turns must be a non-negative integer or null, got {row['turns']!r}")

    if row["elapsed_seconds"] is not None and not _is_number(row["elapsed_seconds"]):
        raise ReplaySchemaError(f"{where}elapsed_seconds must be a finite number or null")

    if row["run_seed"] is not None and not _is_int(row["run_seed"]):
        raise ReplaySchemaError(f"{where}run_seed must be an integer or null, got {row['run_seed']!r}")

    if row["error"] is not None and not isinstance(row["error"], str):
        raise ReplaySchemaError(f"{where}error must be a string or null")

    _validate_outcome_state(row, index)


def _validate_outcome_state(row: dict[str, Any], index: int | None) -> None:
    """Reject rows that are neither completed nor clearly crashed/timed out."""
    where = _where(index)
    crashed = row["crashed"]
    timed_out = row["timed_out"]
    has_result = row["result"] is not None
    failed = crashed or timed_out

    if not has_result and not failed:
        raise ReplaySchemaError(
            f"{where}row has no result but is not marked crashed or timed_out"
        )

    if not failed:
        # A cleanly completed game: win/draw flags must be coherent with result
        # and seat so a downstream consumer never has to recompute them.
        result = row["result"]
        seat = row["agent_seat"]
        expected_draw = result == _DRAW_RESULT
        expected_agent_win = (not expected_draw) and result == seat
        expected_opponent_win = (not expected_draw) and result != seat
        if row["draw"] != expected_draw:
            raise ReplaySchemaError(f"{where}draw={row['draw']} is inconsistent with result={result}")
        if row["agent_win"] != expected_agent_win:
            raise ReplaySchemaError(
                f"{where}agent_win={row['agent_win']} is inconsistent with result={result} seat={seat}"
            )
        if row["opponent_win"] != expected_opponent_win:
            raise ReplaySchemaError(
                f"{where}opponent_win={row['opponent_win']} is inconsistent with result={result} seat={seat}"
            )


def validate_matchup_row(row: dict[str, Any], *, index: int | None = None) -> dict[str, Any]:
    """Validate a raw V3.7.1 matchup row (before provenance is attached)."""
    if not isinstance(row, dict):
        raise ReplaySchemaError(f"{_where(index)}matchup row must be a JSON object")
    _validate_identity(row, index)
    return row


def validate_replay_row(row: dict[str, Any], *, index: int | None = None) -> dict[str, Any]:
    """Validate a fully-formed replay row, including provenance."""
    if not isinstance(row, dict):
        raise ReplaySchemaError(f"{_where(index)}replay row must be a JSON object")
    _require_present(row, PROVENANCE_FIELDS, index)
    _require_non_null(row, PROVENANCE_FIELDS, index)
    where = _where(index)
    if row["schema_version"] != SCHEMA_VERSION:
        raise ReplaySchemaError(
            f"{where}unsupported schema_version {row['schema_version']!r}; expected {SCHEMA_VERSION!r}"
        )
    for field in ("source_kind", "source_run_id", "source_path"):
        if not isinstance(row[field], str) or not row[field].strip():
            raise ReplaySchemaError(f"{where}{field} must be a non-empty string")
    _validate_identity(row, index)
    return row


def make_replay_row(
    matchup_row: dict[str, Any],
    *,
    source_kind: str,
    source_run_id: str,
    source_path: str,
    index: int | None = None,
) -> dict[str, Any]:
    """Normalize a validated matchup row into a versioned replay row.

    Identity fields are copied verbatim; only the provenance header is added.
    The result is itself validated so a malformed source row never escapes.
    """
    validate_matchup_row(matchup_row, index=index)
    replay_row = {
        "schema_version": SCHEMA_VERSION,
        "source_kind": source_kind,
        "source_run_id": source_run_id,
        "source_path": source_path,
    }
    for field in MATCHUP_IDENTITY_FIELDS:
        replay_row[field] = matchup_row[field]
    return validate_replay_row(replay_row, index=index)
