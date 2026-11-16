"""Versioned raw replay-action row schema for DeepMew (Bible §2.16 aligned).

Unlike `deepmew.replay_decision_row.v1` (normalized, identity-only), this layer
keeps the raw per-decision observation (`state_json`) and the raw legal option
payload (`legal_options_json`) so it can serve as Bible §2.16
`state/action/result(+result_reason)` supervised rows. It is built from existing
`deepmew.decision_frame.v1` `frames.jsonl` artifacts (no game replay), and never
fakes raw state: when a frame lacks a raw observation, `state_json` /
`legal_options_json` are left null and counted, not fabricated.

Bible §2.16 minimum schema: game_id, player_id, turn, step, state_json,
legal_options_json, chosen_indices, result, result_reason, deck_id,
source_rating. `source_rating` is null for arena/trace-origin rows (local sims
carry no ladder rating); this is documented, not a defect.
"""

from __future__ import annotations

import math
from typing import Any


SCHEMA_VERSION = "deepmew.replay_action_row.v1"

PROVENANCE_FIELDS: tuple[str, ...] = (
    "schema_version",
    "source_kind",
    "source_run_id",
    "source_path",
)

# Bible §2.16-aligned fields (plus frame_id/seed/deck_hash as reproducibility
# extras). `state_json` and `legal_options_json` are the raw payloads the
# normalized decision row intentionally omits.
ACTION_FIELDS: tuple[str, ...] = (
    "game_id",
    "player_id",
    "turn",
    "step",
    "frame_id",
    "seed",
    "state_json",
    "legal_options_json",
    "chosen_indices",
    "result",
    "result_reason",
    "deck_id",
    "deck_hash",
    "source_rating",
)

ACTION_ROW_FIELDS: tuple[str, ...] = PROVENANCE_FIELDS + ACTION_FIELDS

# Fields allowed to be null. Everything else must be present and non-null.
# state_json / legal_options_json are nullable so a frame missing a raw
# observation is recorded honestly (counted) rather than faked.
NULLABLE_FIELDS: frozenset[str] = frozenset(
    {
        "player_id",
        "turn",
        "seed",
        "state_json",
        "legal_options_json",
        "result",
        "result_reason",
        "deck_id",
        "deck_hash",
        "source_rating",
    }
)

_DRAW_RESULT = 2
_NON_EMPTY_STR_FIELDS: tuple[str, ...] = ("source_kind", "source_run_id", "source_path", "game_id", "frame_id")


class ReplayActionSchemaError(ValueError):
    """Raised when a raw action row violates the versioned schema."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _where(index: int | None) -> str:
    return "" if index is None else f"row {index}: "


def validate_action_row(row: dict[str, Any], *, index: int | None = None) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ReplayActionSchemaError(f"{_where(index)}action row must be a JSON object")

    missing = [field for field in ACTION_ROW_FIELDS if field not in row]
    if missing:
        raise ReplayActionSchemaError(f"{_where(index)}missing required fields: {', '.join(sorted(missing))}")

    null_required = [
        field for field in ACTION_ROW_FIELDS if field not in NULLABLE_FIELDS and row.get(field) is None
    ]
    if null_required:
        raise ReplayActionSchemaError(
            f"{_where(index)}fields must not be null: {', '.join(sorted(null_required))}"
        )

    where = _where(index)
    if row["schema_version"] != SCHEMA_VERSION:
        raise ReplayActionSchemaError(
            f"{where}unsupported schema_version {row['schema_version']!r}; expected {SCHEMA_VERSION!r}"
        )
    for field in _NON_EMPTY_STR_FIELDS:
        if not isinstance(row[field], str) or not row[field].strip():
            raise ReplayActionSchemaError(f"{where}{field} must be a non-empty string")

    if not _is_int(row["step"]) or row["step"] < 0:
        raise ReplayActionSchemaError(f"{where}step must be a non-negative integer, got {row['step']!r}")
    if row["player_id"] is not None and row["player_id"] not in (0, 1):
        raise ReplayActionSchemaError(f"{where}player_id must be 0, 1, or null, got {row['player_id']!r}")
    if row["turn"] is not None and (not _is_int(row["turn"]) or row["turn"] < 0):
        raise ReplayActionSchemaError(f"{where}turn must be a non-negative integer or null, got {row['turn']!r}")
    if row["seed"] is not None and not _is_int(row["seed"]):
        raise ReplayActionSchemaError(f"{where}seed must be an integer or null, got {row['seed']!r}")

    # Raw payloads: state_json is a JSON object; legal_options_json is a list.
    if row["state_json"] is not None and not isinstance(row["state_json"], dict):
        raise ReplayActionSchemaError(f"{where}state_json must be a JSON object or null")
    if row["legal_options_json"] is not None and not isinstance(row["legal_options_json"], list):
        raise ReplayActionSchemaError(f"{where}legal_options_json must be a list or null")

    if not isinstance(row["chosen_indices"], list):
        raise ReplayActionSchemaError(f"{where}chosen_indices must be a list")
    legal = row["legal_options_json"]
    for chosen in row["chosen_indices"]:
        if not _is_int(chosen) or chosen < 0:
            raise ReplayActionSchemaError(f"{where}chosen index {chosen!r} must be a non-negative integer")
        if isinstance(legal, list) and chosen >= len(legal):
            raise ReplayActionSchemaError(
                f"{where}chosen index {chosen!r} out of legal range [0, {len(legal)})"
            )

    result = row["result"]
    if result is not None and (not _is_int(result) or result not in (0, 1, _DRAW_RESULT)):
        raise ReplayActionSchemaError(f"{where}result must be null, 0, 1, or {_DRAW_RESULT}, got {result!r}")
    if row["result_reason"] is not None and not isinstance(row["result_reason"], str):
        raise ReplayActionSchemaError(f"{where}result_reason must be a string or null")

    for field in ("deck_id", "deck_hash"):
        if row[field] is not None and not isinstance(row[field], str):
            raise ReplayActionSchemaError(f"{where}{field} must be a string or null")
    if row["source_rating"] is not None and not _is_number(row["source_rating"]):
        raise ReplayActionSchemaError(f"{where}source_rating must be a number or null, got {row['source_rating']!r}")

    return row
