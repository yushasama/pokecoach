"""Ingest V3.7.1 opponent-mixture artifacts into versioned replay rows.

Preferred source is ``results.json`` (the ``matchups`` array, already typed).
Fallback is ``matchups.csv``, whose string cells are coerced back to typed
values. Identity that already lives in the row is never re-derived by joining
against ``mixture_summary.json`` or the manifest.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepmew.replay.schema import (
    MATCHUP_IDENTITY_FIELDS,
    ReplaySchemaError,
    make_replay_row,
)


SOURCE_KIND = "opponent_mixtures"

RESULTS_FILENAME = "results.json"
MATCHUPS_CSV_FILENAME = "matchups.csv"

_BOOL_CSV_FIELDS = ("agent_win", "opponent_win", "draw", "crashed", "timed_out")
_INT_CSV_FIELDS = ("game_index", "run_seed", "opponent_entry_index", "agent_seat", "result", "turns")
_FLOAT_CSV_FIELDS = ("opponent_weight", "opponent_normalized_weight", "elapsed_seconds")
_JSON_CSV_FIELDS = ("agent_policy_options", "opponent_policy_options")


class ReplayIngestError(ValueError):
    """Raised when a source artifact cannot be ingested."""


@dataclass(frozen=True)
class IngestResult:
    rows: list[dict[str, Any]]
    source_kind: str
    source_run_id: str
    source_path: str

    @property
    def row_count(self) -> int:
        return len(self.rows)


def _run_id_for(path: Path) -> str:
    """Derive the source run id from the artifact's containing directory name."""
    return path.resolve().parent.name


def _coerce_bool(value: str, field: str) -> bool:
    normalized = value.strip().lower()
    if normalized in ("true", "1"):
        return True
    if normalized in ("false", "0"):
        return False
    raise ReplayIngestError(f"{field}: cannot parse boolean from {value!r}")


def _coerce_int(value: str, field: str) -> int | None:
    text = value.strip()
    if text == "":
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ReplayIngestError(f"{field}: cannot parse integer from {value!r}") from exc


def _coerce_float(value: str, field: str) -> float | None:
    text = value.strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ReplayIngestError(f"{field}: cannot parse float from {value!r}") from exc


def _coerce_json(value: str, field: str) -> Any:
    text = value.strip()
    if text == "":
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReplayIngestError(f"{field}: cannot parse JSON from {value!r}") from exc


def _csv_row_to_matchup(raw: dict[str, str]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for field in MATCHUP_IDENTITY_FIELDS:
        if field not in raw:
            # Leave missing so schema validation reports it precisely.
            continue
        value = raw[field]
        if field in _BOOL_CSV_FIELDS:
            row[field] = _coerce_bool(value, field)
        elif field in _INT_CSV_FIELDS:
            row[field] = _coerce_int(value, field)
        elif field in _FLOAT_CSV_FIELDS:
            row[field] = _coerce_float(value, field)
        elif field in _JSON_CSV_FIELDS:
            row[field] = _coerce_json(value, field)
        elif field == "error":
            row[field] = value if value != "" else None
        else:
            row[field] = value
    return row


def _build_rows(
    matchups: list[Any],
    *,
    source_run_id: str,
    source_path: str,
    expected_count: int | None,
) -> list[dict[str, Any]]:
    if expected_count is not None and len(matchups) != expected_count:
        raise ReplayIngestError(
            f"inconsistent row count: {len(matchups)} matchup rows but artifact reports {expected_count} games"
        )
    rows: list[dict[str, Any]] = []
    for index, matchup in enumerate(matchups):
        if not isinstance(matchup, dict):
            raise ReplayIngestError(f"row {index}: matchup must be a JSON object, got {type(matchup).__name__}")
        try:
            rows.append(
                make_replay_row(
                    matchup,
                    source_kind=SOURCE_KIND,
                    source_run_id=source_run_id,
                    source_path=source_path,
                    index=index,
                )
            )
        except ReplaySchemaError as exc:
            raise ReplayIngestError(str(exc)) from exc
    return rows


def ingest_results_json(path: str | Path) -> IngestResult:
    """Ingest the preferred ``results.json`` artifact."""
    results_path = Path(path)
    if not results_path.is_file():
        raise ReplayIngestError(f"results.json not found: {results_path}")
    data = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ReplayIngestError(f"results.json root must be a JSON object: {results_path}")
    matchups = data.get("matchups")
    if not isinstance(matchups, list):
        raise ReplayIngestError(f"results.json has no 'matchups' array: {results_path}")
    # Cross-check the declared game count where available, but never join against
    # sidecars for identity. ``records`` and ``summary.games`` are internal to
    # this same file.
    expected = None
    summary = data.get("summary")
    if isinstance(summary, dict) and isinstance(summary.get("games"), int):
        expected = summary["games"]
    records = data.get("records")
    if isinstance(records, list) and len(records) != len(matchups):
        raise ReplayIngestError(
            f"inconsistent row count: {len(matchups)} matchups but {len(records)} records in {results_path}"
        )
    source_run_id = _run_id_for(results_path)
    rows = _build_rows(
        matchups,
        source_run_id=source_run_id,
        source_path=str(results_path),
        expected_count=expected,
    )
    return IngestResult(
        rows=rows,
        source_kind=SOURCE_KIND,
        source_run_id=source_run_id,
        source_path=str(results_path),
    )


def ingest_matchups_csv(path: str | Path) -> IngestResult:
    """Ingest the fallback ``matchups.csv`` artifact."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise ReplayIngestError(f"matchups.csv not found: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        raw_rows = list(reader)
    if not raw_rows:
        raise ReplayIngestError(f"matchups.csv has no data rows: {csv_path}")
    matchups = [_csv_row_to_matchup(raw) for raw in raw_rows]
    source_run_id = _run_id_for(csv_path)
    rows = _build_rows(
        matchups,
        source_run_id=source_run_id,
        source_path=str(csv_path),
        expected_count=None,
    )
    return IngestResult(
        rows=rows,
        source_kind=SOURCE_KIND,
        source_run_id=source_run_id,
        source_path=str(csv_path),
    )


def ingest_run_dir(run_dir: str | Path) -> IngestResult:
    """Ingest one opponent-mixture run directory.

    Prefers ``results.json``; falls back to ``matchups.csv`` only if the
    preferred artifact is absent.
    """
    directory = Path(run_dir)
    if not directory.is_dir():
        raise ReplayIngestError(f"run directory not found: {directory}")
    results_path = directory / RESULTS_FILENAME
    if results_path.is_file():
        return ingest_results_json(results_path)
    csv_path = directory / MATCHUPS_CSV_FILENAME
    if csv_path.is_file():
        return ingest_matchups_csv(csv_path)
    raise ReplayIngestError(
        f"no {RESULTS_FILENAME} or {MATCHUPS_CSV_FILENAME} found in {directory}"
    )
