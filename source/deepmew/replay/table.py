"""Build and export the normalized replay table.

The table is a flat list of versioned replay rows (one per game) plus a
self-describing manifest. Exports are intentionally plain: a JSONL row stream, an
optional CSV mirror, and a manifest carrying schema version, source run ids, row
count, crash/timeout counts, and deck/policy/mixture coverage.
"""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from deepmew.artifacts.manifest import build_manifest, write_manifest
from deepmew.artifacts.writers import write_json, write_jsonl, write_text
from deepmew.replay.schema import (
    REPLAY_ROW_FIELDS,
    SCHEMA_VERSION,
    validate_replay_row,
)


REPLAY_TABLE_KIND = "replay_table"

JSONL_FILENAME = "replay_table.jsonl"
CSV_FILENAME = "replay_table.csv"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class ReplayTableExport:
    out_dir: Path
    jsonl_path: Path
    csv_path: Path | None
    manifest_path: Path
    summary: dict[str, Any]


def _sorted_unique(values: Iterable[Any]) -> list[Any]:
    return sorted({value for value in values if value is not None})


def table_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the manifest summary, including failure counts and coverage."""
    completed = sum(
        1 for row in rows if row["result"] is not None and not row["crashed"] and not row["timed_out"]
    )
    crashes = sum(1 for row in rows if row["crashed"])
    timeouts = sum(1 for row in rows if row["timed_out"])
    draws = sum(1 for row in rows if row["draw"])
    agent_wins = sum(1 for row in rows if row["agent_win"])
    opponent_wins = sum(1 for row in rows if row["opponent_win"])

    policy_ids = _sorted_unique(
        [row["agent_policy_id"] for row in rows] + [row["opponent_policy_id"] for row in rows]
    )
    deck_hashes = _sorted_unique(
        [row["agent_deck_hash"] for row in rows] + [row["opponent_deck_hash"] for row in rows]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "row_count": len(rows),
        "source_run_ids": _sorted_unique(row["source_run_id"] for row in rows),
        "completed": completed,
        "crashes": crashes,
        "timeouts": timeouts,
        "draws": draws,
        "agent_wins": agent_wins,
        "opponent_wins": opponent_wins,
        "coverage": {
            "mixtures": _sorted_unique(row["mixture_id"] for row in rows),
            "policies": policy_ids,
            "deck_hashes": deck_hashes,
            "agents": _sorted_unique(row["agent_canonical_label"] for row in rows),
            "opponents": _sorted_unique(row["opponent_canonical_label"] for row in rows),
            "sampled_opponent_counts": dict(
                sorted(Counter(str(row["opponent_canonical_label"]) for row in rows).items())
            ),
        },
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if value is None:
        return ""
    return value


def table_csv_text(rows: list[dict[str, Any]]) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(REPLAY_ROW_FIELDS), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_value(row.get(field)) for field in REPLAY_ROW_FIELDS})
    return stream.getvalue()


def validate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-validate every row against the versioned schema before export."""
    for index, row in enumerate(rows):
        validate_replay_row(row, index=index)
    return rows


def write_replay_table(
    rows: list[dict[str, Any]],
    out_dir: str | Path,
    *,
    run_id: str | None = None,
    command: str | list[str] | None = None,
    source_run_ids: list[str] | None = None,
    write_csv: bool = True,
) -> ReplayTableExport:
    """Validate and write the JSONL table, optional CSV, and manifest."""
    validate_rows(rows)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    jsonl_path = write_jsonl(out_path / JSONL_FILENAME, rows)

    csv_path: Path | None = None
    if write_csv:
        csv_path = write_text(out_path / CSV_FILENAME, table_csv_text(rows))

    summary = table_summary(rows)
    outputs: dict[str, str] = {"replay_table_jsonl": str(jsonl_path)}
    if csv_path is not None:
        outputs["replay_table_csv"] = str(csv_path)

    manifest = build_manifest(
        run_id=run_id or out_path.name,
        kind=REPLAY_TABLE_KIND,
        command=command or "",
        inputs={
            "source_kind": "opponent_mixtures",
            "source_run_ids": source_run_ids or summary["source_run_ids"],
        },
        outputs=outputs,
        summary=summary,
    )
    manifest_path = write_manifest(out_path / MANIFEST_FILENAME, manifest)
    return ReplayTableExport(
        out_dir=out_path,
        jsonl_path=jsonl_path,
        csv_path=csv_path,
        manifest_path=manifest_path,
        summary=summary,
    )


def read_replay_table(jsonl_path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL replay table back, validating each row."""
    path = Path(jsonl_path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(validate_replay_row(row, index=index))
    return rows
