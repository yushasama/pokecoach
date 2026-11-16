"""Build and export the normalized per-decision replay table.

JSONL row stream, optional CSV mirror, and a self-describing manifest carrying
schema version, source run ids, row/frame counts, resolution counts, failure and
draw counts, and agent/opponent/policy/select coverage.
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
from deepmew.artifacts.writers import write_jsonl, write_text
from deepmew.replay.decision_schema import (
    DECISION_ROW_FIELDS,
    SCHEMA_VERSION,
    validate_decision_row,
)


DECISION_TABLE_KIND = "replay_decision_table"

JSONL_FILENAME = "replay_decision_table.jsonl"
CSV_FILENAME = "replay_decision_table.csv"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class DecisionTableExport:
    out_dir: Path
    jsonl_path: Path
    csv_path: Path | None
    manifest_path: Path
    summary: dict[str, Any]


def _sorted_unique(values: Iterable[Any]) -> list[Any]:
    return sorted({value for value in values if value is not None})


def decision_table_summary(
    rows: list[dict[str, Any]],
    *,
    frame_count: int,
    skipped_unresolved: int,
    pre_enrichment_frames: int,
    source_paths: list[str],
) -> dict[str, Any]:
    resolved = sum(1 for row in rows if row["chosen_resolved"])
    return {
        "schema_version": SCHEMA_VERSION,
        "row_count": len(rows),
        "frame_count": frame_count,
        "skipped_unresolved": skipped_unresolved,
        "pre_enrichment_frames": pre_enrichment_frames,
        "resolved_chosen": resolved,
        "unresolved_chosen": len(rows) - resolved,
        "crashes": sum(1 for row in rows if row["crashed"]),
        "timeouts": sum(1 for row in rows if row["timed_out"]),
        "draws": sum(1 for row in rows if row["draw"]),
        "agent_wins": sum(1 for row in rows if row["agent_win"]),
        "source_run_ids": _sorted_unique(row["source_run_id"] for row in rows),
        "source_kinds": _sorted_unique(row["source_kind"] for row in rows),
        "source_paths": sorted(set(source_paths)),
        "coverage": {
            "agents": _sorted_unique(row["agent_label"] for row in rows),
            "opponents": _sorted_unique(row["opponent_label"] for row in rows),
            "policies": _sorted_unique(row["policy_version"] for row in rows),
            "deck_hashes": _sorted_unique(row["deck_hash"] for row in rows),
            "select_types": dict(sorted(Counter(str(row["select_type"]) for row in rows).items())),
            "select_contexts": dict(sorted(Counter(str(row["select_context"]) for row in rows).items())),
        },
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if value is None:
        return ""
    return value


def decision_table_csv_text(rows: list[dict[str, Any]]) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(DECISION_ROW_FIELDS), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_value(row.get(field)) for field in DECISION_ROW_FIELDS})
    return stream.getvalue()


def validate_decision_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for index, row in enumerate(rows):
        validate_decision_row(row, index=index)
    return rows


def write_decision_replay_table(
    rows: list[dict[str, Any]],
    out_dir: str | Path,
    *,
    run_id: str | None = None,
    command: str | list[str] | None = None,
    source_run_ids: list[str] | None = None,
    source_paths: list[str] | None = None,
    frame_count: int | None = None,
    skipped_unresolved: int = 0,
    pre_enrichment_frames: int = 0,
    write_csv: bool = True,
) -> DecisionTableExport:
    """Validate and write the JSONL table, optional CSV, and manifest."""
    validate_decision_rows(rows)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    jsonl_path = write_jsonl(out_path / JSONL_FILENAME, rows)
    csv_path: Path | None = None
    if write_csv:
        csv_path = write_text(out_path / CSV_FILENAME, decision_table_csv_text(rows))

    summary = decision_table_summary(
        rows,
        frame_count=frame_count if frame_count is not None else len(rows),
        skipped_unresolved=skipped_unresolved,
        pre_enrichment_frames=pre_enrichment_frames,
        source_paths=source_paths or [],
    )
    outputs: dict[str, str] = {"replay_decision_table_jsonl": str(jsonl_path)}
    if csv_path is not None:
        outputs["replay_decision_table_csv"] = str(csv_path)

    manifest = build_manifest(
        run_id=run_id or out_path.name,
        kind=DECISION_TABLE_KIND,
        command=command or "",
        inputs={
            "source_kinds": summary["source_kinds"],
            "source_run_ids": source_run_ids or summary["source_run_ids"],
            "source_paths": summary["source_paths"],
        },
        outputs=outputs,
        summary=summary,
    )
    manifest_path = write_manifest(out_path / MANIFEST_FILENAME, manifest)
    return DecisionTableExport(
        out_dir=out_path,
        jsonl_path=jsonl_path,
        csv_path=csv_path,
        manifest_path=manifest_path,
        summary=summary,
    )


def read_decision_replay_table(jsonl_path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL decision replay table back, validating each row."""
    path = Path(jsonl_path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            rows.append(validate_decision_row(json.loads(line), index=index))
    return rows
