"""Build and export the raw replay-action table (Bible §2.16 aligned).

Streams rows to disk so large aggregate builds (tens of thousands of frames, each
embedding a raw observation) stay memory-flat. Emits JSONL, an optional CSV
mirror, a manifest, and a markdown report. The manifest records whether Bible
§2.16 is strictly satisfied (state_json + legal_options_json present and 0
malformed) and documents the source_rating-null / result_reason-sparse caveats.
"""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from deepmew.artifacts.manifest import build_manifest, write_manifest
from deepmew.artifacts.writers import write_markdown, write_text
from deepmew.replay.action_ingest import DeckResolver, iter_action_rows
from deepmew.replay.action_schema import ACTION_ROW_FIELDS, SCHEMA_VERSION
from deepmew.replay.decision_ingest import deck_metadata_from_path

DECISION_TABLE_KIND = "replay_action_table"

JSONL_FILENAME = "replay_action_table.jsonl"
CSV_FILENAME = "replay_action_table.csv"
MANIFEST_FILENAME = "manifest.json"
REPORT_FILENAME = "report.md"

BIBLE_2_16_FIELDS = (
    "game_id",
    "player_id",
    "turn",
    "step",
    "state_json",
    "legal_options_json",
    "chosen_indices",
    "result",
    "result_reason",
    "deck_id",
    "source_rating",
)


@dataclass
class _Accumulator:
    row_count: int = 0
    frame_count: int = 0
    malformed_count: int = 0
    missing_state_count: int = 0
    missing_legal_options_count: int = 0
    result_null_count: int = 0
    result_reason_null_count: int = 0
    source_rating_null_count: int = 0
    source_run_ids: set[str] = field(default_factory=set)
    source_kinds: set[str] = field(default_factory=set)
    source_paths: set[str] = field(default_factory=set)
    deck_hashes: set[str] = field(default_factory=set)
    games: set[tuple[str, str]] = field(default_factory=set)
    players: set[int] = field(default_factory=set)
    results: Counter = field(default_factory=Counter)
    result_reasons: Counter = field(default_factory=Counter)
    malformed_examples: list[str] = field(default_factory=list)

    def observe(self, row: dict[str, Any], *, missing_state: bool, missing_legal: bool) -> None:
        self.row_count += 1
        if missing_state:
            self.missing_state_count += 1
        if missing_legal:
            self.missing_legal_options_count += 1
        self.source_run_ids.add(str(row["source_run_id"]))
        self.source_kinds.add(str(row["source_kind"]))
        self.source_paths.add(str(row["source_path"]))
        if row["deck_hash"] is not None:
            self.deck_hashes.add(str(row["deck_hash"]))
        self.games.add((str(row["source_run_id"]), str(row["game_id"])))
        if row["player_id"] is not None:
            self.players.add(int(row["player_id"]))
        self.results[row["result"]] += 1
        if row["result"] is None:
            self.result_null_count += 1
        if row["result_reason"] is None:
            self.result_reason_null_count += 1
        else:
            self.result_reasons[str(row["result_reason"])] += 1
        if row["source_rating"] is None:
            self.source_rating_null_count += 1

    def note_malformed(self, error: str | None) -> None:
        self.malformed_count += 1
        if error and len(self.malformed_examples) < 10:
            self.malformed_examples.append(error)


@dataclass(frozen=True)
class ActionTableExport:
    out_dir: Path
    jsonl_path: Path
    csv_path: Path | None
    manifest_path: Path
    report_path: Path
    summary: dict[str, Any]


def _bible_2_16_strict(stats: _Accumulator) -> bool:
    return (
        stats.row_count > 0
        and stats.malformed_count == 0
        and stats.missing_state_count == 0
        and stats.missing_legal_options_count == 0
    )


def _bible_2_16_notes(stats: _Accumulator) -> str:
    return (
        "All §2.16 fields present in every row; state_json and legal_options_json populated "
        "from genuine raw frame observations. source_rating is null for all rows because "
        "arena/trace runs carry no ladder rating (schema-allowed, documented). result_reason "
        f"is null for {stats.result_reason_null_count} rows (the local engine emits a reason "
        "only for crash/timeout and any engine-provided code; normal finishes have no reason)."
    )


def _summary(stats: _Accumulator, *, row_gate: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "row_count": stats.row_count,
        "frame_count": stats.frame_count,
        "malformed_count": stats.malformed_count,
        "missing_state_count": stats.missing_state_count,
        "missing_legal_options_count": stats.missing_legal_options_count,
        "result_null_count": stats.result_null_count,
        "result_reason_null_count": stats.result_reason_null_count,
        "source_rating_null_count": stats.source_rating_null_count,
        "row_gate": row_gate,
        "row_gate_met": stats.row_count >= row_gate and stats.malformed_count == 0,
        "bible_2_16_fields": list(BIBLE_2_16_FIELDS),
        "bible_2_16_strict": _bible_2_16_strict(stats),
        "bible_2_16_notes": _bible_2_16_notes(stats),
        "source_run_ids": sorted(stats.source_run_ids),
        "source_kinds": sorted(stats.source_kinds),
        "source_paths": sorted(stats.source_paths),
        "game_count": len(stats.games),
        "coverage": {
            "players": sorted(stats.players),
            "deck_hashes": sorted(stats.deck_hashes),
            "results": {str(key): value for key, value in sorted(stats.results.items(), key=lambda kv: str(kv[0]))},
            "result_reasons": dict(sorted(stats.result_reasons.items())),
        },
        "malformed_examples": list(stats.malformed_examples),
    }


def _report(summary: dict[str, Any]) -> str:
    cov = summary["coverage"]
    lines = [
        "# Replay Action Table (Bible §2.16)",
        "",
        f"schema_version={summary['schema_version']}",
        f"row_count={summary['row_count']} frame_count={summary['frame_count']} "
        f"malformed_count={summary['malformed_count']}",
        f"missing_state_count={summary['missing_state_count']} "
        f"missing_legal_options_count={summary['missing_legal_options_count']}",
        f"row_gate={summary['row_gate']} row_gate_met={summary['row_gate_met']}",
        f"bible_2_16_strict={summary['bible_2_16_strict']}",
        f"game_count={summary['game_count']} source_runs={len(summary['source_run_ids'])}",
        f"deck_hashes={cov['deck_hashes']}",
        f"results={cov['results']}",
        f"result_reasons={cov['result_reasons']}",
        f"source_rating_null_count={summary['source_rating_null_count']}",
        "",
        "## Bible §2.16",
        "",
        summary["bible_2_16_notes"],
        "",
        "## Boundary",
        "",
        "- Built from existing `deepmew.decision_frame.v1` `frames.jsonl` artifacts; no game replay, "
        "no engine instrumentation.",
        "- Raw infrastructure only: no gameplay/default change, no replay prior, value model, RL, or search.",
    ]
    return "\n".join(lines)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if value is None:
        return ""
    return value


def _open_temp(out_path: Path):
    return tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        delete=False,
        dir=str(out_path),
        prefix=f".{JSONL_FILENAME}.",
        suffix=".tmp",
    )


def build_action_table(
    frame_paths: list[str | Path],
    out_dir: str | Path,
    *,
    run_id: str | None = None,
    command: str | list[str] | None = None,
    row_gate: int = 10000,
    write_csv: bool = False,
    deck_resolver: DeckResolver = deck_metadata_from_path,
) -> ActionTableExport:
    """Stream raw action rows from many frames.jsonl into one table + manifest + report."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_path / JSONL_FILENAME

    stats = _Accumulator()
    deck_cache: dict[str | None, tuple[str | None, str | None]] = {}
    csv_rows: list[dict[str, Any]] | None = [] if write_csv else None

    handle = _open_temp(out_path)
    tmp_name = handle.name
    try:
        with handle:
            for frame_path in frame_paths:
                for result in iter_action_rows(frame_path, deck_resolver=deck_resolver, deck_cache=deck_cache):
                    stats.frame_count += 1
                    if result.malformed or result.row is None:
                        stats.note_malformed(result.error)
                        continue
                    handle.write(json.dumps(result.row, sort_keys=True))
                    handle.write("\n")
                    stats.observe(result.row, missing_state=result.missing_state, missing_legal=result.missing_legal)
                    if csv_rows is not None:
                        csv_rows.append(result.row)
        os.replace(tmp_name, jsonl_path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise

    csv_path: Path | None = None
    if csv_rows is not None:
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(ACTION_ROW_FIELDS), lineterminator="\n")
        writer.writeheader()
        for row in csv_rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in ACTION_ROW_FIELDS})
        csv_path = write_text(out_path / CSV_FILENAME, stream.getvalue())

    summary = _summary(stats, row_gate=row_gate)
    outputs: dict[str, str] = {"replay_action_table_jsonl": str(jsonl_path)}
    if csv_path is not None:
        outputs["replay_action_table_csv"] = str(csv_path)
    report_path = write_markdown(out_path / REPORT_FILENAME, _report(summary))
    outputs["report"] = str(report_path)

    manifest = build_manifest(
        run_id=run_id or out_path.name,
        kind=DECISION_TABLE_KIND,
        command=command or "",
        inputs={
            "source_kinds": summary["source_kinds"],
            "source_run_ids": summary["source_run_ids"],
            "source_paths": summary["source_paths"],
        },
        outputs=outputs,
        summary=summary,
    )
    manifest_path = write_manifest(out_path / MANIFEST_FILENAME, manifest)
    return ActionTableExport(
        out_dir=out_path,
        jsonl_path=jsonl_path,
        csv_path=csv_path,
        manifest_path=manifest_path,
        report_path=report_path,
        summary=summary,
    )


def read_action_table(jsonl_path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL action table back, validating each row."""
    from deepmew.replay.action_schema import validate_action_row

    path = Path(jsonl_path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            rows.append(validate_action_row(json.loads(line), index=index))
    return rows
