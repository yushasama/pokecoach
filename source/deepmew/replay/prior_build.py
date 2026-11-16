"""Build the offline replay-prior count table from decision replay rows.

Reads `deepmew.replay_decision_row.v1` rows, filters crashed/timed-out (and
optionally non-matching agents), aggregates win counts per
`select_context + option_type`, and emits a `deepmew.replay_prior.v1` artifact
with smoothed win-advantage entries. Pure offline statistics; no gameplay code,
no online behavior, no value model.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from deepmew.replay.decision_schema import SCHEMA_VERSION as DECISION_SCHEMA_VERSION
from deepmew.replay.prior_schema import KEY_FIELDS, SCHEMA_VERSION, TARGET, validate_prior_artifact

DECISION_TABLE_FILENAME = "replay_decision_table.jsonl"

# Decision-row fields the prior builder consumes.
_REQUIRED_ROW_FIELDS = (
    "schema_version",
    "select_context",
    "chosen_option_types",
    "agent_win",
    "crashed",
    "timed_out",
    "agent_label",
    "policy_version",
)

LOW_COVERAGE_MULTIPLIER = 10  # low_coverage if rows_used < LOW_COVERAGE_MULTIPLIER * min_count

# Stable per-decision key used to de-duplicate rows during input aggregation, so
# running `--all` over a source root holding both a standalone decision table and
# an aggregate that contains it never double-counts the shared rows. `frame_id`
# already encodes game_id + decision_index within a trace run; trace_run_id and
# source_run_id are both included so copies of one decision collapse while truly
# distinct decisions stay distinct.
DEDUPE_KEY_FIELDS = ("trace_run_id", "source_run_id", "game_id", "frame_id", "decision_index")

# Transient per-row tag (not a schema field) recording which decision table a row
# was read from, so duplicate-source-table coverage is cheap to report.
_SOURCE_TABLE_TAG = "__dedupe_source_table__"


class ReplayPriorBuildError(ValueError):
    """Raised when decision rows cannot be turned into a prior artifact."""


@dataclass
class PriorBuildResult:
    artifact: dict[str, Any]
    stats: dict[str, Any]
    warnings: dict[str, bool]
    source_table_paths: list[str] = field(default_factory=list)


def resolve_decision_table_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_dir():
        return p / DECISION_TABLE_FILENAME
    return p


def read_decision_rows(paths: list[str | Path]) -> tuple[list[dict[str, Any]], list[str]]:
    """Read decision rows from one or more decision tables (file or run dir)."""
    rows: list[dict[str, Any]] = []
    table_paths: list[str] = []
    for path in paths:
        table_path = resolve_decision_table_path(path)
        if not table_path.is_file():
            raise ReplayPriorBuildError(f"{DECISION_TABLE_FILENAME} not found: {table_path}")
        table_paths.append(str(table_path))
        with table_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ReplayPriorBuildError(f"{table_path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(row, dict):
                    raise ReplayPriorBuildError(f"{table_path}:{line_number}: row must be a JSON object")
                row[_SOURCE_TABLE_TAG] = str(table_path)
                rows.append(row)
    return rows, table_paths


def _check_required_fields(row: dict[str, Any], *, index: int) -> None:
    missing = [field for field in _REQUIRED_ROW_FIELDS if field not in row]
    if missing:
        raise ReplayPriorBuildError(f"decision row {index}: missing required fields: {', '.join(sorted(missing))}")
    if row["schema_version"] != DECISION_SCHEMA_VERSION:
        raise ReplayPriorBuildError(
            f"decision row {index}: unsupported schema_version {row['schema_version']!r}; "
            f"expected {DECISION_SCHEMA_VERSION!r}"
        )
    if not isinstance(row["chosen_option_types"], list):
        raise ReplayPriorBuildError(f"decision row {index}: chosen_option_types must be a list")
    for flag in ("agent_win", "crashed", "timed_out"):
        if not isinstance(row[flag], bool):
            raise ReplayPriorBuildError(f"decision row {index}: {flag} must be a boolean")


def _dedupe_key(row: dict[str, Any]) -> tuple[Any, ...] | None:
    """Stable per-decision key, or None if the row lacks a complete key (kept as-is)."""
    key = tuple(row.get(field) for field in DEDUPE_KEY_FIELDS)
    return key if all(part is not None for part in key) else None


def _dedupe_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, list[str]]:
    """Drop duplicate copies of the same decision. Returns (rows, skipped, dup_tables)."""
    seen: set[tuple[Any, ...]] = set()
    deduped: list[dict[str, Any]] = []
    skipped = 0
    duplicate_tables: set[str] = set()
    for row in rows:
        key = _dedupe_key(row)
        if key is not None and key in seen:
            skipped += 1
            table = row.get(_SOURCE_TABLE_TAG)
            if isinstance(table, str):
                duplicate_tables.add(table)
            continue
        if key is not None:
            seen.add(key)
        deduped.append(row)
    return deduped, skipped, sorted(duplicate_tables)


def build_replay_prior(
    rows: list[dict[str, Any]],
    *,
    min_count: int = 30,
    alpha: float = 50.0,
    agent_filter: str | None = None,
    source_table_paths: list[str] | None = None,
) -> PriorBuildResult:
    """Aggregate decision rows into a deterministic replay-prior artifact.

    Input rows are de-duplicated by a stable per-decision key first, so an
    aggregate decision table that contains a standalone table never double-counts.
    """
    if min_count < 0:
        raise ReplayPriorBuildError("min_count must be non-negative")
    if alpha < 0:
        raise ReplayPriorBuildError("alpha must be non-negative")

    rows_before_dedupe = len(rows)
    rows, duplicate_rows_skipped, duplicate_source_tables = _dedupe_rows(rows)

    cell_count: dict[tuple[str, str], int] = defaultdict(int)
    cell_wins: dict[tuple[str, str], int] = defaultdict(int)
    context_count: dict[str, int] = defaultdict(int)
    context_wins: dict[str, int] = defaultdict(int)
    global_count = global_wins = 0

    rows_total = len(rows)
    rows_filtered_out = rows_used = rows_no_contribution = 0
    unresolved_rows = 0
    used_agents: set[str] = set()
    used_policies: set[str] = set()
    source_run_ids: set[str] = set()
    source_paths: set[str] = set()

    for index, row in enumerate(rows):
        _check_required_fields(row, index=index)
        if row["crashed"] or row["timed_out"]:
            rows_filtered_out += 1
            continue
        if agent_filter is not None and row["agent_label"] != agent_filter:
            rows_filtered_out += 1
            continue
        option_types = sorted({str(value) for value in row["chosen_option_types"]})
        if not option_types:
            rows_no_contribution += 1
            continue
        rows_used += 1
        used_agents.add(str(row["agent_label"]))
        used_policies.add(str(row["policy_version"]))
        if row.get("source_run_id") is not None:
            source_run_ids.add(str(row["source_run_id"]))
        if row.get("source_path") is not None:
            source_paths.add(str(row["source_path"]))
        if row.get("chosen_resolved") is False:
            unresolved_rows += 1

        context = str(row["select_context"])
        win = bool(row["agent_win"])
        for option_type in option_types:
            key = (context, option_type)
            cell_count[key] += 1
            context_count[context] += 1
            global_count += 1
            if win:
                cell_wins[key] += 1
                context_wins[context] += 1
                global_wins += 1

    per_context = {
        context: {
            "context_count": context_count[context],
            "context_wins": context_wins[context],
            "context_win_rate": context_wins[context] / context_count[context] if context_count[context] else 0.0,
        }
        for context in sorted(context_count)
    }
    global_win_rate = global_wins / global_count if global_count else 0.0

    entries: list[dict[str, Any]] = []
    omitted_low_count = 0
    for key in sorted(cell_count):
        context, option_type = key
        count = cell_count[key]
        wins = cell_wins[key]
        ctx_count = context_count[context]
        ctx_wins = context_wins[context]
        if count < min_count:
            omitted_low_count += 1
            continue
        context_baseline = ctx_wins / ctx_count if ctx_count else 0.0
        smoothed_rate = (wins + alpha * context_baseline) / (count + alpha) if (count + alpha) else context_baseline
        entries.append(
            {
                "select_context": context,
                "option_type": option_type,
                "count": count,
                "wins": wins,
                "win_rate": wins / count if count else 0.0,
                "context_count": ctx_count,
                "context_wins": ctx_wins,
                "context_win_rate": context_baseline,
                "smoothed_rate": smoothed_rate,
                "advantage": smoothed_rate - context_baseline,
            }
        )

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "key_fields": list(KEY_FIELDS),
        "target": TARGET,
        "smoothing": {"alpha": float(alpha), "baseline": "select_context"},
        "min_count": int(min_count),
        "source": {
            "row_schema": DECISION_SCHEMA_VERSION,
            "table_paths": sorted(set(source_table_paths or [])),
            "source_run_ids": sorted(source_run_ids),
            "source_paths": sorted(source_paths),
            "agent_filter": agent_filter,
        },
        "baseline": {
            "global_count": global_count,
            "global_wins": global_wins,
            "global_win_rate": global_win_rate,
            "per_context": per_context,
        },
        "entries": entries,
    }
    validate_prior_artifact(artifact)

    stats = {
        "rows_before_dedupe": rows_before_dedupe,
        "rows_after_dedupe": rows_total,
        "duplicate_rows_skipped": duplicate_rows_skipped,
        "dedupe_key_fields": list(DEDUPE_KEY_FIELDS),
        "duplicate_source_tables": duplicate_source_tables,
        "rows_total": rows_total,
        "rows_used": rows_used,
        "rows_filtered_out": rows_filtered_out,
        "rows_no_contribution": rows_no_contribution,
        "contribution_count": global_count,
        "unresolved_rows": unresolved_rows,
        "entry_count": len(entries),
        "omitted_low_count": omitted_low_count,
        "agents": sorted(used_agents),
        "policies": sorted(used_policies),
        "select_contexts": sorted(context_count),
        "option_types": sorted({option_type for _, option_type in cell_count}),
        "source_run_ids": sorted(source_run_ids),
        "source_paths": sorted(source_paths),
        "low_coverage_threshold": LOW_COVERAGE_MULTIPLIER * min_count,
    }
    warnings = {
        "single_source": len(source_run_ids) <= 1,
        "low_coverage": len(entries) == 0 or rows_used < LOW_COVERAGE_MULTIPLIER * min_count,
        "unresolved_rows_present": unresolved_rows > 0,
        "mixed_agents_present": len(used_agents) > 1,
        "no_entries": len(entries) == 0,
        "duplicates_skipped": duplicate_rows_skipped > 0,
    }
    return PriorBuildResult(
        artifact=artifact,
        stats=stats,
        warnings=warnings,
        source_table_paths=sorted(set(source_table_paths or [])),
    )
