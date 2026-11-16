"""Write the offline replay-prior artifact, manifest, and report."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepmew.artifacts.manifest import build_manifest, write_manifest
from deepmew.artifacts.writers import write_json, write_markdown
from deepmew.replay.prior_build import PriorBuildResult
from deepmew.replay.prior_schema import SCHEMA_VERSION, validate_prior_artifact

REPLAY_PRIOR_KIND = "replay_prior"

PRIOR_FILENAME = "replay_prior.json"
MANIFEST_FILENAME = "manifest.json"
REPORT_FILENAME = "report.md"

_TOP_N = 10


@dataclass(frozen=True)
class PriorExport:
    out_dir: Path
    prior_path: Path
    manifest_path: Path
    report_path: Path
    summary: dict[str, Any]


def _manifest_summary(result: PriorBuildResult) -> dict[str, Any]:
    artifact = result.artifact
    stats = result.stats
    return {
        "schema_version": SCHEMA_VERSION,
        "source_row_schema": artifact["source"]["row_schema"],
        "source_table_paths": result.source_table_paths,
        "source_run_ids": stats["source_run_ids"],
        "source_paths": stats["source_paths"],
        "rows_before_dedupe": stats["rows_before_dedupe"],
        "rows_after_dedupe": stats["rows_after_dedupe"],
        "duplicate_rows_skipped": stats["duplicate_rows_skipped"],
        "dedupe_key_fields": stats["dedupe_key_fields"],
        "duplicate_source_tables": stats["duplicate_source_tables"],
        "rows_total": stats["rows_total"],
        "rows_used": stats["rows_used"],
        "rows_filtered_out": stats["rows_filtered_out"],
        "rows_no_contribution": stats["rows_no_contribution"],
        "contribution_count": stats["contribution_count"],
        "min_count": artifact["min_count"],
        "alpha": artifact["smoothing"]["alpha"],
        "agent_filter": artifact["source"]["agent_filter"],
        "agent_coverage": stats["agents"],
        "policy_coverage": stats["policies"],
        "select_context_coverage": stats["select_contexts"],
        "option_type_coverage": stats["option_types"],
        "entry_count": stats["entry_count"],
        "omitted_low_count": stats["omitted_low_count"],
        "unresolved_rows": stats["unresolved_rows"],
        "global_win_rate": artifact["baseline"]["global_win_rate"],
        "warnings": result.warnings,
    }


def _format_entry(entry: dict[str, Any]) -> str:
    return (
        f"| {entry['select_context']} | {entry['option_type']} | {entry['count']} | "
        f"{entry['win_rate']:.3f} | {entry['context_win_rate']:.3f} | "
        f"{entry['smoothed_rate']:.3f} | {entry['advantage']:+.4f} |"
    )


def _report(result: PriorBuildResult) -> str:
    artifact = result.artifact
    stats = result.stats
    entries = artifact["entries"]
    by_adv = sorted(entries, key=lambda e: (e["advantage"], e["select_context"], e["option_type"]))
    top_pos = list(reversed(by_adv[-_TOP_N:]))
    top_neg = by_adv[:_TOP_N]
    header = "| select_context | option_type | count | win_rate | ctx_win_rate | smoothed | advantage |"
    sep = "|---|---|---:|---:|---:|---:|---:|"

    lines = [
        "# Replay Prior (offline, smoothed win advantage)",
        "",
        f"schema_version={artifact['schema_version']} key={artifact['key_fields']} target={artifact['target']}",
        f"min_count={artifact['min_count']} alpha={artifact['smoothing']['alpha']} "
        f"agent_filter={artifact['source']['agent_filter']}",
        "",
        "## Source",
        "",
        f"row_schema={artifact['source']['row_schema']}",
        f"source_run_ids={stats['source_run_ids']}",
        f"table_paths={result.source_table_paths}",
        "",
        "## Row counts",
        "",
        f"rows_before_dedupe={stats['rows_before_dedupe']} rows_after_dedupe={stats['rows_after_dedupe']} "
        f"duplicate_rows_skipped={stats['duplicate_rows_skipped']}",
        f"rows_total={stats['rows_total']} rows_used={stats['rows_used']} "
        f"rows_filtered_out={stats['rows_filtered_out']} rows_no_contribution={stats['rows_no_contribution']}",
        f"contribution_count={stats['contribution_count']} entry_count={stats['entry_count']} "
        f"omitted_low_count={stats['omitted_low_count']} unresolved_rows={stats['unresolved_rows']}",
        "",
        "## De-duplication",
        "",
        f"dedupe_key_fields={stats['dedupe_key_fields']}",
        f"duplicate_rows_skipped={stats['duplicate_rows_skipped']}",
        f"duplicate_source_tables={stats['duplicate_source_tables']}",
        "- Historical artifacts are preserved; duplicate copies of the same decision (a standalone "
        "table also contained inside an aggregate table) are ignored during aggregation, not deleted.",
        "",
        "## Baselines",
        "",
        f"global_win_rate={artifact['baseline']['global_win_rate']:.4f} "
        f"(wins={artifact['baseline']['global_wins']}/{artifact['baseline']['global_count']})",
        "",
        "| select_context | context_count | context_win_rate |",
        "|---|---:|---:|",
    ]
    for context, info in artifact["baseline"]["per_context"].items():
        lines.append(f"| {context} | {info['context_count']} | {info['context_win_rate']:.3f} |")

    lines += ["", "## Top positive advantages", "", header, sep]
    lines += [_format_entry(entry) for entry in top_pos] or ["| (none) |"]
    lines += ["", "## Top negative advantages", "", header, sep]
    lines += [_format_entry(entry) for entry in top_neg] or ["| (none) |"]

    lines += ["", "## Warnings", ""]
    lines += [f"- {name}: {value}" for name, value in result.warnings.items()]

    lines += [
        "",
        "## Boundary",
        "",
        "- This is an offline count-table prior, NOT a value model and NOT online behavior.",
        "- The win label is the whole-game `agent_win` outcome attributed to every decision "
        "(crude credit assignment), used only as a weak smoothed signal.",
        "- The artifact is never loaded by the eval/runtime submission path; no gameplay/default "
        "`v2.4` behavior is changed by building it.",
    ]
    return "\n".join(lines)


def write_replay_prior(
    result: PriorBuildResult,
    out_dir: str | Path,
    *,
    run_id: str | None = None,
    command: str | list[str] | None = None,
) -> PriorExport:
    validate_prior_artifact(result.artifact)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    prior_path = write_json(out_path / PRIOR_FILENAME, result.artifact)
    report_path = write_markdown(out_path / REPORT_FILENAME, _report(result))
    summary = _manifest_summary(result)
    manifest = build_manifest(
        run_id=run_id or out_path.name,
        kind=REPLAY_PRIOR_KIND,
        command=command or "",
        inputs={
            "row_schema": result.artifact["source"]["row_schema"],
            "source_table_paths": result.source_table_paths,
            "min_count": result.artifact["min_count"],
            "alpha": result.artifact["smoothing"]["alpha"],
            "agent_filter": result.artifact["source"]["agent_filter"],
        },
        outputs={"replay_prior": str(prior_path), "report": str(report_path)},
        summary=summary,
    )
    manifest_path = write_manifest(out_path / MANIFEST_FILENAME, manifest)
    return PriorExport(
        out_dir=out_path,
        prior_path=prior_path,
        manifest_path=manifest_path,
        report_path=report_path,
        summary=summary,
    )


def read_replay_prior(path: str | Path) -> dict[str, Any]:
    import json

    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_prior_artifact(artifact)
