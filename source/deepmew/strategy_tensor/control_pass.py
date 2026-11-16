"""Strict-public coverage receipt for fresh control-pass collection."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from deepmew.artifacts.writers import write_json, write_text
from deepmew.replay.prospective_trace import (
    PUBLIC_PROJECTION_VERSION,
    assert_public_trace_row_safe,
)


SCHEMA_VERSION = "deepmew.control_pass_collection.v1"
MANIFEST_VERSION = "deepmew.run_manifest.v1"
MIN_RAW_CONTROL_PASS_ROWS = 5_000
MIN_GAMES = 50
MIN_MATCHUP_CELLS = 4
MIN_DECKS = 4
MIN_POLICIES = 4


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


def raw_coverage_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    conditions = {
        "minimum_raw_control_pass_rows": int(summary["raw_control_pass_rows"]) >= MIN_RAW_CONTROL_PASS_ROWS,
        "minimum_games": int(summary["games"]) >= MIN_GAMES,
        "minimum_matchup_cells": int(summary["matchup_cells"]) >= MIN_MATCHUP_CELLS,
        "minimum_decks": int(summary["deck_hashes"]) >= MIN_DECKS,
        "minimum_policies": int(summary["policy_ids"]) >= MIN_POLICIES,
        "safety_clean": all(int(summary[key]) == 0 for key in ("crashes", "timeouts", "illegal")),
        "strict_public": int(summary["invalid_public_rows"]) == 0,
        "unique_decisions": int(summary["duplicate_decision_ids"]) == 0,
    }

    return {
        "passed": all(conditions.values()),
        "conditions": conditions,
        "thresholds": {
            "minimum_raw_control_pass_rows": MIN_RAW_CONTROL_PASS_ROWS,
            "minimum_games": MIN_GAMES,
            "minimum_matchup_cells": MIN_MATCHUP_CELLS,
            "minimum_decks": MIN_DECKS,
            "minimum_policies": MIN_POLICIES,
        },
        "boundary": "raw support only; canonical conversion and target eligibility are still required",
    }


def _actor_identity(row: Mapping[str, Any]) -> tuple[str, str]:
    deck = row.get("actor_deck") or {}
    policy = row.get("actor_policy") or {}

    return (
        str(deck.get("deck_hash") or "unattributed"),
        str(policy.get("feature_policy_id") or policy.get("policy_spec") or "unattributed"),
    )


def render_report(result: Mapping[str, Any]) -> str:
    summary = result["summary"]
    gate = result["raw_coverage_gate"]
    lines = [
        "# Fresh strict-public control-pass collection",
        "",
        f"Verdict: **{result['verdict']}**.",
        "",
        f"Collected {summary['decisions']:,} decisions across {summary['games']:,} complete games and {summary['matchup_cells']} matchup cells. {summary['raw_control_pass_rows']:,} rows have an exact `control_passed=true` transition.",
        "",
        f"Raw coverage gate: **{'PASS' if gate['passed'] else 'FAIL'}**. This is not the frozen Q support gate yet: conversion into a new actor-visible canonical dataset must determine how many rows are target-eligible, without reopening the old sealed test.",
        "",
        "## Safety and visibility",
        "",
        f"- Crashes/timeouts/illegal: {summary['crashes']}/{summary['timeouts']}/{summary['illegal']}.",
        f"- Invalid strict-public rows: {summary['invalid_public_rows']}; duplicate decision IDs: {summary['duplicate_decision_ids']}.",
        f"- Deck hashes: {summary['deck_hashes']}; policy IDs: {summary['policy_ids']}; both acting seats: {summary['acting_seats']}.",
        "- Empirical strength remains `not_instrumented`; the unavailable rating snapshot was not converted into a fake bucket.",
        "",
        "## Matchup coverage",
        "",
        "| Matchup | Games | Decisions | Control-pass rows |",
        "|---|---:|---:|---:|",
    ]
    for row in result["by_matchup"]:
        lines.append(
            f"| `{row['matchup']}` | {row['games']:,} | {row['decisions']:,} | "
            f"{row['raw_control_pass_rows']:,} |"
        )
    lines += [
        "",
        "No policy, V, Q, league generation, runtime, submission, or protected default changed.",
        "",
    ]

    return "\n".join(lines)


def summarize_control_pass_collection(
    *,
    collection_root: str | Path,
) -> dict[str, Any]:
    root = Path(collection_root)
    plan_paths = sorted(root.glob("*/plan.json"))
    if not plan_paths:
        raise ValueError(f"no prospective plans found under {root}")
    decision_ids = set()
    duplicate_decisions = 0
    game_ids = set()
    acting_decks = Counter()
    acting_policies = Counter()
    acting_seats = Counter()
    matchup_counts: dict[str, Counter[str]] = {}
    input_files = []
    totals = Counter()
    for plan_path in plan_paths:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if len(plan.get("tranches") or []) < 3:
            raise ValueError(f"plan is not chronological three-tranche collection: {plan_path}")
        if any(tranche.get("status") != "completed" for tranche in plan["tranches"]):
            raise ValueError(f"plan has incomplete tranches: {plan_path}")
        matchup = plan_path.parent.name
        matchup_counts[matchup] = Counter()
        input_files.append({"path": str(plan_path), "sha256": _sha256(plan_path)})
        for tranche in plan["tranches"]:
            manifest_path = plan_path.parent / "tranches" / str(tranche["tranche_id"]) / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            input_files.append({"path": str(manifest_path), "sha256": _sha256(manifest_path)})
            if manifest.get("information_mode") != "public_belief":
                raise ValueError(f"non-public tranche in control-pass collection: {manifest_path}")
            if manifest.get("public_projection_version") != PUBLIC_PROJECTION_VERSION:
                raise ValueError(f"projection version drift: {manifest_path}")
            for key in ("crashes", "timeouts", "illegal"):
                value = int(manifest["safety"][key])
                totals[key] += value
                matchup_counts[matchup][key] += value
            shard_paths = sorted((manifest_path.parent / "shards").glob("decisions_shard_*.jsonl.gz"))
            if len(shard_paths) != int(manifest["coverage"]["games"]):
                raise ValueError(f"one-shard-per-game contract failed: {manifest_path}")
            for shard_path in shard_paths:
                input_files.append({"path": str(shard_path), "sha256": _sha256(shard_path)})
                with gzip.open(shard_path, "rt", encoding="utf-8") as handle:
                    for line in handle:
                        row = json.loads(line)
                        totals["decisions"] += 1
                        matchup_counts[matchup]["decisions"] += 1
                        decision_id = str(row.get("decision_id") or "")
                        if decision_id in decision_ids:
                            duplicate_decisions += 1
                        decision_ids.add(decision_id)
                        game_id = str(row.get("game_id") or "")
                        game_ids.add(game_id)
                        try:
                            assert_public_trace_row_safe(row)
                        except (TypeError, ValueError):
                            totals["invalid_public_rows"] += 1
                        if row.get("control_passed") is True:
                            totals["raw_control_pass_rows"] += 1
                            matchup_counts[matchup]["raw_control_pass_rows"] += 1
                        deck_hash, policy_id = _actor_identity(row)
                        acting_decks[deck_hash] += 1
                        acting_policies[policy_id] += 1
                        acting_seats[str(row.get("actor_seat"))] += 1
        matchup_games = {
            game_id
            for game_id in game_ids
            if f"{plan.get('run_id')}:" in game_id
        }
        matchup_counts[matchup]["games"] = len(matchup_games)

    totals["games"] = len(game_ids)
    summary = {
        "plans": len(plan_paths),
        "matchup_cells": len(matchup_counts),
        "games": totals["games"],
        "decisions": totals["decisions"],
        "raw_control_pass_rows": totals["raw_control_pass_rows"],
        "deck_hashes": len(acting_decks),
        "policy_ids": len(acting_policies),
        "acting_seats": len(acting_seats),
        "crashes": totals["crashes"],
        "timeouts": totals["timeouts"],
        "illegal": totals["illegal"],
        "invalid_public_rows": totals["invalid_public_rows"],
        "duplicate_decision_ids": duplicate_decisions,
        "strength_status": "not_instrumented",
    }
    gate = raw_coverage_gate(summary)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": (
            "RAW_CONTROL_PASS_SUPPORT_PASS_CONVERSION_GATE_PENDING"
            if gate["passed"]
            else "RAW_CONTROL_PASS_SUPPORT_INSUFFICIENT"
        ),
        "summary": summary,
        "raw_coverage_gate": gate,
        "by_matchup": [
            {
                "matchup": matchup,
                "games": counts["games"],
                "decisions": counts["decisions"],
                "raw_control_pass_rows": counts["raw_control_pass_rows"],
            }
            for matchup, counts in sorted(matchup_counts.items())
        ],
        "by_acting_deck": dict(acting_decks.most_common()),
        "by_acting_policy": dict(acting_policies.most_common()),
        "by_acting_seat": dict(acting_seats.most_common()),
        "input_files": input_files,
        "information_mode": "public_belief",
        "public_projection_version": PUBLIC_PROJECTION_VERSION,
        "promotion_status": "data_receipt_only_no_control_authority",
        "protected_surfaces_changed": [],
    }
    result_path = root / "collection_result.json"
    report_path = root / "REPORT.md"
    write_json(result_path, result)
    write_text(report_path, render_report(result))
    manifest = {
        "schema_version": MANIFEST_VERSION,
        "run_id": "control-pass-strategy-v1-20260711",
        "kind": "strict_public_control_pass_collection",
        "status": "completed",
        "created_at_utc": result["created_at_utc"],
        "inputs": {
            "collection_root": str(root),
            "input_file_count": len(input_files),
            "input_set_sha256": hashlib.sha256(
                json.dumps(input_files, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "implementation_sha256": _sha256(Path(__file__)),
            "entrypoint_sha256": _sha256(Path(__file__).resolve().parents[2] / "scripts" / "summarize_control_pass_collection.py"),
        },
        "outputs": {
            "result": {"path": str(result_path), "bytes": result_path.stat().st_size, "sha256": _sha256(result_path)},
            "report": {"path": str(report_path), "bytes": report_path.stat().st_size, "sha256": _sha256(report_path)},
        },
        "summary": {
            "verdict": result["verdict"],
            "games": summary["games"],
            "raw_control_pass_rows": summary["raw_control_pass_rows"],
            "raw_coverage_gate_passed": gate["passed"],
            "canonical_eligibility_gate": "pending",
        },
        "protected_surfaces_changed": [],
    }
    write_json(root / "collection_manifest.json", manifest)

    return result
