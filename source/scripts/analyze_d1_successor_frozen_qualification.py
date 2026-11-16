"""Analyze the frozen D1-successor 14,336-game qualification panel."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts import analyze_h2_x3_frozen_qualification as analysis


FROZEN_GAMES = 14_336
FROZEN_TASKS = 34
CONTROL_IDS = ("h2", "e1", "r7", "x2", "r12_hardtail")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")

    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_batches(root: Path, prefix: str, focal: str) -> list[list[analysis.PayoffGame]]:
    return [
        analysis.load_payoff_games(root / "payoff" / f"{prefix}_{batch}", focal)
        for batch in range(1, 4)
    ]


def _flatten(batches: Sequence[Sequence[analysis.PayoffGame]]) -> list[analysis.PayoffGame]:
    return [game for batch in batches for game in batch]


def _wilson_severe(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    candidate_interval = candidate["decisive_wilson95"]
    baseline_interval = baseline["decisive_wilson95"]

    return bool(candidate_interval[1] < baseline_interval[0])


def _load_execution_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    if manifest.get("status") != "PASS":
        raise ValueError("qualification execution manifest is not PASS")
    if int(manifest.get("planned_games", -1)) != FROZEN_GAMES:
        raise ValueError("qualification execution manifest has the wrong game budget")
    if int(manifest.get("completed_games", -1)) != FROZEN_GAMES:
        raise ValueError("qualification execution manifest is not exactly complete")
    results = manifest.get("results")
    if not isinstance(results, list) or len(results) != FROZEN_TASKS:
        raise ValueError("qualification execution manifest has the wrong task count")
    names = [row.get("name") for row in results if isinstance(row, dict)]
    if len(names) != FROZEN_TASKS or len(set(names)) != FROZEN_TASKS:
        raise ValueError("qualification execution manifest task identities are incomplete or duplicated")
    if any(row.get("status") not in {"PASS", "SKIPPED_COMPLETE"} for row in results):
        raise ValueError("qualification execution manifest contains an incomplete task")

    return manifest


def _same_schedule_control(
    root: Path,
    opponent_id: str,
    *,
    seed_offset: int,
) -> dict[str, Any]:
    successor_batches = _load_batches(root, f"successor_vs_{opponent_id}_batch", "SUCCESSOR")
    d1_batches = _load_batches(root, f"d1_vs_{opponent_id}_batch", "D1")
    successor = analysis.summarize_games(_flatten(successor_batches))
    d1 = analysis.summarize_games(_flatten(d1_batches))
    delta = analysis.compare_game_scores(successor_batches, d1_batches, seed_offset=seed_offset)
    delta.update({
        "definition": f"SUCCESSOR point rate versus {opponent_id} minus D1 point rate versus {opponent_id}",
        "successor": successor,
        "d1_same_schedule": d1,
        "retention_floor_pass": delta["estimate"] >= -0.05,
        "wilson_severe_regression": _wilson_severe(successor, d1),
        "fresh_batch_point_rate_deltas": [
            float(
                np.mean([game.score for game in successor_batch])
                - np.mean([game.score for game in d1_batch])
            )
            for successor_batch, d1_batch in zip(successor_batches, d1_batches, strict=True)
        ],
    })

    return delta


def _hardtail_slices(root: Path) -> dict[str, Any]:
    successor = _load_batches(root, "successor_vs_r12_hardtail_batch", "SUCCESSOR")
    d1 = _load_batches(root, "d1_vs_r12_hardtail_batch", "D1")
    contracts = (
        ("first_attack_turn_capped_11", True, "tempo"),
        ("attack_by_turn_3", False, "tempo"),
        ("first_prize_turn_capped_11", True, "prize_plan"),
        ("prize_by_turn_4", False, "prize_plan"),
        ("zero_prize_game", True, "prize_plan"),
        ("positive_prize_lead_by_turn_5", False, "prize_plan"),
        ("mega_starmie_evolution_by_turn_4", False, "evolution"),
    )
    slices = {}
    for offset, (metric, lower_is_better, family) in enumerate(contracts, start=100):
        receipt = analysis.compare_telemetry_metric(
            successor,
            d1,
            metric,
            lower_is_better=lower_is_better,
            seed_offset=offset,
        )
        receipt["family"] = family
        slices[metric] = receipt

    return slices


def _gate_receipt(path: Path) -> dict[str, Any]:
    receipt = _read_json(path)
    required = {
        "status": "PASS",
        "leakage": False,
        "runtime_contamination": False,
        "successor_checkpoint_finite": True,
    }
    mismatches = {
        key: {"expected": expected, "observed": receipt.get(key)}
        for key, expected in required.items()
        if receipt.get(key) != expected
    }

    return {"path": str(path), "receipt": receipt, "required": required, "mismatches": mismatches, "pass": not mismatches}


def analyze(root: Path, gate_path: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = _load_execution_manifest(manifest_path)
    primary_batches = _load_batches(root, "successor_vs_d1_batch", "SUCCESSOR")
    primary_games = _flatten(primary_batches)
    primary = analysis.summarize_games(primary_games)
    primary["fresh_batches"] = [analysis.summarize_games(batch) for batch in primary_batches]
    primary["point_rate_above_half"] = primary["point_rate"] > 0.5
    primary["decisive_wilson_lower_above_half"] = primary["decisive_wilson95"][0] > 0.5
    primary["every_fresh_batch_above_half"] = all(
        batch["point_rate"] > 0.5 for batch in primary["fresh_batches"]
    )
    primary["both_seats_above_half"] = all(
        seat["point_rate"] > 0.5 for seat in primary["by_seat"].values()
    )
    primary["gate_pass"] = all((
        primary["point_rate_above_half"],
        primary["decisive_wilson_lower_above_half"],
        primary["every_fresh_batch_above_half"],
        primary["both_seats_above_half"],
    ))

    controls = {
        opponent_id: _same_schedule_control(root, opponent_id, seed_offset=20 + index)
        for index, opponent_id in enumerate(CONTROL_IDS)
    }
    x2_successor_batches = _load_batches(root, "successor_vs_x2_batch", "X2")
    x2_d1_batches = _load_batches(root, "d1_vs_x2_batch", "X2")
    x2_edge = analysis.compare_game_scores(x2_successor_batches, x2_d1_batches, seed_offset=50)
    x2_edge.update({
        "definition": "X2 point rate versus SUCCESSOR minus X2 point rate versus D1; negative is reduction",
        "x2_vs_successor": analysis.summarize_games(_flatten(x2_successor_batches)),
        "x2_vs_d1": analysis.summarize_games(_flatten(x2_d1_batches)),
    })
    x2_edge["gate_pass"] = (
        x2_edge["x2_vs_successor"]["point_rate"]
        <= x2_edge["x2_vs_d1"]["point_rate"] + 0.05
        and x2_edge["x2_vs_successor"]["point_rate"] <= 0.5
    )

    panel = analysis.summarize_panel(
        analysis.load_panel_games(root / "shared_panel" / "complete_reference_shared_panel"),
        "D1",
        "SUCCESSOR",
    )
    reference_delta = (
        panel["participants"]["SUCCESSOR"]["point_rate"]
        - panel["participants"]["D1"]["point_rate"]
    )
    panel["aggregate_point_rate_delta"] = reference_delta
    panel["gate_pass"] = (
        not panel["wilson_severe_regression_cells"]
        and not panel["paired_severe_regression_cells"]
        and reference_delta >= -0.03
    )
    hardtail = controls["r12_hardtail"]
    hardtail["public_telemetry_slices"] = _hardtail_slices(root)
    hardtail["gate_pass"] = hardtail["retention_floor_pass"] and not hardtail["wilson_severe_regression"]

    all_rows = []
    for task in manifest["tasks"]:
        all_rows.extend(analysis._read_jsonl(Path(task["output_dir"]) / "games.jsonl"))
    safety = analysis.summarize_safety(all_rows)
    external = _gate_receipt(gate_path)
    protected_controls = ("h2", "e1", "r7")
    control_gate_pass = all(
        controls[opponent]["retention_floor_pass"]
        and not controls[opponent]["wilson_severe_regression"]
        for opponent in protected_controls
    )
    regression_counts = {
        "same_schedule_below_minus_0_05": sum(
            not receipt["retention_floor_pass"] for receipt in controls.values()
        ),
        "control_wilson_severe": sum(
            receipt["wilson_severe_regression"] for receipt in controls.values()
        ),
        "reference_wilson_severe_cells": len(panel["wilson_severe_regression_cells"]),
        "reference_paired_severe_cells": len(panel["paired_severe_regression_cells"]),
    }
    all_gates_pass = all((
        primary["gate_pass"],
        x2_edge["gate_pass"],
        control_gate_pass,
        hardtail["gate_pass"],
        panel["gate_pass"],
        safety["safety_clean"],
        external["pass"],
    ))
    if not safety["safety_clean"] or not external["pass"]:
        classification = "REJECTED_SAFETY_LEAKAGE_OR_CONTAMINATION"
    elif primary["gate_pass"] and not all_gates_pass:
        classification = "REJECTED_RETENTION_REGRESSION"
    elif all_gates_pass:
        classification = "QUALIFIED_RESEARCH_LEAGUE_D1_SUCCESSOR_NOT_PROMOTED"
    else:
        classification = "REJECTED_NO_DECISIVE_D1_EDGE"

    return {
        "schema_version": "deepmew.d1_successor_frozen_qualification_analysis.v1",
        "created_at_utc": analysis._utc_now(),
        "qualification_root": str(root),
        "execution_manifest": {"path": str(manifest_path), "planned_games": FROZEN_GAMES, "completed_games": FROZEN_GAMES},
        "bootstrap": {
            "resamples": analysis.BOOTSTRAP_RESAMPLES,
            "engine_rng_control": "not_instrumented",
            "interpretation": "schedule-index bootstrap over independent engine draws; not paired engine samples",
        },
        "successor_vs_d1": primary,
        "controls": controls,
        "x2_edge_reduction_relative_to_d1": x2_edge,
        "reference_panel": panel,
        "hardtail": hardtail,
        "severe_regression_counts": regression_counts,
        "safety": safety,
        "external_safety_leakage_runtime_and_finite_gates": external,
        "classification": classification,
        "active_champion_changed": False,
        "default_v2_4_changed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-root", required=True)
    parser.add_argument("--gate-receipt", required=True)
    parser.add_argument("--execution-manifest", required=True)
    parser.add_argument("--output", required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output).resolve()
    receipt = analyze(
        Path(args.qualification_root).resolve(),
        Path(args.gate_receipt).resolve(),
        Path(args.execution_manifest).resolve(),
    )
    _atomic_json(output, receipt)
    print(json.dumps({"classification": receipt["classification"], "output": str(output)}, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
