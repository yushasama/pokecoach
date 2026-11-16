"""Analyze the prospectively frozen H2/X3 qualification program.

The engine does not expose a seed control.  Comparisons therefore align the
prospectively frozen schedule coordinates, but never claim paired engine
samples.  Bootstrap intervals are schedule-index intervals over independent
engine draws.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_LABEL = "H2_X3_FIRST_WAVE_V1|frozen_qualification|bootstrap_v1"
BOOTSTRAP_SEED = int.from_bytes(hashlib.sha256(BOOTSTRAP_LABEL.encode()).digest()[:4], "little")
CINDERACE_DECK_HASH = "6644aa1457640c4c3d0c132d2b5e493ec808f97162a8ed59ec15448732b9f007"


@dataclass(frozen=True)
class PayoffGame:
    task: str
    coordinate: tuple[str, int, int]
    focal_label: str
    opponent_label: str
    focal_seat: int
    score: float
    telemetry: dict[str, Any] | None
    raw: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _score(outcome: str) -> float:
    values = {"win": 1.0, "draw": 0.5, "loss": 0.0}
    try:
        return values[outcome]
    except KeyError as error:
        raise ValueError(f"unsupported game outcome: {outcome!r}") from error


def _invert_score(score: float) -> float:
    return 1.0 - score


def load_payoff_games(task_dir: Path, focal_label: str) -> list[PayoffGame]:
    task = task_dir.name
    games = []
    for row in _read_jsonl(task_dir / "games.jsonl"):
        row_label = str(row["row_label"])
        column_label = str(row["column_label"])
        row_seat = int(row["row_seat"])
        outcome_score = _score(str(row["outcome"]))
        if row_label == focal_label:
            focal_seat = row_seat
            opponent = column_label
            focal_score = outcome_score
        elif column_label == focal_label:
            focal_seat = 1 - row_seat
            opponent = row_label
            focal_score = _invert_score(outcome_score)
        else:
            raise ValueError(f"{task}: focal label {focal_label!r} is absent from a game")
        telemetry = row.get("public_telemetry")
        member_telemetry = None
        if isinstance(telemetry, dict):
            members = telemetry.get("members")
            if isinstance(members, dict):
                member_telemetry = members.get(focal_label)
        game_index = int(row["game_index"])
        games.append(PayoffGame(
            task=task,
            coordinate=(task, focal_seat, game_index),
            focal_label=focal_label,
            opponent_label=opponent,
            focal_seat=focal_seat,
            score=focal_score,
            telemetry=member_telemetry,
            raw=row,
        ))
    coordinates = [game.coordinate for game in games]
    if len(coordinates) != len(set(coordinates)):
        raise ValueError(f"{task}: duplicate focal schedule coordinates")

    return games


def _wilson(wins: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    proportion = wins / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator

    return [center - half, center + half]


def summarize_games(games: Sequence[PayoffGame]) -> dict[str, Any]:
    wins = sum(game.score == 1.0 for game in games)
    draws = sum(game.score == 0.5 for game in games)
    losses = sum(game.score == 0.0 for game in games)
    decided = wins + losses
    by_seat = {}
    for seat in (0, 1):
        subset = [game for game in games if game.focal_seat == seat]
        if subset:
            by_seat[str(seat)] = summarize_games_without_splits(subset)

    return {
        **summarize_games_without_splits(games),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "by_seat": by_seat,
        "safety": summarize_safety(game.raw for game in games),
    }


def summarize_games_without_splits(games: Sequence[PayoffGame]) -> dict[str, Any]:
    wins = sum(game.score == 1.0 for game in games)
    draws = sum(game.score == 0.5 for game in games)
    losses = sum(game.score == 0.0 for game in games)
    decided = wins + losses

    return {
        "games": len(games),
        "point_rate": sum(game.score for game in games) / len(games) if games else None,
        "decisive_win_rate": wins / decided if decided else None,
        "decisive_wilson95": _wilson(wins, decided),
    }


def summarize_safety(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    receipt = {
        "games": len(rows),
        "illegal_actions": sum(bool(row.get("illegal")) for row in rows),
        "crashes": sum(bool(row.get("crashed")) for row in rows),
        "timeouts": sum(bool(row.get("timed_out")) for row in rows),
        "incomplete_games": sum(bool(row.get("incomplete")) for row in rows),
    }
    receipt["safety_clean"] = not any(receipt[key] for key in (
        "illegal_actions",
        "crashes",
        "timeouts",
        "incomplete_games",
    ))

    return receipt


def _percentile_interval(values: np.ndarray) -> list[float]:
    lower, upper = np.percentile(values, [2.5, 97.5])

    return [float(lower), float(upper)]


def paired_schedule_bootstrap(
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed_offset: int = 0,
) -> dict[str, Any]:
    candidate_array = np.asarray(candidate, dtype=np.float64)
    baseline_array = np.asarray(baseline, dtype=np.float64)
    if candidate_array.shape != baseline_array.shape or candidate_array.size == 0:
        raise ValueError("schedule bootstrap requires equal, non-empty samples")
    differences = candidate_array - baseline_array
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    chunk_size = 256
    means = []
    for start in range(0, resamples, chunk_size):
        count = min(chunk_size, resamples - start)
        indices = rng.integers(0, differences.size, size=(count, differences.size))
        means.append(differences[indices].mean(axis=1))
    bootstrap = np.concatenate(means)

    return {
        "estimate": float(differences.mean()),
        "bootstrap_95": _percentile_interval(bootstrap),
        "bootstrap_standard_error": float(bootstrap.std(ddof=1)),
        "resamples": resamples,
        "seed": BOOTSTRAP_SEED + seed_offset,
        "unit": "prospectively_frozen_schedule_index",
        "engine_sample_relationship": "independent; engine RNG not instrumented",
        "paired_engine_sample_claim": False,
    }


def align_schedule(
    candidate_batches: Sequence[Sequence[PayoffGame]],
    baseline_batches: Sequence[Sequence[PayoffGame]],
) -> tuple[list[PayoffGame], list[PayoffGame]]:
    if len(candidate_batches) != len(baseline_batches):
        raise ValueError("schedule comparison requires the same number of batches")
    candidate_aligned = []
    baseline_aligned = []
    for batch_index, (candidate, baseline) in enumerate(zip(candidate_batches, baseline_batches, strict=True)):
        candidate_by_coordinate = {
            (batch_index, game.focal_seat, game.coordinate[2]): game
            for game in candidate
        }
        baseline_by_coordinate = {
            (batch_index, game.focal_seat, game.coordinate[2]): game
            for game in baseline
        }
        if candidate_by_coordinate.keys() != baseline_by_coordinate.keys():
            raise ValueError(f"schedule coordinates differ in batch {batch_index + 1}")
        for coordinate in sorted(candidate_by_coordinate):
            candidate_aligned.append(candidate_by_coordinate[coordinate])
            baseline_aligned.append(baseline_by_coordinate[coordinate])

    return candidate_aligned, baseline_aligned


def compare_game_scores(
    candidate_batches: Sequence[Sequence[PayoffGame]],
    baseline_batches: Sequence[Sequence[PayoffGame]],
    *,
    seed_offset: int,
) -> dict[str, Any]:
    candidate, baseline = align_schedule(candidate_batches, baseline_batches)
    overall = paired_schedule_bootstrap(
        [game.score for game in candidate],
        [game.score for game in baseline],
        seed_offset=seed_offset,
    )
    splits = []
    for batch_index, (candidate_batch, baseline_batch) in enumerate(
        zip(candidate_batches, baseline_batches, strict=True),
        start=1,
    ):
        aligned_candidate, aligned_baseline = align_schedule([candidate_batch], [baseline_batch])
        for seat in (0, 1):
            candidate_scores = [game.score for game in aligned_candidate if game.focal_seat == seat]
            baseline_scores = [game.score for game in aligned_baseline if game.focal_seat == seat]
            splits.append({
                "batch": batch_index,
                "seat": seat,
                "point_rate_delta": float(np.mean(candidate_scores) - np.mean(baseline_scores)),
                "games": len(candidate_scores),
            })
    overall["batch_seat_splits"] = splits

    return overall


def _mean_metric(games: Sequence[PayoffGame], key: str) -> float:
    values = [game.telemetry.get(key) for game in games if isinstance(game.telemetry, dict)]
    if len(values) != len(games) or any(value is None for value in values):
        raise ValueError(f"public telemetry metric {key!r} is incomplete")

    return float(np.mean(np.asarray(values, dtype=np.float64)))


def compare_telemetry_metric(
    candidate_batches: Sequence[Sequence[PayoffGame]],
    baseline_batches: Sequence[Sequence[PayoffGame]],
    key: str,
    *,
    lower_is_better: bool,
    seed_offset: int,
) -> dict[str, Any]:
    candidate, baseline = align_schedule(candidate_batches, baseline_batches)
    candidate_values = [float(game.telemetry[key]) for game in candidate if game.telemetry is not None]
    baseline_values = [float(game.telemetry[key]) for game in baseline if game.telemetry is not None]
    if len(candidate_values) != len(candidate) or len(baseline_values) != len(baseline):
        raise ValueError(f"metric {key!r} is absent from one or more games")
    if lower_is_better:
        comparison = paired_schedule_bootstrap(
            baseline_values,
            candidate_values,
            seed_offset=seed_offset,
        )
    else:
        comparison = paired_schedule_bootstrap(
            candidate_values,
            baseline_values,
            seed_offset=seed_offset,
        )
    comparison.update({
        "metric": key,
        "candidate_mean": float(np.mean(candidate_values)),
        "baseline_mean": float(np.mean(baseline_values)),
        "improvement_direction": "baseline_minus_candidate" if lower_is_better else "candidate_minus_baseline",
    })

    return comparison


def behavioral_diagnostics(games: Sequence[PayoffGame]) -> dict[str, Any]:
    telemetry = [game.telemetry for game in games if isinstance(game.telemetry, dict)]
    if len(telemetry) != len(games):
        raise ValueError("behavioral diagnostics require telemetry for every game")
    decisions = sum(int(row["decision_count"]) for row in telemetry)
    forced = sum(int(row["forced_decision_count"]) for row in telemetry)
    categories = Counter()
    identities = Counter()
    identities_by_cardinality: dict[str, Counter[str]] = defaultdict(Counter)
    for row in telemetry:
        categories.update(row["action_category_counts"])
        identities.update(row["unforced_action_identity_counts"])
        for cardinality, counts in row["unforced_action_identities_by_legal_cardinality"].items():
            identities_by_cardinality[cardinality].update(counts)
    selected_actions = sum(categories.values())
    unforced_actions = sum(identities.values())
    entropy_by_cardinality = {}
    for cardinality, counts in sorted(identities_by_cardinality.items(), key=lambda item: int(item[0])):
        total = sum(counts.values())
        probabilities = [count / total for count in counts.values()]
        entropy_by_cardinality[cardinality] = {
            "decisions": total,
            "entropy_bits": -sum(probability * math.log2(probability) for probability in probabilities),
            "maximum_identity_share": max(probabilities),
        }
    maximum_identity_share = max(identities.values()) / unforced_actions if unforced_actions else None

    return {
        "games": len(games),
        "decision_count": decisions,
        "forced_decision_rate": forced / decisions if decisions else None,
        "action_category_density_per_decision": {
            category: count / decisions for category, count in sorted(categories.items())
        } if decisions else {},
        "attack_density_per_selected_action": categories.get("ATTACK", 0) / selected_actions if selected_actions else None,
        "play_density_per_selected_action": categories.get("PLAY", 0) / selected_actions if selected_actions else None,
        "attachment_density_per_selected_action": categories.get("ATTACH", 0) / selected_actions if selected_actions else None,
        "mean_first_attack_turn_capped_11": _mean_metric(games, "first_attack_turn_capped_11"),
        "mean_first_prize_turn_capped_11": _mean_metric(games, "first_prize_turn_capped_11"),
        "attack_by_turn_3_rate": _mean_metric(games, "attack_by_turn_3"),
        "prize_by_turn_4_rate": _mean_metric(games, "prize_by_turn_4"),
        "zero_prize_game_rate": _mean_metric(games, "zero_prize_game"),
        "positive_prize_lead_by_turn_5_rate": _mean_metric(games, "positive_prize_lead_by_turn_5"),
        "mega_starmie_evolution_by_turn_4_rate": _mean_metric(games, "mega_starmie_evolution_by_turn_4"),
        "unforced_action_identity_count": unforced_actions,
        "maximum_unforced_action_identity_share": maximum_identity_share,
        "unforced_action_entropy_by_legal_cardinality": entropy_by_cardinality,
        "degenerate_single_action_95pct": maximum_identity_share is not None and maximum_identity_share >= 0.95,
    }


def load_panel_games(task_dir: Path) -> list[dict[str, Any]]:
    return _read_jsonl(task_dir / "games.jsonl")


def summarize_panel(rows: Sequence[dict[str, Any]], champion: str, candidate: str) -> dict[str, Any]:
    participants = {str(row["participant"]) for row in rows}
    if participants != {champion, candidate}:
        raise ValueError(f"unexpected panel participants: {participants}")
    safety = summarize_safety(rows)
    output: dict[str, Any] = {"safety": safety, "participants": {}}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["participant"]), str(row["cell_id"]))].append(row)
    for participant in (champion, candidate):
        participant_rows = [row for row in rows if row["participant"] == participant]
        scores = [_score(str(row["outcome"])) for row in participant_rows]
        wins = sum(score == 1.0 for score in scores)
        losses = sum(score == 0.0 for score in scores)
        output["participants"][participant] = {
            "games": len(scores),
            "point_rate": float(np.mean(scores)),
            "decisive_win_rate": wins / (wins + losses),
            "decisive_wilson95": _wilson(wins, wins + losses),
        }
    severe_cells = []
    paired_severe_cells = []
    cell_results = []
    cell_ids = sorted({cell_id for _, cell_id in grouped})
    for cell_id in cell_ids:
        champion_rows = grouped[(champion, cell_id)]
        candidate_rows = grouped[(candidate, cell_id)]
        champion_scores = [_score(str(row["outcome"])) for row in champion_rows]
        candidate_scores = [_score(str(row["outcome"])) for row in candidate_rows]
        champion_wins = sum(score == 1.0 for score in champion_scores)
        champion_losses = sum(score == 0.0 for score in champion_scores)
        candidate_wins = sum(score == 1.0 for score in candidate_scores)
        candidate_losses = sum(score == 0.0 for score in candidate_scores)
        champion_wilson = _wilson(champion_wins, champion_wins + champion_losses)
        candidate_wilson = _wilson(candidate_wins, candidate_wins + candidate_losses)
        severe = bool(candidate_wilson[1] < champion_wilson[0])
        champion_by_coordinate = {
            (int(row["candidate_seat"]), int(row["game_index"])): _score(str(row["outcome"]))
            for row in champion_rows
        }
        candidate_by_coordinate = {
            (int(row["candidate_seat"]), int(row["game_index"])): _score(str(row["outcome"]))
            for row in candidate_rows
        }
        if champion_by_coordinate.keys() != candidate_by_coordinate.keys():
            raise ValueError(f"panel schedule coordinates differ for {cell_id}")
        coordinates = sorted(champion_by_coordinate)
        delta_bootstrap = paired_schedule_bootstrap(
            [candidate_by_coordinate[key] for key in coordinates],
            [champion_by_coordinate[key] for key in coordinates],
            seed_offset=100 + len(cell_results),
        )
        paired_severe = delta_bootstrap["estimate"] <= -0.10 and delta_bootstrap["bootstrap_95"][1] < 0.0
        receipt = {
            "cell_id": cell_id,
            "champion_point_rate": float(np.mean(champion_scores)),
            "candidate_point_rate": float(np.mean(candidate_scores)),
            "point_rate_delta": float(np.mean(candidate_scores) - np.mean(champion_scores)),
            "champion_decisive_wilson95": champion_wilson,
            "candidate_decisive_wilson95": candidate_wilson,
            "wilson_severe_regression": severe,
            "schedule_delta_bootstrap": delta_bootstrap,
            "paired_severe_regression": paired_severe,
        }
        cell_results.append(receipt)
        if severe:
            severe_cells.append(cell_id)
        if paired_severe:
            paired_severe_cells.append(cell_id)
    output["cells"] = cell_results
    output["wilson_severe_regression_cells"] = severe_cells
    output["paired_severe_regression_cells"] = paired_severe_cells

    return output


def _load_batches(root: Path, prefix: str, focal: str, count: int = 3) -> list[list[PayoffGame]]:
    return [load_payoff_games(root / "payoff" / f"{prefix}_{index}", focal) for index in range(1, count + 1)]


def load_external_gates(path: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "status": "PASS",
        "leakage": False,
        "runtime_contamination": False,
        "h2_checkpoint_finite": True,
        "x3_checkpoint_finite": True,
    }
    mismatches = {
        key: {"expected": expected, "observed": receipt.get(key)}
        for key, expected in required.items()
        if receipt.get(key) != expected
    }

    return {
        "path": str(path),
        "receipt": receipt,
        "required": required,
        "mismatches": mismatches,
        "pass": not mismatches,
    }


def load_execution_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or int(manifest.get("planned_games", -1)) != 16_384:
        raise ValueError("qualification execution manifest is not the complete frozen PASS receipt")
    results = manifest.get("results")
    if not isinstance(results, list) or len(results) != 24:
        raise ValueError("qualification execution manifest does not contain the frozen 24-task schedule")
    names = [row.get("name") for row in results if isinstance(row, dict)]
    if len(names) != 24 or len(set(names)) != 24:
        raise ValueError("qualification execution manifest task identities are incomplete or duplicated")
    statuses = [row.get("status") for row in results]
    if any(status not in {"PASS", "SKIPPED_COMPLETE"} for status in statuses):
        raise ValueError("qualification execution manifest contains a non-complete task")
    if int(manifest.get("completed_games_after_resume", 16_384)) != 16_384:
        raise ValueError("qualification execution manifest does not close the full game budget")

    return manifest


def analyze(root: Path, gate_receipt_path: Path, execution_manifest_path: Path) -> dict[str, Any]:
    manifest = load_execution_manifest(execution_manifest_path)

    x2_e1 = _load_batches(root, "h2_x2_vs_e1_seed", "X2")
    x2_h2 = _load_batches(root, "h2_x2_vs_h2_seed", "X2")
    repair = compare_game_scores(x2_h2, x2_e1, seed_offset=10)
    repair["definition"] = "X2 point rate versus H2 minus X2 point rate versus E1"
    repair["baseline_x2_vs_e1"] = summarize_games([game for batch in x2_e1 for game in batch])
    repair["x2_vs_h2"] = summarize_games([game for batch in x2_h2 for game in batch])
    repair["material_repair_pass"] = repair["estimate"] <= -0.05
    repair["direction_supported"] = repair["bootstrap_95"][1] < 0.0
    repair["seat_point_rate_deltas"] = {
        str(seat): float(
            np.mean([game.score for batch in x2_h2 for game in batch if game.focal_seat == seat])
            - np.mean([game.score for batch in x2_e1 for game in batch if game.focal_seat == seat])
        )
        for seat in (0, 1)
    }
    repair["both_seats_improve"] = all(delta < 0.0 for delta in repair["seat_point_rate_deltas"].values())
    repair["each_batch_seat_regression_floor_pass"] = all(
        split["point_rate_delta"] <= 0.05 for split in repair["batch_seat_splits"]
    )

    h2_controls = {
        label: summarize_games(load_payoff_games(root / "payoff" / f"h2_control_{label}", "H2"))
        for label in ("e1", "d1", "r7")
    }
    h2_same_control_retention = {}
    for offset, label in enumerate(("d1", "r7"), start=60):
        h2_games = load_payoff_games(root / "payoff" / f"h2_control_{label}", "H2")
        e1_games = load_payoff_games(root / "payoff" / f"h2_baseline_e1_vs_{label}", "E1")
        comparison = compare_game_scores([h2_games], [e1_games], seed_offset=offset)
        comparison.update({
            "definition": f"H2 point rate versus {label} minus E1 point rate versus {label}",
            "h2": summarize_games(h2_games),
            "e1_same_control_baseline": summarize_games(e1_games),
        })
        comparison["new_collapse"] = (
            comparison["estimate"] <= -0.10
            and comparison["bootstrap_95"][1] < 0.0
        )
        h2_same_control_retention[label] = comparison
    h2_panel = summarize_panel(
        load_panel_games(root / "shared_panel" / "h2_complete_reference_panel"),
        "E1",
        "H2",
    )
    h2_behavior_games = [game for batch in x2_h2 for game in batch]
    h2_behavior = behavioral_diagnostics(h2_behavior_games)

    x3_d1 = _load_batches(root, "x3_vs_d1_seed", "X3")
    r12_d1 = _load_batches(root, "r12_vs_d1_seed", "r12")
    x3_target_games = [game for batch in x3_d1 for game in batch]
    x3_direct = summarize_games(x3_target_games)
    x3_slice = compare_game_scores(x3_d1, r12_d1, seed_offset=20)
    x3_slice["definition"] = "X3 point rate versus D1 minus r12 point rate versus D1"
    x3_slice["material_improvement_pass"] = x3_slice["estimate"] >= 0.05
    x3_slice["direction_supported"] = x3_slice["bootstrap_95"][0] > 0.0
    x3_slice["each_batch_seat_regression_floor_pass"] = all(
        split["point_rate_delta"] >= -0.05 for split in x3_slice["batch_seat_splits"]
    )
    mechanisms = {}
    metric_contract = (
        ("first_attack_turn_capped_11", True, 0.5, "tempo"),
        ("attack_by_turn_3", False, 0.05, "tempo"),
        ("first_prize_turn_capped_11", True, 0.5, "prize_plan"),
        ("prize_by_turn_4", False, 0.05, "prize_plan"),
        ("zero_prize_game", True, 0.05, "prize_plan"),
        ("positive_prize_lead_by_turn_5", False, 0.05, "prize_plan"),
        ("mega_starmie_evolution_by_turn_4", False, 0.05, "prize_plan"),
    )
    for offset, (metric, lower_is_better, threshold, family) in enumerate(metric_contract, start=30):
        receipt = compare_telemetry_metric(
            x3_d1,
            r12_d1,
            metric,
            lower_is_better=lower_is_better,
            seed_offset=offset,
        )
        receipt["family"] = family
        receipt["material_threshold"] = threshold
        receipt["passes"] = receipt["estimate"] >= threshold and receipt["bootstrap_95"][0] > 0.0
        mechanisms[metric] = receipt
    x3_slice["mechanisms"] = mechanisms
    x3_slice["mechanism_gate_pass"] = (
        any(item["passes"] for item in mechanisms.values() if item["family"] == "tempo")
        and any(item["passes"] for item in mechanisms.values() if item["family"] == "prize_plan")
    )
    x3_slice["mechanism_degeneracy_pass"] = all(
        item["estimate"] >= (-0.5 if "turn_capped" in key else -0.05)
        for key, item in mechanisms.items()
    )
    x3_controls = {
        label: summarize_games(load_payoff_games(root / "payoff" / f"x3_control_{label}", "X3"))
        for label in ("e1", "r7", "x2", "r12", "h2")
    }
    x3_panel = summarize_panel(
        load_panel_games(root / "shared_panel" / "x3_complete_reference_panel"),
        "r12",
        "X3",
    )
    x3_behavior = behavioral_diagnostics(x3_target_games)
    d1_behavior = behavioral_diagnostics([
        game
        for index in range(1, 4)
        for game in load_payoff_games(root / "payoff" / f"x3_vs_d1_seed_{index}", "D1")
    ])
    x2_behavior = behavioral_diagnostics(
        load_payoff_games(root / "payoff" / "x3_control_x2", "X2")
    )
    x3_against_x2_behavior = behavioral_diagnostics(
        load_payoff_games(root / "payoff" / "x3_control_x2", "X3")
    )

    all_rows = []
    for task in manifest["tasks"]:
        task_dir = Path(task["output_dir"])
        all_rows.extend(_read_jsonl(task_dir / "games.jsonl"))
    safety = summarize_safety(all_rows)
    external_gates = load_external_gates(gate_receipt_path)
    h2_reference = h2_panel["participants"]["H2"]
    x3_reference = x3_panel["participants"]["X3"]
    h2_broad_regression = (
        bool(h2_panel["wilson_severe_regression_cells"])
        or h2_reference["decisive_win_rate"] < 0.45
        or any(item["new_collapse"] for item in h2_same_control_retention.values())
    )
    h2_degenerate = (
        not safety["safety_clean"]
        or not external_gates["pass"]
        or h2_behavior["degenerate_single_action_95pct"]
    )
    if h2_degenerate:
        h2_classification = "REJECTED_DEGENERATE_CHILD"
    elif h2_broad_regression:
        h2_classification = "REJECTED_BROAD_REGRESSION"
    elif repair["material_repair_pass"] and repair["direction_supported"] and repair["both_seats_improve"]:
        h2_classification = "QUALIFIED_REPAIR_CHILD"
    elif repair["estimate"] < 0.0 and repair["direction_supported"]:
        h2_classification = "DIAGNOSTIC_REPAIR_CHILD"
    else:
        h2_classification = "REJECTED_REPAIR_INSUFFICIENT"

    x3_direct_pass = (
        x3_direct["decisive_win_rate"] >= 0.55
        and x3_direct["decisive_wilson95"][0] > 0.5
        and all(split["decisive_win_rate"] >= 0.5 for split in x3_direct["by_seat"].values())
        and all(
            summarize_games(batch)["decisive_win_rate"] >= 0.5
            for batch in x3_d1
        )
    )
    x3_slice_pass = (
        x3_slice["material_improvement_pass"]
        and x3_slice["direction_supported"]
        and x3_slice["each_batch_seat_regression_floor_pass"]
        and x3_slice["mechanism_gate_pass"]
        and x3_slice["mechanism_degeneracy_pass"]
    )
    x3_retention_pass = not x3_panel["paired_severe_regression_cells"] and x3_reference["decisive_win_rate"] >= 0.45
    x3_degenerate = (
        not safety["safety_clean"]
        or not external_gates["pass"]
        or x3_behavior["degenerate_single_action_95pct"]
        or not x3_slice["mechanism_degeneracy_pass"]
    )
    diagnostic_mechanism = any(item["bootstrap_95"][0] > 0.0 for item in mechanisms.values())
    if x3_degenerate:
        x3_classification = "REJECTED_DEGENERATE_EXPLOITER"
    elif (x3_direct_pass or x3_slice_pass) and not x3_retention_pass:
        x3_classification = "REJECTED_OVERFIT_EXPLOITER"
    elif x3_direct_pass and x3_retention_pass:
        x3_classification = "QUALIFIED_ACTIVE_CROSS_DECK_EXPLOITER"
    elif x3_slice_pass and x3_retention_pass:
        x3_classification = "QUALIFIED_HISTORICAL_CROSS_DECK_EXPLOITER"
    elif diagnostic_mechanism and x3_retention_pass:
        x3_classification = "DIAGNOSTIC_ONLY_CROSS_DECK_EXPLOITER"
    else:
        x3_classification = "REJECTED_NO_EDGE"

    return {
        "schema_version": "deepmew.h2_x3_frozen_qualification_analysis.v1",
        "created_at_utc": _utc_now(),
        "qualification_root": str(root),
        "execution_manifest": {
            "path": str(execution_manifest_path),
            "schema_version": manifest.get("schema_version"),
            "status": manifest.get("status"),
            "completed_games_before_resume": manifest.get("completed_games_before_resume"),
            "completed_games_after_resume": manifest.get("completed_games_after_resume", 16_384),
            "skipped_complete_task_count": sum(
                row.get("status") == "SKIPPED_COMPLETE" for row in manifest["results"]
            ),
            "resumed_task_count": sum(row.get("status") == "PASS" for row in manifest["results"]),
            "preserved_original_hard_stop_manifest": manifest.get("preserved_original_hard_stop_manifest"),
        },
        "bootstrap": {
            "label": BOOTSTRAP_LABEL,
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "engine_rng_control": "not_instrumented",
            "interpretation": "schedule-index bootstrap over independent engine draws; not paired engine samples",
        },
        "safety": safety,
        "external_safety_leakage_runtime_and_finite_gates": external_gates,
        "h2": {
            "repair": repair,
            "controls": h2_controls,
            "reference_panel": h2_panel,
            "behavior": h2_behavior,
            "d1_r7_relative_to_e1_gate": {
                "status": "PASS" if not any(
                    item["new_collapse"] for item in h2_same_control_retention.values()
                ) else "FAIL_NEW_COLLAPSE",
                "authorization": "exactly 512 supplemental free local games explicitly authorized by the user",
                "games_added": 512,
                "controls": h2_same_control_retention,
                "collapse_definition": "H2 point rate is at least 0.10 below E1 on the same control with bootstrap 95% upper bound below zero",
            },
            "classification": h2_classification,
        },
        "x3": {
            "direct_d1": x3_direct,
            "direct_route_pass": x3_direct_pass,
            "failure_slice": x3_slice,
            "slice_route_pass": x3_slice_pass,
            "controls": x3_controls,
            "reference_panel": x3_panel,
            "retention_pass": x3_retention_pass,
            "behavior": x3_behavior,
            "behavioral_distinctness": {
                "D1": {
                    "x3_profile": x3_behavior,
                    "comparison_profile": d1_behavior,
                    "caveat": "measured in exact target telemetry; deck-confounded diagnostic comparison",
                },
                "X2": {
                    "x3_profile": x3_against_x2_behavior,
                    "comparison_profile": x2_behavior,
                    "caveat": "measured in x3_control_x2 telemetry; deck-confounded diagnostic comparison",
                },
                "X1": {
                    "status": "HISTORICAL_TELEMETRY_NOT_INSTRUMENTED",
                    "reason": "historical qualification did not persist the preregistered action/tempo telemetry; no new X1 games were added to the exact frozen 6912-game X3 schedule",
                },
                "gate_role": "diagnostic only; cannot rescue failure of both target routes",
            },
            "classification": x3_classification,
        },
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
    result = analyze(
        Path(args.qualification_root).resolve(),
        Path(args.gate_receipt).resolve(),
        Path(args.execution_manifest).resolve(),
    )
    _atomic_json(Path(args.output).resolve(), result)
    print(json.dumps({
        "h2_classification": result["h2"]["classification"],
        "x3_classification": result["x3"]["classification"],
        "output": str(Path(args.output).resolve()),
    }, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
