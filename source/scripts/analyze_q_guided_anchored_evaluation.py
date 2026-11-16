"""Validate and aggregate the frozen Q-guided Starmie anchored evaluation.

This research-only postprocessor reads aggregate runner receipts.  It does not
run gameplay, load teacher rows, open model checkpoints, select a policy, or
mutate runtime/submission surfaces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


ADDENDUM_SCHEMA_VERSION = "deepmew.q_guided.anchored_evaluation_identity_addendum.v1"
SHARED_PANEL_SCHEMA_VERSION = "deepmew.league_shared_panel.v2"
PAYOFF_SCHEMA_VERSION = "deepmew.league_payoff_matrix.v2"
ACCEPTANCE_SCHEMA_VERSION = "deepmew.q_guided.acceptance_adapter_receipt.v1"
EVIDENCE_SCHEMA_VERSION = "deepmew.q_guided.anchored_evaluation_evidence.v1"
ARCHIVE_EQUIVALENCE_SCHEMA_VERSION = "deepmew.q_guided.archive_equivalence.v1"
EXPECTED_SEEDS = (40009, 50021, 60013)
T_CRITICAL_DF2_95 = 4.302652729911275
NORMAL_95 = 1.959963984540054
VERDICT_FAMILIES = (
    "TINY_DATA_Q_PROTOTYPE_PROMISING",
    "TINY_DATA_Q_PROTOTYPE_INCONCLUSIVE",
    "TINY_DATA_Q_CHECKPOINT_REJECTED_METHOD_UNTESTED",
    "TECHNICAL_EVALUATION_FAILURE",
)


class EvaluationValidationError(ValueError):
    """Raised when an aggregate evaluation artifact fails closed."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise EvaluationValidationError("artifact contains non-canonical JSON values") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise EvaluationValidationError(f"artifact is unreadable: {path}") from exc

    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise EvaluationValidationError(f"{name} must be a SHA-256 string")
    normalized = value.lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise EvaluationValidationError(f"{name} must be a 64-character SHA-256")

    return normalized


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationValidationError(f"{name} must be an object")

    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise EvaluationValidationError(f"{name} must be an array")

    return value


def _required(value: Mapping[str, Any], fields: Sequence[str], name: str) -> None:
    missing = sorted(set(fields) - set(value))
    if missing:
        raise EvaluationValidationError(f"{name} is missing fields: {missing}")


def _finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationValidationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise EvaluationValidationError(f"{name} is outside its finite bound")

    return result


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EvaluationValidationError(f"{name} must be an integer >= {minimum}")

    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvaluationValidationError(f"{name} must be a non-empty string")

    return value


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationValidationError(f"{name} is not readable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise EvaluationValidationError(f"{name} must contain one JSON object")

    return value


def _resolve_path(raw_path: Any, base_dir: Path, name: str) -> Path:
    path_text = _string(raw_path, name)
    path = Path(path_text)
    if not path.is_absolute():
        path = base_dir / path

    return path.resolve()


def _read_hash_bound_json(path: Path, expected_sha256: str, name: str) -> tuple[dict[str, Any], str]:
    expected = _sha256(expected_sha256, f"{name} expected sha256")
    actual = _sha256_file(path)
    if actual != expected:
        raise EvaluationValidationError(f"{name} hash mismatch")

    return _read_json(path, name), actual


def _validate_receipt(receipt: Any, expected_sha256: str, name: str) -> None:
    value = _mapping(receipt, name)
    actual = _sha256(value.get("sha256"), f"{name} sha256")
    if actual != _sha256(expected_sha256, f"{name} expected sha256"):
        raise EvaluationValidationError(f"{name} identity mismatch")


def _validate_config_receipt(
    result_path: Path,
    result: Mapping[str, Any],
    *,
    config_key: str,
    name: str,
) -> Mapping[str, Any]:
    config_receipt = _read_json(result_path.parent / "config.json", f"{name} config")
    _required(config_receipt, ("schema_version", "run_id", "config_fingerprint", config_key), f"{name} config")
    config = _mapping(config_receipt[config_key], f"{name} config payload")
    fingerprint = _sha256_json(config)
    if config_receipt["config_fingerprint"] != fingerprint:
        raise EvaluationValidationError(f"{name} config fingerprint is not canonical")
    if result.get("config_fingerprint") != fingerprint:
        raise EvaluationValidationError(f"{name} result/config fingerprint mismatch")
    if result.get("run_id") != config_receipt["run_id"]:
        raise EvaluationValidationError(f"{name} result/config run_id mismatch")

    return config


def _validate_source_protocol(addendum: Mapping[str, Any], base_dir: Path) -> dict[str, Any]:
    source = _mapping(addendum["source_protocol"], "source protocol binding")
    _required(source, ("path", "sha256"), "source protocol binding")
    path = _resolve_path(source["path"], base_dir, "source protocol path")
    protocol, _ = _read_hash_bound_json(path, source["sha256"], "source protocol")
    if protocol.get("schema_version") != "deepmew.q_guided.anchored_evaluation_protocol.v1":
        raise EvaluationValidationError("source protocol schema mismatch")
    if protocol.get("status") != "frozen_not_executed":
        raise EvaluationValidationError("source protocol is not frozen_not_executed")

    return protocol


def _identity(addendum: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    identities = _mapping(addendum["identities"], "addendum identities")
    identity = _mapping(identities.get(key), f"{key} identity")
    _required(
        identity,
        ("label", "league_member_id", "artifact_sha256", "deck_sha256", "feature_policy_id"),
        f"{key} identity",
    )
    _string(identity["label"], f"{key} label")
    _string(identity["league_member_id"], f"{key} league_member_id")
    _sha256(identity["artifact_sha256"], f"{key} artifact sha256")
    _sha256(identity["deck_sha256"], f"{key} deck sha256")
    _string(identity["feature_policy_id"], f"{key} feature_policy_id")

    return identity


def _validate_archive_equivalence(
    addendum: Mapping[str, Any],
    base_dir: Path,
    r18: Mapping[str, Any],
) -> dict[str, Any]:
    binding = _mapping(addendum["archive_equivalence"], "archive equivalence binding")
    _required(binding, ("path", "sha256"), "archive equivalence binding")
    path = _resolve_path(binding["path"], base_dir, "archive equivalence path")
    receipt, actual_hash = _read_hash_bound_json(path, binding["sha256"], "archive equivalence")
    if (
        receipt.get("schema_version") != ARCHIVE_EQUIVALENCE_SCHEMA_VERSION
        or receipt.get("status") != "verified_byte_equivalent"
    ):
        raise EvaluationValidationError("R18 archive equivalence receipt is not verified")
    archive = _mapping(receipt.get("archive"), "archive equivalence archive")
    runnable = _mapping(receipt.get("runnable_directory"), "archive equivalence directory")
    comparison = _mapping(receipt.get("comparison"), "archive equivalence comparison")
    if _sha256(archive.get("sha256"), "equivalent archive sha256") != r18["artifact_sha256"]:
        raise EvaluationValidationError("R18 archive hash is not bound to the addendum")
    if _sha256(runnable.get("sha256"), "equivalent directory sha256") != _sha256(
        r18.get("runnable_directory_sha256"), "R18 runnable directory sha256"
    ):
        raise EvaluationValidationError("R18 runnable directory hash is not bound to the addendum")
    if comparison.get("all_regular_files_byte_equal") is not True:
        raise EvaluationValidationError("R18 archive and runnable directory are not byte-equivalent")
    for field in ("archive_only_paths", "directory_only_paths", "byte_mismatch_paths"):
        if comparison.get(field) != []:
            raise EvaluationValidationError(f"R18 archive equivalence contains {field}")

    return {
        "receipt_sha256": actual_hash,
        "archive_sha256": archive["sha256"],
        "runnable_directory_sha256": runnable["sha256"],
        "regular_file_count": _integer(
            comparison.get("archive_regular_file_count"),
            "R18 archive regular file count",
            minimum=1,
        ),
        "byte_equivalent": True,
    }


def _validate_protocol_identities(
    protocol: Mapping[str, Any],
    candidate: Mapping[str, Any],
    r7: Mapping[str, Any],
) -> None:
    if protocol.get("candidate_id") != candidate["label"]:
        raise EvaluationValidationError("source protocol candidate label mismatch")
    if protocol.get("reference_id") != r7["label"]:
        raise EvaluationValidationError("source protocol R7 label mismatch")
    if _sha256(protocol.get("candidate_sha256"), "source candidate sha256") != candidate["artifact_sha256"]:
        raise EvaluationValidationError("source protocol candidate artifact mismatch")
    if _sha256(protocol.get("reference_sha256"), "source R7 sha256") != r7["artifact_sha256"]:
        raise EvaluationValidationError("source protocol R7 artifact mismatch")
    panel = _mapping(protocol.get("panel"), "source protocol panel")
    if _sha256(panel.get("deck_binding_sha256"), "source deck sha256") != candidate["deck_sha256"]:
        raise EvaluationValidationError("source protocol candidate deck mismatch")
    if tuple(protocol.get("frozen_seeds") or ()) != EXPECTED_SEEDS:
        raise EvaluationValidationError("source protocol frozen seeds mismatch")


def _zero_safety(value: Any, name: str) -> dict[str, int]:
    counters = _mapping(value, name)
    result = {}
    for field in ("crashes", "timeouts", "illegal"):
        count = _integer(counters.get(field), f"{name} {field}")
        if count != 0:
            raise EvaluationValidationError(f"{name} has {field}={count}")
        result[field] = count
    if "incomplete" in counters:
        incomplete = _integer(counters["incomplete"], f"{name} incomplete")
        if incomplete != 0:
            raise EvaluationValidationError(f"{name} has incomplete={incomplete}")
        result["incomplete"] = incomplete

    return result


def _confidence_interval(mean: float, standard_error: float, multiplier: float) -> list[float]:
    return [mean - multiplier * standard_error, mean + multiplier * standard_error]


def _aggregate_cell_group(
    group_name: str,
    cell_ids: Sequence[Any],
    candidate_cells: Mapping[str, Mapping[str, Any]],
    r7_cells: Mapping[str, Mapping[str, Any]],
    comparison_cells: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    ids = [_string(cell_id, f"{group_name} cell id") for cell_id in cell_ids]
    if not ids or len(ids) != len(set(ids)):
        raise EvaluationValidationError(f"{group_name} retention group must contain unique cells")
    unknown = sorted(set(ids) - set(candidate_cells))
    if unknown or set(ids) - set(r7_cells) or set(ids) - set(comparison_cells):
        raise EvaluationValidationError(f"{group_name} retention cells are absent from shared-panel results: {unknown}")

    def totals(cells: Mapping[str, Mapping[str, Any]]) -> tuple[int, int, int, int, float]:
        wins = sum(_integer(cells[cell]["wins"], f"{group_name} wins") for cell in ids)
        losses = sum(_integer(cells[cell]["losses"], f"{group_name} losses") for cell in ids)
        draws = sum(_integer(cells[cell]["draws"], f"{group_name} draws") for cell in ids)
        completed = sum(_integer(cells[cell]["completed"], f"{group_name} completed") for cell in ids)
        elapsed = sum(_finite(cells[cell]["elapsed_game_seconds"], f"{group_name} elapsed", minimum=0.0) for cell in ids)
        if completed != wins + losses + draws:
            raise EvaluationValidationError(f"{group_name} retention totals are inconsistent")

        return wins, losses, draws, completed, elapsed

    candidate_totals = totals(candidate_cells)
    r7_totals = totals(r7_cells)
    if candidate_totals[3] != r7_totals[3]:
        raise EvaluationValidationError(f"{group_name} retention budget differs between candidate and R7")
    candidate_rate = (candidate_totals[0] + 0.5 * candidate_totals[2]) / candidate_totals[3]
    r7_rate = (r7_totals[0] + 0.5 * r7_totals[2]) / r7_totals[3]
    regression_count = sum(_finite(comparison_cells[cell]["delta"], f"{group_name} cell delta") < 0.0 for cell in ids)
    severe_count = sum(comparison_cells[cell].get("severe_regression") is True for cell in ids)

    return {
        "cell_ids": ids,
        "cell_count": len(ids),
        "games_per_participant": candidate_totals[3],
        "candidate_wins_losses_draws": list(candidate_totals[:3]),
        "r7_wins_losses_draws": list(r7_totals[:3]),
        "candidate_point_rate": candidate_rate,
        "r7_point_rate": r7_rate,
        "candidate_minus_r7_point_rate": candidate_rate - r7_rate,
        "regression_cell_count": regression_count,
        "severe_regression_cell_count": severe_count,
        "candidate_elapsed_game_seconds": candidate_totals[4],
        "r7_elapsed_game_seconds": r7_totals[4],
    }


def _validate_shared_panel(
    result_path: Path,
    expected_result_sha256: str,
    addendum: Mapping[str, Any],
    candidate: Mapping[str, Any],
    r7: Mapping[str, Any],
) -> dict[str, Any]:
    result, actual_result_hash = _read_hash_bound_json(
        result_path,
        expected_result_sha256,
        "shared-panel result",
    )
    shared = _mapping(addendum["shared_panel"], "shared-panel addendum")
    if result.get("schema_version") != SHARED_PANEL_SCHEMA_VERSION:
        raise EvaluationValidationError("shared-panel result schema mismatch")
    if result.get("run_id") != shared.get("run_id"):
        raise EvaluationValidationError("shared-panel run_id mismatch")
    if result.get("config_fingerprint") != shared.get("config_fingerprint"):
        raise EvaluationValidationError("shared-panel frozen config fingerprint mismatch")
    if result.get("status") != "completed" or result.get("comparable") is not True:
        raise EvaluationValidationError("shared-panel result is incomplete or non-comparable")
    if result.get("safety_clean") is not True:
        raise EvaluationValidationError("shared-panel result is not safety-clean")
    config = _validate_config_receipt(result_path, result, config_key="panel_config", name="shared panel")
    if config.get("run_id") != shared.get("run_id"):
        raise EvaluationValidationError("shared-panel config run_id mismatch")
    receipts = _mapping(config.get("input_receipts"), "shared-panel input receipts")
    _validate_receipt(receipts.get("evaluator_source"), shared.get("runner_sha256"), "shared-panel runner")
    _validate_receipt(
        receipts.get(f"participant:{candidate['label']}"),
        candidate["artifact_sha256"],
        "shared-panel candidate",
    )
    _validate_receipt(
        receipts.get(f"participant:{r7['label']}"),
        r7["artifact_sha256"],
        "shared-panel R7",
    )
    _validate_receipt(receipts.get("deck_binding"), candidate["deck_sha256"], "shared-panel deck")
    _validate_receipt(
        receipts.get("exploiter_checkpoint"),
        shared.get("exploiter_checkpoint_sha256"),
        "shared-panel exploiter checkpoint",
    )
    participants_config = _sequence(config.get("participants"), "shared-panel config participants")
    expected_participants = {
        (candidate["label"], "", "candidate"),
        (r7["label"], "", "champion"),
    }
    observed_participants = {
        (row.get("label"), row.get("league_member_id"), row.get("role"))
        for row in map(lambda item: _mapping(item, "shared-panel config participant"), participants_config)
    }
    if observed_participants != expected_participants:
        raise EvaluationValidationError("shared-panel participant identity mismatch")
    if (
        tuple(config.get("panel_seeds") or ()) != EXPECTED_SEEDS
        or tuple(shared.get("seeds") or ()) != EXPECTED_SEEDS
    ):
        raise EvaluationValidationError("shared-panel seed identity mismatch")
    if config.get("games_per_cell_seed") != shared.get("games_per_cell_per_seed_per_participant"):
        raise EvaluationValidationError("shared-panel game budget mismatch")
    cells_config = _sequence(config.get("cells"), "shared-panel config cells")
    if len(cells_config) != shared.get("cell_count"):
        raise EvaluationValidationError("shared-panel cell count mismatch")
    schedule = _mapping(result.get("schedule"), "shared-panel schedule")
    expected_games = _integer(shared.get("expected_games_per_participant"), "shared expected games", minimum=1)
    if (
        schedule.get("complete") is not True
        or schedule.get("jobs_completed") != schedule.get("jobs_expected")
        or schedule.get("games_expected_per_participant") != expected_games
    ):
        raise EvaluationValidationError("shared-panel expected schedule is incomplete")
    identities = _mapping(result.get("member_identities"), "shared-panel member identities")
    expected_identity_fields = {
        "candidate_member_id": None,
        "champion_member_id": None,
        "candidate_checkpoint_sha256": candidate["artifact_sha256"],
        "champion_checkpoint_sha256": r7["artifact_sha256"],
        "candidate_deck_sha256": candidate["deck_sha256"],
        "champion_deck_sha256": r7["deck_sha256"],
        "feature_policy_id": candidate["feature_policy_id"],
    }
    for field, expected in expected_identity_fields.items():
        if identities.get(field) != expected:
            raise EvaluationValidationError(f"shared-panel {field} mismatch")
    participants = _mapping(result.get("participants"), "shared-panel participants")
    if set(participants) != {candidate["label"], r7["label"]}:
        raise EvaluationValidationError("shared-panel result participant set mismatch")
    candidate_result = _mapping(participants[candidate["label"]], "shared-panel candidate result")
    r7_result = _mapping(participants[r7["label"]], "shared-panel R7 result")
    for label, participant, expected_role in (
        ("candidate", candidate_result, "candidate"),
        ("R7", r7_result, "champion"),
    ):
        if participant.get("league_member_id") is not None or participant.get("role") != expected_role:
            raise EvaluationValidationError(
                f"shared-panel {label} must preserve the frozen command's empty member ID and exact role"
            )
        aggregate = _mapping(participant.get("aggregate"), f"shared-panel {label} aggregate")
        if aggregate.get("budget_complete") is not True or aggregate.get("safety_clean") is not True:
            raise EvaluationValidationError(f"shared-panel {label} aggregate is incomplete or unsafe")
        if aggregate.get("attempted") != expected_games or aggregate.get("completed") != expected_games:
            raise EvaluationValidationError(f"shared-panel {label} aggregate budget mismatch")
        _zero_safety(aggregate, f"shared-panel {label} aggregate")
    comparisons = _sequence(result.get("comparisons"), "shared-panel comparisons")
    if len(comparisons) != 1:
        raise EvaluationValidationError("shared-panel must contain exactly one candidate/R7 comparison")
    comparison = _mapping(comparisons[0], "shared-panel comparison")
    if (
        comparison.get("candidate_member_id") is not None
        or comparison.get("champion_member_id") is not None
        or comparison.get("equal_completed_budget") is not True
    ):
        raise EvaluationValidationError("shared-panel comparison identity or budget mismatch")
    aggregate_delta = _finite(comparison.get("aggregate_win_rate_delta"), "shared-panel aggregate delta")
    aggregate_se = _finite(
        comparison.get("independent_aggregate_delta_se"),
        "shared-panel aggregate delta SE",
        minimum=0.0,
    )
    aggregate_ci = _confidence_interval(aggregate_delta, aggregate_se, NORMAL_95)
    batch_mean = _finite(
        comparison.get("independent_cell_seed_batch_delta_mean"),
        "shared-panel cell-seed delta mean",
    )
    batch_se = _finite(
        comparison.get("independent_cell_seed_batch_delta_se"),
        "shared-panel cell-seed delta SE",
        minimum=0.0,
    )
    batch_ci = _confidence_interval(batch_mean, batch_se, NORMAL_95)
    candidate_cells = {
        _string(row.get("cell_id"), "candidate shared-panel cell id"): row
        for row in map(lambda item: _mapping(item, "candidate shared-panel cell"), candidate_result.get("cells") or [])
    }
    r7_cells = {
        _string(row.get("cell_id"), "R7 shared-panel cell id"): row
        for row in map(lambda item: _mapping(item, "R7 shared-panel cell"), r7_result.get("cells") or [])
    }
    comparison_cells = {
        _string(row.get("cell_id"), "comparison cell id"): row
        for row in map(lambda item: _mapping(item, "shared-panel comparison cell"), comparison.get("cell_deltas") or [])
    }
    if (
        len(candidate_cells) != len(cells_config)
        or set(candidate_cells) != set(r7_cells)
        or set(candidate_cells) != set(comparison_cells)
    ):
        raise EvaluationValidationError("shared-panel per-cell result inventory mismatch")
    severe_cells = _sequence(comparison.get("severe_regression_cells"), "shared-panel severe cells")
    severe_count = _integer(comparison.get("severe_regression_count"), "shared-panel severe count")
    if severe_count != len(severe_cells):
        raise EvaluationValidationError("shared-panel severe regression count mismatch")
    retention_config = _mapping(shared.get("retention_groups"), "shared-panel retention groups")
    for required_group in ("reference", "exploiter"):
        if required_group not in retention_config:
            raise EvaluationValidationError(f"shared-panel retention group {required_group} is missing")
    retention = {
        group_name: _aggregate_cell_group(
            group_name,
            _sequence(cell_ids, f"{group_name} retention cells"),
            candidate_cells,
            r7_cells,
            comparison_cells,
        )
        for group_name, cell_ids in retention_config.items()
    }
    candidate_aggregate = _mapping(candidate_result["aggregate"], "candidate aggregate")
    r7_aggregate = _mapping(r7_result["aggregate"], "R7 aggregate")
    safety = {
        "illegal": candidate_aggregate["illegal"] + r7_aggregate["illegal"],
        "crashes": candidate_aggregate["crashes"] + r7_aggregate["crashes"],
        "timeouts": candidate_aggregate["timeouts"] + r7_aggregate["timeouts"],
        "incomplete": candidate_aggregate["incomplete"] + r7_aggregate["incomplete"],
    }

    return {
        "result_sha256": actual_result_hash,
        "run_id": result["run_id"],
        "config_fingerprint": result["config_fingerprint"],
        "games_per_participant": expected_games,
        "aggregate_win_rate_delta": aggregate_delta,
        "independent_aggregate_delta_se": aggregate_se,
        "independent_aggregate_delta_ci95": aggregate_ci,
        "cell_seed_batch_delta_mean": batch_mean,
        "cell_seed_batch_delta_se": batch_se,
        "cell_seed_batch_delta_ci95": batch_ci,
        "regression_cell_count": _integer(comparison.get("regression_cell_count"), "regression cell count"),
        "maximum_cell_regression": _finite(comparison.get("maximum_cell_regression"), "maximum cell regression"),
        "severe_regression_count": severe_count,
        "severe_regression_cells": list(severe_cells),
        "cell_deltas": list(comparison_cells.values()),
        "retention": retention,
        "safety": safety,
        "runtime": {
            "elapsed_wall_seconds": _finite(result.get("elapsed_wall_seconds"), "shared elapsed wall", minimum=0.0),
            "candidate_elapsed_game_seconds": _finite(
                candidate_aggregate.get("elapsed_game_seconds"),
                "candidate elapsed game seconds",
                minimum=0.0,
            ),
            "r7_elapsed_game_seconds": _finite(
                r7_aggregate.get("elapsed_game_seconds"),
                "R7 elapsed game seconds",
                minimum=0.0,
            ),
            "wall_clock_timeout": config.get("wall_clock_timeout"),
            "max_steps": _integer(config.get("max_steps"), "shared max_steps", minimum=1),
        },
    }


def _wilson_interval(wins: int, decided: int) -> list[float]:
    if decided == 0:
        return [0.0, 1.0]
    z = NORMAL_95
    phat = wins / decided
    denominator = 1.0 + z * z / decided
    center = (phat + z * z / (2.0 * decided)) / denominator
    half = z * math.sqrt(phat * (1.0 - phat) / decided + z * z / (4.0 * decided * decided)) / denominator

    return [max(0.0, center - half), min(1.0, center + half)]


def _pair_counts(row: Mapping[str, Any], name: str, planned_games: int) -> tuple[int, int, int]:
    if row.get("budget_complete") is not True or row.get("safety_clean") is not True:
        raise EvaluationValidationError(f"{name} payoff pair is incomplete or unsafe")
    if row.get("planned_games") != planned_games or row.get("attempted") != planned_games or row.get("completed") != planned_games:
        raise EvaluationValidationError(f"{name} payoff pair budget mismatch")
    _zero_safety(row.get("safety"), f"{name} payoff pair safety")
    wins = _integer(row.get("wins"), f"{name} wins")
    losses = _integer(row.get("losses"), f"{name} losses")
    draws = _integer(row.get("draws"), f"{name} draws")
    if wins + losses + draws != planned_games:
        raise EvaluationValidationError(f"{name} payoff outcome counts are inconsistent")
    expected_rate = (wins + 0.5 * draws) / planned_games
    observed_rate = _finite(row.get("W_ij"), f"{name} W_ij")
    if not math.isclose(observed_rate, round(expected_rate, 6), abs_tol=5e-7):
        raise EvaluationValidationError(f"{name} W_ij is inconsistent with outcome counts")

    return wins, losses, draws


def _member_vs_historical(
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
    member_label: str,
    historical_label: str,
    planned_games: int,
    name: str,
) -> dict[str, Any]:
    forward = _pair_counts(rows[(member_label, historical_label)], f"{name} forward", planned_games)
    reverse = _pair_counts(rows[(historical_label, member_label)], f"{name} reverse", planned_games)
    wins = forward[0] + reverse[1]
    losses = forward[1] + reverse[0]
    draws = forward[2] + reverse[2]
    completed = wins + losses + draws
    points = wins + 0.5 * draws
    historical_points = losses + 0.5 * draws
    decided = wins + losses

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "completed": completed,
        "points": points,
        "historical_points": historical_points,
        "point_rate": points / completed,
        "delta_from_even": points / completed - 0.5,
        "decided_win_rate": wins / decided if decided else None,
        "wilson95_decided": _wilson_interval(wins, decided),
        "both_ordered_directions_combined": True,
    }


def _validate_payoff_seed(
    seed: int,
    result_path: Path,
    expected_result_sha256: str,
    addendum: Mapping[str, Any],
    identities: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    result, result_hash = _read_hash_bound_json(
        result_path,
        expected_result_sha256,
        f"ordered payoff seed {seed}",
    )
    ordered = _mapping(addendum["ordered_payoff"], "ordered payoff addendum")
    if result.get("schema_version") != PAYOFF_SCHEMA_VERSION:
        raise EvaluationValidationError(f"ordered payoff seed {seed} schema mismatch")
    expected_run_id = _mapping(ordered.get("run_ids_by_seed"), "payoff run IDs").get(str(seed))
    if result.get("run_id") != expected_run_id:
        raise EvaluationValidationError(f"ordered payoff seed {seed} run_id mismatch")
    if result.get("status") != "completed" or result.get("analysis_ready") is not True:
        raise EvaluationValidationError(f"ordered payoff seed {seed} is incomplete")
    if result.get("safety_clean") is not True:
        raise EvaluationValidationError(f"ordered payoff seed {seed} is unsafe")
    config = _validate_config_receipt(result_path, result, config_key="config", name=f"payoff seed {seed}")
    if config.get("base_seed") != seed or config.get("run_id") != expected_run_id:
        raise EvaluationValidationError(f"ordered payoff seed {seed} config identity mismatch")
    planned_games = _integer(ordered.get("games_per_ordered_pair"), "games per ordered pair", minimum=1)
    if config.get("games_per_ordered_pair") != planned_games:
        raise EvaluationValidationError(f"ordered payoff seed {seed} game budget mismatch")
    schedule = _mapping(result.get("schedule"), f"ordered payoff seed {seed} schedule")
    if (
        schedule.get("complete") is not True
        or schedule.get("all_pair_budgets_complete") is not True
        or schedule.get("member_count") != 3
        or schedule.get("ordered_pair_count") != ordered.get("ordered_pair_count_per_seed")
        or schedule.get("games_expected") != ordered.get("expected_games_per_seed")
        or schedule.get("games_attempted") != ordered.get("expected_games_per_seed")
    ):
        raise EvaluationValidationError(f"ordered payoff seed {seed} schedule mismatch")
    receipts = _mapping(config.get("input_receipts"), f"payoff seed {seed} input receipts")
    _validate_receipt(receipts.get("payoff_runner"), ordered.get("runner_sha256"), f"payoff seed {seed} runner")
    _validate_receipt(receipts.get("roster"), ordered.get("roster_sha256"), f"payoff seed {seed} roster")
    for key, identity in identities.items():
        expected_artifact = (
            identity.get("runnable_directory_sha256") if key == "r18" else identity["artifact_sha256"]
        )
        _validate_receipt(
            receipts.get(f"member_artifact:{identity['label']}"),
            expected_artifact,
            f"payoff seed {seed} {key} artifact",
        )
        if key != "r18":
            _validate_receipt(
                receipts.get(f"member_deck_binding:{identity['label']}"),
                identity["deck_sha256"],
                f"payoff seed {seed} {key} deck",
            )
    members = {
        row.get("label"): row
        for row in map(lambda item: _mapping(item, f"payoff seed {seed} member"), result.get("members") or [])
    }
    if set(members) != {identity["label"] for identity in identities.values()}:
        raise EvaluationValidationError(f"ordered payoff seed {seed} member set mismatch")
    for key, identity in identities.items():
        member = members[identity["label"]]
        if member.get("league_member_id") != identity["league_member_id"]:
            raise EvaluationValidationError(f"ordered payoff seed {seed} {key} member ID mismatch")
        if member.get("feature_policy_id") != identity["feature_policy_id"]:
            raise EvaluationValidationError(f"ordered payoff seed {seed} {key} feature policy mismatch")
        if key in {"candidate", "r7"} and member.get("deck_hash") != identity.get("normalized_deck_hash"):
            raise EvaluationValidationError(f"ordered payoff seed {seed} {key} normalized deck mismatch")
        if key == "r18" and member.get("feature_engine_deck_match") is not False:
            raise EvaluationValidationError("exact R18 feature/engine deck mismatch sentinel changed")
    rows_list = _sequence(_mapping(result.get("matrix"), f"payoff seed {seed} matrix").get("rows"), f"payoff seed {seed} rows")
    rows = {}
    expected_labels = {identity["label"] for identity in identities.values()}
    for raw_row in rows_list:
        row = _mapping(raw_row, f"payoff seed {seed} matrix row")
        key = (row.get("row_label"), row.get("column_label"))
        if key in rows:
            raise EvaluationValidationError(f"ordered payoff seed {seed} contains duplicate pair {key}")
        if key[0] not in expected_labels or key[1] not in expected_labels or key[0] == key[1]:
            raise EvaluationValidationError(f"ordered payoff seed {seed} contains unexpected pair {key}")
        rows[key] = row
        row_identity = next(identity for identity in identities.values() if identity["label"] == key[0])
        column_identity = next(identity for identity in identities.values() if identity["label"] == key[1])
        if (
            row.get("row_member_id") != row_identity["league_member_id"]
            or row.get("column_member_id") != column_identity["league_member_id"]
        ):
            raise EvaluationValidationError(f"ordered payoff seed {seed} pair member identity mismatch")
    expected_pairs = {(left, right) for left in expected_labels for right in expected_labels if left != right}
    if set(rows) != expected_pairs:
        raise EvaluationValidationError(f"ordered payoff seed {seed} ordered-pair inventory mismatch")
    for pair, row in rows.items():
        _pair_counts(row, f"payoff seed {seed} {pair[0]}->{pair[1]}", planned_games)
    candidate = identities["candidate"]
    r7 = identities["r7"]
    r18 = identities["r18"]

    return {
        "seed": seed,
        "result_sha256": result_hash,
        "run_id": result["run_id"],
        "config_fingerprint": result["config_fingerprint"],
        "m3_vs_r18": _member_vs_historical(
            rows,
            candidate["label"],
            r18["label"],
            planned_games,
            f"seed {seed} M3/R18",
        ),
        "r7_vs_r18": _member_vs_historical(
            rows,
            r7["label"],
            r18["label"],
            planned_games,
            f"seed {seed} R7/R18",
        ),
        "elapsed_wall_seconds": _finite(result.get("elapsed_wall_seconds"), f"payoff seed {seed} wall time", minimum=0.0),
        "elapsed_game_seconds": sum(
            _finite(row.get("elapsed_game_seconds"), f"payoff seed {seed} pair elapsed", minimum=0.0)
            for row in rows.values()
        ),
    }


def _summarize_seed_blocks(seed_results: Sequence[Mapping[str, Any]], comparison: str) -> dict[str, Any]:
    blocks = [_mapping(row[comparison], f"{comparison} seed aggregate") for row in seed_results]
    deltas = [_finite(block["delta_from_even"], f"{comparison} seed delta") for block in blocks]
    if len(deltas) != 3:
        raise EvaluationValidationError(f"{comparison} requires exactly three seed blocks")
    mean_delta = statistics.fmean(deltas)
    standard_error = statistics.stdev(deltas) / math.sqrt(len(deltas))
    wins = sum(_integer(block["wins"], f"{comparison} wins") for block in blocks)
    losses = sum(_integer(block["losses"], f"{comparison} losses") for block in blocks)
    draws = sum(_integer(block["draws"], f"{comparison} draws") for block in blocks)
    completed = wins + losses + draws
    points = wins + 0.5 * draws
    historical_points = losses + 0.5 * draws

    return {
        "per_seed": [
            {
                "seed": seed_result["seed"],
                **dict(_mapping(seed_result[comparison], f"{comparison} seed result")),
            }
            for seed_result in seed_results
        ],
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "completed": completed,
        "points": points,
        "historical_points": historical_points,
        "pooled_point_rate": points / completed,
        "pooled_delta_from_even": points / completed - 0.5,
        "mean_seed_delta_from_even": mean_delta,
        "sample_standard_error": standard_error,
        "mean_seed_delta_ci95_student_t_df2": _confidence_interval(
            mean_delta,
            standard_error,
            T_CRITICAL_DF2_95,
        ),
        "wilson95_decided": _wilson_interval(wins, wins + losses),
        "directions_combined_per_seed": 2,
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise EvaluationValidationError("bootstrap produced no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower

    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _elo_style_index(points: float, historical_points: float, anchor: float, scale: float, pseudocount: float) -> float:
    return anchor + scale * math.log10((points + pseudocount) / (historical_points + pseudocount))


def _rating_summary(
    seed_results: Sequence[Mapping[str, Any]],
    rating_method: Mapping[str, Any],
) -> dict[str, Any]:
    anchor = _finite(rating_method.get("anchor"), "rating anchor")
    scale = _finite(rating_method.get("scale"), "rating scale", minimum=0.0)
    pseudocount = _finite(rating_method.get("pseudocount_per_side"), "rating pseudocount", minimum=0.0)
    if scale == 0.0 or pseudocount == 0.0:
        raise EvaluationValidationError("rating scale and pseudocount must be positive")
    if rating_method.get("draw_points_per_side") != 0.5:
        raise EvaluationValidationError("frozen draw-points rule changed")
    bootstrap_seed = _integer(rating_method.get("bootstrap_seed"), "bootstrap seed")
    bootstrap_samples = _integer(rating_method.get("bootstrap_samples"), "bootstrap samples", minimum=1)
    percentiles = _sequence(rating_method.get("bootstrap_interval_percentiles"), "bootstrap percentiles")
    if len(percentiles) != 2:
        raise EvaluationValidationError("bootstrap interval requires two percentiles")
    lower_percentile = _finite(percentiles[0], "lower bootstrap percentile", minimum=0.0)
    upper_percentile = _finite(percentiles[1], "upper bootstrap percentile", minimum=0.0)
    if not lower_percentile < upper_percentile <= 100.0:
        raise EvaluationValidationError("bootstrap percentiles are invalid")
    if (
        anchor != 677.9
        or scale != 400.0
        or pseudocount != 0.5
        or bootstrap_seed != 7132026
        or bootstrap_samples != 10000
        or rating_method.get("external_anchor_provenance") != "user_supplied_mission_brief_2026-07-13"
        or rating_method.get("locally_verified") is not False
        or rating_method.get("interpretation") != "local_index_only_not_a_Kaggle_score_prediction"
    ):
        raise EvaluationValidationError("frozen local rating method identity changed")
    blocks = [
        (
            _finite(row["m3_vs_r18"]["points"], "M3 seed points", minimum=0.0),
            _finite(row["m3_vs_r18"]["historical_points"], "R18 seed points", minimum=0.0),
        )
        for row in seed_results
    ]
    total_points = sum(block[0] for block in blocks)
    total_historical = sum(block[1] for block in blocks)
    point_index = _elo_style_index(total_points, total_historical, anchor, scale, pseudocount)
    rng = random.Random(bootstrap_seed)
    bootstrap_values = []
    for _ in range(bootstrap_samples):
        sampled = [blocks[rng.randrange(len(blocks))] for _ in blocks]
        bootstrap_values.append(
            _elo_style_index(
                sum(block[0] for block in sampled),
                sum(block[1] for block in sampled),
                anchor,
                scale,
                pseudocount,
            )
        )

    return {
        "label": rating_method.get("label"),
        "formula": rating_method.get("formula"),
        "anchor": anchor,
        "anchor_provenance": rating_method.get("external_anchor_provenance"),
        "anchor_locally_verified": False,
        "anchor_submission_id": rating_method.get("external_submission_id"),
        "anchor_episode_count": rating_method.get("external_episode_count"),
        "m3_points": total_points,
        "r18_points": total_historical,
        "local_elo_style_index": point_index,
        "bootstrap_unit": rating_method.get("bootstrap_unit"),
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_standard_error": statistics.stdev(bootstrap_values),
        "bootstrap_interval_percentiles": [lower_percentile, upper_percentile],
        "bootstrap_interval": [
            _percentile(bootstrap_values, lower_percentile),
            _percentile(bootstrap_values, upper_percentile),
        ],
        "interpretation": "local_index_only_not_a_Kaggle_score_prediction",
        "kaggle_score_prediction": False,
    }


def _validate_offline_receipt(
    path: Path | None,
    expected_sha256: str | None,
    candidate: Mapping[str, Any],
    r7: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any] | None:
    if path is None and expected_sha256 is None:
        return None
    if path is None or expected_sha256 is None:
        raise EvaluationValidationError("acceptance receipt path and hash must be supplied together")
    receipt, receipt_hash = _read_hash_bound_json(path, expected_sha256, "acceptance receipt")
    if receipt.get("schema_version") != ACCEPTANCE_SCHEMA_VERSION or receipt.get("status") != "validated":
        raise EvaluationValidationError("acceptance receipt schema or status mismatch")
    lineage = _mapping(receipt.get("verified_lineage"), "acceptance verified lineage")
    hashes = _mapping(receipt.get("verified_artifact_hashes"), "acceptance verified hashes")
    trainer_hashes = _mapping(_mapping(hashes.get("trainers"), "acceptance trainer hashes").get("m3"), "M3 hashes")
    if lineage.get("source_policy_sha256") != r7["artifact_sha256"]:
        raise EvaluationValidationError("acceptance receipt R7 lineage mismatch")
    if trainer_hashes.get("artifact") != candidate["artifact_sha256"]:
        raise EvaluationValidationError("acceptance receipt candidate lineage mismatch")
    offline = receipt.get("normalized_offline")
    if offline is None:
        return {
            "receipt_sha256": receipt_hash,
            "available": False,
            "support_met": False,
            "audit_all_passed": False,
        }
    offline = _mapping(offline, "normalized offline evidence")
    _required(offline, ("heldout_rows", "heldout_groups", "mean_teacher_regret_delta"), "normalized offline evidence")
    effect = _mapping(offline["mean_teacher_regret_delta"], "offline teacher-regret effect")
    mean = _finite(effect.get("mean"), "offline regret mean")
    lower = _finite(effect.get("lower"), "offline regret lower")
    upper = _finite(effect.get("upper"), "offline regret upper")
    if not lower <= mean <= upper:
        raise EvaluationValidationError("offline teacher-regret confidence interval is invalid")
    minimum = _mapping(protocol.get("minimum_support"), "source minimum support")
    heldout_rows = _integer(offline["heldout_rows"], "offline heldout rows")
    heldout_groups = _integer(offline["heldout_groups"], "offline heldout groups")
    support_met = (
        heldout_rows >= _integer(minimum.get("heldout_rows"), "minimum heldout rows", minimum=1)
        and heldout_groups >= _integer(minimum.get("heldout_groups"), "minimum heldout groups", minimum=1)
    )
    audit = _mapping(receipt.get("normalized_audit"), "normalized acceptance audit")
    audit_status = dict(audit)
    audit_status["equal_budget"] = "passed"
    audit_all_passed = all(value == "passed" for value in audit_status.values())

    return {
        "receipt_sha256": receipt_hash,
        "available": True,
        "heldout_rows": heldout_rows,
        "heldout_groups": heldout_groups,
        "mean_teacher_regret_delta": {"mean": mean, "lower": lower, "upper": upper},
        "support_met": support_met,
        "audit": audit_status,
        "audit_all_passed": audit_all_passed,
    }


def _validate_verdict_rules(addendum: Mapping[str, Any]) -> None:
    rules = _mapping(addendum["verdict_rules"], "verdict rules")
    allowed = tuple(_sequence(rules.get("allowed_families"), "allowed verdict families"))
    if set(allowed) != set(VERDICT_FAMILIES) or len(allowed) != len(VERDICT_FAMILIES):
        raise EvaluationValidationError("allowed tiny-data verdict families changed")
    if rules.get("forbidden_family") != "Q_REJECTED":
        raise EvaluationValidationError("Q_REJECTED forbidden-family rule changed")


def _recommend_verdict(
    shared: Mapping[str, Any],
    historical: Mapping[str, Any],
    offline: Mapping[str, Any] | None,
) -> dict[str, Any]:
    reasons = []
    shared_ci = _sequence(shared["independent_aggregate_delta_ci95"], "shared-panel delta CI")
    historical_severe = historical["severe_regression_vs_r7"]
    if shared["severe_regression_count"] > 0:
        reasons.append(f"shared_panel_severe_regression_cells={shared['severe_regression_count']}")
    if _finite(shared_ci[1], "shared delta CI upper") < 0.0:
        reasons.append("shared_panel_delta_confidence_upper_less_than_zero")
    if historical_severe:
        reasons.append("m3_vs_exact_r18_wilson95_upper_less_than_r7_vs_exact_r18_wilson95_lower")
    if offline is not None and offline.get("available") is True:
        effect = _mapping(offline["mean_teacher_regret_delta"], "offline effect")
        if _finite(effect["lower"], "offline regret lower") > 0.0:
            reasons.append("offline_teacher_regret_delta_confidence_lower_greater_than_zero")
    if reasons:
        return {
            "family": "TINY_DATA_Q_CHECKPOINT_REJECTED_METHOD_UNTESTED",
            "reasons": reasons,
            "parent_owns_final_verdict": True,
        }
    promising_reasons = []
    if _finite(shared_ci[0], "shared delta CI lower") <= 0.0:
        promising_reasons.append("shared_panel_improvement_not_resolved")
    if offline is None or offline.get("available") is not True:
        promising_reasons.append("offline_teacher_regret_not_supplied")
    else:
        effect = _mapping(offline["mean_teacher_regret_delta"], "offline effect")
        if _finite(effect["upper"], "offline regret upper") >= 0.0:
            promising_reasons.append("offline_teacher_regret_improvement_not_resolved")
        if offline.get("support_met") is not True:
            promising_reasons.append("offline_minimum_support_not_met")
        if offline.get("audit_all_passed") is not True:
            promising_reasons.append("offline_audit_not_all_passed")
    if promising_reasons:
        return {
            "family": "TINY_DATA_Q_PROTOTYPE_INCONCLUSIVE",
            "reasons": promising_reasons,
            "parent_owns_final_verdict": True,
        }

    return {
        "family": "TINY_DATA_Q_PROTOTYPE_PROMISING",
        "reasons": [
            "shared_panel_delta_confidence_lower_greater_than_zero",
            "zero_severe_regressions",
            "offline_teacher_regret_delta_confidence_upper_less_than_zero",
            "offline_minimum_support_and_audit_gates_met",
        ],
        "parent_owns_final_verdict": True,
    }


def analyze_evaluation(
    addendum_path: Path,
    *,
    expected_addendum_sha256: str,
    expected_shared_result_sha256: str,
    expected_payoff_sha256s: Mapping[int, str],
    acceptance_receipt_path: Path | None = None,
    expected_acceptance_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Return aggregate-only evidence after strict result validation."""

    addendum_path = Path(addendum_path).resolve()
    addendum, addendum_hash = _read_hash_bound_json(
        addendum_path,
        expected_addendum_sha256,
        "frozen identity addendum",
    )
    _required(
        addendum,
        (
            "schema_version",
            "status",
            "classification",
            "research_only",
            "promotion_authorized",
            "source_protocol",
            "identities",
            "archive_equivalence",
            "shared_panel",
            "ordered_payoff",
            "rating_method",
            "verdict_rules",
        ),
        "frozen identity addendum",
    )
    if addendum.get("schema_version") != ADDENDUM_SCHEMA_VERSION:
        raise EvaluationValidationError("frozen identity addendum schema mismatch")
    if (
        addendum.get("status") != "frozen_not_executed"
        or addendum.get("classification") != "quarantine"
        or addendum.get("research_only") is not True
        or addendum.get("promotion_authorized") is not False
    ):
        raise EvaluationValidationError("frozen identity addendum boundary mismatch")
    _validate_verdict_rules(addendum)
    base_dir = addendum_path.parent
    protocol = _validate_source_protocol(addendum, base_dir)
    candidate = _identity(addendum, "candidate")
    r7 = _identity(addendum, "r7")
    r18 = _identity(addendum, "r18")
    _validate_protocol_identities(protocol, candidate, r7)
    if r18.get("feature_engine_deck_match") is not False:
        raise EvaluationValidationError("exact R18 mismatch sentinel is not frozen false")
    archive_equivalence = _validate_archive_equivalence(addendum, base_dir, r18)
    shared_config = _mapping(addendum["shared_panel"], "shared-panel addendum")
    shared_path = _resolve_path(shared_config.get("result_path"), base_dir, "shared-panel result path")
    shared = _validate_shared_panel(
        shared_path,
        expected_shared_result_sha256,
        addendum,
        candidate,
        r7,
    )
    supplied_seed_hashes = {int(seed): value for seed, value in expected_payoff_sha256s.items()}
    if set(supplied_seed_hashes) != set(EXPECTED_SEEDS):
        raise EvaluationValidationError("payoff result hashes must be supplied for exactly the three frozen seeds")
    ordered = _mapping(addendum["ordered_payoff"], "ordered payoff addendum")
    frozen_seeds = tuple(ordered.get("seeds") or ())
    if frozen_seeds != EXPECTED_SEEDS:
        raise EvaluationValidationError("ordered payoff frozen seeds changed")
    result_paths = _mapping(ordered.get("results_by_seed"), "ordered payoff result paths")
    if set(result_paths) != {str(seed) for seed in EXPECTED_SEEDS}:
        raise EvaluationValidationError("ordered payoff result path inventory mismatch")
    identities = {"candidate": candidate, "r7": r7, "r18": r18}
    seed_results = [
        _validate_payoff_seed(
            seed,
            _resolve_path(result_paths[str(seed)], base_dir, f"ordered payoff seed {seed} result path"),
            supplied_seed_hashes[seed],
            addendum,
            identities,
        )
        for seed in EXPECTED_SEEDS
    ]
    m3_historical = _summarize_seed_blocks(seed_results, "m3_vs_r18")
    r7_historical = _summarize_seed_blocks(seed_results, "r7_vs_r18")
    historical_severe = (
        m3_historical["wilson95_decided"][1] < r7_historical["wilson95_decided"][0]
    )
    historical = {
        "historical_member_id": ordered.get("historical_member_id"),
        "comparison_semantics": ordered.get("comparison_semantics"),
        "m3_vs_exact_r18": m3_historical,
        "r7_vs_exact_r18": r7_historical,
        "severe_regression_rule": "m3_wilson95_upper < r7_wilson95_lower against exact R18",
        "severe_regression_vs_r7": historical_severe,
        "result_artifacts": [
            {
                "seed": row["seed"],
                "run_id": row["run_id"],
                "config_fingerprint": row["config_fingerprint"],
                "sha256": row["result_sha256"],
                "elapsed_wall_seconds": row["elapsed_wall_seconds"],
                "elapsed_game_seconds": row["elapsed_game_seconds"],
            }
            for row in seed_results
        ],
    }
    if historical["historical_member_id"] != r18["league_member_id"]:
        raise EvaluationValidationError("ordered payoff historical member ID mismatch")
    rating = _rating_summary(seed_results, _mapping(addendum["rating_method"], "rating method"))
    offline = _validate_offline_receipt(
        Path(acceptance_receipt_path).resolve() if acceptance_receipt_path is not None else None,
        expected_acceptance_receipt_sha256,
        candidate,
        r7,
        protocol,
    )
    recommendation = _recommend_verdict(shared, historical, offline)
    evidence = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "validated",
        "classification": "quarantine",
        "research_only": True,
        "promotion_authorized": False,
        "tiny_data_interpretation": "64_unique_row_checkpoint_prototype_only",
        "input_bindings": {
            "addendum_path": str(addendum_path),
            "addendum_sha256": addendum_hash,
            "source_protocol_sha256": addendum["source_protocol"]["sha256"],
            "shared_panel_result_sha256": shared["result_sha256"],
            "ordered_payoff_result_sha256_by_seed": {
                str(row["seed"]): row["result_sha256"] for row in seed_results
            },
            "acceptance_receipt_sha256": offline.get("receipt_sha256") if offline is not None else None,
        },
        "identities": {
            key: {
                field: value
                for field, value in identity.items()
                if field in {
                    "label",
                    "league_member_id",
                    "artifact_sha256",
                    "runnable_directory_sha256",
                    "weights_sha256",
                    "deck_sha256",
                    "normalized_deck_hash",
                    "feature_policy_id",
                    "feature_engine_deck_match",
                    "artifact_interpretation",
                    "data_disclosure",
                }
            }
            for key, identity in identities.items()
        },
        "archive_equivalence": archive_equivalence,
        "shared_panel_candidate_vs_r7": shared,
        "historical_retention": historical,
        "r18_anchored_local_rating": rating,
        "offline_evidence": offline,
        "aggregate_safety": {
            "illegal": shared["safety"]["illegal"],
            "crashes": shared["safety"]["crashes"],
            "timeouts": shared["safety"]["timeouts"],
            "incomplete": shared["safety"]["incomplete"],
            "ordered_payoff_all_pairs_safety_clean": True,
        },
        "verdict_recommendation": recommendation,
        "forbidden_verdict_family": "Q_REJECTED",
        "rating_is_kaggle_prediction": False,
        "runtime_mutated": False,
        "submission_mutated": False,
    }
    evidence["evidence_sha256"] = _sha256_json(evidence)

    return evidence


def _technical_failure(reason: str) -> dict[str, Any]:
    value = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "technical_evaluation_failure",
        "classification": "quarantine",
        "research_only": True,
        "promotion_authorized": False,
        "verdict_recommendation": {
            "family": "TECHNICAL_EVALUATION_FAILURE",
            "reasons": [reason],
            "parent_owns_final_verdict": True,
        },
        "forbidden_verdict_family": "Q_REJECTED",
        "rating_is_kaggle_prediction": False,
        "runtime_mutated": False,
        "submission_mutated": False,
    }
    value["evidence_sha256"] = _sha256_json(value)

    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(value) + b"\n"
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_seed_hash(value: str) -> tuple[int, str]:
    seed_text, separator, sha256 = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("ordered payoff hash must be SEED=SHA256")
    try:
        seed = int(seed_text)
        normalized = _sha256(sha256, "ordered payoff sha256")
    except (ValueError, EvaluationValidationError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc

    return seed, normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addendum", required=True, help="frozen identity addendum JSON")
    parser.add_argument("--addendum-sha256", required=True, help="expected frozen addendum SHA-256")
    parser.add_argument("--shared-result-sha256", required=True, help="completed shared-panel results SHA-256")
    parser.add_argument(
        "--ordered-payoff-sha256",
        action="append",
        type=_parse_seed_hash,
        required=True,
        metavar="SEED=SHA256",
        help="completed seed-specific ordered-payoff results SHA-256; repeat for all three seeds",
    )
    parser.add_argument("--acceptance-receipt", help="optional normalized offline/audit acceptance receipt")
    parser.add_argument("--acceptance-receipt-sha256", help="expected optional acceptance receipt SHA-256")
    parser.add_argument("--out", required=True, help="atomic aggregate evidence JSON output")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payoff_hashes = dict(args.ordered_payoff_sha256)
    if len(payoff_hashes) != len(args.ordered_payoff_sha256):
        failure = _technical_failure("duplicate ordered-payoff seed hash arguments")
        _write_json(Path(args.out), failure)
        print(json.dumps(failure, indent=2, sort_keys=True))

        return 2
    try:
        result = analyze_evaluation(
            Path(args.addendum),
            expected_addendum_sha256=args.addendum_sha256,
            expected_shared_result_sha256=args.shared_result_sha256,
            expected_payoff_sha256s=payoff_hashes,
            acceptance_receipt_path=Path(args.acceptance_receipt) if args.acceptance_receipt else None,
            expected_acceptance_receipt_sha256=args.acceptance_receipt_sha256,
        )
    except EvaluationValidationError as exc:
        result = _technical_failure(str(exc))
        _write_json(Path(args.out), result)
        print(json.dumps(result, indent=2, sort_keys=True))

        return 2
    _write_json(Path(args.out), result)
    print(json.dumps(result, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
