"""One-shot aggregate-only scorer for frozen label-blind value predictions."""

from __future__ import annotations

import gzip
import json
import math
import os
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import deepmew.value_lab.calibration as value_calibration
import deepmew.value_lab.calibration_selection as value_calibration_selection
import deepmew.value_lab.control_pass_mask as value_control_pass_mask
import deepmew.value_lab.crossfit as value_crossfit
import deepmew.value_lab.data as value_data
import deepmew.value_lab.mc_screen as value_mc_screen
import deepmew.value_lab.metrics as value_metrics
import deepmew.value_lab.models as value_models
import deepmew.value_lab.sealed_test_prediction as value_prediction
from deepmew.artifacts.manifest import build_manifest, read_manifest, write_manifest
from deepmew.artifacts.writers import write_json
from deepmew.value_lab.crossfit import (
    _file_receipts,
    _sha256,
    _stable_hash,
    _verify_output_files,
)
from deepmew.value_lab.data import LabData
from deepmew.value_lab.metrics import full_metrics, paired_game_cluster_bootstrap


SEALED_SCORE_SCHEMA_VERSION = "deepmew.value_sealed_test_score.v1"
PREDICTION_ARRAY_KEYS = {
    "row_id",
    "block_id",
    "game_gidx",
    "source_idx",
    "policy_gidx",
    "opp_policy_gidx",
    "deck_gidx",
    "opp_deck_gidx",
    "seat",
    "turn",
    "decision_index",
    "select_context",
    "legal_count",
    "is_target_eligible",
    "control_pass_known",
    "control_passed",
    "raw_probability",
    "calibrated_probability",
    "phase4_screen_seed_915001_probability",
    "phase4_screen_seed_915002_probability",
    "phase4_screen_ensemble_probability",
    "s0_probability",
}


def _current_prediction_code_hashes() -> dict[str, str]:
    return {
        "sealed_test_prediction": _sha256(Path(value_prediction.__file__)),
        "calibration": _sha256(Path(value_calibration.__file__)),
        "calibration_selection": _sha256(Path(value_calibration_selection.__file__)),
        "control_pass_mask": _sha256(Path(value_control_pass_mask.__file__)),
        "crossfit": _sha256(Path(value_crossfit.__file__)),
        "data": _sha256(Path(value_data.__file__)),
        "mc_screen": _sha256(Path(value_mc_screen.__file__)),
        "models": _sha256(Path(value_models.__file__)),
    }


def verify_prediction_artifact(prediction_dir: Path) -> dict[str, Any]:
    prediction_dir = prediction_dir.resolve()
    manifest = read_manifest(prediction_dir / "manifest.json")
    config = manifest.get("inputs", {}).get("config")
    summary = manifest.get("summary", {})
    if (
        manifest.get("kind") != "value_sealed_test_prediction"
        or manifest.get("status") != "completed"
        or summary.get("status") != "completed_label_blind_predictions_frozen"
        or not isinstance(config, dict)
    ):
        raise ValueError("sealed prediction manifest contract mismatch")
    declared_config_sha256 = config.get("sha256")
    config_payload = {key: value for key, value in config.items() if key != "sha256"}
    if (
        config.get("schema_version") != value_prediction.SEALED_PREDICTION_SCHEMA_VERSION
        or _stable_hash(config_payload) != declared_config_sha256
        or manifest["inputs"].get("config_sha256") != declared_config_sha256
        or config.get("code_sha256") != _current_prediction_code_hashes()
        or config.get("test_targets_consumed") is not False
        or summary.get("label_arrays_loaded") is not False
        or summary.get("test_targets_consumed") is not False
        or summary.get("nonfinite_predictions") != 0
    ):
        raise ValueError("sealed prediction lineage or label-blind contract mismatch")
    files = manifest.get("outputs", {}).get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("sealed prediction output inventory is missing")
    _verify_output_files(prediction_dir, files)
    plan_path = prediction_dir / "SCORING_PLAN.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        plan.get("schema_version") != value_prediction.SCORING_PLAN_SCHEMA_VERSION
        or plan.get("status") != "frozen_before_test_labels"
        or plan.get("test_targets_consumed") is not False
        or _stable_hash(plan) != config.get("scoring_plan_sha256")
        or _sha256(plan_path) != summary.get("scoring_plan_sha256")
    ):
        raise ValueError("sealed scoring plan hash or state mismatch")
    expected_scoring_code = {
        "sealed_test_scoring": _sha256(Path(__file__)),
        "data": _sha256(Path(value_data.__file__)),
        "mc_screen": _sha256(Path(value_mc_screen.__file__)),
        "metrics": _sha256(Path(value_metrics.__file__)),
    }
    plan_bindings = plan.get("bindings", {})
    if (
        plan_bindings != config.get("bindings")
        or plan_bindings.get("scoring_code_sha256") != expected_scoring_code
        or plan.get("canonical_test_lock_path")
        != str(value_prediction.canonical_test_lock_path(plan_bindings["split_manifest_sha256"]))
    ):
        raise ValueError("sealed scoring plan dependency or lock binding mismatch")
    result = json.loads((prediction_dir / "result.json").read_text(encoding="utf-8"))
    prediction_path = prediction_dir / "test_predictions.npz"
    if (
        result.get("status") != "completed_label_blind_predictions_frozen"
        or result.get("label_arrays_loaded") is not False
        or result.get("test_targets_consumed") is not False
        or result.get("bindings") != plan_bindings
        or result.get("prediction_sha256") != summary.get("prediction_sha256")
        or _sha256(prediction_path) != result.get("prediction_sha256")
    ):
        raise ValueError("sealed prediction result contract mismatch")
    with np.load(prediction_path, allow_pickle=False) as stored:
        if set(stored.files) != PREDICTION_ARRAY_KEYS:
            raise ValueError("sealed prediction array inventory is invalid")
        arrays = {key: stored[key] for key in stored.files}
    rows = int(summary["context_rows"])
    if any(np.asarray(value).shape != (rows,) for value in arrays.values()):
        raise ValueError("sealed prediction arrays have inconsistent shapes")
    row_id = arrays["row_id"].astype(np.int64, copy=False)
    probabilities = [
        arrays[key].astype(np.float64, copy=False)
        for key in PREDICTION_ARRAY_KEYS
        if key.endswith("probability")
    ]
    if (
        np.any(np.diff(row_id) <= 0)
        or int(arrays["is_target_eligible"].sum()) != summary.get("eligible_targets")
        or rows - int(arrays["is_target_eligible"].sum()) != summary.get("excluded_targets")
        or np.any(arrays["control_passed"] & ~arrays["control_pass_known"])
        or any(
            not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0))
            for values in probabilities
        )
        or float(np.ptp(arrays["s0_probability"].astype(np.float64))) != 0.0
        or not np.isclose(
            float(arrays["s0_probability"][0]),
            float(plan["comparators"]["s0"]["probability"]),
            rtol=0.0,
            atol=np.finfo(np.float32).eps,
        )
    ):
        raise ValueError("sealed prediction coverage or probability contract mismatch")
    with gzip.open(prediction_dir / "game_vocab.json.gz", "rt", encoding="utf-8") as handle:
        game_vocab = json.load(handle)["game_ids"]
    if (
        len(game_vocab) == 0
        or np.any(arrays["game_gidx"] < 0)
        or np.any(arrays["game_gidx"] >= len(game_vocab))
        or len(np.unique(arrays["game_gidx"])) != summary.get("games")
    ):
        raise ValueError("sealed prediction game vocabulary is invalid")

    return {
        "manifest": manifest,
        "config": config,
        "summary": summary,
        "plan": plan,
        "result": result,
        "arrays": arrays,
        "game_vocab": game_vocab,
    }


def _preflight_context(
    *,
    cache_dir: Path,
    split_path: Path,
    prediction_dir: Path,
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], LabData]:
    cache_dir = cache_dir.resolve()
    split_path = split_path.resolve()
    out_dir = out_dir.resolve()
    verified = verify_prediction_artifact(prediction_dir)
    bindings = verified["plan"]["bindings"]
    cache_manifest_sha256 = _sha256(cache_dir / "cache_manifest.json")
    split_manifest_sha256 = _sha256(split_path)
    if (
        cache_manifest_sha256 != bindings.get("cache_manifest_sha256")
        or split_manifest_sha256 != bindings.get("split_manifest_sha256")
    ):
        raise ValueError("sealed scoring cache or split lineage mismatch")
    data = LabData(
        str(cache_dir),
        split_manifest_path=str(split_path),
        load_labels=False,
    )
    arrays = verified["arrays"]
    test_index = data.split_names.index("test")
    expected_rows = np.flatnonzero(data.meta["split_idx"] == test_index).astype(np.int64)
    starts = data.blocks["row_start"].astype(np.int64, copy=False)
    lengths = data.blocks["row_len"].astype(np.int64, copy=False)
    expected_block = np.searchsorted(starts, expected_rows, side="right") - 1
    if np.any(expected_block < 0) or np.any(
        expected_rows >= starts[expected_block] + lengths[expected_block]
    ):
        raise ValueError("accepted test rows do not map to actor blocks")
    block_game, game_ids = value_mc_screen.canonical_game_indices(data)
    expected_metadata = {
        "block_id": expected_block.astype(np.int32),
        "game_gidx": block_game[expected_block].astype(np.int32, copy=False),
        "source_idx": data.meta["source_idx"][expected_rows].astype(np.int32, copy=False),
        "policy_gidx": data.blocks["policy_gidx"][expected_block].astype(np.int32, copy=False),
        "opp_policy_gidx": data.blocks["opp_policy_gidx"][expected_block].astype(
            np.int32, copy=False,
        ),
        "deck_gidx": data.blocks["deck_gidx"][expected_block].astype(np.int32, copy=False),
        "opp_deck_gidx": data.blocks["opp_deck_gidx"][expected_block].astype(
            np.int32, copy=False,
        ),
        "seat": data.meta["seat"][expected_rows].astype(np.int32, copy=False),
        "turn": data.meta["turn"][expected_rows].astype(np.int32, copy=False),
        "decision_index": data.meta["decision_index"][expected_rows].astype(
            np.int32, copy=False,
        ),
        "select_context": data.meta["select_context"][expected_rows].astype(
            np.int32, copy=False,
        ),
        "legal_count": data.meta["legal_count"][expected_rows].astype(np.int32, copy=False),
    }
    if (
        data.has_labels
        or not np.array_equal(arrays["row_id"], expected_rows)
        or not np.array_equal(
            arrays["is_target_eligible"],
            data.meta["is_target_eligible"][expected_rows],
        )
        or any(
            not np.array_equal(arrays[key], expected)
            for key, expected in expected_metadata.items()
        )
        or verified["game_vocab"] != game_ids
    ):
        raise ValueError("sealed prediction rows do not match label-blind accepted test coverage")
    lock_path = value_prediction.canonical_test_lock_path(split_manifest_sha256)

    receipt = {
        "status": "preflight_passed_labels_still_sealed",
        "prediction_manifest_sha256": _sha256(Path(prediction_dir) / "manifest.json"),
        "scoring_plan_sha256": _sha256(Path(prediction_dir) / "SCORING_PLAN.json"),
        "cache_manifest_sha256": cache_manifest_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "context_rows": int(len(expected_rows)),
        "eligible_targets": int(arrays["is_target_eligible"].sum()),
        "lock_path": str(lock_path),
        "lock_exists": lock_path.exists(),
        "output_exists": out_dir.exists(),
        "label_arrays_loaded": False,
        "test_targets_consumed": False,
    }

    return receipt, verified, data


def preflight_sealed_scoring(
    *,
    cache_dir: Path,
    split_path: Path,
    prediction_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    receipt, _verified, _data = _preflight_context(
        cache_dir=cache_dir,
        split_path=split_path,
        prediction_dir=prediction_dir,
        out_dir=out_dir,
    )

    return receipt


def _light_metrics(p: np.ndarray, z: np.ndarray) -> dict[str, Any]:
    return {
        "n": int(len(z)),
        "mean_p": float(np.mean(p)),
        "mean_z": float(np.mean(z)),
        "log_loss": value_metrics.log_loss(p, z),
        "brier": value_metrics.brier(p, z),
        "ece": value_metrics.ece(p, z)["ece"],
        "auc": value_metrics.auc(p, z),
    }


def _slice_metrics(
    p: np.ndarray,
    s0: np.ndarray,
    z: np.ndarray,
    game_gidx: np.ndarray,
    keys: np.ndarray,
    *,
    names: list[str] | tuple[str, ...] | None,
    minimum_rows: int,
    minimum_games: int,
) -> dict[str, dict[str, Any]]:
    result = {}
    for raw_key in np.unique(keys):
        mask = keys == raw_key
        rows = int(mask.sum())
        games = int(len(np.unique(game_gidx[mask])))
        index = int(raw_key)
        name = names[index] if names is not None and 0 <= index < len(names) else str(index)
        supported = rows >= minimum_rows and games >= minimum_games
        result[name] = {
            "rows": rows,
            "games": games,
            "supported": supported,
        }
        if not supported:
            continue
        candidate = _light_metrics(p[mask], z[mask])
        reference = _light_metrics(s0[mask], z[mask])
        result[name].update({
            "calibrated": candidate,
            "s0": reference,
            "log_loss_damage_vs_s0": candidate["log_loss"] - reference["log_loss"],
            "brier_damage_vs_s0": candidate["brier"] - reference["brier"],
        })

    return result


def _worst_supported_slice(slices: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any] | None:
    candidates = []
    for family, values in slices.items():
        for name, receipt in values.items():
            if receipt["supported"]:
                candidates.append({"family": family, "name": name, **receipt})
    if not candidates:
        return None

    return max(candidates, key=lambda item: item["log_loss_damage_vs_s0"])


def _paired_comparison(
    candidate: np.ndarray,
    reference: np.ndarray,
    z: np.ndarray,
    game_gidx: np.ndarray,
    plan: dict[str, Any],
) -> dict[str, Any]:
    bootstrap = plan["bootstrap"]

    return {
        metric: paired_game_cluster_bootstrap(
            candidate,
            reference,
            z,
            game_gidx,
            metric=metric,
            replicates=int(bootstrap["replicates"]),
            seed=int(bootstrap["seed"]),
        )
        for metric in ("log_loss", "brier")
    }


def score_aggregate_arrays(
    arrays: dict[str, np.ndarray],
    z: np.ndarray,
    *,
    source_ids: list[str],
    policy_vocab: list[str],
    deck_vocab: list[str],
    plan: dict[str, Any],
    inference: dict[str, Any],
) -> dict[str, Any]:
    eligible = arrays["is_target_eligible"].astype(np.bool_, copy=False)
    if z.shape != (int(eligible.sum()),):
        raise ValueError("sealed labels are not aligned to eligible prediction rows")
    selected = {key: np.asarray(value)[eligible] for key, value in arrays.items()}
    game_gidx = selected["game_gidx"].astype(np.int32, copy=False)
    calibrated = selected["calibrated_probability"].astype(np.float64, copy=False)
    raw = selected["raw_probability"].astype(np.float64, copy=False)
    screening = selected["phase4_screen_ensemble_probability"].astype(np.float64, copy=False)
    s0 = selected["s0_probability"].astype(np.float64, copy=False)
    metrics = {
        "s0": full_metrics(s0, z),
        "raw_final_v": full_metrics(raw, z),
        "calibrated_final_v": full_metrics(calibrated, z),
        "phase4_screening_ensemble": full_metrics(screening, z),
    }
    comparisons = {
        "calibrated_final_v_vs_s0": _paired_comparison(calibrated, s0, z, game_gidx, plan),
        "raw_final_v_vs_s0": _paired_comparison(raw, s0, z, game_gidx, plan),
        "calibrated_final_v_vs_raw_final_v": _paired_comparison(
            calibrated, raw, z, game_gidx, plan,
        ),
        "calibrated_final_v_vs_phase4_screening": _paired_comparison(
            calibrated, screening, z, game_gidx, plan,
        ),
    }
    support = plan["slice_support"]
    minimum_rows = int(support["minimum_eligible_rows"])
    minimum_games = int(support["minimum_games"])
    phase = np.where(selected["turn"] <= 5, 0, np.where(selected["turn"] <= 9, 1, 2))
    slices = {
        "source": _slice_metrics(
            calibrated,
            s0,
            z,
            game_gidx,
            selected["source_idx"],
            names=source_ids,
            minimum_rows=minimum_rows,
            minimum_games=minimum_games,
        ),
        "deck": _slice_metrics(
            calibrated,
            s0,
            z,
            game_gidx,
            selected["deck_gidx"],
            names=deck_vocab,
            minimum_rows=minimum_rows,
            minimum_games=minimum_games,
        ),
        "policy": _slice_metrics(
            calibrated,
            s0,
            z,
            game_gidx,
            selected["policy_gidx"],
            names=policy_vocab,
            minimum_rows=minimum_rows,
            minimum_games=minimum_games,
        ),
        "phase": _slice_metrics(
            calibrated,
            s0,
            z,
            game_gidx,
            phase,
            names=("early_turn_0_5", "mid_turn_6_9", "late_turn_10_plus"),
            minimum_rows=minimum_rows,
            minimum_games=minimum_games,
        ),
    }
    source_damages = [
        value["log_loss_damage_vs_s0"]
        for value in slices["source"].values()
        if value["supported"]
    ]
    macro_source_damage = float(np.mean(source_damages)) if source_damages else float("nan")
    worst = _worst_supported_slice({
        family: slices[family] for family in ("source", "deck", "policy")
    })
    control_mask = selected["control_pass_known"] & selected["control_passed"]
    control_games = int(len(np.unique(game_gidx[control_mask]))) if control_mask.any() else 0
    control_source_indices = np.unique(selected["source_idx"][control_mask])
    control_sources = [
        source_ids[int(index)] if 0 <= int(index) < len(source_ids) else str(int(index))
        for index in control_source_indices
    ]
    control_pass = {
        "rows": int(control_mask.sum()),
        "games": control_games,
        "supported": int(control_mask.sum()) >= minimum_rows and control_games >= minimum_games,
        "sources": control_sources,
        "source_concentrated": len(control_sources) <= 1,
    }
    if control_pass["supported"]:
        control_candidate = _light_metrics(calibrated[control_mask], z[control_mask])
        control_s0 = _light_metrics(s0[control_mask], z[control_mask])
        control_pass.update({
            "calibrated": control_candidate,
            "s0": control_s0,
            "log_loss_damage_vs_s0": control_candidate["log_loss"] - control_s0["log_loss"],
            "brier_damage_vs_s0": control_candidate["brier"] - control_s0["brier"],
        })
    result = {
        "metrics": metrics,
        "comparisons": comparisons,
        "slices": slices,
        "macro_source_log_loss_damage_vs_s0": macro_source_damage,
        "worst_supported_source_deck_or_policy": worst,
        "control_pass": control_pass,
        "inference": inference,
        "old_value_model": "not_admissible_due_accepted_split_overlap",
    }
    result["gate"] = evaluate_gate(result, plan)

    return result


def evaluate_gate(result: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]["calibrated_final_v"]
    comparison = result["comparisons"]["calibrated_final_v_vs_s0"]
    research_plan = plan["selected_research_v_gate"]
    latency = float(result["inference"]["latency_microseconds_per_row"])
    research_checks = {
        "safety_exact_coverage_and_finite": True,
        "log_loss_vs_s0_ci95_upper_below_zero": comparison["log_loss"]["ci95"][1] < 0.0,
        "brier_vs_s0_ci95_upper_at_most_zero": comparison["brier"]["ci95"][1] <= 0.0,
        "ece_at_most": metrics["ece"] <= float(research_plan["ece_at_most"]),
        "calibration_slope_in_range": (
            math.isfinite(metrics["slope"])
            and research_plan["calibration_slope_range"][0]
            <= metrics["slope"]
            <= research_plan["calibration_slope_range"][1]
        ),
        "absolute_calibration_intercept_at_most": (
            math.isfinite(metrics["intercept"])
            and abs(metrics["intercept"])
            <= float(research_plan["absolute_calibration_intercept_at_most"])
        ),
        "latency_at_most": latency
        <= float(research_plan["batch_latency_microseconds_per_row_at_most"]),
    }
    if all(research_checks.values()):
        research_status = "pass"
    elif (
        not research_checks["safety_exact_coverage_and_finite"]
        or comparison["log_loss"]["ci95"][0] > 0.0
        or comparison["brier"]["ci95"][0] > 0.0
        or not research_checks["ece_at_most"]
        or not research_checks["calibration_slope_in_range"]
        or not research_checks["absolute_calibration_intercept_at_most"]
    ):
        research_status = "reject"
    else:
        research_status = "ambiguous"
    branch_plan = plan["branch_q_gate"]
    worst = result["worst_supported_source_deck_or_policy"]
    control = result["control_pass"]
    macro_source_available = math.isfinite(result["macro_source_log_loss_damage_vs_s0"])
    branch_checks = {
        "selected_research_v_gate": research_status == "pass",
        "macro_source_available": macro_source_available,
        "macro_source_nonworse": (
            macro_source_available
            and result["macro_source_log_loss_damage_vs_s0"]
            <= float(branch_plan["macro_source_log_loss_damage_vs_s0_at_most"])
        ),
        "worst_supported_slice_available": worst is not None,
        "worst_supported_slice_within_margin": (
            worst is not None
            and worst["log_loss_damage_vs_s0"]
            <= float(
                branch_plan[
                    "worst_supported_source_deck_or_policy_log_loss_damage_vs_s0_at_most"
                ]
            )
        ),
        "control_pass_coverage": (
            control["rows"] >= int(branch_plan["control_pass_minimum_eligible_rows"])
            and control["games"] >= int(branch_plan["control_pass_minimum_games"])
        ),
        "control_pass_within_margin": (
            control.get("log_loss_damage_vs_s0", float("inf"))
            <= float(branch_plan["control_pass_log_loss_damage_vs_s0_at_most"])
        ),
    }
    if all(branch_checks.values()):
        branch_status = "pass"
        verdict = plan["verdicts"]["pass"]
    elif research_status == "reject":
        branch_status = "reject"
        verdict = plan["verdicts"]["reject"]
    elif (
        research_status != "pass"
        or not branch_checks["macro_source_available"]
        or not branch_checks["worst_supported_slice_available"]
        or not branch_checks["control_pass_coverage"]
    ):
        branch_status = "ambiguous"
        verdict = plan["verdicts"]["ambiguous"]
    else:
        branch_status = "reject"
        verdict = plan["verdicts"]["reject"]

    return {
        "selected_research_v_status": research_status,
        "selected_research_v_checks": research_checks,
        "branch_q_status": branch_status,
        "branch_q_checks": branch_checks,
        "verdict": verdict,
        "retuning_after_test": "forbidden",
    }


def _write_test_lock(lock_path: Path, payload: dict[str, Any]) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())


def run_sealed_test_score(
    *,
    cache_dir: Path,
    split_path: Path,
    prediction_dir: Path,
    out_dir: Path,
    command: list[str] | None = None,
) -> dict[str, Any]:
    cache_dir = cache_dir.resolve()
    split_path = split_path.resolve()
    prediction_dir = prediction_dir.resolve()
    out_dir = out_dir.resolve()
    if out_dir.exists():
        manifest = read_manifest(out_dir / "manifest.json")
        _verify_output_files(out_dir, manifest["outputs"]["files"])

        return {"action": "reused_verified_without_label_access", "manifest": manifest}
    preflight, verified, data = _preflight_context(
        cache_dir=cache_dir,
        split_path=split_path,
        prediction_dir=prediction_dir,
        out_dir=out_dir,
    )
    if preflight["lock_exists"]:
        raise RuntimeError("sealed test split was already opened; automatic retry is forbidden")
    lock_path = Path(preflight["lock_path"])
    temp_dir = out_dir.with_name(f".{out_dir.name}.running.{uuid.uuid4().hex}")
    temp_dir.mkdir(parents=True)
    preopen_path = temp_dir / "PREOPEN_RECEIPT.json"
    write_json(preopen_path, {
        "schema_version": SEALED_SCORE_SCHEMA_VERSION,
        "status": "output_staging_reserved_labels_still_sealed",
        **preflight,
    })
    lock_payload = {
        "schema_version": SEALED_SCORE_SCHEMA_VERSION,
        "status": "TEST_OPENED_BEFORE_LABEL_LOAD",
        "opened_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "split_manifest_sha256": preflight["split_manifest_sha256"],
        "prediction_manifest_sha256": preflight["prediction_manifest_sha256"],
        "scoring_plan_sha256": preflight["scoring_plan_sha256"],
        "scorer_source_sha256": _sha256(Path(__file__)),
        "intended_output": str(out_dir),
    }
    _write_test_lock(lock_path, lock_payload)
    lock_sha256 = _sha256(lock_path)
    try:
        arrays = verified["arrays"]
        row_id = arrays["row_id"].astype(np.int64, copy=False)
        eligible = arrays["is_target_eligible"].astype(np.bool_, copy=False)
        with np.load(cache_dir / "meta.npz", allow_pickle=False) as label_store:
            if "z" not in label_store.files:
                raise ValueError("sealed cache target array is missing")
            z = label_store["z"][row_id[eligible]].astype(np.float64, copy=False)
        if (
            z.shape != (int(eligible.sum()),)
            or not np.isfinite(z).all()
            or np.any((z < 0.0) | (z > 1.0))
        ):
            raise ValueError("sealed test targets are invalid")
        aggregate = score_aggregate_arrays(
            arrays,
            z,
            source_ids=data.source_ids,
            policy_vocab=data.policy_vocab,
            deck_vocab=data.deck_vocab,
            plan=verified["plan"],
            inference=verified["result"]["final_inference"],
        )
        result = {
            "schema_version": SEALED_SCORE_SCHEMA_VERSION,
            "status": "completed_one_shot_sealed_test",
            "bindings": verified["plan"]["bindings"],
            "test_open_lock_path": str(lock_path),
            "test_open_lock_sha256": lock_sha256,
            "prediction_manifest_sha256": preflight["prediction_manifest_sha256"],
            "scoring_plan_sha256": preflight["scoring_plan_sha256"],
            "eligible_targets": int(len(z)),
            "games": int(len(np.unique(arrays["game_gidx"][eligible]))),
            "aggregate": aggregate,
            "labels_or_row_losses_published": False,
            "test_targets_consumed": True,
            "protected_runtime_surfaces_changed": False,
            "q_work_started": False,
            "ladder_improvement_established": False,
        }
        config = {
            "schema_version": SEALED_SCORE_SCHEMA_VERSION,
            "cache_path": str(cache_dir),
            "cache_manifest_sha256": _sha256(cache_dir / "cache_manifest.json"),
            "split_path": str(split_path),
            "split_manifest_sha256": _sha256(split_path),
            "prediction_path": str(prediction_dir),
            "prediction_manifest_sha256": preflight["prediction_manifest_sha256"],
            "scoring_plan_sha256": preflight["scoring_plan_sha256"],
            "test_open_lock_sha256": lock_sha256,
            "code_sha256": {
                "sealed_test_scoring": _sha256(Path(__file__)),
                "sealed_test_prediction": _sha256(Path(value_prediction.__file__)),
                "data": _sha256(Path(value_data.__file__)),
                "metrics": _sha256(Path(value_metrics.__file__)),
            },
            "environment": {
                "python": platform.python_version(),
                "python_executable": sys.executable,
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
            "test_targets_consumed": True,
        }
        config["sha256"] = _stable_hash(config)
        config_path = temp_dir / "config.json"
        write_json(config_path, config)
        result_path = temp_dir / "result.json"
        write_json(result_path, value_mc_screen._json_safe(result))
        files = _file_receipts(temp_dir, [preopen_path, config_path, result_path])
        manifest = build_manifest(
            run_id=out_dir.name,
            kind="value_sealed_test_score",
            command=command or [],
            inputs={"config": config, "config_sha256": config["sha256"]},
            outputs={"files": files},
            summary={
                "status": "completed_one_shot_sealed_test",
                "eligible_targets": int(len(z)),
                "games": result["games"],
                "selected_research_v_status": aggregate["gate"]["selected_research_v_status"],
                "branch_q_status": aggregate["gate"]["branch_q_status"],
                "verdict": aggregate["gate"]["verdict"],
                "test_open_lock_sha256": lock_sha256,
                "labels_or_row_losses_published": False,
                "test_targets_consumed": True,
                "protected_runtime_surfaces_changed": False,
                "q_work_started": False,
                "ladder_improvement_established": False,
            },
        )
        write_manifest(temp_dir / "manifest.json", manifest)
        os.replace(temp_dir, out_dir)
    except BaseException as error:
        failure = {
            "schema_version": SEALED_SCORE_SCHEMA_VERSION,
            "status": "FAILED_AFTER_TEST_OPEN",
            "error_type": type(error).__name__,
            "error_message": str(error)[:1_000],
            "test_open_lock_path": str(lock_path),
            "test_open_lock_sha256": lock_sha256,
            "prediction_manifest_sha256": preflight["prediction_manifest_sha256"],
            "scoring_plan_sha256": preflight["scoring_plan_sha256"],
            "labels_or_row_losses_published": False,
            "test_targets_consumed": True,
            "retry_forbidden": True,
        }
        failure_path = temp_dir / "FAILED_AFTER_TEST_OPEN.json"
        try:
            write_json(failure_path, failure)
            files = _file_receipts(temp_dir, [preopen_path, failure_path])
            failed_manifest = build_manifest(
                run_id=out_dir.name,
                kind="value_sealed_test_score",
                command=command or [],
                status="failed",
                inputs={
                    "prediction_manifest_sha256": preflight["prediction_manifest_sha256"],
                    "scoring_plan_sha256": preflight["scoring_plan_sha256"],
                },
                outputs={"files": files},
                summary=failure,
            )
            write_manifest(temp_dir / "manifest.json", failed_manifest)
            os.replace(temp_dir, out_dir)
        except BaseException as receipt_error:
            raise RuntimeError(
                "sealed test failed after opening and the terminal failure receipt could not publish: "
                f"{type(receipt_error).__name__}"
            ) from error
        raise RuntimeError(
            f"sealed test failed after opening; durable receipt: {out_dir}"
        ) from error

    return {"action": "completed", "manifest": manifest, "result": result}
