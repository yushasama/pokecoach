"""Label-blind prediction stage for the one-shot sealed value-model test."""

from __future__ import annotations

import json
import os
import platform
import sys
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import deepmew.value_lab.calibration as value_calibration
import deepmew.value_lab.calibration_selection as value_calibration_selection
import deepmew.value_lab.control_pass_mask as value_control_pass_mask
import deepmew.value_lab.crossfit as value_crossfit
import deepmew.value_lab.data as value_data
import deepmew.value_lab.mc_screen as value_mc_screen
import deepmew.value_lab.metrics as value_metrics
import deepmew.value_lab.models as value_models
from deepmew.artifacts.manifest import build_manifest, read_manifest, write_manifest
from deepmew.artifacts.writers import write_json
from deepmew.value_lab.calibration import apply_calibrator
from deepmew.value_lab.calibration_selection import (
    load_calibrator_bundle,
    verify_crossfit_artifact,
)
from deepmew.value_lab.control_pass_mask import CONTROL_PASS_MASK_SCHEMA_VERSION
from deepmew.value_lab.crossfit import (
    CROSSFIT_FOLDS,
    FOLD_UNIT_SCHEMA_VERSION,
    SNAPSHOT_SPEC,
    _atomic_gzip_json,
    _atomic_npz,
    _file_receipts,
    _sha256,
    _stable_hash,
    _verify_output_files,
    canonical_game_indices,
)
from deepmew.value_lab.data import LabData
from deepmew.value_lab.mc_screen import _predict, build_snapshot_model
from deepmew.value_lab.models import parameter_count


SEALED_PREDICTION_SCHEMA_VERSION = "deepmew.value_sealed_test_prediction.v1"
SCORING_PLAN_SCHEMA_VERSION = "deepmew.value_sealed_test_scoring_plan.v1"
SCORING_BOOTSTRAP_REPLICATES = 2_000
SCORING_BOOTSTRAP_SEED = 915_004
SCREENING_CHECKPOINT_SCHEMA_VERSION = "deepmew.value_mc_temporal_checkpoint.v1"
DEFAULT_BATCH_SIZE = 4_096


def canonical_test_lock_path(split_manifest_sha256: str) -> Path:
    if (
        not isinstance(split_manifest_sha256, str)
        or len(split_manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in split_manifest_sha256.lower())
    ):
        raise ValueError("split manifest SHA-256 is invalid")
    repo_root = Path(__file__).resolve().parents[2]

    return (
        repo_root
        / ".research"
        / "value_q_experiment"
        / "sealed_test_locks"
        / f"{split_manifest_sha256.lower()}.TEST_OPENED.json"
    )


def _current_calibration_code_hashes() -> dict[str, str]:
    return {
        "calibration_selection": _sha256(Path(value_calibration_selection.__file__)),
        "calibration": _sha256(Path(value_calibration.__file__)),
        "crossfit": _sha256(Path(value_crossfit.__file__)),
        "data": _sha256(Path(value_data.__file__)),
        "feature_confirmation": value_calibration_selection._sha256(
            Path(value_calibration_selection.value_feature_confirmation.__file__)
        ),
        "metrics": value_calibration_selection._sha256(
            Path(value_calibration_selection.value_metrics.__file__)
        ),
    }


def verify_calibration_artifact(
    calibration_dir: Path,
    *,
    cache_manifest_sha256: str,
    split_manifest_sha256: str,
    crossfit_manifest_sha256: str,
) -> dict[str, Any]:
    calibration_dir = calibration_dir.resolve()
    manifest = read_manifest(calibration_dir / "manifest.json")
    config = manifest.get("inputs", {}).get("config")
    summary = manifest.get("summary", {})
    if (
        manifest.get("kind") != "value_grouped_calibration_selection"
        or manifest.get("status") != "completed"
        or summary.get("status") != "completed_calibrator_frozen"
        or not isinstance(config, dict)
    ):
        raise ValueError("calibration root manifest contract mismatch")
    declared_config_sha256 = config.get("sha256")
    config_payload = {key: value for key, value in config.items() if key != "sha256"}
    if (
        not isinstance(declared_config_sha256, str)
        or _stable_hash(config_payload) != declared_config_sha256
        or manifest["inputs"].get("config_sha256") != declared_config_sha256
        or config.get("cache_manifest_sha256") != cache_manifest_sha256
        or config.get("split_manifest_sha256") != split_manifest_sha256
        or config.get("crossfit_manifest_sha256") != crossfit_manifest_sha256
        or config.get("code_sha256") != _current_calibration_code_hashes()
        or config.get("test_targets_consumed") is not False
        or summary.get("test_targets_consumed") is not False
    ):
        raise ValueError("calibration lineage or code closure mismatch")
    files = manifest.get("outputs", {}).get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("calibration output inventory is missing")
    _verify_output_files(calibration_dir, files)
    result = json.loads((calibration_dir / "result.json").read_text(encoding="utf-8"))
    receipt = result.get("final_calibrator")
    if (
        result.get("status") != "completed_calibrator_frozen"
        or result.get("selected_method") != summary.get("selected_method")
        or result.get("test_targets_consumed") is not False
        or not isinstance(receipt, dict)
        or receipt.get("bundle_sha256") != summary.get("final_calibrator_sha256")
    ):
        raise ValueError("calibration result contract mismatch")
    calibrator = load_calibrator_bundle(calibration_dir, receipt)

    return {
        "manifest": manifest,
        "result": result,
        "calibrator": calibrator,
        "calibrator_receipt": receipt,
    }


def verify_control_pass_artifact(
    control_pass_dir: Path,
    *,
    cache_manifest_sha256: str,
    split_manifest_sha256: str,
    total_rows: int,
) -> dict[str, Any]:
    control_pass_dir = control_pass_dir.resolve()
    manifest = read_manifest(control_pass_dir / "manifest.json")
    config = manifest.get("inputs", {}).get("config")
    summary = manifest.get("summary", {})
    if (
        manifest.get("kind") != "value_control_pass_mask"
        or manifest.get("status") != "completed"
        or summary.get("status") != "completed_tri_state"
        or not isinstance(config, dict)
    ):
        raise ValueError("control-pass manifest contract mismatch")
    declared_config_sha256 = config.get("sha256")
    config_payload = {key: value for key, value in config.items() if key != "sha256"}
    expected_code = {
        "control_pass_mask": _sha256(Path(value_control_pass_mask.__file__)),
        "data": _sha256(Path(value_data.__file__)),
    }
    if (
        config.get("schema_version") != CONTROL_PASS_MASK_SCHEMA_VERSION
        or _stable_hash(config_payload) != declared_config_sha256
        or manifest["inputs"].get("config_sha256") != declared_config_sha256
        or config.get("cache_manifest_sha256") != cache_manifest_sha256
        or config.get("split_manifest_sha256") != split_manifest_sha256
        or config.get("code_sha256") != expected_code
        or summary.get("rows") != total_rows
        or summary.get("test_targets_consumed") is not False
    ):
        raise ValueError("control-pass lineage or code closure mismatch")
    _verify_output_files(control_pass_dir, manifest["outputs"]["files"])
    with np.load(control_pass_dir / "control_pass_mask.npz", allow_pickle=False) as arrays:
        row_id = arrays["row_id"].astype(np.int64, copy=False)
        known = arrays["is_control_pass_known"].astype(np.bool_, copy=False)
        passed = arrays["control_passed"].astype(np.bool_, copy=False)
    if (
        not np.array_equal(row_id, np.arange(total_rows, dtype=np.int64))
        or known.shape != (total_rows,)
        or passed.shape != (total_rows,)
        or np.any(passed & ~known)
        or int(known.sum()) != summary.get("known_rows")
        or int((known & passed).sum()) != summary.get("passed_rows")
    ):
        raise ValueError("control-pass tri-state arrays are invalid")

    return {"manifest": manifest, "known": known, "passed": passed}


def _load_snapshot_checkpoint(
    checkpoint_path: Path,
    *,
    expected_sha256: str,
    expected_root_config_sha256: str | None,
    expected_schema_version: str,
    expected_role: str | None,
    expected_spec: dict[str, Any] | None,
) -> torch.nn.Module:
    if _sha256(checkpoint_path) != expected_sha256:
        raise ValueError(f"value checkpoint hash mismatch: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("schema_version") != expected_schema_version
        or checkpoint.get("model_config", {}).get("input_dim") != 49
        or checkpoint.get("model_config", {}).get("hidden_dims") != [256, 128]
        or float(checkpoint.get("model_config", {}).get("dropout", -1.0)) != 0.0
        or checkpoint.get("parameter_count") != 45_825
    ):
        raise ValueError(f"value checkpoint architecture mismatch: {checkpoint_path}")
    if expected_root_config_sha256 is not None and (
        checkpoint.get("root_config_sha256") != expected_root_config_sha256
    ):
        raise ValueError(f"value checkpoint root lineage mismatch: {checkpoint_path}")
    if expected_role is not None and checkpoint.get("role") != expected_role:
        raise ValueError(f"value checkpoint role mismatch: {checkpoint_path}")
    if expected_spec is not None and checkpoint.get("spec") != expected_spec:
        raise ValueError(f"value checkpoint feature/spec mismatch: {checkpoint_path}")
    model = build_snapshot_model(SNAPSHOT_SPEC)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if parameter_count(model) != 45_825:
        raise ValueError("loaded value model parameter count drifted")

    return model


def verify_screening_artifact(
    screening_dir: Path,
    *,
    cache_manifest_sha256: str,
    split_manifest_sha256: str,
) -> dict[str, Any]:
    screening_dir = screening_dir.resolve()
    manifest = read_manifest(screening_dir / "manifest.json")
    config = manifest.get("inputs", {}).get("config", {})
    summary = manifest.get("summary", {})
    if (
        manifest.get("kind") != "value_mc_temporal_screen"
        or manifest.get("status") != "completed"
        or summary.get("status") != "screening_only"
        or config.get("cache_manifest_sha256") != cache_manifest_sha256
        or config.get("split_manifest_sha256") != split_manifest_sha256
        or summary.get("calibration_targets_consumed") is not False
        or summary.get("test_targets_consumed") is not False
    ):
        raise ValueError("Phase 4 screening artifact contract mismatch")
    _verify_output_files(screening_dir, manifest["outputs"]["files"])
    checkpoints = []
    for seed in (915_001, 915_002):
        path = screening_dir / f"seed_{seed}" / "v0_snapshot_s1.pt"
        receipt = next(
            (
                item for item in manifest["outputs"]["files"]
                if item["path"] == f"seed_{seed}/v0_snapshot_s1.pt"
            ),
            None,
        )
        if receipt is None:
            raise ValueError(f"Phase 4 screening checkpoint is missing for seed {seed}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (
            checkpoint.get("schema_version") != SCREENING_CHECKPOINT_SCHEMA_VERSION
            or checkpoint.get("seed") != seed
            or checkpoint.get("cache_manifest_sha256") != cache_manifest_sha256
            or checkpoint.get("split_manifest_sha256") != split_manifest_sha256
            or checkpoint.get("spec", {}).get("name") != "v0_snapshot_s1"
            or checkpoint.get("spec", {}).get("feature_start") != 0
            or checkpoint.get("spec", {}).get("feature_stop") != 49
        ):
            raise ValueError(f"Phase 4 screening checkpoint lineage mismatch for seed {seed}")
        model = _load_snapshot_checkpoint(
            path,
            expected_sha256=receipt["sha256"],
            expected_root_config_sha256=None,
            expected_schema_version=SCREENING_CHECKPOINT_SCHEMA_VERSION,
            expected_role=None,
            expected_spec=None,
        )
        checkpoints.append({
            "seed": seed,
            "path": path,
            "sha256": receipt["sha256"],
            "model": model,
        })

    return {"manifest": manifest, "checkpoints": checkpoints}


def _outer_train_base_rate(final_manifest: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    stream = final_manifest["summary"]["train_row_stream"]
    counts = stream["rows_by_outcome"]
    presentations = int(stream["presentations"])
    numerator = sum(float(outcome) * int(count) for outcome, count in counts.items())
    base_rate = numerator / presentations

    return base_rate, {
        "name": "outer_train_actor_outcome_rate",
        "probability": base_rate,
        "presentations": presentations,
        "rows_by_outcome": counts,
        "row_stream_sha256": stream["sha256"],
    }


def _test_index(data: LabData) -> dict[str, np.ndarray]:
    if data.has_labels:
        raise ValueError("sealed prediction requires label-blind LabData")
    split_index = data.split_names.index("test")
    row_id = np.flatnonzero(data.meta["split_idx"] == split_index).astype(np.int64, copy=False)
    starts = data.blocks["row_start"].astype(np.int64, copy=False)
    lengths = data.blocks["row_len"].astype(np.int64, copy=False)
    block_id = np.searchsorted(starts, row_id, side="right") - 1
    if np.any(block_id < 0) or np.any(row_id >= starts[block_id] + lengths[block_id]):
        raise ValueError("test rows do not map to canonical actor blocks")
    block_game, game_ids = canonical_game_indices(data)

    return {
        "row_id": row_id,
        "block_id": block_id.astype(np.int32),
        "game_gidx": block_game[block_id].astype(np.int32, copy=False),
        "source_idx": data.meta["source_idx"][row_id].astype(np.int32, copy=False),
        "policy_gidx": data.blocks["policy_gidx"][block_id].astype(np.int32, copy=False),
        "opp_policy_gidx": data.blocks["opp_policy_gidx"][block_id].astype(np.int32, copy=False),
        "deck_gidx": data.blocks["deck_gidx"][block_id].astype(np.int32, copy=False),
        "opp_deck_gidx": data.blocks["opp_deck_gidx"][block_id].astype(np.int32, copy=False),
        "seat": data.meta["seat"][row_id].astype(np.int32, copy=False),
        "turn": data.meta["turn"][row_id].astype(np.int32, copy=False),
        "decision_index": data.meta["decision_index"][row_id].astype(np.int32, copy=False),
        "select_context": data.meta["select_context"][row_id].astype(np.int32, copy=False),
        "legal_count": data.meta["legal_count"][row_id].astype(np.int32, copy=False),
        "is_target_eligible": data.meta["is_target_eligible"][row_id].astype(np.bool_, copy=False),
        "game_ids": game_ids,
    }


def _scoring_plan(
    *,
    bindings: dict[str, Any],
    base_rate: dict[str, Any],
    control_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCORING_PLAN_SCHEMA_VERSION,
        "status": "frozen_before_test_labels",
        "bindings": bindings,
        "comparators": {
            "s0": base_rate,
            "phase4_screening": "mean_of_seed_915001_and_915002_raw_probabilities",
            "old_value_model": (
                "not_admissible_accepted_split_overlap_recorded_in_live_ledger"
            ),
        },
        "metrics": ["log_loss", "brier", "ece_15_bin", "auc_secondary", "slope", "intercept"],
        "bootstrap": {
            "unit": "complete_game",
            "replicates": SCORING_BOOTSTRAP_REPLICATES,
            "seed": SCORING_BOOTSTRAP_SEED,
        },
        "slice_support": {"minimum_eligible_rows": 5_000, "minimum_games": 50},
        "selected_research_v_gate": {
            "safety_exact_coverage_and_finite": True,
            "log_loss_vs_s0_ci95_upper_below_zero": True,
            "brier_vs_s0_ci95_upper_at_most_zero": True,
            "ece_at_most": 0.03,
            "calibration_slope_range": [0.8, 1.2],
            "absolute_calibration_intercept_at_most": 0.10,
            "batch_latency_microseconds_per_row_at_most": 5.0,
        },
        "branch_q_gate": {
            "requires_selected_research_v_gate": True,
            "macro_source_log_loss_damage_vs_s0_at_most": 0.0,
            "worst_supported_source_deck_or_policy_log_loss_damage_vs_s0_at_most": 0.01,
            "control_pass_minimum_eligible_rows": 5_000,
            "control_pass_minimum_games": 50,
            "control_pass_log_loss_damage_vs_s0_at_most": 0.01,
            "insufficient_control_pass_coverage_verdict": "AMBIGUOUS_NO_Q",
        },
        "control_pass_pretest_coverage": control_summary,
        "canonical_test_lock_path": str(
            canonical_test_lock_path(bindings["split_manifest_sha256"])
        ),
        "verdicts": {
            "pass": "PASS_TO_BRANCH_Q",
            "ambiguous": "AMBIGUOUS_NO_Q",
            "reject": "REJECT_NO_Q",
        },
        "one_shot_policy": "no retuning or second candidate after sealed labels are opened",
        "test_targets_consumed": False,
    }


def _write_game_vocab(path: Path, game_ids: list[str]) -> None:
    _atomic_gzip_json(path, {"game_ids": game_ids})


def run_sealed_test_prediction(
    *,
    cache_dir: Path,
    split_path: Path,
    crossfit_dir: Path,
    calibration_dir: Path,
    control_pass_dir: Path,
    screening_dir: Path,
    out_dir: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    command: list[str] | None = None,
) -> dict[str, Any]:
    cache_dir = cache_dir.resolve()
    split_path = split_path.resolve()
    crossfit_dir = crossfit_dir.resolve()
    calibration_dir = calibration_dir.resolve()
    control_pass_dir = control_pass_dir.resolve()
    screening_dir = screening_dir.resolve()
    out_dir = out_dir.resolve()
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    cache_manifest_sha256 = _sha256(cache_dir / "cache_manifest.json")
    split_manifest_sha256 = _sha256(split_path)
    crossfit_manifest_sha256 = _sha256(crossfit_dir / "manifest.json")
    crossfit = verify_crossfit_artifact(
        crossfit_dir,
        cache_manifest_sha256=cache_manifest_sha256,
        split_manifest_sha256=split_manifest_sha256,
    )
    calibration_manifest_sha256 = _sha256(calibration_dir / "manifest.json")
    calibration = verify_calibration_artifact(
        calibration_dir,
        cache_manifest_sha256=cache_manifest_sha256,
        split_manifest_sha256=split_manifest_sha256,
        crossfit_manifest_sha256=crossfit_manifest_sha256,
    )
    data = LabData(
        str(cache_dir),
        split_manifest_path=str(split_path),
        load_labels=False,
    )
    control_pass = verify_control_pass_artifact(
        control_pass_dir,
        cache_manifest_sha256=cache_manifest_sha256,
        split_manifest_sha256=split_manifest_sha256,
        total_rows=int(data.manifest["rows"]),
    )
    screening = verify_screening_artifact(
        screening_dir,
        cache_manifest_sha256=cache_manifest_sha256,
        split_manifest_sha256=split_manifest_sha256,
    )
    final_checkpoint_path = crossfit_dir / "full_train" / "value.pt"
    root_config_sha256 = crossfit["manifest"]["inputs"]["config_sha256"]
    final_checkpoint_sha256 = crossfit["manifest"]["summary"]["final_checkpoint_sha256"]
    final_model = _load_snapshot_checkpoint(
        final_checkpoint_path,
        expected_sha256=final_checkpoint_sha256,
        expected_root_config_sha256=root_config_sha256,
        expected_schema_version=FOLD_UNIT_SCHEMA_VERSION,
        expected_role="final_full_outer_train",
        expected_spec=asdict(SNAPSHOT_SPEC),
    )
    index = _test_index(data)
    row_id = index["row_id"]
    expected_context = int(data.manifest["context_rows_by_split"]["test"])
    expected_eligible = int(data.manifest["eligible_target_rows_by_split"]["test"])
    expected_excluded = int(data.manifest["excluded_target_rows_by_split"]["test"])
    test_games = np.unique(index["game_gidx"])
    if (
        len(row_id) != expected_context
        or int(index["is_target_eligible"].sum()) != expected_eligible
        or expected_context - expected_eligible != expected_excluded
        or len(test_games) != sum(
            value == "test"
            for value in json.loads(split_path.read_text(encoding="utf-8"))["assignment"].values()
        )
    ):
        raise ValueError("sealed test coverage differs from the accepted split")
    raw_probability, final_inference = _predict(
        data,
        final_model,
        SNAPSHOT_SPEC,
        row_id,
        batch_size=batch_size,
    )
    calibrated_probability = apply_calibrator(raw_probability, calibration["calibrator"])
    screening_probabilities = []
    screening_inference = {}
    for checkpoint in screening["checkpoints"]:
        predicted, inference = _predict(
            data,
            checkpoint["model"],
            SNAPSHOT_SPEC,
            row_id,
            batch_size=batch_size,
        )
        screening_probabilities.append(predicted)
        screening_inference[str(checkpoint["seed"])] = inference
    screening_mean = np.mean(np.stack(screening_probabilities), axis=0).astype(np.float32)
    probabilities = [raw_probability, calibrated_probability, *screening_probabilities, screening_mean]
    if any(
        not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0))
        for values in probabilities
    ):
        raise FloatingPointError("sealed test predictions are non-finite or out of range")
    test_known = control_pass["known"][row_id]
    test_passed = control_pass["passed"][row_id]
    eligible_known = index["is_target_eligible"] & test_known
    eligible_passed = eligible_known & test_passed
    base_rate_probability, base_rate = _outer_train_base_rate(crossfit["final_manifest"])
    scorer_source_path = Path(__file__).with_name("sealed_test_scoring.py")
    if not scorer_source_path.is_file():
        raise ValueError("sealed scorer source must exist before the scoring plan is frozen")
    bindings = {
        "cache_manifest_sha256": cache_manifest_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "crossfit_manifest_sha256": crossfit_manifest_sha256,
        "calibration_manifest_sha256": calibration_manifest_sha256,
        "control_pass_manifest_sha256": _sha256(control_pass_dir / "manifest.json"),
        "screening_manifest_sha256": _sha256(screening_dir / "manifest.json"),
        "final_checkpoint_sha256": final_checkpoint_sha256,
        "final_calibrator_sha256": calibration["calibrator_receipt"]["bundle_sha256"],
        "representation_manifest_sha256": crossfit["manifest"]["inputs"]["config"][
            "representation"
        ]["scientific_sha256"],
        "scoring_code_sha256": {
            "sealed_test_scoring": _sha256(scorer_source_path),
            "data": _sha256(Path(value_data.__file__)),
            "mc_screen": _sha256(Path(value_mc_screen.__file__)),
            "metrics": _sha256(Path(value_metrics.__file__)),
        },
        "information_mode": "actor_visible",
        "target": "mc_actor_relative_terminal",
    }
    control_summary = {
        "test_context_known_rows": int(test_known.sum()),
        "test_context_passed_rows": int((test_known & test_passed).sum()),
        "test_eligible_known_rows": int(eligible_known.sum()),
        "test_eligible_passed_rows": int(eligible_passed.sum()),
        "status": "partial_tri_state_source_concentrated",
    }
    scoring_plan = _scoring_plan(
        bindings=bindings,
        base_rate=base_rate,
        control_summary=control_summary,
    )
    scoring_plan_sha256 = _stable_hash(scoring_plan)
    config = {
        "schema_version": SEALED_PREDICTION_SCHEMA_VERSION,
        "bindings": bindings,
        "paths": {
            "cache": str(cache_dir),
            "split": str(split_path),
            "crossfit": str(crossfit_dir),
            "calibration": str(calibration_dir),
            "control_pass": str(control_pass_dir),
            "screening": str(screening_dir),
        },
        "batch_size": batch_size,
        "expected_counts": {
            "context_rows": expected_context,
            "eligible_targets": expected_eligible,
            "excluded_targets": expected_excluded,
            "games": int(len(test_games)),
        },
        "scoring_plan_sha256": scoring_plan_sha256,
        "code_sha256": {
            "sealed_test_prediction": _sha256(Path(__file__)),
            "calibration": _sha256(Path(value_calibration.__file__)),
            "calibration_selection": _sha256(Path(value_calibration_selection.__file__)),
            "control_pass_mask": _sha256(Path(value_control_pass_mask.__file__)),
            "crossfit": _sha256(Path(value_crossfit.__file__)),
            "data": _sha256(Path(value_data.__file__)),
            "mc_screen": _sha256(Path(value_mc_screen.__file__)),
            "models": _sha256(Path(value_models.__file__)),
        },
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "test_targets_consumed": False,
    }
    config["sha256"] = _stable_hash(config)
    if out_dir.exists():
        manifest = read_manifest(out_dir / "manifest.json")
        if manifest.get("inputs", {}).get("config_sha256") != config["sha256"]:
            raise ValueError("existing sealed prediction configuration differs")
        _verify_output_files(out_dir, manifest["outputs"]["files"])

        return {"action": "reused_verified", "manifest": manifest}
    temp_dir = out_dir.with_name(f".{out_dir.name}.running.{uuid.uuid4().hex}")
    temp_dir.mkdir(parents=True)
    output_paths = []
    config_path = temp_dir / "config.json"
    write_json(config_path, config)
    output_paths.append(config_path)
    plan_path = temp_dir / "SCORING_PLAN.json"
    write_json(plan_path, scoring_plan)
    output_paths.append(plan_path)
    prediction_path = temp_dir / "test_predictions.npz"
    _atomic_npz(
        prediction_path,
        **{key: value for key, value in index.items() if key != "game_ids"},
        control_pass_known=test_known,
        control_passed=test_passed,
        raw_probability=raw_probability.astype(np.float32),
        calibrated_probability=calibrated_probability.astype(np.float32),
        phase4_screen_seed_915001_probability=screening_probabilities[0].astype(np.float32),
        phase4_screen_seed_915002_probability=screening_probabilities[1].astype(np.float32),
        phase4_screen_ensemble_probability=screening_mean,
        s0_probability=np.full(len(row_id), base_rate_probability, dtype=np.float32),
    )
    output_paths.append(prediction_path)
    vocab_path = temp_dir / "game_vocab.json.gz"
    _write_game_vocab(vocab_path, index["game_ids"])
    output_paths.append(vocab_path)
    result = {
        "schema_version": SEALED_PREDICTION_SCHEMA_VERSION,
        "status": "completed_label_blind_predictions_frozen",
        "bindings": bindings,
        "counts": config["expected_counts"],
        "control_pass": control_summary,
        "final_inference": final_inference,
        "screening_inference": screening_inference,
        "screening_checkpoint_sha256": {
            str(item["seed"]): item["sha256"] for item in screening["checkpoints"]
        },
        "prediction_sha256": _sha256(prediction_path),
        "game_vocab_sha256": _sha256(vocab_path),
        "scoring_plan_sha256": _sha256(plan_path),
        "nonfinite_predictions": 0,
        "label_arrays_loaded": False,
        "test_targets_consumed": False,
        "protected_runtime_surfaces_changed": False,
    }
    result_path = temp_dir / "result.json"
    write_json(result_path, result)
    output_paths.append(result_path)
    files = _file_receipts(temp_dir, output_paths)
    manifest = build_manifest(
        run_id=out_dir.name,
        kind="value_sealed_test_prediction",
        command=command or [],
        inputs={"config": config, "config_sha256": config["sha256"]},
        outputs={"files": files},
        summary={
            "status": "completed_label_blind_predictions_frozen",
            "context_rows": expected_context,
            "eligible_targets": expected_eligible,
            "excluded_targets": expected_excluded,
            "games": int(len(test_games)),
            "prediction_sha256": result["prediction_sha256"],
            "scoring_plan_sha256": result["scoring_plan_sha256"],
            "final_checkpoint_sha256": final_checkpoint_sha256,
            "final_calibrator_sha256": bindings["final_calibrator_sha256"],
            "nonfinite_predictions": 0,
            "label_arrays_loaded": False,
            "test_targets_consumed": False,
            "protected_runtime_surfaces_changed": False,
        },
    )
    write_manifest(temp_dir / "manifest.json", manifest)
    os.replace(temp_dir, out_dir)

    return {"action": "completed", "manifest": manifest, "result": result}
