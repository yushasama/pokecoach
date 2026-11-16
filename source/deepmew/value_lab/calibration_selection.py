"""Grouped calibration selection for the frozen cross-fitted snapshot value model."""

from __future__ import annotations

import json
import os
import platform
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np

import deepmew.value_lab.calibration as value_calibration
import deepmew.value_lab.crossfit as value_crossfit
import deepmew.value_lab.data as value_data
import deepmew.value_lab.feature_confirmation as value_feature_confirmation
import deepmew.value_lab.metrics as value_metrics
from deepmew.artifacts.manifest import build_manifest, read_manifest, write_manifest
from deepmew.artifacts.writers import write_json
from deepmew.value_lab.calibration import (
    CALIBRATOR_METHODS,
    apply_calibrator,
    fit_isotonic,
    fit_temperature,
    identity_calibrator,
)
from deepmew.value_lab.crossfit import (
    CROSSFIT_FOLDS,
    CROSSFIT_SCHEMA_VERSION,
    FOLD_UNIT_SCHEMA_VERSION,
    _atomic_npz,
    _file_receipts,
    _sha256,
    _stable_hash,
    _verify_output_files,
    derive_signature_fold_bits,
    source_stratified_complete_game_folds,
)
from deepmew.value_lab.data import LabData
from deepmew.value_lab.feature_confirmation import evaluate_confirmation_predictions
from deepmew.value_lab.mc_screen import canonical_game_indices, direct_comparison
from deepmew.value_lab.metrics import full_metrics, paired_game_cluster_bootstrap


CALIBRATION_SELECTION_SCHEMA_VERSION = "deepmew.value_grouped_calibration_selection.v1"
CALIBRATOR_BUNDLE_SCHEMA_VERSION = "deepmew.value_calibrator_bundle.v1"
CALIBRATION_FOLD_SALT = "deepmew.value_v0_calibration_selection.v1"
CALIBRATION_BOOTSTRAP_REPLICATES = 2_000
CALIBRATION_BOOTSTRAP_SEED = 915_003


def eligible_split_index(data: LabData, split: str) -> dict[str, np.ndarray]:
    if split not in data.split_names:
        raise ValueError(f"unknown split {split}")
    split_index = data.split_names.index(split)
    row_id = np.flatnonzero(
        (data.meta["split_idx"] == split_index) & data.meta["is_target_eligible"]
    ).astype(np.int64, copy=False)
    if len(row_id) == 0:
        raise ValueError(f"split {split} has no eligible rows")
    starts = data.blocks["row_start"].astype(np.int64, copy=False)
    lengths = data.blocks["row_len"].astype(np.int64, copy=False)
    block_id = np.searchsorted(starts, row_id, side="right") - 1
    if np.any(block_id < 0) or np.any(row_id >= starts[block_id] + lengths[block_id]):
        raise ValueError(f"split {split} rows do not map to actor blocks")
    block_game, _ = canonical_game_indices(data)

    return {
        "row_id": row_id,
        "block_id": block_id.astype(np.int32),
        "game_gidx": block_game[block_id].astype(np.int32),
        "z": data.meta["z"][row_id].astype(np.float32, copy=False),
        "source_idx": data.meta["source_idx"][row_id].astype(np.int32, copy=False),
        "seat": data.meta["seat"][row_id].astype(np.int32, copy=False),
        "turn": data.meta["turn"][row_id].astype(np.int32, copy=False),
        "legal_count": data.meta["legal_count"][row_id].astype(np.int32, copy=False),
        "select_context": data.meta["select_context"][row_id].astype(np.int32, copy=False),
    }


def _fit_calibrator(method: str, p: np.ndarray, z: np.ndarray) -> dict[str, Any]:
    if method == "identity":
        return identity_calibrator()
    if method == "temperature":
        return fit_temperature(p, z)
    if method == "isotonic":
        return fit_isotonic(p, z)
    raise ValueError(f"unknown calibration method {method}")


def grouped_calibration_selection(
    p: np.ndarray,
    z: np.ndarray,
    game_gidx: np.ndarray,
    fold_id: np.ndarray,
    signature_fold_bits: np.ndarray,
    *,
    folds: int = CROSSFIT_FOLDS,
    bootstrap_replicates: int = CALIBRATION_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = CALIBRATION_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    p = np.asarray(p, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    game_gidx = np.asarray(game_gidx, dtype=np.int32)
    fold_id = np.asarray(fold_id, dtype=np.int8)
    signature_fold_bits = np.asarray(signature_fold_bits, dtype=np.uint8)
    if folds != CROSSFIT_FOLDS:
        raise ValueError(f"calibration selection requires exactly {CROSSFIT_FOLDS} folds")
    if (
        p.ndim != 1
        or z.shape != p.shape
        or game_gidx.shape != p.shape
        or fold_id.shape != p.shape
        or signature_fold_bits.shape != p.shape
        or not np.isfinite(p).all()
        or np.any((p < 0.0) | (p > 1.0))
        or not np.isfinite(z).all()
        or np.any((z < 0.0) | (z > 1.0))
        or np.any(fold_id < 0)
        or np.any(fold_id >= folds)
        or np.any(game_gidx < 0)
    ):
        raise ValueError("grouped calibration arrays are invalid")
    order = np.argsort(game_gidx, kind="stable")
    ordered_games = game_gidx[order]
    starts = np.flatnonzero(np.r_[True, ordered_games[1:] != ordered_games[:-1]])
    ordered_folds = fold_id[order]
    if np.any(
        np.minimum.reduceat(ordered_folds, starts)
        != np.maximum.reduceat(ordered_folds, starts)
    ):
        raise ValueError("calibration folds split at least one complete game")
    own_fold_bits = np.left_shift(np.uint8(1), fold_id.astype(np.uint8))
    if np.any((signature_fold_bits & own_fold_bits) == 0):
        raise ValueError("calibration signature bits omit the row's assigned fold")
    oof_predictions = {
        method: np.full(len(p), np.nan, dtype=np.float64)
        for method in CALIBRATOR_METHODS
    }
    fold_results = []
    for fold in range(folds):
        holdout = fold_id == fold
        train = (fold_id != fold) & ((signature_fold_bits & (1 << fold)) == 0)
        if not holdout.any() or not train.any():
            raise ValueError(f"calibration fold {fold} is empty")
        method_results = {}
        for method in CALIBRATOR_METHODS:
            calibrator = _fit_calibrator(method, p[train], z[train])
            predicted = apply_calibrator(p[holdout], calibrator)
            oof_predictions[method][holdout] = predicted
            method_results[method] = {
                "train_rows": int(train.sum()),
                "holdout_rows": int(holdout.sum()),
                "log_loss": value_metrics.log_loss(predicted, z[holdout]),
                "brier": value_metrics.brier(predicted, z[holdout]),
                "fit_summary": {
                    key: value
                    for key, value in calibrator.items()
                    if key not in {"thresholds", "values"}
                },
            }
        fold_results.append({"fold": fold, "methods": method_results})
    if any(not np.isfinite(values).all() for values in oof_predictions.values()):
        raise FloatingPointError("grouped calibration left non-finite OOF probabilities")
    metrics = {
        method: full_metrics(predicted, z)
        for method, predicted in oof_predictions.items()
    }
    versus_identity = {
        method: {
            metric: paired_game_cluster_bootstrap(
                oof_predictions[method],
                oof_predictions["identity"],
                z,
                game_gidx,
                metric=metric,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            )
            for metric in ("log_loss", "brier")
        }
        for method in ("temperature", "isotonic")
    }
    eligibility = {}
    for method in ("temperature", "isotonic"):
        improved_folds = sum(
            fold["methods"][method]["log_loss"]
            < fold["methods"]["identity"]["log_loss"]
            for fold in fold_results
        )
        eligibility[method] = {
            "log_loss_ci_upper_below_zero": versus_identity[method]["log_loss"]["ci95"][1] < 0.0,
            "brier_nonworse": versus_identity[method]["brier"]["delta_candidate_minus_reference"] <= 0.0,
            "improved_at_least_three_folds": improved_folds >= 3,
            "improved_folds": improved_folds,
        }
        eligibility[method]["eligible"] = all(
            value for key, value in eligibility[method].items()
            if key != "improved_folds"
        )
    eligible = [method for method in ("temperature", "isotonic") if eligibility[method]["eligible"]]
    method = "identity"
    head_to_head = None
    if eligible == ["temperature"]:
        method = "temperature"
    elif eligible == ["isotonic"]:
        method = "isotonic"
    elif len(eligible) == 2:
        head_to_head = {
            metric: paired_game_cluster_bootstrap(
                oof_predictions["isotonic"],
                oof_predictions["temperature"],
                z,
                game_gidx,
                metric=metric,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            )
            for metric in ("log_loss", "brier")
        }
        if (
            head_to_head["log_loss"]["ci95"][1] < 0.0
            and head_to_head["brier"]["delta_candidate_minus_reference"] <= 0.0
        ):
            method = "isotonic"
        else:
            method = "temperature"

    return {
        "selected_method": method,
        "folds": fold_results,
        "metrics": metrics,
        "versus_identity": versus_identity,
        "eligibility": eligibility,
        "isotonic_vs_temperature": head_to_head,
        "oof_predictions": oof_predictions,
        "selection_rule": (
            "nonidentity requires grouped-OOF log-loss CI upper<0, Brier nonworse, "
            "and >=3/5 improved folds; temperature wins unless isotonic credibly beats it"
        ),
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
    }


def write_calibrator_bundle(
    root: Path,
    name: str,
    calibrator: dict[str, Any],
) -> tuple[dict[str, Any], list[Path]]:
    descriptor = {
        key: value for key, value in calibrator.items()
        if key not in {"thresholds", "values"}
    }
    descriptor.update({
        "bundle_schema_version": CALIBRATOR_BUNDLE_SCHEMA_VERSION,
        "name": name,
    })
    paths = []
    if calibrator["method"] == "isotonic":
        arrays_path = root / f"{name}.arrays.npz"
        _atomic_npz(
            arrays_path,
            thresholds=np.asarray(calibrator["thresholds"], dtype=np.float64),
            values=np.asarray(calibrator["values"], dtype=np.float64),
        )
        descriptor["arrays_path"] = arrays_path.name
        descriptor["arrays_sha256"] = _sha256(arrays_path)
        paths.append(arrays_path)
    descriptor_path = root / f"{name}.json"
    write_json(descriptor_path, descriptor)
    paths.append(descriptor_path)
    bundle_hash = _stable_hash({
        "descriptor_sha256": _sha256(descriptor_path),
        "arrays_sha256": descriptor.get("arrays_sha256"),
    })

    return {
        "name": name,
        "method": calibrator["method"],
        "descriptor_path": descriptor_path.name,
        "descriptor_sha256": _sha256(descriptor_path),
        "arrays_path": descriptor.get("arrays_path"),
        "arrays_sha256": descriptor.get("arrays_sha256"),
        "bundle_sha256": bundle_hash,
    }, paths


def load_calibrator_bundle(root: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    required_receipt = {
        "name",
        "method",
        "descriptor_path",
        "descriptor_sha256",
        "arrays_path",
        "arrays_sha256",
        "bundle_sha256",
    }
    if set(receipt) != required_receipt:
        raise ValueError("calibrator receipt inventory is invalid")
    descriptor_path = root / receipt["descriptor_path"]
    descriptor_sha256 = _sha256(descriptor_path)
    if descriptor_sha256 != receipt["descriptor_sha256"]:
        raise ValueError("calibrator descriptor hash mismatch")
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    if (
        descriptor.get("bundle_schema_version") != CALIBRATOR_BUNDLE_SCHEMA_VERSION
        or descriptor.get("name") != receipt["name"]
        or descriptor.get("method") != receipt["method"]
    ):
        raise ValueError("calibrator descriptor identity mismatch")
    calibrator = {
        key: value for key, value in descriptor.items()
        if key not in {"bundle_schema_version", "name", "arrays_path", "arrays_sha256"}
    }
    if descriptor["method"] == "isotonic":
        if (
            descriptor.get("arrays_path") != receipt["arrays_path"]
            or descriptor.get("arrays_sha256") != receipt["arrays_sha256"]
        ):
            raise ValueError("calibrator array receipt mismatch")
        arrays_path = root / descriptor["arrays_path"]
        arrays_sha256 = _sha256(arrays_path)
        if arrays_sha256 != descriptor["arrays_sha256"]:
            raise ValueError("calibrator array hash mismatch")
        with np.load(arrays_path, allow_pickle=False) as arrays:
            calibrator["thresholds"] = arrays["thresholds"].tolist()
            calibrator["values"] = arrays["values"].tolist()
    else:
        arrays_sha256 = None
        if (
            receipt["arrays_path"] is not None
            or receipt["arrays_sha256"] is not None
            or "arrays_path" in descriptor
            or "arrays_sha256" in descriptor
        ):
            raise ValueError("non-isotonic calibrator unexpectedly binds arrays")
    bundle_hash = _stable_hash({
        "descriptor_sha256": descriptor_sha256,
        "arrays_sha256": arrays_sha256,
    })
    if bundle_hash != receipt["bundle_sha256"]:
        raise ValueError("calibrator bundle hash mismatch")

    return calibrator


def _load_prediction_rows(path: Path, expected_rows: np.ndarray) -> np.ndarray:
    with np.load(path, allow_pickle=False) as arrays:
        row_id = arrays["row_id"].astype(np.int64, copy=False)
        p = arrays["p"].astype(np.float64, copy=False)
    if not np.array_equal(row_id, expected_rows) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError(f"calibration prediction rows or probabilities are invalid: {path}")

    return p


def _verified_child_manifest(
    manifest_path: Path,
    *,
    expected_kind: str,
    expected_summary_status: str,
) -> dict[str, Any]:
    manifest = read_manifest(manifest_path)
    if (
        manifest.get("kind") != expected_kind
        or manifest.get("status") != "completed"
        or manifest.get("summary", {}).get("status") != expected_summary_status
    ):
        raise ValueError(f"cross-fit child manifest contract mismatch: {manifest_path}")
    files = manifest.get("outputs", {}).get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"cross-fit child output inventory is missing: {manifest_path}")
    _verify_output_files(manifest_path.parent, files)

    return manifest


def verify_crossfit_artifact(
    crossfit_dir: Path,
    *,
    cache_manifest_sha256: str,
    split_manifest_sha256: str,
) -> dict[str, Any]:
    crossfit_dir = crossfit_dir.resolve()
    manifest_path = crossfit_dir / "manifest.json"
    manifest = read_manifest(manifest_path)
    summary = manifest.get("summary", {})
    config = manifest.get("inputs", {}).get("config")
    if (
        manifest.get("kind") != "value_grouped_crossfit"
        or manifest.get("status") != "completed"
        or summary.get("status") != "completed_raw_uncalibrated"
        or not isinstance(config, dict)
    ):
        raise ValueError("cross-fit root manifest contract mismatch")
    declared_config_sha256 = config.get("sha256")
    config_payload = {key: value for key, value in config.items() if key != "sha256"}
    if (
        not isinstance(declared_config_sha256, str)
        or _stable_hash(config_payload) != declared_config_sha256
        or manifest["inputs"].get("config_sha256") != declared_config_sha256
        or config.get("schema_version") != CROSSFIT_SCHEMA_VERSION
        or config.get("cache_manifest_sha256") != cache_manifest_sha256
        or config.get("split_manifest_sha256") != split_manifest_sha256
        or config.get("folds") != CROSSFIT_FOLDS
        or config.get("information_mode") != "actor_visible"
        or config.get("target") != "mc_actor_relative_terminal"
        or config.get("calibration_targets_consumed") is not False
        or config.get("test_targets_consumed") is not False
        or config.get("representation", {}).get("indices") != list(range(49))
        or summary.get("folds") != CROSSFIT_FOLDS
        or summary.get("calibration_targets_consumed") is not False
        or summary.get("test_targets_consumed") is not False
        or summary.get("effective_forbidden_overlap") != 0
    ):
        raise ValueError("cross-fit lineage or frozen scientific contract mismatch")
    root_files = manifest.get("outputs", {}).get("files")
    if not isinstance(root_files, list) or not root_files:
        raise ValueError("cross-fit root output inventory is missing")
    _verify_output_files(crossfit_dir, root_files)
    representation = config["representation"]
    representation_path = Path(representation["path"])
    if _sha256(representation_path) != representation.get("file_sha256"):
        raise ValueError("cross-fit representation manifest hash mismatch")
    confirmation_path = Path(config["feature_confirmation_manifest_path"])
    if _sha256(confirmation_path) != config.get("feature_confirmation_manifest_sha256"):
        raise ValueError("cross-fit feature-confirmation manifest hash mismatch")
    fold_contract = _verified_child_manifest(
        crossfit_dir / "fold_contract" / "manifest.json",
        expected_kind="value_inner_fold_signature_audit",
        expected_summary_status="pass",
    )
    if (
        fold_contract.get("inputs", {}).get("config_sha256") != declared_config_sha256
        or fold_contract.get("inputs", {}).get("cache_manifest_sha256") != cache_manifest_sha256
        or fold_contract.get("summary", {}).get("assignment_sha256") != summary.get("assignment_sha256")
        or fold_contract.get("summary", {}).get("effective_forbidden_overlap") != 0
    ):
        raise ValueError("cross-fit fold-contract lineage mismatch")
    checkpoint_sha256 = summary.get("fold_checkpoint_sha256")
    if not isinstance(checkpoint_sha256, list) or len(checkpoint_sha256) != CROSSFIT_FOLDS:
        raise ValueError("cross-fit fold checkpoint inventory is invalid")
    fold_manifests = []
    for fold in range(CROSSFIT_FOLDS):
        fold_manifest = _verified_child_manifest(
            crossfit_dir / f"fold_{fold}" / "manifest.json",
            expected_kind="value_grouped_crossfit_fold",
            expected_summary_status="completed",
        )
        child_inputs = fold_manifest.get("inputs", {})
        child_config = child_inputs.get("config")
        child_summary = fold_manifest.get("summary", {})
        if (
            not isinstance(child_config, dict)
            or child_config.get("schema_version") != FOLD_UNIT_SCHEMA_VERSION
            or _stable_hash(child_config) != child_inputs.get("config_sha256")
            or child_config.get("root_config_sha256") != declared_config_sha256
            or child_config.get("fold") != fold
            or child_summary.get("fold") != fold
            or child_summary.get("checkpoint_sha256") != checkpoint_sha256[fold]
        ):
            raise ValueError(f"cross-fit fold {fold} lineage mismatch")
        fold_manifests.append(fold_manifest)
    final_manifest = _verified_child_manifest(
        crossfit_dir / "full_train" / "manifest.json",
        expected_kind="value_grouped_crossfit_final_model",
        expected_summary_status="completed",
    )
    final_inputs = final_manifest.get("inputs", {})
    final_config = final_inputs.get("config")
    if (
        not isinstance(final_config, dict)
        or final_config.get("schema_version") != FOLD_UNIT_SCHEMA_VERSION
        or _stable_hash(final_config) != final_inputs.get("config_sha256")
        or final_config.get("root_config_sha256") != declared_config_sha256
        or final_config.get("role") != "final_full_outer_train"
        or final_manifest.get("summary", {}).get("checkpoint_sha256")
        != summary.get("final_checkpoint_sha256")
    ):
        raise ValueError("cross-fit final-model lineage mismatch")

    return {
        "manifest": manifest,
        "fold_contract": fold_contract,
        "fold_manifests": fold_manifests,
        "final_manifest": final_manifest,
    }


def run_grouped_calibration(
    *,
    cache_dir: Path,
    split_path: Path,
    crossfit_dir: Path,
    out_dir: Path,
    bootstrap_replicates: int = CALIBRATION_BOOTSTRAP_REPLICATES,
    command: list[str] | None = None,
) -> dict[str, Any]:
    cache_dir = cache_dir.resolve()
    split_path = split_path.resolve()
    crossfit_dir = crossfit_dir.resolve()
    out_dir = out_dir.resolve()
    if bootstrap_replicates < 2:
        raise ValueError("calibration bootstrap requires at least two replicates")
    cache_manifest_sha256 = _sha256(cache_dir / "cache_manifest.json")
    split_manifest_sha256 = _sha256(split_path)
    crossfit_manifest_path = crossfit_dir / "manifest.json"
    verified_crossfit = verify_crossfit_artifact(
        crossfit_dir,
        cache_manifest_sha256=cache_manifest_sha256,
        split_manifest_sha256=split_manifest_sha256,
    )
    crossfit_manifest = verified_crossfit["manifest"]
    config = {
        "schema_version": CALIBRATION_SELECTION_SCHEMA_VERSION,
        "cache_path": str(cache_dir),
        "cache_manifest_sha256": cache_manifest_sha256,
        "split_path": str(split_path),
        "split_manifest_sha256": split_manifest_sha256,
        "crossfit_path": str(crossfit_dir),
        "crossfit_manifest_sha256": _sha256(crossfit_manifest_path),
        "final_checkpoint_sha256": crossfit_manifest["summary"]["final_checkpoint_sha256"],
        "fold_checkpoint_sha256": crossfit_manifest["summary"]["fold_checkpoint_sha256"],
        "folds": CROSSFIT_FOLDS,
        "fold_salt": CALIBRATION_FOLD_SALT,
        "methods": list(CALIBRATOR_METHODS),
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": CALIBRATION_BOOTSTRAP_SEED,
        "code_sha256": {
            "calibration_selection": _sha256(Path(__file__)),
            "calibration": _sha256(Path(value_calibration.__file__)),
            "crossfit": _sha256(Path(value_crossfit.__file__)),
            "data": _sha256(Path(value_data.__file__)),
            "feature_confirmation": _sha256(Path(value_feature_confirmation.__file__)),
            "metrics": _sha256(Path(value_metrics.__file__)),
        },
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "test_targets_consumed": False,
    }
    config["sha256"] = _stable_hash(config)
    if out_dir.exists():
        manifest_path = out_dir / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"refusing incomplete calibration directory: {out_dir}")
        manifest = read_manifest(manifest_path)
        if manifest.get("inputs", {}).get("config_sha256") != config["sha256"]:
            raise ValueError("existing calibration config differs")
        _verify_output_files(out_dir, manifest["outputs"]["files"])

        return {"action": "reused_verified", "manifest": manifest}
    data = LabData(str(cache_dir), split_manifest_path=str(split_path))
    index = eligible_split_index(data, "calibration")
    calibration_rows = index["row_id"]
    fold_contract = source_stratified_complete_game_folds(
        data,
        outer_split="calibration",
        folds=CROSSFIT_FOLDS,
        salt=CALIBRATION_FOLD_SALT,
    )
    signature_path = crossfit_dir / "fold_contract" / "effective_signature.v16.npy"
    signatures = np.load(signature_path, mmap_mode="r", allow_pickle=False)
    derived = derive_signature_fold_bits(
        signatures[calibration_rows],
        calibration_rows,
        fold_contract["fold_of_row"][calibration_rows],
        total_rows=len(data.meta["split_idx"]),
    )
    del signatures
    final_prediction_path = crossfit_dir / "full_train" / "calibration_predictions.npz"
    final_p = _load_prediction_rows(final_prediction_path, calibration_rows)
    selection = grouped_calibration_selection(
        final_p,
        index["z"],
        index["game_gidx"],
        fold_contract["fold_of_row"][calibration_rows],
        derived["signature_fold_bits"][calibration_rows],
        bootstrap_replicates=bootstrap_replicates,
    )
    selected_method = selection["selected_method"]
    final_calibrator = _fit_calibrator(selected_method, final_p, index["z"])
    temp_dir = out_dir.with_name(f".{out_dir.name}.running.{uuid.uuid4().hex}")
    temp_dir.mkdir(parents=True)
    output_paths = []
    config_path = temp_dir / "config.json"
    write_json(config_path, config)
    output_paths.append(config_path)
    contract_path = temp_dir / "calibration_fold_contract.npz"
    _atomic_npz(
        contract_path,
        row_id=calibration_rows,
        fold_id=fold_contract["fold_of_row"][calibration_rows].astype(np.int8, copy=False),
        signature_fold_bits=derived["signature_fold_bits"][calibration_rows],
    )
    output_paths.append(contract_path)
    assignment_path = temp_dir / "calibration_assignment.json"
    write_json(assignment_path, fold_contract["assignment"])
    output_paths.append(assignment_path)
    cv_prediction_path = temp_dir / "grouped_oof_method_predictions.npz"
    _atomic_npz(
        cv_prediction_path,
        row_id=calibration_rows,
        **{
            method: predicted.astype(np.float32)
            for method, predicted in selection["oof_predictions"].items()
        },
    )
    output_paths.append(cv_prediction_path)
    calibrator_receipts = {}
    final_receipt, final_paths = write_calibrator_bundle(temp_dir, "final_calibrator", final_calibrator)
    calibrator_receipts["final"] = final_receipt
    output_paths.extend(final_paths)
    fold_calibrators = {}
    for fold in range(CROSSFIT_FOLDS):
        fold_prediction_path = crossfit_dir / f"fold_{fold}" / "calibration_predictions.npz"
        fold_p = _load_prediction_rows(fold_prediction_path, calibration_rows)
        calibrator = _fit_calibrator(selected_method, fold_p, index["z"])
        receipt, paths = write_calibrator_bundle(temp_dir, f"fold_{fold}_calibrator", calibrator)
        fold_calibrators[fold] = calibrator
        calibrator_receipts[str(fold)] = receipt
        output_paths.extend(paths)
    with np.load(crossfit_dir / "oof_predictions.npz", allow_pickle=False) as arrays:
        oof = {key: arrays[key] for key in arrays.files}
    calibrated = np.empty(len(oof["raw_probability"]), dtype=np.float32)
    for fold in range(CROSSFIT_FOLDS):
        mask = oof["fold_id"] == fold
        calibrated[mask] = apply_calibrator(
            oof["raw_probability"][mask],
            fold_calibrators[fold],
        ).astype(np.float32)
    if not np.isfinite(calibrated).all() or np.any((calibrated < 0.0) | (calibrated > 1.0)):
        raise FloatingPointError("calibrated OOF probabilities are invalid")
    calibrated_oof_path = temp_dir / "calibrated_oof_predictions.npz"
    _atomic_npz(
        calibrated_oof_path,
        **oof,
        calibrated_probability=calibrated,
    )
    output_paths.append(calibrated_oof_path)
    final_calibrated = apply_calibrator(final_p, final_calibrator)
    fitted_metrics = evaluate_confirmation_predictions(
        final_calibrated,
        index,
        data,
        final_p,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=CALIBRATION_BOOTSTRAP_SEED,
    )
    fitted_metrics["paired_vs_raw"] = fitted_metrics.pop("paired_vs_s0")
    raw_metrics = evaluate_confirmation_predictions(
        final_p,
        index,
        data,
        final_p,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=CALIBRATION_BOOTSTRAP_SEED,
    )
    raw_metrics["paired_vs_raw"] = raw_metrics.pop("paired_vs_s0")
    selection_output = {
        key: value for key, value in selection.items() if key != "oof_predictions"
    }
    result = {
        "schema_version": CALIBRATION_SELECTION_SCHEMA_VERSION,
        "status": "completed_calibrator_frozen",
        "selected_method": selected_method,
        "selection": selection_output,
        "calibration_assignment_sha256": fold_contract["assignment_sha256"],
        "calibration_games": len(fold_contract["assignment"]["assignment"]),
        "calibration_rows": int(len(calibration_rows)),
        "unique_effective_signatures": derived["unique_signatures"],
        "crossing_signature_count": derived["crossing_signature_count"],
        "crossing_signature_count_by_fold": derived["crossing_signature_count_by_fold"],
        "effective_forbidden_overlap": 0,
        "final_calibrator": final_receipt,
        "fold_calibrators": calibrator_receipts,
        "raw_full_calibration_metrics": raw_metrics,
        "fitted_full_calibration_metrics": fitted_metrics,
        "calibrated_oof_predictions_sha256": _sha256(calibrated_oof_path),
        "calibration_targets_consumed": True,
        "test_targets_consumed": False,
        "control_pass": "not_instrumented_cache_has_no_control_pass_row_mask",
    }
    result_path = temp_dir / "result.json"
    write_json(result_path, result)
    output_paths.append(result_path)
    files = _file_receipts(temp_dir, output_paths)
    manifest = build_manifest(
        run_id=out_dir.name,
        kind="value_grouped_calibration_selection",
        command=command or [],
        inputs={"config": config, "config_sha256": config["sha256"]},
        outputs={"files": files},
        summary={
            "status": "completed_calibrator_frozen",
            "selected_method": selected_method,
            "final_calibrator_sha256": final_receipt["bundle_sha256"],
            "final_checkpoint_sha256": config["final_checkpoint_sha256"],
            "calibrated_oof_predictions_sha256": result["calibrated_oof_predictions_sha256"],
            "calibration_rows": int(len(calibration_rows)),
            "calibration_targets_consumed": True,
            "test_targets_consumed": False,
            "protected_runtime_surfaces_changed": False,
        },
    )
    write_manifest(temp_dir / "manifest.json", manifest)
    os.replace(temp_dir, out_dir)

    return {"action": "completed", "manifest": manifest, "result": result}
