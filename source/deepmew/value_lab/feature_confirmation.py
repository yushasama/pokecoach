"""Equal-budget nonlinear confirmation for selected value representations.

This runner is deliberately separate from the accepted Phase-4 snapshot and
temporal runners.  Those runners consume contiguous feature slices; a compact
representation is an ordered index manifest and must never be approximated by
another slice.
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

import deepmew.value_lab.data as value_data
import deepmew.value_lab.feature_contract as value_feature_contract
import deepmew.value_lab.metrics as value_metrics
import deepmew.value_lab.models as value_models
from deepmew.artifacts.manifest import build_manifest, read_manifest, write_manifest
from deepmew.artifacts.writers import write_json
from deepmew.value_lab.data import LabData
from deepmew.value_lab.feature_contract import representation_manifest
from deepmew.value_lab.mc_screen import (
    _atomic_torch_save,
    _json_safe,
    _output_receipt,
    _sha256,
    _stable_hash,
    direct_comparison,
    eligible_row_stream,
    row_stream_receipt,
)
from deepmew.value_lab.metrics import full_metrics, paired_game_cluster_bootstrap
from deepmew.value_lab.models import (
    SnapshotConfig,
    SnapshotMLP,
    matched_model_config,
    parameter_count,
)
from deepmew.value_lab.temporal_screen import temporal_validation_index


FEATURE_CONFIRMATION_SCHEMA_VERSION = "deepmew.value_feature_confirmation.v1"
FEATURE_CONFIRMATION_CHECKPOINT_VERSION = "deepmew.value_feature_confirmation_checkpoint.v1"
RUNG_TARGET_ROWS = {
    "smoke": 65_536,
    "screen_10pct": 720_896,
    "serious_25pct": 1_802_240,
}
DEFAULT_SEEDS = (915_001, 915_002)
DEFAULT_BATCH_SIZE = 4_096
DEFAULT_HIDDEN_DIMS = (256, 128)
COMPACT_LOG_LOSS_MARGIN = 0.001
COMPACT_BRIER_MARGIN = 0.0005
COMPACT_MACRO_SOURCE_MARGIN = 0.001
MIN_FEATURE_REDUCTION = 0.20
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class RepresentationSpec:
    name: str
    indices: tuple[int, ...]
    scientific_sha256: str
    file_sha256: str
    path: str


@dataclass(frozen=True)
class ConfirmationModelSpec:
    name: str
    representation_name: str
    indices: tuple[int, ...]
    hidden_dims: tuple[int, ...]
    role: str
    variant: str


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")

    return payload


def load_selected_representations(
    selection_path: Path,
    representation_paths: tuple[Path, ...],
) -> tuple[tuple[RepresentationSpec, ...], dict[str, Any]]:
    """Load exact F1-selected representation manifests and fail closed on drift."""
    if len(representation_paths) < 2 or len(representation_paths) > 3:
        raise ValueError("nonlinear confirmation requires two or three representations")
    selection = _load_json_object(selection_path)
    selected = selection.get("representation_selection", {}).get(
        "nonlinear_confirmation_representations"
    )
    if not isinstance(selected, list) or not selected:
        raise ValueError("selection artifact lacks nonlinear confirmation representations")
    selected_by_name = {
        str(item.get("name")): item
        for item in selected
        if isinstance(item, dict) and item.get("name")
    }
    specs: list[RepresentationSpec] = []
    for path in representation_paths:
        payload = _load_json_object(path)
        name = str(payload.get("name") or "")
        raw_indices = payload.get("indices")
        if not _SAFE_NAME.fullmatch(name) or not isinstance(raw_indices, list):
            raise ValueError(f"invalid representation identity: {path}")
        indices = tuple(int(index) for index in raw_indices)
        expected = representation_manifest(name, indices)
        if _stable_hash(payload) != _stable_hash(expected):
            raise ValueError(f"representation manifest does not match the live feature contract: {path}")
        selected_item = selected_by_name.get(name)
        if selected_item is None:
            raise ValueError(f"representation was not selected by F1: {name}")
        if (
            selected_item.get("indices") != list(indices)
            or selected_item.get("sha256") != expected["sha256"]
        ):
            raise ValueError(f"F1 selection and representation manifest disagree: {name}")
        specs.append(RepresentationSpec(
            name=name,
            indices=indices,
            scientific_sha256=expected["sha256"],
            file_sha256=_sha256(path),
            path=str(path.resolve()),
        ))
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("representation names must be unique")
    if specs[0].indices != tuple(range(49)):
        raise ValueError("the first confirmation representation must be raw S1 [0:49]")
    if selection.get("representation_selection", {}).get("s2_advancement_gate") is not False:
        raise ValueError("this bounded F3 runner requires the accepted negative S2 advancement gate")
    if any(any(index >= 49 for index in spec.indices) for spec in specs):
        raise ValueError("S2 features are not authorized by the accepted F1 gate")

    return tuple(specs), selection


def confirmation_model_specs(
    representations: tuple[RepresentationSpec, ...],
    *,
    include_matched_controls: bool,
) -> tuple[ConfirmationModelSpec, ...]:
    baseline = representations[0]
    baseline_model = SnapshotMLP(len(baseline.indices), DEFAULT_HIDDEN_DIMS)
    parameter_budget = parameter_count(baseline_model)
    models = [ConfirmationModelSpec(
        name=f"{baseline.name}__default",
        representation_name=baseline.name,
        indices=baseline.indices,
        hidden_dims=DEFAULT_HIDDEN_DIMS,
        role="raw_s1_reference",
        variant="default",
    )]
    for representation in representations[1:]:
        models.append(ConfirmationModelSpec(
            name=f"{representation.name}__default",
            representation_name=representation.name,
            indices=representation.indices,
            hidden_dims=DEFAULT_HIDDEN_DIMS,
            role="deployable_same_hidden_compact_candidate",
            variant="default",
        ))
        if include_matched_controls:
            matched = matched_model_config(
                "s1",
                len(representation.indices),
                len(representation.indices),
                parameter_budget,
            )
            if not isinstance(matched, SnapshotConfig):
                raise AssertionError("snapshot parameter matching returned the wrong config type")
            models.append(ConfirmationModelSpec(
                name=f"{representation.name}__matched",
                representation_name=representation.name,
                indices=representation.indices,
                hidden_dims=matched.hidden_dims,
                role="equal_parameter_feature_contribution_control",
                variant="matched",
            ))

    return tuple(models)


def build_confirmation_model(spec: ConfirmationModelSpec) -> SnapshotMLP:
    return SnapshotMLP(len(spec.indices), spec.hidden_dims, dropout=0.0)


def feature_batch(
    data: LabData,
    rows: np.ndarray,
    spec: ConfirmationModelSpec,
) -> torch.Tensor:
    row_index = np.asarray(rows, dtype=np.int64)
    column_index = np.asarray(spec.indices, dtype=np.int64)
    values = np.asarray(data.X[np.ix_(row_index, column_index)], dtype=np.float32)
    if values.shape != (len(row_index), len(column_index)):
        raise ValueError("indexed representation batch has the wrong shape")

    return torch.from_numpy(values)


def train_confirmation_model(
    data: LabData,
    spec: ConfirmationModelSpec,
    row_stream: np.ndarray,
    *,
    seed: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[SnapshotMLP, dict[str, Any]]:
    torch.manual_seed(seed)
    model = build_confirmation_model(spec)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train()
    weighted_loss = 0.0
    updates = 0
    loader_seconds = 0.0
    compute_seconds = 0.0
    nonfinite = {"features": 0, "logits": 0, "losses": 0, "gradients": 0}
    started = time.perf_counter()
    for start in range(0, len(row_stream), batch_size):
        rows = row_stream[start:start + batch_size]
        loader_started = time.perf_counter()
        features = feature_batch(data, rows, spec)
        labels = torch.from_numpy(data.meta["z"][rows].astype(np.float32, copy=False))
        sequence_mask = torch.ones((len(rows), 1), dtype=torch.bool)
        loader_seconds += time.perf_counter() - loader_started
        nonfinite["features"] += int((~torch.isfinite(features)).sum().item())
        compute_started = time.perf_counter()
        logits = model(features[:, None, :], sequence_mask)[:, 0]
        nonfinite["logits"] += int((~torch.isfinite(logits)).sum().item())
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        nonfinite["losses"] += int(not bool(torch.isfinite(loss).item()))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nonfinite["gradients"] += sum(
            int((~torch.isfinite(parameter.grad)).sum().item())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        if any(nonfinite.values()):
            raise FloatingPointError(f"non-finite confirmation training values: {nonfinite}")
        optimizer.step()
        compute_seconds += time.perf_counter() - compute_started
        weighted_loss += float(loss.item()) * len(rows)
        updates += 1
    elapsed = time.perf_counter() - started

    return model, {
        "presentations": int(len(row_stream)),
        "updates": updates,
        "batch_size": batch_size,
        "mean_training_loss": weighted_loss / len(row_stream),
        "elapsed_seconds": elapsed,
        "rows_per_second": len(row_stream) / max(elapsed, 1e-12),
        "data_loader_seconds": loader_seconds,
        "optimizer_compute_seconds": compute_seconds,
        "nonfinite_counts": nonfinite,
    }


def train_confirmation_models(
    data: LabData,
    specs: tuple[ConfirmationModelSpec, ...],
    row_stream: np.ndarray,
    *,
    seed: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, tuple[SnapshotMLP, dict[str, Any]]]:
    """Train the paired matrix in one cache pass while keeping optimizers independent."""
    models: dict[str, SnapshotMLP] = {}
    optimizers = {}
    state = {}
    for spec in specs:
        torch.manual_seed(seed)
        model = build_confirmation_model(spec)
        models[spec.name] = model
        optimizers[spec.name] = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay,
        )
        model.train()
        state[spec.name] = {
            "weighted_loss": 0.0,
            "updates": 0,
            "compute_seconds": 0.0,
            "nonfinite": {"features": 0, "logits": 0, "losses": 0, "gradients": 0},
        }
    shared_loader_seconds = 0.0
    started = time.perf_counter()
    for start in range(0, len(row_stream), batch_size):
        rows = row_stream[start:start + batch_size]
        loader_started = time.perf_counter()
        full_features = np.asarray(data.X[rows], dtype=np.float32)
        labels = torch.from_numpy(data.meta["z"][rows].astype(np.float32, copy=False))
        sequence_mask = torch.ones((len(rows), 1), dtype=torch.bool)
        features_by_indices = {
            spec.indices: torch.from_numpy(
                np.ascontiguousarray(full_features[:, np.asarray(spec.indices, dtype=np.int64)])
            )
            for spec in specs
        }
        shared_loader_seconds += time.perf_counter() - loader_started
        for spec in specs:
            model = models[spec.name]
            optimizer = optimizers[spec.name]
            observed = state[spec.name]
            features = features_by_indices[spec.indices]
            observed["nonfinite"]["features"] += int((~torch.isfinite(features)).sum().item())
            compute_started = time.perf_counter()
            logits = model(features[:, None, :], sequence_mask)[:, 0]
            observed["nonfinite"]["logits"] += int((~torch.isfinite(logits)).sum().item())
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            observed["nonfinite"]["losses"] += int(not bool(torch.isfinite(loss).item()))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            observed["nonfinite"]["gradients"] += sum(
                int((~torch.isfinite(parameter.grad)).sum().item())
                for parameter in model.parameters()
                if parameter.grad is not None
            )
            if any(observed["nonfinite"].values()):
                raise FloatingPointError(
                    f"non-finite confirmation training values for {spec.name}: "
                    f"{observed['nonfinite']}"
                )
            optimizer.step()
            observed["compute_seconds"] += time.perf_counter() - compute_started
            observed["weighted_loss"] += float(loss.item()) * len(rows)
            observed["updates"] += 1
    elapsed = time.perf_counter() - started
    results = {}
    allocated_loader_seconds = shared_loader_seconds / len(specs)
    for spec in specs:
        observed = state[spec.name]
        effective_seconds = observed["compute_seconds"] + allocated_loader_seconds
        results[spec.name] = (
            models[spec.name],
            {
                "presentations": int(len(row_stream)),
                "updates": int(observed["updates"]),
                "batch_size": batch_size,
                "mean_training_loss": observed["weighted_loss"] / len(row_stream),
                "elapsed_seconds": effective_seconds,
                "rows_per_second": len(row_stream) / max(effective_seconds, 1e-12),
                "shared_matrix_wall_seconds": elapsed,
                "shared_data_loader_seconds": shared_loader_seconds,
                "allocated_data_loader_seconds": allocated_loader_seconds,
                "data_loader_seconds": allocated_loader_seconds,
                "optimizer_compute_seconds": observed["compute_seconds"],
                "nonfinite_counts": observed["nonfinite"],
            },
        )

    return results


def predict_confirmation_model(
    data: LabData,
    model: SnapshotMLP,
    spec: ConfirmationModelSpec,
    row_ids: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    model.eval()
    predictions = np.empty(len(row_ids), dtype=np.float32)
    loader_seconds = 0.0
    compute_seconds = 0.0
    nonfinite = {"features": 0, "logits": 0, "predictions": 0}
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(row_ids), batch_size):
            stop = min(start + batch_size, len(row_ids))
            loader_started = time.perf_counter()
            features = feature_batch(data, row_ids[start:stop], spec)
            sequence_mask = torch.ones((len(features), 1), dtype=torch.bool)
            loader_seconds += time.perf_counter() - loader_started
            nonfinite["features"] += int((~torch.isfinite(features)).sum().item())
            compute_started = time.perf_counter()
            logits = model(features[:, None, :], sequence_mask)[:, 0]
            probabilities = torch.sigmoid(logits)
            compute_seconds += time.perf_counter() - compute_started
            nonfinite["logits"] += int((~torch.isfinite(logits)).sum().item())
            nonfinite["predictions"] += int((~torch.isfinite(probabilities)).sum().item())
            predictions[start:stop] = probabilities.cpu().numpy()
    elapsed = time.perf_counter() - started
    if any(nonfinite.values()):
        raise FloatingPointError(f"non-finite confirmation inference values: {nonfinite}")

    return predictions, {
        "elapsed_seconds": elapsed,
        "data_loader_seconds": loader_seconds,
        "model_compute_seconds": compute_seconds,
        "rows_per_second": len(predictions) / max(elapsed, 1e-12),
        "latency_microseconds_per_row": 1e6 * elapsed / len(predictions),
        "nonfinite_counts": nonfinite,
    }


def _light_metrics(p: np.ndarray, z: np.ndarray) -> dict[str, Any]:
    return {
        "n": int(len(z)),
        "mean_p": float(np.mean(p)),
        "mean_z": float(np.mean(z)),
        "log_loss": value_metrics.log_loss(p, z),
        "brier": value_metrics.brier(p, z),
        "ece": value_metrics.ece(p, z)["ece"],
    }


def _light_slices(
    p: np.ndarray,
    z: np.ndarray,
    keys: np.ndarray,
    *,
    names: list[str] | tuple[str, ...] | None = None,
    min_n: int = 500,
) -> dict[str, dict[str, Any]]:
    result = {}
    for raw_key in np.unique(keys):
        mask = keys == raw_key
        if int(mask.sum()) < min_n:
            continue
        index = int(raw_key)
        name = (
            names[index]
            if names is not None and 0 <= index < len(names)
            else str(index)
        )
        result[name] = _light_metrics(p[mask], z[mask])

    return result


def evaluate_confirmation_predictions(
    p: np.ndarray,
    index: dict[str, np.ndarray],
    data: LabData,
    reference_p: np.ndarray,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    z = index["z"]
    source = _light_slices(p, z, index["source_idx"], names=data.source_ids, min_n=1)
    block_ids = index["block_id"].astype(np.int64, copy=False)
    policy = data.blocks["policy_gidx"][block_ids]
    opponent_policy = data.blocks["opp_policy_gidx"][block_ids]
    deck = data.blocks["deck_gidx"][block_ids]
    opponent_deck = data.blocks["opp_deck_gidx"][block_ids]
    turn = index["turn"]
    phase = np.where(turn <= 5, 0, np.where(turn <= 9, 1, 2)).astype(np.int8)
    phase_names = ("early_turn_0_5", "mid_turn_6_9", "late_turn_10_plus")

    return {
        "overall": full_metrics(p, z),
        "by_source": source,
        "macro_source": {
            "log_loss": float(np.mean([metrics["log_loss"] for metrics in source.values()])),
            "brier": float(np.mean([metrics["brier"] for metrics in source.values()])),
        },
        "by_seat": _light_slices(p, z, index["seat"], min_n=1),
        "by_phase": _light_slices(p, z, phase, names=phase_names, min_n=1),
        "by_policy": _light_slices(p, z, policy, names=data.policy_vocab),
        "by_opponent_policy": _light_slices(
            p, z, opponent_policy, names=data.policy_vocab,
        ),
        "by_deck": _light_slices(p, z, deck, names=data.deck_vocab),
        "by_opponent_deck": _light_slices(
            p, z, opponent_deck, names=data.deck_vocab,
        ),
        "paired_vs_s0": {
            metric: paired_game_cluster_bootstrap(
                p,
                reference_p,
                z,
                index["game_gidx"],
                metric=metric,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            )
            for metric in ("log_loss", "brier")
        },
        "control_pass": "not_instrumented_cache_has_no_control_pass_row_mask",
    }


def _mean_se(values: list[float]) -> dict[str, Any]:
    observed = np.asarray(values, dtype=np.float64)
    finite = observed[np.isfinite(observed)]

    return {
        "mean": float(finite.mean()) if len(finite) else float("nan"),
        "se": float(finite.std(ddof=1) / math.sqrt(len(finite))) if len(finite) > 1 else float("nan"),
        "finite_seeds": int(len(finite)),
    }


def _slice_damage(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    damage = {}
    for key in sorted(set(candidate) & set(reference)):
        damage[key] = {
            "n": int(candidate[key]["n"]),
            "log_loss_damage": float(candidate[key]["log_loss"] - reference[key]["log_loss"]),
            "brier_damage": float(candidate[key]["brier"] - reference[key]["brier"]),
        }

    return damage


def aggregate_confirmation(
    results: dict[str, Any],
    specs: tuple[ConfirmationModelSpec, ...],
    predictions: dict[str, list[np.ndarray]],
    index: dict[str, np.ndarray],
    data: LabData,
    *,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "seed_count": len(results),
        "models": {},
        "direct_comparisons": {},
        "ensemble_mean": {},
        "ensemble_mean_comparisons": {},
        "ensemble_slice_damage": {},
    }
    for spec in specs:
        aggregate["models"][spec.name] = {
            metric: _mean_se([
                result["models"][spec.name]["metrics"]["overall"][metric]
                for result in results.values()
            ])
            for metric in ("log_loss", "brier", "ece", "auc")
        }
        aggregate["models"][spec.name]["macro_source_log_loss"] = _mean_se([
            result["models"][spec.name]["metrics"]["macro_source"]["log_loss"]
            for result in results.values()
        ])
    baseline_name = specs[0].name
    for spec in specs[1:]:
        comparison_name = f"{spec.name}_vs_{baseline_name}"
        aggregate["direct_comparisons"][comparison_name] = {
            metric: {
                "per_seed": [
                    result["direct_comparisons"][comparison_name][metric][
                        "delta_candidate_minus_reference"
                    ]
                    for result in results.values()
                ]
            }
            for metric in ("log_loss", "brier")
        }
        for metric in ("log_loss", "brier"):
            values = aggregate["direct_comparisons"][comparison_name][metric]["per_seed"]
            aggregate["direct_comparisons"][comparison_name][metric].update(
                _mean_se(values)
            )
    ensemble_base_rate = float(np.mean([
        result["s0"]["base_rate"] for result in results.values()
    ]))
    ensemble_s0 = np.full(len(index["z"]), ensemble_base_rate, dtype=np.float32)
    ensemble_predictions = {
        spec.name: np.mean(np.stack(predictions[spec.name]), axis=0)
        for spec in specs
    }
    for spec in specs:
        aggregate["ensemble_mean"][spec.name] = evaluate_confirmation_predictions(
            ensemble_predictions[spec.name],
            index,
            data,
            ensemble_s0,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=DEFAULT_SEEDS[0],
        )
    baseline_metrics = aggregate["ensemble_mean"][baseline_name]
    for spec in specs[1:]:
        comparison_name = f"{spec.name}_vs_{baseline_name}"
        aggregate["ensemble_mean_comparisons"][comparison_name] = direct_comparison(
            ensemble_predictions[spec.name],
            ensemble_predictions[baseline_name],
            index,
            candidate=spec.name,
            reference=baseline_name,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=DEFAULT_SEEDS[0],
        )
        candidate_metrics = aggregate["ensemble_mean"][spec.name]
        aggregate["ensemble_slice_damage"][comparison_name] = {
            slice_name: _slice_damage(candidate_metrics[slice_name], baseline_metrics[slice_name])
            for slice_name in ("by_source", "by_deck", "by_policy", "by_phase")
        }

    return aggregate


def decide_confirmation_gate(
    results: dict[str, Any],
    aggregate: dict[str, Any],
    representations: tuple[RepresentationSpec, ...],
    specs: tuple[ConfirmationModelSpec, ...],
    *,
    rung: str,
) -> dict[str, Any]:
    baseline = specs[0]
    if rung == "smoke":
        return {
            "status": "plumbing_smoke_complete",
            "advance_representations": [representation.name for representation in representations],
            "winner": None,
            "selection_metric": "not_selected_at_smoke",
        }
    by_representation: dict[str, dict[str, ConfirmationModelSpec]] = {}
    for spec in specs:
        by_representation.setdefault(spec.representation_name, {})[spec.variant] = spec
    baseline_metrics = aggregate["models"][baseline.name]
    candidates = {}
    advancing = [representations[0].name]
    for representation in representations[1:]:
        variants = by_representation[representation.name]
        default_spec = variants["default"]
        matched_spec = variants.get("matched", default_spec)
        default_comparison = aggregate["ensemble_mean_comparisons"][
            f"{default_spec.name}_vs_{baseline.name}"
        ]
        matched_comparison = aggregate["ensemble_mean_comparisons"][
            f"{matched_spec.name}_vs_{baseline.name}"
        ]
        default_seed_delta = aggregate["direct_comparisons"][
            f"{default_spec.name}_vs_{baseline.name}"
        ]["log_loss"]["per_seed"]
        feature_reduction = 1.0 - len(representation.indices) / len(representations[0].indices)
        default_macro_damage = (
            aggregate["models"][default_spec.name]["macro_source_log_loss"]["mean"]
            - baseline_metrics["macro_source_log_loss"]["mean"]
        )
        checks = {
            "feature_reduction_at_least_20pct": feature_reduction >= MIN_FEATURE_REDUCTION,
            "default_all_seed_log_loss_within_margin": all(
                delta <= COMPACT_LOG_LOSS_MARGIN for delta in default_seed_delta
            ),
            "default_ensemble_log_loss_ci_upper_within_margin": (
                default_comparison["log_loss"]["ci95"][1] <= COMPACT_LOG_LOSS_MARGIN
            ),
            "default_ensemble_brier_within_margin": (
                default_comparison["brier"]["delta_candidate_minus_reference"]
                <= COMPACT_BRIER_MARGIN
            ),
            "default_macro_source_log_loss_within_margin": (
                default_macro_damage <= COMPACT_MACRO_SOURCE_MARGIN
            ),
            "matched_ensemble_log_loss_ci_upper_within_margin": (
                matched_comparison["log_loss"]["ci95"][1] <= COMPACT_LOG_LOSS_MARGIN
            ),
            "matched_ensemble_brier_within_margin": (
                matched_comparison["brier"]["delta_candidate_minus_reference"]
                <= COMPACT_BRIER_MARGIN
            ),
        }
        passes = all(checks.values())
        if passes:
            advancing.append(representation.name)
        candidates[representation.name] = {
            "passes_compact_gate": passes,
            "checks": checks,
            "feature_count": len(representation.indices),
            "feature_reduction": feature_reduction,
            "default_model": default_spec.name,
            "matched_control": matched_spec.name,
            "default_parameter_count": parameter_count(build_confirmation_model(default_spec)),
            "matched_parameter_count": parameter_count(build_confirmation_model(matched_spec)),
            "reference_parameter_count": parameter_count(build_confirmation_model(baseline)),
            "default_log_loss_delta_per_seed": default_seed_delta,
            "default_macro_source_log_loss_damage": default_macro_damage,
            "default_ensemble_comparison": default_comparison,
            "matched_ensemble_comparison": matched_comparison,
        }
    winner = None
    status = "screen_complete"
    if rung == "serious_25pct":
        passing = [name for name, evidence in candidates.items() if evidence["passes_compact_gate"]]
        winner = (
            min(
                passing,
                key=lambda name: aggregate["models"][by_representation[name]["default"].name][
                    "log_loss"
                ]["mean"],
            )
            if passing else representations[0].name
        )
        status = "serious_complete_representation_frozen"

    return {
        "status": status,
        "selection_metric": "raw_validation_log_loss_with_game_clustered_noninferiority",
        "margins": {
            "log_loss": COMPACT_LOG_LOSS_MARGIN,
            "brier": COMPACT_BRIER_MARGIN,
            "macro_source_log_loss": COMPACT_MACRO_SOURCE_MARGIN,
            "minimum_feature_reduction": MIN_FEATURE_REDUCTION,
        },
        "candidates": candidates,
        "advance_representations": advancing,
        "winner": winner,
        "calibration_targets_consumed": False,
        "test_targets_consumed": False,
    }


def _expected_output_paths(config: dict[str, Any]) -> set[str]:
    expected = {
        "config.json",
        "representation_contract.json",
        "validation_index.npz",
        "result.json",
        "selection.json",
    }
    model_names = [model["name"] for model in config["models"]]
    for seed in config["seeds"]:
        prefix = f"seed_{seed}"
        expected.update({
            f"{prefix}/train_row_stream.npz",
            f"{prefix}/s0_predictions.npz",
            f"{prefix}/result.json",
        })
        for name in model_names:
            expected.add(f"{prefix}/{name}.pt")
            expected.add(f"{prefix}/{name}.predictions.npz")

    return expected


def _verify_completed(out_dir: Path, config: dict[str, Any], config_sha: str) -> dict[str, Any] | None:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = read_manifest(manifest_path)
    if manifest.get("status") != "completed":
        raise ValueError("existing confirmation manifest is not completed")
    if manifest.get("inputs", {}).get("config_sha256") != config_sha:
        raise ValueError("existing confirmation config does not match")
    files = manifest.get("outputs", {}).get("files")
    if not isinstance(files, list):
        raise ValueError("existing confirmation manifest lacks output inventory")
    observed = {entry.get("path") for entry in files if isinstance(entry, dict)}
    if observed != _expected_output_paths(config):
        raise ValueError("existing confirmation manifest has an incomplete output inventory")
    for entry in files:
        path = out_dir / entry["path"]
        if not path.is_file() or path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"confirmation output missing or size-mismatched: {entry['path']}")
        if _sha256(path) != entry["sha256"]:
            raise ValueError(f"confirmation output hash mismatch: {entry['path']}")

    return manifest


def run_feature_confirmation(
    *,
    cache_dir: Path,
    split_path: Path,
    selection_path: Path,
    representation_paths: tuple[Path, ...],
    out_dir: Path,
    rung: str,
    presentations: int | None = None,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    threads: int = 8,
    bootstrap_replicates: int = 500,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    include_matched_controls: bool = True,
    command: list[str] | None = None,
) -> dict[str, Any]:
    if rung not in RUNG_TARGET_ROWS:
        raise ValueError(f"rung must be one of {sorted(RUNG_TARGET_ROWS)}")
    row_budget = int(presentations if presentations is not None else RUNG_TARGET_ROWS[rung])
    if row_budget < 1 or batch_size < 1 or row_budget % batch_size:
        raise ValueError("presentations must be positive and divisible by batch_size")
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be unique nonnegative integers")
    if rung != "smoke" and len(seeds) < 2:
        raise ValueError("scientific confirmation rungs require at least two seeds")
    if threads < 1 or bootstrap_replicates < 2:
        raise ValueError("threads and bootstrap replicates must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("optimizer configuration is invalid")
    cache_dir = cache_dir.resolve()
    split_path = split_path.resolve()
    selection_path = selection_path.resolve()
    representation_paths = tuple(path.resolve() for path in representation_paths)
    out_dir = out_dir.resolve()
    representations, selection = load_selected_representations(
        selection_path,
        representation_paths,
    )
    specs = confirmation_model_specs(
        representations,
        include_matched_controls=include_matched_controls,
    )
    cache_manifest_path = cache_dir / "cache_manifest.json"
    parameter_counts = {
        spec.name: parameter_count(build_confirmation_model(spec)) for spec in specs
    }
    config = {
        "schema_version": FEATURE_CONFIRMATION_SCHEMA_VERSION,
        "rung": rung,
        "presentations": row_budget,
        "seeds": list(seeds),
        "batch_size": batch_size,
        "threads": threads,
        "bootstrap_replicates": bootstrap_replicates,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "target": "mc_actor_relative_terminal",
        "normalization": "identity_canonical_float32",
        "checkpoint_selection": "final_update_only_no_validation_early_stopping",
        "calibration_protocol": "none_screening_raw_sigmoid_calibration_fold_untouched",
        "include_matched_controls": include_matched_controls,
        "representations": [asdict(representation) for representation in representations],
        "models": [
            {**asdict(spec), "parameter_count": parameter_counts[spec.name]}
            for spec in specs
        ],
        "selection_path": str(selection_path),
        "selection_sha256": _sha256(selection_path),
        "selection_status": selection.get("status"),
        "cache_manifest_sha256": _sha256(cache_manifest_path),
        "split_manifest_sha256": _sha256(split_path),
        "thresholds": {
            "compact_log_loss_margin": COMPACT_LOG_LOSS_MARGIN,
            "compact_brier_margin": COMPACT_BRIER_MARGIN,
            "compact_macro_source_margin": COMPACT_MACRO_SOURCE_MARGIN,
            "minimum_feature_reduction": MIN_FEATURE_REDUCTION,
        },
        "code_sha256": {
            "feature_confirmation": _sha256(Path(__file__)),
            "feature_contract": _sha256(Path(value_feature_contract.__file__)),
            "data": _sha256(Path(value_data.__file__)),
            "metrics": _sha256(Path(value_metrics.__file__)),
            "models": _sha256(Path(value_models.__file__)),
        },
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "device": "cpu",
            "gpu_utilization": "not_instrumented_cpu_only_execution",
        },
        "calibration_targets_consumed": False,
        "test_targets_consumed": False,
    }
    config_sha = _stable_hash(config)
    if out_dir.exists():
        completed = _verify_completed(out_dir, config, config_sha)
        if completed is not None:
            return {"action": "reused_verified", "manifest": completed}
        preview = ", ".join(sorted(path.name for path in out_dir.iterdir())[:10]) or "empty"
        raise RuntimeError(f"refusing to overwrite incomplete confirmation run: {out_dir} ({preview})")
    data = LabData(str(cache_dir), split_manifest_path=str(split_path))
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = out_dir.with_name(f".{out_dir.name}.running.{uuid.uuid4().hex}")
    temp_dir.mkdir()
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    index = temporal_validation_index(data)
    index_path = temp_dir / "validation_index.npz"
    np.savez_compressed(index_path, **index)
    config_path = temp_dir / "config.json"
    write_json(config_path, _json_safe(config))
    representation_path = temp_dir / "representation_contract.json"
    write_json(representation_path, {
        "schema_version": FEATURE_CONFIRMATION_SCHEMA_VERSION,
        "representations": [asdict(representation) for representation in representations],
        "models": [
            {**asdict(spec), "parameter_count": parameter_counts[spec.name]}
            for spec in specs
        ],
    })
    output_paths = [config_path, representation_path, index_path]
    results: dict[str, Any] = {}
    predictions_across_seeds: dict[str, list[np.ndarray]] = {spec.name: [] for spec in specs}
    for seed in seeds:
        row_stream = eligible_row_stream(data, split="train", presentations=row_budget, seed=seed)
        stream_receipt = row_stream_receipt(data, row_stream)
        seed_dir = temp_dir / f"seed_{seed}"
        seed_dir.mkdir()
        stream_path = seed_dir / "train_row_stream.npz"
        np.savez_compressed(stream_path, row_id=row_stream.astype(np.int64, copy=False))
        train_labels = data.meta["z"][row_stream].astype(np.float64, copy=False)
        base_rate = float(train_labels.mean())
        s0_predictions = np.full(len(index["z"]), base_rate, dtype=np.float32)
        s0_path = seed_dir / "s0_predictions.npz"
        np.savez_compressed(s0_path, p=s0_predictions)
        s0_metrics = evaluate_confirmation_predictions(
            s0_predictions,
            index,
            data,
            s0_predictions,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=seed,
        )
        seed_result: dict[str, Any] = {
            "seed": seed,
            "presentations": row_budget,
            "row_stream_sha256": stream_receipt["sha256"],
            "row_stream": stream_receipt,
            "s0": {
                "base_rate": base_rate,
                "metrics": s0_metrics,
            },
            "models": {},
            "direct_comparisons": {},
        }
        output_paths.extend([stream_path, s0_path])
        predictions_by_name = {}
        trained_models = train_confirmation_models(
            data,
            specs,
            row_stream,
            seed=seed,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        for spec in specs:
            model, train_receipt = trained_models[spec.name]
            predictions, inference_receipt = predict_confirmation_model(
                data,
                model,
                spec,
                index["row_id"],
                batch_size=batch_size,
            )
            metrics = evaluate_confirmation_predictions(
                predictions,
                index,
                data,
                s0_predictions,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=seed,
            )
            checkpoint_path = seed_dir / f"{spec.name}.pt"
            checkpoint = {
                "schema_version": FEATURE_CONFIRMATION_CHECKPOINT_VERSION,
                "spec": asdict(spec),
                "model_config": {
                    "input_dim": len(spec.indices),
                    "hidden_dims": list(spec.hidden_dims),
                    "dropout": 0.0,
                },
                "seed": seed,
                "row_stream_sha256": stream_receipt["sha256"],
                "cache_manifest_sha256": config["cache_manifest_sha256"],
                "split_manifest_sha256": config["split_manifest_sha256"],
                "outer_config_sha256": config_sha,
                "code_sha256": config["code_sha256"],
                "parameter_count": parameter_counts[spec.name],
                "model_state": model.state_dict(),
                "optimizer": {
                    "name": "AdamW",
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                },
                "train": train_receipt,
                "validation": metrics["overall"],
            }
            _atomic_torch_save(checkpoint_path, checkpoint)
            prediction_path = seed_dir / f"{spec.name}.predictions.npz"
            np.savez_compressed(prediction_path, p=predictions)
            output_paths.extend([checkpoint_path, prediction_path])
            predictions_by_name[spec.name] = predictions
            predictions_across_seeds[spec.name].append(predictions)
            seed_result["models"][spec.name] = {
                "spec": asdict(spec),
                "parameter_count": parameter_counts[spec.name],
                "parameter_bytes_float32": parameter_counts[spec.name] * 4,
                "train": train_receipt,
                "inference": inference_receipt,
                "metrics": metrics,
            }
        baseline_name = specs[0].name
        seed_result["direct_comparisons"] = {
            f"{spec.name}_vs_{baseline_name}": direct_comparison(
                predictions_by_name[spec.name],
                predictions_by_name[baseline_name],
                index,
                candidate=spec.name,
                reference=baseline_name,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=seed,
            )
            for spec in specs[1:]
        }
        results[str(seed)] = seed_result
        seed_result_path = seed_dir / "result.json"
        write_json(seed_result_path, _json_safe(seed_result))
        output_paths.append(seed_result_path)
    aggregate = aggregate_confirmation(
        results,
        specs,
        predictions_across_seeds,
        index,
        data,
        bootstrap_replicates=bootstrap_replicates,
    )
    gate = decide_confirmation_gate(
        results,
        aggregate,
        representations,
        specs,
        rung=rung,
    )
    result_path = temp_dir / "result.json"
    write_json(result_path, _json_safe({
        "schema_version": FEATURE_CONFIRMATION_SCHEMA_VERSION,
        "config": config,
        "config_sha256": config_sha,
        "seeds": results,
        "aggregate": aggregate,
        "gate": gate,
    }))
    selection_output_path = temp_dir / "selection.json"
    write_json(selection_output_path, _json_safe(gate))
    output_paths.extend([result_path, selection_output_path])
    files = [_output_receipt(path, temp_dir) for path in sorted(output_paths)]
    nonfinite = {
        "train": sum(
            sum(model["train"]["nonfinite_counts"].values())
            for result in results.values()
            for model in result["models"].values()
        ),
        "inference": sum(
            sum(model["inference"]["nonfinite_counts"].values())
            for result in results.values()
            for model in result["models"].values()
        ),
    }
    manifest = build_manifest(
        run_id=out_dir.name,
        kind="value_feature_confirmation",
        command=command or [],
        inputs={
            "config": config,
            "config_sha256": config_sha,
            "cache_dir": str(cache_dir),
            "split_manifest_path": str(split_path),
            "selection_path": str(selection_path),
            "feature_confirmation_code_sha256": _sha256(Path(__file__)),
        },
        outputs={"files": files},
        summary={
            "status": "screening_only",
            "rung": rung,
            "seeds": list(seeds),
            "presentations_per_model_seed": row_budget,
            "validation_rows": int(len(index["z"])),
            "gate": _json_safe(gate),
            "nonfinite_counts": nonfinite,
            "calibration_targets_consumed": False,
            "test_targets_consumed": False,
            "control_pass": "not_instrumented_cache_has_no_control_pass_row_mask",
            "gpu_utilization": config["environment"]["gpu_utilization"],
            "protected_runtime_surfaces_changed": False,
            "offline_gameplay_safety_monitors": "not_applicable_no_gameplay_executed",
        },
    )
    write_manifest(temp_dir / "manifest.json", manifest)
    os.replace(temp_dir, out_dir)

    return {"action": "completed", "manifest": manifest, "results": results, "gate": gate}
