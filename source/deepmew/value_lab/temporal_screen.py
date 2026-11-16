"""Compact MC-only temporal value screen over accepted actor sequences.

The primary comparison isolates temporal history on the accepted S1 feature
prefix.  Every temporal update consumes the exact same eligible target-row
stream as V0 while replaying complete game-seat sequences as causal context.
Target-excluded rows remain sequence context and never receive direct loss.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

import deepmew.value_lab.data as value_data
import deepmew.value_lab.metrics as value_metrics
import deepmew.value_lab.models as value_models
from deepmew.artifacts.manifest import build_manifest, read_manifest, write_manifest
from deepmew.artifacts.writers import write_json
from deepmew.replay.canonical_actor_dataset import S1_SLICE
from deepmew.value_lab.data import LabData
from deepmew.value_lab.mc_screen import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SEED,
    SnapshotSpec,
    _atomic_torch_save,
    _json_safe,
    _output_receipt,
    _predict,
    _sha256,
    _stable_hash,
    direct_comparison,
    eligible_row_stream,
    evaluate_predictions,
    row_stream_receipt,
    snapshot_specs,
    train_snapshot,
    validation_index,
)
from deepmew.value_lab.metrics import slice_metrics
from deepmew.value_lab.models import build_model, parameter_count
from deepmew.value_lab.targets import mc_targets


TEMPORAL_SCREEN_SCHEMA_VERSION = "deepmew.value_mc_temporal_screen.v1"
TEMPORAL_CHECKPOINT_SCHEMA_VERSION = "deepmew.value_mc_temporal_checkpoint.v1"
RUNG_TARGET_ROWS = {
    "smoke": 65_536,
    "ten_percent": 720_896,
    "serious_25pct": 1_802_240,
}
DEFAULT_PARAMETER_BUDGET = 45_825
DEFAULT_WINDOW = 8
DEFAULT_SEQUENCE_BLOCKS_PER_BATCH = 64


@dataclass(frozen=True)
class TemporalSpec:
    name: str
    arch: str
    feature_start: int
    feature_stop: int
    parameter_budget: int
    window: int
    role: str


def temporal_specs(
    *,
    parameter_budget: int = DEFAULT_PARAMETER_BUDGET,
    window: int = DEFAULT_WINDOW,
) -> tuple[TemporalSpec, ...]:
    return (
        TemporalSpec(
            "v0_snapshot_s1",
            "s1",
            S1_SLICE[0],
            S1_SLICE[1],
            parameter_budget,
            1,
            "current_actor_visible_state_control",
        ),
        TemporalSpec(
            f"v1_window{window}_s1",
            "s3",
            S1_SLICE[0],
            S1_SLICE[1],
            parameter_budget,
            window,
            "fixed_causal_history",
        ),
        TemporalSpec(
            "v2_lstm_s1",
            "s5",
            S1_SLICE[0],
            S1_SLICE[1],
            parameter_budget,
            1,
            "long_range_unidirectional_history",
        ),
    )


def build_temporal_model(spec: TemporalSpec) -> torch.nn.Module:
    input_dim = spec.feature_stop - spec.feature_start
    if spec.arch == "s1" and spec.parameter_budget == 45_825:
        model, consumed = build_model("s1", input_dim, input_dim)
    else:
        model, consumed = build_model(
            spec.arch,
            input_dim,
            input_dim,
            variant="matched",
            parameter_budget=spec.parameter_budget,
            matched_window=spec.window,
        )
    if consumed != input_dim:
        raise AssertionError("temporal model input contract drifted")

    return model


def _model_config(model: torch.nn.Module) -> dict[str, Any]:
    if isinstance(model, value_models.SnapshotMLP):
        return {
            "type": "snapshot_mlp",
            "input_dim": model.input_dim,
            "hidden_dims": list(model.hidden_dims),
            "dropout": model.dropout,
        }
    if isinstance(model, value_models.CausalWindowMLP):
        return {
            "type": "causal_window_mlp",
            "input_dim": model.input_dim,
            "window": model.window,
            "hidden_dims": list(model.hidden_dims),
            "dropout": model.dropout,
        }
    if isinstance(model, value_models.RecurrentValue):
        return {
            "type": "recurrent_value",
            "cell": model.cell,
            "input_dim": model.input_dim,
            "hidden_size": model.hidden_size,
            "head_dim": model.head_dim,
            "num_layers": model.num_layers,
            "dropout": model.dropout,
            "bidirectional": False,
        }
    raise TypeError(f"unsupported temporal model type: {type(model).__name__}")


def _update_mapping(data: LabData, row_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_ids = np.asarray(row_ids, dtype=np.int64)
    starts = data.blocks["row_start"].astype(np.int64, copy=False)
    lengths = data.blocks["row_len"].astype(np.int64, copy=False)
    block_ids = np.searchsorted(starts, row_ids, side="right") - 1
    if np.any(block_ids < 0) or np.any(row_ids >= starts[block_ids] + lengths[block_ids]):
        raise ValueError("temporal target row does not map to a canonical actor block")
    if not bool(data.meta["is_target_eligible"][row_ids].all()):
        raise ValueError("temporal target stream contains an excluded target row")
    unique_blocks, batch_index = np.unique(block_ids, return_inverse=True)
    local_index = row_ids - starts[block_ids]

    return unique_blocks.astype(np.int64), batch_index.astype(np.int64), local_index.astype(np.int64)


def load_sequence_update(
    data: LabData,
    row_ids: np.ndarray,
    *,
    feature_start: int,
    feature_stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load complete causal blocks and an exact ordered target-row mapping."""
    block_ids, batch_index, local_index = _update_mapping(data, row_ids)
    starts = data.blocks["row_start"][block_ids].astype(np.int64, copy=False)
    lengths = data.blocks["row_len"][block_ids].astype(np.int64, copy=False)
    horizon = int(lengths.max())
    input_dim = feature_stop - feature_start
    x = np.zeros((len(block_ids), horizon, input_dim), dtype=np.float32)
    sequence_mask = np.zeros((len(block_ids), horizon), dtype=np.bool_)
    for batch_row, (start, length) in enumerate(zip(starts, lengths)):
        stop = int(start + length)
        x[batch_row, :length] = data.X[int(start):stop, feature_start:feature_stop]
        sequence_mask[batch_row, :length] = True
    z = data.blocks["z"][block_ids].astype(np.float32, copy=False)
    gathered_z = z[batch_index]
    if not np.array_equal(gathered_z, data.meta["z"][row_ids]):
        raise ValueError("temporal actor-relative target disagrees with its sequence block")

    return x, sequence_mask, z, batch_index, local_index


def _nonfinite_gradient_count(model: torch.nn.Module) -> int:
    return sum(
        int((~torch.isfinite(parameter.grad)).sum().item())
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def train_temporal(
    data: LabData,
    spec: TemporalSpec,
    row_stream: np.ndarray,
    *,
    seed: int,
    target_batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if spec.arch == "s1":
        snapshot_spec = SnapshotSpec(
            name="s1_default",
            feature_start=spec.feature_start,
            feature_stop=spec.feature_stop,
            variant="default",
            role=spec.role,
        )
        model, receipt = train_snapshot(
            data,
            snapshot_spec,
            row_stream,
            seed=seed,
            batch_size=target_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        receipt["context_rows_loaded"] = int(len(row_stream))
        receipt["padded_context_positions"] = int(len(row_stream))
        receipt["context_efficiency"] = 1.0
        receipt["sequence_blocks_replayed"] = "not_applicable_snapshot"

        return model, receipt
    torch.manual_seed(seed)
    model = build_temporal_model(spec)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train()
    process = psutil.Process()
    cpu_start = process.cpu_times()
    weighted_loss = 0.0
    updates = 0
    data_loader_seconds = 0.0
    optimizer_compute_seconds = 0.0
    context_rows_loaded = 0
    padded_context_positions = 0
    sequence_blocks_replayed = 0
    peak_batch_bytes = 0
    nonfinite_counts = {
        "features": 0,
        "targets": 0,
        "logits": 0,
        "losses": 0,
        "gradients": 0,
    }
    started = time.perf_counter()
    for start in range(0, len(row_stream), target_batch_size):
        rows = row_stream[start:start + target_batch_size]
        loader_started = time.perf_counter()
        x, sequence_mask, z, batch_index, local_index = load_sequence_update(
            data,
            rows,
            feature_start=spec.feature_start,
            feature_stop=spec.feature_stop,
        )
        x_tensor = torch.from_numpy(x)
        sequence_tensor = torch.from_numpy(sequence_mask)
        z_tensor = torch.from_numpy(z)
        selected_batch = torch.from_numpy(batch_index)
        selected_time = torch.from_numpy(local_index)
        data_loader_seconds += time.perf_counter() - loader_started
        context_rows_loaded += int(sequence_mask.sum())
        padded_context_positions += int(sequence_mask.size)
        sequence_blocks_replayed += len(z)
        peak_batch_bytes = max(
            peak_batch_bytes,
            x.nbytes + sequence_mask.nbytes + z.nbytes + batch_index.nbytes + local_index.nbytes,
        )
        nonfinite_counts["features"] += int((~torch.isfinite(x_tensor)).sum().item())
        compute_started = time.perf_counter()
        logits = model(x_tensor, sequence_tensor)
        targets = mc_targets(z_tensor, sequence_tensor)
        selected_logits = logits[selected_batch, selected_time]
        selected_targets = targets[selected_batch, selected_time]
        if len(selected_logits) != len(rows):
            raise AssertionError("temporal update target count drifted")
        nonfinite_counts["targets"] += int((~torch.isfinite(selected_targets)).sum().item())
        nonfinite_counts["logits"] += int((~torch.isfinite(selected_logits)).sum().item())
        loss = torch.nn.functional.binary_cross_entropy_with_logits(selected_logits, selected_targets)
        nonfinite_counts["losses"] += int(not bool(torch.isfinite(loss).item()))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nonfinite_counts["gradients"] += _nonfinite_gradient_count(model)
        if any(nonfinite_counts.values()):
            raise FloatingPointError(f"non-finite temporal training values: {nonfinite_counts}")
        optimizer.step()
        optimizer_compute_seconds += time.perf_counter() - compute_started
        weighted_loss += float(loss.item()) * len(rows)
        updates += 1
    elapsed = time.perf_counter() - started
    cpu_end = process.cpu_times()
    cpu_seconds = (cpu_end.user + cpu_end.system) - (cpu_start.user + cpu_start.system)

    return model, {
        "presentations": int(len(row_stream)),
        "updates": updates,
        "target_batch_size": target_batch_size,
        "mean_training_loss": weighted_loss / len(row_stream),
        "elapsed_seconds": elapsed,
        "rows_per_second": len(row_stream) / max(elapsed, 1e-12),
        "data_loader_seconds": data_loader_seconds,
        "optimizer_compute_seconds": optimizer_compute_seconds,
        "process_cpu_seconds": cpu_seconds,
        "process_cpu_percent_of_one_core": 100.0 * cpu_seconds / max(elapsed, 1e-12),
        "context_rows_loaded": context_rows_loaded,
        "padded_context_positions": padded_context_positions,
        "context_efficiency": context_rows_loaded / padded_context_positions,
        "sequence_blocks_replayed": sequence_blocks_replayed,
        "peak_batch_bytes": peak_batch_bytes,
        "nonfinite_counts": nonfinite_counts,
    }


def predict_temporal_validation(
    data: LabData,
    model: torch.nn.Module,
    spec: TemporalSpec,
    index: dict[str, np.ndarray],
    *,
    blocks_per_batch: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if spec.arch == "s1":
        snapshot_spec = SnapshotSpec(
            name="s1_default",
            feature_start=spec.feature_start,
            feature_stop=spec.feature_stop,
            variant="default",
            role=spec.role,
        )

        return _predict(data, model, snapshot_spec, index["row_id"], batch_size=DEFAULT_BATCH_SIZE)
    validation_blocks = data.block_ids("validation").astype(np.int64, copy=False)
    predictions = np.empty(len(index["row_id"]), dtype=np.float32)
    filled = np.zeros(len(predictions), dtype=np.bool_)
    model.eval()
    data_loader_seconds = 0.0
    model_compute_seconds = 0.0
    context_rows = 0
    padded_positions = 0
    nonfinite_counts = {"features": 0, "logits": 0, "predictions": 0}
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(validation_blocks), blocks_per_batch):
            block_ids = validation_blocks[start:start + blocks_per_batch]
            row_chunks = []
            for block_id in block_ids:
                row_start = int(data.blocks["row_start"][block_id])
                row_len = int(data.blocks["row_len"][block_id])
                eligible = np.flatnonzero(
                    data.meta["is_target_eligible"][row_start:row_start + row_len]
                )
                if len(eligible):
                    row_chunks.append(row_start + eligible.astype(np.int64, copy=False))
            if not row_chunks:
                continue
            rows = np.concatenate(row_chunks)
            loader_started = time.perf_counter()
            x, sequence_mask, _, batch_index, local_index = load_sequence_update(
                data,
                rows,
                feature_start=spec.feature_start,
                feature_stop=spec.feature_stop,
            )
            x_tensor = torch.from_numpy(x)
            sequence_tensor = torch.from_numpy(sequence_mask)
            selected_batch = torch.from_numpy(batch_index)
            selected_time = torch.from_numpy(local_index)
            data_loader_seconds += time.perf_counter() - loader_started
            context_rows += int(sequence_mask.sum())
            padded_positions += int(sequence_mask.size)
            nonfinite_counts["features"] += int((~torch.isfinite(x_tensor)).sum().item())
            compute_started = time.perf_counter()
            logits = model(x_tensor, sequence_tensor)[selected_batch, selected_time]
            probabilities = torch.sigmoid(logits)
            model_compute_seconds += time.perf_counter() - compute_started
            nonfinite_counts["logits"] += int((~torch.isfinite(logits)).sum().item())
            nonfinite_counts["predictions"] += int((~torch.isfinite(probabilities)).sum().item())
            positions = np.searchsorted(index["row_id"], rows)
            if not np.array_equal(index["row_id"][positions], rows) or bool(filled[positions].any()):
                raise ValueError("temporal validation predictions are misaligned or duplicated")
            predictions[positions] = probabilities.cpu().numpy()
            filled[positions] = True
    if not bool(filled.all()):
        raise ValueError(f"temporal validation left {int((~filled).sum())} rows unpredicted")
    if any(nonfinite_counts.values()):
        raise FloatingPointError(f"non-finite temporal validation values: {nonfinite_counts}")
    elapsed = time.perf_counter() - started

    return predictions, {
        "elapsed_seconds": elapsed,
        "data_loader_seconds": data_loader_seconds,
        "model_compute_seconds": model_compute_seconds,
        "rows_per_second": len(predictions) / max(elapsed, 1e-12),
        "latency_microseconds_per_row": 1e6 * elapsed / len(predictions),
        "context_rows_loaded": context_rows,
        "padded_context_positions": padded_positions,
        "context_efficiency": context_rows / padded_positions,
        "blocks_per_batch": blocks_per_batch,
        "nonfinite_counts": nonfinite_counts,
    }


def _named_slice(
    p: np.ndarray,
    z: np.ndarray,
    keys: np.ndarray,
    names: list[str],
    *,
    min_n: int = 500,
) -> dict[str, dict]:
    indexed = slice_metrics(p, z, keys, min_n=min_n)

    return {
        names[int(key)] if 0 <= int(key) < len(names) else f"unknown:{key}": value
        for key, value in indexed.items()
    }


def evaluate_temporal_predictions(
    p: np.ndarray,
    index: dict[str, np.ndarray],
    data: LabData,
    reference_p: np.ndarray,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    metrics = evaluate_predictions(
        p,
        index,
        data.source_ids,
        reference_p,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    z = index["z"]
    metrics["by_policy"] = _named_slice(p, z, index["policy_gidx"], data.policy_vocab)
    metrics["by_opponent_policy"] = _named_slice(
        p, z, index["opp_policy_gidx"], data.policy_vocab,
    )
    metrics["by_deck"] = _named_slice(p, z, index["deck_gidx"], data.deck_vocab)
    metrics["by_opponent_deck"] = _named_slice(
        p, z, index["opp_deck_gidx"], data.deck_vocab,
    )
    metrics["control_pass"] = "not_instrumented_cache_has_no_control_pass_row_mask"

    return metrics


def temporal_validation_index(data: LabData) -> dict[str, np.ndarray]:
    index = validation_index(data)
    block_ids = index["block_id"].astype(np.int64, copy=False)
    for key in ("policy_gidx", "opp_policy_gidx", "deck_gidx", "opp_deck_gidx"):
        index[key] = data.blocks[key][block_ids].astype(np.int32, copy=False)

    return index


def _aggregate_seeds(
    results: dict[str, Any],
    specs: tuple[TemporalSpec, ...],
    predictions: dict[str, list[np.ndarray]],
    index: dict[str, np.ndarray],
    *,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    names = ["s0", *[spec.name for spec in specs]]
    aggregate: dict[str, Any] = {
        "seed_count": len(results),
        "models": {},
        "direct_comparisons": {},
        "ensemble_mean_comparisons": {},
    }
    for name in names:
        metrics = {}
        for metric in ("log_loss", "brier", "ece", "auc"):
            observed = np.asarray([
                (result["s0"] if name == "s0" else result["models"][name])["metrics"]["overall"][metric]
                for result in results.values()
            ], dtype=np.float64)
            finite = observed[np.isfinite(observed)]
            metrics[metric] = {
                "mean": float(finite.mean()) if len(finite) else float("nan"),
                "se": (
                    float(finite.std(ddof=1) / math.sqrt(len(finite)))
                    if len(finite) > 1 else float("nan")
                ),
                "finite_seeds": int(len(finite)),
            }
        aggregate["models"][name] = metrics
    first = next(iter(results.values()))
    for comparison_name in first["direct_comparisons"]:
        values = {}
        for metric in ("log_loss", "brier"):
            observed = np.asarray([
                result["direct_comparisons"][comparison_name][metric]["delta_candidate_minus_reference"]
                for result in results.values()
            ])
            values[metric] = {
                "mean_delta_candidate_minus_reference": float(observed.mean()),
                "se_across_seeds": (
                    float(observed.std(ddof=1) / math.sqrt(len(observed)))
                    if len(observed) > 1 else float("nan")
                ),
                "per_seed": observed.tolist(),
            }
        aggregate["direct_comparisons"][comparison_name] = values
    reference_name = specs[0].name
    reference = np.mean(np.stack(predictions[reference_name]), axis=0)
    for spec in specs[1:]:
        candidate = np.mean(np.stack(predictions[spec.name]), axis=0)
        aggregate["ensemble_mean_comparisons"][f"{spec.name}_vs_{reference_name}"] = direct_comparison(
            candidate,
            reference,
            index,
            candidate=spec.name,
            reference=reference_name,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=DEFAULT_SEED,
        )

    return aggregate


def decide_temporal_gate(
    results: dict[str, Any],
    aggregate: dict[str, Any],
    specs: tuple[TemporalSpec, ...],
    *,
    rung: str,
) -> dict[str, Any]:
    ranked = sorted(
        (spec.name for spec in specs),
        key=lambda name: aggregate["models"][name]["log_loss"]["mean"],
    )
    reference = specs[0].name
    if len(results) < 2:
        return {
            "status": "plumbing_only_insufficient_seeds",
            "history_helps": "untested_at_required_two_seed_gate",
            "ranked_models": ranked,
            "next_models": ranked[:2],
        }
    history = {}
    for spec in specs[1:]:
        name = spec.name
        comparison_name = f"{name}_vs_{reference}"
        per_seed_ll = aggregate["direct_comparisons"][comparison_name]["log_loss"]["per_seed"]
        ensemble = aggregate["ensemble_mean_comparisons"][comparison_name]
        calibration_ok = all(
            result["models"][name]["metrics"]["overall"]["calibration_fit_status"] == "converged"
            and result["models"][name]["metrics"]["overall"]["slope"] > 0
            for result in results.values()
        )
        history[name] = {
            "both_seed_log_loss_deltas_negative": all(delta < 0 for delta in per_seed_ll),
            "ensemble_log_loss_ci_upper_below_zero": ensemble["log_loss"]["ci95"][1] < 0,
            "ensemble_brier_not_worse": ensemble["brier"]["delta_candidate_minus_reference"] <= 0,
            "calibration_identifiable_positive": calibration_ok,
        }
        history[name]["passes_history_gate"] = all(history[name].values())

    return {
        "status": (
            "serious_25pct_complete" if rung == "serious_25pct" else "ten_percent_screen_complete"
        ),
        "history_helps": any(value["passes_history_gate"] for value in history.values()),
        "ranked_models": ranked,
        "next_models": ranked[:2],
        "history_candidates": history,
        "selection_metric": "mean_raw_validation_log_loss",
    }


def _expected_output_paths(config: dict[str, Any]) -> set[str]:
    expected = {"validation_index.npz", "result.json"}
    names = [spec["name"] for spec in config["models"]]
    for seed in config["seeds"]:
        prefix = f"seed_{seed}"
        expected.update({
            f"{prefix}/train_row_stream.npz",
            f"{prefix}/s0_predictions.npz",
            f"{prefix}/result.json",
        })
        for name in names:
            expected.add(f"{prefix}/{name}.pt")
            expected.add(f"{prefix}/{name}.predictions.npz")

    return expected


def _verify_completed(out_dir: Path, config: dict[str, Any], config_sha: str) -> dict | None:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = read_manifest(manifest_path)
    inputs = manifest.get("inputs") or {}
    if (
        manifest.get("kind") != "value_mc_temporal_screen"
        or manifest.get("status") != "completed"
        or inputs.get("config_sha256") != config_sha
        or inputs.get("config") != config
        or _stable_hash(inputs.get("config")) != config_sha
    ):
        raise ValueError("existing temporal screen does not match the requested completed run")
    receipts = (manifest.get("outputs") or {}).get("files")
    if not isinstance(receipts, list) or not receipts:
        raise ValueError("existing temporal screen lacks output receipts")
    if {receipt.get("path") for receipt in receipts} != _expected_output_paths(config):
        raise ValueError("existing temporal screen output inventory is incomplete")
    resolved_root = out_dir.resolve()
    for receipt in receipts:
        path = (out_dir / receipt["path"]).resolve()
        if not path.is_relative_to(resolved_root):
            raise ValueError("temporal output receipt escapes its run directory")
        if (
            not path.is_file()
            or path.stat().st_size != int(receipt["bytes"])
            or _sha256(path) != receipt["sha256"]
        ):
            raise ValueError(f"temporal output hash mismatch: {path}")

    return manifest


def _assert_v0_parity(
    model: torch.nn.Module,
    predictions: np.ndarray,
    row_stream_sha: str,
    checkpoint_path: Path,
    predictions_path: Path,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("row_stream_sha256") != row_stream_sha:
        raise ValueError("V0 parity reference used a different row stream")
    reference_state = checkpoint.get("model_state") or {}
    if set(reference_state) != set(model.state_dict()):
        raise ValueError("V0 parity reference model-state inventory differs")
    if any(not torch.equal(value, reference_state[name]) for name, value in model.state_dict().items()):
        raise ValueError("V0 snapshot tensor state did not reproduce exactly")
    with np.load(predictions_path, allow_pickle=False) as saved:
        reference_predictions = saved["p"]
    if not np.array_equal(predictions, reference_predictions):
        raise ValueError("V0 validation predictions did not reproduce exactly")

    return {
        "status": "exact",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "predictions_path": str(predictions_path),
        "predictions_sha256": _sha256(predictions_path),
        "row_stream_sha256": row_stream_sha,
    }


def run_temporal_screen(
    *,
    cache_dir: Path,
    split_path: Path,
    out_dir: Path,
    rung: str,
    presentations: int | None = None,
    seeds: tuple[int, ...] = (DEFAULT_SEED,),
    target_batch_size: int = DEFAULT_BATCH_SIZE,
    validation_blocks_per_batch: int = DEFAULT_SEQUENCE_BLOCKS_PER_BATCH,
    threads: int = 8,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    bootstrap_replicates: int = 500,
    parameter_budget: int = DEFAULT_PARAMETER_BUDGET,
    window: int = DEFAULT_WINDOW,
    v0_reference_checkpoint: Path | None = None,
    v0_reference_predictions: Path | None = None,
    model_names: tuple[str, ...] | None = None,
    command: list[str] | None = None,
) -> dict[str, Any]:
    if rung not in RUNG_TARGET_ROWS:
        raise ValueError(f"rung must be one of {sorted(RUNG_TARGET_ROWS)}")
    row_budget = int(presentations if presentations is not None else RUNG_TARGET_ROWS[rung])
    if row_budget < 1 or not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("presentations and unique nonnegative seeds are required")
    if rung == "ten_percent" and len(seeds) < 2:
        raise ValueError("ten_percent requires at least two seeds")
    if target_batch_size < 1 or validation_blocks_per_batch < 1 or threads < 1:
        raise ValueError("batch sizes and threads must be positive")
    if bootstrap_replicates < 2 or parameter_budget < 1 or not 4 <= window <= 8:
        raise ValueError("bootstrap, parameter budget, or causal window is invalid")
    if not math.isfinite(learning_rate) or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("optimizer configuration is invalid")
    if (v0_reference_checkpoint is None) != (v0_reference_predictions is None):
        raise ValueError("V0 parity requires both checkpoint and prediction references")
    cache_dir = cache_dir.resolve()
    split_path = split_path.resolve()
    out_dir = out_dir.resolve()
    reference_checkpoint = v0_reference_checkpoint.resolve() if v0_reference_checkpoint else None
    reference_predictions = v0_reference_predictions.resolve() if v0_reference_predictions else None
    data = LabData(str(cache_dir), split_manifest_path=str(split_path))
    cache_manifest_path = cache_dir / "cache_manifest.json"
    all_specs = temporal_specs(parameter_budget=parameter_budget, window=window)
    if model_names is None:
        specs = all_specs
    else:
        by_name = {spec.name: spec for spec in all_specs}
        if len(set(model_names)) != len(model_names) or any(name not in by_name for name in model_names):
            raise ValueError("model_names must be unique accepted temporal model names")
        specs = tuple(by_name[name] for name in model_names)
    if not specs or specs[0].arch != "s1":
        raise ValueError("temporal screens require V0 snapshot as the first model")
    if rung in {"smoke", "ten_percent"} and specs != all_specs:
        raise ValueError(f"{rung} requires the complete primary V0/V1/V2 comparison")
    if rung == "serious_25pct" and len(specs) != 2:
        raise ValueError("serious_25pct requires exactly the selected top two models")
    config = {
        "schema_version": TEMPORAL_SCREEN_SCHEMA_VERSION,
        "rung": rung,
        "presentations": row_budget,
        "seeds": list(seeds),
        "target_batch_size": target_batch_size,
        "validation_blocks_per_batch": validation_blocks_per_batch,
        "threads": threads,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "bootstrap_replicates": bootstrap_replicates,
        "parameter_budget": parameter_budget,
        "window": window,
        "target": "mc_actor_relative_terminal",
        "feature_surface": "s1_actor_visible_prefix_0_49",
        "models": [asdict(spec) for spec in specs],
        "gru": "not_run_default_off_cpu_cost_would_delay_lstm",
        "dense144": "quarantined_after_negative_feature_contribution_smoke",
        "cache_manifest_sha256": _sha256(cache_manifest_path),
        "split_manifest_sha256": _sha256(split_path),
        "v0_reference": (
            {
                "checkpoint": str(reference_checkpoint),
                "checkpoint_sha256": _sha256(reference_checkpoint),
                "predictions": str(reference_predictions),
                "predictions_sha256": _sha256(reference_predictions),
            }
            if reference_checkpoint else None
        ),
        "code_sha256": {
            "temporal_screen": _sha256(Path(__file__)),
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
            "gpu_utilization": "not_instrumented_cpu_only_torch_build",
        },
    }
    config_sha = _stable_hash(config)
    if out_dir.exists():
        completed = _verify_completed(out_dir, config, config_sha)
        if completed is not None:
            return {"action": "reused_verified", "manifest": completed}
        preview = ", ".join(sorted(path.name for path in out_dir.iterdir())[:10]) or "empty"
        raise RuntimeError(f"refusing to overwrite incomplete temporal screen: {out_dir} ({preview})")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = out_dir.with_name(f".{out_dir.name}.running.{uuid.uuid4().hex}")
    temp_dir.mkdir()
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    index = temporal_validation_index(data)
    index_path = temp_dir / "validation_index.npz"
    np.savez_compressed(index_path, **index)
    results: dict[str, Any] = {}
    predictions_across_seeds: dict[str, list[np.ndarray]] = {
        spec.name: [] for spec in specs
    }
    output_paths = [index_path]
    parity_receipts = []
    for seed in seeds:
        row_stream = eligible_row_stream(data, split="train", presentations=row_budget, seed=seed)
        stream_receipt = row_stream_receipt(data, row_stream)
        stream_sha = stream_receipt["sha256"]
        seed_dir = temp_dir / f"seed_{seed}"
        seed_dir.mkdir()
        stream_path = seed_dir / "train_row_stream.npz"
        np.savez_compressed(stream_path, row_id=row_stream.astype(np.int64, copy=False))
        train_labels = data.meta["z"][row_stream].astype(np.float64, copy=False)
        base_rate = float(train_labels.mean())
        s0_predictions = np.full(len(index["z"]), base_rate, dtype=np.float32)
        s0_path = seed_dir / "s0_predictions.npz"
        np.savez_compressed(s0_path, p=s0_predictions)
        s0_metrics = evaluate_temporal_predictions(
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
            "row_stream_sha256": stream_sha,
            "row_stream": stream_receipt,
            "s0": {"base_rate": base_rate, "metrics": s0_metrics},
            "models": {},
        }
        output_paths.extend([stream_path, s0_path])
        prediction_by_name = {"s0": s0_predictions}
        for spec in specs:
            model, train_receipt = train_temporal(
                data,
                spec,
                row_stream,
                seed=seed,
                target_batch_size=target_batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )
            predictions, inference_receipt = predict_temporal_validation(
                data,
                model,
                spec,
                index,
                blocks_per_batch=validation_blocks_per_batch,
            )
            metrics = evaluate_temporal_predictions(
                predictions,
                index,
                data,
                s0_predictions,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=seed,
            )
            if spec.arch == "s1" and reference_checkpoint is not None:
                parity_receipts.append(_assert_v0_parity(
                    model,
                    predictions,
                    stream_sha,
                    reference_checkpoint,
                    reference_predictions,
                ))
            checkpoint_path = seed_dir / f"{spec.name}.pt"
            checkpoint = {
                "schema_version": TEMPORAL_CHECKPOINT_SCHEMA_VERSION,
                "spec": asdict(spec),
                "model_config": _model_config(model),
                "seed": seed,
                "row_stream_sha256": stream_sha,
                "cache_manifest_sha256": config["cache_manifest_sha256"],
                "split_manifest_sha256": config["split_manifest_sha256"],
                "outer_config_sha256": config_sha,
                "code_sha256": config["code_sha256"],
                "parameter_count": parameter_count(model),
                "model_state": model.state_dict(),
                "optimizer": {
                    "name": "AdamW",
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "target_batch_size": target_batch_size,
                },
                "train": train_receipt,
                "validation": metrics["overall"],
            }
            _atomic_torch_save(checkpoint_path, checkpoint)
            prediction_path = seed_dir / f"{spec.name}.predictions.npz"
            np.savez_compressed(prediction_path, p=predictions)
            output_paths.extend([checkpoint_path, prediction_path])
            prediction_by_name[spec.name] = predictions
            predictions_across_seeds[spec.name].append(predictions)
            seed_result["models"][spec.name] = {
                "spec": asdict(spec),
                "parameter_count": parameter_count(model),
                "parameter_bytes_float32": parameter_count(model) * 4,
                "train": train_receipt,
                "inference": inference_receipt,
                "metrics": metrics,
            }
        reference_name = specs[0].name
        pairs = tuple((spec.name, reference_name) for spec in specs[1:])
        if len(specs) >= 3:
            pairs += ((specs[2].name, specs[1].name),)
        seed_result["direct_comparisons"] = {
            f"{candidate}_vs_{reference}": direct_comparison(
                prediction_by_name[candidate],
                prediction_by_name[reference],
                index,
                candidate=candidate,
                reference=reference,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=seed,
            )
            for candidate, reference in pairs
        }
        results[str(seed)] = seed_result
        seed_result_path = seed_dir / "result.json"
        write_json(seed_result_path, _json_safe(seed_result))
        output_paths.append(seed_result_path)
    aggregate = _aggregate_seeds(
        results,
        specs,
        predictions_across_seeds,
        index,
        bootstrap_replicates=bootstrap_replicates,
    )
    gate = decide_temporal_gate(results, aggregate, specs, rung=rung)
    reference_name = specs[0].name
    for spec in specs[1:]:
        name = spec.name
        gain = (
            aggregate["models"][reference_name]["log_loss"]["mean"]
            - aggregate["models"][name]["log_loss"]["mean"]
        )
        train_seconds = float(np.mean([
            result["models"][name]["train"]["elapsed_seconds"]
            for result in results.values()
        ]))
        gate.setdefault("marginal_value_per_second", {})[name] = {
            "validation_log_loss_gain_vs_v0": gain,
            "mean_candidate_train_seconds": train_seconds,
            "gain_per_candidate_train_second": gain / max(train_seconds, 1e-12),
        }
    result_path = temp_dir / "result.json"
    write_json(result_path, _json_safe({
        "config": config,
        "config_sha256": config_sha,
        "seeds": results,
        "aggregate": aggregate,
        "gate": gate,
        "v0_parity": parity_receipts,
    }))
    output_paths.append(result_path)
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
        kind="value_mc_temporal_screen",
        command=command or [],
        inputs={
            "config": config,
            "config_sha256": config_sha,
            "cache_dir": str(cache_dir),
            "split_manifest_path": str(split_path),
            "temporal_screen_code_sha256": _sha256(Path(__file__)),
        },
        outputs={"files": files},
        summary={
            "status": "screening_only",
            "rung": rung,
            "seeds": list(seeds),
            "presentations_per_model_seed": row_budget,
            "validation_rows": int(len(index["z"])),
            "aggregate": _json_safe(aggregate),
            "gate": _json_safe(gate),
            "v0_parity": parity_receipts,
            "nonfinite_counts": nonfinite,
            "gpu_utilization": config["environment"]["gpu_utilization"],
            "calibration_targets_consumed": False,
            "test_targets_consumed": False,
            "control_pass": "not_instrumented_cache_has_no_control_pass_row_mask",
            "gru": config["gru"],
            "dense144": config["dense144"],
            "offline_gameplay_safety_monitors": "not_applicable_no_gameplay_executed",
            "midrun_resume": "not_implemented_for_bounded_temporal_rungs",
        },
    )
    write_manifest(temp_dir / "manifest.json", manifest)
    os.replace(temp_dir, out_dir)

    return {"action": "completed", "manifest": manifest, "results": results, "gate": gate}
