"""Deterministic Monte Carlo snapshot screening on the accepted value cache."""

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
import torch

import deepmew.value_lab.data as value_data
import deepmew.value_lab.metrics as value_metrics
import deepmew.value_lab.models as value_models
from deepmew.artifacts.manifest import build_manifest, read_manifest, write_manifest
from deepmew.artifacts.writers import write_json
from deepmew.replay.canonical_actor_dataset import FEATURE_DIM, S1_SLICE
from deepmew.value_lab.data import LabData
from deepmew.value_lab.metrics import (
    full_metrics,
    paired_game_cluster_bootstrap,
    slice_metrics,
)
from deepmew.value_lab.models import (
    SnapshotMLP,
    build_model,
    matched_model_config,
    parameter_count,
)


MC_SCREEN_SCHEMA_VERSION = "deepmew.value_mc_snapshot_screen.v1"
CHECKPOINT_SCHEMA_VERSION = "deepmew.value_mc_checkpoint.v1"
RUNG_TARGET_ROWS = {"smoke": 65_536, "screen": 1_048_576}
DEFAULT_PARAMETER_BUDGET = 100_000
DEFAULT_SEED = 915_001
DEFAULT_BATCH_SIZE = 4_096


@dataclass(frozen=True)
class SnapshotSpec:
    name: str
    feature_start: int
    feature_stop: int
    variant: str
    role: str


def snapshot_specs() -> tuple[SnapshotSpec, ...]:
    return (
        SnapshotSpec("s1_default", S1_SLICE[0], S1_SLICE[1], "default", "existing_value_mlp_baseline"),
        SnapshotSpec("s1_matched", S1_SLICE[0], S1_SLICE[1], "matched", "capacity_control_baseline"),
        SnapshotSpec("s2_dense144_default", 0, FEATURE_DIM, "default", "baseline_plus_features_width_matched"),
        SnapshotSpec("s2_dense144_matched", 0, FEATURE_DIM, "matched", "strict_feature_contribution"),
        SnapshotSpec("s2_extra_only_matched", S1_SLICE[1], FEATURE_DIM, "matched", "feature_only_diagnostic"),
    )


def _sha256(path: Path, chunk_size: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)

    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)

        return number if np.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)

    return value


def canonical_game_id(block_key: str, seat: int | None = None) -> str:
    if not isinstance(block_key, str) or ":s" not in block_key:
        raise ValueError(f"block key lacks canonical seat suffix: {block_key!r}")
    game_id, raw_seat = block_key.rsplit(":s", 1)
    if not game_id or raw_seat not in {"0", "1"}:
        raise ValueError(f"block key has invalid canonical seat suffix: {block_key!r}")
    if seat is not None and int(raw_seat) != int(seat):
        raise ValueError(f"block key seat disagrees with block metadata: {block_key!r}")

    return game_id


def canonical_game_indices(data: LabData) -> tuple[np.ndarray, list[str]]:
    game_to_index: dict[str, int] = {}
    game_ids: list[str] = []
    block_game = np.empty(len(data.block_keys), dtype=np.int32)
    game_sources: dict[str, int] = {}
    game_seats: dict[str, set[int]] = {}
    for block_id, block_key in enumerate(data.block_keys):
        seat = int(data.blocks["seat"][block_id])
        game_id = canonical_game_id(block_key, seat)
        source = int(data.blocks["source_idx"][block_id])
        if game_id in game_sources and game_sources[game_id] != source:
            raise ValueError(f"canonical game spans multiple sources: {game_id}")
        game_sources[game_id] = source
        seats = game_seats.setdefault(game_id, set())
        if seat in seats:
            raise ValueError(f"canonical game repeats seat block: {game_id}:s{seat}")
        seats.add(seat)
        if game_id not in game_to_index:
            game_to_index[game_id] = len(game_ids)
            game_ids.append(game_id)
        block_game[block_id] = game_to_index[game_id]

    return block_game, game_ids


def eligible_row_stream(data: LabData, *, split: str, presentations: int, seed: int) -> np.ndarray:
    if isinstance(presentations, bool) or not isinstance(presentations, int) or presentations < 1:
        raise ValueError("presentations must be a positive integer")
    block_ids = np.asarray(data.block_ids(split), dtype=np.int64)
    if len(block_ids) == 0:
        raise ValueError(f"split {split} contains no actor blocks")
    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    remaining = presentations
    eligible_per_pass = 0
    while remaining:
        order = rng.permutation(block_ids)
        pass_rows = 0
        for block_id in order:
            start = int(data.blocks["row_start"][block_id])
            length = int(data.blocks["row_len"][block_id])
            local_eligible = np.flatnonzero(data.meta["is_target_eligible"][start:start + length])
            if len(local_eligible) == 0:
                continue
            rows = start + local_eligible.astype(np.int64, copy=False)
            take = min(remaining, len(rows))
            chunks.append(rows[:take])
            remaining -= take
            pass_rows += take
            if remaining == 0:
                break
        if eligible_per_pass == 0:
            eligible_per_pass = pass_rows
        if pass_rows == 0:
            raise ValueError(f"split {split} contains no eligible target rows")
    stream = np.concatenate(chunks)
    if len(stream) != presentations:
        raise AssertionError("eligible row stream length drifted")
    if not bool(data.meta["is_target_eligible"][stream].all()):
        raise AssertionError("eligible row stream contains a target-excluded row")

    return stream


def row_stream_sha256(row_stream: np.ndarray) -> str:
    rows = np.asarray(row_stream, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(str(len(rows)).encode("ascii"))
    digest.update(b"\0")
    digest.update(rows.tobytes(order="C"))

    return digest.hexdigest()


def row_stream_receipt(data: LabData, row_stream: np.ndarray) -> dict[str, Any]:
    starts = data.blocks["row_start"].astype(np.int64, copy=False)
    lengths = data.blocks["row_len"].astype(np.int64, copy=False)
    block_ids = np.searchsorted(starts, row_stream, side="right") - 1
    if np.any(block_ids < 0) or np.any(row_stream >= starts[block_ids] + lengths[block_ids]):
        raise ValueError("training row stream does not map to canonical actor blocks")
    block_game, _ = canonical_game_indices(data)
    source_indices, source_counts = np.unique(data.meta["source_idx"][row_stream], return_counts=True)
    seats, seat_counts = np.unique(data.meta["seat"][row_stream], return_counts=True)
    outcomes, outcome_counts = np.unique(data.meta["z"][row_stream], return_counts=True)
    unique_rows = int(len(np.unique(row_stream)))

    return {
        "sha256": row_stream_sha256(row_stream),
        "presentations": int(len(row_stream)),
        "unique_target_rows": unique_rows,
        "duplicate_presentations": int(len(row_stream) - unique_rows),
        "unique_blocks": int(len(np.unique(block_ids))),
        "unique_games": int(len(np.unique(block_game[block_ids]))),
        "rows_by_source": {
            data.source_ids[int(index)]: int(count)
            for index, count in zip(source_indices, source_counts)
        },
        "rows_by_seat": {str(int(seat)): int(count) for seat, count in zip(seats, seat_counts)},
        "rows_by_outcome": {
            str(float(outcome)): int(count)
            for outcome, count in zip(outcomes, outcome_counts)
        },
        "numpy_version": np.__version__,
        "sampling": "seeded_numpy_generator_block_permutation_then_within_block_eligible_rows",
    }


def validation_index(data: LabData) -> dict[str, np.ndarray]:
    validation_idx = data.split_names.index("validation")
    row_ids = np.flatnonzero(
        (data.meta["split_idx"] == validation_idx) & data.meta["is_target_eligible"]
    ).astype(np.int64, copy=False)
    if len(row_ids) == 0:
        raise ValueError("validation split contains no eligible targets")
    starts = data.blocks["row_start"].astype(np.int64, copy=False)
    lengths = data.blocks["row_len"].astype(np.int64, copy=False)
    block_ids = np.searchsorted(starts, row_ids, side="right") - 1
    if np.any(block_ids < 0) or np.any(row_ids >= starts[block_ids] + lengths[block_ids]):
        raise ValueError("validation rows do not map to canonical actor blocks")
    block_game, _ = canonical_game_indices(data)

    return {
        "row_id": row_ids,
        "block_id": block_ids.astype(np.int32),
        "game_gidx": block_game[block_ids],
        "z": data.meta["z"][row_ids].astype(np.float32, copy=False),
        "source_idx": data.meta["source_idx"][row_ids].astype(np.int32, copy=False),
        "seat": data.meta["seat"][row_ids].astype(np.int32, copy=False),
        "turn": data.meta["turn"][row_ids].astype(np.int32, copy=False),
        "legal_count": data.meta["legal_count"][row_ids].astype(np.int32, copy=False),
        "select_context": data.meta["select_context"][row_ids].astype(np.int32, copy=False),
    }


def build_snapshot_model(spec: SnapshotSpec) -> SnapshotMLP:
    input_dim = spec.feature_stop - spec.feature_start
    if input_dim < 1:
        raise ValueError(f"invalid feature slice for {spec.name}")
    if spec.name.startswith("s1_"):
        model, consumed = build_model(
            "s1",
            FEATURE_DIM,
            S1_SLICE[1],
            variant=spec.variant,
            parameter_budget=DEFAULT_PARAMETER_BUDGET if spec.variant == "matched" else None,
        )
        if consumed != input_dim or not isinstance(model, SnapshotMLP):
            raise AssertionError("S1 model/input contract drifted")

        return model
    if spec.name == "s2_extra_only_matched":
        config = matched_model_config(
            "s2",
            input_dim,
            min(S1_SLICE[1], input_dim),
            DEFAULT_PARAMETER_BUDGET,
        )
        return SnapshotMLP(input_dim, config.hidden_dims, config.dropout)
    model, consumed = build_model(
        "s2",
        FEATURE_DIM,
        S1_SLICE[1],
        variant=spec.variant,
        parameter_budget=DEFAULT_PARAMETER_BUDGET if spec.variant == "matched" else None,
    )
    if consumed != input_dim or not isinstance(model, SnapshotMLP):
        raise AssertionError("S2 model/input contract drifted")

    return model


def _feature_batch(data: LabData, rows: np.ndarray, spec: SnapshotSpec) -> torch.Tensor:
    values = np.asarray(data.X[rows, spec.feature_start:spec.feature_stop], dtype=np.float32)

    return torch.from_numpy(values)


def _predict(
    data: LabData,
    model: SnapshotMLP,
    spec: SnapshotSpec,
    row_ids: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    model.eval()
    predictions = np.empty(len(row_ids), dtype=np.float32)
    started = time.perf_counter()
    data_loader_seconds = 0.0
    model_compute_seconds = 0.0
    nonfinite_features = 0
    nonfinite_logits = 0
    nonfinite_predictions = 0
    with torch.no_grad():
        for start in range(0, len(row_ids), batch_size):
            stop = min(start + batch_size, len(row_ids))
            loader_started = time.perf_counter()
            features = _feature_batch(data, row_ids[start:stop], spec)
            sequence_mask = torch.ones((len(features), 1), dtype=torch.bool)
            data_loader_seconds += time.perf_counter() - loader_started
            nonfinite_features += int((~torch.isfinite(features)).sum().item())
            compute_started = time.perf_counter()
            logits = model(features[:, None, :], sequence_mask)[:, 0]
            probabilities = torch.sigmoid(logits)
            model_compute_seconds += time.perf_counter() - compute_started
            nonfinite_logits += int((~torch.isfinite(logits)).sum().item())
            nonfinite_predictions += int((~torch.isfinite(probabilities)).sum().item())
            predictions[start:stop] = probabilities.cpu().numpy()
    elapsed = time.perf_counter() - started
    nonfinite_counts = {
        "features": nonfinite_features,
        "logits": nonfinite_logits,
        "predictions": nonfinite_predictions,
    }
    if any(nonfinite_counts.values()):
        raise FloatingPointError(f"non-finite snapshot inference values: {nonfinite_counts}")

    return predictions, {
        "elapsed_seconds": elapsed,
        "data_loader_seconds": data_loader_seconds,
        "model_compute_seconds": model_compute_seconds,
        "rows_per_second": len(predictions) / max(elapsed, 1e-12),
        "latency_microseconds_per_row": 1e6 * elapsed / len(predictions),
        "nonfinite_counts": nonfinite_counts,
    }


def train_snapshot(
    data: LabData,
    spec: SnapshotSpec,
    row_stream: np.ndarray,
    *,
    seed: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[SnapshotMLP, dict[str, Any]]:
    torch.manual_seed(seed)
    model = build_snapshot_model(spec)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train()
    weighted_loss = 0.0
    updates = 0
    data_loader_seconds = 0.0
    optimizer_compute_seconds = 0.0
    nonfinite_features = 0
    nonfinite_logits = 0
    nonfinite_losses = 0
    nonfinite_gradients = 0
    started = time.perf_counter()
    for start in range(0, len(row_stream), batch_size):
        rows = row_stream[start:start + batch_size]
        loader_started = time.perf_counter()
        features = _feature_batch(data, rows, spec)
        labels = torch.from_numpy(data.meta["z"][rows].astype(np.float32, copy=False))
        sequence_mask = torch.ones((len(rows), 1), dtype=torch.bool)
        data_loader_seconds += time.perf_counter() - loader_started
        nonfinite_features += int((~torch.isfinite(features)).sum().item())
        compute_started = time.perf_counter()
        logits = model(features[:, None, :], sequence_mask)[:, 0]
        nonfinite_logits += int((~torch.isfinite(logits)).sum().item())
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        nonfinite_losses += int(not bool(torch.isfinite(loss).item()))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nonfinite_gradients += sum(
            int((~torch.isfinite(parameter.grad)).sum().item())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        if nonfinite_logits or nonfinite_losses or nonfinite_gradients:
            raise FloatingPointError(
                "non-finite snapshot training values: "
                f"logits={nonfinite_logits} losses={nonfinite_losses} "
                f"gradients={nonfinite_gradients}"
            )
        optimizer.step()
        optimizer_compute_seconds += time.perf_counter() - compute_started
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
        "data_loader_seconds": data_loader_seconds,
        "optimizer_compute_seconds": optimizer_compute_seconds,
        "nonfinite_counts": {
            "features": nonfinite_features,
            "logits": nonfinite_logits,
            "losses": nonfinite_losses,
            "gradients": nonfinite_gradients,
        },
    }


def _macro_source(metrics_by_source: dict[str, dict]) -> dict[str, float]:
    if not metrics_by_source:
        raise ValueError("macro-source metrics require at least one source")

    return {
        "log_loss": float(np.mean([value["log_loss"] for value in metrics_by_source.values()])),
        "brier": float(np.mean([value["brier"] for value in metrics_by_source.values()])),
    }


def evaluate_predictions(
    p: np.ndarray,
    index: dict[str, np.ndarray],
    source_ids: list[str],
    reference_p: np.ndarray,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    z = index["z"]
    by_source_index = slice_metrics(p, z, index["source_idx"], min_n=1)
    by_source = {
        source_ids[int(key)]: value
        for key, value in by_source_index.items()
    }

    return {
        "overall": full_metrics(p, z),
        "by_source": by_source,
        "macro_source": _macro_source(by_source),
        "by_seat": slice_metrics(p, z, index["seat"], min_n=1),
        "by_turn": slice_metrics(p, z, index["turn"], min_n=1),
        "by_legal_count": slice_metrics(p, z, index["legal_count"], min_n=1),
        "by_select_context": slice_metrics(p, z, index["select_context"], min_n=1),
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
    }


def direct_comparison(
    candidate_p: np.ndarray,
    reference_p: np.ndarray,
    index: dict[str, np.ndarray],
    *,
    candidate: str,
    reference: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    return {
        "candidate": candidate,
        "reference": reference,
        "delta_definition": "candidate_minus_reference; negative_loss_is_better",
        **{
            metric: paired_game_cluster_bootstrap(
                candidate_p,
                reference_p,
                index["z"],
                index["game_gidx"],
                metric=metric,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            )
            for metric in ("log_loss", "brier")
        },
    }


def aggregate_seeds(results: dict[str, Any]) -> dict[str, Any]:
    names = ["s0", *[spec.name for spec in snapshot_specs()]]
    aggregate: dict[str, Any] = {"seed_count": len(results), "models": {}, "direct_comparisons": {}}
    for name in names:
        values = {}
        for metric in ("log_loss", "brier", "auc", "ece"):
            observed = np.asarray(
                [
                    result["s0"]["metrics"]["overall"][metric]
                    if name == "s0"
                    else result["models"][name]["metrics"]["overall"][metric]
                    for result in results.values()
                ],
                dtype=np.float64,
            )
            finite = observed[np.isfinite(observed)]
            values[metric] = {
                "mean": float(finite.mean()) if len(finite) else float("nan"),
                "se": float(finite.std(ddof=1) / math.sqrt(len(finite))) if len(finite) > 1 else float("nan"),
                "finite_seeds": int(len(finite)),
            }
        aggregate["models"][name] = values
    first_result = next(iter(results.values()))
    for comparison_name in first_result["direct_comparisons"]:
        comparison_values = {}
        for metric in ("log_loss", "brier"):
            observed = np.asarray([
                result["direct_comparisons"][comparison_name][metric][
                    "delta_candidate_minus_reference"
                ]
                for result in results.values()
            ], dtype=np.float64)
            comparison_values[metric] = {
                "mean_delta_candidate_minus_reference": float(observed.mean()),
                "se_across_seeds": (
                    float(observed.std(ddof=1) / math.sqrt(len(observed)))
                    if len(observed) > 1
                    else float("nan")
                ),
                "seeds": int(len(observed)),
            }
        aggregate["direct_comparisons"][comparison_name] = comparison_values

    return aggregate


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def _output_receipt(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _expected_output_paths(config: dict[str, Any]) -> set[str]:
    expected = {"validation_index.npz", "result.json"}
    model_names = [spec["name"] for spec in config["models"]]
    for seed in config["seeds"]:
        prefix = f"seed_{seed}"
        expected.add(f"{prefix}/train_row_stream.npz")
        expected.add(f"{prefix}/s0_predictions.npz")
        expected.add(f"{prefix}/result.json")
        for name in model_names:
            expected.add(f"{prefix}/{name}.pt")
            expected.add(f"{prefix}/{name}.predictions.npz")

    return expected


def _verify_completed(
    out_dir: Path,
    expected_config: dict[str, Any],
    config_sha256: str,
) -> dict[str, Any] | None:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = read_manifest(manifest_path)
    inputs = manifest.get("inputs") or {}
    embedded_config = inputs.get("config")
    if (
        manifest.get("kind") != "value_mc_snapshot_screen"
        or manifest.get("status") != "completed"
        or inputs.get("config_sha256") != config_sha256
        or embedded_config != expected_config
        or _stable_hash(embedded_config) != config_sha256
    ):
        raise ValueError("existing MC screen manifest does not match the requested completed run")
    receipts = (manifest.get("outputs") or {}).get("files")
    if not isinstance(receipts, list) or not receipts:
        raise ValueError("existing MC screen manifest lacks a complete output inventory")
    expected_paths = _expected_output_paths(expected_config)
    receipt_paths = [receipt.get("path") for receipt in receipts if isinstance(receipt, dict)]
    if len(receipt_paths) != len(receipts) or len(set(receipt_paths)) != len(receipts):
        raise ValueError("existing MC screen output receipts are malformed or duplicated")
    if set(receipt_paths) != expected_paths:
        raise ValueError("existing MC screen output inventory is missing or unexpected")
    resolved_root = out_dir.resolve()
    for receipt in receipts:
        relative = Path(receipt["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("existing MC screen output receipt contains an unsafe path")
        path = (out_dir / relative).resolve()
        if not path.is_relative_to(resolved_root):
            raise ValueError("existing MC screen output receipt escapes the run directory")
        if (
            not path.is_file()
            or path.stat().st_size != int(receipt["bytes"])
            or _sha256(path) != receipt["sha256"]
        ):
            raise ValueError(f"existing MC screen output hash mismatch: {path}")

    return manifest


def run_mc_screen(
    *,
    cache_dir: Path,
    split_path: Path,
    out_dir: Path,
    rung: str,
    presentations: int | None = None,
    seeds: tuple[int, ...] = (DEFAULT_SEED,),
    batch_size: int = DEFAULT_BATCH_SIZE,
    threads: int = 8,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    bootstrap_replicates: int = 500,
    command: list[str] | None = None,
) -> dict[str, Any]:
    if rung not in RUNG_TARGET_ROWS:
        raise ValueError(f"rung must be one of {sorted(RUNG_TARGET_ROWS)}")
    row_budget = int(presentations if presentations is not None else RUNG_TARGET_ROWS[rung])
    if row_budget < 1 or not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("presentations and unique seeds are required")
    if batch_size < 1 or threads < 1 or bootstrap_replicates < 2:
        raise ValueError("batch_size, threads, and bootstrap_replicates must be positive")
    if row_budget % batch_size != 0:
        raise ValueError("presentations must be divisible by batch_size for equal-weight optimizer updates")
    if (
        not math.isfinite(learning_rate)
        or learning_rate <= 0.0
        or not math.isfinite(weight_decay)
        or weight_decay < 0.0
    ):
        raise ValueError("learning_rate must be positive and weight_decay nonnegative")
    cache_dir = cache_dir.resolve()
    split_path = split_path.resolve()
    out_dir = out_dir.resolve()
    data = LabData(str(cache_dir), split_manifest_path=str(split_path))
    cache_manifest_path = cache_dir / "cache_manifest.json"
    config = {
        "schema_version": MC_SCREEN_SCHEMA_VERSION,
        "rung": rung,
        "presentations": row_budget,
        "seeds": list(seeds),
        "batch_size": batch_size,
        "threads": threads,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "bootstrap_replicates": bootstrap_replicates,
        "normalization": "identity_canonical",
        "target": "mc_actor_relative_terminal",
        "models": [asdict(spec) for spec in snapshot_specs()],
        "cache_manifest_sha256": _sha256(cache_manifest_path),
        "split_manifest_sha256": _sha256(split_path),
        "code_sha256": {
            "mc_screen": _sha256(Path(__file__)),
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
        preview = ", ".join(sorted(path.name for path in out_dir.iterdir())[:10]) or "empty directory"
        raise RuntimeError(f"refusing to overwrite incomplete MC screen directory: {out_dir} ({preview})")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = out_dir.with_name(f".{out_dir.name}.running.{uuid.uuid4().hex}")
    temp_dir.mkdir()
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    index = validation_index(data)
    index_path = temp_dir / "validation_index.npz"
    np.savez_compressed(index_path, **index)
    results: dict[str, Any] = {}
    output_paths = [index_path]
    for seed in seeds:
        stream = eligible_row_stream(data, split="train", presentations=row_budget, seed=seed)
        stream_receipt = row_stream_receipt(data, stream)
        stream_sha = stream_receipt["sha256"]
        train_labels = data.meta["z"][stream].astype(np.float64, copy=False)
        base_rate = float(train_labels.mean())
        s0_predictions = np.full(len(index["z"]), base_rate, dtype=np.float32)
        seed_dir = temp_dir / f"seed_{seed}"
        seed_dir.mkdir()
        s0_metrics = evaluate_predictions(
            s0_predictions,
            index,
            data.source_ids,
            s0_predictions,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=seed,
        )
        seed_result: dict[str, Any] = {
            "seed": seed,
            "row_stream_sha256": stream_sha,
            "row_stream": stream_receipt,
            "presentations": row_budget,
            "s0": {"base_rate": base_rate, "metrics": s0_metrics},
            "models": {},
        }
        s0_path = seed_dir / "s0_predictions.npz"
        np.savez_compressed(s0_path, p=s0_predictions)
        stream_path = seed_dir / "train_row_stream.npz"
        np.savez_compressed(stream_path, row_id=stream.astype(np.int64, copy=False))
        output_paths.extend([s0_path, stream_path])
        prediction_by_name = {"s0": s0_predictions}
        for spec in snapshot_specs():
            model, train_receipt = train_snapshot(
                data,
                spec,
                stream,
                seed=seed,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )
            predictions, inference_receipt = _predict(
                data,
                model,
                spec,
                index["row_id"],
                batch_size=batch_size,
            )
            metrics = evaluate_predictions(
                predictions,
                index,
                data.source_ids,
                s0_predictions,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=seed,
            )
            checkpoint_path = seed_dir / f"{spec.name}.pt"
            checkpoint = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "spec": asdict(spec),
                "model_config": {
                    "input_dim": model.input_dim,
                    "hidden_dims": list(model.hidden_dims),
                    "dropout": model.dropout,
                },
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
                    "batch_size": batch_size,
                },
                "train": train_receipt,
                "validation": metrics["overall"],
            }
            _atomic_torch_save(checkpoint_path, checkpoint)
            prediction_path = seed_dir / f"{spec.name}.predictions.npz"
            np.savez_compressed(prediction_path, p=predictions)
            output_paths.extend([checkpoint_path, prediction_path])
            prediction_by_name[spec.name] = predictions
            seed_result["models"][spec.name] = {
                "spec": asdict(spec),
                "parameter_count": parameter_count(model),
                "parameter_bytes_float32": parameter_count(model) * 4,
                "train": train_receipt,
                "inference_seconds": inference_receipt["elapsed_seconds"],
                "inference_rows_per_second": inference_receipt["rows_per_second"],
                "inference": inference_receipt,
                "metrics": metrics,
            }
        comparison_pairs = (
            ("s2_dense144_default", "s1_default"),
            ("s2_dense144_matched", "s1_matched"),
            ("s2_extra_only_matched", "s1_matched"),
            ("s2_extra_only_matched", "s2_dense144_matched"),
        )
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
            for candidate, reference in comparison_pairs
        }
        results[str(seed)] = seed_result
        seed_result_path = seed_dir / "result.json"
        write_json(seed_result_path, _json_safe(seed_result))
        output_paths.append(seed_result_path)
    result_path = temp_dir / "result.json"
    aggregate = aggregate_seeds(results)
    write_json(
        result_path,
        _json_safe({
            "config": config,
            "config_sha256": config_sha,
            "seeds": results,
            "aggregate": aggregate,
        }),
    )
    output_paths.append(result_path)
    files = [_output_receipt(path, temp_dir) for path in sorted(output_paths)]
    nonfinite_summary = {
        "train_features": sum(
            model["train"]["nonfinite_counts"]["features"]
            for result in results.values()
            for model in result["models"].values()
        ),
        "train_logits": sum(
            model["train"]["nonfinite_counts"]["logits"]
            for result in results.values()
            for model in result["models"].values()
        ),
        "train_losses": sum(
            model["train"]["nonfinite_counts"]["losses"]
            for result in results.values()
            for model in result["models"].values()
        ),
        "train_gradients": sum(
            model["train"]["nonfinite_counts"]["gradients"]
            for result in results.values()
            for model in result["models"].values()
        ),
        "inference_features": sum(
            model["inference"]["nonfinite_counts"]["features"]
            for result in results.values()
            for model in result["models"].values()
        ),
        "inference_logits": sum(
            model["inference"]["nonfinite_counts"]["logits"]
            for result in results.values()
            for model in result["models"].values()
        ),
        "inference_predictions": sum(
            model["inference"]["nonfinite_counts"]["predictions"]
            for result in results.values()
            for model in result["models"].values()
        ),
    }
    manifest = build_manifest(
        run_id=out_dir.name,
        kind="value_mc_snapshot_screen",
        command=command or [],
        inputs={
            "config": config,
            "config_sha256": config_sha,
            "cache_dir": str(cache_dir),
            "split_manifest_path": str(split_path),
            "mc_screen_code_sha256": _sha256(Path(__file__)),
        },
        outputs={"files": files},
        summary={
            "status": "screening_only",
            "seeds": list(seeds),
            "presentations_per_model_seed": row_budget,
            "validation_rows": int(len(index["z"])),
            "aggregate": _json_safe(aggregate),
            "calibration_targets_consumed": False,
            "test_targets_consumed": False,
            "gpu_utilization": config["environment"]["gpu_utilization"],
            "nonfinite_counts": nonfinite_summary,
            "offline_gameplay_safety_monitors": "not_applicable_no_gameplay_executed",
            "midrun_resume": "not_implemented_for_smoke_or_screen_rungs",
        },
    )
    write_manifest(temp_dir / "manifest.json", manifest)
    os.replace(temp_dir, out_dir)

    return {"action": "completed", "manifest": manifest, "results": results}
