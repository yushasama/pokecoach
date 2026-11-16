"""CUDA-only, resumable research training for M3 policy-Q and M4 specialist V."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for _path in (ROOT, SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from cg.api import SelectContext  # noqa: E402
import run_r9_r4f_retrain as r4f  # noqa: E402
import run_unified_r3_value as value_source  # noqa: E402
from deepmew.replay.contract.preflight import (  # noqa: E402
    require_accepted_canonical_decision_receipt,
)
from deepmew.q_guided.training import (  # noqa: E402
    Q_FEATURE_CLASSIFICATION,
    ManifestBoundTeacherDataset,
    QDistributionHead,
    ShardBinding,
    TrainingProgress,
    atomic_save_training_checkpoint,
    encode_actor_visible_actions,
    load_manifest_bound_shards,
    load_training_checkpoint,
    masked_gaussian_q_loss,
)
from deepmew.second_order.reader import augment_option_features  # noqa: E402
from deepmew.value_lab.calibration import apply_calibrator, fit_isotonic  # noqa: E402
from deepmew.value_lab.metrics import brier, log_loss  # noqa: E402
from deepmew.value_lab.models import SnapshotMLP  # noqa: E402


SCHEMA_VERSION = "deepmew.q_guided.gpu_train.v1"
CALIBRATOR_LINEAGE_SCHEMA_VERSION = "deepmew.q_guided.specialist_v_calibrator.v1"
POLICY_EXPORT_SCHEMA_VERSION = "deepmew.q_guided.r4f_research_export.v1"
CHECKPOINT_INTERVAL_SECONDS = 300.0
DEFAULT_BATCH_SIZE = 32
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
KL_ANCHOR_WEIGHT = 0.25
Q_REGRESSION_WEIGHT = 1.0
POLICY_GUIDANCE_WEIGHT = 1.0
TARGET_TEMPERATURE = 0.10
USE_AMP = False
MIN_POLICY_ROWS = 2
MIN_SPECIALIST_ROWS = 3
SAFETY_COUNTERS = {"illegal_actions": 0, "crashes": 0, "timeouts": 0}
PROBABILITY_ROUNDOFF_TOLERANCE = 1e-6


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _verify_file_hash(path: Path, expected: str, name: str) -> str:
    expected = str(expected).lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError(f"{name} hash must be a 64-character SHA-256")
    if not path.is_file() or _sha256_file(path) != expected:
        raise ValueError(f"{name} hash mismatch")

    return expected


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(_canonical_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        with temporary.open("r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _shard_for_manifest(manifest_path: Path) -> Path:
    name = manifest_path.name
    suffix = ".manifest.json"
    if not name.endswith(suffix):
        raise ValueError("--manifest must end in .manifest.json")
    shard = manifest_path.with_name(f"{name[:-len(suffix)]}.jsonl")
    if not shard.is_file():
        raise ValueError("manifest-bound sibling JSONL shard is missing")

    return shard


def _load_dataset(manifest_path: Path) -> ManifestBoundTeacherDataset:
    manifest_path = manifest_path.resolve()

    return load_manifest_bound_shards([ShardBinding(_shard_for_manifest(manifest_path), manifest_path)])


def _load_source_policy(path: Path, expected_sha256: str) -> tuple[nn.Module, dict[str, Any]]:
    _verify_file_hash(path, expected_sha256, "source policy checkpoint")
    model, payload = r4f.load_r4f_model(path)
    if payload.get("model_kind") != "r4f_intent_autoregressive":
        raise ValueError("source policy is not an R4F-compatible checkpoint")
    if any(not bool(torch.isfinite(value).all().item()) for value in model.state_dict().values()):
        raise ValueError("source policy contains non-finite parameters")

    return model, payload


def _load_snapshot_v(path: Path, expected_sha256: str) -> tuple[SnapshotMLP, dict[str, Any]]:
    _verify_file_hash(path, expected_sha256, "V checkpoint")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("model_config")
    if not isinstance(config, dict) or "model_state" not in payload:
        raise ValueError("V checkpoint lacks the SnapshotMLP adapter contract")
    model = SnapshotMLP(
        int(config["input_dim"]),
        tuple(int(value) for value in config["hidden_dims"]),
        float(config.get("dropout", 0.0)),
    )
    model.load_state_dict(payload["model_state"], strict=True)
    if model.input_dim != value_source.FEATURE_DIM_V2:
        raise ValueError("specialist V requires the frozen 49-dimensional S1 representation")
    if any(not bool(torch.isfinite(value).all().item()) for value in model.state_dict().values()):
        raise ValueError("V checkpoint contains non-finite parameters")

    return model, payload


def _reconstruct_atomic_options(
    row: dict[str, Any],
) -> tuple[list[dict[str, Any]], tuple[tuple[int, ...], ...]]:
    actions = row["actor_visible_legal_options"]
    indexed: dict[int, dict[str, Any]] = {}
    selections: list[tuple[int, ...]] = []
    for action in actions:
        selection = action.get("selection") if isinstance(action, dict) else None
        semantics = action.get("semantics") if isinstance(action, dict) else None
        if (
            not isinstance(selection, list)
            or not isinstance(semantics, list)
            or len(selection) != len(semantics)
            or any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in selection)
            or len(selection) != len(set(selection))
            or any(not isinstance(semantic, dict) for semantic in semantics)
        ):
            raise RuntimeError("eligible sibling action has an invalid actor-visible selection contract")
        for index, semantic in zip(selection, semantics, strict=True):
            normalized = dict(semantic)
            previous = indexed.setdefault(index, normalized)
            if _canonical_bytes(previous) != _canonical_bytes(normalized):
                raise RuntimeError("actor-visible atomic option semantics conflict across sibling actions")
        selections.append(tuple(selection))
    atomic_indices = tuple(sorted(indexed))
    positions = {original: offset for offset, original in enumerate(atomic_indices)}
    mapped = tuple(tuple(positions[index] for index in selection) for selection in selections)

    return [indexed[index] for index in atomic_indices], mapped


def _encode_sibling_actions(row: dict[str, Any]) -> tuple[np.ndarray, tuple[str, ...]]:
    encoded_options = []
    for action in row["actor_visible_legal_options"]:
        encoded = copy.deepcopy(action)
        if encoded.get("selection") == [] and encoded.get("semantics") == []:
            encoded["semantics"] = [{
                "option_type": "STOP",
                "select_type": "",
                "select_context": "",
                "semantic_key": "STOP",
            }]
        encoded_options.append(encoded)

    return encode_actor_visible_actions(encoded_options)


@dataclass(frozen=True)
class PolicyExample:
    row_sha256: str
    state: np.ndarray
    actions: np.ndarray
    policy_features: list[list[float]]
    sibling_selections: tuple[tuple[int, ...], ...]
    min_count: int
    max_count: int
    order_sensitive: bool
    q_mean: np.ndarray
    q_variance: np.ndarray
    families: tuple[str, ...]


def _policy_example(loaded_row: Any, source_input_dim: int) -> PolicyExample:
    row = dict(loaded_row.row)
    root = dict(row["actor_visible_root_observation"])
    semantics, sibling_selections = _reconstruct_atomic_options(row)
    if not semantics:
        raise RuntimeError("R4F adapter requires at least one actor-visible atomic option")
    encoded = r4f.r4.build_live_features(
        root,
        semantics,
        policy_id=str(row["continuation_policy_identity"]),
        deck_hash="",
    )
    if encoded is None:
        raise RuntimeError("R4F live feature adapter rejected an eligible teacher row")
    policy_features = augment_option_features(root, semantics, encoded.features)
    if not policy_features or len(policy_features[0]) != source_input_dim:
        raise RuntimeError("R4F source input dimension is incompatible with teacher action features")
    actions, action_ids = _encode_sibling_actions(row)
    if set(action_ids) != set(row["aggregate_q_mean"]):
        raise RuntimeError("teacher action identities do not align with Q targets")
    state = np.asarray(
        value_source._features({"raw_observation": root}, value_source.FEATURE_CONFIG_V2),
        dtype=np.float32,
    )
    q_mean = np.asarray([row["aggregate_q_mean"][key] for key in action_ids], dtype=np.float32)
    q_variance = np.asarray([row["q_variance"][key] for key in action_ids], dtype=np.float32)
    families = tuple(
        "+".join(str(semantics[index].get("option_type") or "UNKNOWN") for index in selection)
        or "EMPTY"
        for selection in sibling_selections
    )
    select = root.get("select") if isinstance(root.get("select"), dict) else {}
    min_count = int(select.get("minCount") or 0)
    max_count = max(min_count, min(int(select.get("maxCount") or len(semantics)), len(semantics)))
    order_sensitive = select.get("context") == int(SelectContext.SKILL_ORDER) or any(
        str(option.get("select_context") or "") == SelectContext.SKILL_ORDER.name
        for option in semantics
    )
    if (
        state.shape != (value_source.FEATURE_DIM_V2,)
        or not np.isfinite(state).all()
        or not np.isfinite(q_mean).all()
        or not np.isfinite(q_variance).all()
    ):
        raise ValueError("teacher example contains a non-finite or malformed training tensor")

    return PolicyExample(
        row_sha256=loaded_row.row_sha256,
        state=state,
        actions=actions,
        policy_features=policy_features,
        sibling_selections=sibling_selections,
        min_count=min_count,
        max_count=max_count,
        order_sensitive=order_sensitive,
        q_mean=q_mean,
        q_variance=q_variance,
        families=families,
    )


def _prepare_examples(dataset: ManifestBoundTeacherDataset, source_input_dim: int) -> list[PolicyExample]:
    return [_policy_example(row, source_input_dim) for row in dataset.rows]


def _split_examples(examples: list[PolicyExample], parts: int) -> tuple[list[PolicyExample], ...]:
    if len(examples) < parts:
        raise ValueError(f"eligible tranche requires at least {parts} rows")
    ordered = sorted(examples, key=lambda example: example.row_sha256)
    groups = [[] for _ in range(parts)]
    for index, example in enumerate(ordered):
        groups[index % parts].append(example)
    if parts == 2:
        return groups[1], groups[0]

    return tuple(groups)


class PolicyTrainingBundle(nn.Module):
    def __init__(self, policy: nn.Module, q_head: QDistributionHead) -> None:
        super().__init__()
        self.policy = policy
        self.q_head = q_head


def _step_batch(examples: list[PolicyExample], device: torch.device) -> tuple[Any, ...]:
    max_actions = max(len(example.q_mean) for example in examples)
    action_dim = examples[0].actions.shape[1]
    state = torch.zeros((len(examples), value_source.FEATURE_DIM_V2), dtype=torch.float32)
    actions = torch.zeros((len(examples), max_actions, action_dim), dtype=torch.float32)
    q_mean = torch.zeros((len(examples), max_actions), dtype=torch.float32)
    q_variance = torch.zeros_like(q_mean)
    q_mask = torch.zeros_like(q_mean, dtype=torch.bool)
    for row_index, example in enumerate(examples):
        count = len(example.q_mean)
        state[row_index] = torch.from_numpy(example.state)
        actions[row_index, :count] = torch.from_numpy(example.actions)
        q_mean[row_index, :count] = torch.from_numpy(example.q_mean)
        q_variance[row_index, :count] = torch.from_numpy(example.q_variance)
        q_mask[row_index, :count] = True
    return (
        state.to(device),
        actions.to(device),
        q_mean.to(device),
        q_variance.to(device),
        q_mask.to(device),
        [example.families for example in examples],
        examples,
    )


def _prefix_log_probabilities(
    model: nn.Module,
    example: PolicyExample,
    selected: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    atomic_count = len(example.policy_features)
    sample = r4f.r4.StepSample(
        features=example.policy_features,
        legal_count=atomic_count,
        selected=list(selected),
        target=atomic_count,
        step_index=len(selected),
        min_count=example.min_count,
        max_count=example.max_count,
        weight=1.0,
        metadata={},
    )
    packed = r4f.batch_steps_r4f([sample])
    features = packed.features.to(device)
    option_meta = packed.option_meta.to(device)
    mask = packed.mask.to(device)
    logits = model(features, option_meta).masked_fill(~mask, -torch.inf)

    return torch.log_softmax(logits[0], dim=0)


def _ordered_sequence_log_probability(
    model: nn.Module,
    example: PolicyExample,
    selection: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    prefix: tuple[int, ...] = ()
    total = torch.zeros((), device=device)
    for choice in selection:
        total = total + _prefix_log_probabilities(model, example, prefix, device)[choice]
        prefix = (*prefix, choice)
    stop_index = len(example.policy_features)

    return total + _prefix_log_probabilities(model, example, prefix, device)[stop_index]


def _unordered_sequence_log_probability(
    model: nn.Module,
    example: PolicyExample,
    selection: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    if not selection:
        return _ordered_sequence_log_probability(model, example, selection, device)
    states: dict[int, torch.Tensor] = {0: torch.zeros((), device=device)}
    full_mask = (1 << len(selection)) - 1
    for selected_count in range(len(selection)):
        for mask in tuple(states):
            if mask.bit_count() != selected_count:
                continue
            prefix = tuple(selection[index] for index in range(len(selection)) if mask & (1 << index))
            log_probability = _prefix_log_probabilities(model, example, prefix, device)
            for index, choice in enumerate(selection):
                bit = 1 << index
                if mask & bit:
                    continue
                next_mask = mask | bit
                candidate = states[mask] + log_probability[choice]
                if next_mask in states:
                    states[next_mask] = torch.logaddexp(states[next_mask], candidate)
                else:
                    states[next_mask] = candidate
    stop_index = len(example.policy_features)
    full_prefix = tuple(sorted(selection))

    return states[full_mask] + _prefix_log_probabilities(
        model,
        example,
        full_prefix,
        device,
    )[stop_index]


def _complete_action_log_probabilities(
    model: nn.Module,
    example: PolicyExample,
    device: torch.device,
) -> torch.Tensor:
    values = []
    for selection in example.sibling_selections:
        if example.order_sensitive:
            value = _ordered_sequence_log_probability(model, example, selection, device)
        else:
            value = _unordered_sequence_log_probability(model, example, selection, device)
        values.append(value)

    return torch.log_softmax(torch.stack(values), dim=0)


def _anchor_probabilities(
    anchor: nn.Module,
    examples: list[PolicyExample],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    anchor.to(device).eval()
    with torch.no_grad():
        return {
            example.row_sha256: _complete_action_log_probabilities(anchor, example, device).exp().clone()
            for example in examples
        }


def _policy_losses(
    bundle: PolicyTrainingBundle,
    batch: tuple[Any, ...],
    anchor_probabilities: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    state, actions, q_mean, q_variance, q_mask, _families, examples = batch
    distribution = bundle.q_head(state, actions, q_mask)
    q_loss = masked_gaussian_q_loss(distribution, q_mean, q_variance, q_mask)
    guidance_rows = []
    anchor_kl_rows = []
    for row_index, example in enumerate(examples):
        complete_log_probability = _complete_action_log_probabilities(
            bundle.policy,
            example,
            device,
        )
        count = len(example.q_mean)
        target = torch.softmax(q_mean[row_index, :count] / TARGET_TEMPERATURE, dim=0)
        guidance_rows.append(-(target * complete_log_probability).sum())
        anchor_kl_rows.append(torch.nn.functional.kl_div(
            complete_log_probability,
            anchor_probabilities[example.row_sha256],
            reduction="sum",
        ))
    guidance = torch.stack(guidance_rows).mean()
    anchor_kl = torch.stack(anchor_kl_rows).mean()
    total = Q_REGRESSION_WEIGHT * q_loss + POLICY_GUIDANCE_WEIGHT * guidance + KL_ANCHOR_WEIGHT * anchor_kl

    return total, {
        "q_regression_loss": float(q_loss.detach()),
        "policy_guidance_loss": float(guidance.detach()),
        "r7_anchor_kl": float(anchor_kl.detach()),
        "total_loss": float(total.detach()),
    }


def _policy_metrics(bundle: PolicyTrainingBundle, examples: list[PolicyExample], device: torch.device) -> dict[str, float]:
    bundle.eval()
    regrets = []
    correct = 0
    predicted_advantages = []
    target_advantages = []
    selected_families = set()
    q_squared_error = []
    with torch.inference_mode():
        for start in range(0, len(examples), DEFAULT_BATCH_SIZE):
            chunk = examples[start:start + DEFAULT_BATCH_SIZE]
            batch = _step_batch(chunk, device)
            state, actions, q_mean, _q_variance, q_mask, families, batch_examples = batch
            distribution = bundle.q_head(state, actions, q_mask)
            for row_index, (family_names, example) in enumerate(zip(families, batch_examples, strict=True)):
                count = int(q_mask[row_index].sum().item())
                complete_log_probability = _complete_action_log_probabilities(
                    bundle.policy,
                    example,
                    device,
                )
                predicted_index = int(complete_log_probability.argmax().item())
                target_index = int(q_mean[row_index, :count].argmax().item())
                correct += int(predicted_index == target_index)
                regrets.append(float(q_mean[row_index, target_index] - q_mean[row_index, predicted_index]))
                selected_families.add(family_names[predicted_index])
                predicted_advantages.extend((distribution.mean[row_index, :count] - distribution.mean[row_index, 0]).cpu().tolist())
                target_advantages.extend((q_mean[row_index, :count] - q_mean[row_index, 0]).cpu().tolist())
                q_squared_error.extend((distribution.mean[row_index, :count] - q_mean[row_index, :count]).square().cpu().tolist())
    advantage_error = np.asarray(predicted_advantages) - np.asarray(target_advantages)

    return {
        "q_regression_mse": float(np.mean(q_squared_error)),
        "sibling_ranking_accuracy": correct / len(examples),
        "material_advantage_mae": float(np.mean(np.abs(advantage_error))),
        "mean_regret": float(np.mean(regrets)),
        "max_regret": float(np.max(regrets)),
        "action_family_diversity": float(len(selected_families)),
    }


def _progress(
    prior: TrainingProgress | None,
    updates: int,
    presentations: int,
    cumulative_seconds: float,
    metrics: dict[str, float],
) -> TrainingProgress:
    best = dict(prior.best_metrics) if prior else {}
    for name, value in metrics.items():
        if math.isfinite(value):
            best[name] = min(float(value), float(best.get(name, value)))

    return TrainingProgress(
        epoch=0,
        batch_cursor=(prior.batch_cursor if prior else 0) + updates,
        global_step=(prior.global_step if prior else 0) + updates,
        presentations=(prior.presentations if prior else 0) + presentations,
        cumulative_gpu_seconds=cumulative_seconds,
        best_metrics=best,
        last_metrics=metrics,
        safety_counters=SAFETY_COUNTERS,
    )


def _train_policy(
    bundle: PolicyTrainingBundle,
    anchor: nn.Module,
    train_rows: list[PolicyExample],
    heldout_rows: list[PolicyExample],
    *,
    device: torch.device,
    minimum_seconds: float,
    checkpoint_path: Path,
    identity: dict[str, Any],
    resume: bool,
) -> tuple[TrainingProgress, dict[str, float]]:
    bundle.to(device)
    anchor.to(device).eval()
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    anchor_probability_by_row = _anchor_probabilities(anchor, train_rows, device)
    optimizer = torch.optim.AdamW(bundle.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and device.type == "cuda")
    sampler = torch.Generator().manual_seed(20260712)
    prior = None
    if resume:
        prior = load_training_checkpoint(
            checkpoint_path,
            model=bundle,
            optimizer=optimizer,
            scheduler=scheduler,
            amp_scaler=scaler,
            sampler_generator=sampler,
            expected_dataset_sha256=identity["dataset_sha256"],
            expected_config_sha256=identity["config_sha256"],
            expected_source_sha256s=identity["source_sha256s"],
            map_location=device,
        )
    cumulative = float(prior.cumulative_gpu_seconds if prior else 0.0)
    last_checkpoint = time.monotonic()
    updates = 0
    presentations = 0
    last_metrics = dict(prior.last_metrics) if prior else {}
    while cumulative < minimum_seconds:
        indices = torch.randint(len(train_rows), (min(DEFAULT_BATCH_SIZE, len(train_rows)),), generator=sampler).tolist()
        batch = _step_batch([train_rows[index] for index in indices], device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=USE_AMP and device.type == "cuda"):
            loss, last_metrics = _policy_losses(
                bundle,
                batch,
                anchor_probability_by_row,
                device,
            )
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("non-finite policy/Q loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(bundle.parameters(), 10.0)
        if not bool(torch.isfinite(gradient_norm).item()):
            raise FloatingPointError("non-finite policy/Q gradient")
        scaler.step(optimizer)
        scaler.update()
        scheduler.step((prior.global_step if prior else 0) + updates + 1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        cumulative += time.perf_counter() - started
        updates += 1
        presentations += len(indices)
        last_metrics.update({
            "optimizer_updates": float((prior.global_step if prior else 0) + updates),
            "nonfinite_count": 0.0,
            "mask_failure_count": 0.0,
            "collapse_count": 0.0,
        })
        if time.monotonic() - last_checkpoint >= CHECKPOINT_INTERVAL_SECONDS:
            progress = _progress(prior, updates, presentations, cumulative, last_metrics)
            atomic_save_training_checkpoint(
                checkpoint_path,
                model=bundle,
                optimizer=optimizer,
                scheduler=scheduler,
                amp_scaler=scaler,
                sampler_generator=sampler,
                progress=progress,
                **identity,
            )
            last_checkpoint = time.monotonic()
    metrics = _policy_metrics(bundle, heldout_rows, device)
    metrics.update({
        "optimizer_updates": float((prior.global_step if prior else 0) + updates),
        "nonfinite_count": 0.0,
        "mask_failure_count": 0.0,
        "collapse_count": 0.0,
    })
    progress = _progress(prior, updates, presentations, cumulative, metrics)
    atomic_save_training_checkpoint(
        checkpoint_path,
        model=bundle,
        optimizer=optimizer,
        scheduler=scheduler,
        amp_scaler=scaler,
        sampler_generator=sampler,
        progress=progress,
        **identity,
    )

    return progress, metrics


def _anchor_expectations(anchor: nn.Module, examples: list[PolicyExample], device: torch.device) -> np.ndarray:
    values = []
    probabilities = _anchor_probabilities(anchor, examples, device)
    for example in examples:
        probability = probabilities[example.row_sha256].cpu().numpy()
        values.append(float(np.dot(probability, example.q_mean)))

    targets = np.asarray(values, dtype=np.float64)
    if not np.isfinite(targets).all():
        raise FloatingPointError("R7 aggregate-Q expectations must be finite")
    if np.any(
        (targets < -PROBABILITY_ROUNDOFF_TOLERANCE)
        | (targets > 1.0 + PROBABILITY_ROUNDOFF_TOLERANCE)
    ):
        raise ValueError("R7 aggregate-Q expectations must be on the probability scale [0, 1]")

    return np.clip(targets, 0.0, 1.0).astype(np.float32)


def _value_probabilities(model: SnapshotMLP, examples: list[PolicyExample], device: torch.device) -> np.ndarray:
    state = torch.from_numpy(np.stack([example.state for example in examples])).to(device)
    mask = torch.ones((len(examples), 1), dtype=torch.bool, device=device)
    with torch.inference_mode():
        logits = model(state[:, None, :], mask)[:, 0]

    return torch.sigmoid(logits).cpu().numpy().astype(np.float64)


def _train_specialist_v(
    model: SnapshotMLP,
    anchor: nn.Module,
    train_rows: list[PolicyExample],
    calibration_rows: list[PolicyExample],
    heldout_rows: list[PolicyExample],
    *,
    device: torch.device,
    minimum_seconds: float,
    checkpoint_path: Path,
    identity: dict[str, Any],
    resume: bool,
) -> tuple[TrainingProgress, dict[str, float], dict[str, Any]]:
    model.to(device)
    targets = _anchor_expectations(anchor, train_rows, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and device.type == "cuda")
    sampler = torch.Generator().manual_seed(20260713)
    prior = None
    if resume:
        prior = load_training_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            amp_scaler=scaler,
            sampler_generator=sampler,
            expected_dataset_sha256=identity["dataset_sha256"],
            expected_config_sha256=identity["config_sha256"],
            expected_source_sha256s=identity["source_sha256s"],
            map_location=device,
        )
    cumulative = float(prior.cumulative_gpu_seconds if prior else 0.0)
    last_checkpoint = time.monotonic()
    updates = 0
    presentations = 0
    last_metrics = dict(prior.last_metrics) if prior else {}
    states = torch.from_numpy(np.stack([example.state for example in train_rows])).to(device)
    target_tensor = torch.from_numpy(targets).to(device)
    while cumulative < minimum_seconds:
        indices = torch.randint(len(train_rows), (min(DEFAULT_BATCH_SIZE, len(train_rows)),), generator=sampler)
        batch_state = states[indices.to(device)]
        batch_target = target_tensor[indices.to(device)]
        mask = torch.ones((len(indices), 1), dtype=torch.bool, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=USE_AMP and device.type == "cuda"):
            prediction = torch.sigmoid(model(batch_state[:, None, :], mask)[:, 0])
            loss = torch.nn.functional.mse_loss(prediction, batch_target)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("non-finite specialist-V loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        if not bool(torch.isfinite(gradient_norm).item()):
            raise FloatingPointError("non-finite specialist-V gradient")
        scaler.step(optimizer)
        scaler.update()
        scheduler.step((prior.global_step if prior else 0) + updates + 1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        cumulative += time.perf_counter() - started
        updates += 1
        presentations += len(indices)
        last_metrics = {
            "loss": float(loss.detach()),
            "optimizer_updates": float((prior.global_step if prior else 0) + updates),
            "nonfinite_count": 0.0,
            "collapse_count": 0.0,
        }
        if time.monotonic() - last_checkpoint >= CHECKPOINT_INTERVAL_SECONDS:
            progress = _progress(prior, updates, presentations, cumulative, last_metrics)
            atomic_save_training_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                amp_scaler=scaler,
                sampler_generator=sampler,
                progress=progress,
                **identity,
            )
            last_checkpoint = time.monotonic()
    calibration_raw = _value_probabilities(model, calibration_rows, device)
    calibration_targets = _anchor_expectations(anchor, calibration_rows, device).astype(np.float64)
    calibrator = fit_isotonic(calibration_raw, calibration_targets)
    heldout_raw = _value_probabilities(model, heldout_rows, device)
    heldout_targets = _anchor_expectations(anchor, heldout_rows, device).astype(np.float64)
    heldout = apply_calibrator(heldout_raw, calibrator)
    metrics = {
        "heldout_log_loss": float(log_loss(heldout, heldout_targets)),
        "heldout_brier": float(brier(heldout, heldout_targets)),
        "heldout_mae": float(np.mean(np.abs(heldout - heldout_targets))),
        "heldout_target_variance": float(np.var(heldout_targets)),
        "optimizer_updates": float((prior.global_step if prior else 0) + updates),
        "nonfinite_count": 0.0,
        "collapse_count": 0.0,
    }
    progress = _progress(prior, updates, presentations, cumulative, metrics)
    atomic_save_training_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        amp_scaler=scaler,
        sampler_generator=sampler,
        progress=progress,
        **identity,
    )

    return progress, metrics, calibrator


def _config_hash(args: argparse.Namespace, manifest_sha256: str) -> str:
    return _sha256_bytes(
        _canonical_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "mode": args.mode,
                "manifest_sha256": manifest_sha256,
                "source_policy_sha256": args.source_policy_sha256.lower(),
                "v_checkpoint_sha256": args.v_checkpoint_sha256.lower(),
                "batch_size": DEFAULT_BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "kl_anchor_weight": KL_ANCHOR_WEIGHT,
                "target_temperature": TARGET_TEMPERATURE,
                "use_amp": USE_AMP,
            }
        )
    )


def _export_policy(run_dir: Path, bundle: PolicyTrainingBundle, source_payload: dict[str, Any], identity: dict[str, Any]) -> Path:
    payload = copy.deepcopy(source_payload)
    payload["model_state"] = bundle.policy.cpu().state_dict()
    payload["q_guided_research"] = {
        "schema_version": POLICY_EXPORT_SCHEMA_VERSION,
        "classification": Q_FEATURE_CLASSIFICATION,
        "dataset_sha256": identity["dataset_sha256"],
        "config_sha256": identity["config_sha256"],
        "q_head_state": bundle.q_head.cpu().state_dict(),
    }
    target = run_dir / "research_r4f_q_guided.pt"
    _atomic_torch(target, payload)
    verified, _ = r4f.load_r4f_model(target)
    if set(verified.state_dict()) != set(bundle.policy.state_dict()):
        raise RuntimeError("R4F research export verification failed")

    return target


def run(args: argparse.Namespace, *, _device: torch.device | None = None) -> dict[str, Any]:
    if args.mode not in {"policy", "specialist-v"}:
        raise ValueError("--mode must be policy or specialist-v")
    if not math.isfinite(args.minimum_cumulative_seconds) or args.minimum_cumulative_seconds <= 0.0:
        raise ValueError("--minimum-cumulative-seconds must be finite and positive")
    if _device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("Q-guided GPU training requires CUDA")
        device = torch.device("cuda")
    else:
        device = _device
    manifest_path = Path(args.manifest).resolve()
    source_path = Path(args.source_policy_checkpoint).resolve()
    v_path = Path(args.v_checkpoint).resolve()
    dataset = _load_dataset(manifest_path)
    source_policy, source_payload = _load_source_policy(source_path, args.source_policy_sha256)
    source_policy.eval()
    source_input_dim = int(source_payload["input_dim"])
    examples = _prepare_examples(dataset, source_input_dim)
    manifest_sha256 = _sha256_file(manifest_path)
    _verify_file_hash(v_path, args.v_checkpoint_sha256, "V checkpoint")
    config_sha256 = _config_hash(args, manifest_sha256)
    identity = {
        "dataset_sha256": dataset.dataset_sha256,
        "config_sha256": config_sha256,
        "source_sha256s": tuple(sorted({
            args.source_policy_sha256.lower(),
            args.v_checkpoint_sha256.lower(),
            *dataset.shard_sha256s,
            *dataset.manifest_sha256s,
        })),
    }
    run_dir = Path(args.run_dir).resolve()
    checkpoint_path = run_dir / "training.pt"
    if args.resume:
        if not checkpoint_path.is_file():
            raise ValueError("--resume requires an existing complete checkpoint")
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("new run directory must be absent or empty")
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if args.mode == "policy":
        if len(examples) < MIN_POLICY_ROWS:
            raise ValueError(f"policy tranche requires at least {MIN_POLICY_ROWS} eligible rows")
        train_rows, heldout_rows = _split_examples(examples, 2)
        policy = copy.deepcopy(source_policy)
        action_dim = examples[0].actions.shape[1]
        if any(example.actions.shape[1] != action_dim for example in examples):
            raise ValueError("actor-visible action encoding dimension drifted")
        q_head = QDistributionHead(value_source.FEATURE_DIM_V2, action_dim)
        bundle = PolicyTrainingBundle(policy, q_head)
        progress, metrics = _train_policy(
            bundle,
            copy.deepcopy(source_policy),
            train_rows,
            heldout_rows,
            device=device,
            minimum_seconds=float(args.minimum_cumulative_seconds),
            checkpoint_path=checkpoint_path,
            identity=identity,
            resume=bool(args.resume),
        )
        export_path = _export_policy(run_dir, bundle, source_payload, identity)
        artifacts = {"research_r4f_checkpoint": str(export_path), "calibrator": None}
    else:
        if len(examples) < MIN_SPECIALIST_ROWS:
            raise ValueError(f"specialist-V tranche requires at least {MIN_SPECIALIST_ROWS} eligible rows")
        train_rows, calibration_rows, heldout_rows = _split_examples(examples, 3)
        model, _v_payload = _load_snapshot_v(v_path, args.v_checkpoint_sha256)
        progress, metrics, calibrator = _train_specialist_v(
            model,
            copy.deepcopy(source_policy),
            train_rows,
            calibration_rows,
            heldout_rows,
            device=device,
            minimum_seconds=float(args.minimum_cumulative_seconds),
            checkpoint_path=checkpoint_path,
            identity=identity,
            resume=bool(args.resume),
        )
        calibrator_lineage = {
            "schema_version": CALIBRATOR_LINEAGE_SCHEMA_VERSION,
            "classification": Q_FEATURE_CLASSIFICATION,
            "dataset_sha256": dataset.dataset_sha256,
            "config_sha256": config_sha256,
            "source_v_sha256": args.v_checkpoint_sha256.lower(),
            "source_policy_sha256": args.source_policy_sha256.lower(),
            "calibrator": calibrator,
        }
        calibrator_path = run_dir / "specialist_v_calibrator.json"
        _atomic_json(calibrator_path, calibrator_lineage)
        artifacts = {"research_r4f_checkpoint": None, "calibrator": str(calibrator_path)}
    if progress.cumulative_gpu_seconds < args.minimum_cumulative_seconds:
        raise RuntimeError("training stopped before the requested cumulative GPU floor")
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_cumulative_floor",
        "classification": Q_FEATURE_CLASSIFICATION,
        "mode": args.mode,
        "dataset_sha256": dataset.dataset_sha256,
        "config_sha256": config_sha256,
        "manifest_sha256": manifest_sha256,
        "source_policy_sha256": args.source_policy_sha256.lower(),
        "source_v_sha256": args.v_checkpoint_sha256.lower(),
        "cumulative_gpu_seconds": progress.cumulative_gpu_seconds,
        "minimum_cumulative_gpu_seconds": float(args.minimum_cumulative_seconds),
        "checkpoint": str(checkpoint_path),
        "metrics": metrics,
        "safety_counters": dict(progress.safety_counters),
        "artifacts": artifacts,
        "wall_seconds_this_invocation": time.time() - started,
    }
    _atomic_json(run_dir / "metrics.json", result)

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-decision-receipt", default="")
    parser.add_argument("--mode", required=True, choices=("policy", "specialist-v"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--minimum-cumulative-seconds", required=True, type=float)
    parser.add_argument("--source-policy-checkpoint", required=True)
    parser.add_argument("--source-policy-sha256", required=True)
    parser.add_argument("--v-checkpoint", required=True)
    parser.add_argument("--v-checkpoint-sha256", required=True)
    parser.add_argument("--resume", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    accepted_receipt = require_accepted_canonical_decision_receipt(
        args.canonical_decision_receipt,
        consumer=__name__,
    )
    args.canonical_decision_receipt = str(accepted_receipt.path)
    result = run(args)
    print(json.dumps(result, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
