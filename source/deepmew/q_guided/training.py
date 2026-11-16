"""Research-only training primitives for quarantined Q and specialist-V models.

This module deliberately stops before policy control.  It accepts only
manifest-bound, aggregate teacher labels, encodes already actor-visible legal
semantics, and persists enough state for bitwise-equivalent resumptions where
the underlying device and PyTorch kernels are deterministic.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from deepmew.q_guided.teacher_contract import validate_safe_shard
from deepmew.rl1_teacher_bc.semantics import build_option_features
from deepmew.value_lab.models import SnapshotMLP


TEACHER_MANIFEST_SCHEMA_VERSION = "deepmew.q_guided.teacher_shard_manifest.v1"
TRAINING_CHECKPOINT_SCHEMA_VERSION = "deepmew.q_guided.training_checkpoint.v1"
CHECKPOINT_RECEIPT_SCHEMA_VERSION = "deepmew.q_guided.checkpoint_receipt.v1"
CORPUS_MANIFEST_SCHEMA_VERSION = "deepmew.q_guided.corpus_manifest.v1"
STREAMING_TRAINER_SCHEMA_VERSION = "deepmew.q_guided.streaming_trainer.v1"
STREAMING_TRAINING_STATE_SCHEMA_VERSION = "deepmew.q_guided.streaming_training_state.v1"
STREAMING_CHECKPOINT_SCHEMA_VERSION = "deepmew.q_guided.streaming_checkpoint.v2"
STREAMING_CHECKPOINT_RECEIPT_SCHEMA_VERSION = "deepmew.q_guided.streaming_checkpoint_receipt.v1"
REPLAY_POLICY_SCHEMA_VERSION = "deepmew.q_guided.replay_policy.v1"
ROW_SPLIT_POLICY_SCHEMA_VERSION = "deepmew.q_guided.row_split_policy.v1"
M3_HISTORICAL_SPLIT_SCHEMA_VERSION = "deepmew.q_guided.m3_historical_split.v1"
ELIGIBLE_TEACHER_STATUS = "eligible_aggregate_teacher_target"
Q_FEATURE_CLASSIFICATION = "quarantine"
MIN_AGGREGATE_WORLD_SAMPLES = 2
MIN_PAIRED_SAMPLES = 2
ACTION_HASH_BINS = 16
LOG_VARIANCE_MIN = -10.0
LOG_VARIANCE_MAX = 4.0

_MANIFEST_FIELDS = frozenset({
    "schema_version",
    "split",
    "shard_sha256",
    "row_count",
    "row_receipts",
})
_ROW_RECEIPT_FIELDS = frozenset({
    "status",
    "world_sample_count",
    "paired_sample_count",
    "row_sha256",
    "actor_visible_signature_sha256",
    "actor_visible_source_sha256",
})
_CORPUS_MANIFEST_FIELDS = frozenset({
    "schema_version",
    "classification",
    "status",
    "observed_at",
    "generation",
    "corpus_lineage_id",
    "compatibility",
    "scope",
    "minimum_useful_seconds_per_miner",
    "ordered_shard_identity_sha256",
    "ordered_shards",
    "miners",
    "coverage",
    "duplication",
    "m3_consumption",
    "provisional_vs_final",
    "gates",
})
_SOURCE_CORPUS_COMPATIBILITY_FIELDS = frozenset({
    "teacher_manifest_schema_version",
    "teacher_config_sha256s",
    "label_v_checkpoint_sha256s",
    "label_v_calibrator_sha256s",
    "miner_config_sha256s",
    "split_values",
})
_TRAINER_COMPATIBILITY_FIELDS = frozenset({
    "state_feature_schema_sha256",
    "action_feature_schema_sha256",
    "deck_manifest_sha256",
    "split_policy_sha256",
    "trainer_schema_version",
})
_CORPUS_SHARD_FIELDS = frozenset({
    "ordinal",
    "miner_id",
    "source_shard_index",
    "split",
    "shard",
    "manifest",
    "shard_sha256",
    "manifest_sha256",
    "row_count",
    "m3_consumed_row_count",
    "m3_consumed_row_offsets_inclusive",
    "unseen_row_count",
    "unseen_row_offsets_inclusive",
    "coverage",
})
_CORPUS_GATE_FIELDS = frozenset({
    "m1_completed",
    "m2_completed",
    "minimum_useful_seconds_met",
    "completion_receipts_valid",
    "all_shards_and_sidecars_hash_validated",
    "ordered_shard_inventory_unique",
    "zero_duplicate_roots_or_states",
    "zero_safety_failures",
    "v0_provenance_valid",
    "m3_consumed_cursor_exact",
    "labels_modified",
    "policy_control_eligible",
})
_M3_CONSUMPTION_FIELDS = frozenset({
    "tranche_dataset_sha256",
    "ordered_consumed_row_sha256",
    "consumed_unique_rows",
    "unseen_unique_rows",
    "final_immutable_rows",
    "consumed_percentage",
})


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("payload must be finite and JSON-serializable") from exc

    return text.encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _validate_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ValueError(f"{name} must be a 64-character SHA-256")

    return value.lower()


def _require_exact_fields(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    fields = set(value)
    if fields != expected:
        missing = sorted(expected - fields)
        unexpected = sorted(fields - expected)
        raise ValueError(f"{name} field inventory mismatch: missing={missing}, unexpected={unexpected}")


@dataclass(frozen=True)
class ShardBinding:
    shard_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class LoadedTeacherRow:
    split: str
    row: Mapping[str, Any]
    row_sha256: str
    actor_visible_signature_sha256: str
    actor_visible_source_sha256: str


@dataclass(frozen=True)
class ManifestBoundTeacherDataset:
    rows: tuple[LoadedTeacherRow, ...]
    dataset_sha256: str
    shard_sha256s: tuple[str, ...]
    manifest_sha256s: tuple[str, ...]

    def rows_for_split(self, split: str) -> tuple[LoadedTeacherRow, ...]:
        return tuple(row for row in self.rows if row.split == split)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON object from {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")

    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    raise ValueError(f"{path}:{line_number} contains a blank row")
                value = json.loads(raw_line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} must contain one JSON object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSONL shard {path}") from exc

    return rows


def _validate_target_map(row: Mapping[str, Any]) -> None:
    action_keys = set(row["aggregate_q_mean"])
    for action_key in action_keys:
        mean = row["aggregate_q_mean"][action_key]
        variance = row["q_variance"][action_key]
        standard_error = row["q_standard_error"][action_key]
        if isinstance(mean, bool) or not isinstance(mean, (int, float)):
            raise TypeError("aggregate Q means must be numeric")
        if not math.isfinite(float(mean)) or not 0.0 <= float(mean) <= 1.0:
            raise ValueError("aggregate Q means must be finite and in [0, 1]")
        for value, name in ((variance, "Q variance"), (standard_error, "Q standard error")):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


def _validate_manifest_receipt(receipt: Mapping[str, Any]) -> None:
    _require_exact_fields(receipt, _ROW_RECEIPT_FIELDS, "teacher row receipt")
    if receipt["status"] != ELIGIBLE_TEACHER_STATUS:
        raise ValueError("teacher row receipt is not an eligible aggregate target")
    for field, minimum in (
        ("world_sample_count", MIN_AGGREGATE_WORLD_SAMPLES),
        ("paired_sample_count", MIN_PAIRED_SAMPLES),
    ):
        value = receipt[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"teacher row receipt {field} must be at least {minimum}")
    for field in (
        "row_sha256",
        "actor_visible_signature_sha256",
        "actor_visible_source_sha256",
    ):
        _validate_sha256(receipt[field], field)


def load_manifest_bound_shards(
    bindings: Sequence[ShardBinding],
) -> ManifestBoundTeacherDataset:
    """Load aggregate labels only after external receipts and leakage gates pass."""
    if not bindings:
        raise ValueError("at least one manifest-bound teacher shard is required")
    loaded_rows: list[LoadedTeacherRow] = []
    shard_hashes: list[str] = []
    manifest_hashes: list[str] = []
    seen_shards: set[str] = set()
    seen_rows: set[str] = set()
    signature_split: dict[str, str] = {}
    source_split: dict[str, str] = {}
    for binding in bindings:
        if not isinstance(binding, ShardBinding):
            raise TypeError("bindings must contain ShardBinding values")
        shard_path = Path(binding.shard_path).resolve()
        manifest_path = Path(binding.manifest_path).resolve()
        manifest = _read_json_object(manifest_path)
        _require_exact_fields(manifest, _MANIFEST_FIELDS, "teacher shard manifest")
        if manifest["schema_version"] != TEACHER_MANIFEST_SCHEMA_VERSION:
            raise ValueError("teacher shard manifest schema mismatch")
        split = manifest["split"]
        if not isinstance(split, str) or not split:
            raise ValueError("teacher shard split must be a non-empty string")
        shard_sha256 = _validate_sha256(manifest["shard_sha256"], "shard_sha256")
        if _sha256_file(shard_path) != shard_sha256:
            raise ValueError("teacher shard hash mismatch")
        if shard_sha256 in seen_shards:
            raise ValueError("teacher shard hash is bound more than once")
        seen_shards.add(shard_sha256)
        rows = _read_jsonl(shard_path)
        row_count = manifest["row_count"]
        receipts = manifest["row_receipts"]
        if (
            isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or row_count <= 0
            or row_count != len(rows)
            or not isinstance(receipts, list)
            or len(receipts) != len(rows)
        ):
            raise ValueError("teacher shard row count or receipt coverage mismatch")
        for row, receipt in zip(rows, receipts, strict=True):
            if not isinstance(receipt, Mapping):
                raise TypeError("teacher row receipt must be a mapping")
            _validate_manifest_receipt(receipt)
            validate_safe_shard(row)
            _validate_target_map(row)
            row_sha256 = _sha256_bytes(_canonical_json_bytes(row))
            if row_sha256 != receipt["row_sha256"].lower():
                raise ValueError("teacher row hash mismatch")
            if row_sha256 in seen_rows:
                raise ValueError("teacher row hash overlaps another bound shard")
            seen_rows.add(row_sha256)
            signature = receipt["actor_visible_signature_sha256"].lower()
            source = receipt["actor_visible_source_sha256"].lower()
            previous_signature_split = signature_split.setdefault(signature, split)
            if previous_signature_split != split:
                raise ValueError("actor-visible signature overlaps dataset splits")
            previous_source_split = source_split.setdefault(source, split)
            if previous_source_split != split:
                raise ValueError("actor-visible source hash overlaps dataset splits")
            loaded_rows.append(
                LoadedTeacherRow(
                    split=split,
                    row=row,
                    row_sha256=row_sha256,
                    actor_visible_signature_sha256=signature,
                    actor_visible_source_sha256=source,
                )
            )
        shard_hashes.append(shard_sha256)
        manifest_hashes.append(_sha256_file(manifest_path))
    dataset_payload = {
        "classification": Q_FEATURE_CLASSIFICATION,
        "rows": [
            {
                "row_sha256": row.row_sha256,
                "split": row.split,
                "signature": row.actor_visible_signature_sha256,
                "source": row.actor_visible_source_sha256,
            }
            for row in loaded_rows
        ],
        "shards": shard_hashes,
        "manifests": manifest_hashes,
    }

    return ManifestBoundTeacherDataset(
        rows=tuple(loaded_rows),
        dataset_sha256=_sha256_bytes(_canonical_json_bytes(dataset_payload)),
        shard_sha256s=tuple(shard_hashes),
        manifest_sha256s=tuple(manifest_hashes),
    )


def _hash_features(value: Any, bins: int = ACTION_HASH_BINS) -> list[float]:
    digest = hashlib.sha256(_canonical_json_bytes(value)).digest()

    return [digest[index] / 255.0 for index in range(bins)]


def _semantic_options(option: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    semantics = option.get("semantics")
    if semantics is None:
        return (dict(option),)
    if not isinstance(semantics, list) or not all(isinstance(value, dict) for value in semantics):
        raise ValueError("composite legal option semantics must be a list of objects")

    return tuple(dict(value) for value in semantics)


def _canonical_action_id(option: Mapping[str, Any], position: int) -> str:
    selection = option.get("selection")
    if selection is not None:
        if (
            not isinstance(selection, list)
            or any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in selection)
        ):
            raise ValueError("legal option selection must contain non-negative integers")

        return json.dumps(selection, separators=(",", ":"))
    index = option.get("index", position)
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("legal option index must be a non-negative integer")

    return str(index)


def encode_actor_visible_actions(
    legal_options: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Encode legal selections using the existing actor-visible semantics."""
    if not isinstance(legal_options, Sequence) or isinstance(legal_options, (str, bytes)):
        raise TypeError("legal_options must be a sequence")
    if not legal_options:
        raise ValueError("legal_options cannot be empty")
    rows: list[list[float]] = []
    action_ids: list[str] = []
    for position, raw_option in enumerate(legal_options):
        if not isinstance(raw_option, Mapping):
            raise TypeError("legal options must be mappings")
        semantics = _semantic_options(raw_option)
        if not semantics:
            raise ValueError("legal option must contain at least one semantic option")
        semantic_matrix = np.asarray(
            [build_option_features(option) for option in semantics],
            dtype=np.float32,
        )
        if semantic_matrix.ndim != 2 or not np.isfinite(semantic_matrix).all():
            raise ValueError("legal semantic encoding must be a finite matrix")
        selection = raw_option.get("selection", [position])
        vector = np.concatenate(
            [
                semantic_matrix.mean(axis=0),
                semantic_matrix.max(axis=0),
                np.asarray(
                    [
                        min(len(semantics), 16) / 16.0,
                        float(len(semantics) > 1),
                    ],
                    dtype=np.float32,
                ),
                np.asarray(_hash_features(selection), dtype=np.float32),
            ]
        )
        rows.append(vector.tolist())
        action_ids.append(_canonical_action_id(raw_option, position))
    if len(action_ids) != len(set(action_ids)):
        raise ValueError("actor-visible legal action identifiers must be unique")

    return np.asarray(rows, dtype=np.float32), tuple(action_ids)


@dataclass(frozen=True)
class EncodedQExample:
    state_features: torch.Tensor
    action_features: torch.Tensor
    action_mask: torch.Tensor
    q_mean: torch.Tensor
    q_variance: torch.Tensor
    selected_action_index: int
    row_sha256: str


def encode_teacher_row(
    loaded_row: LoadedTeacherRow,
    state_encoder: Callable[[Mapping[str, Any]], np.ndarray],
) -> EncodedQExample:
    """Build one model example while keeping the raw observation actor-visible."""
    state = np.asarray(
        state_encoder(loaded_row.row["actor_visible_root_observation"]),
        dtype=np.float32,
    )
    if state.ndim != 1 or state.size == 0 or not np.isfinite(state).all():
        raise ValueError("state_encoder must return one finite non-empty float vector")
    actions, action_ids = encode_actor_visible_actions(
        loaded_row.row["actor_visible_legal_options"]
    )
    target_keys = set(loaded_row.row["aggregate_q_mean"])
    if target_keys != set(action_ids):
        raise ValueError("actor-visible action encoding and aggregate Q targets disagree")
    q_mean = np.asarray(
        [loaded_row.row["aggregate_q_mean"][key] for key in action_ids],
        dtype=np.float32,
    )
    q_variance = np.asarray(
        [loaded_row.row["q_variance"][key] for key in action_ids],
        dtype=np.float32,
    )
    selected_key = str(loaded_row.row["selected_root_action"])
    if selected_key not in action_ids:
        raise ValueError("selected root action is outside the actor-visible legal set")

    return EncodedQExample(
        state_features=torch.from_numpy(state.copy()),
        action_features=torch.from_numpy(actions.copy()),
        action_mask=torch.ones(len(action_ids), dtype=torch.bool),
        q_mean=torch.from_numpy(q_mean),
        q_variance=torch.from_numpy(q_variance),
        selected_action_index=action_ids.index(selected_key),
        row_sha256=loaded_row.row_sha256,
    )


@dataclass(frozen=True)
class QDistribution:
    mean: torch.Tensor
    variance: torch.Tensor
    log_variance: torch.Tensor


class QDistributionHead(nn.Module):
    """Small heteroscedastic Q head; research classification stays quarantine."""

    classification = Q_FEATURE_CLASSIFICATION

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        state_hidden_dims: tuple[int, ...] = (256, 128),
        action_hidden_dim: int = 128,
        joint_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        dimensions = (state_dim, action_dim, action_hidden_dim, joint_hidden_dim, *state_hidden_dims)
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in dimensions):
            raise ValueError("Q head dimensions must be positive integers")
        self.state_tower = self._build_tower(state_dim, state_hidden_dims)
        self.action_tower = self._build_tower(action_dim, (action_hidden_dim,))
        state_width = state_hidden_dims[-1]
        self.joint = nn.Sequential(
            nn.Linear(state_width + action_hidden_dim, joint_hidden_dim),
            nn.ReLU(),
            nn.Linear(joint_hidden_dim, 2),
        )
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_hidden_dims = state_hidden_dims

    @staticmethod
    def _build_tower(input_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width

        return nn.Sequential(*layers)

    def forward(
        self,
        state_features: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> QDistribution:
        if state_features.ndim != 2 or state_features.shape[1] != self.state_dim:
            raise ValueError("state_features must have shape [batch, state_dim]")
        if (
            action_features.ndim != 3
            or action_features.shape[0] != state_features.shape[0]
            or action_features.shape[2] != self.action_dim
        ):
            raise ValueError("action_features must have shape [batch, actions, action_dim]")
        if action_mask.dtype is not torch.bool or tuple(action_mask.shape) != tuple(action_features.shape[:2]):
            raise ValueError("action_mask must be bool with shape [batch, actions]")
        if not bool(action_mask.any(dim=1).all().item()):
            raise ValueError("every batch row must contain at least one legal action")
        if not bool(torch.isfinite(state_features).all().item()) or not bool(
            torch.isfinite(action_features).all().item()
        ):
            raise ValueError("Q head inputs must be finite")
        batch, action_count, _ = action_features.shape
        state_embedding = self.state_tower(state_features)
        state_embedding = state_embedding[:, None, :].expand(-1, action_count, -1)
        action_embedding = self.action_tower(action_features)
        raw = self.joint(torch.cat((state_embedding, action_embedding), dim=-1))
        mean = torch.sigmoid(raw[..., 0]).masked_fill(~action_mask, 0.0)
        log_variance = raw[..., 1].clamp(LOG_VARIANCE_MIN, LOG_VARIANCE_MAX)
        log_variance = log_variance.masked_fill(~action_mask, 0.0)
        variance = torch.exp(log_variance).masked_fill(~action_mask, 0.0)
        if not bool(torch.isfinite(mean).all().item()) or not bool(torch.isfinite(variance).all().item()):
            raise FloatingPointError("Q distribution head produced non-finite output")

        return QDistribution(mean=mean, variance=variance, log_variance=log_variance)


def masked_gaussian_q_loss(
    distribution: QDistribution,
    target_mean: torch.Tensor,
    target_variance: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    if (
        tuple(target_mean.shape) != tuple(distribution.mean.shape)
        or tuple(target_variance.shape) != tuple(distribution.mean.shape)
        or tuple(action_mask.shape) != tuple(distribution.mean.shape)
        or action_mask.dtype is not torch.bool
    ):
        raise ValueError("Q targets and mask must match the predicted action matrix")
    if not bool(action_mask.any().item()):
        raise ValueError("Q loss requires at least one legal target")
    if not bool(torch.isfinite(target_mean).all().item()) or not bool(
        torch.isfinite(target_variance).all().item()
    ):
        raise ValueError("Q targets must be finite")
    if bool(((target_mean < 0.0) | (target_mean > 1.0)).any().item()):
        raise ValueError("Q target means must be in [0, 1]")
    if bool((target_variance < 0.0).any().item()):
        raise ValueError("Q target variances must be non-negative")
    residual_second_moment = (distribution.mean - target_mean).square() + target_variance
    losses = 0.5 * (distribution.log_variance + residual_second_moment / distribution.variance)

    return losses[action_mask].mean()


def warm_start_q_from_specialist_v(
    q_model: QDistributionHead,
    specialist_v: SnapshotMLP,
) -> dict[str, Any]:
    """Copy the SnapshotMLP representation layers into the Q state tower."""
    if q_model.state_dim != specialist_v.input_dim:
        raise ValueError("specialist-V input dimension does not match the Q state input")
    if q_model.state_hidden_dims != specialist_v.hidden_dims:
        raise ValueError("specialist-V hidden dimensions do not match the Q state tower")
    source_layers = [layer for layer in specialist_v.net if isinstance(layer, nn.Linear)][:-1]
    target_layers = [layer for layer in q_model.state_tower if isinstance(layer, nn.Linear)]
    if len(source_layers) != len(target_layers):
        raise ValueError("specialist-V and Q state towers have different depths")
    copied_parameters = 0
    with torch.no_grad():
        for source, target in zip(source_layers, target_layers, strict=True):
            if source.weight.shape != target.weight.shape or source.bias.shape != target.bias.shape:
                raise ValueError("specialist-V and Q state layer shapes do not match")
            target.weight.copy_(source.weight)
            target.bias.copy_(source.bias)
            copied_parameters += source.weight.numel() + source.bias.numel()

    return {
        "classification": Q_FEATURE_CLASSIFICATION,
        "source_model": "SnapshotMLP",
        "input_dim": specialist_v.input_dim,
        "hidden_dims": list(specialist_v.hidden_dims),
        "copied_parameters": copied_parameters,
    }


@dataclass(frozen=True)
class TrainingProgress:
    epoch: int
    batch_cursor: int
    global_step: int
    presentations: int
    cumulative_gpu_seconds: float
    best_metrics: Mapping[str, float]
    last_metrics: Mapping[str, float]
    safety_counters: Mapping[str, int]

    def validate(self) -> None:
        for field in ("epoch", "batch_cursor", "global_step", "presentations"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"training progress {field} must be a non-negative integer")
        if not isinstance(self.cumulative_gpu_seconds, (int, float)) or isinstance(
            self.cumulative_gpu_seconds, bool
        ):
            raise TypeError("cumulative_gpu_seconds must be numeric")
        if not math.isfinite(float(self.cumulative_gpu_seconds)) or self.cumulative_gpu_seconds < 0.0:
            raise ValueError("cumulative_gpu_seconds must be finite and non-negative")
        for metrics_name in ("best_metrics", "last_metrics"):
            metrics = getattr(self, metrics_name)
            if not isinstance(metrics, Mapping) or any(
                not isinstance(key, str)
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for key, value in metrics.items()
            ):
                raise ValueError(f"{metrics_name} must contain finite numeric metrics")
        if not isinstance(self.safety_counters, Mapping) or any(
            not isinstance(key, str)
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for key, value in self.safety_counters.items()
        ):
            raise ValueError("safety_counters must contain non-negative integer counts")


def _assert_finite_model(model: nn.Module) -> None:
    if any(not bool(torch.isfinite(value).all().item()) for value in model.state_dict().values()):
        raise ValueError("model state contains non-finite tensors")


def _atomic_bytes_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _checkpoint_receipt_path(path: Path) -> Path:
    return path.with_suffix(f"{path.suffix}.sha256.json")


def atomic_save_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    amp_scaler: Any,
    sampler_generator: torch.Generator,
    progress: TrainingProgress,
    dataset_sha256: str,
    config_sha256: str,
    source_sha256s: Sequence[str],
) -> str:
    """Atomically save complete training and RNG state plus a hash receipt."""
    path = Path(path).resolve()
    progress.validate()
    _assert_finite_model(model)
    dataset_sha256 = _validate_sha256(dataset_sha256, "dataset_sha256")
    config_sha256 = _validate_sha256(config_sha256, "config_sha256")
    normalized_sources = tuple(sorted(_validate_sha256(value, "source_sha256") for value in source_sha256s))
    if not normalized_sources or len(normalized_sources) != len(set(normalized_sources)):
        raise ValueError("source_sha256s must be non-empty and unique")
    if not isinstance(sampler_generator, torch.Generator):
        raise TypeError("sampler_generator must be a torch.Generator")
    for component, name in ((scheduler, "scheduler"), (amp_scaler, "amp_scaler")):
        if not hasattr(component, "state_dict") or not hasattr(component, "load_state_dict"):
            raise TypeError(f"{name} must provide state_dict and load_state_dict")
    checkpoint = {
        "schema_version": TRAINING_CHECKPOINT_SCHEMA_VERSION,
        "classification": Q_FEATURE_CLASSIFICATION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "amp_scaler_state": amp_scaler.state_dict(),
        "progress": asdict(progress),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.random.get_rng_state(),
        "torch_cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "sampler_rng_state": sampler_generator.get_state(),
        "dataset_sha256": dataset_sha256,
        "config_sha256": config_sha256,
        "source_sha256s": normalized_sources,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    temporary_path = Path(temporary_name)
    try:
        torch.save(checkpoint, temporary_path)
        with temporary_path.open("r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    checkpoint_sha256 = _sha256_file(path)
    receipt = {
        "schema_version": CHECKPOINT_RECEIPT_SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_sha256,
    }
    _atomic_bytes_write(_checkpoint_receipt_path(path), _canonical_json_bytes(receipt) + b"\n")

    return checkpoint_sha256


def load_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    amp_scaler: Any,
    sampler_generator: torch.Generator,
    expected_dataset_sha256: str,
    expected_config_sha256: str,
    expected_source_sha256s: Sequence[str],
    map_location: str | torch.device = "cpu",
) -> TrainingProgress:
    """Verify identity, restore all states, and return the persisted cursor."""
    path = Path(path).resolve()
    receipt = _read_json_object(_checkpoint_receipt_path(path))
    _require_exact_fields(
        receipt,
        frozenset({"schema_version", "checkpoint_sha256"}),
        "checkpoint receipt",
    )
    if receipt["schema_version"] != CHECKPOINT_RECEIPT_SCHEMA_VERSION:
        raise ValueError("checkpoint receipt schema mismatch")
    if _sha256_file(path) != _validate_sha256(receipt["checkpoint_sha256"], "checkpoint_sha256"):
        raise ValueError("checkpoint hash mismatch")
    expected_dataset_sha256 = _validate_sha256(expected_dataset_sha256, "expected_dataset_sha256")
    expected_config_sha256 = _validate_sha256(expected_config_sha256, "expected_config_sha256")
    expected_sources = tuple(
        sorted(_validate_sha256(value, "expected_source_sha256") for value in expected_source_sha256s)
    )
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise ValueError("training checkpoint could not be decoded") from exc
    required_fields = {
        "schema_version",
        "classification",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "amp_scaler_state",
        "progress",
        "python_rng_state",
        "numpy_rng_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_states",
        "sampler_rng_state",
        "dataset_sha256",
        "config_sha256",
        "source_sha256s",
    }
    if not isinstance(checkpoint, dict) or set(checkpoint) != required_fields:
        raise ValueError("training checkpoint is incomplete or has unexpected fields")
    if (
        checkpoint["schema_version"] != TRAINING_CHECKPOINT_SCHEMA_VERSION
        or checkpoint["classification"] != Q_FEATURE_CLASSIFICATION
    ):
        raise ValueError("training checkpoint schema or classification mismatch")
    if checkpoint["dataset_sha256"] != expected_dataset_sha256:
        raise ValueError("training checkpoint dataset hash mismatch")
    if checkpoint["config_sha256"] != expected_config_sha256:
        raise ValueError("training checkpoint config hash mismatch")
    if tuple(checkpoint["source_sha256s"]) != expected_sources:
        raise ValueError("training checkpoint source hash mismatch")
    progress = TrainingProgress(**checkpoint["progress"])
    progress.validate()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    _assert_finite_model(model)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    amp_scaler.load_state_dict(checkpoint["amp_scaler_state"])
    if not isinstance(sampler_generator, torch.Generator):
        raise TypeError("sampler_generator must be a torch.Generator")
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.random.set_rng_state(checkpoint["torch_cpu_rng_state"].cpu())
    cuda_states = checkpoint["torch_cuda_rng_states"]
    if cuda_states:
        if not torch.cuda.is_available():
            raise ValueError("checkpoint has CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])
    sampler_generator.set_state(checkpoint["sampler_rng_state"].cpu())

    return progress


@dataclass(frozen=True)
class StreamingShardIdentity:
    ordinal: int
    miner_id: str
    source_shard_index: int
    split: str
    shard: str
    manifest: str
    shard_sha256: str
    manifest_sha256: str
    row_count: int

    def validate(self) -> None:
        for value, name in (
            (self.ordinal, "ordinal"),
            (self.source_shard_index, "source_shard_index"),
            (self.row_count, "row_count"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"streaming shard {name} must be a non-negative integer")
        if self.row_count == 0:
            raise ValueError("streaming shard row_count must be positive")
        for value, name in (
            (self.miner_id, "miner_id"),
            (self.split, "split"),
            (self.shard, "shard"),
            (self.manifest, "manifest"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"streaming shard {name} must be a non-empty string")
        for value, name in ((self.shard, "shard"), (self.manifest, "manifest")):
            relative = Path(value)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"streaming shard {name} must be a safe relative path")
        _validate_sha256(self.shard_sha256, "streaming shard_sha256")
        _validate_sha256(self.manifest_sha256, "streaming manifest_sha256")

    def to_dict(self) -> dict[str, Any]:
        self.validate()

        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StreamingShardIdentity:
        expected = frozenset({
            "ordinal",
            "miner_id",
            "source_shard_index",
            "split",
            "shard",
            "manifest",
            "shard_sha256",
            "manifest_sha256",
            "row_count",
        })
        _require_exact_fields(value, expected, "streaming shard identity")
        identity = cls(**dict(value))
        identity.validate()

        return identity


@dataclass(frozen=True)
class BoundCorpusShard:
    identity: StreamingShardIdentity
    shard_path: Path
    manifest_path: Path
    row_sha256s: tuple[str, ...]
    m3_consumed_row_offsets: tuple[int, ...]
    unseen_row_offsets: tuple[int, ...]


@dataclass(frozen=True)
class ManifestBoundCorpus:
    root_manifest_path: Path
    root_manifest_sha256: str
    corpus_lineage_id: str
    generation: str
    ordered_shard_identity_sha256: str
    source_compatibility: Mapping[str, Any]
    trainer_compatibility: Mapping[str, str]
    compatibility_sha256: str
    shards: tuple[BoundCorpusShard, ...]
    dataset: ManifestBoundTeacherDataset
    m3_tranche_dataset_sha256: str
    ordered_m3_consumed_row_sha256: str
    m3_historical_row_sha256s: tuple[str, ...]
    m3_source_mapping_sha256: str

    @property
    def ordered_shards(self) -> tuple[StreamingShardIdentity, ...]:
        return tuple(shard.identity for shard in self.shards)

    @property
    def total_rows(self) -> int:
        return sum(shard.identity.row_count for shard in self.shards)

    @property
    def historical_row_cursors(self) -> dict[str, int]:
        return {
            shard.identity.shard_sha256: len(shard.m3_consumed_row_offsets)
            for shard in self.shards
        }

    @property
    def historical_row_sha256s(self) -> tuple[str, ...]:
        return self.m3_historical_row_sha256s


def _validate_sha256_sequence(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    normalized = [_validate_sha256(item, name) for item in value]
    if len(normalized) != len(set(normalized)) or normalized != sorted(normalized):
        raise ValueError(f"{name} must be unique and sorted")

    return normalized


def _validate_source_corpus_compatibility(value: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_fields(value, _SOURCE_CORPUS_COMPATIBILITY_FIELDS, "source corpus compatibility")
    if value["teacher_manifest_schema_version"] != TEACHER_MANIFEST_SCHEMA_VERSION:
        raise ValueError("corpus teacher-manifest schema is incompatible")
    miner_config = value["miner_config_sha256s"]
    if not isinstance(miner_config, Mapping) or not miner_config:
        raise ValueError("miner_config_sha256s must be a non-empty mapping")
    normalized_miner_config = {}
    for miner_id, config_sha256 in miner_config.items():
        if not isinstance(miner_id, str) or not miner_id:
            raise ValueError("miner ids must be non-empty strings")
        normalized_miner_config[miner_id] = _validate_sha256(config_sha256, "miner config SHA-256")
    split_values = value["split_values"]
    if (
        not isinstance(split_values, list)
        or not split_values
        or any(not isinstance(item, str) or not item for item in split_values)
        or len(split_values) != len(set(split_values))
        or split_values != sorted(split_values)
    ):
        raise ValueError("split_values must be non-empty, unique, and sorted")

    return {
        "teacher_manifest_schema_version": TEACHER_MANIFEST_SCHEMA_VERSION,
        "teacher_config_sha256s": _validate_sha256_sequence(
            value["teacher_config_sha256s"], "teacher_config_sha256s"
        ),
        "label_v_checkpoint_sha256s": _validate_sha256_sequence(
            value["label_v_checkpoint_sha256s"], "label_v_checkpoint_sha256s"
        ),
        "label_v_calibrator_sha256s": _validate_sha256_sequence(
            value["label_v_calibrator_sha256s"], "label_v_calibrator_sha256s"
        ),
        "miner_config_sha256s": dict(sorted(normalized_miner_config.items())),
        "split_values": list(split_values),
    }


def _validate_trainer_compatibility(value: Mapping[str, Any]) -> dict[str, str]:
    _require_exact_fields(value, _TRAINER_COMPATIBILITY_FIELDS, "trainer compatibility")
    normalized = {str(key): str(item) for key, item in value.items()}
    if normalized["trainer_schema_version"] != STREAMING_TRAINER_SCHEMA_VERSION:
        raise ValueError("streaming trainer schema is incompatible")
    for name in (
        "state_feature_schema_sha256",
        "action_feature_schema_sha256",
        "deck_manifest_sha256",
        "split_policy_sha256",
    ):
        normalized[name] = _validate_sha256(normalized[name], name)

    return normalized


def _expand_offset_ranges(value: Any, row_count: int, name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    offsets = []
    for item in value:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or any(isinstance(offset, bool) or not isinstance(offset, int) for offset in item)
        ):
            raise ValueError(f"{name} entries must be two-integer inclusive ranges")
        start, end = item
        if start < 0 or end < start or end >= row_count:
            raise ValueError(f"{name} contains an out-of-range interval")
        offsets.extend(range(start, end + 1))
    if offsets != sorted(set(offsets)):
        raise ValueError(f"{name} ranges must be ordered and non-overlapping")

    return tuple(offsets)


def _validate_corpus_gates(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("root corpus gates must be a mapping")
    _require_exact_fields(value, _CORPUS_GATE_FIELDS, "root corpus gates")
    required_true = _CORPUS_GATE_FIELDS - {"labels_modified", "policy_control_eligible"}
    if any(value[name] is not True for name in required_true):
        raise ValueError("root corpus has an unsatisfied safety or completion gate")
    if value["labels_modified"] is not False or value["policy_control_eligible"] is not False:
        raise ValueError("root corpus label or policy-control quarantine gate mismatch")


def _resolve_corpus_relative(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"corpus {name} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"corpus {name} must be a safe relative path")
    if not relative.parts or relative.parts[0] != "runs" or len(relative.parts) == 1:
        raise ValueError(f"corpus {name} must be in the frozen runs/ namespace")
    relative = Path(*relative.parts[1:])
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"corpus {name} escapes the explicit corpus root") from exc

    return resolved


def load_manifest_bound_corpus(
    root_manifest_path: Path,
    *,
    corpus_root: Path,
    m3_tranche_binding: ShardBinding,
    expected_trainer_compatibility: Mapping[str, Any],
) -> ManifestBoundCorpus:
    """Load the authoritative frozen-final corpus and bind trainer compatibility."""
    root_manifest_path = Path(root_manifest_path).resolve()
    corpus_root = Path(corpus_root).resolve()
    root = _read_json_object(root_manifest_path)
    _require_exact_fields(root, _CORPUS_MANIFEST_FIELDS, "root corpus manifest")
    if root["schema_version"] != CORPUS_MANIFEST_SCHEMA_VERSION:
        raise ValueError("root corpus manifest schema mismatch")
    if root["classification"] != Q_FEATURE_CLASSIFICATION:
        raise ValueError("root corpus manifest classification mismatch")
    if root["status"] != "frozen_final":
        raise ValueError("root corpus manifest is not frozen final")
    _validate_corpus_gates(root["gates"])
    lineage = root["corpus_lineage_id"]
    if (
        not isinstance(lineage, str)
        or not lineage.startswith("qcl1-")
        or len(lineage) != 37
        or any(character not in "0123456789abcdef" for character in lineage[5:].lower())
    ):
        raise ValueError("corpus_lineage_id is invalid")
    generation = root["generation"]
    if not isinstance(generation, str) or not generation:
        raise ValueError("corpus generation must be a non-empty string")
    if not isinstance(root["compatibility"], Mapping):
        raise TypeError("root corpus compatibility must be a mapping")
    source_compatibility = _validate_source_corpus_compatibility(root["compatibility"])
    trainer_compatibility = _validate_trainer_compatibility(expected_trainer_compatibility)
    compatibility_payload = {
        "source_manifest_compatibility": source_compatibility,
        "trainer_compatibility": trainer_compatibility,
    }
    compatibility_sha256 = _sha256_bytes(_canonical_json_bytes(compatibility_payload))
    entries = root["ordered_shards"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("root corpus must contain at least one ordered shard")
    coverage = root["coverage"]
    if not isinstance(coverage, Mapping):
        raise TypeError("root corpus coverage must be a mapping")
    total_rows = coverage.get("rows")
    if isinstance(total_rows, bool) or not isinstance(total_rows, int) or total_rows <= 0:
        raise ValueError("root corpus rows must be a positive integer")
    if coverage.get("eligible_rows") != total_rows:
        raise ValueError("root corpus eligible_rows and rows must agree")
    ordered_identity_sha256 = _validate_sha256(
        root["ordered_shard_identity_sha256"], "ordered_shard_identity_sha256"
    )
    identities: list[StreamingShardIdentity] = []
    bindings: list[ShardBinding] = []
    entry_details: list[tuple[Path, Path, tuple[int, ...], tuple[int, ...]]] = []
    ordered_identity_payload = []
    source_keys: set[tuple[str, int]] = set()
    for ordinal, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, Mapping):
            raise TypeError("root corpus ordered_shards must contain mappings")
        _require_exact_fields(raw_entry, _CORPUS_SHARD_FIELDS, "root corpus shard")
        identity = StreamingShardIdentity(
            ordinal=raw_entry["ordinal"],
            miner_id=raw_entry["miner_id"],
            source_shard_index=raw_entry["source_shard_index"],
            split=raw_entry["split"],
            shard=raw_entry["shard"],
            manifest=raw_entry["manifest"],
            shard_sha256=str(raw_entry["shard_sha256"]).lower(),
            manifest_sha256=str(raw_entry["manifest_sha256"]).lower(),
            row_count=raw_entry["row_count"],
        )
        identity.validate()
        if identity.ordinal != ordinal:
            raise ValueError("root corpus shard ordinals must be contiguous and ordered")
        source_key = (identity.miner_id, identity.source_shard_index)
        if source_key in source_keys:
            raise ValueError("root corpus source shard is bound more than once")
        source_keys.add(source_key)
        consumed_offsets = _expand_offset_ranges(
            raw_entry["m3_consumed_row_offsets_inclusive"], identity.row_count, "M3 consumed offsets"
        )
        unseen_offsets = _expand_offset_ranges(
            raw_entry["unseen_row_offsets_inclusive"], identity.row_count, "unseen offsets"
        )
        if len(consumed_offsets) != raw_entry["m3_consumed_row_count"]:
            raise ValueError("M3 consumed offset coverage disagrees with its count")
        if len(unseen_offsets) != raw_entry["unseen_row_count"]:
            raise ValueError("unseen offset coverage disagrees with its count")
        if set(consumed_offsets) & set(unseen_offsets) or sorted(consumed_offsets + unseen_offsets) != list(
            range(identity.row_count)
        ):
            raise ValueError("M3 consumed and unseen offsets must partition each shard")
        if consumed_offsets != tuple(range(len(consumed_offsets))):
            raise ValueError("historical M3 membership must be an exact shard prefix")
        shard_path = _resolve_corpus_relative(corpus_root, identity.shard, "shard")
        manifest_path = _resolve_corpus_relative(corpus_root, identity.manifest, "manifest")
        if _sha256_file(shard_path) != identity.shard_sha256:
            raise ValueError("root corpus shard hash mismatch")
        if _sha256_file(manifest_path) != identity.manifest_sha256:
            raise ValueError("root corpus sidecar hash mismatch")
        sidecar = _read_json_object(manifest_path)
        if sidecar.get("split") != identity.split or sidecar.get("row_count") != identity.row_count:
            raise ValueError("root corpus shard identity disagrees with its sidecar")
        identities.append(identity)
        entry_details.append((shard_path, manifest_path, consumed_offsets, unseen_offsets))
        bindings.append(ShardBinding(shard_path, manifest_path))
        ordered_identity_payload.append({
            "ordinal": identity.ordinal,
            "miner_id": identity.miner_id,
            "source_shard_index": identity.source_shard_index,
            "split": identity.split,
            "shard": identity.shard,
            "manifest": identity.manifest,
            "shard_sha256": identity.shard_sha256,
            "manifest_sha256": identity.manifest_sha256,
            "row_count": identity.row_count,
        })
    if _sha256_bytes(_canonical_json_bytes(ordered_identity_payload)) != ordered_identity_sha256:
        raise ValueError("root corpus ordered-shard identity hash mismatch")
    if sum(identity.row_count for identity in identities) != total_rows:
        raise ValueError("root corpus rows disagree with ordered shard coverage")
    dataset = load_manifest_bound_shards(bindings)
    bound_shards = []
    row_offset = 0
    for identity, details in zip(identities, entry_details, strict=True):
        shard_path, manifest_path, consumed_offsets, unseen_offsets = details
        next_offset = row_offset + identity.row_count
        row_hashes = tuple(row.row_sha256 for row in dataset.rows[row_offset:next_offset])
        if len(row_hashes) != identity.row_count:
            raise ValueError("root corpus row slicing disagrees with shard coverage")
        bound_shards.append(
            BoundCorpusShard(
                identity=identity,
                shard_path=shard_path,
                manifest_path=manifest_path,
                row_sha256s=row_hashes,
                m3_consumed_row_offsets=consumed_offsets,
                unseen_row_offsets=unseen_offsets,
            )
        )
        row_offset = next_offset
    consumption = root["m3_consumption"]
    if not isinstance(consumption, Mapping):
        raise TypeError("root corpus m3_consumption must be a mapping")
    _require_exact_fields(consumption, _M3_CONSUMPTION_FIELDS, "root corpus M3 consumption")
    corpus_consumed_hashes = tuple(
        shard.row_sha256s[offset]
        for shard in bound_shards
        for offset in shard.m3_consumed_row_offsets
    )
    consumed_count = len(corpus_consumed_hashes)
    if (
        consumption["consumed_unique_rows"] != consumed_count
        or consumption["unseen_unique_rows"] != total_rows - consumed_count
        or consumption["final_immutable_rows"] != total_rows
    ):
        raise ValueError("root corpus M3 consumption counts disagree with ordered shards")
    expected_consumed_hash = _validate_sha256(
        consumption["ordered_consumed_row_sha256"], "ordered_consumed_row_sha256"
    )
    tranche_dataset = load_manifest_bound_shards([m3_tranche_binding])
    tranche_hashes = tuple(row.row_sha256 for row in tranche_dataset.rows)
    if tranche_dataset.dataset_sha256 != _validate_sha256(
        consumption["tranche_dataset_sha256"], "tranche_dataset_sha256"
    ):
        raise ValueError("frozen M3 tranche dataset hash mismatch")
    if _sha256_bytes(_canonical_json_bytes(list(tranche_hashes))) != expected_consumed_hash:
        raise ValueError("frozen M3 tranche ordered-row hash mismatch")
    if len(tranche_hashes) != len(set(tranche_hashes)) or set(tranche_hashes) != set(corpus_consumed_hashes):
        raise ValueError("frozen M3 tranche membership disagrees with corpus consumed offsets")
    corpus_location_by_row = {
        row_sha256: (shard.identity.miner_id, shard.identity.source_shard_index, offset)
        for shard in bound_shards
        for offset, row_sha256 in enumerate(shard.row_sha256s)
        if offset in shard.m3_consumed_row_offsets
    }
    source_mapping = []
    for tranche_position, row_sha256 in enumerate(tranche_hashes):
        miner_id, shard_index, row_offset = corpus_location_by_row[row_sha256]
        source_mapping.append({
            "tranche_position": tranche_position,
            "miner_id": miner_id,
            "shard_index": shard_index,
            "row_offset": row_offset,
            "row_sha256": row_sha256,
        })
    source_mapping_sha256 = _sha256_bytes(_canonical_json_bytes(source_mapping))
    expected_percentage = 100.0 * consumed_count / total_rows
    percentage = consumption["consumed_percentage"]
    if (
        isinstance(percentage, bool)
        or not isinstance(percentage, (int, float))
        or not math.isfinite(float(percentage))
        or not math.isclose(float(percentage), expected_percentage, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("root corpus consumed percentage mismatch")

    return ManifestBoundCorpus(
        root_manifest_path=root_manifest_path,
        root_manifest_sha256=_sha256_file(root_manifest_path),
        corpus_lineage_id=lineage,
        generation=generation,
        ordered_shard_identity_sha256=ordered_identity_sha256,
        source_compatibility=source_compatibility,
        trainer_compatibility=trainer_compatibility,
        compatibility_sha256=compatibility_sha256,
        shards=tuple(bound_shards),
        dataset=dataset,
        m3_tranche_dataset_sha256=tranche_dataset.dataset_sha256,
        ordered_m3_consumed_row_sha256=expected_consumed_hash,
        m3_historical_row_sha256s=tranche_hashes,
        m3_source_mapping_sha256=source_mapping_sha256,
    )


@dataclass(frozen=True)
class ReplayPolicy:
    mode: str = "unseen_only"

    def validate(self) -> None:
        if self.mode not in {"unseen_only", "unseen_first_then_replay"}:
            raise ValueError("replay policy mode must be unseen_only or unseen_first_then_replay")

    @property
    def allows_replay(self) -> bool:
        return self.mode == "unseen_first_then_replay"

    def to_dict(self) -> dict[str, str]:
        self.validate()

        return {"schema_version": REPLAY_POLICY_SCHEMA_VERSION, "mode": self.mode}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReplayPolicy:
        _require_exact_fields(value, frozenset({"schema_version", "mode"}), "replay policy")
        if value["schema_version"] != REPLAY_POLICY_SCHEMA_VERSION:
            raise ValueError("replay policy schema mismatch")
        policy = cls(mode=value["mode"])
        policy.validate()

        return policy


@dataclass(frozen=True)
class RowSplitPolicy:
    modulus: int = 5
    heldout_residues: tuple[int, ...] = (0,)
    algorithm: str = "row_sha256_modulo"
    schema_version: str = ROW_SPLIT_POLICY_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != ROW_SPLIT_POLICY_SCHEMA_VERSION:
            raise ValueError("row split policy schema mismatch")
        if self.algorithm != "row_sha256_modulo":
            raise ValueError("row split policy algorithm mismatch")
        if isinstance(self.modulus, bool) or not isinstance(self.modulus, int) or self.modulus < 2:
            raise ValueError("row split modulus must be an integer of at least two")
        if (
            not self.heldout_residues
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < self.modulus
                for value in self.heldout_residues
            )
            or tuple(sorted(set(self.heldout_residues))) != self.heldout_residues
            or len(self.heldout_residues) >= self.modulus
        ):
            raise ValueError("heldout residues must be unique, ordered, and leave training coverage")

    @property
    def sha256(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.to_dict()))

    def split_for_row(self, row_sha256: str) -> str:
        self.validate()
        normalized = _validate_sha256(row_sha256, "row_sha256")
        residue = int(normalized, 16) % self.modulus

        return "heldout" if residue in self.heldout_residues else "train"

    def to_dict(self) -> dict[str, Any]:
        self.validate()

        return {
            "schema_version": self.schema_version,
            "algorithm": self.algorithm,
            "modulus": self.modulus,
            "heldout_residues": list(self.heldout_residues),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RowSplitPolicy:
        _require_exact_fields(
            value,
            frozenset({"schema_version", "algorithm", "modulus", "heldout_residues"}),
            "row split policy",
        )
        residues = value["heldout_residues"]
        if not isinstance(residues, list):
            raise TypeError("row split heldout_residues must be a list")
        policy = cls(
            schema_version=value["schema_version"],
            algorithm=value["algorithm"],
            modulus=value["modulus"],
            heldout_residues=tuple(residues),
        )
        policy.validate()

        return policy


_STREAMING_STATE_FIELDS = frozenset({
    "schema_version",
    "trainer_schema_version",
    "root_corpus_manifest_sha256",
    "corpus_lineage_id",
    "corpus_generation",
    "ordered_shard_identity_sha256",
    "compatibility_sha256",
    "row_split_policy_sha256",
    "ordered_shards",
    "consumed_shard_sha256s",
    "per_shard_row_cursor",
    "unique_row_splits",
    "total_unique_rows_consumed",
    "unique_train_rows",
    "unique_heldout_rows",
    "legacy_presentations",
    "unseen_presentations",
    "replay_presentations",
    "total_presentations",
    "replay_policy",
    "legacy_dataset_sha256",
    "legacy_mapping_sha256",
    "legacy_source_mapping_sha256",
    "legacy_checkpoint_sha256",
    "legacy_config_sha256",
    "legacy_source_sha256s",
    "dataloader_rng_origin",
    "dataloader_rng_seed",
})


@dataclass(frozen=True)
class StreamingTrainingState:
    root_corpus_manifest_sha256: str
    corpus_lineage_id: str
    corpus_generation: str
    ordered_shard_identity_sha256: str
    compatibility_sha256: str
    row_split_policy_sha256: str
    ordered_shards: tuple[StreamingShardIdentity, ...]
    consumed_shard_sha256s: tuple[str, ...]
    per_shard_row_cursor: tuple[tuple[str, int], ...]
    unique_row_splits: tuple[tuple[str, str], ...]
    total_unique_rows_consumed: int
    unique_train_rows: int
    unique_heldout_rows: int
    legacy_presentations: int
    unseen_presentations: int
    replay_presentations: int
    replay_policy: ReplayPolicy
    legacy_dataset_sha256: str | None = None
    legacy_mapping_sha256: str | None = None
    legacy_source_mapping_sha256: str | None = None
    legacy_checkpoint_sha256: str | None = None
    legacy_config_sha256: str | None = None
    legacy_source_sha256s: tuple[str, ...] = ()
    dataloader_rng_origin: str = "native"
    dataloader_rng_seed: int | None = None
    schema_version: str = STREAMING_TRAINING_STATE_SCHEMA_VERSION
    trainer_schema_version: str = STREAMING_TRAINER_SCHEMA_VERSION

    @property
    def cursor_by_shard(self) -> dict[str, int]:
        return dict(self.per_shard_row_cursor)

    @property
    def total_presentations(self) -> int:
        return self.legacy_presentations + self.unseen_presentations + self.replay_presentations

    def validate(self) -> None:
        if self.schema_version != STREAMING_TRAINING_STATE_SCHEMA_VERSION:
            raise ValueError("streaming training-state schema mismatch")
        if self.trainer_schema_version != STREAMING_TRAINER_SCHEMA_VERSION:
            raise ValueError("streaming trainer schema mismatch")
        for value, name in (
            (self.root_corpus_manifest_sha256, "root_corpus_manifest_sha256"),
            (self.ordered_shard_identity_sha256, "ordered_shard_identity_sha256"),
            (self.compatibility_sha256, "compatibility_sha256"),
            (self.row_split_policy_sha256, "row_split_policy_sha256"),
        ):
            _validate_sha256(value, name)
        if not isinstance(self.corpus_lineage_id, str) or not self.corpus_lineage_id.startswith("qcl1-"):
            raise ValueError("streaming corpus_lineage_id is invalid")
        if not isinstance(self.corpus_generation, str) or not self.corpus_generation:
            raise ValueError("streaming corpus_generation must be a non-empty string")
        if not self.ordered_shards:
            raise ValueError("streaming state requires at least one ordered shard")
        shard_hashes = []
        for ordinal, shard in enumerate(self.ordered_shards):
            shard.validate()
            if shard.ordinal != ordinal:
                raise ValueError("streaming state shard ordinals are not contiguous")
            shard_hashes.append(shard.shard_sha256)
        if len(shard_hashes) != len(set(shard_hashes)):
            raise ValueError("streaming state contains duplicate shard hashes")
        cursor = self.cursor_by_shard
        if len(cursor) != len(self.per_shard_row_cursor) or tuple(cursor) != tuple(shard_hashes):
            raise ValueError("streaming per-shard cursor must exactly follow ordered shards")
        for shard in self.ordered_shards:
            offset = cursor[shard.shard_sha256]
            if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset <= shard.row_count:
                raise ValueError("streaming per-shard row cursor is outside shard coverage")
        expected_consumed = tuple(
            shard.shard_sha256
            for shard in self.ordered_shards
            if cursor[shard.shard_sha256] == shard.row_count
        )
        if self.consumed_shard_sha256s != expected_consumed:
            raise ValueError("streaming consumed-shard inventory disagrees with row cursors")
        expected_unique = sum(cursor.values())
        if self.total_unique_rows_consumed != expected_unique:
            raise ValueError("streaming unique-row count disagrees with row cursors")
        if len(self.unique_row_splits) != self.total_unique_rows_consumed:
            raise ValueError("streaming unique-row membership disagrees with row cursors")
        row_hashes = []
        for row_sha256, split in self.unique_row_splits:
            row_hashes.append(_validate_sha256(row_sha256, "streaming unique row SHA-256"))
            if split not in {"train", "heldout"}:
                raise ValueError("streaming unique row split must be train or heldout")
        if len(row_hashes) != len(set(row_hashes)):
            raise ValueError("streaming unique-row membership contains a duplicate")
        expected_train = sum(split == "train" for _, split in self.unique_row_splits)
        expected_heldout = sum(split == "heldout" for _, split in self.unique_row_splits)
        if self.unique_train_rows != expected_train or self.unique_heldout_rows != expected_heldout:
            raise ValueError("streaming train/heldout unique counters disagree with membership")
        if self.total_unique_rows_consumed != self.unique_train_rows + self.unique_heldout_rows:
            raise ValueError("streaming total unique rows must equal train plus heldout")
        for value, name in (
            (self.total_unique_rows_consumed, "total_unique_rows_consumed"),
            (self.unique_train_rows, "unique_train_rows"),
            (self.unique_heldout_rows, "unique_heldout_rows"),
            (self.legacy_presentations, "legacy_presentations"),
            (self.unseen_presentations, "unseen_presentations"),
            (self.replay_presentations, "replay_presentations"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"streaming {name} must be a non-negative integer")
        if self.legacy_presentations and not self.total_unique_rows_consumed:
            raise ValueError("legacy presentations require an exact historical row cursor")
        lineage_values = (
            self.legacy_dataset_sha256,
            self.legacy_mapping_sha256,
            self.legacy_source_mapping_sha256,
            self.legacy_checkpoint_sha256,
            self.legacy_config_sha256,
        )
        if any(value is None for value in lineage_values) and any(value is not None for value in lineage_values):
            raise ValueError("legacy dataset, split mapping, and source mapping identities must be present together")
        for value, name in zip(
            lineage_values,
            (
                "legacy_dataset_sha256",
                "legacy_mapping_sha256",
                "legacy_source_mapping_sha256",
                "legacy_checkpoint_sha256",
                "legacy_config_sha256",
            ),
            strict=True,
        ):
            if value is not None:
                _validate_sha256(value, name)
        normalized_legacy_sources = tuple(
            _validate_sha256(value, "legacy_source_sha256") for value in self.legacy_source_sha256s
        )
        if lineage_values[0] is None and normalized_legacy_sources:
            raise ValueError("legacy source identities require complete legacy lineage")
        if lineage_values[0] is not None and (
            not normalized_legacy_sources
            or normalized_legacy_sources != tuple(sorted(set(normalized_legacy_sources)))
        ):
            raise ValueError("legacy source identities must be non-empty, unique, and sorted")
        if self.legacy_presentations and self.legacy_dataset_sha256 is None:
            raise ValueError("legacy presentations require preserved dataset and mapping identities")
        if self.dataloader_rng_origin not in {"native", "initialized_at_v1_migration"}:
            raise ValueError("streaming dataloader RNG origin is invalid")
        if self.dataloader_rng_origin == "native" and self.dataloader_rng_seed is not None:
            raise ValueError("native dataloader RNG must not claim a migration seed")
        if self.dataloader_rng_origin == "initialized_at_v1_migration" and (
            isinstance(self.dataloader_rng_seed, bool)
            or not isinstance(self.dataloader_rng_seed, int)
            or self.dataloader_rng_seed < 0
        ):
            raise ValueError("migrated dataloader RNG requires a non-negative deterministic seed")
        self.replay_policy.validate()
        if not self.replay_policy.allows_replay and self.replay_presentations:
            raise ValueError("replay presentations require an explicit replay policy")

    def to_dict(self) -> dict[str, Any]:
        self.validate()

        return {
            "schema_version": self.schema_version,
            "trainer_schema_version": self.trainer_schema_version,
            "root_corpus_manifest_sha256": self.root_corpus_manifest_sha256,
            "corpus_lineage_id": self.corpus_lineage_id,
            "corpus_generation": self.corpus_generation,
            "ordered_shard_identity_sha256": self.ordered_shard_identity_sha256,
            "compatibility_sha256": self.compatibility_sha256,
            "row_split_policy_sha256": self.row_split_policy_sha256,
            "ordered_shards": [shard.to_dict() for shard in self.ordered_shards],
            "consumed_shard_sha256s": list(self.consumed_shard_sha256s),
            "per_shard_row_cursor": [list(item) for item in self.per_shard_row_cursor],
            "unique_row_splits": [list(item) for item in self.unique_row_splits],
            "total_unique_rows_consumed": self.total_unique_rows_consumed,
            "unique_train_rows": self.unique_train_rows,
            "unique_heldout_rows": self.unique_heldout_rows,
            "legacy_presentations": self.legacy_presentations,
            "unseen_presentations": self.unseen_presentations,
            "replay_presentations": self.replay_presentations,
            "total_presentations": self.total_presentations,
            "replay_policy": self.replay_policy.to_dict(),
            "legacy_dataset_sha256": self.legacy_dataset_sha256,
            "legacy_mapping_sha256": self.legacy_mapping_sha256,
            "legacy_source_mapping_sha256": self.legacy_source_mapping_sha256,
            "legacy_checkpoint_sha256": self.legacy_checkpoint_sha256,
            "legacy_config_sha256": self.legacy_config_sha256,
            "legacy_source_sha256s": list(self.legacy_source_sha256s),
            "dataloader_rng_origin": self.dataloader_rng_origin,
            "dataloader_rng_seed": self.dataloader_rng_seed,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StreamingTrainingState:
        _require_exact_fields(value, _STREAMING_STATE_FIELDS, "streaming training state")
        raw_shards = value["ordered_shards"]
        raw_cursor = value["per_shard_row_cursor"]
        raw_row_splits = value["unique_row_splits"]
        if not isinstance(raw_shards, list) or not isinstance(raw_cursor, list) or not isinstance(raw_row_splits, list):
            raise TypeError("streaming shard inventory, cursor, and row membership must be lists")
        cursor = []
        for item in raw_cursor:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError("streaming per-shard cursor entries must be two-item lists")
            cursor.append((item[0], item[1]))
        row_splits = []
        for item in raw_row_splits:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError("streaming unique-row split entries must be two-item lists")
            row_splits.append((item[0], item[1]))
        if not isinstance(value["replay_policy"], Mapping):
            raise TypeError("streaming replay policy must be a mapping")
        state = cls(
            schema_version=value["schema_version"],
            trainer_schema_version=value["trainer_schema_version"],
            root_corpus_manifest_sha256=value["root_corpus_manifest_sha256"],
            corpus_lineage_id=value["corpus_lineage_id"],
            corpus_generation=value["corpus_generation"],
            ordered_shard_identity_sha256=value["ordered_shard_identity_sha256"],
            compatibility_sha256=value["compatibility_sha256"],
            row_split_policy_sha256=value["row_split_policy_sha256"],
            ordered_shards=tuple(StreamingShardIdentity.from_dict(item) for item in raw_shards),
            consumed_shard_sha256s=tuple(value["consumed_shard_sha256s"]),
            per_shard_row_cursor=tuple(cursor),
            unique_row_splits=tuple(row_splits),
            total_unique_rows_consumed=value["total_unique_rows_consumed"],
            unique_train_rows=value["unique_train_rows"],
            unique_heldout_rows=value["unique_heldout_rows"],
            legacy_presentations=value["legacy_presentations"],
            unseen_presentations=value["unseen_presentations"],
            replay_presentations=value["replay_presentations"],
            replay_policy=ReplayPolicy.from_dict(value["replay_policy"]),
            legacy_dataset_sha256=value["legacy_dataset_sha256"],
            legacy_mapping_sha256=value["legacy_mapping_sha256"],
            legacy_source_mapping_sha256=value["legacy_source_mapping_sha256"],
            legacy_checkpoint_sha256=value["legacy_checkpoint_sha256"],
            legacy_config_sha256=value["legacy_config_sha256"],
            legacy_source_sha256s=tuple(value["legacy_source_sha256s"]),
            dataloader_rng_origin=value["dataloader_rng_origin"],
            dataloader_rng_seed=value["dataloader_rng_seed"],
        )
        state.validate()
        if value["total_presentations"] != state.total_presentations:
            raise ValueError("streaming serialized total presentations disagrees with counters")

        return state


def _consumed_shards(
    ordered_shards: Sequence[StreamingShardIdentity],
    cursor: Mapping[str, int],
) -> tuple[str, ...]:
    return tuple(
        shard.shard_sha256
        for shard in ordered_shards
        if cursor[shard.shard_sha256] == shard.row_count
    )


def _historical_split_mapping_sha256(row_splits: Sequence[tuple[str, str]]) -> str:
    payload = {
        "schema_version": M3_HISTORICAL_SPLIT_SCHEMA_VERSION,
        "train_row_sha256s": sorted(row_sha256 for row_sha256, split in row_splits if split == "train"),
        "heldout_row_sha256s": sorted(
            row_sha256 for row_sha256, split in row_splits if split == "heldout"
        ),
    }

    return _sha256_bytes(_canonical_json_bytes(payload))


def initialize_streaming_training_state(
    corpus: ManifestBoundCorpus,
    *,
    replay_policy: ReplayPolicy,
    row_split_policy: RowSplitPolicy,
    historical_row_cursors: Mapping[str, int] | None = None,
    historical_row_splits: Mapping[str, str] | None = None,
    legacy_presentations: int = 0,
    legacy_dataset_sha256: str | None = None,
    legacy_mapping_sha256: str | None = None,
    legacy_checkpoint_sha256: str | None = None,
    legacy_config_sha256: str | None = None,
    legacy_source_sha256s: Sequence[str] = (),
    dataloader_rng_origin: str = "native",
    dataloader_rng_seed: int | None = None,
) -> StreamingTrainingState:
    replay_policy.validate()
    row_split_policy.validate()
    if corpus.trainer_compatibility["split_policy_sha256"] != row_split_policy.sha256:
        raise ValueError("row split policy disagrees with bound trainer compatibility")
    supplied = dict(historical_row_cursors or {})
    known_hashes = {shard.identity.shard_sha256 for shard in corpus.shards}
    if set(supplied) - known_hashes:
        raise ValueError("historical cursor names a shard outside the root corpus")
    cursor = tuple(
        (shard.identity.shard_sha256, supplied.get(shard.identity.shard_sha256, 0))
        for shard in corpus.shards
    )
    cursor_map = dict(cursor)
    for shard in corpus.shards:
        offset = cursor_map[shard.identity.shard_sha256]
        if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset <= shard.identity.row_count:
            raise ValueError("historical row cursor is outside shard coverage")
    if legacy_presentations and cursor_map != corpus.historical_row_cursors:
        raise ValueError("legacy M3 cursor must exactly match audited root-manifest membership")
    corpus_order_consumed_hashes = tuple(
        shard.row_sha256s[offset]
        for shard in corpus.shards
        for offset in range(cursor_map[shard.identity.shard_sha256])
    )
    ordered_consumed_hashes = (
        corpus.historical_row_sha256s
        if cursor_map == corpus.historical_row_cursors
        else corpus_order_consumed_hashes
    )
    if set(ordered_consumed_hashes) != set(corpus_order_consumed_hashes):
        raise ValueError("historical tranche order does not match the supplied corpus cursor")
    supplied_splits = dict(historical_row_splits or {})
    if set(supplied_splits) != set(ordered_consumed_hashes):
        raise ValueError("historical row-split mapping must exactly cover the consumed row cursor")
    unique_row_splits = tuple((row_sha256, supplied_splits[row_sha256]) for row_sha256 in ordered_consumed_hashes)
    for _, split in unique_row_splits:
        if split not in {"train", "heldout"}:
            raise ValueError("historical row split must be train or heldout")
    computed_mapping_sha256 = _historical_split_mapping_sha256(unique_row_splits) if unique_row_splits else None
    if legacy_mapping_sha256 is not None:
        expected_mapping = _validate_sha256(legacy_mapping_sha256, "legacy_mapping_sha256")
        if expected_mapping != computed_mapping_sha256:
            raise ValueError("legacy row-split mapping hash mismatch")
    state = StreamingTrainingState(
        root_corpus_manifest_sha256=corpus.root_manifest_sha256,
        corpus_lineage_id=corpus.corpus_lineage_id,
        corpus_generation=corpus.generation,
        ordered_shard_identity_sha256=corpus.ordered_shard_identity_sha256,
        compatibility_sha256=corpus.compatibility_sha256,
        row_split_policy_sha256=row_split_policy.sha256,
        ordered_shards=corpus.ordered_shards,
        consumed_shard_sha256s=_consumed_shards(corpus.ordered_shards, cursor_map),
        per_shard_row_cursor=cursor,
        unique_row_splits=unique_row_splits,
        total_unique_rows_consumed=sum(cursor_map.values()),
        unique_train_rows=sum(split == "train" for _, split in unique_row_splits),
        unique_heldout_rows=sum(split == "heldout" for _, split in unique_row_splits),
        legacy_presentations=legacy_presentations,
        unseen_presentations=0,
        replay_presentations=0,
        replay_policy=replay_policy,
        legacy_dataset_sha256=legacy_dataset_sha256,
        legacy_mapping_sha256=computed_mapping_sha256,
        legacy_source_mapping_sha256=corpus.m3_source_mapping_sha256 if unique_row_splits else None,
        legacy_checkpoint_sha256=legacy_checkpoint_sha256,
        legacy_config_sha256=legacy_config_sha256,
        legacy_source_sha256s=tuple(
            sorted(_validate_sha256(value, "legacy_source_sha256") for value in legacy_source_sha256s)
        ),
        dataloader_rng_origin=dataloader_rng_origin,
        dataloader_rng_seed=dataloader_rng_seed,
    )
    state.validate()

    return state


def extend_streaming_training_state(
    state: StreamingTrainingState,
    corpus: ManifestBoundCorpus,
) -> StreamingTrainingState:
    """Rebind only the exact frozen-final root; cross-generation append is invalid."""
    state.validate()
    identities = (
        corpus.root_manifest_sha256,
        corpus.corpus_lineage_id,
        corpus.generation,
        corpus.ordered_shard_identity_sha256,
        corpus.compatibility_sha256,
        corpus.ordered_shards,
    )
    expected = (
        state.root_corpus_manifest_sha256,
        state.corpus_lineage_id,
        state.corpus_generation,
        state.ordered_shard_identity_sha256,
        state.compatibility_sha256,
        state.ordered_shards,
    )
    if identities != expected:
        raise ValueError("streaming state is not bound to the exact frozen-final corpus")

    return state


@dataclass(frozen=True)
class StreamingRowRef:
    shard_sha256: str
    row_offset: int
    row_sha256: str
    split: str

    def validate(self) -> None:
        _validate_sha256(self.shard_sha256, "streaming row shard SHA-256")
        _validate_sha256(self.row_sha256, "streaming row SHA-256")
        if isinstance(self.row_offset, bool) or not isinstance(self.row_offset, int) or self.row_offset < 0:
            raise ValueError("streaming row offset must be a non-negative integer")
        if self.split not in {"train", "heldout"}:
            raise ValueError("streaming row split must be train or heldout")


def validate_streaming_state_against_corpus(
    state: StreamingTrainingState,
    corpus: ManifestBoundCorpus,
    row_split_policy: RowSplitPolicy,
) -> None:
    state.validate()
    extend_streaming_training_state(state, corpus)
    row_split_policy.validate()
    if state.row_split_policy_sha256 != row_split_policy.sha256:
        raise ValueError("streaming state row-split policy mismatch")
    if corpus.trainer_compatibility["split_policy_sha256"] != row_split_policy.sha256:
        raise ValueError("corpus trainer compatibility row-split policy mismatch")
    cursor = state.cursor_by_shard
    expected_hashes = {
        row_sha256
        for shard in corpus.shards
        for row_sha256 in shard.row_sha256s[:cursor[shard.identity.shard_sha256]]
    }
    membership = dict(state.unique_row_splits)
    if set(membership) != expected_hashes:
        raise ValueError("streaming unique-row membership disagrees with the corpus cursor")
    historical_hashes = corpus.historical_row_sha256s
    historical_set = set(historical_hashes)
    if state.legacy_presentations:
        if not historical_set <= expected_hashes:
            raise ValueError("streaming state omitted an audited historical M3 row")
        historical_mapping = tuple((row_sha256, membership[row_sha256]) for row_sha256 in historical_hashes)
        mapping_sha256 = _historical_split_mapping_sha256(historical_mapping)
        if mapping_sha256 != state.legacy_mapping_sha256:
            raise ValueError("streaming historical row-split mapping hash mismatch")
        if state.legacy_dataset_sha256 != corpus.m3_tranche_dataset_sha256:
            raise ValueError("streaming historical dataset hash is not the frozen M3 tranche")
        if state.legacy_source_mapping_sha256 != corpus.m3_source_mapping_sha256:
            raise ValueError("streaming historical source mapping hash mismatch")
    for row_sha256 in expected_hashes - historical_set:
        if membership[row_sha256] != row_split_policy.split_for_row(row_sha256):
            raise ValueError("streaming new-row split is not stable under the bound policy")


def _remaining_streaming_rows(
    state: StreamingTrainingState,
    corpus: ManifestBoundCorpus,
    row_split_policy: RowSplitPolicy,
) -> tuple[StreamingRowRef, ...]:
    validate_streaming_state_against_corpus(state, corpus, row_split_policy)
    cursor = state.cursor_by_shard
    rows = []
    for shard in corpus.shards:
        start = cursor[shard.identity.shard_sha256]
        for offset in range(start, shard.identity.row_count):
            row_sha256 = shard.row_sha256s[offset]
            rows.append(
                StreamingRowRef(
                    shard_sha256=shard.identity.shard_sha256,
                    row_offset=offset,
                    row_sha256=row_sha256,
                    split=row_split_policy.split_for_row(row_sha256),
                )
            )

    return tuple(rows)


def plan_streaming_update(
    state: StreamingTrainingState,
    corpus: ManifestBoundCorpus,
    row_split_policy: RowSplitPolicy,
    *,
    maximum_train_rows: int,
) -> tuple[StreamingRowRef, ...]:
    """Select the deterministic unseen prefix needed for one bounded update."""
    if (
        isinstance(maximum_train_rows, bool)
        or not isinstance(maximum_train_rows, int)
        or maximum_train_rows <= 0
    ):
        raise ValueError("maximum_train_rows must be a positive integer")
    selected = []
    train_rows = 0
    for row in _remaining_streaming_rows(state, corpus, row_split_policy):
        selected.append(row)
        train_rows += row.split == "train"
        if train_rows == maximum_train_rows:
            break

    return tuple(selected)


def commit_streaming_update(
    state: StreamingTrainingState,
    corpus: ManifestBoundCorpus,
    row_split_policy: RowSplitPolicy,
    *,
    unique_rows: Sequence[StreamingRowRef],
    replay_row_sha256s: Sequence[str] = (),
) -> StreamingTrainingState:
    """Advance a contiguous unseen cursor and account for explicit train-only replay."""
    remaining = _remaining_streaming_rows(state, corpus, row_split_policy)
    normalized_unique = tuple(unique_rows)
    if normalized_unique != remaining[:len(normalized_unique)]:
        raise ValueError("streaming update must consume one exact deterministic unseen prefix")
    for row in normalized_unique:
        row.validate()
    replay_hashes = tuple(_validate_sha256(value, "replay row SHA-256") for value in replay_row_sha256s)
    if replay_hashes and not state.replay_policy.allows_replay:
        raise ValueError("replay is disabled by the streaming replay policy")
    if len(replay_hashes) != len(set(replay_hashes)):
        raise ValueError("one streaming update cannot replay a row more than once")
    prior_membership = dict(state.unique_row_splits)
    if any(prior_membership.get(row_sha256) != "train" for row_sha256 in replay_hashes):
        raise ValueError("replay rows must be previously consumed training rows")
    if not normalized_unique and not replay_hashes:
        raise ValueError("streaming update must consume or replay at least one row")
    cursor = state.cursor_by_shard
    appended_splits = []
    for row in normalized_unique:
        cursor[row.shard_sha256] = row.row_offset + 1
        appended_splits.append((row.row_sha256, row.split))
    unique_row_splits = state.unique_row_splits + tuple(appended_splits)
    updated = replace(
        state,
        consumed_shard_sha256s=_consumed_shards(state.ordered_shards, cursor),
        per_shard_row_cursor=tuple(
            (shard.shard_sha256, cursor[shard.shard_sha256]) for shard in state.ordered_shards
        ),
        unique_row_splits=unique_row_splits,
        total_unique_rows_consumed=len(unique_row_splits),
        unique_train_rows=sum(split == "train" for _, split in unique_row_splits),
        unique_heldout_rows=sum(split == "heldout" for _, split in unique_row_splits),
        unseen_presentations=state.unseen_presentations + sum(row.split == "train" for row in normalized_unique),
        replay_presentations=state.replay_presentations + len(replay_hashes),
    )
    validate_streaming_state_against_corpus(updated, corpus, row_split_policy)

    return updated


def _assert_finite_nested_state(value: Any, name: str) -> None:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{name} contains a non-finite tensor")
    elif isinstance(value, Mapping):
        for child in value.values():
            _assert_finite_nested_state(child, name)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_finite_nested_state(child, name)


def _validate_zero_nonfinite_counters(value: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("nonfinite_counters must be a non-empty mapping")
    normalized = {}
    for name, count in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError("nonfinite counter names must be non-empty strings")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("nonfinite counters must be non-negative integers")
        normalized[name] = count
    if any(normalized.values()):
        raise ValueError("nonfinite counters must be zero for a resumable checkpoint")

    return dict(sorted(normalized.items()))


def _streaming_receipt(
    *,
    checkpoint_sha256: str,
    state: StreamingTrainingState,
    progress: TrainingProgress,
    config_sha256: str,
    source_sha256s: tuple[str, ...],
    nonfinite_counters: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "schema_version": STREAMING_CHECKPOINT_RECEIPT_SCHEMA_VERSION,
        "classification": Q_FEATURE_CLASSIFICATION,
        "checkpoint_sha256": checkpoint_sha256,
        "root_corpus_manifest_sha256": state.root_corpus_manifest_sha256,
        "corpus_lineage_id": state.corpus_lineage_id,
        "ordered_shard_identity_sha256": state.ordered_shard_identity_sha256,
        "compatibility_sha256": state.compatibility_sha256,
        "config_sha256": config_sha256,
        "source_sha256s": list(source_sha256s),
        "global_step": progress.global_step,
        "total_unique_rows_consumed": state.total_unique_rows_consumed,
        "unique_train_rows": state.unique_train_rows,
        "unique_heldout_rows": state.unique_heldout_rows,
        "legacy_presentations": state.legacy_presentations,
        "unseen_presentations": state.unseen_presentations,
        "replay_presentations": state.replay_presentations,
        "total_presentations": state.total_presentations,
        "nonfinite_counters": dict(nonfinite_counters),
    }


def atomic_save_streaming_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    amp_scaler: Any,
    sampler_generator: torch.Generator,
    dataloader_generator: torch.Generator,
    progress: TrainingProgress,
    streaming_state: StreamingTrainingState,
    corpus: ManifestBoundCorpus,
    row_split_policy: RowSplitPolicy,
    config_sha256: str,
    source_sha256s: Sequence[str],
    nonfinite_counters: Mapping[str, int],
    preserved_cuda_rng_states: Sequence[torch.Tensor] | None = None,
    cuda_rng_state_origin: str = "captured_current",
) -> str:
    """Atomically persist model, update, cursor, and every RNG in one v2 payload."""
    path = Path(path).resolve()
    progress.validate()
    validate_streaming_state_against_corpus(streaming_state, corpus, row_split_policy)
    if progress.presentations != streaming_state.total_presentations:
        raise ValueError("training progress presentations disagree with streaming accounting")
    _assert_finite_model(model)
    _assert_finite_nested_state(optimizer.state_dict(), "optimizer state")
    normalized_nonfinite = _validate_zero_nonfinite_counters(nonfinite_counters)
    config_sha256 = _validate_sha256(config_sha256, "config_sha256")
    normalized_sources = tuple(sorted(_validate_sha256(value, "source_sha256") for value in source_sha256s))
    if not normalized_sources or len(normalized_sources) != len(set(normalized_sources)):
        raise ValueError("source_sha256s must be non-empty and unique")
    if not isinstance(sampler_generator, torch.Generator) or not isinstance(
        dataloader_generator, torch.Generator
    ):
        raise TypeError("sampler and dataloader generators must be torch.Generator values")
    for component, name in ((scheduler, "scheduler"), (amp_scaler, "amp_scaler")):
        if not hasattr(component, "state_dict") or not hasattr(component, "load_state_dict"):
            raise TypeError(f"{name} must provide state_dict and load_state_dict")
    if cuda_rng_state_origin not in {"captured_current", "preserved_v1_migration"}:
        raise ValueError("CUDA RNG state origin is invalid")
    if preserved_cuda_rng_states is None:
        cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    else:
        cuda_rng_states = [state.detach().cpu().clone() for state in preserved_cuda_rng_states]
    checkpoint = {
        "schema_version": STREAMING_CHECKPOINT_SCHEMA_VERSION,
        "classification": Q_FEATURE_CLASSIFICATION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "amp_scaler_state": amp_scaler.state_dict(),
        "progress": asdict(progress),
        "streaming_state": streaming_state.to_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.random.get_rng_state(),
        "torch_cuda_rng_states": cuda_rng_states,
        "cuda_rng_state_origin": cuda_rng_state_origin,
        "sampler_rng_state": sampler_generator.get_state(),
        "dataloader_rng_state": dataloader_generator.get_state(),
        "config_sha256": config_sha256,
        "source_sha256s": normalized_sources,
        "nonfinite_counters": normalized_nonfinite,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    temporary_path = Path(temporary_name)
    try:
        torch.save(checkpoint, temporary_path)
        with temporary_path.open("r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    checkpoint_sha256 = _sha256_file(path)
    receipt = _streaming_receipt(
        checkpoint_sha256=checkpoint_sha256,
        state=streaming_state,
        progress=progress,
        config_sha256=config_sha256,
        source_sha256s=normalized_sources,
        nonfinite_counters=normalized_nonfinite,
    )
    _atomic_bytes_write(_checkpoint_receipt_path(path), _canonical_json_bytes(receipt) + b"\n")

    return checkpoint_sha256


_STREAMING_CHECKPOINT_FIELDS = frozenset({
    "schema_version",
    "classification",
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "amp_scaler_state",
    "progress",
    "streaming_state",
    "python_rng_state",
    "numpy_rng_state",
    "torch_cpu_rng_state",
    "torch_cuda_rng_states",
    "cuda_rng_state_origin",
    "sampler_rng_state",
    "dataloader_rng_state",
    "config_sha256",
    "source_sha256s",
    "nonfinite_counters",
})


def load_streaming_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    amp_scaler: Any,
    sampler_generator: torch.Generator,
    dataloader_generator: torch.Generator,
    corpus: ManifestBoundCorpus,
    row_split_policy: RowSplitPolicy,
    expected_config_sha256: str,
    expected_source_sha256s: Sequence[str],
    map_location: str | torch.device = "cpu",
    allow_deferred_cuda_rng: bool = False,
) -> tuple[TrainingProgress, StreamingTrainingState, dict[str, int], str]:
    """Verify one exact v2 checkpoint, restore all state, and return its cursor."""
    path = Path(path).resolve()
    receipt = _read_json_object(_checkpoint_receipt_path(path))
    expected_receipt_fields = frozenset(_streaming_receipt(
        checkpoint_sha256="0" * 64,
        state=initialize_streaming_training_state(
            corpus,
            replay_policy=ReplayPolicy(),
            row_split_policy=row_split_policy,
        ),
        progress=TrainingProgress(0, 0, 0, 0, 0.0, {}, {}, {}),
        config_sha256="0" * 64,
        source_sha256s=("1" * 64,),
        nonfinite_counters={"placeholder": 0},
    ))
    _require_exact_fields(receipt, expected_receipt_fields, "streaming checkpoint receipt")
    if (
        receipt["schema_version"] != STREAMING_CHECKPOINT_RECEIPT_SCHEMA_VERSION
        or receipt["classification"] != Q_FEATURE_CLASSIFICATION
    ):
        raise ValueError("streaming checkpoint receipt schema or classification mismatch")
    if _sha256_file(path) != _validate_sha256(receipt["checkpoint_sha256"], "checkpoint_sha256"):
        raise ValueError("streaming checkpoint hash mismatch")
    expected_config = _validate_sha256(expected_config_sha256, "expected_config_sha256")
    expected_sources = tuple(
        sorted(_validate_sha256(value, "expected_source_sha256") for value in expected_source_sha256s)
    )
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise ValueError("streaming checkpoint could not be decoded") from exc
    if not isinstance(checkpoint, dict) or set(checkpoint) != _STREAMING_CHECKPOINT_FIELDS:
        raise ValueError("streaming checkpoint is incomplete or has unexpected fields")
    if (
        checkpoint["schema_version"] != STREAMING_CHECKPOINT_SCHEMA_VERSION
        or checkpoint["classification"] != Q_FEATURE_CLASSIFICATION
    ):
        raise ValueError("streaming checkpoint schema or classification mismatch")
    if checkpoint["cuda_rng_state_origin"] not in {"captured_current", "preserved_v1_migration"}:
        raise ValueError("streaming checkpoint CUDA RNG state origin mismatch")
    if checkpoint["config_sha256"] != expected_config or tuple(checkpoint["source_sha256s"]) != expected_sources:
        raise ValueError("streaming checkpoint config or source identity mismatch")
    state = StreamingTrainingState.from_dict(checkpoint["streaming_state"])
    validate_streaming_state_against_corpus(state, corpus, row_split_policy)
    progress = TrainingProgress(**checkpoint["progress"])
    progress.validate()
    if progress.presentations != state.total_presentations:
        raise ValueError("streaming checkpoint presentation accounting mismatch")
    nonfinite = _validate_zero_nonfinite_counters(checkpoint["nonfinite_counters"])
    expected_receipt = _streaming_receipt(
        checkpoint_sha256=receipt["checkpoint_sha256"],
        state=state,
        progress=progress,
        config_sha256=expected_config,
        source_sha256s=expected_sources,
        nonfinite_counters=nonfinite,
    )
    if receipt != expected_receipt:
        raise ValueError("streaming checkpoint receipt disagrees with checkpoint state")
    _assert_finite_nested_state(checkpoint["model_state"], "model state")
    _assert_finite_nested_state(checkpoint["optimizer_state"], "optimizer state")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    _assert_finite_model(model)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    amp_scaler.load_state_dict(checkpoint["amp_scaler_state"])
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.random.set_rng_state(checkpoint["torch_cpu_rng_state"].cpu())
    cuda_states = checkpoint["torch_cuda_rng_states"]
    if cuda_states:
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])
            cuda_status = "applied"
        elif allow_deferred_cuda_rng and checkpoint["cuda_rng_state_origin"] == "preserved_v1_migration":
            cuda_status = "deferred_unavailable_structurally_preserved"
        else:
            raise ValueError("streaming checkpoint has CUDA RNG state but CUDA is unavailable")
    else:
        cuda_status = "not_present"
    sampler_generator.set_state(checkpoint["sampler_rng_state"].cpu())
    dataloader_generator.set_state(checkpoint["dataloader_rng_state"].cpu())

    return progress, state, nonfinite, cuda_status


_V1_CHECKPOINT_FIELDS = frozenset({
    "schema_version",
    "classification",
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "amp_scaler_state",
    "progress",
    "python_rng_state",
    "numpy_rng_state",
    "torch_cpu_rng_state",
    "torch_cuda_rng_states",
    "sampler_rng_state",
    "dataset_sha256",
    "config_sha256",
    "source_sha256s",
})


@dataclass(frozen=True)
class StreamingMigrationResult:
    checkpoint_sha256: str
    progress: TrainingProgress
    streaming_state: StreamingTrainingState
    cuda_rng_application_status: str
    preserved_cuda_rng_state_count: int


def _read_verified_v1_checkpoint(
    path: Path,
    *,
    expected_checkpoint_sha256: str,
    expected_dataset_sha256: str,
    expected_config_sha256: str,
    expected_source_sha256s: Sequence[str],
    map_location: str | torch.device,
) -> tuple[dict[str, Any], str]:
    path = Path(path).resolve()
    receipt = _read_json_object(_checkpoint_receipt_path(path))
    _require_exact_fields(
        receipt,
        frozenset({"schema_version", "checkpoint_sha256"}),
        "v1 checkpoint receipt",
    )
    if receipt["schema_version"] != CHECKPOINT_RECEIPT_SCHEMA_VERSION:
        raise ValueError("v1 checkpoint receipt schema mismatch")
    checkpoint_sha256 = _validate_sha256(receipt["checkpoint_sha256"], "v1 checkpoint SHA-256")
    if checkpoint_sha256 != _validate_sha256(
        expected_checkpoint_sha256, "expected v1 checkpoint SHA-256"
    ):
        raise ValueError("v1 checkpoint identity mismatch")
    if _sha256_file(path) != checkpoint_sha256:
        raise ValueError("v1 checkpoint hash mismatch")
    expected_dataset = _validate_sha256(expected_dataset_sha256, "expected v1 dataset SHA-256")
    expected_config = _validate_sha256(expected_config_sha256, "expected v1 config SHA-256")
    expected_sources = tuple(
        sorted(_validate_sha256(value, "expected v1 source SHA-256") for value in expected_source_sha256s)
    )
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise ValueError("v1 training checkpoint could not be decoded") from exc
    if not isinstance(checkpoint, dict) or set(checkpoint) != _V1_CHECKPOINT_FIELDS:
        raise ValueError("v1 training checkpoint is incomplete or has unexpected fields")
    if (
        checkpoint["schema_version"] != TRAINING_CHECKPOINT_SCHEMA_VERSION
        or checkpoint["classification"] != Q_FEATURE_CLASSIFICATION
        or checkpoint["dataset_sha256"] != expected_dataset
        or checkpoint["config_sha256"] != expected_config
        or tuple(checkpoint["source_sha256s"]) != expected_sources
    ):
        raise ValueError("v1 checkpoint schema, classification, or bound identity mismatch")
    progress = TrainingProgress(**checkpoint["progress"])
    progress.validate()
    _assert_finite_nested_state(checkpoint["model_state"], "v1 model state")
    _assert_finite_nested_state(checkpoint["optimizer_state"], "v1 optimizer state")
    cuda_states = checkpoint["torch_cuda_rng_states"]
    if not isinstance(cuda_states, list) or any(not isinstance(value, torch.Tensor) for value in cuda_states):
        raise ValueError("v1 CUDA RNG state inventory is invalid")

    return checkpoint, checkpoint_sha256


def migrate_v1_checkpoint_to_streaming(
    legacy_path: Path,
    streaming_path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    amp_scaler: Any,
    sampler_generator: torch.Generator,
    dataloader_generator: torch.Generator,
    corpus: ManifestBoundCorpus,
    row_split_policy: RowSplitPolicy,
    replay_policy: ReplayPolicy,
    historical_row_splits: Mapping[str, str],
    expected_legacy_checkpoint_sha256: str,
    expected_legacy_dataset_sha256: str,
    expected_legacy_config_sha256: str,
    expected_legacy_source_sha256s: Sequence[str],
    expected_legacy_split_mapping_sha256: str,
    expected_legacy_source_mapping_sha256: str,
    streaming_config_sha256: str,
    streaming_source_sha256s: Sequence[str],
    dataloader_seed: int,
    nonfinite_counters: Mapping[str, int],
    map_location: str | torch.device = "cpu",
) -> StreamingMigrationResult:
    """Migrate v1 bytes without weakening its normal CUDA resume boundary."""
    expected_source_mapping = _validate_sha256(
        expected_legacy_source_mapping_sha256, "expected legacy source mapping SHA-256"
    )
    if corpus.m3_source_mapping_sha256 != expected_source_mapping:
        raise ValueError("frozen tranche source mapping identity mismatch")
    if (
        isinstance(dataloader_seed, bool)
        or not isinstance(dataloader_seed, int)
        or not 0 <= dataloader_seed < 2**63
    ):
        raise ValueError("dataloader_seed must be a non-negative signed 64-bit integer")
    checkpoint, _ = _read_verified_v1_checkpoint(
        legacy_path,
        expected_checkpoint_sha256=expected_legacy_checkpoint_sha256,
        expected_dataset_sha256=expected_legacy_dataset_sha256,
        expected_config_sha256=expected_legacy_config_sha256,
        expected_source_sha256s=expected_legacy_source_sha256s,
        map_location=map_location,
    )
    if checkpoint["dataset_sha256"] != corpus.m3_tranche_dataset_sha256:
        raise ValueError("v1 dataset is not the frozen M3 tranche")
    progress = TrainingProgress(**checkpoint["progress"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    _assert_finite_model(model)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    amp_scaler.load_state_dict(checkpoint["amp_scaler_state"])
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.random.set_rng_state(checkpoint["torch_cpu_rng_state"].cpu())
    sampler_generator.set_state(checkpoint["sampler_rng_state"].cpu())
    dataloader_generator.manual_seed(dataloader_seed)
    cuda_states = [value.detach().cpu().clone() for value in checkpoint["torch_cuda_rng_states"]]
    if cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)
        cuda_status = "applied"
    elif cuda_states:
        cuda_status = "deferred_unavailable_structurally_preserved"
    else:
        cuda_status = "not_present"
    streaming_state = initialize_streaming_training_state(
        corpus,
        replay_policy=replay_policy,
        row_split_policy=row_split_policy,
        historical_row_cursors=corpus.historical_row_cursors,
        historical_row_splits=historical_row_splits,
        legacy_presentations=progress.presentations,
        legacy_dataset_sha256=checkpoint["dataset_sha256"],
        legacy_mapping_sha256=expected_legacy_split_mapping_sha256,
        legacy_checkpoint_sha256=expected_legacy_checkpoint_sha256,
        legacy_config_sha256=checkpoint["config_sha256"],
        legacy_source_sha256s=checkpoint["source_sha256s"],
        dataloader_rng_origin="initialized_at_v1_migration",
        dataloader_rng_seed=dataloader_seed,
    )
    checkpoint_sha256 = atomic_save_streaming_checkpoint(
        streaming_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        amp_scaler=amp_scaler,
        sampler_generator=sampler_generator,
        dataloader_generator=dataloader_generator,
        progress=progress,
        streaming_state=streaming_state,
        corpus=corpus,
        row_split_policy=row_split_policy,
        config_sha256=streaming_config_sha256,
        source_sha256s=streaming_source_sha256s,
        nonfinite_counters=nonfinite_counters,
        preserved_cuda_rng_states=cuda_states,
        cuda_rng_state_origin="preserved_v1_migration",
    )

    return StreamingMigrationResult(
        checkpoint_sha256=checkpoint_sha256,
        progress=progress,
        streaming_state=streaming_state,
        cuda_rng_application_status=cuda_status,
        preserved_cuda_rng_state_count=len(cuda_states),
    )
