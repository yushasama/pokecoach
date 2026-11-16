"""Fail-closed adapter from post-M artifacts to the frozen acceptance gate.

The adapter validates local artifact mirrors only.  It never contacts Modal,
starts evaluation, changes a policy, or authorizes runtime/submission use.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from deepmew.q_guided.acceptance import (
    AcceptanceDecision,
    AcceptanceThresholds,
    AuditEvidence,
    CheckStatus,
    EffectEstimate,
    GameplayEvidence,
    OfflineEvidence,
    ResearchVerdict,
    SafetyEvidence,
    decide_research_candidate,
)
from deepmew.q_guided.training import (
    CHECKPOINT_RECEIPT_SCHEMA_VERSION,
    ManifestBoundTeacherDataset,
    Q_FEATURE_CLASSIFICATION,
    TRAINING_CHECKPOINT_SCHEMA_VERSION,
    ShardBinding,
    TrainingProgress,
    load_manifest_bound_shards,
)
from deepmew.value_lab.calibration import apply_calibrator


CONTRACT_SCHEMA_VERSION = "deepmew.q_guided.acceptance_adapter_contract.v1"
RECEIPT_SCHEMA_VERSION = "deepmew.q_guided.acceptance_adapter_receipt.v1"
AUDIT_SCHEMA_VERSION = "deepmew.q_guided.partition_audit.v1"
OFFLINE_SCHEMA_VERSION = "deepmew.q_guided.offline_evidence.v1"
GAMEPLAY_SCHEMA_VERSION = "deepmew.q_guided.gameplay_evidence.v1"
ANCHORED_EVALUATION_SCHEMA_VERSION = "deepmew.q_guided.anchored_evaluation_preparation.v2"
SPLIT_DERIVATION_SCHEMA_VERSION = "deepmew.q_guided.split_derivation.v1"
SPLIT_DERIVATION_SEMANTICS = "immutable_subset_relabel_preserves_exact_row_receipts"
MINER_SCHEMA_VERSION = "deepmew.q_guided.branch_miner.v1"
MINER_PROGRESS_SCHEMA_VERSION = "deepmew.q_guided.branch_miner_progress.v1"
GPU_TRAIN_SCHEMA_VERSION = "deepmew.q_guided.gpu_train.v1"
POLICY_EXPORT_SCHEMA_VERSION = "deepmew.q_guided.r4f_research_export.v1"
SPECIALIST_CALIBRATOR_SCHEMA_VERSION = "deepmew.q_guided.specialist_v_calibrator.v1"
MODAL_COMPLETION_SCHEMA_VERSION = "deepmew.modal_first_wave_job.v1"
FROZEN_EVAL_SEEDS = (40009, 50021, 60013)
SAFETY_FIELDS = ("illegal_actions", "crashes", "timeouts")
REQUIRED_GPU_SECONDS = MappingProxyType({"m3": 28_800.0, "m4": 14_400.0})
FROZEN_ACCEPTANCE_THRESHOLDS = AcceptanceThresholds(
    minimum_heldout_rows=32,
    minimum_heldout_groups=16,
    minimum_games_per_participant=256,
)
REQUIRED_SOURCE_POLICY_ID = "R7"
REQUIRED_SOURCE_V_ID = "V0"

_COMPLETION_FIELDS = frozenset({
    "schema_version",
    "job_id",
    "kind",
    "status",
    "process_return_code",
    "stopped_at_budget_bound",
    "required_outputs_complete",
    "volume_commit_count",
    "cumulative_useful_seconds",
    "eligible_row_count",
    "research_output_exists",
    "metrics",
    "safety_counters",
    "failure_category",
})
_CHECKPOINT_FIELDS = frozenset({
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
_POLICY_METRICS = frozenset({
    "q_regression_mse",
    "sibling_ranking_accuracy",
    "material_advantage_mae",
    "mean_regret",
    "max_regret",
    "action_family_diversity",
    "optimizer_updates",
    "nonfinite_count",
    "mask_failure_count",
    "collapse_count",
})
_SPECIALIST_METRICS = frozenset({
    "heldout_log_loss",
    "heldout_brier",
    "heldout_mae",
    "heldout_target_variance",
    "optimizer_updates",
    "nonfinite_count",
    "collapse_count",
})
_ROW_RECEIPT_FIELDS = frozenset({
    "status",
    "world_sample_count",
    "paired_sample_count",
    "row_sha256",
    "actor_visible_signature_sha256",
    "actor_visible_source_sha256",
})


class ArtifactValidationError(ValueError):
    """Raised when an artifact set cannot safely reach the acceptance gate."""


@dataclass(frozen=True)
class ArtifactRef:
    path: Path
    sha256: str


@dataclass(frozen=True)
class MinerValidation:
    job_id: str
    miner_id: str
    split: str
    config_sha256: str
    config_artifact_sha256: str
    state_turn_interval: tuple[int, int]
    matchup_identity: tuple[str, str]
    eligible_rows: int
    cumulative_useful_seconds: float
    manifest_sha256: str
    progress_sha256: str
    completion_sha256: str
    completion_safety: Mapping[str, int] | None
    bindings: tuple[ShardBinding, ...]
    row_sha256s: frozenset[str]


@dataclass(frozen=True)
class TrainerValidation:
    job_id: str
    mode: str
    dataset_sha256: str
    config_sha256: str
    checkpoint_sha256: str
    artifact: ArtifactRef
    cumulative_gpu_seconds: float
    metrics: Mapping[str, float]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ArtifactValidationError(f"artifact is unreadable: {path}") from exc

    return digest.hexdigest()


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ArtifactValidationError(f"{name} must be a SHA-256 string")
    normalized = value.lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ArtifactValidationError(f"{name} must be a 64-character SHA-256")

    return normalized


def _fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
    *,
    allowed_extra: frozenset[str] = frozenset(),
) -> None:
    if not isinstance(value, Mapping):
        raise ArtifactValidationError(f"{name} must be an object")
    fields = frozenset(value)
    missing = sorted(expected - fields)
    unexpected = sorted(fields - expected - allowed_extra)
    if missing or unexpected:
        raise ArtifactValidationError(
            f"{name} field inventory mismatch: missing={missing}, unexpected={unexpected}"
        )


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"{name} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"{name} must contain one JSON object")

    return value


def _artifact_ref(value: Any, base_dir: Path, name: str) -> ArtifactRef:
    if not isinstance(value, Mapping):
        raise ArtifactValidationError(f"{name} artifact reference must be an object")
    _fields(value, frozenset({"path", "sha256"}), f"{name} artifact reference")
    raw_path = value["path"]
    if not isinstance(raw_path, str) or not raw_path:
        raise ArtifactValidationError(f"{name} artifact path must be a non-empty string")
    path = Path(raw_path)
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    expected = _sha256(value["sha256"], f"{name} artifact sha256")
    actual = _sha256_file(path)
    if actual != expected:
        raise ArtifactValidationError(f"{name} artifact hash mismatch")

    return ArtifactRef(path=path, sha256=actual)


def _json_artifact(value: Any, base_dir: Path, name: str) -> tuple[ArtifactRef, dict[str, Any]]:
    reference = _artifact_ref(value, base_dir, name)

    return reference, _read_json(reference.path, name)


def _finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactValidationError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise ArtifactValidationError(f"{name} is outside its finite bound")

    return number


def _positive(value: Any, name: str) -> float:
    number = _finite(value, name, minimum=0.0)
    if number == 0.0:
        raise ArtifactValidationError(f"{name} must be positive")

    return number


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ArtifactValidationError(f"{name} must be an integer >= {minimum}")

    return value


def _clean_safety(value: Any, name: str, *, allow_uninstrumented: bool = False) -> dict[str, int]:
    if allow_uninstrumented and value == {}:
        return {}
    if not isinstance(value, Mapping) or frozenset(value) != frozenset(SAFETY_FIELDS):
        raise ArtifactValidationError(f"{name} safety counters are not fully instrumented")
    counters = {field: _integer(value[field], f"{name}.{field}") for field in SAFETY_FIELDS}
    failures = {field: count for field, count in counters.items() if count != 0}
    if failures:
        raise ArtifactValidationError(f"{name} safety hard stop: {failures}")

    return counters


def _load_bound_dataset(
    bindings: Sequence[ShardBinding],
    name: str,
) -> ManifestBoundTeacherDataset:
    try:
        return load_manifest_bound_shards(bindings)
    except (OSError, TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"{name} failed manifest-bound validation") from exc


def _row_receipt_identities(
    bindings: Sequence[ShardBinding],
    name: str,
) -> tuple[dict[str, str], frozenset[str]]:
    identities: dict[str, str] = {}
    splits: set[str] = set()
    for binding in bindings:
        manifest = _read_json(Path(binding.manifest_path), f"{name} shard manifest")
        split = manifest.get("split")
        if not isinstance(split, str) or not split:
            raise ArtifactValidationError(f"{name} shard split provenance is malformed")
        splits.add(split)
        entries = manifest.get("row_receipts")
        if not isinstance(entries, list):
            raise ArtifactValidationError(f"{name} shard receipt coverage is malformed")
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise ArtifactValidationError(f"{name} contains a malformed row receipt")
            _fields(entry, _ROW_RECEIPT_FIELDS, f"{name} row receipt {index}")
            normalized = dict(entry)
            for field in (
                "row_sha256",
                "actor_visible_signature_sha256",
                "actor_visible_source_sha256",
            ):
                normalized[field] = _sha256(entry[field], f"{name} row receipt {index} {field}")
            row_sha256 = normalized["row_sha256"]
            if row_sha256 in identities:
                raise ArtifactValidationError(f"{name} binds a row receipt more than once")
            identities[row_sha256] = hashlib.sha256(_canonical_bytes(normalized)).hexdigest()

    return identities, frozenset(splits)


def _validate_completion(
    value: Mapping[str, Any],
    *,
    job_id: str,
    kind: str,
) -> dict[str, Any]:
    _fields(
        value,
        _COMPLETION_FIELDS,
        f"{job_id} completion receipt",
        allowed_extra=frozenset({"call_id", "observed_at"}),
    )
    if value["schema_version"] != MODAL_COMPLETION_SCHEMA_VERSION:
        raise ArtifactValidationError(f"{job_id} completion schema mismatch")
    if value["job_id"] != job_id or value["kind"] != kind:
        raise ArtifactValidationError(f"{job_id} completion identity mismatch")
    if (
        value["status"] != "completed_checkpointed"
        or value["process_return_code"] != 0
        or value["stopped_at_budget_bound"] is not False
        or value["required_outputs_complete"] is not True
        or value["failure_category"] is not None
    ):
        raise ArtifactValidationError(f"{job_id} is not a clean completed artifact set")
    _integer(value["volume_commit_count"], f"{job_id}.volume_commit_count", minimum=1)
    if not isinstance(value["metrics"], Mapping):
        raise ArtifactValidationError(f"{job_id} completion metrics must be an object")

    return dict(value)


def _safe_child(root: Path, relative: Any, name: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ArtifactValidationError(f"{name} must be a non-empty relative path")
    candidate = (root / relative).resolve()
    if candidate == root or not candidate.is_relative_to(root):
        raise ArtifactValidationError(f"{name} escapes its run directory")

    return candidate


def _validate_miner(spec: Mapping[str, Any], base_dir: Path) -> MinerValidation:
    expected_fields = frozenset({
        "job_id",
        "expected_miner_id",
        "completion",
        "config",
        "run_manifest",
        "progress",
        "expected_config_sha256",
        "expected_split",
        "minimum_useful_seconds",
    })
    _fields(spec, expected_fields, "miner specification")
    job_id = spec["job_id"]
    miner_id = spec["expected_miner_id"]
    split = spec["expected_split"]
    if not all(isinstance(value, str) and value for value in (job_id, miner_id, split)):
        raise ArtifactValidationError("miner identities and split must be non-empty strings")
    config_sha256 = _sha256(spec["expected_config_sha256"], f"{job_id} config sha256")
    minimum_seconds = _positive(spec["minimum_useful_seconds"], f"{job_id} minimum useful seconds")
    completion_ref, completion = _json_artifact(spec["completion"], base_dir, f"{job_id} completion")
    completion = _validate_completion(completion, job_id=job_id, kind="miner")
    if completion["research_output_exists"] is not None:
        raise ArtifactValidationError(f"{job_id} miner completion has an invalid research-output flag")
    completion_safety = _clean_safety(
        completion["safety_counters"],
        f"{job_id} completion",
        allow_uninstrumented=True,
    )
    config_ref, config = _json_artifact(spec["config"], base_dir, f"{job_id} config")
    _fields(
        config,
        frozenset({
            "schema_version",
            "miner_id",
            "deck0",
            "deck1",
            "v_checkpoint",
            "v_calibrator_dir",
            "policy0",
            "policy1",
            "minimum_source_turn",
            "maximum_source_turn",
            "source_seed",
            "branch_seed",
            "world_samples",
            "rollouts_per_action",
            "root_action_cap",
            "batch_rows",
            "split",
            "execution_mode",
            "config_sha256",
        }),
        f"{job_id} config",
    )
    if config["schema_version"] != MINER_SCHEMA_VERSION or config["execution_mode"] != "fork":
        raise ArtifactValidationError(f"{job_id} config schema or execution mode mismatch")
    if config["miner_id"] != miner_id or config["split"] != split:
        raise ArtifactValidationError(f"{job_id} config identity mismatch")
    config_payload = {key: value for key, value in config.items() if key != "config_sha256"}
    computed_config_sha256 = hashlib.sha256(_canonical_bytes(config_payload)).hexdigest()
    artifact_config_sha256 = _sha256(config["config_sha256"], f"{job_id} config artifact sha256")
    if artifact_config_sha256 != config_sha256 or computed_config_sha256 != config_sha256:
        raise ArtifactValidationError(f"{job_id} config artifact hash binding mismatch")
    minimum_turn = _integer(config["minimum_source_turn"], f"{job_id} minimum source turn", minimum=1)
    maximum_turn = _integer(config["maximum_source_turn"], f"{job_id} maximum source turn")
    if maximum_turn and maximum_turn < minimum_turn:
        raise ArtifactValidationError(f"{job_id} source-turn interval is invalid")
    for field in ("source_seed", "branch_seed", "root_action_cap", "batch_rows"):
        _integer(config[field], f"{job_id} config {field}", minimum=1)
    for field in ("world_samples", "rollouts_per_action"):
        _integer(config[field], f"{job_id} config {field}", minimum=2)
    matchup_identity = (config["policy0"], config["policy1"])
    if not all(isinstance(value, str) and value for value in matchup_identity):
        raise ArtifactValidationError(f"{job_id} matchup identity is invalid")
    manifest_ref, manifest = _json_artifact(spec["run_manifest"], base_dir, f"{job_id} run manifest")
    progress_ref, progress = _json_artifact(spec["progress"], base_dir, f"{job_id} progress")
    _fields(
        manifest,
        frozenset({
            "schema_version",
            "status",
            "classification",
            "config_sha256",
            "miner_id",
            "split",
            "eligible_rows",
            "cumulative_useful_seconds",
            "safety",
            "shards",
        }),
        f"{job_id} run manifest",
    )
    _fields(
        progress,
        frozenset({
            "schema_version",
            "status",
            "config_sha256",
            "attempts",
            "eligible_rows",
            "duplicate_rows",
            "safe_no_root_failures",
            "cumulative_useful_seconds",
            "v0_validated",
            "next_shard_index",
            "shard_count",
            "safety",
            "split",
        }),
        f"{job_id} progress",
    )
    if (
        manifest["schema_version"] != MINER_SCHEMA_VERSION
        or progress["schema_version"] != MINER_PROGRESS_SCHEMA_VERSION
        or manifest["classification"] != Q_FEATURE_CLASSIFICATION
    ):
        raise ArtifactValidationError(f"{job_id} miner schema or classification mismatch")
    if manifest["status"] != "completed_useful_floor" or progress["status"] != "completed_useful_floor":
        raise ArtifactValidationError(f"{job_id} miner did not complete its useful-runtime floor")
    if manifest["miner_id"] != miner_id or manifest["split"] != split or progress["split"] != split:
        raise ArtifactValidationError(f"{job_id} miner or split identity mismatch")
    manifest_config_sha256 = _sha256(
        manifest["config_sha256"],
        f"{job_id} run manifest config sha256",
    )
    progress_config_sha256 = _sha256(
        progress["config_sha256"],
        f"{job_id} progress config sha256",
    )
    if manifest_config_sha256 != config_sha256 or progress_config_sha256 != config_sha256:
        raise ArtifactValidationError(f"{job_id} config hash mismatch")
    if progress["v0_validated"] is not True:
        raise ArtifactValidationError(f"{job_id} V0 provenance was not validated")
    for field in ("attempts", "eligible_rows", "duplicate_rows", "safe_no_root_failures", "next_shard_index", "shard_count"):
        _integer(progress[field], f"{job_id}.{field}")
    eligible_rows = _integer(manifest["eligible_rows"], f"{job_id}.eligible_rows", minimum=1)
    if eligible_rows != progress["eligible_rows"]:
        raise ArtifactValidationError(f"{job_id} eligible-row counts disagree")
    cumulative = _finite(manifest["cumulative_useful_seconds"], f"{job_id} cumulative useful seconds", minimum=minimum_seconds)
    progress_cumulative = _finite(progress["cumulative_useful_seconds"], f"{job_id} progress cumulative seconds", minimum=minimum_seconds)
    if not math.isclose(cumulative, progress_cumulative, rel_tol=0.0, abs_tol=1e-6):
        raise ArtifactValidationError(f"{job_id} cumulative useful seconds disagree")
    completion_cumulative = _finite(completion["cumulative_useful_seconds"], f"{job_id} completion cumulative seconds", minimum=minimum_seconds)
    if not math.isclose(cumulative, completion_cumulative, rel_tol=0.0, abs_tol=1e-6):
        raise ArtifactValidationError(f"{job_id} completion runtime disagrees with its manifest")
    if completion["eligible_row_count"] != eligible_rows:
        raise ArtifactValidationError(f"{job_id} completion row count disagrees with its manifest")
    _clean_safety(manifest["safety"], f"{job_id} manifest")
    _clean_safety(progress["safety"], f"{job_id} progress")
    shards = manifest["shards"]
    if not isinstance(shards, list) or not shards:
        raise ArtifactValidationError(f"{job_id} has no immutable shards")
    run_dir = manifest_ref.path.parent.resolve()
    bindings: list[ShardBinding] = []
    row_total = 0
    listed_files: set[Path] = set()
    for index, entry in enumerate(shards):
        _fields(
            entry,
            frozenset({"shard", "manifest", "shard_sha256", "manifest_sha256", "row_count"}),
            f"{job_id} shard entry {index}",
        )
        shard_path = _safe_child(run_dir, entry["shard"], f"{job_id} shard path")
        shard_manifest_path = _safe_child(run_dir, entry["manifest"], f"{job_id} shard manifest path")
        if _sha256_file(shard_path) != _sha256(entry["shard_sha256"], f"{job_id} shard sha256"):
            raise ArtifactValidationError(f"{job_id} shard hash mismatch")
        if _sha256_file(shard_manifest_path) != _sha256(entry["manifest_sha256"], f"{job_id} shard manifest sha256"):
            raise ArtifactValidationError(f"{job_id} shard manifest hash mismatch")
        row_total += _integer(entry["row_count"], f"{job_id} shard row count", minimum=1)
        listed_files.update((shard_path, shard_manifest_path))
        bindings.append(ShardBinding(shard_path, shard_manifest_path))
    actual_files = set((run_dir / "shards").glob("*.jsonl")) | set((run_dir / "shards").glob("*.manifest.json"))
    actual_files = {path.resolve() for path in actual_files}
    if actual_files != listed_files:
        raise ArtifactValidationError(f"{job_id} immutable shard inventory is incomplete or contains extras")
    if row_total != eligible_rows or progress["shard_count"] != len(bindings) or progress["next_shard_index"] != len(bindings):
        raise ArtifactValidationError(f"{job_id} shard cursor or row coverage mismatch")
    dataset = _load_bound_dataset(bindings, job_id)
    if len(dataset.rows) != eligible_rows:
        raise ArtifactValidationError(f"{job_id} loaded row count disagrees with its manifest")

    return MinerValidation(
        job_id=job_id,
        miner_id=miner_id,
        split=split,
        config_sha256=config_sha256,
        config_artifact_sha256=config_ref.sha256,
        state_turn_interval=(minimum_turn, maximum_turn),
        matchup_identity=matchup_identity,
        eligible_rows=eligible_rows,
        cumulative_useful_seconds=cumulative,
        manifest_sha256=manifest_ref.sha256,
        progress_sha256=progress_ref.sha256,
        completion_sha256=completion_ref.sha256,
        completion_safety=dict(completion_safety) if completion_safety else None,
        bindings=tuple(bindings),
        row_sha256s=frozenset(row.row_sha256 for row in dataset.rows),
    )


def _load_checkpoint(reference: ArtifactRef, receipt: Mapping[str, Any], name: str) -> dict[str, Any]:
    _fields(receipt, frozenset({"schema_version", "checkpoint_sha256"}), f"{name} checkpoint receipt")
    if receipt["schema_version"] != CHECKPOINT_RECEIPT_SCHEMA_VERSION:
        raise ArtifactValidationError(f"{name} checkpoint receipt schema mismatch")
    if _sha256(receipt["checkpoint_sha256"], f"{name} checkpoint receipt sha256") != reference.sha256:
        raise ArtifactValidationError(f"{name} checkpoint receipt hash mismatch")
    try:
        value = torch.load(reference.path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ArtifactValidationError(f"{name} checkpoint could not be decoded") from exc
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"{name} checkpoint must contain one object")
    _fields(value, _CHECKPOINT_FIELDS, f"{name} checkpoint")

    return value


def _validate_optimizer_state(value: Any, name: str) -> None:
    if not isinstance(value, Mapping) or frozenset(value) != frozenset({"state", "param_groups"}):
        raise ArtifactValidationError(f"{name} optimizer state is incomplete")
    states = value["state"]
    groups = value["param_groups"]
    if not isinstance(states, Mapping) or not isinstance(groups, list) or not groups:
        raise ArtifactValidationError(f"{name} optimizer state is incomplete")
    parameter_ids: list[int] = []
    for group in groups:
        if not isinstance(group, Mapping) or any(not isinstance(key, str) for key in group):
            raise ArtifactValidationError(f"{name} optimizer state is incomplete")
        parameters = group.get("params")
        if not isinstance(parameters, list) or not parameters or any(
            isinstance(parameter, bool) or not isinstance(parameter, int) or parameter < 0
            for parameter in parameters
        ):
            raise ArtifactValidationError(f"{name} optimizer state is incomplete")
        parameter_ids.extend(parameters)
    if len(parameter_ids) != len(set(parameter_ids)):
        raise ArtifactValidationError(f"{name} optimizer state is incomplete")
    parameter_id_set = set(parameter_ids)
    if any(
        isinstance(parameter, bool)
        or not isinstance(parameter, int)
        or parameter not in parameter_id_set
        or not isinstance(state, Mapping)
        for parameter, state in states.items()
    ):
        raise ArtifactValidationError(f"{name} optimizer state is incomplete")


def _validate_scheduler_state(value: Any, name: str) -> None:
    if (
        not isinstance(value, Mapping)
        or not value
        or any(not isinstance(key, str) for key in value)
    ):
        raise ArtifactValidationError(f"{name} scheduler state is incomplete")
    last_epoch = value.get("last_epoch")
    if isinstance(last_epoch, bool) or not isinstance(last_epoch, int) or last_epoch < -1:
        raise ArtifactValidationError(f"{name} scheduler state is incomplete")
    step_count = value.get("_step_count")
    if step_count is not None and (
        isinstance(step_count, bool) or not isinstance(step_count, int) or step_count < 0
    ):
        raise ArtifactValidationError(f"{name} scheduler state is incomplete")


def _validate_amp_scaler_state(value: Any, name: str) -> None:
    if not isinstance(value, Mapping):
        raise ArtifactValidationError(f"{name} AMP scaler state is incomplete")
    if not value:
        return
    required = frozenset({"scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"})
    if frozenset(value) != required:
        raise ArtifactValidationError(f"{name} AMP scaler state is incomplete")
    for field in ("scale", "growth_factor", "backoff_factor"):
        scalar = value[field]
        if (
            isinstance(scalar, bool)
            or not isinstance(scalar, (int, float))
            or not math.isfinite(float(scalar))
            or float(scalar) <= 0.0
        ):
            raise ArtifactValidationError(f"{name} AMP scaler state is incomplete")
    for field, minimum in (("growth_interval", 1), ("_growth_tracker", 0)):
        scalar = value[field]
        if isinstance(scalar, bool) or not isinstance(scalar, int) or scalar < minimum:
            raise ArtifactValidationError(f"{name} AMP scaler state is incomplete")


def _validate_python_rng_state(value: Any, name: str) -> None:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ArtifactValidationError(f"{name} Python RNG state is incomplete")
    version, internal, gaussian = value
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version <= 0
        or not isinstance(internal, tuple)
        or not internal
        or any(isinstance(item, bool) or not isinstance(item, int) for item in internal)
        or (
            gaussian is not None
            and (
                isinstance(gaussian, bool)
                or not isinstance(gaussian, (int, float))
                or not math.isfinite(float(gaussian))
            )
        )
    ):
        raise ArtifactValidationError(f"{name} Python RNG state is incomplete")
    try:
        random.Random().setstate(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ArtifactValidationError(f"{name} Python RNG state is incomplete") from exc


def _validate_numpy_rng_state(value: Any, name: str) -> None:
    if not isinstance(value, tuple) or len(value) != 5:
        raise ArtifactValidationError(f"{name} NumPy RNG state is incomplete")
    generator, keys, position, has_gaussian, cached_gaussian = value
    if (
        not isinstance(generator, str)
        or not generator
        or not isinstance(keys, np.ndarray)
        or keys.dtype != np.uint32
        or keys.ndim != 1
        or keys.size == 0
        or isinstance(position, bool)
        or not isinstance(position, int)
        or not 0 <= position <= keys.size
        or isinstance(has_gaussian, bool)
        or not isinstance(has_gaussian, int)
        or has_gaussian not in (0, 1)
        or isinstance(cached_gaussian, bool)
        or not isinstance(cached_gaussian, (int, float))
        or not math.isfinite(float(cached_gaussian))
    ):
        raise ArtifactValidationError(f"{name} NumPy RNG state is incomplete")
    try:
        np.random.RandomState().set_state(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ArtifactValidationError(f"{name} NumPy RNG state is incomplete") from exc


def _validate_torch_rng_state(value: Any, name: str, *, require_cpu_compatible: bool) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype is not torch.uint8
        or value.ndim != 1
        or value.numel() == 0
    ):
        raise ArtifactValidationError(f"{name} is incomplete")
    if require_cpu_compatible:
        try:
            torch.Generator(device="cpu").set_state(value.detach().cpu())
        except (TypeError, RuntimeError) as exc:
            raise ArtifactValidationError(f"{name} is incomplete") from exc


def _recorded_source_path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactValidationError(f"{name} must be a non-empty path")

    return value


def _finite_state(value: Any, name: str) -> None:
    if not isinstance(value, Mapping) or not value:
        raise ArtifactValidationError(f"{name} must be a non-empty state mapping")
    for key, item in value.items():
        if not isinstance(key, str):
            raise ArtifactValidationError(f"{name} contains a non-string key")
        if not isinstance(item, torch.Tensor):
            raise ArtifactValidationError(f"{name} contains a non-tensor value")
        if not bool(torch.isfinite(item).all().item()):
            raise ArtifactValidationError(f"{name} contains non-finite tensors")


def _state_equal(left: Mapping[str, Any], right: Mapping[str, Any], name: str) -> None:
    if not left or not right:
        raise ArtifactValidationError(f"{name} state must be non-empty")
    if frozenset(left) != frozenset(right):
        raise ArtifactValidationError(f"{name} state keys do not match")
    for key in left:
        left_value = left[key]
        right_value = right[key]
        if not isinstance(left_value, torch.Tensor) or not isinstance(right_value, torch.Tensor):
            raise ArtifactValidationError(f"{name} state contains a non-tensor value")
        if not torch.equal(left_value.cpu(), right_value.cpu()):
            raise ArtifactValidationError(f"{name} state tensor mismatch")


def _metric_map(value: Any, expected: frozenset[str], name: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ArtifactValidationError(f"{name} metrics must be an object")
    _fields(value, expected, f"{name} metrics")
    metrics = {key: _finite(item, f"{name}.{key}") for key, item in value.items()}
    for counter in ("nonfinite_count", "collapse_count", "mask_failure_count"):
        if counter in metrics and metrics[counter] != 0.0:
            raise ArtifactValidationError(f"{name} model-failure counter {counter} is nonzero")

    return metrics


def _validate_policy_export(
    reference: ArtifactRef,
    checkpoint: Mapping[str, Any],
    *,
    dataset_sha256: str,
    config_sha256: str,
) -> None:
    try:
        payload = torch.load(reference.path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ArtifactValidationError("M3 policy export could not be decoded") from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model_state"), Mapping):
        raise ArtifactValidationError("M3 policy export lacks its model state")
    lineage = payload.get("q_guided_research")
    if not isinstance(lineage, Mapping):
        raise ArtifactValidationError("M3 policy export lacks Q-guided lineage")
    _fields(
        lineage,
        frozenset({"schema_version", "classification", "dataset_sha256", "config_sha256", "q_head_state"}),
        "M3 policy export lineage",
    )
    if (
        lineage["schema_version"] != POLICY_EXPORT_SCHEMA_VERSION
        or lineage["classification"] != Q_FEATURE_CLASSIFICATION
        or lineage["dataset_sha256"] != dataset_sha256
        or lineage["config_sha256"] != config_sha256
    ):
        raise ArtifactValidationError("M3 policy export lineage mismatch")
    checkpoint_state = checkpoint["model_state"]
    policy_state = {
        key.removeprefix("policy."): value
        for key, value in checkpoint_state.items()
        if isinstance(key, str) and key.startswith("policy.")
    }
    q_state = {
        key.removeprefix("q_head."): value
        for key, value in checkpoint_state.items()
        if isinstance(key, str) and key.startswith("q_head.")
    }
    _state_equal(policy_state, payload["model_state"], "M3 exported policy")
    if not isinstance(lineage["q_head_state"], Mapping):
        raise ArtifactValidationError("M3 export Q-head state is invalid")
    _state_equal(q_state, lineage["q_head_state"], "M3 exported Q head")


def _validate_calibrator(
    reference: ArtifactRef,
    *,
    dataset_sha256: str,
    config_sha256: str,
    source_policy_sha256: str,
    source_v_sha256: str,
) -> None:
    value = _read_json(reference.path, "M4 calibrator")
    _fields(
        value,
        frozenset({
            "schema_version",
            "classification",
            "dataset_sha256",
            "config_sha256",
            "source_v_sha256",
            "source_policy_sha256",
            "calibrator",
        }),
        "M4 calibrator",
    )
    if (
        value["schema_version"] != SPECIALIST_CALIBRATOR_SCHEMA_VERSION
        or value["classification"] != Q_FEATURE_CLASSIFICATION
        or value["dataset_sha256"] != dataset_sha256
        or value["config_sha256"] != config_sha256
        or value["source_policy_sha256"] != source_policy_sha256
        or value["source_v_sha256"] != source_v_sha256
    ):
        raise ArtifactValidationError("M4 calibrator lineage mismatch")
    try:
        calibrated = apply_calibrator(np.asarray([0.0, 0.5, 1.0]), value["calibrator"])
    except Exception as exc:
        raise ArtifactValidationError("M4 calibrator payload is invalid") from exc
    if not np.isfinite(calibrated).all():
        raise ArtifactValidationError("M4 calibrator emitted non-finite probabilities")


def _validate_trainer(
    spec: Mapping[str, Any],
    base_dir: Path,
    *,
    dataset_sha256: str,
    tranche_manifest_sha256: str,
    expected_source_sha256s: Sequence[str],
    source_policy_sha256: str,
    source_v_sha256: str,
) -> TrainerValidation:
    _fields(
        spec,
        frozenset({
            "job_id",
            "mode",
            "completion",
            "metrics",
            "checkpoint",
            "checkpoint_receipt",
            "artifact",
            "recorded_source_paths",
            "expected_config_sha256",
            "minimum_gpu_seconds",
        }),
        "trainer specification",
    )
    job_id = spec["job_id"]
    mode = spec["mode"]
    if (job_id, mode) not in {("m3", "policy"), ("m4", "specialist-v")}:
        raise ArtifactValidationError("trainer identity and mode must be the M3/M4 pair")
    config_sha256 = _sha256(spec["expected_config_sha256"], f"{job_id} config sha256")
    independent_floor = _positive(spec["minimum_gpu_seconds"], f"{job_id} independent GPU floor")
    if independent_floor != REQUIRED_GPU_SECONDS[job_id]:
        raise ArtifactValidationError(
            f"{job_id} independent GPU floor must be {REQUIRED_GPU_SECONDS[job_id]:.0f} seconds"
        )
    completion_ref, completion = _json_artifact(spec["completion"], base_dir, f"{job_id} completion")
    completion = _validate_completion(completion, job_id=job_id, kind="trainer")
    if completion["research_output_exists"] is not True:
        raise ArtifactValidationError(f"{job_id} completion lacks its research artifact")
    _clean_safety(completion["safety_counters"], f"{job_id} completion")
    metrics_ref, result = _json_artifact(spec["metrics"], base_dir, f"{job_id} metrics")
    _fields(
        result,
        frozenset({
            "schema_version",
            "status",
            "classification",
            "mode",
            "dataset_sha256",
            "config_sha256",
            "manifest_sha256",
            "source_policy_sha256",
            "source_v_sha256",
            "cumulative_gpu_seconds",
            "minimum_cumulative_gpu_seconds",
            "checkpoint",
            "metrics",
            "safety_counters",
            "artifacts",
            "wall_seconds_this_invocation",
        }),
        f"{job_id} metrics receipt",
    )
    if (
        result["schema_version"] != GPU_TRAIN_SCHEMA_VERSION
        or result["status"] != "completed_cumulative_floor"
        or result["classification"] != Q_FEATURE_CLASSIFICATION
        or result["mode"] != mode
    ):
        raise ArtifactValidationError(f"{job_id} result schema, status, classification, or mode mismatch")
    if (
        result["dataset_sha256"] != dataset_sha256
        or result["config_sha256"] != config_sha256
        or result["manifest_sha256"] != tranche_manifest_sha256
        or result["source_policy_sha256"] != source_policy_sha256
        or result["source_v_sha256"] != source_v_sha256
    ):
        raise ArtifactValidationError(f"{job_id} result lineage mismatch")
    declared_floor = _finite(result["minimum_cumulative_gpu_seconds"], f"{job_id} declared GPU floor", minimum=independent_floor)
    cumulative = _finite(result["cumulative_gpu_seconds"], f"{job_id} cumulative GPU seconds", minimum=independent_floor)
    if cumulative < declared_floor:
        raise ArtifactValidationError(f"{job_id} completed result is below its declared floor")
    _finite(result["wall_seconds_this_invocation"], f"{job_id} wall seconds", minimum=0.0)
    _clean_safety(result["safety_counters"], f"{job_id} metrics receipt")
    expected_metrics = _POLICY_METRICS if mode == "policy" else _SPECIALIST_METRICS
    metrics = _metric_map(result["metrics"], expected_metrics, job_id)
    if dict(completion["metrics"]) != dict(result["metrics"]):
        raise ArtifactValidationError(f"{job_id} completion metrics disagree with metrics.json")
    completion_cumulative = _finite(completion["cumulative_useful_seconds"], f"{job_id} completion cumulative seconds", minimum=independent_floor)
    if not math.isclose(cumulative, completion_cumulative, rel_tol=0.0, abs_tol=1e-6):
        raise ArtifactValidationError(f"{job_id} completion GPU seconds disagree with metrics.json")
    checkpoint_ref = _artifact_ref(spec["checkpoint"], base_dir, f"{job_id} checkpoint")
    checkpoint_receipt_ref, checkpoint_receipt = _json_artifact(
        spec["checkpoint_receipt"], base_dir, f"{job_id} checkpoint receipt"
    )
    checkpoint = _load_checkpoint(checkpoint_ref, checkpoint_receipt, job_id)
    if (
        checkpoint["schema_version"] != TRAINING_CHECKPOINT_SCHEMA_VERSION
        or checkpoint["classification"] != Q_FEATURE_CLASSIFICATION
        or checkpoint["dataset_sha256"] != dataset_sha256
        or checkpoint["config_sha256"] != config_sha256
    ):
        raise ArtifactValidationError(f"{job_id} checkpoint lineage mismatch")
    normalized_sources = tuple(sorted(_sha256(value, f"{job_id} source sha256") for value in checkpoint["source_sha256s"]))
    if normalized_sources != tuple(sorted(expected_source_sha256s)):
        raise ArtifactValidationError(f"{job_id} checkpoint source lineage mismatch")
    _validate_optimizer_state(checkpoint["optimizer_state"], job_id)
    _validate_scheduler_state(checkpoint["scheduler_state"], job_id)
    _validate_amp_scaler_state(checkpoint["amp_scaler_state"], job_id)
    _validate_python_rng_state(checkpoint["python_rng_state"], job_id)
    _validate_numpy_rng_state(checkpoint["numpy_rng_state"], job_id)
    _validate_torch_rng_state(
        checkpoint["torch_cpu_rng_state"],
        f"{job_id} torch_cpu_rng_state",
        require_cpu_compatible=True,
    )
    _validate_torch_rng_state(
        checkpoint["sampler_rng_state"],
        f"{job_id} sampler_rng_state",
        require_cpu_compatible=True,
    )
    cuda_states = checkpoint["torch_cuda_rng_states"]
    if not isinstance(cuda_states, list):
        raise ArtifactValidationError(f"{job_id} CUDA RNG state inventory is incomplete")
    for index, state in enumerate(cuda_states):
        _validate_torch_rng_state(
            state,
            f"{job_id} CUDA RNG state {index}",
            require_cpu_compatible=False,
        )
    try:
        progress = TrainingProgress(**checkpoint["progress"])
        progress.validate()
    except Exception as exc:
        raise ArtifactValidationError(f"{job_id} checkpoint progress is invalid") from exc
    _clean_safety(progress.safety_counters, f"{job_id} checkpoint progress")
    if progress.cumulative_gpu_seconds < independent_floor or not math.isclose(
        progress.cumulative_gpu_seconds,
        cumulative,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ArtifactValidationError(f"{job_id} checkpoint progress does not bind the completed floor")
    if dict(progress.last_metrics) != dict(result["metrics"]):
        raise ArtifactValidationError(f"{job_id} checkpoint metrics disagree with metrics.json")
    if progress.global_step != int(metrics["optimizer_updates"]):
        raise ArtifactValidationError(f"{job_id} optimizer step does not match checkpoint progress")
    _finite_state(checkpoint["model_state"], f"{job_id} model state")
    artifacts = result["artifacts"]
    _fields(artifacts, frozenset({"research_r4f_checkpoint", "calibrator"}), f"{job_id} artifact inventory")
    artifact_ref = _artifact_ref(spec["artifact"], base_dir, f"{job_id} research artifact")
    recorded_source_paths = spec["recorded_source_paths"]
    _fields(
        recorded_source_paths,
        frozenset({"checkpoint", "research_artifact"}),
        f"{job_id} recorded source paths",
    )
    recorded_checkpoint_path = _recorded_source_path(
        recorded_source_paths["checkpoint"],
        f"{job_id} recorded checkpoint source path",
    )
    recorded_artifact_path = _recorded_source_path(
        recorded_source_paths["research_artifact"],
        f"{job_id} recorded research artifact source path",
    )
    run_dir = metrics_ref.path.parent
    if (
        checkpoint_ref.path.parent != run_dir
        or checkpoint_receipt_ref.path.parent != run_dir
        or artifact_ref.path.parent != run_dir
    ):
        raise ArtifactValidationError(f"{job_id} trainer artifacts are not co-located in one run directory")
    expected_result_path = artifacts["research_r4f_checkpoint"] if mode == "policy" else artifacts["calibrator"]
    other_result_path = artifacts["calibrator"] if mode == "policy" else artifacts["research_r4f_checkpoint"]
    if other_result_path is not None:
        raise ArtifactValidationError(f"{job_id} research artifact inventory mismatch")
    if expected_result_path != recorded_artifact_path:
        raise ArtifactValidationError(f"{job_id} recorded research artifact source path mismatch")
    if result["checkpoint"] != recorded_checkpoint_path:
        raise ArtifactValidationError(f"{job_id} recorded checkpoint source path mismatch")
    if mode == "policy":
        _validate_policy_export(
            artifact_ref,
            checkpoint,
            dataset_sha256=dataset_sha256,
            config_sha256=config_sha256,
        )
    else:
        _validate_calibrator(
            artifact_ref,
            dataset_sha256=dataset_sha256,
            config_sha256=config_sha256,
            source_policy_sha256=source_policy_sha256,
            source_v_sha256=source_v_sha256,
        )

    return TrainerValidation(
        job_id=job_id,
        mode=mode,
        dataset_sha256=dataset_sha256,
        config_sha256=config_sha256,
        checkpoint_sha256=checkpoint_ref.sha256,
        artifact=artifact_ref,
        cumulative_gpu_seconds=cumulative,
        metrics=metrics,
    )


def _check_status(value: Any, name: str) -> CheckStatus:
    try:
        return CheckStatus(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"{name} must be passed, failed, or not_instrumented") from exc


def _effect(value: Any, name: str) -> EffectEstimate:
    if not isinstance(value, Mapping):
        raise ArtifactValidationError(f"{name} must be an object")
    _fields(value, frozenset({"mean", "lower", "upper"}), name)
    try:
        return EffectEstimate(value["mean"], value["lower"], value["upper"])
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"{name} is invalid") from exc


def _load_audit(value: Any, base_dir: Path, dataset_sha256: str) -> tuple[AuditEvidence, str | None]:
    if value is None:
        missing = CheckStatus.NOT_INSTRUMENTED
        return AuditEvidence(missing, missing, missing, missing, missing, missing), None
    reference, receipt = _json_artifact(value, base_dir, "partition audit")
    _fields(
        receipt,
        frozenset({"schema_version", "classification", "dataset_sha256", "checks"}),
        "partition audit receipt",
    )
    if (
        receipt["schema_version"] != AUDIT_SCHEMA_VERSION
        or receipt["classification"] != Q_FEATURE_CLASSIFICATION
        or receipt["dataset_sha256"] != dataset_sha256
    ):
        raise ArtifactValidationError("partition audit lineage mismatch")
    checks = receipt["checks"]
    expected = frozenset({"leakage", "game_disjoint", "signature_disjoint", "source_disjoint", "identities_bound"})
    _fields(checks, expected, "partition audit checks")
    statuses = {name: _check_status(checks[name], f"partition audit {name}") for name in expected}
    if statuses["identities_bound"] is CheckStatus.FAILED:
        raise ArtifactValidationError("partition audit identity check failed")

    return AuditEvidence(
        statuses["leakage"],
        statuses["game_disjoint"],
        statuses["signature_disjoint"],
        statuses["source_disjoint"],
        CheckStatus.NOT_INSTRUMENTED,
        CheckStatus.NOT_INSTRUMENTED,
    ), reference.sha256


def _load_offline(
    value: Any,
    base_dir: Path,
    *,
    dataset_sha256: str,
    candidate_sha256: str,
    reference_sha256: str,
) -> tuple[OfflineEvidence | None, str | None]:
    if value is None:
        return None, None
    reference, receipt = _json_artifact(value, base_dir, "offline evidence")
    _fields(
        receipt,
        frozenset({
            "schema_version",
            "classification",
            "dataset_sha256",
            "candidate_sha256",
            "reference_sha256",
            "heldout_rows",
            "heldout_groups",
            "mean_teacher_regret_delta",
        }),
        "offline evidence receipt",
    )
    receipt_candidate_sha256 = _sha256(
        receipt["candidate_sha256"],
        "offline evidence candidate sha256",
    )
    receipt_reference_sha256 = _sha256(
        receipt["reference_sha256"],
        "offline evidence reference sha256",
    )
    if (
        receipt["schema_version"] != OFFLINE_SCHEMA_VERSION
        or receipt["classification"] != Q_FEATURE_CLASSIFICATION
        or receipt["dataset_sha256"] != dataset_sha256
        or receipt_candidate_sha256 != candidate_sha256
        or receipt_reference_sha256 != reference_sha256
    ):
        raise ArtifactValidationError("offline evidence lineage mismatch")
    try:
        evidence = OfflineEvidence(
            heldout_rows=receipt["heldout_rows"],
            heldout_groups=receipt["heldout_groups"],
            mean_teacher_regret_delta=_effect(receipt["mean_teacher_regret_delta"], "offline effect"),
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError("offline evidence is invalid") from exc

    return evidence, reference.sha256


def _load_gameplay(
    value: Any,
    base_dir: Path,
    *,
    candidate_sha256: str,
    reference_sha256: str,
) -> tuple[GameplayEvidence | None, CheckStatus, str | None]:
    if value is None:
        return None, CheckStatus.NOT_INSTRUMENTED, None
    reference, receipt = _json_artifact(value, base_dir, "gameplay evidence")
    _fields(
        receipt,
        frozenset({
            "schema_version",
            "classification",
            "candidate_sha256",
            "reference_sha256",
            "equal_budget",
            "safety",
            "games_per_participant",
            "win_rate_delta",
            "severe_regression_cells",
        }),
        "gameplay evidence receipt",
    )
    if (
        receipt["schema_version"] != GAMEPLAY_SCHEMA_VERSION
        or receipt["classification"] != Q_FEATURE_CLASSIFICATION
        or receipt["candidate_sha256"] != candidate_sha256
        or receipt["reference_sha256"] != reference_sha256
    ):
        raise ArtifactValidationError("gameplay evidence lineage mismatch")
    _clean_safety(receipt["safety"], "gameplay evidence")
    equal_budget = _check_status(receipt["equal_budget"], "gameplay equal-budget check")
    try:
        evidence = GameplayEvidence(
            games_per_participant=receipt["games_per_participant"],
            win_rate_delta=_effect(receipt["win_rate_delta"], "gameplay effect"),
            severe_regression_cells=receipt["severe_regression_cells"],
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError("gameplay evidence is invalid") from exc

    return evidence, equal_budget, reference.sha256


def _evaluation_preparation(
    value: Mapping[str, Any],
    base_dir: Path,
    *,
    candidate: ArtifactRef,
    source_policy_sha256: str,
) -> dict[str, Any]:
    _fields(
        value,
        frozenset({
            "runner",
            "pools",
            "reference_checkpoint",
            "deck_binding",
            "reference_label",
            "candidate_label",
            "deck_policy",
            "feature_policy_id",
            "stage",
            "top_n",
            "carry_n",
            "pilots",
            "exploiter_checkpoint",
            "exploiter_top",
            "games",
            "seeds",
            "max_steps",
            "out_dir",
            "run_id",
        }),
        "anchored evaluation specification",
    )
    runner = _artifact_ref(value["runner"], base_dir, "anchored evaluation runner")
    pools = _artifact_ref(value["pools"], base_dir, "anchored evaluation pools")
    reference = _artifact_ref(value["reference_checkpoint"], base_dir, "anchored reference checkpoint")
    deck_binding = _artifact_ref(value["deck_binding"], base_dir, "anchored deck binding")
    if reference.sha256 != source_policy_sha256:
        raise ArtifactValidationError("anchored reference is not the bound source policy")
    reference_label = value["reference_label"]
    candidate_label = value["candidate_label"]
    for label, name in ((reference_label, "reference label"), (candidate_label, "candidate label")):
        if not isinstance(label, str) or not label or "=" in label:
            raise ArtifactValidationError(f"anchored evaluation {name} is invalid")
    if reference_label == candidate_label:
        raise ArtifactValidationError("anchored evaluation labels must be distinct")
    for field in ("deck_policy", "feature_policy_id", "stage", "run_id"):
        if not isinstance(value[field], str) or not value[field]:
            raise ArtifactValidationError(f"anchored evaluation {field} must be non-empty")
    top_n = _integer(value["top_n"], "anchored evaluation top_n", minimum=1)
    carry_n = _integer(value["carry_n"], "anchored evaluation carry_n")
    games = _integer(value["games"], "anchored evaluation games", minimum=1)
    max_steps = _integer(value["max_steps"], "anchored evaluation max_steps", minimum=1)
    pilots = value["pilots"]
    if not isinstance(pilots, list) or not pilots or not all(isinstance(item, str) and item for item in pilots):
        raise ArtifactValidationError("anchored evaluation pilots must be a non-empty string list")
    if len(pilots) != len(set(pilots)):
        raise ArtifactValidationError("anchored evaluation pilots must be unique")
    exploiter_enabled = "exploiter" in pilots
    exploiter_top = _integer(value["exploiter_top"], "anchored evaluation exploiter_top")
    exploiter_checkpoint = None
    if exploiter_enabled:
        if exploiter_top <= 0:
            raise ArtifactValidationError("anchored evaluation exploiter pilot requires exploiter_top > 0")
        exploiter_checkpoint = _artifact_ref(
            value["exploiter_checkpoint"],
            base_dir,
            "anchored exploiter checkpoint",
        )
    elif value["exploiter_checkpoint"] is not None or exploiter_top != 0:
        raise ArtifactValidationError(
            "anchored evaluation exploiter inputs must be null/zero when the pilot is absent"
        )
    seeds = value["seeds"]
    if not isinstance(seeds, list) or tuple(seeds) != FROZEN_EVAL_SEEDS:
        raise ArtifactValidationError("anchored evaluation must use the frozen eval-only seed panel")
    out_dir_value = value["out_dir"]
    if not isinstance(out_dir_value, str) or not out_dir_value:
        raise ArtifactValidationError("anchored evaluation out_dir must be non-empty")
    out_dir = Path(out_dir_value)
    if not out_dir.is_absolute():
        out_dir = base_dir / out_dir
    out_dir = out_dir.resolve()
    command = [
        sys.executable,
        str(runner.path),
        "--pools",
        str(pools.path),
        "--champion",
        f"{reference_label}=checkpoint:{reference.path}",
        "--candidate",
        f"{candidate_label}=checkpoint:{candidate.path}",
        "--deck-binding",
        f"deck:{deck_binding.path}",
        "--deck-policy",
        value["deck_policy"],
        "--feature-policy-id",
        value["feature_policy_id"],
        "--stage",
        value["stage"],
        "--top-n",
        str(top_n),
        "--carry-n",
        str(carry_n),
        "--pilots",
        *pilots,
    ]
    if exploiter_checkpoint is not None:
        command.extend((
            "--exploiter-top",
            str(exploiter_top),
            "--exploiter-checkpoint",
            str(exploiter_checkpoint.path),
        ))
    command.extend((
        "--games",
        str(games),
        "--seeds",
        *(str(seed) for seed in seeds),
        "--max-steps",
        str(max_steps),
        "--out-dir",
        str(out_dir),
        "--run-id",
        value["run_id"],
    ))
    receipt = {
        "schema_version": ANCHORED_EVALUATION_SCHEMA_VERSION,
        "status": "prepared_not_executed",
        "classification": Q_FEATURE_CLASSIFICATION,
        "execution_authorized": False,
        "execution_gate": "offline acceptance and split audits must pass before gameplay execution",
        "candidate_sha256": candidate.sha256,
        "reference_sha256": reference.sha256,
        "runner_sha256": runner.sha256,
        "pools_sha256": pools.sha256,
        "deck_binding_sha256": deck_binding.sha256,
        "exploiter_checkpoint_sha256": (
            exploiter_checkpoint.sha256 if exploiter_checkpoint is not None else None
        ),
        "exploiter_top": exploiter_top,
        "seeds": list(FROZEN_EVAL_SEEDS),
        "command": command,
    }
    receipt["preparation_sha256"] = hashlib.sha256(_canonical_bytes(receipt)).hexdigest()

    return receipt


def _decision_with_missing_evidence(
    *,
    safety: SafetyEvidence,
    audit: AuditEvidence,
    offline: OfflineEvidence | None,
    gameplay: GameplayEvidence | None,
    thresholds: AcceptanceThresholds,
) -> AcceptanceDecision:
    missing = []
    if offline is None:
        missing.append("offline evidence is not_instrumented")
        offline = OfflineEvidence(0, 0, EffectEstimate(0.0, 0.0, 0.0))
    if gameplay is None:
        missing.append("equal-budget gameplay evidence is not_instrumented")
        gameplay = GameplayEvidence(0, EffectEstimate(0.0, 0.0, 0.0), 0)
    decision = decide_research_candidate(
        safety=safety,
        audit=audit,
        offline=offline,
        gameplay=gameplay,
        thresholds=thresholds,
    )
    if missing and decision.verdict is not ResearchVerdict.REJECTED:
        return AcceptanceDecision(ResearchVerdict.AMBIGUOUS, tuple(missing) + decision.reasons)

    return decision


def validate_and_prepare(contract_path: Path) -> dict[str, Any]:
    """Validate a synthetic/local post-M artifact contract and prepare evaluation."""

    contract_path = Path(contract_path).resolve()
    contract = _read_json(contract_path, "acceptance adapter contract")
    _fields(
        contract,
        frozenset({
            "schema_version",
            "classification",
            "expected_lineage",
            "miners",
            "tranche",
            "trainers",
            "audit_evidence",
            "offline_evidence",
            "gameplay_evidence",
            "thresholds",
            "anchored_evaluation",
        }),
        "acceptance adapter contract",
    )
    if contract["schema_version"] != CONTRACT_SCHEMA_VERSION or contract["classification"] != Q_FEATURE_CLASSIFICATION:
        raise ArtifactValidationError("acceptance adapter contract schema or classification mismatch")
    base_dir = contract_path.parent
    lineage = contract["expected_lineage"]
    _fields(
        lineage,
        frozenset({
            "source_policy_id",
            "source_policy_sha256",
            "source_v_id",
            "source_v_sha256",
        }),
        "expected lineage",
    )
    source_policy_id = lineage["source_policy_id"]
    source_v_id = lineage["source_v_id"]
    if not all(isinstance(value, str) and value for value in (source_policy_id, source_v_id)):
        raise ArtifactValidationError("source policy and V lineage identities must be non-empty strings")
    if source_policy_id != REQUIRED_SOURCE_POLICY_ID or source_v_id != REQUIRED_SOURCE_V_ID:
        raise ArtifactValidationError("source lineage identities must be exactly R7 and V0")
    source_policy_sha256 = _sha256(lineage["source_policy_sha256"], "source policy sha256")
    source_v_sha256 = _sha256(lineage["source_v_sha256"], "source V sha256")
    miners = contract["miners"]
    if not isinstance(miners, list) or len(miners) != 2:
        raise ArtifactValidationError("contract must contain exactly M1 and M2")
    miner_results = tuple(_validate_miner(spec, base_dir) for spec in miners)
    if {result.job_id for result in miner_results} != {"m1", "m2"}:
        raise ArtifactValidationError("miner contract identities must be exactly m1 and m2")
    if len({result.miner_id for result in miner_results}) != len(miner_results):
        raise ArtifactValidationError("M1 and M2 miner identities must be distinct")
    if len({result.config_sha256 for result in miner_results}) != len(miner_results):
        raise ArtifactValidationError("M1 and M2 stratum config hashes must be distinct")
    if len({result.matchup_identity for result in miner_results}) != len(miner_results):
        raise ArtifactValidationError("M1 and M2 matchup strata must be distinct")
    first_interval, second_interval = (result.state_turn_interval for result in miner_results)
    first_upper = math.inf if first_interval[1] == 0 else first_interval[1]
    second_upper = math.inf if second_interval[1] == 0 else second_interval[1]
    if max(first_interval[0], second_interval[0]) <= min(first_upper, second_upper):
        raise ArtifactValidationError("M1 and M2 state-turn strata overlap")
    combined_bindings = tuple(binding for result in miner_results for binding in result.bindings)
    combined_dataset = _load_bound_dataset(combined_bindings, "combined M1/M2 dataset")
    tranche_spec = contract["tranche"]
    _fields(
        tranche_spec,
        frozenset({"manifest", "shard", "expected_dataset_sha256", "split_derivation"}),
        "tranche specification",
    )
    split_derivation = tranche_spec["split_derivation"]
    _fields(
        split_derivation,
        frozenset({"schema_version", "source_splits", "target_split", "semantics"}),
        "tranche split derivation",
    )
    if (
        split_derivation["schema_version"] != SPLIT_DERIVATION_SCHEMA_VERSION
        or split_derivation["semantics"] != SPLIT_DERIVATION_SEMANTICS
    ):
        raise ArtifactValidationError("tranche split-derivation schema or semantics mismatch")
    declared_source_splits = split_derivation["source_splits"]
    if (
        not isinstance(declared_source_splits, list)
        or not declared_source_splits
        or not all(isinstance(value, str) and value for value in declared_source_splits)
        or declared_source_splits != sorted(set(declared_source_splits))
    ):
        raise ArtifactValidationError("tranche source split set must be a sorted unique non-empty list")
    target_split = split_derivation["target_split"]
    if not isinstance(target_split, str) or not target_split:
        raise ArtifactValidationError("tranche target split must be a non-empty string")
    tranche_manifest_ref = _artifact_ref(tranche_spec["manifest"], base_dir, "tranche manifest")
    tranche_shard_ref = _artifact_ref(tranche_spec["shard"], base_dir, "tranche shard")
    tranche_dataset = _load_bound_dataset(
        [ShardBinding(tranche_shard_ref.path, tranche_manifest_ref.path)],
        "frozen tranche",
    )
    expected_dataset_sha256 = _sha256(tranche_spec["expected_dataset_sha256"], "tranche dataset sha256")
    if tranche_dataset.dataset_sha256 != expected_dataset_sha256:
        raise ArtifactValidationError("tranche dataset hash mismatch")
    if tuple(tranche_dataset.shard_sha256s) != (tranche_shard_ref.sha256,):
        raise ArtifactValidationError("tranche shard reference is not bound by its manifest")
    if tuple(tranche_dataset.manifest_sha256s) != (tranche_manifest_ref.sha256,):
        raise ArtifactValidationError("tranche manifest hash mismatch")
    mined_rows = {row.row_sha256 for row in combined_dataset.rows}
    tranche_rows = {row.row_sha256 for row in tranche_dataset.rows}
    if not tranche_rows or not tranche_rows.issubset(mined_rows):
        raise ArtifactValidationError("tranche contains rows not bound to the validated miners")
    mined_receipts, observed_source_splits = _row_receipt_identities(
        combined_bindings,
        "combined M1/M2 dataset",
    )
    tranche_receipts, observed_target_splits = _row_receipt_identities(
        [ShardBinding(tranche_shard_ref.path, tranche_manifest_ref.path)],
        "frozen tranche",
    )
    if observed_source_splits != frozenset(declared_source_splits):
        raise ArtifactValidationError("tranche source split set differs from the declared derivation")
    if observed_target_splits != frozenset({target_split}):
        raise ArtifactValidationError("tranche target split differs from the declared derivation")
    if set(mined_receipts) != mined_rows or set(tranche_receipts) != tranche_rows:
        raise ArtifactValidationError("row receipt provenance coverage disagrees with validated rows")
    if any(tranche_receipts[row_sha256] != mined_receipts[row_sha256] for row_sha256 in tranche_rows):
        raise ArtifactValidationError("tranche rewrites a miner row receipt or provenance identity")
    expected_sources = tuple(sorted({
        source_policy_sha256,
        source_v_sha256,
        *tranche_dataset.shard_sha256s,
        *tranche_dataset.manifest_sha256s,
    }))
    trainers = contract["trainers"]
    if not isinstance(trainers, list) or len(trainers) != 2:
        raise ArtifactValidationError("contract must contain exactly M3 and M4")
    trainer_results = tuple(
        _validate_trainer(
            spec,
            base_dir,
            dataset_sha256=tranche_dataset.dataset_sha256,
            tranche_manifest_sha256=tranche_manifest_ref.sha256,
            expected_source_sha256s=expected_sources,
            source_policy_sha256=source_policy_sha256,
            source_v_sha256=source_v_sha256,
        )
        for spec in trainers
    )
    by_job = {result.job_id: result for result in trainer_results}
    if frozenset(by_job) != frozenset({"m3", "m4"}):
        raise ArtifactValidationError("trainer contract identities must be exactly m3 and m4")
    thresholds_value = contract["thresholds"]
    _fields(
        thresholds_value,
        frozenset({"minimum_heldout_rows", "minimum_heldout_groups", "minimum_games_per_participant"}),
        "acceptance thresholds",
    )
    try:
        thresholds = AcceptanceThresholds(**thresholds_value)
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError("acceptance thresholds are invalid") from exc
    if thresholds != FROZEN_ACCEPTANCE_THRESHOLDS:
        raise ArtifactValidationError("acceptance thresholds differ from the frozen gate")
    audit, audit_sha256 = _load_audit(contract["audit_evidence"], base_dir, tranche_dataset.dataset_sha256)
    offline, offline_sha256 = _load_offline(
        contract["offline_evidence"],
        base_dir,
        dataset_sha256=tranche_dataset.dataset_sha256,
        candidate_sha256=by_job["m3"].artifact.sha256,
        reference_sha256=source_policy_sha256,
    )
    anchored = _evaluation_preparation(
        contract["anchored_evaluation"],
        base_dir,
        candidate=by_job["m3"].artifact,
        source_policy_sha256=source_policy_sha256,
    )
    gameplay, equal_budget, gameplay_sha256 = _load_gameplay(
        contract["gameplay_evidence"],
        base_dir,
        candidate_sha256=by_job["m3"].artifact.sha256,
        reference_sha256=source_policy_sha256,
    )
    audit = AuditEvidence(
        audit.leakage,
        audit.game_disjoint,
        audit.signature_disjoint,
        audit.source_disjoint,
        audit.identities_bound,
        equal_budget,
    )
    gameplay_safety = SafetyEvidence(0, 0, 0) if gameplay is not None else SafetyEvidence(None, None, None)
    decision = _decision_with_missing_evidence(
        safety=gameplay_safety,
        audit=audit,
        offline=offline,
        gameplay=gameplay,
        thresholds=thresholds,
    )
    verified_hashes = {
        "miners": {
            result.job_id: {
                "completion": result.completion_sha256,
                "config_artifact": result.config_artifact_sha256,
                "progress": result.progress_sha256,
                "run_manifest": result.manifest_sha256,
            }
            for result in miner_results
        },
        "tranche": {
            "manifest": tranche_manifest_ref.sha256,
            "shard": tranche_shard_ref.sha256,
        },
        "trainers": {
            result.job_id: {
                "checkpoint": result.checkpoint_sha256,
                "artifact": result.artifact.sha256,
            }
            for result in trainer_results
        },
        "evidence": {
            "audit": audit_sha256,
            "offline": offline_sha256,
            "gameplay": gameplay_sha256,
        },
    }
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "validated",
        "classification": Q_FEATURE_CLASSIFICATION,
        "adapter_classification": "diagnostic only",
        "verified_artifact_hashes": verified_hashes,
        "verified_lineage": {
            "source_policy_id": source_policy_id,
            "source_policy_sha256": source_policy_sha256,
            "source_v_id": source_v_id,
            "source_v_sha256": source_v_sha256,
            "dataset_sha256": tranche_dataset.dataset_sha256,
            "trainer_source_sha256": "not_instrumented",
            "split_derivation": {
                "schema_version": SPLIT_DERIVATION_SCHEMA_VERSION,
                "source_splits": declared_source_splits,
                "target_split": target_split,
                "semantics": SPLIT_DERIVATION_SEMANTICS,
            },
        },
        "floor_receipts": {
            result.job_id: {
                "minimum_seconds": next(
                    float(spec["minimum_gpu_seconds"])
                    for spec in trainers
                    if spec["job_id"] == result.job_id
                ),
                "cumulative_seconds": result.cumulative_gpu_seconds,
            }
            for result in trainer_results
        },
        "normalized_safety": {
            "artifact_generation": {
                "miners": {
                    result.job_id: {
                        "completion_receipt": (
                            dict(result.completion_safety)
                            if result.completion_safety is not None
                            else "not_instrumented"
                        ),
                        "authoritative_run_manifest": {field: 0 for field in SAFETY_FIELDS},
                        "authoritative_progress": {field: 0 for field in SAFETY_FIELDS},
                    }
                    for result in miner_results
                },
                "trainers": {
                    result.job_id: {field: 0 for field in SAFETY_FIELDS}
                    for result in trainer_results
                },
            },
            "evaluation": asdict(gameplay_safety),
        },
        "normalized_audit": asdict(audit),
        "normalized_offline": asdict(offline) if offline is not None else None,
        "normalized_gameplay": asdict(gameplay) if gameplay is not None else None,
        "acceptance_decision": {
            "verdict": decision.verdict.value,
            "reasons": list(decision.reasons),
        },
        "anchored_evaluation": anchored,
    }
