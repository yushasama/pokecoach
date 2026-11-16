"""Research-only streaming-v2 continuation for the quarantined M3 policy-Q model.

This is a parallel path.  It deliberately does not share control flow with the
exact-dataset v1 runner, so the v1 resume contract remains unchanged.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for _path in (ROOT, SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_q_guided_gpu_train as v1  # noqa: E402
import deepmew.q_guided.training as q_training  # noqa: E402
import deepmew.rl1_teacher_bc.semantics as semantic_features  # noqa: E402
from deepmew.replay.contract.preflight import (  # noqa: E402
    require_accepted_canonical_decision_receipt,
)
from deepmew.q_guided.training import (  # noqa: E402
    ACTION_HASH_BINS,
    Q_FEATURE_CLASSIFICATION,
    ReplayPolicy,
    RowSplitPolicy,
    ShardBinding,
    STREAMING_TRAINER_SCHEMA_VERSION,
    TrainingProgress,
    StreamingTrainingState,
    atomic_save_streaming_checkpoint,
    commit_streaming_update,
    load_manifest_bound_corpus,
    load_streaming_checkpoint,
    migrate_v1_checkpoint_to_streaming,
    plan_streaming_update,
    validate_streaming_state_against_corpus,
)


CONFIG_SCHEMA_VERSION = "deepmew.q_guided.streaming_train_config.v1"
RESULT_SCHEMA_VERSION = "deepmew.q_guided.streaming_train_result.v1"
HISTORICAL_SPLIT_SCHEMA_VERSION = "deepmew.q_guided.m3_historical_split.v1"
SOURCE_MAPPING_RECEIPT_SCHEMA_VERSION = "deepmew.q_guided.branch_corpus_audit.v1"
M3_HISTORICAL_ROW_COUNT = 64
M3_HISTORICAL_TRAIN_ROW_COUNT = 32
M3_HISTORICAL_HELDOUT_ROW_COUNT = 32
ZERO_SAFETY_COUNTERS = {"illegal_actions": 0, "crashes": 0, "timeouts": 0}
ZERO_NONFINITE_COUNTERS = {"loss": 0, "gradient": 0, "model": 0, "optimizer": 0}

_CONFIG_FIELDS = frozenset({
    "schema_version",
    "root_manifest",
    "root_manifest_sha256",
    "corpus_root",
    "m3_tranche",
    "trainer_compatibility",
    "source_mapping_receipt",
    "legacy_v1",
    "source_policy",
    "historical_split_mapping_sha256",
    "row_split_policy",
    "replay_policy",
    "optimizer",
    "deck",
    "trainer_bytes",
    "batch_size",
    "seed",
})
_TRANCHE_FIELDS = frozenset({"shard", "manifest"})
_SOURCE_MAPPING_FIELDS = frozenset({"path", "sha256", "ordered_source_mapping_sha256"})
_LEGACY_FIELDS = frozenset({
    "checkpoint",
    "checkpoint_sha256",
    "dataset_sha256",
    "config_sha256",
    "source_sha256s",
    "presentations",
})
_SOURCE_POLICY_FIELDS = frozenset({"checkpoint", "sha256"})
_DECK_FIELDS = frozenset({"path", "sha256"})
_OPTIMIZER_FIELDS = frozenset({
    "learning_rate",
    "weight_decay",
    "gradient_clip_norm",
    "scheduler",
    "amp_enabled",
})
_SCHEDULER_FIELDS = frozenset({"kind", "t_0"})
_TRAINER_BYTES_FIELDS = frozenset({"streaming_runner_sha256", "policy_training_module_sha256"})


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


def _validate_sha256(value: Any, name: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{name} must be a 64-character SHA-256")

    return normalized


def _require_fields(value: Mapping[str, Any], fields: frozenset[str], name: str) -> None:
    if set(value) != fields:
        raise ValueError(f"{name} has missing or unexpected fields")


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} could not be decoded") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")

    return value


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


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    config = _read_json(path, "streaming train config")
    _require_fields(config, _CONFIG_FIELDS, "streaming train config")
    if config["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ValueError("streaming train config schema mismatch")
    for key, fields in (
        ("m3_tranche", _TRANCHE_FIELDS),
        ("source_mapping_receipt", _SOURCE_MAPPING_FIELDS),
        ("legacy_v1", _LEGACY_FIELDS),
        ("source_policy", _SOURCE_POLICY_FIELDS),
        ("optimizer", _OPTIMIZER_FIELDS),
        ("deck", _DECK_FIELDS),
        ("trainer_bytes", _TRAINER_BYTES_FIELDS),
    ):
        if not isinstance(config[key], dict):
            raise TypeError(f"streaming train config {key} must be an object")
        _require_fields(config[key], fields, f"streaming train config {key}")
    legacy = config["legacy_v1"]
    sources = legacy["source_sha256s"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("legacy source_sha256s must be a non-empty list")
    normalized_sources = [_validate_sha256(value, "legacy source SHA-256") for value in sources]
    if normalized_sources != sorted(set(normalized_sources)):
        raise ValueError("legacy source_sha256s must be unique and sorted")
    legacy["source_sha256s"] = normalized_sources
    if (
        isinstance(legacy["presentations"], bool)
        or not isinstance(legacy["presentations"], int)
        or legacy["presentations"] <= 0
    ):
        raise ValueError("legacy presentations must be a positive integer")
    path_values = (
        config["root_manifest"],
        config["corpus_root"],
        config["m3_tranche"]["shard"],
        config["m3_tranche"]["manifest"],
        config["source_mapping_receipt"]["path"],
        legacy["checkpoint"],
        config["source_policy"]["checkpoint"],
        config["deck"]["path"],
    )
    if any(not isinstance(value, str) or not value for value in path_values):
        raise ValueError("streaming config paths must be non-empty strings")
    for name in ("trainer_compatibility", "row_split_policy", "replay_policy"):
        if not isinstance(config[name], dict):
            raise TypeError(f"streaming config {name} must be an object")
    for field in ("checkpoint_sha256", "dataset_sha256", "config_sha256"):
        legacy[field] = _validate_sha256(legacy[field], f"legacy {field}")
    config["source_policy"]["sha256"] = _validate_sha256(
        config["source_policy"]["sha256"], "source policy SHA-256"
    )
    config["root_manifest_sha256"] = _validate_sha256(
        config["root_manifest_sha256"], "root manifest SHA-256"
    )
    config["deck"]["sha256"] = _validate_sha256(config["deck"]["sha256"], "deck SHA-256")
    config["historical_split_mapping_sha256"] = _validate_sha256(
        config["historical_split_mapping_sha256"], "historical split mapping SHA-256"
    )
    optimizer = config["optimizer"]
    scheduler = optimizer["scheduler"]
    if not isinstance(scheduler, dict):
        raise TypeError("optimizer scheduler must be an object")
    _require_fields(scheduler, _SCHEDULER_FIELDS, "optimizer scheduler")
    if scheduler["kind"] != "cosine_annealing_warm_restarts":
        raise ValueError("streaming scheduler kind mismatch")
    if isinstance(scheduler["t_0"], bool) or not isinstance(scheduler["t_0"], int) or scheduler["t_0"] <= 0:
        raise ValueError("streaming scheduler t_0 must be a positive integer")
    if not isinstance(optimizer["amp_enabled"], bool):
        raise TypeError("optimizer amp_enabled must be boolean")
    for field in ("learning_rate", "weight_decay", "gradient_clip_norm"):
        value = optimizer[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"optimizer {field} must be finite")
    if optimizer["learning_rate"] <= 0.0 or optimizer["weight_decay"] < 0.0 or optimizer["gradient_clip_norm"] <= 0.0:
        raise ValueError("optimizer hyperparameters are outside their valid range")
    if isinstance(config["seed"], bool) or not isinstance(config["seed"], int) or config["seed"] < 0:
        raise ValueError("streaming seed must be a non-negative integer")
    if isinstance(config["batch_size"], bool) or not isinstance(config["batch_size"], int) or config["batch_size"] <= 0:
        raise ValueError("streaming batch_size must be a positive integer")
    trainer_bytes = config["trainer_bytes"]
    expected_trainer_bytes = {
        "streaming_runner_sha256": _sha256_file(Path(__file__).resolve()),
        "policy_training_module_sha256": _sha256_file(Path(v1.__file__).resolve()),
    }
    if trainer_bytes != expected_trainer_bytes:
        raise ValueError("streaming trainer bytes do not match the bound config")

    return config, _sha256_file(path)


def _verify_optimizer_contract(
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    amp_scaler: Any,
    config: Mapping[str, Any],
) -> None:
    expected_learning_rate = float(config["learning_rate"])
    expected_weight_decay = float(config["weight_decay"])
    if not optimizer.param_groups:
        raise ValueError("streaming optimizer has no parameter groups")
    scheduler_state = scheduler.state_dict()
    base_lrs = scheduler_state.get("base_lrs")
    if not isinstance(base_lrs, list) or len(base_lrs) != len(optimizer.param_groups):
        raise ValueError("restored scheduler lacks an exact base learning-rate inventory")
    if any(not math.isclose(float(value), expected_learning_rate, rel_tol=0.0, abs_tol=0.0) for value in base_lrs):
        raise ValueError("restored scheduler base learning rate disagrees with the streaming config")
    if scheduler_state.get("T_0") != config["scheduler"]["t_0"]:
        raise ValueError("restored scheduler period disagrees with the streaming config")
    last_lrs = scheduler_state.get("_last_lr")
    if not isinstance(last_lrs, list) or len(last_lrs) != len(optimizer.param_groups):
        raise ValueError("restored scheduler lacks an exact current learning-rate inventory")
    for group, last_learning_rate in zip(optimizer.param_groups, last_lrs, strict=True):
        current_learning_rate = float(group.get("lr", math.nan))
        if not math.isclose(current_learning_rate, float(last_learning_rate), rel_tol=0.0, abs_tol=0.0):
            raise ValueError("restored optimizer and scheduler current learning rates disagree")
        if not math.isclose(
            float(group.get("weight_decay", math.nan)),
            expected_weight_decay,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("restored optimizer weight decay disagrees with the streaming config")
    if bool(amp_scaler.is_enabled()) != bool(config["amp_enabled"]):
        raise ValueError("restored AMP scaler mode disagrees with the streaming config")


def _derived_trainer_compatibility(
    deck_identity: Mapping[str, Any],
    row_split_policy: RowSplitPolicy,
) -> dict[str, str]:
    deck_path = Path(deck_identity["path"]).resolve()
    deck_sha256 = _validate_sha256(deck_identity["sha256"], "deck SHA-256")
    if _sha256_file(deck_path) != deck_sha256:
        raise ValueError("deck bytes do not match the bound deck identity")
    state_descriptor = {
        "schema_version": "deepmew.q_guided.state_feature_schema.v1",
        "encoder": "scripts.run_unified_r3_value._features",
        "encoder_module_sha256": _sha256_file(Path(v1.value_source.__file__).resolve()),
        "feature_config": v1.value_source.FEATURE_CONFIG_V2,
        "feature_dimension": v1.value_source.FEATURE_DIM_V2,
        "second_order_state_schema": list(v1.value_source.SECOND_ORDER_STATE_SCHEMA),
    }
    semantic_dimension = len(semantic_features.build_option_features({}))
    action_descriptor = {
        "schema_version": "deepmew.q_guided.action_feature_schema.v1",
        "encoder": "deepmew.q_guided.training.encode_actor_visible_actions",
        "encoder_module_sha256": _sha256_file(Path(q_training.__file__).resolve()),
        "semantic_encoder": "deepmew.rl1_teacher_bc.semantics.build_option_features",
        "semantic_encoder_module_sha256": _sha256_file(Path(semantic_features.__file__).resolve()),
        "semantic_feature_dimension": semantic_dimension,
        "action_hash_bins": ACTION_HASH_BINS,
        "action_feature_dimension": semantic_dimension * 2 + 2 + ACTION_HASH_BINS,
    }

    return {
        "trainer_schema_version": STREAMING_TRAINER_SCHEMA_VERSION,
        "state_feature_schema_sha256": _sha256_bytes(_canonical_bytes(state_descriptor)),
        "action_feature_schema_sha256": _sha256_bytes(_canonical_bytes(action_descriptor)),
        "deck_manifest_sha256": deck_sha256,
        "split_policy_sha256": row_split_policy.sha256,
    }


def _verify_source_mapping_receipt(config: Mapping[str, Any], tranche: ShardBinding) -> str:
    identity = config["source_mapping_receipt"]
    path = Path(identity["path"]).resolve()
    expected_file_sha256 = _validate_sha256(identity["sha256"], "source mapping receipt SHA-256")
    if _sha256_file(path) != expected_file_sha256:
        raise ValueError("source mapping receipt hash mismatch")
    receipt = _read_json(path, "source mapping receipt")
    if receipt.get("schema_version") != SOURCE_MAPPING_RECEIPT_SCHEMA_VERSION:
        raise ValueError("source mapping receipt schema mismatch")
    frozen = receipt.get("frozen_m3_tranche")
    if not isinstance(frozen, dict) or frozen.get("status") != "frozen_immutable":
        raise ValueError("source mapping receipt does not bind a frozen M3 tranche")
    expected_mapping = _validate_sha256(
        identity["ordered_source_mapping_sha256"], "ordered source mapping SHA-256"
    )
    if frozen.get("ordered_source_mapping_sha256") != expected_mapping:
        raise ValueError("source mapping digest mismatch")
    if frozen.get("row_count") != M3_HISTORICAL_ROW_COUNT:
        raise ValueError("source mapping receipt does not bind exactly 64 historical rows")
    if frozen.get("tranche_shard_sha256") != _sha256_file(tranche.shard_path):
        raise ValueError("source mapping receipt tranche shard mismatch")
    if frozen.get("tranche_manifest_sha256") != _sha256_file(tranche.manifest_path):
        raise ValueError("source mapping receipt tranche manifest mismatch")

    return expected_file_sha256


def _streaming_source_sha256s(
    config: Mapping[str, Any],
    source_mapping_receipt_sha256: str,
) -> tuple[str, ...]:
    return tuple(sorted({
        *config["legacy_v1"]["source_sha256s"],
        config["legacy_v1"]["checkpoint_sha256"],
        config["deck"]["sha256"],
        source_mapping_receipt_sha256,
    }))


def _historical_split_mapping(corpus: Any) -> tuple[dict[str, str], str, dict[str, Any]]:
    row_sha256s = tuple(corpus.historical_row_sha256s)
    if len(row_sha256s) != M3_HISTORICAL_ROW_COUNT or len(set(row_sha256s)) != len(row_sha256s):
        raise ValueError("M3 migration requires exactly 64 unique frozen historical rows")
    sorted_hashes = sorted(row_sha256s)
    split_by_hash = {
        row_sha256: ("heldout" if index % 2 == 0 else "train")
        for index, row_sha256 in enumerate(sorted_hashes)
    }
    artifact = {
        "schema_version": HISTORICAL_SPLIT_SCHEMA_VERSION,
        "train_row_sha256s": sorted(
            row_sha256 for row_sha256, split in split_by_hash.items() if split == "train"
        ),
        "heldout_row_sha256s": sorted(
            row_sha256 for row_sha256, split in split_by_hash.items() if split == "heldout"
        ),
    }
    mapping_sha256 = _sha256_bytes(_canonical_bytes(artifact))
    train_count = sum(split == "train" for split in split_by_hash.values())
    heldout_count = sum(split == "heldout" for split in split_by_hash.values())
    if train_count != M3_HISTORICAL_TRAIN_ROW_COUNT or heldout_count != M3_HISTORICAL_HELDOUT_ROW_COUNT:
        raise ValueError("M3 historical split must contain exactly 32 train and 32 heldout rows")
    return split_by_hash, mapping_sha256, artifact


def _examples_by_hash(corpus: Any, source_input_dim: int) -> dict[str, Any]:
    examples = v1._prepare_examples(corpus.dataset, source_input_dim)
    indexed = {example.row_sha256: example for example in examples}
    if len(indexed) != len(examples):
        raise ValueError("streaming corpus contains duplicate example identities")

    return indexed


def _ordered_batch(examples: Sequence[Any], generator: torch.Generator) -> list[Any]:
    if not examples:
        return []
    order = torch.randperm(len(examples), generator=generator).tolist()

    return [examples[index] for index in order]


def _choose_replay_rows(
    state: StreamingTrainingState,
    maximum_rows: int,
    sampler_generator: torch.Generator,
) -> tuple[str, ...]:
    if not state.replay_policy.allows_replay:
        return ()
    train_hashes = [row_sha256 for row_sha256, split in state.unique_row_splits if split == "train"]
    if not train_hashes:
        raise ValueError("explicit replay requires a previously consumed training row")
    count = min(maximum_rows, len(train_hashes))
    order = torch.randperm(len(train_hashes), generator=sampler_generator)[:count].tolist()

    return tuple(train_hashes[index] for index in order)


def _finite_optimizer_step(
    loss: torch.Tensor,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    amp_scaler: Any,
    gradient_clip_norm: float,
) -> tuple[float, float]:
    if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("streaming loss is non-finite or non-scalar")
    optimizer.zero_grad(set_to_none=True)
    amp_scaler.scale(loss).backward()
    amp_scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    if not bool(torch.isfinite(gradient_norm).item()):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("streaming gradient is non-finite")
    amp_scaler.step(optimizer)
    amp_scaler.update()
    scheduler.step()
    if any(not bool(torch.isfinite(value).all().item()) for value in model.state_dict().values()):
        raise FloatingPointError("streaming optimizer produced non-finite parameters")

    return float(loss.detach()), float(gradient_norm.detach())


def _advance_progress(
    progress: TrainingProgress,
    state: StreamingTrainingState,
    metrics: Mapping[str, float],
    *,
    optimizer_step_performed: bool,
    useful_gpu_seconds: float,
) -> TrainingProgress:
    best = dict(progress.best_metrics)
    for name, value in metrics.items():
        if math.isfinite(float(value)):
            best[name] = min(float(value), float(best.get(name, value)))
    update_increment = int(optimizer_step_performed)

    return TrainingProgress(
        epoch=progress.epoch,
        batch_cursor=progress.batch_cursor + update_increment,
        global_step=progress.global_step + update_increment,
        presentations=state.total_presentations,
        cumulative_gpu_seconds=progress.cumulative_gpu_seconds + useful_gpu_seconds,
        best_metrics=best,
        last_metrics=dict(metrics),
        safety_counters=dict(ZERO_SAFETY_COUNTERS),
    )


def _build_components(config: Mapping[str, Any], corpus: Any, device: torch.device) -> tuple[Any, ...]:
    source_identity = config["source_policy"]
    source_policy, source_payload = v1._load_source_policy(
        Path(source_identity["checkpoint"]).resolve(), source_identity["sha256"]
    )
    source_policy.eval()
    examples = _examples_by_hash(corpus, int(source_payload["input_dim"]))
    first = next(iter(examples.values()))
    action_dim = first.actions.shape[1]
    if any(example.actions.shape[1] != action_dim for example in examples.values()):
        raise ValueError("actor-visible action encoding dimension drifted across streaming shards")
    policy = copy.deepcopy(source_policy)
    q_head = v1.QDistributionHead(v1.value_source.FEATURE_DIM_V2, action_dim)
    bundle = v1.PolicyTrainingBundle(policy, q_head).to(device)
    anchor = copy.deepcopy(source_policy).to(device).eval()
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(
        bundle.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(optimizer_config["scheduler"]["t_0"]),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=bool(optimizer_config["amp_enabled"]))
    sampler = torch.Generator().manual_seed(int(config["seed"]))
    dataloader = torch.Generator().manual_seed(int(config["seed"]) + 1)

    return bundle, anchor, examples, optimizer, scheduler, scaler, sampler, dataloader


def _run_one_update(
    *,
    state: StreamingTrainingState,
    corpus: Any,
    row_split_policy: RowSplitPolicy,
    bundle: torch.nn.Module,
    anchor: torch.nn.Module,
    examples_by_hash: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    amp_scaler: Any,
    sampler_generator: torch.Generator,
    dataloader_generator: torch.Generator,
    device: torch.device,
    batch_size: int,
    gradient_clip_norm: float,
) -> tuple[StreamingTrainingState, dict[str, float], bool, float, float] | None:
    planned = plan_streaming_update(
        state,
        corpus,
        row_split_policy,
        maximum_train_rows=batch_size,
    )
    replay_hashes = () if planned else _choose_replay_rows(state, batch_size, sampler_generator)
    training_hashes = [row.row_sha256 for row in planned if row.split == "train"] + list(replay_hashes)
    if not training_hashes and not planned:
        return None
    candidate_state = commit_streaming_update(
        state,
        corpus,
        row_split_policy,
        unique_rows=planned,
        replay_row_sha256s=replay_hashes,
    )
    if not training_hashes:
        return candidate_state, {"heldout_rows_drained": float(len(planned))}, False, 0.0, 0.0
    batch_examples = _ordered_batch(
        [examples_by_hash[row_sha256] for row_sha256 in training_hashes],
        dataloader_generator,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    anchor_probabilities = v1._anchor_probabilities(anchor, batch_examples, device)
    loss, metrics = v1._policy_losses(
        bundle,
        v1._step_batch(batch_examples, device),
        anchor_probabilities,
        device,
    )
    loss_value, gradient_norm = _finite_optimizer_step(
        loss,
        model=bundle,
        optimizer=optimizer,
        scheduler=scheduler,
        amp_scaler=amp_scaler,
        gradient_clip_norm=gradient_clip_norm,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    device_step_seconds = time.perf_counter() - started
    metrics.update({
        "total_loss": loss_value,
        "gradient_norm": gradient_norm,
        "nonfinite_count": 0.0,
    })

    useful_gpu_seconds = device_step_seconds if device.type == "cuda" else 0.0
    local_cpu_seconds = device_step_seconds if device.type == "cpu" else 0.0

    return candidate_state, metrics, True, useful_gpu_seconds, local_cpu_seconds


def run(args: argparse.Namespace, *, _device: torch.device | None = None) -> dict[str, Any]:
    if isinstance(args.maximum_updates, bool) or args.maximum_updates <= 0:
        raise ValueError("--maximum-updates must be positive")
    config_path = Path(args.config).resolve()
    config, config_sha256 = _load_config(config_path)
    row_split_policy = RowSplitPolicy.from_dict(config["row_split_policy"])
    replay_policy = ReplayPolicy.from_dict(config["replay_policy"])
    derived_compatibility = _derived_trainer_compatibility(config["deck"], row_split_policy)
    if config["trainer_compatibility"] != derived_compatibility:
        raise ValueError("trainer compatibility does not match current feature, action, deck, and split bytes")
    tranche = ShardBinding(
        Path(config["m3_tranche"]["shard"]).resolve(),
        Path(config["m3_tranche"]["manifest"]).resolve(),
    )
    root_manifest_path = Path(config["root_manifest"]).resolve()
    if _sha256_file(root_manifest_path) != config["root_manifest_sha256"]:
        raise ValueError("root corpus manifest hash mismatch")
    corpus = load_manifest_bound_corpus(
        root_manifest_path,
        corpus_root=Path(config["corpus_root"]).resolve(),
        m3_tranche_binding=tranche,
        expected_trainer_compatibility=config["trainer_compatibility"],
    )
    source_mapping_receipt_sha256 = _verify_source_mapping_receipt(config, tranche)
    streaming_source_sha256s = _streaming_source_sha256s(config, source_mapping_receipt_sha256)
    split_mapping, split_mapping_sha256, split_artifact = _historical_split_mapping(corpus)
    if split_mapping_sha256 != config["historical_split_mapping_sha256"]:
        raise ValueError("historical 32/32 train/heldout mapping hash mismatch")
    device = _device or torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA streaming training but CUDA is unavailable")
    components = _build_components(config, corpus, device)
    bundle, anchor, examples, optimizer, scheduler, scaler, sampler, dataloader = components
    run_dir = Path(args.run_dir).resolve()
    checkpoint_path = run_dir / "streaming_training.pt"
    split_artifact_path = run_dir / "historical_split_mapping.json"
    legacy = config["legacy_v1"]
    if _sha256_file(Path(legacy["checkpoint"]).resolve()) != legacy["checkpoint_sha256"]:
        raise ValueError("legacy v1 checkpoint hash mismatch")
    if args.resume:
        if not checkpoint_path.is_file():
            raise ValueError("--resume requires an existing streaming-v2 checkpoint")
        progress, state, nonfinite_counters, migration_status = load_streaming_checkpoint(
            checkpoint_path,
            model=bundle,
            optimizer=optimizer,
            scheduler=scheduler,
            amp_scaler=scaler,
            sampler_generator=sampler,
            dataloader_generator=dataloader,
            corpus=corpus,
            row_split_policy=row_split_policy,
            expected_config_sha256=config_sha256,
            expected_source_sha256s=streaming_source_sha256s,
            map_location=device,
            allow_deferred_cuda_rng=bool(args.migration_only),
        )
        preserved_cuda_rng_state_count = None
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError("new streaming run directory must be absent or empty")
        run_dir.mkdir(parents=True, exist_ok=True)
        migration = migrate_v1_checkpoint_to_streaming(
            Path(legacy["checkpoint"]).resolve(),
            checkpoint_path,
            model=bundle,
            optimizer=optimizer,
            scheduler=scheduler,
            amp_scaler=scaler,
            sampler_generator=sampler,
            dataloader_generator=dataloader,
            corpus=corpus,
            row_split_policy=row_split_policy,
            replay_policy=replay_policy,
            historical_row_splits=split_mapping,
            expected_legacy_checkpoint_sha256=legacy["checkpoint_sha256"],
            expected_legacy_dataset_sha256=legacy["dataset_sha256"],
            expected_legacy_config_sha256=legacy["config_sha256"],
            expected_legacy_source_sha256s=legacy["source_sha256s"],
            expected_legacy_split_mapping_sha256=split_mapping_sha256,
            expected_legacy_source_mapping_sha256=config["source_mapping_receipt"]["ordered_source_mapping_sha256"],
            streaming_config_sha256=config_sha256,
            streaming_source_sha256s=streaming_source_sha256s,
            dataloader_seed=int(config["seed"]) + 1,
            nonfinite_counters=ZERO_NONFINITE_COUNTERS,
            map_location=device,
        )
        progress = migration.progress
        state = migration.streaming_state
        nonfinite_counters = dict(ZERO_NONFINITE_COUNTERS)
        migration_status = migration.cuda_rng_application_status
        preserved_cuda_rng_state_count = migration.preserved_cuda_rng_state_count
        if progress.presentations != int(legacy["presentations"]):
            raise ValueError("legacy v1 presentation count disagrees with the migration contract")
        validate_streaming_state_against_corpus(state, corpus, row_split_policy)
        _atomic_json(split_artifact_path, split_artifact)
    _verify_optimizer_contract(optimizer, scheduler, scaler, config["optimizer"])
    if progress.safety_counters != ZERO_SAFETY_COUNTERS:
        raise ValueError("streaming source checkpoint is not safety-clean")
    if migration_status == "deferred_unavailable_structurally_preserved" and not args.migration_only:
        raise RuntimeError("CUDA RNG bytes were structurally preserved; use --migration-only or resume on CUDA")
    latest_metrics: dict[str, float] = {}
    updates_this_invocation = 0
    cursor_transactions_this_invocation = 0
    local_cpu_step_seconds = 0.0
    scheduled_updates = 0 if args.migration_only else args.maximum_updates
    for _ in range(scheduled_updates):
        outcome = _run_one_update(
            state=state,
            corpus=corpus,
            row_split_policy=row_split_policy,
            bundle=bundle,
            anchor=anchor,
            examples_by_hash=examples,
            optimizer=optimizer,
            scheduler=scheduler,
            amp_scaler=scaler,
            sampler_generator=sampler,
            dataloader_generator=dataloader,
            device=device,
            batch_size=int(config["batch_size"]),
            gradient_clip_norm=float(config["optimizer"]["gradient_clip_norm"]),
        )
        if outcome is None:
            break
        state, latest_metrics, optimizer_step_performed, useful_gpu_seconds, local_cpu_seconds = outcome
        updates_this_invocation += int(optimizer_step_performed)
        cursor_transactions_this_invocation += 1
        local_cpu_step_seconds += local_cpu_seconds
        progress = _advance_progress(
            progress,
            state,
            latest_metrics,
            optimizer_step_performed=optimizer_step_performed,
            useful_gpu_seconds=useful_gpu_seconds,
        )
        atomic_save_streaming_checkpoint(
            checkpoint_path,
            model=bundle,
            optimizer=optimizer,
            scheduler=scheduler,
            amp_scaler=scaler,
            sampler_generator=sampler,
            dataloader_generator=dataloader,
            progress=progress,
            streaming_state=state,
            corpus=corpus,
            row_split_policy=row_split_policy,
            config_sha256=config_sha256,
            source_sha256s=streaming_source_sha256s,
            nonfinite_counters=nonfinite_counters,
        )
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "classification": Q_FEATURE_CLASSIFICATION,
        "status": "migrated_only" if args.migration_only else "checkpointed",
        "root_corpus_manifest_sha256": corpus.root_manifest_sha256,
        "corpus_lineage_id": corpus.corpus_lineage_id,
        "config_sha256": config_sha256,
        "source_mapping_receipt_sha256": source_mapping_receipt_sha256,
        "historical_split_mapping_sha256": split_mapping_sha256,
        "checkpoint": str(checkpoint_path),
        "updates_this_invocation": updates_this_invocation,
        "cursor_transactions_this_invocation": cursor_transactions_this_invocation,
        "optimizer_updates": progress.global_step,
        "presentations": progress.presentations,
        "cumulative_gpu_seconds": progress.cumulative_gpu_seconds,
        "local_cpu_step_seconds_this_invocation": local_cpu_step_seconds,
        "cuda_rng_application_status": migration_status,
        "preserved_cuda_rng_state_count": preserved_cuda_rng_state_count,
        "streaming_state": state.to_dict(),
        "latest_metrics": latest_metrics,
        "safety_counters": dict(ZERO_SAFETY_COUNTERS),
        "paid_compute_launched": False,
        "policy_control_eligible": False,
    }
    _atomic_json(run_dir / "result.json", result)

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-decision-receipt", default="")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--maximum-updates", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--migration-only", action="store_true")

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
