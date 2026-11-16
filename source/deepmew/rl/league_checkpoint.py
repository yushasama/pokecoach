"""Fail-closed full-state checkpoints for research league learners.

The helper is deliberately independent of concrete environments and samplers.
Callers provide explicit state hooks; environment hooks must expose only the
RNG/cursor state needed to resume and must never serialize private game state.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import nn


CHECKPOINT_SCHEMA_VERSION = "deepmew.rl.league_checkpoint.v1"
CHECKPOINT_RECEIPT_SCHEMA_VERSION = "deepmew.rl.league_checkpoint_receipt.v1"
RESTORE_RECEIPT_SCHEMA_VERSION = "deepmew.rl.league_checkpoint_restore.v1"
RESUME_PROOF_SCHEMA_VERSION = "deepmew.rl.league_resume_counter_proof.v1"
CHECKPOINT_READY_SCHEMA_VERSION = "deepmew.rl.league_checkpoint_ready.v1"
CHECKPOINT_POINTER_SCHEMA_VERSION = "deepmew.rl.league_checkpoint_pointer.v1"

_BINDING_FIELDS = frozenset({
    "learner_id",
    "training_config_sha256",
    "opponent_mixture_receipt",
    "league_snapshot_receipt",
    "deck_binding_receipt",
    "observation_schema_sha256",
    "action_schema_sha256",
})
_PROGRESS_FIELDS = frozenset({
    "environment_decisions",
    "update_count",
    "cumulative_training_seconds",
    "counters",
})
_CHECKPOINT_FIELDS = frozenset({
    "schema_version",
    "actor_state",
    "critic_state",
    "optimizer_state",
    "scheduler_state",
    "scaler_state",
    "progress",
    "python_rng_state",
    "numpy_rng_state",
    "torch_cpu_rng_state",
    "torch_cuda_rng_states",
    "environment_state",
    "sampler_state",
    "bindings",
    "validation_receipt",
})
_RECEIPT_FIELDS = frozenset({
    "schema_version",
    "checkpoint_schema_version",
    "checkpoint_sha256",
    "bindings_sha256",
    "binding_receipts",
    "validation_receipt_sha256",
    "validation_receipt",
    "progress",
    "cuda_rng_state_count",
})
_READY_FIELDS = frozenset({
    "schema_version",
    "generation_id",
    "generation_index",
    "checkpoint_file",
    "checkpoint_sha256",
    "receipt_file",
    "receipt_sha256",
    "progress",
})
_POINTER_FIELDS = frozenset({
    "schema_version",
    "generation_id",
    "generation_index",
    "ready_file",
    "ready_sha256",
    "checkpoint_sha256",
})


class LeagueCheckpointError(ValueError):
    """A checkpoint failed integrity, compatibility, or exact restore."""


def _canonical_json_bytes(value: Any, *, name: str) -> bytes:
    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise LeagueCheckpointError(f"{name} must be finite JSON data") from exc

    return rendered.encode("ascii")


def _json_object(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise LeagueCheckpointError(f"{name} must be a non-empty mapping")
    normalized = json.loads(_canonical_json_bytes(dict(value), name=name))
    if not isinstance(normalized, dict):
        raise LeagueCheckpointError(f"{name} must encode a JSON object")

    return normalized


def _json_sha256(value: Mapping[str, Any], *, name: str) -> str:
    return hashlib.sha256(_canonical_json_bytes(dict(value), name=name)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _snapshot_file_receipt(path: Path, *, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _normalized_sha256(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise LeagueCheckpointError(f"{name} must be a SHA-256 string")
    normalized = value.lower().removeprefix("sha256:")
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise LeagueCheckpointError(f"{name} must be a SHA-256 string")

    return normalized


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LeagueCheckpointError(f"{name} must be a non-negative integer")

    return value


def _atomic_bytes_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def checkpoint_receipt_path(path: str | Path) -> Path:
    """Return the receipt path used by legacy fixed-file checkpoints."""

    checkpoint_path = Path(path)

    return checkpoint_path.with_suffix(f"{checkpoint_path.suffix}.sha256.json")


def checkpoint_latest_ready_path(path: str | Path) -> Path:
    """Return the stable pointer written after a generation is fully ready."""

    return Path(f"{Path(path)}.latest.json")


def _checkpoint_generation_dir(path: Path) -> Path:
    return Path(f"{path}.generations")


def _generation_paths(path: Path, generation_id: str) -> tuple[Path, Path, Path]:
    generation_dir = _checkpoint_generation_dir(path)

    return (
        generation_dir / f"{generation_id}.pt",
        generation_dir / f"{generation_id}.receipt.json",
        generation_dir / f"{generation_id}.ready.json",
    )


@dataclass(frozen=True)
class StateHooks:
    """Explicit, concrete-type-free state capture and restore hooks."""

    getter: Callable[[], Any]
    setter: Callable[[Any], None]
    label: str

    def __post_init__(self) -> None:
        if not callable(self.getter) or not callable(self.setter):
            raise TypeError(f"{self.label} getter and setter must be callable")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("state hook label must be non-empty")

    def capture(self) -> Any:
        try:
            return copy.deepcopy(self.getter())
        except Exception as exc:
            raise LeagueCheckpointError(f"could not capture {self.label} state") from exc

    def restore(self, state: Any) -> None:
        try:
            self.setter(copy.deepcopy(state))
        except Exception as exc:
            raise LeagueCheckpointError(f"could not restore {self.label} state") from exc


@dataclass(frozen=True)
class LeagueCheckpointBindings:
    """Immutable identities that must match before any learner state is loaded."""

    learner_id: str
    training_config_sha256: str
    opponent_mixture_receipt: Mapping[str, Any]
    league_snapshot_receipt: Mapping[str, Any]
    deck_binding_receipt: Mapping[str, Any]
    observation_schema_sha256: str
    action_schema_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.learner_id, str) or not self.learner_id.strip():
            raise LeagueCheckpointError("learner_id must be non-empty")
        object.__setattr__(
            self,
            "training_config_sha256",
            _normalized_sha256(self.training_config_sha256, name="training_config_sha256"),
        )
        object.__setattr__(
            self,
            "observation_schema_sha256",
            _normalized_sha256(self.observation_schema_sha256, name="observation_schema_sha256"),
        )
        object.__setattr__(
            self,
            "action_schema_sha256",
            _normalized_sha256(self.action_schema_sha256, name="action_schema_sha256"),
        )
        for field_name in (
            "opponent_mixture_receipt",
            "league_snapshot_receipt",
            "deck_binding_receipt",
        ):
            object.__setattr__(
                self,
                field_name,
                _json_object(getattr(self, field_name), name=field_name),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "learner_id": self.learner_id,
            "training_config_sha256": self.training_config_sha256,
            "opponent_mixture_receipt": copy.deepcopy(dict(self.opponent_mixture_receipt)),
            "league_snapshot_receipt": copy.deepcopy(dict(self.league_snapshot_receipt)),
            "deck_binding_receipt": copy.deepcopy(dict(self.deck_binding_receipt)),
            "observation_schema_sha256": self.observation_schema_sha256,
            "action_schema_sha256": self.action_schema_sha256,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "LeagueCheckpointBindings":
        if not isinstance(value, dict) or set(value) != _BINDING_FIELDS:
            raise LeagueCheckpointError("checkpoint bindings are incomplete or unexpected")

        return cls(**value)

    def receipt_hashes(self) -> dict[str, str]:
        return {
            "training_config_sha256": self.training_config_sha256,
            "opponent_mixture_receipt_sha256": _json_sha256(
                self.opponent_mixture_receipt,
                name="opponent_mixture_receipt",
            ),
            "league_snapshot_receipt_sha256": _json_sha256(
                self.league_snapshot_receipt,
                name="league_snapshot_receipt",
            ),
            "deck_binding_receipt_sha256": _json_sha256(
                self.deck_binding_receipt,
                name="deck_binding_receipt",
            ),
            "observation_schema_sha256": self.observation_schema_sha256,
            "action_schema_sha256": self.action_schema_sha256,
        }


@dataclass(frozen=True)
class LeagueTrainingProgress:
    """Cumulative counters that a retry must restore without double counting."""

    environment_decisions: int
    update_count: int
    cumulative_training_seconds: float
    counters: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "environment_decisions",
            _nonnegative_int(self.environment_decisions, name="environment_decisions"),
        )
        object.__setattr__(
            self,
            "update_count",
            _nonnegative_int(self.update_count, name="update_count"),
        )
        seconds = float(self.cumulative_training_seconds)
        if not math.isfinite(seconds) or seconds < 0.0:
            raise LeagueCheckpointError("cumulative_training_seconds must be finite and non-negative")
        object.__setattr__(self, "cumulative_training_seconds", seconds)
        if not isinstance(self.counters, Mapping):
            raise LeagueCheckpointError("counters must be a mapping")
        normalized: dict[str, int] = {}
        for name, value in self.counters.items():
            if not isinstance(name, str) or not name:
                raise LeagueCheckpointError("counter names must be non-empty strings")
            normalized[name] = _nonnegative_int(value, name=f"counter {name}")
        object.__setattr__(self, "counters", dict(sorted(normalized.items())))

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment_decisions": self.environment_decisions,
            "update_count": self.update_count,
            "cumulative_training_seconds": self.cumulative_training_seconds,
            "counters": dict(self.counters),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "LeagueTrainingProgress":
        if not isinstance(value, dict) or set(value) != _PROGRESS_FIELDS:
            raise LeagueCheckpointError("checkpoint progress is incomplete or unexpected")

        return cls(**value)


@dataclass(frozen=True)
class LeagueCheckpointLoadResult:
    progress: LeagueTrainingProgress
    validation_receipt: Mapping[str, Any]
    checkpoint_sha256: str
    restore_receipt: Mapping[str, Any]


@dataclass(frozen=True)
class _ResolvedCheckpoint:
    checkpoint: Mapping[str, Any]
    receipt: Mapping[str, Any]
    checkpoint_path: Path
    checkpoint_sha256: str
    storage_format: str
    generation_id: str | None
    generation_index: int | None


def _component_state(component: Any, *, name: str) -> Any:
    if component is None:
        return None
    if not hasattr(component, "state_dict") or not hasattr(component, "load_state_dict"):
        raise TypeError(f"{name} must provide state_dict and load_state_dict")

    return copy.deepcopy(component.state_dict())


def _validate_optional_component(component: Any, state: Any, *, name: str) -> None:
    if (component is None) != (state is None):
        raise LeagueCheckpointError(f"checkpoint {name} presence is incompatible")
    if component is not None and (
        not hasattr(component, "state_dict") or not hasattr(component, "load_state_dict")
    ):
        raise TypeError(f"{name} must provide state_dict and load_state_dict")


def _assert_finite_nested(value: Any, *, name: str) -> None:
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all().item():
            raise LeagueCheckpointError(f"{name} contains non-finite tensors")
        return
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"f", "c"} and not np.isfinite(value).all():
            raise LeagueCheckpointError(f"{name} contains non-finite arrays")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LeagueCheckpointError(f"{name} contains non-finite values")
        return
    if isinstance(value, Mapping):
        for child in value.values():
            _assert_finite_nested(child, name=name)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _assert_finite_nested(child, name=name)


def _validate_module_state(module: nn.Module, state: Any, *, name: str) -> None:
    if not isinstance(module, nn.Module):
        raise TypeError(f"{name} must be a torch.nn.Module")
    if not isinstance(state, Mapping):
        raise LeagueCheckpointError(f"checkpoint {name} state must be a mapping")
    current = module.state_dict()
    if set(current) != set(state):
        raise LeagueCheckpointError(f"checkpoint {name} parameter names are incompatible")
    for key, current_value in current.items():
        saved_value = state[key]
        if isinstance(current_value, torch.Tensor):
            if not isinstance(saved_value, torch.Tensor):
                raise LeagueCheckpointError(f"checkpoint {name}.{key} type is incompatible")
            if current_value.shape != saved_value.shape or current_value.dtype != saved_value.dtype:
                raise LeagueCheckpointError(f"checkpoint {name}.{key} shape or dtype is incompatible")
    _assert_finite_nested(state, name=f"checkpoint {name} state")


def _validate_optimizer_state(optimizer: Any, state: Any) -> None:
    if not hasattr(optimizer, "state_dict") or not hasattr(optimizer, "load_state_dict"):
        raise TypeError("optimizer must provide state_dict and load_state_dict")
    if not isinstance(state, dict) or set(state) != {"state", "param_groups"}:
        raise LeagueCheckpointError("checkpoint optimizer state is malformed")
    current = optimizer.state_dict()
    saved_groups = state["param_groups"]
    current_groups = current.get("param_groups")
    if not isinstance(saved_groups, list) or not isinstance(current_groups, list):
        raise LeagueCheckpointError("checkpoint optimizer parameter groups are malformed")
    if len(saved_groups) != len(current_groups):
        raise LeagueCheckpointError("checkpoint optimizer group count is incompatible")
    for saved_group, current_group in zip(saved_groups, current_groups, strict=True):
        if not isinstance(saved_group, dict) or not isinstance(current_group, dict):
            raise LeagueCheckpointError("checkpoint optimizer parameter group is malformed")
        if len(saved_group.get("params", [])) != len(current_group.get("params", [])):
            raise LeagueCheckpointError("checkpoint optimizer parameter count is incompatible")
    _assert_finite_nested(state, name="checkpoint optimizer state")


def _checkpoint_receipt(
    *,
    checkpoint_sha256: str,
    bindings: LeagueCheckpointBindings,
    progress: LeagueTrainingProgress,
    validation_receipt: Mapping[str, Any],
    cuda_rng_state_count: int,
) -> dict[str, Any]:
    binding_payload = bindings.to_dict()
    validation_payload = _json_object(validation_receipt, name="validation_receipt")

    return {
        "schema_version": CHECKPOINT_RECEIPT_SCHEMA_VERSION,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_sha256": _normalized_sha256(checkpoint_sha256, name="checkpoint_sha256"),
        "bindings_sha256": _json_sha256(binding_payload, name="checkpoint bindings"),
        "binding_receipts": bindings.receipt_hashes(),
        "validation_receipt_sha256": _json_sha256(
            validation_payload,
            name="validation_receipt",
        ),
        "validation_receipt": validation_payload,
        "progress": progress.to_dict(),
        "cuda_rng_state_count": _nonnegative_int(
            cuda_rng_state_count,
            name="cuda_rng_state_count",
        ),
    }


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise LeagueCheckpointError(f"immutable checkpoint artifact already exists: {path.name}")
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(handle)
    temporary_path = Path(temporary_name)
    try:
        torch.save(value, temporary_path)
        with temporary_path.open("r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise LeagueCheckpointError(
                f"immutable checkpoint artifact already exists: {path.name}"
            )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_immutable_bytes_write(path: Path, payload: bytes) -> None:
    if path.exists():
        raise LeagueCheckpointError(f"immutable checkpoint artifact already exists: {path.name}")
    _atomic_bytes_write(path, payload)


def _read_json_file(
    path: Path,
    *,
    fields: frozenset[str],
    name: str,
) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LeagueCheckpointError(f"{name} could not be decoded") from exc
    if not isinstance(value, dict) or set(value) != fields:
        raise LeagueCheckpointError(f"{name} is incomplete or unexpected")

    return value


def _validate_generation_identity(
    generation_id: Any,
    generation_index: Any,
) -> tuple[str, int]:
    index = _nonnegative_int(generation_index, name="checkpoint generation index")
    if not isinstance(generation_id, str):
        raise LeagueCheckpointError("checkpoint generation id is malformed")
    prefix, separator, unique_suffix = generation_id.partition("-")
    if (
        separator != "-"
        or len(prefix) != 21
        or not prefix.startswith("g")
        or not prefix[1:].isdigit()
        or len(unique_suffix) != 32
        or any(character not in "0123456789abcdef" for character in unique_suffix)
        or int(prefix[1:]) != index
    ):
        raise LeagueCheckpointError("checkpoint generation id is malformed")

    return generation_id, index


def _read_ready_marker(path: Path) -> dict[str, Any]:
    ready = _read_json_file(
        path,
        fields=_READY_FIELDS,
        name="checkpoint ready marker",
    )
    if ready["schema_version"] != CHECKPOINT_READY_SCHEMA_VERSION:
        raise LeagueCheckpointError("checkpoint ready marker schema is incompatible")
    generation_id, generation_index = _validate_generation_identity(
        ready["generation_id"],
        ready["generation_index"],
    )
    if path.name != f"{generation_id}.ready.json":
        raise LeagueCheckpointError("checkpoint ready marker name is incompatible")
    if ready["checkpoint_file"] != f"{generation_id}.pt":
        raise LeagueCheckpointError("checkpoint ready marker checkpoint path is incompatible")
    if ready["receipt_file"] != f"{generation_id}.receipt.json":
        raise LeagueCheckpointError("checkpoint ready marker receipt path is incompatible")
    ready["generation_id"] = generation_id
    ready["generation_index"] = generation_index
    ready["checkpoint_sha256"] = _normalized_sha256(
        ready["checkpoint_sha256"],
        name="checkpoint ready marker checkpoint sha256",
    )
    ready["receipt_sha256"] = _normalized_sha256(
        ready["receipt_sha256"],
        name="checkpoint ready marker receipt sha256",
    )
    ready["progress"] = LeagueTrainingProgress.from_dict(ready["progress"]).to_dict()

    return ready


def _read_latest_pointer(path: Path) -> dict[str, Any]:
    pointer = _read_json_file(
        path,
        fields=_POINTER_FIELDS,
        name="checkpoint latest pointer",
    )
    if pointer["schema_version"] != CHECKPOINT_POINTER_SCHEMA_VERSION:
        raise LeagueCheckpointError("checkpoint latest pointer schema is incompatible")
    generation_id, generation_index = _validate_generation_identity(
        pointer["generation_id"],
        pointer["generation_index"],
    )
    expected_ready_file = f"{generation_id}.ready.json"
    if pointer["ready_file"] != expected_ready_file:
        raise LeagueCheckpointError("checkpoint latest pointer ready path is incompatible")
    pointer["generation_id"] = generation_id
    pointer["generation_index"] = generation_index
    pointer["ready_sha256"] = _normalized_sha256(
        pointer["ready_sha256"],
        name="checkpoint latest pointer ready sha256",
    )
    pointer["checkpoint_sha256"] = _normalized_sha256(
        pointer["checkpoint_sha256"],
        name="checkpoint latest pointer checkpoint sha256",
    )

    return pointer


def _next_generation_index(checkpoint_path: Path) -> int:
    highest_index = 0
    pointer_path = checkpoint_latest_ready_path(checkpoint_path)
    if pointer_path.is_file():
        try:
            highest_index = _read_latest_pointer(pointer_path)["generation_index"]
        except LeagueCheckpointError:
            pass
    generation_dir = _checkpoint_generation_dir(checkpoint_path)
    if generation_dir.is_dir():
        for ready_path in generation_dir.glob("*.ready.json"):
            try:
                ready = _read_ready_marker(ready_path)
            except LeagueCheckpointError:
                continue
            highest_index = max(highest_index, ready["generation_index"])

    return highest_index + 1


def atomic_save_league_checkpoint(
    path: str | Path,
    *,
    actor: nn.Module,
    critic: nn.Module,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    environment_state: StateHooks,
    sampler_state: StateHooks,
    progress: LeagueTrainingProgress,
    bindings: LeagueCheckpointBindings,
    validation_receipt: Mapping[str, Any],
) -> str:
    """Commit an immutable checkpoint generation and then advance its pointer."""

    checkpoint_path = Path(path).resolve()
    validation_payload = _json_object(validation_receipt, name="validation_receipt")
    actor_state = copy.deepcopy(actor.state_dict())
    critic_state = copy.deepcopy(critic.state_dict())
    optimizer_state = _component_state(optimizer, name="optimizer")
    scheduler_state = _component_state(scheduler, name="scheduler")
    scaler_state = _component_state(scaler, name="scaler")
    _validate_module_state(actor, actor_state, name="actor")
    _validate_module_state(critic, critic_state, name="critic")
    _validate_optimizer_state(optimizer, optimizer_state)
    _assert_finite_nested(scheduler_state, name="scheduler state")
    _assert_finite_nested(scaler_state, name="scaler state")
    cuda_rng_states = (
        [state.detach().cpu().clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else []
    )
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "actor_state": actor_state,
        "critic_state": critic_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "scaler_state": scaler_state,
        "progress": progress.to_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.random.get_rng_state(),
        "torch_cuda_rng_states": cuda_rng_states,
        "environment_state": environment_state.capture(),
        "sampler_state": sampler_state.capture(),
        "bindings": bindings.to_dict(),
        "validation_receipt": validation_payload,
    }
    generation_index = _next_generation_index(checkpoint_path)
    generation_id = f"g{generation_index:020d}-{uuid.uuid4().hex}"
    generation_checkpoint_path, generation_receipt_path, ready_path = _generation_paths(
        checkpoint_path,
        generation_id,
    )
    _atomic_torch_save(generation_checkpoint_path, checkpoint)
    checkpoint_sha256 = _file_sha256(generation_checkpoint_path)
    receipt = _checkpoint_receipt(
        checkpoint_sha256=checkpoint_sha256,
        bindings=bindings,
        progress=progress,
        validation_receipt=validation_payload,
        cuda_rng_state_count=len(cuda_rng_states),
    )
    _atomic_immutable_bytes_write(
        generation_receipt_path,
        _canonical_json_bytes(receipt, name="checkpoint receipt") + b"\n",
    )
    receipt_sha256 = _file_sha256(generation_receipt_path)
    ready = {
        "schema_version": CHECKPOINT_READY_SCHEMA_VERSION,
        "generation_id": generation_id,
        "generation_index": generation_index,
        "checkpoint_file": generation_checkpoint_path.name,
        "checkpoint_sha256": checkpoint_sha256,
        "receipt_file": generation_receipt_path.name,
        "receipt_sha256": receipt_sha256,
        "progress": progress.to_dict(),
    }
    _atomic_immutable_bytes_write(
        ready_path,
        _canonical_json_bytes(ready, name="checkpoint ready marker") + b"\n",
    )
    pointer = {
        "schema_version": CHECKPOINT_POINTER_SCHEMA_VERSION,
        "generation_id": generation_id,
        "generation_index": generation_index,
        "ready_file": ready_path.name,
        "ready_sha256": _file_sha256(ready_path),
        "checkpoint_sha256": checkpoint_sha256,
    }
    _atomic_bytes_write(
        checkpoint_latest_ready_path(checkpoint_path),
        _canonical_json_bytes(pointer, name="checkpoint latest pointer") + b"\n",
    )

    return checkpoint_sha256


def _read_receipt(path: Path) -> dict[str, Any]:
    return _read_json_file(
        path,
        fields=_RECEIPT_FIELDS,
        name="checkpoint receipt",
    )


def _load_checkpoint_pair(
    checkpoint_path: Path,
    receipt_path: Path,
    *,
    map_location: str | torch.device,
    storage_format: str,
    generation_id: str | None,
    generation_index: int | None,
) -> _ResolvedCheckpoint:
    receipt = _read_receipt(receipt_path)
    try:
        checkpoint_sha256 = _file_sha256(checkpoint_path)
    except OSError as exc:
        raise LeagueCheckpointError("league checkpoint could not be read") from exc
    if checkpoint_sha256 != _normalized_sha256(
        receipt["checkpoint_sha256"],
        name="checkpoint receipt sha256",
    ):
        raise LeagueCheckpointError("checkpoint hash does not match its receipt")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise LeagueCheckpointError("league checkpoint could not be decoded") from exc
    if not isinstance(checkpoint, dict) or set(checkpoint) != _CHECKPOINT_FIELDS:
        raise LeagueCheckpointError("league checkpoint is incomplete or unexpected")
    if checkpoint["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise LeagueCheckpointError("league checkpoint schema is incompatible")
    bindings = LeagueCheckpointBindings.from_dict(checkpoint["bindings"])
    progress = LeagueTrainingProgress.from_dict(checkpoint["progress"])
    validation_payload = _json_object(
        checkpoint["validation_receipt"],
        name="validation_receipt",
    )
    cuda_states = checkpoint["torch_cuda_rng_states"]
    if not isinstance(cuda_states, list) or any(
        not isinstance(value, torch.Tensor) for value in cuda_states
    ):
        raise LeagueCheckpointError("checkpoint CUDA RNG state is malformed")
    expected_receipt = _checkpoint_receipt(
        checkpoint_sha256=checkpoint_sha256,
        bindings=bindings,
        progress=progress,
        validation_receipt=validation_payload,
        cuda_rng_state_count=len(cuda_states),
    )
    if receipt != expected_receipt:
        raise LeagueCheckpointError("checkpoint receipt disagrees with checkpoint contents")

    return _ResolvedCheckpoint(
        checkpoint=checkpoint,
        receipt=receipt,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        storage_format=storage_format,
        generation_id=generation_id,
        generation_index=generation_index,
    )


def _load_ready_generation(
    checkpoint_path: Path,
    ready_path: Path,
    *,
    map_location: str | torch.device,
    expected_pointer: Mapping[str, Any] | None = None,
) -> _ResolvedCheckpoint:
    ready = _read_ready_marker(ready_path)
    generation_id = ready["generation_id"]
    generation_index = ready["generation_index"]
    generation_checkpoint_path, generation_receipt_path, expected_ready_path = _generation_paths(
        checkpoint_path,
        generation_id,
    )
    if ready_path != expected_ready_path:
        raise LeagueCheckpointError("checkpoint ready marker location is incompatible")
    if expected_pointer is not None:
        if (
            expected_pointer["generation_id"] != generation_id
            or expected_pointer["generation_index"] != generation_index
            or expected_pointer["ready_file"] != ready_path.name
        ):
            raise LeagueCheckpointError("checkpoint latest pointer disagrees with ready marker")
        try:
            ready_sha256 = _file_sha256(ready_path)
        except OSError as exc:
            raise LeagueCheckpointError("checkpoint ready marker could not be read") from exc
        if ready_sha256 != expected_pointer["ready_sha256"]:
            raise LeagueCheckpointError("checkpoint ready marker hash disagrees with latest pointer")
        if ready["checkpoint_sha256"] != expected_pointer["checkpoint_sha256"]:
            raise LeagueCheckpointError("checkpoint hash disagrees with latest pointer")
    try:
        receipt_sha256 = _file_sha256(generation_receipt_path)
    except OSError as exc:
        raise LeagueCheckpointError("checkpoint generation receipt could not be read") from exc
    if receipt_sha256 != ready["receipt_sha256"]:
        raise LeagueCheckpointError("checkpoint generation receipt hash disagrees with ready marker")
    resolved = _load_checkpoint_pair(
        generation_checkpoint_path,
        generation_receipt_path,
        map_location=map_location,
        storage_format="immutable_generation_v1",
        generation_id=generation_id,
        generation_index=generation_index,
    )
    if resolved.checkpoint_sha256 != ready["checkpoint_sha256"]:
        raise LeagueCheckpointError("checkpoint generation hash disagrees with ready marker")
    if resolved.checkpoint["progress"] != ready["progress"]:
        raise LeagueCheckpointError("checkpoint generation progress disagrees with ready marker")

    return resolved


def _scan_ready_generations(
    checkpoint_path: Path,
    *,
    map_location: str | torch.device,
    maximum_generation_index: int | None = None,
) -> _ResolvedCheckpoint | None:
    generation_dir = _checkpoint_generation_dir(checkpoint_path)
    if not generation_dir.is_dir():
        return None
    candidates: list[tuple[int, str, Path]] = []
    for ready_path in generation_dir.glob("*.ready.json"):
        try:
            ready = _read_ready_marker(ready_path)
        except LeagueCheckpointError:
            continue
        generation_index = ready["generation_index"]
        if maximum_generation_index is not None and generation_index > maximum_generation_index:
            continue
        candidates.append((generation_index, ready["generation_id"], ready_path))
    candidates.sort(reverse=True)
    for _, _, ready_path in candidates:
        try:
            return _load_ready_generation(
                checkpoint_path,
                ready_path,
                map_location=map_location,
            )
        except LeagueCheckpointError:
            continue

    return None


def _load_legacy_checkpoint(
    checkpoint_path: Path,
    *,
    map_location: str | torch.device,
) -> _ResolvedCheckpoint:
    return _load_checkpoint_pair(
        checkpoint_path,
        checkpoint_receipt_path(checkpoint_path),
        map_location=map_location,
        storage_format="legacy_fixed_file_v1",
        generation_id=None,
        generation_index=None,
    )


def _resolve_checkpoint(
    checkpoint_path: Path,
    *,
    map_location: str | torch.device,
) -> _ResolvedCheckpoint:
    pointer_path = checkpoint_latest_ready_path(checkpoint_path)
    last_error: LeagueCheckpointError | None = None
    maximum_visible_generation_index: int | None = None
    if pointer_path.is_file():
        try:
            pointer = _read_latest_pointer(pointer_path)
        except LeagueCheckpointError as exc:
            last_error = exc
            recovered = _scan_ready_generations(
                checkpoint_path,
                map_location=map_location,
            )
            if recovered is not None:
                return recovered
        else:
            maximum_visible_generation_index = pointer["generation_index"]
            ready_path = _checkpoint_generation_dir(checkpoint_path) / pointer["ready_file"]
            try:
                return _load_ready_generation(
                    checkpoint_path,
                    ready_path,
                    map_location=map_location,
                    expected_pointer=pointer,
                )
            except LeagueCheckpointError as exc:
                last_error = exc
                recovered = _scan_ready_generations(
                    checkpoint_path,
                    map_location=map_location,
                    maximum_generation_index=pointer["generation_index"] - 1,
                )
                if recovered is not None:
                    return recovered
        if checkpoint_path.is_file():
            try:
                return _load_legacy_checkpoint(
                    checkpoint_path,
                    map_location=map_location,
                )
            except LeagueCheckpointError as exc:
                last_error = exc
    elif checkpoint_path.is_file():
        try:
            return _load_legacy_checkpoint(
                checkpoint_path,
                map_location=map_location,
            )
        except LeagueCheckpointError as exc:
            last_error = exc
    recovered = _scan_ready_generations(
        checkpoint_path,
        map_location=map_location,
        maximum_generation_index=maximum_visible_generation_index,
    )
    if recovered is not None:
        return recovered
    if last_error is not None:
        raise LeagueCheckpointError(
            "no complete checkpoint generation or valid legacy checkpoint could be resolved"
        ) from last_error

    raise LeagueCheckpointError("no checkpoint generation or legacy checkpoint exists")


def snapshot_league_checkpoint(
    source_path: str | Path,
    destination_path: str | Path,
) -> dict[str, Any]:
    """Copy one validated logical checkpoint into a self-contained snapshot.

    Generation-backed checkpoints retain the selected immutable generation and
    receive a fresh pointer. Legacy fixed-file checkpoints retain their
    checkpoint and receipt pair. Existing destination files are never replaced.
    """

    source_checkpoint_path = Path(source_path)
    destination_checkpoint_path = Path(destination_path)
    destination_root = destination_checkpoint_path.parent
    resolved = _resolve_checkpoint(source_checkpoint_path, map_location="cpu")
    copied_paths: list[Path] = []
    if resolved.storage_format == "immutable_generation_v1":
        if resolved.generation_id is None or resolved.generation_index is None:
            raise LeagueCheckpointError("resolved generation checkpoint lacks generation identity")
        source_generation_paths = _generation_paths(
            source_checkpoint_path,
            resolved.generation_id,
        )
        destination_generation_paths = _generation_paths(
            destination_checkpoint_path,
            resolved.generation_id,
        )
        for destination in destination_generation_paths:
            if destination.exists():
                raise LeagueCheckpointError(f"checkpoint snapshot destination already exists: {destination}")
        for source, destination in zip(source_generation_paths, destination_generation_paths):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied_paths.append(destination)
        destination_checkpoint, _, destination_ready = destination_generation_paths
        ready = _read_ready_marker(destination_ready)
        pointer = {
            "schema_version": CHECKPOINT_POINTER_SCHEMA_VERSION,
            "generation_id": resolved.generation_id,
            "generation_index": resolved.generation_index,
            "ready_file": destination_ready.name,
            "ready_sha256": _file_sha256(destination_ready),
            "checkpoint_sha256": _file_sha256(destination_checkpoint),
        }
        destination_pointer = checkpoint_latest_ready_path(destination_checkpoint_path)
        if destination_pointer.exists():
            raise LeagueCheckpointError(
                f"checkpoint snapshot destination already exists: {destination_pointer}"
            )
        if ready["checkpoint_sha256"] != pointer["checkpoint_sha256"]:
            raise LeagueCheckpointError("copied checkpoint hash disagrees with ready marker")
        _atomic_bytes_write(
            destination_pointer,
            _canonical_json_bytes(pointer, name="checkpoint snapshot pointer") + b"\n",
        )
        copied_paths.append(destination_pointer)
    elif resolved.storage_format == "legacy_fixed_file_v1":
        source_receipt = checkpoint_receipt_path(source_checkpoint_path)
        destination_receipt = checkpoint_receipt_path(destination_checkpoint_path)
        for destination in (destination_checkpoint_path, destination_receipt):
            if destination.exists():
                raise LeagueCheckpointError(f"checkpoint snapshot destination already exists: {destination}")
        destination_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_checkpoint_path, destination_checkpoint_path)
        shutil.copy2(source_receipt, destination_receipt)
        copied_paths.extend((destination_checkpoint_path, destination_receipt))
    else:
        raise LeagueCheckpointError(
            f"unsupported checkpoint storage format: {resolved.storage_format}"
        )
    copied = _resolve_checkpoint(destination_checkpoint_path, map_location="cpu")
    if copied.checkpoint_sha256 != resolved.checkpoint_sha256:
        raise LeagueCheckpointError("checkpoint snapshot hash disagrees with source checkpoint")

    return {
        "storage_format": copied.storage_format,
        "generation_id": copied.generation_id,
        "generation_index": copied.generation_index,
        "checkpoint_sha256": copied.checkpoint_sha256,
        "files": {
            receipt["path"]: receipt
            for receipt in (
                _snapshot_file_receipt(path, root=destination_root) for path in copied_paths
            )
        },
    }


def _capture_runtime_state(
    *,
    actor: nn.Module,
    critic: nn.Module,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    environment_state: StateHooks,
    sampler_state: StateHooks,
) -> dict[str, Any]:
    return {
        "actor_state": copy.deepcopy(actor.state_dict()),
        "critic_state": copy.deepcopy(critic.state_dict()),
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "scheduler_state": _component_state(scheduler, name="scheduler"),
        "scaler_state": _component_state(scaler, name="scaler"),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.random.get_rng_state(),
        "torch_cuda_rng_states": (
            [state.detach().cpu().clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
        "environment_state": environment_state.capture(),
        "sampler_state": sampler_state.capture(),
    }


def _apply_runtime_state(
    state: Mapping[str, Any],
    *,
    actor: nn.Module,
    critic: nn.Module,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    environment_state: StateHooks,
    sampler_state: StateHooks,
) -> None:
    actor.load_state_dict(state["actor_state"], strict=True)
    critic.load_state_dict(state["critic_state"], strict=True)
    optimizer.load_state_dict(state["optimizer_state"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler_state"])
    if scaler is not None:
        scaler.load_state_dict(state["scaler_state"])
    environment_state.restore(state["environment_state"])
    sampler_state.restore(state["sampler_state"])
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.random.set_rng_state(state["torch_cpu_rng_state"].cpu())
    cuda_states = state["torch_cuda_rng_states"]
    if cuda_states:
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])


def load_league_checkpoint(
    path: str | Path,
    *,
    actor: nn.Module,
    critic: nn.Module,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    environment_state: StateHooks,
    sampler_state: StateHooks,
    expected_bindings: LeagueCheckpointBindings,
    map_location: str | torch.device = "cpu",
) -> LeagueCheckpointLoadResult:
    """Verify compatibility and integrity, then restore the exact saved state."""

    checkpoint_path = Path(path).resolve()
    resolved = _resolve_checkpoint(
        checkpoint_path,
        map_location=map_location,
    )
    checkpoint = resolved.checkpoint
    checkpoint_sha256 = resolved.checkpoint_sha256
    bindings = LeagueCheckpointBindings.from_dict(checkpoint["bindings"])
    if bindings.to_dict() != expected_bindings.to_dict():
        raise LeagueCheckpointError("league checkpoint bindings are incompatible")
    progress = LeagueTrainingProgress.from_dict(checkpoint["progress"])
    validation_payload = _json_object(
        checkpoint["validation_receipt"],
        name="validation_receipt",
    )
    cuda_states = checkpoint["torch_cuda_rng_states"]
    if not isinstance(cuda_states, list) or any(not isinstance(value, torch.Tensor) for value in cuda_states):
        raise LeagueCheckpointError("checkpoint CUDA RNG state is malformed")
    expected_receipt = resolved.receipt
    _validate_module_state(actor, checkpoint["actor_state"], name="actor")
    _validate_module_state(critic, checkpoint["critic_state"], name="critic")
    _validate_optimizer_state(optimizer, checkpoint["optimizer_state"])
    _validate_optional_component(scheduler, checkpoint["scheduler_state"], name="scheduler")
    _validate_optional_component(scaler, checkpoint["scaler_state"], name="scaler")
    _assert_finite_nested(checkpoint["scheduler_state"], name="checkpoint scheduler state")
    _assert_finite_nested(checkpoint["scaler_state"], name="checkpoint scaler state")
    if cuda_states:
        if not torch.cuda.is_available():
            raise LeagueCheckpointError("checkpoint has CUDA RNG state but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise LeagueCheckpointError("checkpoint CUDA RNG device count is incompatible")
    elif torch.cuda.is_available():
        raise LeagueCheckpointError("CUDA is available but checkpoint has no CUDA RNG state")
    rollback = _capture_runtime_state(
        actor=actor,
        critic=critic,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        environment_state=environment_state,
        sampler_state=sampler_state,
    )
    try:
        _apply_runtime_state(
            checkpoint,
            actor=actor,
            critic=critic,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            environment_state=environment_state,
            sampler_state=sampler_state,
        )
    except Exception as exc:
        try:
            _apply_runtime_state(
                rollback,
                actor=actor,
                critic=critic,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                environment_state=environment_state,
                sampler_state=sampler_state,
            )
        except Exception as rollback_exc:
            raise LeagueCheckpointError(
                "checkpoint restore failed and runtime rollback also failed"
            ) from rollback_exc
        raise LeagueCheckpointError("checkpoint restore failed; runtime state was rolled back") from exc
    restore_receipt = {
        "schema_version": RESTORE_RECEIPT_SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_sha256,
        "environment_decisions": progress.environment_decisions,
        "update_count": progress.update_count,
        "cumulative_training_seconds": progress.cumulative_training_seconds,
        "counters": dict(progress.counters),
        "bindings_sha256": expected_receipt["bindings_sha256"],
        "checkpoint_hash_verified": True,
        "compatibility_verified": True,
        "rng_restored": True,
        "environment_state_restored": True,
        "sampler_state_restored": True,
        "checkpoint_storage_format": resolved.storage_format,
        "checkpoint_generation_id": resolved.generation_id,
        "checkpoint_generation_index": resolved.generation_index,
    }

    return LeagueCheckpointLoadResult(
        progress=progress,
        validation_receipt=validation_payload,
        checkpoint_sha256=checkpoint_sha256,
        restore_receipt=restore_receipt,
    )


def prove_resume_counter_advance(
    *,
    restored: LeagueCheckpointLoadResult,
    current: LeagueTrainingProgress,
) -> dict[str, Any]:
    """Prove that resumed work advanced beyond the exact restored cursor."""

    normalized_checkpoint_sha256 = _normalized_sha256(
        restored.checkpoint_sha256,
        name="checkpoint_sha256",
    )
    restored_progress = restored.progress
    if (
        restored.restore_receipt.get("checkpoint_sha256") != normalized_checkpoint_sha256
        or restored.restore_receipt.get("environment_decisions")
        != restored_progress.environment_decisions
        or restored.restore_receipt.get("update_count") != restored_progress.update_count
        or restored.restore_receipt.get("cumulative_training_seconds")
        != restored_progress.cumulative_training_seconds
        or restored.restore_receipt.get("counters") != dict(restored_progress.counters)
    ):
        raise LeagueCheckpointError("restore receipt does not match the restored cursor")
    if current.environment_decisions <= restored_progress.environment_decisions:
        raise LeagueCheckpointError("resumed environment decision count did not advance")
    if current.update_count < restored_progress.update_count:
        raise LeagueCheckpointError("resumed update count moved backward")
    if current.cumulative_training_seconds < restored_progress.cumulative_training_seconds:
        raise LeagueCheckpointError("resumed cumulative training seconds moved backward")
    counter_names = sorted(set(restored_progress.counters) | set(current.counters))
    counter_deltas: dict[str, int] = {}
    for name in counter_names:
        before = restored_progress.counters.get(name, 0)
        after = current.counters.get(name, 0)
        if after < before:
            raise LeagueCheckpointError(f"resumed counter {name} moved backward")
        counter_deltas[name] = after - before

    return {
        "schema_version": RESUME_PROOF_SCHEMA_VERSION,
        "checkpoint_sha256": normalized_checkpoint_sha256,
        "restored": restored_progress.to_dict(),
        "current": current.to_dict(),
        "delta": {
            "environment_decisions": (
                current.environment_decisions - restored_progress.environment_decisions
            ),
            "update_count": current.update_count - restored_progress.update_count,
            "cumulative_training_seconds": (
                current.cumulative_training_seconds
                - restored_progress.cumulative_training_seconds
            ),
            "counters": counter_deltas,
        },
        "checks": {
            "restored_cursor_exact": True,
            "environment_decisions_advanced": True,
            "update_count_not_decreased": True,
            "cumulative_training_seconds_not_decreased": True,
            "counters_not_decreased": True,
        },
    }
