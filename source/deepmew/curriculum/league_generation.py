"""Durable research-only state for one closed-loop league generation.

This module executes no games or training. It verifies artifacts produced by
existing tools and advances an atomic state receipt only when each prerequisite
contract is complete, identity-consistent, and safety-clean.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from deepmew.curriculum.awr_train_receipt import validate_awr_train_receipt
from deepmew.curriculum.league_registry import (
    LeagueRegistryConflictError,
    LeagueRole,
    load_registry,
    sha256_path,
)
from deepmew.curriculum.league_schedule import SCHEDULE_VERSION


GENERATION_SCHEMA_VERSION = "deepmew.league_generation.v1"
DATAGEN_SCHEMA_VERSION = "deepmew.r19.league_datagen.v2"
TRAIN_RECEIPT_SCHEMA_VERSION = "deepmew.awr_train_receipt.v1"
SHARED_PANEL_SCHEMA_VERSION = "deepmew.league_shared_panel.v2"
PAYOFF_SCHEMA_VERSION = "deepmew.league_payoff_matrix.v2"
PUBLIC_INFORMATION_MODE = "public_belief"


def _require_sha256(value: Any, *, field: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text.lower()):
        raise ValueError(f"{field} must be a 64-character SHA-256")

    return text.lower()


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")

    return value


def _require_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")

    return value.strip()


def _require_nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be an explicit non-negative integer")

    return value


def derive_evaluation_panel_id(
    *,
    generation_id: str,
    snapshot_id: str,
    shared_results_sha256: str,
    payoff_results_sha256: str,
    shared_config_fingerprint: str,
    payoff_config_fingerprint: str,
) -> str:
    payload = {
        "generation_id": generation_id,
        "snapshot_id": snapshot_id,
        "shared_results_sha256": _require_sha256(
            shared_results_sha256,
            field="shared_results_sha256",
        ),
        "payoff_results_sha256": _require_sha256(
            payoff_results_sha256,
            field="payoff_results_sha256",
        ),
        "shared_config_fingerprint": _require_sha256(
            shared_config_fingerprint,
            field="shared_config_fingerprint",
        ),
        "payoff_config_fingerprint": _require_sha256(
            payoff_config_fingerprint,
            field="payoff_config_fingerprint",
        ),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return f"lgate1-{digest}"


class GenerationPhase(str, Enum):
    INITIALIZED = "initialized"
    DATA_READY = "data_ready"
    CANDIDATE_READY = "candidate_ready"
    EVALUATION_READY = "evaluation_ready"
    DECISION_RECORDED = "decision_recorded"


_NEXT_PHASE = {
    GenerationPhase.INITIALIZED: GenerationPhase.DATA_READY,
    GenerationPhase.DATA_READY: GenerationPhase.CANDIDATE_READY,
    GenerationPhase.CANDIDATE_READY: GenerationPhase.EVALUATION_READY,
    GenerationPhase.EVALUATION_READY: GenerationPhase.DECISION_RECORDED,
}


@dataclass(frozen=True)
class GenerationArtifactV1:
    path: str
    sha256: str
    schema_version: str
    status: str

    def to_json(self) -> dict[str, str]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "schema_version": self.schema_version,
            "status": self.status,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "GenerationArtifactV1":
        return cls(
            path=str(data["path"]),
            sha256=str(data["sha256"]),
            schema_version=str(data.get("schema_version", "")),
            status=str(data.get("status", "")),
        )


@dataclass(frozen=True)
class LeagueGenerationV1:
    generation_id: str
    revision: int
    phase: GenerationPhase
    registry_path: str
    registry_sha256_at_init: str
    snapshot_id: str
    champion_member_id: str
    champion_artifact_sha256: str
    champion_deck_sha256: str
    feature_policy_id: str
    candidate_member_id: str | None
    candidate_deck_sha256: str | None
    evaluation_panel_id: str | None
    artifacts: Mapping[str, GenerationArtifactV1]
    state_fingerprint: str
    schema_version: str = GENERATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase", GenerationPhase(self.phase))
        if self.schema_version != GENERATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported generation schema {self.schema_version!r}")
        if not self.generation_id or not self.snapshot_id or not self.champion_member_id:
            raise ValueError("generation, snapshot, and champion IDs must be non-empty")
        if not self.feature_policy_id:
            raise ValueError("feature_policy_id must be non-empty")
        if self.revision < 0:
            raise ValueError("generation revision must be non-negative")
        _require_sha256(self.champion_artifact_sha256, field="champion_artifact_sha256")
        _require_sha256(self.champion_deck_sha256, field="champion_deck_sha256")
        candidate_required = self.phase in {
            GenerationPhase.CANDIDATE_READY,
            GenerationPhase.EVALUATION_READY,
            GenerationPhase.DECISION_RECORDED,
        }
        evaluation_required = self.phase in {
            GenerationPhase.EVALUATION_READY,
            GenerationPhase.DECISION_RECORDED,
        }
        if candidate_required:
            if not self.candidate_member_id or not self.candidate_deck_sha256:
                raise ValueError(f"phase {self.phase.value} requires explicit candidate identity")
            _require_sha256(self.candidate_deck_sha256, field="candidate_deck_sha256")
        elif self.candidate_member_id is not None or self.candidate_deck_sha256 is not None:
            raise ValueError(f"phase {self.phase.value} must not carry a candidate identity")
        if evaluation_required and not self.evaluation_panel_id:
            raise ValueError(f"phase {self.phase.value} requires evaluation_panel_id")
        if not evaluation_required and self.evaluation_panel_id is not None:
            raise ValueError(f"phase {self.phase.value} must not carry evaluation_panel_id")
        expected = generation_fingerprint(self.identity_payload())
        if self.state_fingerprint != expected:
            raise ValueError(
                f"generation state fingerprint mismatch: {self.state_fingerprint} != {expected}"
            )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generation_id": self.generation_id,
            "revision": self.revision,
            "phase": self.phase.value,
            "registry_path": self.registry_path,
            "registry_sha256_at_init": self.registry_sha256_at_init,
            "snapshot_id": self.snapshot_id,
            "champion_member_id": self.champion_member_id,
            "champion_artifact_sha256": self.champion_artifact_sha256,
            "champion_deck_sha256": self.champion_deck_sha256,
            "feature_policy_id": self.feature_policy_id,
            "candidate_member_id": self.candidate_member_id,
            "candidate_deck_sha256": self.candidate_deck_sha256,
            "evaluation_panel_id": self.evaluation_panel_id,
            "artifacts": {
                name: artifact.to_json()
                for name, artifact in sorted(self.artifacts.items())
            },
        }

    def to_json(self) -> dict[str, Any]:
        return {**self.identity_payload(), "state_fingerprint": self.state_fingerprint}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "LeagueGenerationV1":
        return cls(
            generation_id=str(data["generation_id"]),
            revision=int(data["revision"]),
            phase=GenerationPhase(str(data["phase"])),
            registry_path=str(data["registry_path"]),
            registry_sha256_at_init=str(data["registry_sha256_at_init"]),
            snapshot_id=str(data["snapshot_id"]),
            champion_member_id=str(data["champion_member_id"]),
            champion_artifact_sha256=str(data["champion_artifact_sha256"]),
            champion_deck_sha256=str(data["champion_deck_sha256"]),
            feature_policy_id=str(data["feature_policy_id"]),
            candidate_member_id=(
                str(data["candidate_member_id"])
                if data.get("candidate_member_id") is not None else None
            ),
            candidate_deck_sha256=(
                str(data["candidate_deck_sha256"])
                if data.get("candidate_deck_sha256") is not None else None
            ),
            evaluation_panel_id=(
                str(data["evaluation_panel_id"])
                if data.get("evaluation_panel_id") is not None else None
            ),
            artifacts={
                str(name): GenerationArtifactV1.from_json(artifact)
                for name, artifact in (data.get("artifacts") or {}).items()
            },
            state_fingerprint=str(data["state_fingerprint"]),
            schema_version=str(data.get("schema_version", "")),
        )


def generation_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def _with_fingerprint(payload: dict[str, Any]) -> LeagueGenerationV1:
    fingerprint = generation_fingerprint(payload)

    return LeagueGenerationV1.from_json({**payload, "state_fingerprint": fingerprint})


def initialize_generation(
    *,
    generation_id: str,
    registry_path: str | Path,
    snapshot_id: str,
    champion_member_id: str,
    feature_policy_id: str,
) -> LeagueGenerationV1:
    registry_file = Path(registry_path).resolve()
    registry = load_registry(registry_file)
    champion = registry.active_champion()
    if champion is None or champion.league_member_id != champion_member_id:
        actual = champion.league_member_id if champion is not None else None
        raise LeagueRegistryConflictError(
            f"generation champion mismatch: expected {champion_member_id!r}, found {actual!r}"
        )
    if champion.feature_policy_id != feature_policy_id:
        raise LeagueRegistryConflictError(
            f"generation feature-policy mismatch: {feature_policy_id!r} != "
            f"{champion.feature_policy_id!r}"
        )
    snapshot = next(
        (item for item in registry.snapshots if item.snapshot_id == snapshot_id),
        None,
    )
    if snapshot is None:
        raise ValueError(f"unknown generation snapshot {snapshot_id!r}")
    frozen_champion = next(
        (item for item in snapshot.members if item.league_member_id == champion_member_id),
        None,
    )
    if frozen_champion is None or frozen_champion.role is not LeagueRole.CHAMPION:
        raise ValueError("snapshot does not freeze the requested member as champion")
    if (
        frozen_champion.artifact_sha256 != champion.artifact_sha256
        or frozen_champion.deck_sha256 != champion.deck_sha256
        or frozen_champion.feature_policy_id != champion.feature_policy_id
    ):
        raise ValueError("snapshot champion immutable identity does not match live registry member")
    payload = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "generation_id": generation_id,
        "revision": 0,
        "phase": GenerationPhase.INITIALIZED.value,
        "registry_path": str(registry_file),
        "registry_sha256_at_init": sha256_path(registry_file),
        "snapshot_id": snapshot_id,
        "champion_member_id": champion_member_id,
        "champion_artifact_sha256": champion.artifact_sha256,
        "champion_deck_sha256": champion.deck_sha256,
        "feature_policy_id": feature_policy_id,
        "candidate_member_id": None,
        "candidate_deck_sha256": None,
        "evaluation_panel_id": None,
        "artifacts": {},
    }

    return _with_fingerprint(payload)


def _read_json_artifact(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"generation artifact must contain a JSON object: {resolved}")

    return resolved, payload


def _strict_safety_totals(payload: Mapping[str, Any], *, field: str) -> Mapping[str, Any]:
    totals = _require_mapping(payload.get(field), field=field)
    for key in ("crashes", "timeouts", "illegal", "incomplete"):
        if _require_nonnegative_int(totals.get(key), field=f"{field}.{key}") != 0:
            raise ValueError(f"{field}.{key} is not zero")

    return totals


def _artifact_already_recorded(
    state: LeagueGenerationV1,
    *,
    name: str,
    artifact: GenerationArtifactV1,
) -> bool:
    existing = state.artifacts.get(name)
    if existing is None:
        return False
    if existing != artifact:
        raise ValueError(f"generation already binds {name} to a different immutable receipt")

    return True


def _advance(
    state: LeagueGenerationV1,
    phase: GenerationPhase,
    name: str,
    artifact: GenerationArtifactV1,
    *,
    state_updates: Mapping[str, Any] | None = None,
) -> LeagueGenerationV1:
    expected = _NEXT_PHASE.get(state.phase)
    if expected is not phase:
        raise ValueError(f"cannot advance generation {state.phase.value} -> {phase.value}")
    artifacts = dict(state.artifacts)
    artifacts[name] = artifact
    payload = state.identity_payload()
    payload.update({
        "revision": state.revision + 1,
        "phase": phase.value,
        "artifacts": {
            key: value.to_json()
            for key, value in sorted(artifacts.items())
        },
    })
    payload.update(state_updates or {})

    return _with_fingerprint(payload)


def mark_data_ready(state: LeagueGenerationV1, manifest_path: str | Path) -> LeagueGenerationV1:
    path, manifest = _read_json_artifact(manifest_path)
    if manifest.get("schema_version") != DATAGEN_SCHEMA_VERSION:
        raise ValueError(f"datagen schema must be {DATAGEN_SCHEMA_VERSION!r}")
    if manifest.get("status") != "completed":
        raise ValueError("datagen manifest status must be completed")
    if manifest.get("schedule_version") != SCHEDULE_VERSION:
        raise ValueError(
            f"datagen schedule must be {SCHEDULE_VERSION!r}, got {manifest.get('schedule_version')!r}"
        )
    generation = _require_mapping(manifest.get("generation_binding"), field="generation_binding")
    expected_generation = {
        "generation_id": state.generation_id,
        "snapshot_id": state.snapshot_id,
        "champion_member_id": state.champion_member_id,
        "champion_artifact_sha256": state.champion_artifact_sha256,
        "champion_deck_sha256": state.champion_deck_sha256,
        "feature_policy_id": state.feature_policy_id,
        "registry_sha256_at_start": state.registry_sha256_at_init,
    }
    for key, expected in expected_generation.items():
        if generation.get(key) != expected:
            raise ValueError(f"datagen generation binding mismatch for {key}")
    if generation.get("verified") is not True:
        raise ValueError("datagen generation binding was not registry-verified")
    registry_path = _require_string(generation.get("registry_path"), field="generation_binding.registry_path")
    if Path(registry_path).resolve() != Path(state.registry_path).resolve():
        raise ValueError("datagen registry path does not match generation")
    _require_sha256(manifest.get("schedule_fingerprint"), field="schedule_fingerprint")
    shard_receipts = manifest.get("shard_receipts")
    if not isinstance(shard_receipts, list) or not shard_receipts:
        raise ValueError("datagen manifest requires non-empty shard_receipts")
    for index, receipt_value in enumerate(shard_receipts):
        receipt = _require_mapping(receipt_value, field=f"shard_receipts[{index}]")
        _require_string(receipt.get("path"), field=f"shard_receipts[{index}].path")
        _require_sha256(receipt.get("sha256"), field=f"shard_receipts[{index}].sha256")
        if _require_nonnegative_int(receipt.get("rows"), field=f"shard_receipts[{index}].rows") <= 0:
            raise ValueError("every datagen shard receipt must contain rows")
    our = _require_mapping(manifest.get("our"), field="our")
    if our.get("league_member_id") != state.champion_member_id:
        raise ValueError("datagen champion member ID does not match generation")
    policy_identity = _require_mapping(our.get("policy_identity"), field="our.policy_identity")
    if policy_identity.get("artifact_sha256") != state.champion_artifact_sha256:
        raise ValueError("datagen policy artifact does not match generation champion")
    deck_binding = _require_mapping(our.get("deck_binding"), field="our.deck_binding")
    if deck_binding.get("sha256") != state.champion_deck_sha256:
        raise ValueError("datagen deck artifact does not match generation champion")
    binding = _require_mapping(our.get("feature_binding"), field="our.feature_binding")
    if binding.get("feature_policy_id") != state.feature_policy_id:
        raise ValueError("datagen feature-policy binding does not match generation")
    totals = _strict_safety_totals(manifest, field="totals")
    games = _require_nonnegative_int(totals.get("games"), field="totals.games")
    completed = _require_nonnegative_int(totals.get("completed"), field="totals.completed")
    rows = _require_nonnegative_int(totals.get("rows"), field="totals.rows")
    if games <= 0 or rows <= 0 or completed != games:
        raise ValueError("datagen requires positive rows/games and completed == games")
    identity = _require_mapping(manifest.get("identity"), field="identity")
    if _require_nonnegative_int(
        identity.get("duplicate_decision_ids"),
        field="identity.duplicate_decision_ids",
    ) != 0:
        raise ValueError("datagen decision identity audit did not pass")
    if _require_nonnegative_int(identity.get("unique_game_ids"), field="identity.unique_game_ids") != games:
        raise ValueError("datagen unique game count does not match totals.games")
    if _require_nonnegative_int(
        identity.get("unique_decision_ids"),
        field="identity.unique_decision_ids",
    ) != rows:
        raise ValueError("datagen unique decision count does not match totals.rows")
    artifact = GenerationArtifactV1(
        path=str(path),
        sha256=sha256_path(path),
        schema_version=str(manifest.get("schema_version", "")),
        status="completed_safety_clean_identity_complete",
    )
    if _artifact_already_recorded(state, name="data_manifest", artifact=artifact):
        return state

    return _advance(state, GenerationPhase.DATA_READY, "data_manifest", artifact)


def mark_candidate_ready(
    state: LeagueGenerationV1,
    *,
    candidate_member_id: str,
    checkpoint_path: str | Path,
    train_manifest_path: str | Path,
) -> LeagueGenerationV1:
    checkpoint = Path(checkpoint_path).resolve(strict=True)
    train_path, train = _read_json_artifact(train_manifest_path)
    if train.get("schema_version") != TRAIN_RECEIPT_SCHEMA_VERSION:
        raise ValueError(f"training receipt schema must be {TRAIN_RECEIPT_SCHEMA_VERSION!r}")
    if train.get("status") != "completed":
        raise ValueError("training receipt status must be completed")
    validate_awr_train_receipt(train)
    if train.get("generation_id") != state.generation_id:
        raise ValueError("training receipt generation_id does not match generation")
    if train.get("snapshot_id") != state.snapshot_id:
        raise ValueError("training receipt snapshot_id does not match generation")
    if train.get("candidate_member_id") != candidate_member_id:
        raise ValueError("training receipt candidate_member_id does not match request")
    if train.get("feature_policy_id") != state.feature_policy_id:
        raise ValueError("training receipt feature_policy_id does not match generation")
    if train.get("information_mode") != PUBLIC_INFORMATION_MODE:
        raise ValueError("training receipt must bind public_belief information_mode")
    if train.get("chosen_advantage_mode") != "off":
        raise ValueError("generation training requires chosen_advantage_mode=off")
    gates = _require_mapping(train.get("gates"), field="gates")
    for gate in ("leakage_passed", "runtime_contamination_passed", "safety_clean"):
        if gates.get(gate) is not True:
            raise ValueError(f"training gate {gate} did not pass")
    training = _require_mapping(train.get("training"), field="training")
    if _require_nonnegative_int(training.get("samples"), field="training.samples") <= 0:
        raise ValueError("training receipt requires positive samples")
    if _require_nonnegative_int(training.get("batches"), field="training.batches") <= 0:
        raise ValueError("training receipt requires positive batches")
    metrics = _require_mapping(train.get("metrics"), field="metrics")
    if not metrics:
        raise ValueError("training receipt requires finite metrics")
    for name, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"training metric {name!r} must be finite")
    receipts = _require_mapping(train.get("receipts"), field="receipts")
    _require_sha256(receipts.get("trainer_code_sha256"), field="receipts.trainer_code_sha256")
    _require_sha256(receipts.get("config_sha256"), field="receipts.config_sha256")
    _require_nonnegative_int(receipts.get("seed"), field="receipts.seed")
    registry = load_registry(state.registry_path)
    candidate = registry.member(candidate_member_id)
    if candidate is None:
        raise ValueError("candidate-ready requires a registered candidate member")
    is_retry = state.candidate_member_id == candidate_member_id
    if candidate.role is not LeagueRole.CANDIDATE and not is_retry:
        raise ValueError("candidate-ready requires a registered candidate member")
    if candidate.parent_member_id != state.champion_member_id:
        raise ValueError("candidate parent does not match generation champion")
    if candidate.feature_policy_id != state.feature_policy_id:
        raise ValueError("candidate feature-policy identity does not match generation")
    if candidate.deck_sha256 != state.champion_deck_sha256:
        raise ValueError("candidate deck identity does not match generation champion deck")
    checkpoint_sha = sha256_path(checkpoint)
    if checkpoint_sha != candidate.artifact_sha256:
        raise ValueError("candidate checkpoint hash does not match registry")
    data_receipt = state.artifacts.get("data_manifest")
    if data_receipt is None:
        raise ValueError("candidate-ready requires a data manifest receipt")
    if train.get("input_data_manifest_sha256") != data_receipt.sha256:
        raise ValueError("training manifest does not bind the generation data manifest")
    if train.get("init_checkpoint_sha256") != state.champion_artifact_sha256:
        raise ValueError("training manifest does not bind the generation champion")
    if train.get("output_checkpoint_sha256") != checkpoint_sha:
        raise ValueError("training manifest output hash does not match candidate")
    if train.get("candidate_deck_sha256") != candidate.deck_sha256:
        raise ValueError("training manifest candidate deck does not match registry")
    train_artifact = GenerationArtifactV1(
        path=str(train_path),
        sha256=sha256_path(train_path),
        schema_version=TRAIN_RECEIPT_SCHEMA_VERSION,
        status="completed_public_safety_clean",
    )
    checkpoint_artifact = GenerationArtifactV1(
        path=str(checkpoint),
        sha256=checkpoint_sha,
        schema_version="binary-checkpoint",
        status="registered_candidate",
    )
    train_recorded = _artifact_already_recorded(
        state,
        name="train_manifest",
        artifact=train_artifact,
    )
    checkpoint_recorded = _artifact_already_recorded(
        state,
        name="candidate_checkpoint",
        artifact=checkpoint_artifact,
    )
    if train_recorded or checkpoint_recorded:
        if not (train_recorded and checkpoint_recorded and is_retry):
            raise ValueError("generation contains a partial or conflicting candidate receipt")
        if state.candidate_deck_sha256 != candidate.deck_sha256:
            raise ValueError("generation candidate deck identity changed")

        return state
    advanced = _advance(
        state,
        GenerationPhase.CANDIDATE_READY,
        "train_manifest",
        train_artifact,
        state_updates={
            "candidate_member_id": candidate_member_id,
            "candidate_deck_sha256": candidate.deck_sha256,
        },
    )
    payload = advanced.identity_payload()
    artifacts = dict(advanced.artifacts)
    artifacts["candidate_checkpoint"] = checkpoint_artifact
    payload["artifacts"] = {
        key: value.to_json()
        for key, value in sorted(artifacts.items())
    }

    return _with_fingerprint(payload)


def mark_evaluation_ready(
    state: LeagueGenerationV1,
    *,
    shared_panel_path: str | Path,
    payoff_path: str | Path,
) -> LeagueGenerationV1:
    shared_path, shared = _read_json_artifact(shared_panel_path)
    payoff_file, payoff = _read_json_artifact(payoff_path)
    if shared.get("schema_version") != SHARED_PANEL_SCHEMA_VERSION:
        raise ValueError(f"shared-panel schema must be {SHARED_PANEL_SCHEMA_VERSION!r}")
    if payoff.get("schema_version") != PAYOFF_SCHEMA_VERSION:
        raise ValueError(f"payoff schema must be {PAYOFF_SCHEMA_VERSION!r}")
    for name, payload in (("shared panel", shared), ("payoff", payoff)):
        if payload.get("status") != "completed" or payload.get("safety_clean") is not True:
            raise ValueError(f"{name} is not completed and safety-clean")
    if shared.get("comparable") is not True:
        raise ValueError("shared panel is not comparable")
    if payoff.get("analysis_ready") is not True:
        raise ValueError("payoff matrix is not analysis-ready")
    candidate_receipt = state.artifacts.get("candidate_checkpoint")
    if candidate_receipt is None or state.candidate_member_id is None or state.candidate_deck_sha256 is None:
        raise ValueError("evaluation-ready requires a candidate checkpoint")
    expected_identities = {
        "generation_id": state.generation_id,
        "snapshot_id": state.snapshot_id,
        "champion_member_id": state.champion_member_id,
        "candidate_member_id": state.candidate_member_id,
        "champion_checkpoint_sha256": state.champion_artifact_sha256,
        "candidate_checkpoint_sha256": candidate_receipt.sha256,
        "champion_deck_sha256": state.champion_deck_sha256,
        "candidate_deck_sha256": state.candidate_deck_sha256,
        "feature_policy_id": state.feature_policy_id,
    }
    for name, payload in (("shared panel", shared), ("payoff", payoff)):
        identities = _require_mapping(payload.get("member_identities"), field=f"{name}.member_identities")
        for key, expected in expected_identities.items():
            if identities.get(key) != expected:
                raise ValueError(f"{name} identity mismatch for {key}")
    shared_schedule = _require_mapping(shared.get("schedule"), field="shared panel.schedule")
    if shared_schedule.get("complete") is not True:
        raise ValueError("shared panel schedule is incomplete")
    shared_jobs = _require_nonnegative_int(
        shared_schedule.get("jobs_expected"),
        field="shared panel.schedule.jobs_expected",
    )
    if shared_jobs <= 0 or _require_nonnegative_int(
        shared_schedule.get("jobs_completed"),
        field="shared panel.schedule.jobs_completed",
    ) != shared_jobs:
        raise ValueError("shared panel job budget is incomplete")
    candidate_comparisons = [
        comparison
        for comparison in shared.get("comparisons") or []
        if isinstance(comparison, Mapping)
        and comparison.get("candidate_member_id") == state.candidate_member_id
    ]
    if len(candidate_comparisons) != 1:
        raise ValueError("shared panel must contain exactly one comparison for the generation candidate")
    if candidate_comparisons[0].get("equal_completed_budget") is not True:
        raise ValueError("generation candidate shared-panel budget is not equal and complete")
    payoff_schedule = _require_mapping(payoff.get("schedule"), field="payoff.schedule")
    if payoff_schedule.get("complete") is not True or payoff_schedule.get("all_pair_budgets_complete") is not True:
        raise ValueError("payoff schedule is incomplete")
    direct_rows = [
        row
        for row in (_require_mapping(payoff.get("matrix"), field="payoff.matrix").get("rows") or [])
        if isinstance(row, Mapping)
        and {
            row.get("row_member_id"),
            row.get("column_member_id"),
        } == {state.champion_member_id, state.candidate_member_id}
        and row.get("budget_complete") is True
        and _require_nonnegative_int(row.get("attempted"), field="payoff.matrix.rows[].attempted") > 0
    ]
    direct_directions = {
        (row.get("row_member_id"), row.get("column_member_id"))
        for row in direct_rows
    }
    if direct_directions != {
        (state.champion_member_id, state.candidate_member_id),
        (state.candidate_member_id, state.champion_member_id),
    }:
        raise ValueError("payoff matrix lacks both direct champion/candidate directions")
    shared_sha = sha256_path(shared_path)
    payoff_sha = sha256_path(payoff_file)
    shared_fingerprint = _require_sha256(
        shared.get("config_fingerprint"),
        field="shared_panel.config_fingerprint",
    )
    payoff_fingerprint = _require_sha256(
        payoff.get("config_fingerprint"),
        field="payoff.config_fingerprint",
    )
    evaluation_panel_id = derive_evaluation_panel_id(
        generation_id=state.generation_id,
        snapshot_id=state.snapshot_id,
        shared_results_sha256=shared_sha,
        payoff_results_sha256=payoff_sha,
        shared_config_fingerprint=shared_fingerprint,
        payoff_config_fingerprint=payoff_fingerprint,
    )
    shared_artifact = GenerationArtifactV1(
        path=str(shared_path),
        sha256=shared_sha,
        schema_version=SHARED_PANEL_SCHEMA_VERSION,
        status="completed_comparable_safety_clean",
    )
    payoff_artifact = GenerationArtifactV1(
        path=str(payoff_file),
        sha256=payoff_sha,
        schema_version=PAYOFF_SCHEMA_VERSION,
        status="analysis_ready_safety_clean",
    )
    shared_recorded = _artifact_already_recorded(
        state,
        name="shared_panel_results",
        artifact=shared_artifact,
    )
    payoff_recorded = _artifact_already_recorded(
        state,
        name="payoff_results",
        artifact=payoff_artifact,
    )
    if shared_recorded or payoff_recorded:
        if not (shared_recorded and payoff_recorded and state.evaluation_panel_id == evaluation_panel_id):
            raise ValueError("generation contains a partial or conflicting evaluation receipt")

        return state
    advanced = _advance(
        state,
        GenerationPhase.EVALUATION_READY,
        "shared_panel_results",
        shared_artifact,
        state_updates={"evaluation_panel_id": evaluation_panel_id},
    )
    payload = advanced.identity_payload()
    artifacts = dict(advanced.artifacts)
    artifacts["payoff_results"] = payoff_artifact
    payload["artifacts"] = {
        key: value.to_json()
        for key, value in sorted(artifacts.items())
    }

    return _with_fingerprint(payload)


def mark_decision_recorded(
    state: LeagueGenerationV1,
    *,
    promotion_id: str,
) -> LeagueGenerationV1:
    registry = load_registry(state.registry_path)
    record = registry.promotion_record(promotion_id)
    if record is None:
        raise ValueError(f"unknown promotion record {promotion_id!r}")
    candidate = state.artifacts.get("candidate_checkpoint")
    if candidate is None or state.candidate_member_id is None or state.candidate_deck_sha256 is None:
        raise ValueError("decision-ready requires explicit candidate identity")
    if record.candidate_member_id != state.candidate_member_id:
        raise ValueError("promotion record candidate member does not match generation")
    if record.candidate_artifact_sha256 != candidate.sha256:
        raise ValueError("promotion record candidate does not match generation")
    if record.candidate_deck_sha256 != state.candidate_deck_sha256:
        raise ValueError("promotion record candidate deck does not match generation")
    if record.incumbent_member_id != state.champion_member_id:
        raise ValueError("promotion record incumbent does not match generation")
    if record.evaluation_panel_id != state.evaluation_panel_id:
        raise ValueError("promotion record evaluation panel does not match generation receipts")
    evaluation = state.artifacts.get("shared_panel_results")
    payoff = state.artifacts.get("payoff_results")
    if evaluation is None or payoff is None:
        raise ValueError("decision-ready requires both evaluation receipts")
    active_champion = registry.active_champion()
    if record.accepted:
        if record.snapshot_id is None:
            raise ValueError("accepted promotion record requires a snapshot")
        snapshot = next(
            (item for item in registry.snapshots if item.snapshot_id == record.snapshot_id),
            None,
        )
        if snapshot is None:
            raise ValueError("accepted promotion snapshot is missing from registry")
        frozen_candidate = next(
            (item for item in snapshot.members if item.league_member_id == state.candidate_member_id),
            None,
        )
        if frozen_candidate is None or frozen_candidate.role is not LeagueRole.CHAMPION:
            raise ValueError("accepted promotion snapshot does not freeze the candidate as champion")
        if active_champion is None or active_champion.league_member_id != state.candidate_member_id:
            raise ValueError("accepted promotion did not make the candidate active champion")
    else:
        if record.snapshot_id is not None:
            raise ValueError("rejected promotion record must not create a snapshot")
        if active_champion is None or active_champion.league_member_id != state.champion_member_id:
            raise ValueError("rejected promotion did not retain the generation champion")
    artifact = GenerationArtifactV1(
        path=str(Path(state.registry_path).resolve()),
        sha256=sha256_path(state.registry_path),
        schema_version="deepmew.league_registry.v1",
        status=f"promotion:{promotion_id}:accepted={record.accepted}",
    )
    if _artifact_already_recorded(state, name="promotion_record", artifact=artifact):
        return state

    return _advance(
        state,
        GenerationPhase.DECISION_RECORDED,
        "promotion_record",
        artifact,
    )


def load_generation(path: str | Path) -> LeagueGenerationV1:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    return LeagueGenerationV1.from_json(payload)


def save_generation_atomic(
    path: str | Path,
    state: LeagueGenerationV1,
    *,
    expected_revision: int | None,
) -> None:
    target = Path(path).resolve()
    if target.exists():
        current = load_generation(target)
        if current.state_fingerprint == state.state_fingerprint:
            if expected_revision is not None and current.revision != expected_revision:
                raise ValueError(
                    f"generation revision conflict: expected {expected_revision}, found {current.revision}"
                )

            return
        if expected_revision is None or current.revision != expected_revision:
            raise ValueError(
                f"generation revision conflict: expected {expected_revision}, found {current.revision}"
            )
    elif expected_revision is not None:
        raise ValueError("generation state does not exist for the expected revision")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(state.to_json(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
