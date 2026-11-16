"""Run the receipt-bound frozen E1 stage probe through the real game engine.

This adapter is research-only.  It preserves the literal frozen probe tuple
while resolving the concrete stage, opponent, and deck artifacts only after
the exact E1 authorization receipts and schemas have been verified.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import run_terminal_league_ppo as terminal
import torch
from deepmew.replay.contract.preflight import require_accepted_canonical_decision_receipt
from deepmew.replay.canonical_actor_dataset import canonical_actor_feature_observation
from deepmew.replay.information_boundary import (
    InformationMode,
    assert_deployable_observation,
    classify_observation_visibility,
)
from deepmew.rl.algorithms.r4f_ppo import R4FNativePPO
from deepmew.rl.deck_identity import (
    normalized_deck_hash,
    normalized_deck_payload,
    r4f_feature_deck_hash,
)
from deepmew.rl.enhanced_stage_probe import (
    ProbeGameResult,
    StageProbeConfig,
    run_fixed_stage_probe,
)
from deepmew.rl.enhanced_trainer import (
    CHECKPOINT_SCHEMA_VERSION,
    STAGE_BOUNDARY_ORDER,
    STAGE_CHECKPOINT_RECEIPT_SCHEMA_VERSION,
    ZERO_SAFETY,
)
from deepmew.rl.opponents import (
    POLICY_DECK_BINDINGS,
    OpponentSampler,
    R7_CHECKPOINT_SHA256,
    resolve_policy_deck_binding,
)
from deepmew.rl.safety import (
    FORBIDDEN_OBSERVATION_INPUTS,
    FORBIDDEN_OFFLINE_LABEL_MODULES,
)


MISSION = "E1_STARMIE_COMPLETE_ENHANCED_GENERATION"
WRAPPER_SCHEMA_VERSION = "deepmew.e1_authorized_stage_probe.v1"
ATTEMPT_SCHEMA_VERSION = "deepmew.e1_authorized_stage_probe_attempt.v1"
FAILURE_SCHEMA_VERSION = "deepmew.e1_authorized_stage_probe_failure.v1"
RESOLUTION_SCHEMA_VERSION = "deepmew.e1_stage_probe_resolution.v1"
ATTEMPT_AUTHORIZATION_SCHEMA_VERSION = (
    "deepmew.e1_stage_probe_resolution_authorization.v1"
)
AUTHORIZED_FROZEN_CONFIG_FILE_SHA256 = (
    "b8a9afc5d3767a569a66d9c6ab303631118e84650d3b6d72a1eb56b7fe2435b4"
)
AUTHORIZED_LAUNCH_RECEIPT_FILE_SHA256 = (
    "69eea3a0632848c6ae7225aa9ac73342a73078cbae4ca0523f76492dfd5961a1"
)
AUTHORIZED_CORRECTION_SOURCE_SHA256 = (
    "08ad2b641831571594caf4cf9d876713d963ad511cdadb9ca68d8c01ba7c7d94"
)
AUTHORIZED_DIAGNOSTIC_RECEIPT_FILE_SHA256 = (
    "4a67bc0ce4b16024ea8e43d9358e94be8a01bf2cde12b3d49cb3ab928c056239"
)
EXPECTED_FIRST_PROJECTED_TENSOR_SHAPE = (2, 304)
EXPECTED_FIRST_PROJECTED_TENSOR_SHA256 = (
    "afd42e7ae62ae32d5b903ec171acb9d860f9a6a125e6b459472646ff5e78dbf8"
)
AUTHORIZED_PROBE_CONFIG_SHA256 = (
    "c658b8ac1cd6ec8cc0fa225b892bcd1aa2c438e3b2aea77ed22584d6b688e1bd"
)
HISTORICAL_PROBE_RECEIPT_FILE_SHA256 = (
    "117bf56591582a6de915b017ff46663ea96f809bd44de9540625a48aa4489723"
)
FAILED_PROBE_ATTEMPT_FILE_SHA256 = (
    "58acceca305411bc061f759c37c9b0a17a69fc86ea903563e85c2d7eef078b62"
)
FAILED_PROBE_RECEIPT_FILE_SHA256 = (
    "0b425323456ef859b4124e65f165ff3c8b656aa431ea7ded7d0e4ee8dd87cf2f"
)
AUTHORIZED_PARTICIPANT = "e1-stage-boundary"
AUTHORIZED_OPPONENT = "r22-r7"
AUTHORIZED_NORMALIZED_DECK_HASH = (
    "sha256:109a82f280159691594de831d3f3f507ea11d10d2bc8eaf54b23ba4ce1aee107"
)
RESOLVED_OPPONENT_BINDING_ID = "starmie-r7-d1"
RESOLVED_OPPONENT_SPEC = f"binding:{RESOLVED_OPPONENT_BINDING_ID}"
AUTHORIZED_FEATURE_DECK_HASH = (
    "b15853cdfb8304f39f879638102a7fd161f1102d81e6d3f36d5beca4db9e01b0"
)
AUTHORIZED_DECK_FILE_SHA256 = (
    "2636fcaf8aedd7a156d0b2e33185b0e3878a9c2e2b77425f9203cc53e96e7b28"
)
ACTION_SCHEMA_IDENTITY = "deepmew.actor_transition.v1"
OBSERVATION_SCHEMA_IDENTITY = "deepmew.replay_action_row.v1:state_json actor_visible"
AUTHORIZED_RUNTIME_ACTION_SCHEMA_SHA256 = (
    "51de2a30e929f04986937201ec58e63b0271f27314d4ca0f1ed4c426b18c8268"
)
AUTHORIZED_RUNTIME_OBSERVATION_SCHEMA_SHA256 = (
    "bf0cc2e34085e01f9e396be5e4a376384343848d1c1f32fb882de4d0a9c4b7e7"
)
AUTHORIZED_PROTECTED_BASELINE_COMMIT = "fd7afe287576562a927e59f382c65b9647ae857e"
AUTHORIZED_PROTECTED_TREE_IDENTITY_SHA256 = (
    "98f2c1381f42a5d7bff9fc09cb2ab633ef6ae72ded36d4fb32fadbb9a01d7155"
)
MODAL_APPS = {
    "ap-L0oyhlqIc4yA1SMkwQDRSw": "deepmew-terminal-league",
    "ap-SidcOl2H187GHLPUS4cNbK": "deepmew-terminal-league-f7f9e1e89b25",
}
PROTECTED_PATHS = (
    "deck.csv",
    "main.py",
    "cg",
    "deepmew/submissions",
    "deepmew/policy",
    "deepmew/agents",
    "deepmew/turn_plan",
)
MISSION_ROOT = (
    ROOT
    / ".research"
    / "terminal_league_starmie_grimmsnarl_v0"
    / "e1_starmie_complete_enhanced_generation_20260714"
)
AUTHORIZED_CALLSITE = (ROOT / "scripts" / "run_e1_real_stage_probe.py").resolve()
CANONICAL_PROJECTOR_PATH = Path(canonical_actor_feature_observation.__code__.co_filename).resolve()
DIAGNOSTIC_RECEIPT = (
    MISSION_ROOT
    / "observation_path_diagnostic_20260714_01"
    / "OBSERVATION_PATH_DIAGNOSTIC_RECEIPT.json"
)
HISTORICAL_PROBE_RECEIPT = MISSION_ROOT / "real_stage_probe_v0" / "receipt_v4.json"
FAILED_PROBE_ROOT = MISSION_ROOT / "real_stage_probe_frozen_contract_superseding_20260714_01"
FAILED_PROBE_ATTEMPT = FAILED_PROBE_ROOT / "attempt.json"
FAILED_PROBE_RECEIPT = FAILED_PROBE_ROOT / "failure_receipt.json"
RAW_ENGINE_VISIBILITY_MODES = frozenset({
    InformationMode.PRIVILEGED_ORACLE.value,
    "engine_control_only",
})
PROJECTED_FORBIDDEN_KEYS = frozenset({
    "private_opponent",
    "privileged_oracle",
    "search_begin_input",
})
FORBIDDEN_PERSISTED_KEYS = frozenset({
    "search_begin_input",
    "private_state",
    "private_hash",
    "hidden_state",
    "opponent_hand",
    "opponent_deck",
    "opponent_prize",
})


class E1ProbeResolutionHardStop(RuntimeError):
    """An exact frozen E1 probe binding or safety condition failed."""


@dataclass(frozen=True)
class FrozenProbeContract:
    config: StageProbeConfig
    frozen_config_path: Path
    frozen_config_sha256: str
    launch_receipt_path: Path
    launch_receipt_sha256: str
    correction_source_path: Path
    correction_source_sha256: str
    diagnostic_receipt_path: Path
    diagnostic_receipt_sha256: str
    projector_path: Path
    projector_sha256: str
    action_schema_identity: str
    observation_schema_identity: str


@dataclass(frozen=True)
class ProbeAttemptAuthorization:
    path: Path
    file_sha256: str
    expected_stage_boundary: str
    expected_stage_receipt_path: Path
    expected_stage_receipt_sha256: str
    expected_stage_checkpoint_path: Path
    expected_stage_checkpoint_sha256: str
    expected_actor_export_path: Path
    expected_actor_export_sha256: str
    output_path: Path


@dataclass(frozen=True)
class ResolvedProbeContext:
    contract: FrozenProbeContract
    attempt_authorization: ProbeAttemptAuthorization
    stage_receipt_path: Path
    stage_receipt_sha256: str
    stage_checkpoint_path: Path
    stage_checkpoint_sha256: str
    stage_checkpoint_payload: Mapping[str, Any]
    actor_export_path: Path
    actor_export_sha256: str
    opponent_checkpoint_path: Path
    opponent_checkpoint_sha256: str
    opponent_binding_manifest_path: Path
    opponent_binding_manifest_sha256: str
    deck: tuple[int, ...]
    deck_path: Path
    deck_file_sha256: str
    normalized_deck_hash: str
    feature_deck_hash: str
    canonical_deck_payload: Mapping[str, Any]
    canonical_multiset_sha256: str
    feature_policy_id: str


@dataclass
class ObservationLeakageAudit:
    raw_observations_checked: int = 0
    observations_checked: int = 0
    projections_completed: int = 0
    tensorizations_completed: int = 0
    model_forwards_completed: int = 0
    projected_policy_calls_without_tensor_capture: int = 0
    engine_control_tokens_seen: int = 0
    engine_control_tokens_removed: int = 0
    violations: int = 0
    raw_visibility_modes: dict[str, int] = field(default_factory=dict)
    visibility_modes: dict[str, int] = field(default_factory=dict)
    first_projected_tensor_shape: tuple[int, ...] | None = None
    first_projected_tensor_sha256: str | None = None
    _pending_tensor_sha256: str | None = None

    def validate_raw_boundary(self, observation: Mapping[str, Any]) -> str:
        self.raw_observations_checked += 1
        if observation.get("search_begin_input") is None:
            self.violations += 1
            raise E1ProbeResolutionHardStop(
                "raw local-engine observation lacks the expected opaque engine-control token"
            )
        self.engine_control_tokens_seen += 1
        mode = classify_observation_visibility(observation)
        self.raw_visibility_modes[mode] = self.raw_visibility_modes.get(mode, 0) + 1
        if mode not in RAW_ENGINE_VISIBILITY_MODES:
            self.violations += 1
            raise E1ProbeResolutionHardStop(
                f"raw local-engine observation classified as unexpected mode {mode!r}"
            )

        return mode

    def validate(self, observation: Mapping[str, Any]) -> None:
        keys = {key.casefold() for key in _nested_keys(observation)}
        forbidden = sorted(keys & (FORBIDDEN_OBSERVATION_INPUTS | PROJECTED_FORBIDDEN_KEYS))
        if forbidden:
            self.violations += 1
            raise E1ProbeResolutionHardStop(
                f"live learner observation contains forbidden inputs: {forbidden}"
            )
        try:
            mode = assert_deployable_observation(observation)
        except ValueError as exc:
            self.violations += 1
            raise E1ProbeResolutionHardStop(str(exc)) from exc
        if mode not in {
            InformationMode.ACTOR_VISIBLE.value,
            InformationMode.PUBLIC_BELIEF.value,
        }:
            self.violations += 1
            raise E1ProbeResolutionHardStop(
                f"live learner observation classified as non-deployable mode {mode!r}"
            )
        self.visibility_modes[mode] = self.visibility_modes.get(mode, 0) + 1
        self.observations_checked += 1

    def project_and_validate(
        self,
        observation: Mapping[str, Any],
        *,
        expected_actor: int,
    ) -> dict[str, Any]:
        self.validate_raw_boundary(observation)
        projected = canonical_actor_feature_observation(observation)
        self.projections_completed += 1
        if projected is observation:
            self.violations += 1
            raise E1ProbeResolutionHardStop("canonical actor projection returned the raw observation object")
        if "search_begin_input" in projected:
            self.violations += 1
            raise E1ProbeResolutionHardStop("canonical actor projection retained the engine-control token")
        current = projected.get("current")
        if not isinstance(current, Mapping):
            self.violations += 1
            raise E1ProbeResolutionHardStop("projected observation lacks current actor state")
        try:
            observed_actor = int(current.get("yourIndex"))
        except (TypeError, ValueError) as exc:
            self.violations += 1
            raise E1ProbeResolutionHardStop("projected observation has invalid current.yourIndex") from exc
        if observed_actor != expected_actor:
            self.violations += 1
            raise E1ProbeResolutionHardStop(
                f"projected learner seat drifted: expected {expected_actor}, got {observed_actor}"
            )
        self.validate(projected)
        self.engine_control_tokens_removed += 1

        return projected

    def record_tensorization(self, tensor: torch.Tensor) -> None:
        if self.tensorizations_completed >= self.observations_checked:
            self.violations += 1
            raise E1ProbeResolutionHardStop(
                "tensorization occurred before projected-observation validation"
            )
        if self._pending_tensor_sha256 is not None:
            self.violations += 1
            raise E1ProbeResolutionHardStop("a projected tensor bypassed its policy-forward audit")
        shape = tuple(int(value) for value in tensor.shape)
        digest = _tensor_sha256(tensor)
        if len(shape) != 2 or shape[1] != EXPECTED_FIRST_PROJECTED_TENSOR_SHAPE[1]:
            self.violations += 1
            raise E1ProbeResolutionHardStop(f"projected tensor shape is invalid: {shape}")
        if self.first_projected_tensor_shape is None:
            if shape != EXPECTED_FIRST_PROJECTED_TENSOR_SHAPE:
                self.violations += 1
            _require_equal(
                shape,
                EXPECTED_FIRST_PROJECTED_TENSOR_SHAPE,
                "first projected tensor shape",
            )
            if digest != EXPECTED_FIRST_PROJECTED_TENSOR_SHA256:
                self.violations += 1
            _require_equal(
                digest,
                EXPECTED_FIRST_PROJECTED_TENSOR_SHA256,
                "first projected tensor hash",
            )
            self.first_projected_tensor_shape = shape
            self.first_projected_tensor_sha256 = digest
        self.tensorizations_completed += 1
        self._pending_tensor_sha256 = digest

    def validate_policy_forward(self, tensor: torch.Tensor) -> None:
        digest = _tensor_sha256(tensor)
        if self._pending_tensor_sha256 is None or digest != self._pending_tensor_sha256:
            self.violations += 1
            raise E1ProbeResolutionHardStop(
                "policy forward did not receive the validated projected tensor"
            )
        self._pending_tensor_sha256 = None
        self.model_forwards_completed += 1

    def record_projected_policy_call_without_tensor_capture(self) -> None:
        if self.tensorizations_completed >= self.observations_checked:
            self.violations += 1
            raise E1ProbeResolutionHardStop(
                "policy call occurred before projected-opponent observation validation"
            )
        self.tensorizations_completed += 1
        self.model_forwards_completed += 1
        self.projected_policy_calls_without_tensor_capture += 1

    def assert_complete(self) -> None:
        counts = {
            self.raw_observations_checked,
            self.projections_completed,
            self.observations_checked,
            self.tensorizations_completed,
            self.model_forwards_completed,
            self.engine_control_tokens_seen,
            self.engine_control_tokens_removed,
        }
        if counts == {0} or len(counts) != 1 or self._pending_tensor_sha256 is not None:
            raise E1ProbeResolutionHardStop(
                "projection, validation, tensorization, and policy-forward counts are not one-to-one"
            )
        if (
            self.first_projected_tensor_shape != EXPECTED_FIRST_PROJECTED_TENSOR_SHAPE
            or self.first_projected_tensor_sha256 != EXPECTED_FIRST_PROJECTED_TENSOR_SHA256
        ):
            raise E1ProbeResolutionHardStop("first learner tensor is not bound to the diagnostic receipt")
        if self.violations:
            raise E1ProbeResolutionHardStop(
                f"projection boundary recorded {self.violations} leakage violations"
            )


@dataclass
class ProbeExecutionState:
    games_completed: int = 0
    current_seed: int | None = None
    current_game_index: int | None = None
    safety: dict[str, int] = field(default_factory=lambda: dict(ZERO_SAFETY))


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(_canonical_json_bytes({
        "dtype": str(cpu.dtype),
        "shape": list(cpu.shape),
    }))
    digest.update(cpu.numpy().tobytes(order="C"))

    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise E1ProbeResolutionHardStop(f"{path}: expected a JSON object")

    return payload


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = _canonical_json_bytes(dict(payload)) + b"\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as exc:
        raise E1ProbeResolutionHardStop(
            f"immutable receipt already exists: {path}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        # Keep even a partial exclusive file as a no-retry sentinel.
        raise


def _require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise E1ProbeResolutionHardStop(
            f"{label} drifted: expected {expected!r}, got {observed!r}"
        )


def _require_file_sha256(path: Path, expected: str, label: str) -> str:
    actual = _sha256(path.resolve(strict=True))
    _require_equal(actual, expected, label)

    return actual


def _validate_research_callsite(callsite: Path) -> None:
    _require_equal(
        callsite.resolve(strict=True),
        AUTHORIZED_CALLSITE,
        "research-only stage-probe callsite",
    )


def _stage_probe_config(payload: Mapping[str, Any]) -> StageProbeConfig:
    stage_probe = payload.get("stage_probe")
    if not isinstance(stage_probe, Mapping):
        raise E1ProbeResolutionHardStop("frozen config lacks stage_probe")
    _require_equal(stage_probe.get("schema_version"), "deepmew.e1_stage_probe.v1", "probe schema")
    _require_equal(
        stage_probe.get("final_confirmatory_seeds_forbidden"),
        True,
        "probe final-seed exclusion",
    )
    config = StageProbeConfig(
        participant=str(stage_probe.get("participant") or ""),
        opponent_policy_identity=str(stage_probe.get("opponent_policy_identity") or ""),
        opponent_checkpoint_sha256=str(stage_probe.get("opponent_checkpoint_sha256") or ""),
        deck_identity=str(stage_probe.get("deck_identity") or ""),
        opponent_deck_identity=str(stage_probe.get("opponent_deck_identity") or ""),
        seeds=tuple(int(seed) for seed in stage_probe.get("seeds") or ()),
        games_per_seed=int(stage_probe.get("games_per_seed") or 0),
        alternating_seats=bool(stage_probe.get("alternating_seats")),
        maximum_game_steps=int(stage_probe.get("maximum_game_steps") or 0),
    )
    _validate_authorized_probe_config(config)

    return config


def _validate_authorized_probe_config(config: StageProbeConfig) -> None:
    config.validate()
    _require_equal(config.identity_sha256, AUTHORIZED_PROBE_CONFIG_SHA256, "frozen probe config hash")
    _require_equal(config.participant, AUTHORIZED_PARTICIPANT, "literal participant")
    _require_equal(config.opponent_policy_identity, AUTHORIZED_OPPONENT, "literal opponent")
    _require_equal(config.opponent_checkpoint_sha256, R7_CHECKPOINT_SHA256, "r22-r7 checkpoint")
    _require_equal(config.deck_identity, AUTHORIZED_NORMALIZED_DECK_HASH, "literal learner deck")
    _require_equal(
        config.opponent_deck_identity,
        AUTHORIZED_NORMALIZED_DECK_HASH,
        "literal opponent deck",
    )


def load_frozen_probe_contract(
    *,
    frozen_config_path: Path,
    launch_receipt_path: Path,
    correction_source_path: Path,
) -> FrozenProbeContract:
    _validate_research_callsite(Path(__file__))
    frozen_path = frozen_config_path.resolve(strict=True)
    launch_path = launch_receipt_path.resolve(strict=True)
    correction_path = correction_source_path.resolve(strict=True)
    diagnostic_path = DIAGNOSTIC_RECEIPT.resolve(strict=True)
    projector_path = CANONICAL_PROJECTOR_PATH.resolve(strict=True)
    frozen_sha256 = _require_file_sha256(
        frozen_path,
        AUTHORIZED_FROZEN_CONFIG_FILE_SHA256,
        "frozen config file",
    )
    launch_sha256 = _require_file_sha256(
        launch_path,
        AUTHORIZED_LAUNCH_RECEIPT_FILE_SHA256,
        "E1 launch authorization receipt",
    )
    correction_sha256 = _require_file_sha256(
        correction_path,
        AUTHORIZED_CORRECTION_SOURCE_SHA256,
        "probe correction authorization source",
    )
    diagnostic_sha256 = _require_file_sha256(
        diagnostic_path,
        AUTHORIZED_DIAGNOSTIC_RECEIPT_FILE_SHA256,
        "observation-path diagnostic receipt",
    )
    projector_sha256 = _sha256(projector_path)
    frozen = _read_json(frozen_path)
    launch = _read_json(launch_path)
    diagnostic = _read_json(diagnostic_path)
    _require_equal(frozen.get("mission"), MISSION, "frozen mission")
    _require_equal(launch.get("mission"), MISSION, "launch mission")
    _require_equal(
        launch.get("schema_version"),
        "deepmew.e1_launch_authorization.v1",
        "launch authorization schema",
    )
    _require_equal(launch.get("exact_resume_required"), True, "exact resume requirement")
    _require_equal(
        diagnostic.get("terminal_classification"),
        "SAFE_PROJECTION_ORDER_BUG_CONFIRMED",
        "observation-path diagnostic classification",
    )
    bound_inputs = launch.get("bound_inputs")
    if not isinstance(bound_inputs, Mapping):
        raise E1ProbeResolutionHardStop("launch authorization lacks bound_inputs")
    _require_equal(
        bound_inputs.get("frozen_config_sha256"),
        frozen_sha256,
        "launch-to-frozen binding",
    )
    _require_equal(
        bound_inputs.get("real_engine_probe_receipt_sha256"),
        HISTORICAL_PROBE_RECEIPT_FILE_SHA256,
        "historical probe binding",
    )
    replay = frozen.get("replay")
    if not isinstance(replay, Mapping):
        raise E1ProbeResolutionHardStop("frozen config lacks replay schema bindings")
    _require_equal(replay.get("action_schema_identity"), ACTION_SCHEMA_IDENTITY, "action schema")
    _require_equal(
        replay.get("observation_schema_identity"),
        OBSERVATION_SCHEMA_IDENTITY,
        "observation schema",
    )
    config = _stage_probe_config(frozen)

    return FrozenProbeContract(
        config=config,
        frozen_config_path=frozen_path,
        frozen_config_sha256=frozen_sha256,
        launch_receipt_path=launch_path,
        launch_receipt_sha256=launch_sha256,
        correction_source_path=correction_path,
        correction_source_sha256=correction_sha256,
        diagnostic_receipt_path=diagnostic_path,
        diagnostic_receipt_sha256=diagnostic_sha256,
        projector_path=projector_path,
        projector_sha256=projector_sha256,
        action_schema_identity=ACTION_SCHEMA_IDENTITY,
        observation_schema_identity=OBSERVATION_SCHEMA_IDENTITY,
    )


def load_probe_attempt_authorization(
    *,
    path: Path,
    contract: FrozenProbeContract,
    actor_export_path: Path,
    stage_receipt_path: Path,
    output_path: Path,
) -> ProbeAttemptAuthorization:
    authorization_path = path.resolve(strict=True)
    if not authorization_path.is_relative_to(MISSION_ROOT.resolve(strict=True)):
        raise E1ProbeResolutionHardStop(
            "probe attempt authorization must remain inside the E1 mission directory"
        )
    payload = _read_json(authorization_path)
    _require_equal(
        payload.get("schema_version"),
        ATTEMPT_AUTHORIZATION_SCHEMA_VERSION,
        "probe attempt authorization schema",
    )
    _require_equal(payload.get("status"), "AUTHORIZED_EXACTLY_ONE_LOCAL_PROBE", "probe attempt status")
    _require_equal(payload.get("mission"), MISSION, "probe attempt mission")
    _require_equal(
        payload.get("correction_authorization_source_sha256"),
        contract.correction_source_sha256,
        "attempt correction authorization",
    )
    _require_equal(
        payload.get("frozen_config_file_sha256"),
        contract.frozen_config_sha256,
        "attempt frozen config",
    )
    _require_equal(
        payload.get("launch_authorization_receipt_sha256"),
        contract.launch_receipt_sha256,
        "attempt launch authorization",
    )
    _require_equal(
        payload.get("probe_config_sha256"),
        contract.config.identity_sha256,
        "attempt probe config",
    )
    diagnostic = payload.get("diagnostic_binding")
    if not isinstance(diagnostic, Mapping):
        raise E1ProbeResolutionHardStop("probe attempt authorization lacks diagnostic binding")
    _require_equal(
        diagnostic.get("receipt_sha256"),
        contract.diagnostic_receipt_sha256,
        "attempt diagnostic receipt",
    )
    _require_equal(
        Path(str(diagnostic.get("receipt_path") or "")).resolve(strict=True),
        contract.diagnostic_receipt_path,
        "attempt diagnostic receipt path",
    )
    literal = payload.get("literal_contract_identity")
    if not isinstance(literal, Mapping):
        raise E1ProbeResolutionHardStop("attempt authorization lacks literal contract identity")
    _require_equal(literal.get("participant"), AUTHORIZED_PARTICIPANT, "attempt literal participant")
    _require_equal(literal.get("opponent"), AUTHORIZED_OPPONENT, "attempt literal opponent")
    _require_equal(literal.get("normalized_deck_hash"), AUTHORIZED_NORMALIZED_DECK_HASH, "attempt literal deck")
    stage = payload.get("participant_stage_artifacts")
    if not isinstance(stage, Mapping):
        raise E1ProbeResolutionHardStop("attempt authorization lacks participant stage artifacts")
    expected_receipt_path = Path(str(stage.get("stage_receipt_path") or "")).resolve(strict=True)
    expected_checkpoint_path = Path(str(stage.get("stage_checkpoint_path") or "")).resolve(strict=True)
    expected_actor_path = Path(str(stage.get("actor_export_path") or "")).resolve(strict=True)
    actual_receipt_path = stage_receipt_path.resolve(strict=True)
    actual_actor_path = actor_export_path.resolve(strict=True)
    _require_equal(actual_receipt_path, expected_receipt_path, "orchestrator stage receipt path")
    _require_equal(_stage_checkpoint_path(actual_receipt_path), expected_checkpoint_path, "orchestrator stage checkpoint path")
    _require_equal(actual_actor_path, expected_actor_path, "orchestrator actor export path")
    resolved = payload.get("resolved_artifacts")
    if not isinstance(resolved, Mapping):
        raise E1ProbeResolutionHardStop("attempt authorization lacks resolved artifacts")
    _require_equal(resolved.get("opponent_binding_id"), RESOLVED_OPPONENT_BINDING_ID, "attempt opponent binding")
    _require_equal(resolved.get("opponent_checkpoint_sha256"), R7_CHECKPOINT_SHA256, "attempt opponent checkpoint")
    _require_equal(resolved.get("deck_file_sha256"), AUTHORIZED_DECK_FILE_SHA256, "attempt deck artifact")
    _require_equal(resolved.get("r4f_feature_deck_hash"), AUTHORIZED_FEATURE_DECK_HASH, "attempt feature deck")
    schemas = payload.get("schemas")
    if not isinstance(schemas, Mapping):
        raise E1ProbeResolutionHardStop("attempt authorization lacks schemas")
    _require_equal(schemas.get("action_schema_identity"), ACTION_SCHEMA_IDENTITY, "attempt action schema")
    _require_equal(
        schemas.get("observation_schema_identity"),
        OBSERVATION_SCHEMA_IDENTITY,
        "attempt observation schema",
    )
    _require_equal(
        schemas.get("runtime_action_schema_sha256"),
        AUTHORIZED_RUNTIME_ACTION_SCHEMA_SHA256,
        "attempt runtime action schema",
    )
    _require_equal(
        schemas.get("runtime_observation_schema_sha256"),
        AUTHORIZED_RUNTIME_OBSERVATION_SCHEMA_SHA256,
        "attempt runtime observation schema",
    )
    resolution = payload.get("resolution_layer")
    if not isinstance(resolution, Mapping):
        raise E1ProbeResolutionHardStop("attempt authorization lacks resolution layer")
    _require_equal(
        Path(str(resolution.get("callsite") or "")).resolve(strict=True),
        AUTHORIZED_CALLSITE,
        "attempt resolution callsite",
    )
    _require_equal(
        resolution.get("implementation_sha256"),
        _sha256(AUTHORIZED_CALLSITE),
        "attempt resolution implementation",
    )
    _require_equal(
        resolution.get("configuration_sha256"),
        _sha256_json(_resolution_configuration()),
        "attempt resolution configuration",
    )
    projection = payload.get("projection_implementation")
    if not isinstance(projection, Mapping):
        raise E1ProbeResolutionHardStop("probe attempt authorization lacks projection implementation")
    _require_equal(
        projection.get("adapter_sha256"),
        _sha256(AUTHORIZED_CALLSITE),
        "attempt projection adapter hash",
    )
    _require_equal(
        Path(str(projection.get("projector_path") or "")).resolve(strict=True),
        contract.projector_path,
        "attempt canonical projector path",
    )
    _require_equal(
        projection.get("projector_sha256"),
        contract.projector_sha256,
        "attempt canonical projector hash",
    )
    expected_output = Path(str(payload.get("output_path") or "")).resolve()
    _require_equal(output_path.resolve(), expected_output, "orchestrator probe output path")
    _require_equal(payload.get("total_games"), 16, "attempt total games")
    _require_equal(payload.get("automatic_retry_allowed"), False, "attempt retry policy")
    _require_equal(payload.get("paid_worker_authorized"), False, "attempt paid-worker policy")
    _require_equal(payload.get("external_launch_authorized"), False, "attempt external-launch policy")

    return ProbeAttemptAuthorization(
        path=authorization_path,
        file_sha256=_sha256(authorization_path),
        expected_stage_boundary=str(stage.get("boundary_stage") or ""),
        expected_stage_receipt_path=expected_receipt_path,
        expected_stage_receipt_sha256=str(stage.get("stage_receipt_sha256") or ""),
        expected_stage_checkpoint_path=expected_checkpoint_path,
        expected_stage_checkpoint_sha256=str(stage.get("stage_checkpoint_sha256") or ""),
        expected_actor_export_path=expected_actor_path,
        expected_actor_export_sha256=str(stage.get("actor_export_sha256") or ""),
        output_path=expected_output,
    )


def _stage_checkpoint_path(stage_receipt_path: Path) -> Path:
    suffix = ".receipt.json"
    raw = str(stage_receipt_path)
    if not raw.endswith(suffix):
        raise E1ProbeResolutionHardStop("stage receipt path must end with .receipt.json")

    return Path(raw[:-len(suffix)]).resolve(strict=True)


def _validate_stage_payload(
    *,
    contract: FrozenProbeContract,
    stage_receipt: Mapping[str, Any],
    checkpoint_payload: Mapping[str, Any],
    checkpoint_sha256: str,
    actor_export_sha256: str,
) -> None:
    _require_equal(
        stage_receipt.get("schema_version"),
        STAGE_CHECKPOINT_RECEIPT_SCHEMA_VERSION,
        "stage receipt schema",
    )
    _require_equal(checkpoint_payload.get("schema_version"), CHECKPOINT_SCHEMA_VERSION, "stage checkpoint schema")
    _require_equal(stage_receipt.get("checkpoint_sha256"), checkpoint_sha256, "stage checkpoint receipt hash")
    _require_equal(stage_receipt.get("actor_export_sha256"), actor_export_sha256, "stage actor export hash")
    boundary_index = int(stage_receipt.get("boundary_index", -1))
    if not 0 <= boundary_index < len(STAGE_BOUNDARY_ORDER):
        raise E1ProbeResolutionHardStop("stage boundary index is outside the frozen stage order")
    boundary_stage = str(stage_receipt.get("boundary_stage") or "")
    _require_equal(boundary_stage, STAGE_BOUNDARY_ORDER[boundary_index], "stage boundary order")
    _require_equal(checkpoint_payload.get("boundary_index"), boundary_index, "checkpoint boundary index")
    _require_equal(checkpoint_payload.get("boundary_stage"), boundary_stage, "checkpoint boundary stage")
    _require_equal(
        checkpoint_payload.get("trainer_lineage_id"),
        stage_receipt.get("trainer_lineage_id"),
        "stage trainer lineage",
    )
    metadata = checkpoint_payload.get("frozen_metadata")
    if not isinstance(metadata, Mapping):
        raise E1ProbeResolutionHardStop("stage checkpoint lacks frozen metadata")
    metadata_sha256 = _sha256_json(dict(metadata))
    _require_equal(metadata_sha256, stage_receipt.get("frozen_metadata_sha256"), "stage metadata receipt hash")
    _require_equal(
        checkpoint_payload.get("frozen_metadata_sha256"),
        metadata_sha256,
        "stage metadata checkpoint hash",
    )
    _require_equal(
        metadata.get("action_schema_identity"),
        contract.action_schema_identity,
        "stage action schema",
    )
    _require_equal(
        metadata.get("observation_schema_identity"),
        contract.observation_schema_identity,
        "stage observation schema",
    )
    _require_equal(
        checkpoint_payload.get("source_checkpoint_sha256"),
        contract.config.opponent_checkpoint_sha256,
        "stage source r7 checkpoint",
    )
    _require_equal(
        checkpoint_payload.get("component_hashes"),
        stage_receipt.get("component_hashes"),
        "stage component hashes",
    )
    _require_equal(
        metadata.get("continuous_stage_chain_sha256"),
        stage_receipt.get("continuous_stage_chain_sha256"),
        "continuous stage chain",
    )
    _require_equal(metadata.get("next_operation"), stage_receipt.get("next_operation"), "stage next operation")


def _canonical_deck_proof(deck: Sequence[int], expected_normalized_hash: str) -> dict[str, Any]:
    card_ids = tuple(int(card_id) for card_id in deck)
    _require_equal(len(card_ids), 60, "canonical deck card count")
    payload = normalized_deck_payload(card_ids)
    observed_normalized_hash = normalized_deck_hash(card_ids)
    _require_equal(observed_normalized_hash, expected_normalized_hash, "canonical deck normalized hash")

    return {
        "card_count": len(card_ids),
        "canonical_payload": payload,
        "canonical_multiset_sha256": _sha256_json(payload),
        "normalized_deck_hash": observed_normalized_hash,
        "r4f_feature_deck_hash": r4f_feature_deck_hash(card_ids),
    }


def resolve_probe_context(
    *,
    contract: FrozenProbeContract,
    attempt_authorization: ProbeAttemptAuthorization,
    actor_export_path: Path,
    stage_receipt_path: Path,
) -> ResolvedProbeContext:
    stage_path = stage_receipt_path.resolve(strict=True)
    actor_path = actor_export_path.resolve(strict=True)
    stage_receipt = _read_json(stage_path)
    checkpoint_path = _stage_checkpoint_path(stage_path)
    expected_actor_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".actor.pt").resolve(strict=True)
    _require_equal(actor_path, expected_actor_path, "stage actor export path")
    checkpoint_sha256 = _sha256(checkpoint_path)
    actor_sha256 = _sha256(actor_path)
    stage_receipt_sha256 = _sha256(stage_path)
    _require_equal(
        stage_receipt_sha256,
        attempt_authorization.expected_stage_receipt_sha256,
        "orchestrator-supplied stage receipt hash",
    )
    _require_equal(
        checkpoint_sha256,
        attempt_authorization.expected_stage_checkpoint_sha256,
        "orchestrator-supplied stage checkpoint hash",
    )
    _require_equal(
        actor_sha256,
        attempt_authorization.expected_actor_export_sha256,
        "orchestrator-supplied actor export hash",
    )
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint_payload, Mapping):
        raise E1ProbeResolutionHardStop("stage checkpoint payload is not a mapping")
    _validate_stage_payload(
        contract=contract,
        stage_receipt=stage_receipt,
        checkpoint_payload=checkpoint_payload,
        checkpoint_sha256=checkpoint_sha256,
        actor_export_sha256=actor_sha256,
    )
    _require_equal(
        stage_receipt.get("boundary_stage"),
        attempt_authorization.expected_stage_boundary,
        "orchestrator-supplied stage boundary",
    )
    binding = POLICY_DECK_BINDINGS.get(RESOLVED_OPPONENT_BINDING_ID)
    if binding is None:
        raise E1ProbeResolutionHardStop("authorized r22-r7 resolved binding is absent")
    _require_equal(binding.policy_artifact_sha256, R7_CHECKPOINT_SHA256, "resolved r22-r7 checkpoint binding")
    _require_equal(binding.feature_deck_hash, AUTHORIZED_FEATURE_DECK_HASH, "resolved feature deck binding")
    _require_equal(binding.expected_deck_file_sha256, AUTHORIZED_DECK_FILE_SHA256, "resolved deck file binding")
    resolved = resolve_policy_deck_binding(binding)
    if resolved.policy_artifact_path is None:
        raise E1ProbeResolutionHardStop("resolved r22-r7 checkpoint path is absent")
    opponent_checkpoint_sha256 = _sha256(resolved.policy_artifact_path)
    _require_equal(opponent_checkpoint_sha256, R7_CHECKPOINT_SHA256, "resolved r22-r7 checkpoint artifact")
    _require_equal(resolved.deck_file_sha256, AUTHORIZED_DECK_FILE_SHA256, "resolved D1 deck artifact")
    deck_proof = _canonical_deck_proof(resolved.deck, contract.config.deck_identity)
    _require_equal(
        deck_proof["r4f_feature_deck_hash"],
        AUTHORIZED_FEATURE_DECK_HASH,
        "resolved ordered feature deck hash",
    )
    manifest_path = Path(binding.deck_manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    manifest_path = manifest_path.resolve(strict=True)

    return ResolvedProbeContext(
        contract=contract,
        attempt_authorization=attempt_authorization,
        stage_receipt_path=stage_path,
        stage_receipt_sha256=stage_receipt_sha256,
        stage_checkpoint_path=checkpoint_path,
        stage_checkpoint_sha256=checkpoint_sha256,
        stage_checkpoint_payload=checkpoint_payload,
        actor_export_path=actor_path,
        actor_export_sha256=actor_sha256,
        opponent_checkpoint_path=resolved.policy_artifact_path,
        opponent_checkpoint_sha256=opponent_checkpoint_sha256,
        opponent_binding_manifest_path=manifest_path,
        opponent_binding_manifest_sha256=_sha256(manifest_path),
        deck=tuple(resolved.deck),
        deck_path=resolved.deck_path,
        deck_file_sha256=resolved.deck_file_sha256,
        normalized_deck_hash=str(deck_proof["normalized_deck_hash"]),
        feature_deck_hash=str(deck_proof["r4f_feature_deck_hash"]),
        canonical_deck_payload=dict(deck_proof["canonical_payload"]),
        canonical_multiset_sha256=str(deck_proof["canonical_multiset_sha256"]),
        feature_policy_id=str(binding.feature_policy_id),
    )


def _runtime_schema_evidence(actor_payload: Mapping[str, Any], context: ResolvedProbeContext) -> dict[str, Any]:
    observation_schema = {
        "schema_version": "deepmew.terminal_league_actor_visible_r4f.v1",
        "input_dim": int(actor_payload["input_dim"]),
        "feature_mode": actor_payload.get("feature_mode"),
        "state_schema": list(actor_payload.get("state_schema") or []),
        "option_schema": list(actor_payload.get("option_schema") or []),
        "feature_policy_id": context.feature_policy_id,
        "deck_hash": context.normalized_deck_hash,
        "feature_deck_hash": context.feature_deck_hash,
        "hidden_state_allowed": False,
    }
    action_schema = {
        "schema_version": "deepmew.terminal_league_r4f_action.v1",
        "semantics": "ordered_unique_live_option_indices_plus_stop",
        "mask": "engine_live_legal_options_and_inclusive_count_bounds",
        "learner_sanitization": False,
        "terminal_rewards": [-1, 0, 1],
    }
    observation_sha256 = _sha256_json(observation_schema)
    action_sha256 = _sha256_json(action_schema)
    _require_equal(
        observation_sha256,
        AUTHORIZED_RUNTIME_OBSERVATION_SCHEMA_SHA256,
        "runtime observation schema",
    )
    _require_equal(action_sha256, AUTHORIZED_RUNTIME_ACTION_SCHEMA_SHA256, "runtime action schema")

    return {
        "frozen_action_schema_identity": context.contract.action_schema_identity,
        "frozen_observation_schema_identity": context.contract.observation_schema_identity,
        "runtime_action_schema": action_schema,
        "runtime_action_schema_sha256": action_sha256,
        "runtime_observation_schema": observation_schema,
        "runtime_observation_schema_sha256": observation_sha256,
    }


def _nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(_nested_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            keys.update(_nested_keys(item))

    return keys


def _negative_leakage_control() -> bool:
    negative = {
        "current": {
            "yourIndex": 0,
            "players": [
                {"hand": [], "prize": [None] * 6},
                {"hand": None, "prize": [None] * 6},
            ],
        },
        "opponent_hand_cards": [999_999],
    }
    audit = ObservationLeakageAudit()
    try:
        audit.validate(negative)
    except (E1ProbeResolutionHardStop, ValueError):
        return True

    return False


def _contains_forbidden_persisted_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() in FORBIDDEN_PERSISTED_KEYS:
                return True
            if _contains_forbidden_persisted_key(item):
                return True

        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_persisted_key(item) for item in value)

    return False


def _runtime_contamination_evidence() -> dict[str, Any]:
    loaded = sorted(
        module
        for module in sys.modules
        if any(module == forbidden or module.startswith(f"{forbidden}.") for forbidden in FORBIDDEN_OFFLINE_LABEL_MODULES)
    )
    if loaded:
        raise E1ProbeResolutionHardStop(f"forbidden runtime modules are loaded: {loaded}")

    return {
        "forbidden_modules_loaded": loaded,
        "runtime_contamination_detected": False,
    }


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except subprocess.TimeoutExpired as exc:
        raise E1ProbeResolutionHardStop(
            f"git {' '.join(args)} timed out"
        ) from exc
    if completed.returncode != 0:
        raise E1ProbeResolutionHardStop(
            f"git {' '.join(args)} failed: {completed.stderr.strip()}"
        )

    return completed


def _protected_surface_evidence() -> dict[str, Any]:
    head = _git("rev-parse", "HEAD").stdout.strip()
    _require_equal(head, AUTHORIZED_PROTECTED_BASELINE_COMMIT, "protected baseline commit")
    status = _git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *PROTECTED_PATHS,
    ).stdout.strip()
    if status:
        raise E1ProbeResolutionHardStop(f"protected surface changed: {status}")
    tracked_raw = _git("ls-files", "-z", "--", *PROTECTED_PATHS).stdout
    tracked = sorted(path for path in tracked_raw.split("\0") if path)
    hashes = {}
    for relative in tracked:
        path = (ROOT / relative).resolve(strict=True)
        hashes[relative.replace("\\", "/")] = _sha256(path)
    tree_identity_sha256 = _sha256_json(hashes)
    _require_equal(
        tree_identity_sha256,
        AUTHORIZED_PROTECTED_TREE_IDENTITY_SHA256,
        "protected baseline tree identity",
    )
    import_violations = []
    scanned_python_files = 0
    for relative in tracked:
        if not relative.casefold().endswith(".py"):
            continue
        scanned_python_files += 1
        source = (ROOT / relative).read_text(encoding="utf-8", errors="replace")
        if "run_e1_real_stage_probe" in source or "e1_stage_probe_resolution" in source:
            import_violations.append(relative.replace("\\", "/"))
    if import_violations:
        raise E1ProbeResolutionHardStop(
            f"protected runtime imports the research resolver: {import_violations}"
        )

    return {
        "paths": list(PROTECTED_PATHS),
        "baseline_commit": head,
        "git_status": "clean",
        "tracked_file_sha256": hashes,
        "tree_identity_sha256": tree_identity_sha256,
        "runtime_import_scan_files": scanned_python_files,
        "runtime_import_violations": import_violations,
    }


def _modal_task_evidence() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["modal", "app", "list", "--json"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise E1ProbeResolutionHardStop("Modal task inventory timed out") from exc
    if completed.returncode != 0:
        raise E1ProbeResolutionHardStop(
            f"Modal task inventory failed: {completed.stderr.strip()}"
        )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, list):
        raise E1ProbeResolutionHardStop("Modal task inventory is not a list")
    by_id = {
        str(row.get("app_id")): row
        for row in payload
        if isinstance(row, Mapping)
    }
    evidence = []
    for app_id, description in MODAL_APPS.items():
        row = by_id.get(app_id)
        if row is None:
            raise E1ProbeResolutionHardStop(f"authorized Modal app {app_id} is absent")
        _require_equal(str(row.get("description")), description, f"Modal app {app_id} description")
        try:
            tasks = int(row.get("tasks", -1))
        except (TypeError, ValueError) as exc:
            raise E1ProbeResolutionHardStop(f"Modal app {app_id} task count is invalid") from exc
        _require_equal(tasks, 0, f"Modal app {app_id} active tasks")
        evidence.append({
            "app_id": app_id,
            "description": description,
            "state": str(row.get("state")),
            "tasks": tasks,
        })

    return {"apps": evidence, "all_tasks_zero": True}


def _resolution_configuration() -> dict[str, Any]:
    return {
        "schema_version": RESOLUTION_SCHEMA_VERSION,
        "mission": MISSION,
        "allowed_callsite": "scripts/run_e1_real_stage_probe.py",
        "allowed_callsite_sha256": _sha256(AUTHORIZED_CALLSITE),
        "frozen_config_file_sha256": AUTHORIZED_FROZEN_CONFIG_FILE_SHA256,
        "launch_authorization_receipt_sha256": AUTHORIZED_LAUNCH_RECEIPT_FILE_SHA256,
        "correction_authorization_source_sha256": AUTHORIZED_CORRECTION_SOURCE_SHA256,
        "diagnostic_receipt_sha256": AUTHORIZED_DIAGNOSTIC_RECEIPT_FILE_SHA256,
        "canonical_projector_path": str(CANONICAL_PROJECTOR_PATH),
        "canonical_projector_sha256": _sha256(CANONICAL_PROJECTOR_PATH),
        "first_projected_tensor_shape": list(EXPECTED_FIRST_PROJECTED_TENSOR_SHAPE),
        "first_projected_tensor_sha256": EXPECTED_FIRST_PROJECTED_TENSOR_SHA256,
        "policy_observation_order": [
            "canonical_actor_feature_observation",
            "projected_observation_leakage_guard",
            "tensorization",
            "policy_forward",
        ],
        "probe_config_sha256": AUTHORIZED_PROBE_CONFIG_SHA256,
        "literal_participant": AUTHORIZED_PARTICIPANT,
        "literal_opponent": AUTHORIZED_OPPONENT,
        "literal_normalized_deck_hash": AUTHORIZED_NORMALIZED_DECK_HASH,
        "resolved_opponent_binding_id": RESOLVED_OPPONENT_BINDING_ID,
        "resolved_opponent_checkpoint_sha256": R7_CHECKPOINT_SHA256,
        "resolved_deck_file_sha256": AUTHORIZED_DECK_FILE_SHA256,
        "resolved_feature_deck_hash": AUTHORIZED_FEATURE_DECK_HASH,
        "action_schema_identity": ACTION_SCHEMA_IDENTITY,
        "observation_schema_identity": OBSERVATION_SCHEMA_IDENTITY,
        "runtime_action_schema_sha256": AUTHORIZED_RUNTIME_ACTION_SCHEMA_SHA256,
        "runtime_observation_schema_sha256": AUTHORIZED_RUNTIME_OBSERVATION_SCHEMA_SHA256,
        "attempt_authorization_schema_version": ATTEMPT_AUTHORIZATION_SCHEMA_VERSION,
        "protected_baseline_commit": AUTHORIZED_PROTECTED_BASELINE_COMMIT,
        "protected_tree_identity_sha256": AUTHORIZED_PROTECTED_TREE_IDENTITY_SHA256,
        "unrestricted_alias_table": False,
    }


def _probe_result(terminal_reward: float) -> int:
    if terminal_reward > 0.0:
        return 1
    if terminal_reward < 0.0:
        return -1

    return 0


def _validate_pair_identity(pair_identity: Mapping[str, Any]) -> None:
    _require_equal(
        pair_identity.get("binding_id"),
        RESOLVED_OPPONENT_BINDING_ID,
        "sampled opponent binding",
    )
    _require_equal(
        pair_identity.get("policy_artifact_sha256"),
        R7_CHECKPOINT_SHA256,
        "sampled opponent checkpoint",
    )
    _require_equal(
        pair_identity.get("deck_hash"),
        AUTHORIZED_NORMALIZED_DECK_HASH,
        "sampled opponent normalized deck",
    )
    _require_equal(
        pair_identity.get("feature_deck_hash"),
        AUTHORIZED_FEATURE_DECK_HASH,
        "sampled opponent feature deck",
    )


def _prepare_samplers(config: StageProbeConfig) -> dict[int, OpponentSampler]:
    samplers = {}
    for seed in config.seeds:
        sampler = OpponentSampler(
            "e1-frozen-r22-r7",
            [(RESOLVED_OPPONENT_SPEC, 1.0)],
            seed=seed,
        )
        sampler.materialize_all()
        if len(sampler.agents) != 1:
            raise E1ProbeResolutionHardStop("frozen probe sampler is not single-opponent")
        _validate_pair_identity(sampler.agents[0].pair_identity)
        samplers[seed] = sampler

    return samplers


def _encode_projected_observation(
    observation: Mapping[str, Any],
    *,
    expected_actor: int,
    audit: ObservationLeakageAudit,
    encoder: Any,
    encoder_kwargs: Mapping[str, Any],
) -> Any:
    projected = audit.project_and_validate(
        observation,
        expected_actor=expected_actor,
    )
    encoded = encoder(projected, **dict(encoder_kwargs))
    features = getattr(encoded, "features", None)
    if not isinstance(features, torch.Tensor):
        audit.violations += 1
        raise E1ProbeResolutionHardStop("projected encoder did not return a tensor")
    audit.record_tensorization(features)

    return encoded


def _select_with_projected_observation(
    observation: Mapping[str, Any],
    rng: random.Random,
    *,
    expected_actor: int,
    audit: ObservationLeakageAudit,
    selector: Any,
) -> list[int]:
    projected = audit.project_and_validate(
        observation,
        expected_actor=expected_actor,
    )
    selection = selector(projected, rng)
    audit.record_projected_policy_call_without_tensor_capture()

    return [int(index) for index in selection]


def _run_probe_games(
    *,
    learner: R4FNativePPO,
    context: ResolvedProbeContext,
    audit: ObservationLeakageAudit,
    execution: ProbeExecutionState,
    samplers: Mapping[int, OpponentSampler],
) -> Sequence[ProbeGameResult]:
    original_encoder = terminal.encode_actor_visible_decision
    original_sample = learner.sample
    original_opponent_selectors = []

    def audited_encoder(observation: dict[str, Any], **kwargs: Any):
        if execution.current_game_index is None:
            raise E1ProbeResolutionHardStop("learner observation arrived outside a frozen probe game")
        expected_actor = execution.current_game_index % 2

        return _encode_projected_observation(
            observation,
            expected_actor=expected_actor,
            audit=audit,
            encoder=original_encoder,
            encoder_kwargs=kwargs,
        )

    def audited_sample(
        option_features: torch.Tensor,
        min_count: int,
        max_count: int,
        *,
        deterministic: bool = False,
    ):
        audit.validate_policy_forward(option_features)

        return original_sample(
            option_features,
            min_count,
            max_count,
            deterministic=deterministic,
        )

    for sampler in samplers.values():
        opponent_spec = sampler.agents[0].spec
        original_selector = opponent_spec.select

        def audited_opponent_select(
            observation: dict[str, Any],
            rng: random.Random,
            *,
            _selector: Any = original_selector,
        ) -> list[int]:
            if execution.current_game_index is None:
                raise E1ProbeResolutionHardStop(
                    "opponent observation arrived outside a frozen probe game"
                )

            return _select_with_projected_observation(
                observation,
                rng,
                expected_actor=1 - (execution.current_game_index % 2),
                audit=audit,
                selector=_selector,
            )

        opponent_spec.select = audited_opponent_select
        original_opponent_selectors.append((opponent_spec, original_selector))

    terminal.encode_actor_visible_decision = audited_encoder
    learner.sample = audited_sample
    games = []
    try:
        for seed in context.contract.config.seeds:
            cursor = terminal.EnvironmentCursor(random.Random(seed))
            sampler = samplers.get(seed)
            if sampler is None:
                raise E1ProbeResolutionHardStop(f"pre-materialized sampler missing seed {seed}")
            for game_index in range(context.contract.config.games_per_seed):
                execution.current_seed = seed
                execution.current_game_index = game_index
                random.seed(seed + game_index)
                torch.manual_seed(seed + game_index)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed + game_index)
                counters = dict(ZERO_SAFETY)
                try:
                    collected = terminal.collect_terminal_episode(
                        learner=learner,
                        sampler=sampler,
                        environment_cursor=cursor,
                        learner_deck=context.deck,
                        learner_deck_hash=context.feature_deck_hash,
                        feature_policy_id=context.feature_policy_id,
                        max_steps=context.contract.config.maximum_game_steps,
                        counters=counters,
                    )
                except Exception:
                    for key in ZERO_SAFETY:
                        execution.safety[key] += int(counters[key])
                    raise
                for key in ZERO_SAFETY:
                    execution.safety[key] += int(counters[key])
                _require_equal(collected.our_seat, game_index % 2, "frozen learner seat")
                _validate_pair_identity(collected.pair_identity)
                target = 1.0 if collected.episode.terminal_reward > 0 else 0.0
                probabilities = tuple(
                    min(max((step.old_value + 1.0) / 2.0, 0.0), 1.0)
                    for step in collected.episode.steps
                )
                legal_counts = tuple(
                    int(step.option_features.shape[0])
                    for step in collected.episode.steps
                )
                with torch.no_grad():
                    entropies = tuple(
                        float(learner.distribution.evaluate(step)[1].detach().cpu())
                        for step in collected.episode.steps
                    )
                games.append(ProbeGameResult(
                    seed=seed,
                    seat=collected.our_seat,
                    result=_probe_result(collected.episode.terminal_reward),
                    steps=len(collected.episode.steps),
                    legal_action_counts=legal_counts,
                    action_entropies=entropies,
                    critic_probabilities=probabilities,
                    critic_targets=tuple(target for _ in probabilities),
                    crash=bool(counters["crashes"]),
                    illegal_action=bool(counters["illegal_actions"]),
                    max_step_timeout=bool(counters["timeouts"]),
                ))
                execution.games_completed += 1
    finally:
        terminal.encode_actor_visible_decision = original_encoder
        learner.sample = original_sample
        for opponent_spec, original_selector in original_opponent_selectors:
            opponent_spec.select = original_selector

    return tuple(games)


def _build_wrapper_receipt(
    *,
    context: ResolvedProbeContext,
    probe_result: Mapping[str, Any],
    schema_evidence: Mapping[str, Any],
    audit: ObservationLeakageAudit,
    protected_before: Mapping[str, Any],
    protected_after: Mapping[str, Any],
    modal_before: Mapping[str, Any],
    modal_after: Mapping[str, Any],
    historical_sha256_after: str,
) -> dict[str, Any]:
    resolution_configuration = _resolution_configuration()
    checkpoint_payload = context.stage_checkpoint_payload
    stage_receipt = _read_json(context.stage_receipt_path)
    extra_state = checkpoint_payload.get("extra_state")
    is_smoke_stage = bool(extra_state.get("smoke")) if isinstance(extra_state, Mapping) else False
    receipt = {
        "schema_version": WRAPPER_SCHEMA_VERSION,
        "status": "PASS_FROZEN_CONTRACT_RESOLVED",
        "mission": MISSION,
        "literal_contract_identity": {
            "probe_config": asdict(context.contract.config),
            "probe_config_sha256": context.contract.config.identity_sha256,
            "participant": context.contract.config.participant,
            "opponent_policy_identity": context.contract.config.opponent_policy_identity,
            "opponent_checkpoint_sha256": context.contract.config.opponent_checkpoint_sha256,
            "deck_identity": context.contract.config.deck_identity,
            "opponent_deck_identity": context.contract.config.opponent_deck_identity,
            "frozen_config_path": str(context.contract.frozen_config_path),
            "frozen_config_file_sha256": context.contract.frozen_config_sha256,
        },
        "resolved_artifact_identity": {
            "participant_resolution": "explicit_current_stage_receipt",
            "participant_artifact_classification": (
                "prelaunch_smoke_stage_boundary"
                if is_smoke_stage
                else "production_stage_boundary"
            ),
            "probe_receipt_counts_as_production_stage_progress": False,
            "participant_stage": stage_receipt["boundary_stage"],
            "participant_stage_checkpoint_path": str(context.stage_checkpoint_path),
            "participant_stage_checkpoint_sha256": context.stage_checkpoint_sha256,
            "participant_actor_export_path": str(context.actor_export_path),
            "participant_actor_export_sha256": context.actor_export_sha256,
            "stage_receipt_path": str(context.stage_receipt_path),
            "stage_receipt_sha256": context.stage_receipt_sha256,
            "trainer_lineage_id": stage_receipt["trainer_lineage_id"],
            "opponent_binding_id": RESOLVED_OPPONENT_BINDING_ID,
            "opponent_engine_spec": RESOLVED_OPPONENT_SPEC,
            "opponent_checkpoint_path": str(context.opponent_checkpoint_path),
            "opponent_checkpoint_sha256": context.opponent_checkpoint_sha256,
            "opponent_binding_manifest_path": str(context.opponent_binding_manifest_path),
            "opponent_binding_manifest_sha256": context.opponent_binding_manifest_sha256,
            "deck_path": str(context.deck_path),
            "deck_file_sha256": context.deck_file_sha256,
            "r4f_feature_deck_hash": context.feature_deck_hash,
        },
        "binding_proof": {
            "status": "pass",
            "literal_contract_preserved": True,
            "participant_checkpoint_hash_matches_stage_receipt": True,
            "opponent_checkpoint_hash_matches_frozen_contract": True,
            "canonical_card_count": len(context.deck),
            "canonical_card_multiset": context.canonical_deck_payload,
            "canonical_multiset_sha256": context.canonical_multiset_sha256,
            "normalized_deck_hash": context.normalized_deck_hash,
            "normalized_deck_hash_matches": context.normalized_deck_hash == AUTHORIZED_NORMALIZED_DECK_HASH,
            "ordered_feature_hash_matches": context.feature_deck_hash == AUTHORIZED_FEATURE_DECK_HASH,
            "schemas": dict(schema_evidence),
            "stage_metadata_sha256": checkpoint_payload["frozen_metadata_sha256"],
            "continuous_stage_chain_sha256": stage_receipt["continuous_stage_chain_sha256"],
            "launch_authorization_receipt_path": str(context.contract.launch_receipt_path),
            "launch_authorization_receipt_sha256": context.contract.launch_receipt_sha256,
            "correction_authorization_source_path": str(context.contract.correction_source_path),
            "correction_authorization_source_sha256": context.contract.correction_source_sha256,
            "observation_path_diagnostic_receipt_path": str(
                context.contract.diagnostic_receipt_path
            ),
            "observation_path_diagnostic_receipt_sha256": (
                context.contract.diagnostic_receipt_sha256
            ),
            "probe_attempt_authorization_path": str(context.attempt_authorization.path),
            "probe_attempt_authorization_sha256": context.attempt_authorization.file_sha256,
        },
        "resolution_layer": {
            "schema_version": RESOLUTION_SCHEMA_VERSION,
            "scope": "research_only_stage_probe_call_site",
            "callsite": str(Path(__file__).resolve()),
            "implementation_sha256": _sha256(Path(__file__).resolve()),
            "projection_implementation": {
                "adapter_sha256": _sha256(Path(__file__).resolve()),
                "canonical_projector": "canonical_actor_feature_observation",
                "projector_path": str(context.contract.projector_path),
                "projector_sha256": context.contract.projector_sha256,
            },
            "configuration": resolution_configuration,
            "configuration_sha256": _sha256_json(resolution_configuration),
            "unrestricted_alias_table": False,
        },
        "probe_result": dict(probe_result),
        "information_boundary": {
            "status": "pass",
            "observation_entrypoint": "run_terminal_league_ppo.encode_actor_visible_decision",
            "terminal_adapter_sha256": _sha256(Path(terminal.__file__).resolve()),
            "raw_observations_checked": audit.raw_observations_checked,
            "observations_checked": audit.observations_checked,
            "projections_completed": audit.projections_completed,
            "tensorizations_completed": audit.tensorizations_completed,
            "model_forwards_completed": audit.model_forwards_completed,
            "projected_opponent_policy_calls": (
                audit.projected_policy_calls_without_tensor_capture
            ),
            "violations": audit.violations,
            "raw_observation_classifications": dict(sorted(audit.raw_visibility_modes.items())),
            "visibility_modes": dict(sorted(audit.visibility_modes.items())),
            "engine_control_token_removal_proof": {
                "token_field_name": "search_begin_input",
                "present_on_raw_count": audit.engine_control_tokens_seen,
                "removed_before_validation_count": audit.engine_control_tokens_removed,
                "absent_from_all_projected_observations": (
                    audit.engine_control_tokens_removed == audit.observations_checked
                ),
            },
            "first_projected_tensor": {
                "shape": list(audit.first_projected_tensor_shape or ()),
                "sha256": audit.first_projected_tensor_sha256,
                "matches_diagnostic_receipt": (
                    audit.first_projected_tensor_shape == EXPECTED_FIRST_PROJECTED_TENSOR_SHAPE
                    and audit.first_projected_tensor_sha256
                    == EXPECTED_FIRST_PROJECTED_TENSOR_SHA256
                ),
            },
            "all_policy_inputs_projected_before_validation": (
                audit.raw_observations_checked
                == audit.projections_completed
                == audit.observations_checked
                == audit.tensorizations_completed
                == audit.model_forwards_completed
            ),
            "raw_observations_forwarded_to_tensorizer": 0,
            "raw_observations_forwarded_to_policy": False,
            "negative_control_rejected": _negative_leakage_control(),
            "raw_observations_persisted": False,
            "private_state_persisted": False,
            "hidden_information_leakage_detected": False,
            **_runtime_contamination_evidence(),
        },
        "protected_surfaces": {
            "before": dict(protected_before),
            "after": dict(protected_after),
            "unchanged": protected_before["tree_identity_sha256"] == protected_after["tree_identity_sha256"],
        },
        "modal_apps": {
            "before": dict(modal_before),
            "after": dict(modal_after),
            "all_tasks_zero": bool(modal_before["all_tasks_zero"] and modal_after["all_tasks_zero"]),
        },
        "supersedes": {
            "failed_probe_attempt_path": str(FAILED_PROBE_ATTEMPT.resolve(strict=True)),
            "failed_probe_attempt_file_sha256": _sha256(FAILED_PROBE_ATTEMPT),
            "failed_probe_receipt_path": str(FAILED_PROBE_RECEIPT.resolve(strict=True)),
            "failed_probe_receipt_file_sha256": _sha256(FAILED_PROBE_RECEIPT),
            "failed_probe_preserved": (
                _sha256(FAILED_PROBE_ATTEMPT) == FAILED_PROBE_ATTEMPT_FILE_SHA256
                and _sha256(FAILED_PROBE_RECEIPT) == FAILED_PROBE_RECEIPT_FILE_SHA256
            ),
            "historical_receipt_path": str(HISTORICAL_PROBE_RECEIPT.resolve(strict=True)),
            "historical_receipt_file_sha256_before": HISTORICAL_PROBE_RECEIPT_FILE_SHA256,
            "historical_receipt_file_sha256_after": historical_sha256_after,
            "historical_receipt_preserved": historical_sha256_after == HISTORICAL_PROBE_RECEIPT_FILE_SHA256,
            "historical_probe_config_sha256": "864404ea5fd3f1b7343be212f758638df879e94464c9b1548617ae483ed73c4f",
            "reason": "historical engineering probe used non-authoritative literal identities",
        },
        "execution": {
            "local_real_engine_probe": True,
            "external_launch_performed": False,
            "paid_worker_started": False,
            "automatic_retry_allowed": False,
        },
    }
    if _contains_forbidden_persisted_key(receipt):
        raise E1ProbeResolutionHardStop("superseding receipt contains a forbidden private-state key")
    _validate_wrapper_receipt(receipt)
    receipt["receipt_sha256"] = _sha256_json(receipt)

    return receipt


def _validate_wrapper_receipt(receipt: Mapping[str, Any]) -> None:
    literal = receipt["literal_contract_identity"]
    result = receipt["probe_result"]
    _require_equal(literal["participant"], AUTHORIZED_PARTICIPANT, "receipt literal participant")
    _require_equal(literal["opponent_policy_identity"], AUTHORIZED_OPPONENT, "receipt literal opponent")
    _require_equal(literal["deck_identity"], AUTHORIZED_NORMALIZED_DECK_HASH, "receipt literal deck")
    _require_equal(literal["probe_config_sha256"], AUTHORIZED_PROBE_CONFIG_SHA256, "receipt probe config")
    _require_equal(result["completed_games"], 16, "superseding probe completed games")
    for field in ("crashes", "illegal_actions", "max_step_timeouts", "incomplete_games"):
        _require_equal(result[field], 0, f"superseding probe {field}")
    _require_equal(receipt["binding_proof"]["status"], "pass", "binding proof")
    _require_equal(
        receipt["information_boundary"]["hidden_information_leakage_detected"],
        False,
        "hidden-information leakage",
    )
    _require_equal(receipt["information_boundary"]["negative_control_rejected"], True, "leakage negative control")
    boundary = receipt["information_boundary"]
    _require_equal(boundary["violations"], 0, "projection-boundary leakage violations")
    _require_equal(
        boundary["all_policy_inputs_projected_before_validation"],
        True,
        "projection-before-policy proof",
    )
    _require_equal(
        boundary["raw_observations_forwarded_to_tensorizer"],
        0,
        "raw observation tensorizer forwarding",
    )
    _require_equal(
        boundary["raw_observations_forwarded_to_policy"],
        False,
        "raw observation policy forwarding",
    )
    _require_equal(
        boundary["raw_observation_classifications"],
        {InformationMode.PRIVILEGED_ORACLE.value: boundary["raw_observations_checked"]},
        "raw observation classification",
    )
    _require_equal(
        boundary["engine_control_token_removal_proof"]["absent_from_all_projected_observations"],
        True,
        "engine-control token removal",
    )
    _require_equal(
        boundary["first_projected_tensor"]["shape"],
        list(EXPECTED_FIRST_PROJECTED_TENSOR_SHAPE),
        "receipt first projected tensor shape",
    )
    _require_equal(
        boundary["first_projected_tensor"]["sha256"],
        EXPECTED_FIRST_PROJECTED_TENSOR_SHA256,
        "receipt first projected tensor hash",
    )
    _require_equal(
        boundary["first_projected_tensor"]["matches_diagnostic_receipt"],
        True,
        "diagnostic tensor binding",
    )
    _require_equal(receipt["protected_surfaces"]["unchanged"], True, "protected surfaces")
    _require_equal(receipt["modal_apps"]["all_tasks_zero"], True, "Modal tasks")
    _require_equal(receipt["supersedes"]["historical_receipt_preserved"], True, "historical receipt preservation")
    _require_equal(receipt["supersedes"]["failed_probe_preserved"], True, "failed probe preservation")
    _require_equal(receipt["execution"]["paid_worker_started"], False, "paid worker start")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).resolve()
    attempt_path = output.with_name("attempt.json")
    failure_path = output.with_name("failure_receipt.json")
    if output.exists() or attempt_path.exists() or failure_path.exists():
        raise E1ProbeResolutionHardStop("superseding probe output or attempt already exists; automatic retry forbidden")
    if not output.is_relative_to(MISSION_ROOT.resolve(strict=True)):
        raise E1ProbeResolutionHardStop("superseding receipt must remain inside the E1 mission directory")
    contract = load_frozen_probe_contract(
        frozen_config_path=Path(args.frozen_config),
        launch_receipt_path=Path(args.launch_authorization),
        correction_source_path=Path(args.correction_authorization),
    )
    attempt_authorization = load_probe_attempt_authorization(
        path=Path(args.resolution_authorization),
        contract=contract,
        actor_export_path=Path(args.actor_export),
        stage_receipt_path=Path(args.stage_receipt),
        output_path=output,
    )
    context = resolve_probe_context(
        contract=contract,
        attempt_authorization=attempt_authorization,
        actor_export_path=Path(args.actor_export),
        stage_receipt_path=Path(args.stage_receipt),
    )
    historical_sha256_before = _require_file_sha256(
        HISTORICAL_PROBE_RECEIPT,
        HISTORICAL_PROBE_RECEIPT_FILE_SHA256,
        "historical probe receipt",
    )
    _require_file_sha256(
        FAILED_PROBE_ATTEMPT,
        FAILED_PROBE_ATTEMPT_FILE_SHA256,
        "failed probe attempt",
    )
    _require_file_sha256(
        FAILED_PROBE_RECEIPT,
        FAILED_PROBE_RECEIPT_FILE_SHA256,
        "failed probe receipt",
    )
    protected_before = _protected_surface_evidence()
    modal_before = _modal_task_evidence()
    implementation_sha256_before = _sha256(Path(__file__).resolve())
    terminal_adapter_sha256_before = _sha256(Path(terminal.__file__).resolve())
    actor_payload = torch.load(context.actor_export_path, map_location="cpu", weights_only=False)
    if not isinstance(actor_payload, Mapping):
        raise E1ProbeResolutionHardStop("stage actor export payload is not a mapping")
    schema_evidence = _runtime_schema_evidence(actor_payload, context)
    input_dim = int(actor_payload["input_dim"])
    learner = R4FNativePPO.from_r4f_checkpoint(
        context.actor_export_path,
        terminal._frozen_equivalence_panel(input_dim),
    )
    terminal._register_artifact_loaders()
    samplers = _prepare_samplers(contract.config)
    attempt_id = _sha256_json({
        "probe_config_sha256": contract.config.identity_sha256,
        "stage_checkpoint_sha256": context.stage_checkpoint_sha256,
        "stage_actor_export_sha256": context.actor_export_sha256,
        "launch_authorization_receipt_sha256": contract.launch_receipt_sha256,
        "correction_authorization_source_sha256": contract.correction_source_sha256,
        "probe_attempt_authorization_sha256": attempt_authorization.file_sha256,
    })
    _exclusive_json(attempt_path, {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "status": "started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempt_id": attempt_id,
        "probe_config_sha256": contract.config.identity_sha256,
        "stage_checkpoint_sha256": context.stage_checkpoint_sha256,
        "probe_attempt_authorization_sha256": attempt_authorization.file_sha256,
        "paid_worker_started": False,
        "automatic_retry_allowed": False,
    })
    audit = ObservationLeakageAudit()
    execution = ProbeExecutionState()
    try:
        games = _run_probe_games(
            learner=learner,
            context=context,
            audit=audit,
            execution=execution,
            samplers=samplers,
        )
        audit.assert_complete()
        probe_result = run_fixed_stage_probe(
            context.actor_export_sha256,
            contract.config,
            lambda checkpoint_sha256, frozen: games
            if checkpoint_sha256 == context.actor_export_sha256 and frozen == contract.config
            else (),
        )
        protected_after = _protected_surface_evidence()
        modal_after = _modal_task_evidence()
        _require_file_sha256(
            Path(__file__).resolve(),
            implementation_sha256_before,
            "probe resolution implementation during execution",
        )
        _require_file_sha256(
            Path(terminal.__file__).resolve(),
            terminal_adapter_sha256_before,
            "terminal adapter during execution",
        )
        for artifact_path, expected_sha256, label in (
            (context.contract.frozen_config_path, context.contract.frozen_config_sha256, "frozen config after probe"),
            (context.contract.launch_receipt_path, context.contract.launch_receipt_sha256, "launch receipt after probe"),
            (context.contract.correction_source_path, context.contract.correction_source_sha256, "correction source after probe"),
            (
                context.contract.diagnostic_receipt_path,
                context.contract.diagnostic_receipt_sha256,
                "observation-path diagnostic receipt after probe",
            ),
            (
                context.contract.projector_path,
                context.contract.projector_sha256,
                "canonical actor projector after probe",
            ),
            (
                FAILED_PROBE_ATTEMPT,
                FAILED_PROBE_ATTEMPT_FILE_SHA256,
                "failed probe attempt after probe",
            ),
            (
                FAILED_PROBE_RECEIPT,
                FAILED_PROBE_RECEIPT_FILE_SHA256,
                "failed probe receipt after probe",
            ),
            (
                context.attempt_authorization.path,
                context.attempt_authorization.file_sha256,
                "probe attempt authorization after probe",
            ),
            (context.stage_receipt_path, context.stage_receipt_sha256, "stage receipt after probe"),
            (context.stage_checkpoint_path, context.stage_checkpoint_sha256, "stage checkpoint after probe"),
            (context.actor_export_path, context.actor_export_sha256, "actor export after probe"),
            (context.opponent_checkpoint_path, context.opponent_checkpoint_sha256, "opponent checkpoint after probe"),
            (
                context.opponent_binding_manifest_path,
                context.opponent_binding_manifest_sha256,
                "opponent binding manifest after probe",
            ),
            (context.deck_path, context.deck_file_sha256, "bound deck after probe"),
        ):
            _require_file_sha256(artifact_path, expected_sha256, label)
        historical_sha256_after = _sha256(HISTORICAL_PROBE_RECEIPT)
        receipt = _build_wrapper_receipt(
            context=context,
            probe_result=probe_result,
            schema_evidence=schema_evidence,
            audit=audit,
            protected_before=protected_before,
            protected_after=protected_after,
            modal_before=modal_before,
            modal_after=modal_after,
            historical_sha256_after=historical_sha256_after,
        )
        receipt["attempt_id"] = attempt_id
        receipt.pop("receipt_sha256", None)
        receipt["receipt_sha256"] = _sha256_json(receipt)
        _exclusive_json(output, receipt)
    except Exception as exc:
        safety_category = getattr(exc, "category", None)
        safety_detail = getattr(exc, "detail", None)
        safety_context = getattr(exc, "context", None)
        failure = {
            "schema_version": FAILURE_SCHEMA_VERSION,
            "status": "hard_failure_no_retry",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "attempt_id": attempt_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failure_category": safety_category,
            "failure_detail": safety_detail,
            "failure_context": dict(safety_context) if isinstance(safety_context, Mapping) else {},
            "games_completed": execution.games_completed,
            "current_seed": execution.current_seed,
            "current_game_index": execution.current_game_index,
            "cumulative_safety": dict(execution.safety),
            "observations_checked": audit.observations_checked,
            "violations": audit.violations,
            "paid_worker_started": False,
            "automatic_retry_allowed": False,
        }
        _exclusive_json(failure_path, failure)
        raise

    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-decision-receipt", default="")
    parser.add_argument("--actor-export", required=True)
    parser.add_argument("--stage-receipt", required=True)
    parser.add_argument("--frozen-config", required=True)
    parser.add_argument("--launch-authorization", required=True)
    parser.add_argument("--correction-authorization", required=True)
    parser.add_argument("--resolution-authorization", required=True)
    parser.add_argument("--output", required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    accepted_receipt = require_accepted_canonical_decision_receipt(
        args.canonical_decision_receipt,
        consumer=__name__,
    )
    args.canonical_decision_receipt = str(accepted_receipt.path)
    try:
        receipt = run(args)
    except Exception as exc:
        print(json.dumps({
            "status": "hard_failure_no_retry",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }, sort_keys=True), flush=True)

        return 1
    print(json.dumps({
        "status": receipt["status"],
        "probe_config_sha256": receipt["literal_contract_identity"]["probe_config_sha256"],
        "completed_games": receipt["probe_result"]["completed_games"],
        "safety_clean": all(
            receipt["probe_result"][field] == 0
            for field in ("crashes", "illegal_actions", "max_step_timeouts", "incomplete_games")
        ),
        "receipt_sha256": receipt["receipt_sha256"],
    }, sort_keys=True), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
