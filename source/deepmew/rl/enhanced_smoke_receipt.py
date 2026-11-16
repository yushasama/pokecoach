"""Fail-closed evidence contract for the enhanced-generation local smoke.

The contract is deliberately isolated from live Modal controls. It accepts a
smoke only when every mandatory signal is present, affirmative, internally
consistent, and bound to one trainer/model lineage through a five-stage hash
chain. It performs no training, launch, deployment, or protected-surface IO.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


SMOKE_RECEIPT_SCHEMA_VERSION = "deepmew.enhanced_generation_smoke.v1"
SMOKE_CLASSIFICATION = "quarantine"
REQUIRED_BRANCH_ROWS = 9_184
REQUIRED_BRANCH_SHARDS = 288
STAGE_CHAIN_GENESIS_SHA256 = "0" * 64
REQUIRED_STAGE_NAMES = (
    "v0_initialization_or_distillation",
    "replay_warm_start",
    "branch_q_supervision",
    "opponent_curriculum",
    "terminal_reward_league_ppo",
)

REQUIRED_RECEIPT_FIELDS = frozenset({
    "schema_version",
    "classification",
    "status",
    "trainer_identity",
    "initial_model_sha256",
    "final_model_sha256",
    "v0_weights_or_distillation_consumed",
    "unique_replay_rows_consumed",
    "unique_branch_rows_consumed",
    "unique_branch_shards_consumed",
    "opponent_deck_strength_metadata_consumed_by_scheduler",
    "actor_parameters_updated",
    "critic_parameters_updated",
    "checkpoint_resume_parity",
    "zero_hidden_information_leakage",
    "same_trainer_identity_stage_hash_chain",
    "v0_consumption",
    "replay_consumption",
    "branch_consumption",
    "curriculum_consumption",
    "parameter_updates",
    "checkpoint_resume",
    "leakage_scan",
    "terminal_ppo",
    "stage_chain",
    "stage_chain_sha256",
    "external_launch_performed",
    "protected_surfaces_mutated",
})

_V0_FIELDS = frozenset({
    "consumed",
    "mode",
    "weights_consumed",
    "distillation_consumed",
    "source_checkpoint_sha256",
    "evidence_sha256",
})
_REPLAY_FIELDS = frozenset({
    "unique_rows_consumed",
    "row_identity_set_sha256",
    "source_catalog_sha256",
    "evidence_sha256",
})
_BRANCH_FIELDS = frozenset({
    "unique_rows_consumed",
    "unique_shards_consumed",
    "row_identity_set_sha256",
    "shard_identity_set_sha256",
    "root_manifest_sha256",
    "evidence_sha256",
})
_CURRICULUM_FIELDS = frozenset({
    "consumed",
    "deck_identity_participated",
    "anchored_strength_bucket_participated",
    "all_strength_buckets_anchored",
    "effective_distribution_l1_delta_from_base",
    "scheduler_config_sha256",
    "scheduler_receipt_sha256",
    "evidence_sha256",
})
_PARAMETER_UPDATE_FIELDS = frozenset({
    "actor_updated",
    "critic_updated",
    "actor_delta_norm",
    "critic_delta_norm",
    "actor_before_sha256",
    "actor_after_sha256",
    "critic_before_sha256",
    "critic_after_sha256",
    "before_model_sha256",
    "after_model_sha256",
    "evidence_sha256",
})
_CHECKPOINT_RESUME_FIELDS = frozenset({
    "parity",
    "checkpoint_sha256",
    "saved_state_sha256",
    "restored_state_sha256",
    "next_step_expected_sha256",
    "next_step_resumed_sha256",
    "evidence_sha256",
})
_LEAKAGE_SCAN_FIELDS = frozenset({
    "zero_hidden_information_leakage",
    "violation_count",
    "scanned_artifact_count",
    "negative_control_rejected",
    "evidence_sha256",
})
_TERMINAL_PPO_FIELDS = frozenset({
    "consumed",
    "terminal_episodes",
    "optimizer_updates",
    "illegal_actions",
    "crashes",
    "timeouts",
    "evidence_sha256",
})
_STAGE_FIELDS = frozenset({
    "stage_index",
    "stage_name",
    "trainer_identity",
    "input_model_sha256",
    "output_model_sha256",
    "signal_consumed",
    "signal_evidence_sha256",
    "previous_stage_sha256",
    "stage_sha256",
})
_STAGE_HASH_FIELDS = (
    "stage_index",
    "stage_name",
    "trainer_identity",
    "input_model_sha256",
    "output_model_sha256",
    "signal_consumed",
    "signal_evidence_sha256",
    "previous_stage_sha256",
)


class EnhancedSmokeReceiptError(ValueError):
    """The enhanced smoke evidence was absent, false, or inconsistent."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise EnhancedSmokeReceiptError("smoke evidence must be finite JSON data") from exc

    return rendered.encode("ascii")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _exact_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise EnhancedSmokeReceiptError(f"{name} must be a non-empty string")
    if value != value.strip():
        raise EnhancedSmokeReceiptError(f"{name} must not contain surrounding whitespace")

    return value


def _sha256_identity(value: Any, *, name: str) -> str:
    text = _exact_text(value, name=name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise EnhancedSmokeReceiptError(f"{name} must be a lowercase SHA-256 identity")

    return text


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EnhancedSmokeReceiptError(f"{name} must be a positive integer")

    return value


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EnhancedSmokeReceiptError(f"{name} must be a non-negative integer")

    return value


def _positive_finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise EnhancedSmokeReceiptError(f"{name} must be finite and positive")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise EnhancedSmokeReceiptError(f"{name} must be finite and positive") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise EnhancedSmokeReceiptError(f"{name} must be finite and positive")

    return number


def _require_true(value: Any, *, name: str) -> None:
    if value is not True:
        raise EnhancedSmokeReceiptError(f"{name} must be explicitly true")


def _require_false(value: Any, *, name: str) -> None:
    if value is not False:
        raise EnhancedSmokeReceiptError(f"{name} must be explicitly false")


def _require_exact_fields(
    value: Any,
    *,
    expected: frozenset[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise EnhancedSmokeReceiptError(f"{name} fields are incomplete or unexpected")

    return value


def _stage_hash_payload(stage: Mapping[str, Any]) -> dict[str, Any]:
    return {field: stage[field] for field in _STAGE_HASH_FIELDS}


def build_stage_record(
    *,
    stage_index: int,
    stage_name: str,
    trainer_identity: str,
    input_model_sha256: str,
    output_model_sha256: str,
    signal_consumed: bool,
    signal_evidence_sha256: str,
    previous_stage_sha256: str,
) -> dict[str, Any]:
    """Build one canonical hash-linked stage record without any defaults."""

    record = {
        "stage_index": _nonnegative_int(stage_index, name="stage_index"),
        "stage_name": _exact_text(stage_name, name="stage_name"),
        "trainer_identity": _exact_text(trainer_identity, name="trainer_identity"),
        "input_model_sha256": _sha256_identity(
            input_model_sha256,
            name="input_model_sha256",
        ),
        "output_model_sha256": _sha256_identity(
            output_model_sha256,
            name="output_model_sha256",
        ),
        "signal_consumed": signal_consumed,
        "signal_evidence_sha256": _sha256_identity(
            signal_evidence_sha256,
            name="signal_evidence_sha256",
        ),
        "previous_stage_sha256": _sha256_identity(
            previous_stage_sha256,
            name="previous_stage_sha256",
        ),
    }
    _require_true(record["signal_consumed"], name="signal_consumed")
    record["stage_sha256"] = _sha256_json(record)

    return record


def compute_stage_chain_sha256(stage_chain: Sequence[Mapping[str, Any]]) -> str:
    """Hash the ordered stage-record identities for the receipt root."""

    if isinstance(stage_chain, (str, bytes)) or not isinstance(stage_chain, Sequence):
        raise EnhancedSmokeReceiptError("stage_chain must be a sequence")
    stage_hashes = []
    for index, stage in enumerate(stage_chain):
        _require_exact_fields(stage, expected=_STAGE_FIELDS, name=f"stage_chain[{index}]")
        stage_hashes.append(
            _sha256_identity(stage["stage_sha256"], name=f"stage_chain[{index}].stage_sha256")
        )

    return _sha256_json(stage_hashes)


def _validate_v0(receipt: Mapping[str, Any]) -> str:
    evidence = _require_exact_fields(
        receipt["v0_consumption"],
        expected=_V0_FIELDS,
        name="v0_consumption",
    )
    _require_true(receipt["v0_weights_or_distillation_consumed"], name="V0 consumption")
    _require_true(evidence["consumed"], name="v0_consumption.consumed")
    mode = _exact_text(evidence["mode"], name="v0_consumption.mode")
    if mode not in {"weights", "distillation"}:
        raise EnhancedSmokeReceiptError("V0 mode must be weights or distillation")
    weights_consumed = evidence["weights_consumed"]
    distillation_consumed = evidence["distillation_consumed"]
    if not isinstance(weights_consumed, bool) or not isinstance(distillation_consumed, bool):
        raise EnhancedSmokeReceiptError("V0 route evidence must use explicit booleans")
    expected_route = (mode == "weights", mode == "distillation")
    if (weights_consumed, distillation_consumed) != expected_route:
        raise EnhancedSmokeReceiptError("exactly the declared V0 route must be consumed")
    _sha256_identity(
        evidence["source_checkpoint_sha256"],
        name="v0_consumption.source_checkpoint_sha256",
    )

    return _sha256_identity(evidence["evidence_sha256"], name="v0_consumption.evidence_sha256")


def _validate_replay(receipt: Mapping[str, Any]) -> str:
    evidence = _require_exact_fields(
        receipt["replay_consumption"],
        expected=_REPLAY_FIELDS,
        name="replay_consumption",
    )
    top_rows = _positive_int(
        receipt["unique_replay_rows_consumed"],
        name="unique_replay_rows_consumed",
    )
    nested_rows = _positive_int(
        evidence["unique_rows_consumed"],
        name="replay_consumption.unique_rows_consumed",
    )
    if top_rows != nested_rows:
        raise EnhancedSmokeReceiptError("replay row counts disagree")
    for field in ("row_identity_set_sha256", "source_catalog_sha256"):
        _sha256_identity(evidence[field], name=f"replay_consumption.{field}")

    return _sha256_identity(
        evidence["evidence_sha256"],
        name="replay_consumption.evidence_sha256",
    )


def _validate_branch(receipt: Mapping[str, Any]) -> str:
    evidence = _require_exact_fields(
        receipt["branch_consumption"],
        expected=_BRANCH_FIELDS,
        name="branch_consumption",
    )
    top_rows = _positive_int(
        receipt["unique_branch_rows_consumed"],
        name="unique_branch_rows_consumed",
    )
    top_shards = _positive_int(
        receipt["unique_branch_shards_consumed"],
        name="unique_branch_shards_consumed",
    )
    nested_rows = _positive_int(
        evidence["unique_rows_consumed"],
        name="branch_consumption.unique_rows_consumed",
    )
    nested_shards = _positive_int(
        evidence["unique_shards_consumed"],
        name="branch_consumption.unique_shards_consumed",
    )
    if top_rows != REQUIRED_BRANCH_ROWS or nested_rows != REQUIRED_BRANCH_ROWS:
        raise EnhancedSmokeReceiptError(
            f"branch-Q consumption must contain exactly {REQUIRED_BRANCH_ROWS} unique rows"
        )
    if top_shards != REQUIRED_BRANCH_SHARDS or nested_shards != REQUIRED_BRANCH_SHARDS:
        raise EnhancedSmokeReceiptError(
            f"branch-Q consumption must contain exactly {REQUIRED_BRANCH_SHARDS} unique shards"
        )
    for field in (
        "row_identity_set_sha256",
        "shard_identity_set_sha256",
        "root_manifest_sha256",
    ):
        _sha256_identity(evidence[field], name=f"branch_consumption.{field}")

    return _sha256_identity(
        evidence["evidence_sha256"],
        name="branch_consumption.evidence_sha256",
    )


def _validate_curriculum(receipt: Mapping[str, Any]) -> str:
    evidence = _require_exact_fields(
        receipt["curriculum_consumption"],
        expected=_CURRICULUM_FIELDS,
        name="curriculum_consumption",
    )
    _require_true(
        receipt["opponent_deck_strength_metadata_consumed_by_scheduler"],
        name="opponent deck/strength scheduler consumption",
    )
    for field in (
        "consumed",
        "deck_identity_participated",
        "anchored_strength_bucket_participated",
        "all_strength_buckets_anchored",
    ):
        _require_true(evidence[field], name=f"curriculum_consumption.{field}")
    _positive_finite(
        evidence["effective_distribution_l1_delta_from_base"],
        name="curriculum_consumption.effective_distribution_l1_delta_from_base",
    )
    for field in ("scheduler_config_sha256", "scheduler_receipt_sha256"):
        _sha256_identity(evidence[field], name=f"curriculum_consumption.{field}")

    return _sha256_identity(
        evidence["evidence_sha256"],
        name="curriculum_consumption.evidence_sha256",
    )


def _validate_parameter_updates(
    receipt: Mapping[str, Any],
    *,
    initial_model_sha256: str,
    final_model_sha256: str,
) -> None:
    evidence = _require_exact_fields(
        receipt["parameter_updates"],
        expected=_PARAMETER_UPDATE_FIELDS,
        name="parameter_updates",
    )
    _require_true(receipt["actor_parameters_updated"], name="actor_parameters_updated")
    _require_true(receipt["critic_parameters_updated"], name="critic_parameters_updated")
    _require_true(evidence["actor_updated"], name="parameter_updates.actor_updated")
    _require_true(evidence["critic_updated"], name="parameter_updates.critic_updated")
    _positive_finite(evidence["actor_delta_norm"], name="parameter_updates.actor_delta_norm")
    _positive_finite(evidence["critic_delta_norm"], name="parameter_updates.critic_delta_norm")
    actor_before = _sha256_identity(
        evidence["actor_before_sha256"],
        name="parameter_updates.actor_before_sha256",
    )
    actor_after = _sha256_identity(
        evidence["actor_after_sha256"],
        name="parameter_updates.actor_after_sha256",
    )
    critic_before = _sha256_identity(
        evidence["critic_before_sha256"],
        name="parameter_updates.critic_before_sha256",
    )
    critic_after = _sha256_identity(
        evidence["critic_after_sha256"],
        name="parameter_updates.critic_after_sha256",
    )
    before_model = _sha256_identity(
        evidence["before_model_sha256"],
        name="parameter_updates.before_model_sha256",
    )
    after_model = _sha256_identity(
        evidence["after_model_sha256"],
        name="parameter_updates.after_model_sha256",
    )
    _sha256_identity(evidence["evidence_sha256"], name="parameter_updates.evidence_sha256")
    if actor_before == actor_after or critic_before == critic_after:
        raise EnhancedSmokeReceiptError("actor and critic identities must both change")
    if before_model != initial_model_sha256 or after_model != final_model_sha256:
        raise EnhancedSmokeReceiptError("parameter-update model lineage disagrees with receipt")


def _validate_checkpoint_resume(receipt: Mapping[str, Any]) -> None:
    evidence = _require_exact_fields(
        receipt["checkpoint_resume"],
        expected=_CHECKPOINT_RESUME_FIELDS,
        name="checkpoint_resume",
    )
    _require_true(receipt["checkpoint_resume_parity"], name="checkpoint_resume_parity")
    _require_true(evidence["parity"], name="checkpoint_resume.parity")
    for field in _CHECKPOINT_RESUME_FIELDS - {"parity"}:
        _sha256_identity(evidence[field], name=f"checkpoint_resume.{field}")
    if evidence["saved_state_sha256"] != evidence["restored_state_sha256"]:
        raise EnhancedSmokeReceiptError("saved and restored trainer states differ")
    if evidence["next_step_expected_sha256"] != evidence["next_step_resumed_sha256"]:
        raise EnhancedSmokeReceiptError("checkpoint/resume next-step parity failed")


def _validate_leakage(receipt: Mapping[str, Any]) -> None:
    evidence = _require_exact_fields(
        receipt["leakage_scan"],
        expected=_LEAKAGE_SCAN_FIELDS,
        name="leakage_scan",
    )
    _require_true(
        receipt["zero_hidden_information_leakage"],
        name="zero_hidden_information_leakage",
    )
    _require_true(
        evidence["zero_hidden_information_leakage"],
        name="leakage_scan.zero_hidden_information_leakage",
    )
    violations = _nonnegative_int(
        evidence["violation_count"],
        name="leakage_scan.violation_count",
    )
    if violations != 0:
        raise EnhancedSmokeReceiptError("hidden-information leakage violations must be zero")
    _positive_int(
        evidence["scanned_artifact_count"],
        name="leakage_scan.scanned_artifact_count",
    )
    _require_true(
        evidence["negative_control_rejected"],
        name="leakage_scan.negative_control_rejected",
    )
    _sha256_identity(evidence["evidence_sha256"], name="leakage_scan.evidence_sha256")


def _validate_terminal_ppo(receipt: Mapping[str, Any]) -> str:
    evidence = _require_exact_fields(
        receipt["terminal_ppo"],
        expected=_TERMINAL_PPO_FIELDS,
        name="terminal_ppo",
    )
    _require_true(evidence["consumed"], name="terminal_ppo.consumed")
    _positive_int(evidence["terminal_episodes"], name="terminal_ppo.terminal_episodes")
    _positive_int(evidence["optimizer_updates"], name="terminal_ppo.optimizer_updates")
    for field in ("illegal_actions", "crashes", "timeouts"):
        value = _nonnegative_int(evidence[field], name=f"terminal_ppo.{field}")
        if value != 0:
            raise EnhancedSmokeReceiptError(f"terminal_ppo.{field} must be zero")

    return _sha256_identity(evidence["evidence_sha256"], name="terminal_ppo.evidence_sha256")


def _validate_stage_chain(
    receipt: Mapping[str, Any],
    *,
    trainer_identity: str,
    initial_model_sha256: str,
    final_model_sha256: str,
    signal_evidence_sha256s: Sequence[str],
) -> None:
    _require_true(
        receipt["same_trainer_identity_stage_hash_chain"],
        name="same_trainer_identity_stage_hash_chain",
    )
    stage_chain = receipt["stage_chain"]
    if isinstance(stage_chain, (str, bytes)) or not isinstance(stage_chain, Sequence):
        raise EnhancedSmokeReceiptError("stage_chain must be a sequence")
    if len(stage_chain) != len(REQUIRED_STAGE_NAMES):
        raise EnhancedSmokeReceiptError("stage_chain must contain exactly all five stages")
    previous_stage_sha256 = STAGE_CHAIN_GENESIS_SHA256
    previous_output_sha256 = initial_model_sha256
    for index, (raw_stage, expected_name, expected_evidence) in enumerate(
        zip(stage_chain, REQUIRED_STAGE_NAMES, signal_evidence_sha256s, strict=True)
    ):
        stage = _require_exact_fields(
            raw_stage,
            expected=_STAGE_FIELDS,
            name=f"stage_chain[{index}]",
        )
        if _nonnegative_int(stage["stage_index"], name=f"stage_chain[{index}].stage_index") != index:
            raise EnhancedSmokeReceiptError("stage indexes must be contiguous and ordered")
        if _exact_text(stage["stage_name"], name=f"stage_chain[{index}].stage_name") != expected_name:
            raise EnhancedSmokeReceiptError("stage names must be complete and ordered")
        if (
            _exact_text(
                stage["trainer_identity"],
                name=f"stage_chain[{index}].trainer_identity",
            )
            != trainer_identity
        ):
            raise EnhancedSmokeReceiptError("every stage must use the same trainer identity")
        input_model_sha256 = _sha256_identity(
            stage["input_model_sha256"],
            name=f"stage_chain[{index}].input_model_sha256",
        )
        output_model_sha256 = _sha256_identity(
            stage["output_model_sha256"],
            name=f"stage_chain[{index}].output_model_sha256",
        )
        _require_true(stage["signal_consumed"], name=f"stage_chain[{index}].signal_consumed")
        if (
            _sha256_identity(
                stage["signal_evidence_sha256"],
                name=f"stage_chain[{index}].signal_evidence_sha256",
            )
            != expected_evidence
        ):
            raise EnhancedSmokeReceiptError("stage signal evidence is not bound to its receipt")
        if (
            _sha256_identity(
                stage["previous_stage_sha256"],
                name=f"stage_chain[{index}].previous_stage_sha256",
            )
            != previous_stage_sha256
        ):
            raise EnhancedSmokeReceiptError("stage hash-chain link is broken")
        if input_model_sha256 != previous_output_sha256:
            raise EnhancedSmokeReceiptError("stage model lineage is broken")
        expected_stage_sha256 = _sha256_json(_stage_hash_payload(stage))
        stage_sha256 = _sha256_identity(
            stage["stage_sha256"],
            name=f"stage_chain[{index}].stage_sha256",
        )
        if stage_sha256 != expected_stage_sha256:
            raise EnhancedSmokeReceiptError("stage evidence hash is invalid")
        previous_stage_sha256 = stage_sha256
        previous_output_sha256 = output_model_sha256
    if previous_output_sha256 != final_model_sha256:
        raise EnhancedSmokeReceiptError("final stage model disagrees with receipt")
    expected_chain_sha256 = compute_stage_chain_sha256(stage_chain)
    actual_chain_sha256 = _sha256_identity(
        receipt["stage_chain_sha256"],
        name="stage_chain_sha256",
    )
    if actual_chain_sha256 != expected_chain_sha256:
        raise EnhancedSmokeReceiptError("stage-chain root hash is invalid")


def validate_enhanced_smoke_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and snapshot one complete local enhanced-generation smoke."""

    receipt = _require_exact_fields(
        receipt,
        expected=REQUIRED_RECEIPT_FIELDS,
        name="enhanced smoke receipt",
    )
    if receipt["schema_version"] != SMOKE_RECEIPT_SCHEMA_VERSION:
        raise EnhancedSmokeReceiptError("enhanced smoke receipt schema is incompatible")
    if receipt["classification"] != SMOKE_CLASSIFICATION:
        raise EnhancedSmokeReceiptError("enhanced smoke must remain classified as quarantine")
    if receipt["status"] != "pass":
        raise EnhancedSmokeReceiptError("enhanced smoke status must be pass")
    trainer_identity = _exact_text(receipt["trainer_identity"], name="trainer_identity")
    initial_model_sha256 = _sha256_identity(
        receipt["initial_model_sha256"],
        name="initial_model_sha256",
    )
    final_model_sha256 = _sha256_identity(
        receipt["final_model_sha256"],
        name="final_model_sha256",
    )
    if initial_model_sha256 == final_model_sha256:
        raise EnhancedSmokeReceiptError("the smoke must update the model identity")
    _require_false(receipt["external_launch_performed"], name="external_launch_performed")
    _require_false(receipt["protected_surfaces_mutated"], name="protected_surfaces_mutated")

    v0_evidence_sha256 = _validate_v0(receipt)
    replay_evidence_sha256 = _validate_replay(receipt)
    branch_evidence_sha256 = _validate_branch(receipt)
    curriculum_evidence_sha256 = _validate_curriculum(receipt)
    _validate_parameter_updates(
        receipt,
        initial_model_sha256=initial_model_sha256,
        final_model_sha256=final_model_sha256,
    )
    _validate_checkpoint_resume(receipt)
    _validate_leakage(receipt)
    terminal_ppo_evidence_sha256 = _validate_terminal_ppo(receipt)
    _validate_stage_chain(
        receipt,
        trainer_identity=trainer_identity,
        initial_model_sha256=initial_model_sha256,
        final_model_sha256=final_model_sha256,
        signal_evidence_sha256s=(
            v0_evidence_sha256,
            replay_evidence_sha256,
            branch_evidence_sha256,
            curriculum_evidence_sha256,
            terminal_ppo_evidence_sha256,
        ),
    )
    try:
        snapshot = copy.deepcopy(dict(receipt))
        _canonical_json_bytes(snapshot)
    except (TypeError, ValueError) as exc:
        raise EnhancedSmokeReceiptError("enhanced smoke receipt cannot be snapshotted") from exc

    return snapshot
