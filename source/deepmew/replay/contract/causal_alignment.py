"""Single-owner causal alignment for canonical replay decisions.

No caller may shift an Admission V0 row or infer a neighboring event.  The
Kaggle adapter binds the exact raw episode and applies the documented runner
convention: ``steps[t][seat].action`` answers the select request observed at
``steps[t - 1][seat]`` and produces ``steps[t][seat].observation``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import secrets
import weakref
from dataclasses import (
    dataclass,
    field,
    fields as dataclass_fields,
    is_dataclass,
    replace,
)
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from deepmew.replay.contract.identities import (
    canonical_json_bytes,
    identity,
    sha256_json,
)
from deepmew.replay.contract.kaggle_sources import (
    KaggleReplaySourceContract,
    kaggle_source_contract_from_metadata,
)
from deepmew.replay.contract.schemas import (
    AlignedDecision,
    AlignmentProof,
    SourceEvent,
    ValidationFailure,
)
from deepmew.replay.contract.source_events import (
    compute_observation_asset_id,
    compute_source_event_id,
    project_actor_observation,
    source_event_integrity_failures,
    source_event_from_admission,
)
from deepmew.replay.information_boundary import InformationMode


CAUSAL_ALIGNER_VERSION = "deepmew.causal_decision_aligner.v1"
KAGGLE_ALIGNMENT_RULE_ID = (
    "KAGGLE_SCHEMA1_CABT_1_0_0_MODULE_1_30_2_CAUSAL_ALIGNMENT_V1"
)
REFGAME_ALIGNMENT_RULE_ID = "REFGAME_SAME_EVENT_CAUSAL_ALIGNMENT_V1"
KAGGLE_SOURCE_SCOPE = (
    "kaggle_episode_replay schema=1 environment=cabt "
    "environment_version=1.0.0 module_version=1.30.2"
)
KAGGLE_ANCHOR_SOURCE_ID = "kaggle_ingest"
KAGGLE_ANCHOR_SOURCE_FORMAT = "kaggle_replay"
KAGGLE_ANCHOR_SOURCE_VERSION = "deepmew.replay_action_row.v1"
EXPECTED_ACTION_SCHEMA_VERSION = "deepmew.actor_transition.v1"
ADMISSION_TRANSITION_REGISTRY_SCHEMA_VERSION = (
    "deepmew.admission_transition_registry.v1"
)
RESULTING_TRANSITION_OWNER_SCHEMA_VERSION = (
    "deepmew.resulting_transition_contract_owner.v1"
)
ALIGNED_DECISION_RECEIPT_SCHEMA_VERSION = (
    "deepmew.causal_aligner_owned_decision_receipt.v1"
)
EXPLICIT_MULTI_EVENT_EVIDENCE_SCHEMA_VERSION = (
    "deepmew.explicit_multi_event_source_evidence.v1"
)
REPAIR_ATTEMPT_NOTE = "deterministic_repair_attempted"
REPAIR_SUCCEEDED_NOTE = "deterministic_repair_succeeded"
_ADMISSION_TRANSITION_REGISTRY_SEAL_KEY = secrets.token_bytes(32)
_MAPPING_PROXY_TYPE = type(MappingProxyType({}))
_BUILDER_OWNED_REGISTRIES: weakref.WeakValueDictionary[int, Any] = (
    weakref.WeakValueDictionary()
)
_BUILDER_REGISTRY_RECEIPTS: dict[int, Any] = {}
_ALIGNED_DECISION_SEAL_KEY = secrets.token_bytes(32)
_ALIGNER_OWNED_DECISIONS: weakref.WeakValueDictionary[int, Any] = (
    weakref.WeakValueDictionary()
)
_ALIGNED_DECISION_RECEIPTS: dict[int, Any] = {}

# V1 accepts no caller-invented multi-event convention.  These two entries are
# frozen synthetic fixtures used to prove the generic adapter itself.
EXPLICIT_MULTI_EVENT_RULE_REGISTRY = frozenset({
    (
        "SYNTHETIC_EXPLICIT_MULTI_EVENT_V1",
        "synthetic multi-event source v1",
        "raw_kaggle_episode",
        "kaggle_episode_replay",
        "schema1:cabt1.0.0:module1.30.2",
    ),
    (
        "SYNTHETIC_CONDITIONAL_MULTI_EVENT_V1",
        "synthetic conditional multi-event source v1",
        "raw_kaggle_episode",
        "kaggle_episode_replay",
        "schema1:cabt1.0.0:module1.30.2",
    ),
})
EXPLICIT_MULTI_EVENT_RULE_REGISTRY_SHA256 = sha256_json(
    sorted(EXPLICIT_MULTI_EVENT_RULE_REGISTRY)
)


class CausalAlignmentError(ValueError):
    """Raised for malformed caller input, never for a classifiable source row."""


@dataclass(frozen=True)
class AdmissionTransitionTarget:
    """Immutable minimum evidence for one admitted target decision."""

    admitted_ordinal: int
    local_row: int
    decision_id: str
    source_game_id: str
    actor_seat: int
    global_decision_index: int
    source_sha256: str
    state_before_feature_row: int
    state_before_representation: str
    source_event_id: str
    observation_id: str
    source_row_sha256: str
    transition_sha256: str
    target_commitment_sha256: str


@dataclass(frozen=True)
class AdmissionTransitionRowCommitment:
    """Compact exact commitment for one row in the admitted binding."""

    admitted_ordinal: int
    local_row: int
    source_event_id: str
    observation_id: str | None
    source_row_sha256: str
    transition_sha256: str
    target: AdmissionTransitionTarget | None


@dataclass(frozen=True)
class AdmissionTransitionRegistry:
    """Per-binding cross-row proof registry with a content-addressed identity."""

    admission_identity: str
    binding_id: str
    row_count: int
    valid_target_count: int
    invalid_target_count: int
    duplicate_decision_ids: tuple[str, ...]
    registry_sha256: str
    _row_commitments: tuple[AdmissionTransitionRowCommitment, ...]
    _targets: Mapping[str, AdmissionTransitionTarget]
    _builder_seal: str = field(repr=False, compare=False, default="")

    def target(self, decision_id: str) -> AdmissionTransitionTarget | None:
        return self._targets.get(decision_id)

    def summary(self) -> Mapping[str, Any]:
        return {
            "schema_version": ADMISSION_TRANSITION_REGISTRY_SCHEMA_VERSION,
            "admission_identity": self.admission_identity,
            "binding_id": self.binding_id,
            "row_count": self.row_count,
            "valid_target_count": self.valid_target_count,
            "invalid_target_count": self.invalid_target_count,
            "duplicate_decision_ids": list(self.duplicate_decision_ids),
            "registry_sha256": self.registry_sha256,
        }


@dataclass(frozen=True)
class _AdmissionTransitionRegistryReceipt:
    admission_identity: str
    binding_id: str
    row_count: int
    valid_target_count: int
    invalid_target_count: int
    duplicate_decision_ids: tuple[str, ...]
    registry_sha256: str
    row_commitments: tuple[AdmissionTransitionRowCommitment, ...]
    targets: Mapping[str, AdmissionTransitionTarget]
    target_commitments: Mapping[str, str]
    builder_seal: str


@dataclass(frozen=True)
class _AlignedDecisionReceipt:
    payload_sha256: str
    builder_seal: str


def _receipt_canonical_value(value: Any) -> Any:
    if value is None or type(value) in (str, int, bool):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value

        return {
            "nonfinite_float": (
                "nan"
                if math.isnan(value)
                else "positive_infinity" if value > 0 else "negative_infinity"
            ),
        }
    if isinstance(value, Enum):
        return _receipt_canonical_value(value.value)
    if is_dataclass(value):
        return {
            item.name: _receipt_canonical_value(getattr(value, item.name))
            for item in dataclass_fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _receipt_canonical_value(item)
            for key, item in sorted(
                value.items(),
                key=lambda pair: str(pair[0]),
            )
        }
    if isinstance(value, (list, tuple)):
        return [_receipt_canonical_value(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()

    raise CausalAlignmentError(
        f"aligned-decision receipt cannot encode {type(value).__name__}"
    )


def _aligned_decision_receipt_payload(
    aligned: AlignedDecision,
) -> Mapping[str, Any]:
    return {
        "schema_version": ALIGNED_DECISION_RECEIPT_SCHEMA_VERSION,
        "causal_aligner_version": CAUSAL_ALIGNER_VERSION,
        "aligned_decision": _receipt_canonical_value(aligned),
    }


def _aligned_decision_builder_seal(
    payload: Mapping[str, Any],
    payload_sha256: str,
) -> str:
    return hmac.new(
        _ALIGNED_DECISION_SEAL_KEY,
        canonical_json_bytes({
            "payload": payload,
            "payload_sha256": payload_sha256,
        }),
        hashlib.sha256,
    ).hexdigest()


def _seal_aligned_decision(aligned: AlignedDecision) -> AlignedDecision:
    if type(aligned) is not AlignedDecision:
        raise CausalAlignmentError(
            "canonical aligner can seal only an exact AlignedDecision"
        )
    payload = _aligned_decision_receipt_payload(aligned)
    payload_sha256 = sha256_json(payload)
    receipt = _AlignedDecisionReceipt(
        payload_sha256=payload_sha256,
        builder_seal=_aligned_decision_builder_seal(payload, payload_sha256),
    )
    aligned_id = id(aligned)
    _ALIGNER_OWNED_DECISIONS[aligned_id] = aligned
    _ALIGNED_DECISION_RECEIPTS[aligned_id] = receipt
    weakref.finalize(
        aligned,
        _ALIGNED_DECISION_RECEIPTS.pop,
        aligned_id,
        None,
    )

    return aligned


def aligned_decision_integrity_failures(
    aligned: Any,
) -> tuple[str, ...]:
    """Replay the aligner-owned payload and verify its process-local receipt."""

    if type(aligned) is not AlignedDecision:
        return ("aligned_decision_type_invalid",)
    receipt = _ALIGNED_DECISION_RECEIPTS.get(id(aligned))
    if (
        _ALIGNER_OWNED_DECISIONS.get(id(aligned)) is not aligned
        or not isinstance(receipt, _AlignedDecisionReceipt)
    ):
        return ("aligned_decision_builder_instance_invalid",)
    try:
        payload = _aligned_decision_receipt_payload(aligned)
        payload_sha256 = sha256_json(payload)
        expected_seal = _aligned_decision_builder_seal(
            payload,
            payload_sha256,
        )
    except (CausalAlignmentError, TypeError, ValueError):
        return ("aligned_decision_payload_replay_invalid",)
    failures = []
    if payload_sha256 != receipt.payload_sha256:
        failures.append("aligned_decision_payload_sha256_invalid")
    if not hmac.compare_digest(expected_seal, receipt.builder_seal):
        failures.append("aligned_decision_builder_seal_invalid")

    return tuple(failures)


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None

    return value


def _selected_values(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, list):
        return None
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        return None

    return tuple(value)


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    normalized = float(value)
    if not math.isfinite(normalized):
        return None

    return normalized


def _strict_next_state(value: Any) -> tuple[str, int, str] | None:
    if not isinstance(value, Mapping) or set(value) != {
        "decision_id",
        "feature_row",
        "representation",
    }:
        return None
    decision_id = value.get("decision_id")
    feature_row = _integer(value.get("feature_row"))
    representation = value.get("representation")
    if (
        not isinstance(decision_id, str)
        or not decision_id
        or feature_row is None
        or feature_row < 0
        or not isinstance(representation, str)
        or not representation
    ):
        return None

    return decision_id, feature_row, representation


def _transition_target_payload(
    target: AdmissionTransitionTarget,
    *,
    include_commitment: bool,
) -> Mapping[str, Any]:
    payload = {
        "admitted_ordinal": target.admitted_ordinal,
        "local_row": target.local_row,
        "decision_id": target.decision_id,
        "source_game_id": target.source_game_id,
        "actor_seat": target.actor_seat,
        "global_decision_index": target.global_decision_index,
        "source_sha256": target.source_sha256,
        "state_before_feature_row": target.state_before_feature_row,
        "state_before_representation": target.state_before_representation,
        "source_event_id": target.source_event_id,
        "observation_id": target.observation_id,
        "source_row_sha256": target.source_row_sha256,
        "transition_sha256": target.transition_sha256,
    }
    if include_commitment:
        payload["target_commitment_sha256"] = (
            target.target_commitment_sha256
        )

    return payload


def _seal_transition_target(
    target: AdmissionTransitionTarget,
) -> AdmissionTransitionTarget:
    commitment = sha256_json(
        _transition_target_payload(target, include_commitment=False)
    )

    return replace(target, target_commitment_sha256=commitment)


def _registry_target(
    event: SourceEvent,
    *,
    admitted_ordinal: int,
    local_row: int,
) -> AdmissionTransitionTarget | None:
    transition = event.transition
    decision_id = transition.get("decision_id")
    source_game_id = transition.get("canonical_game_id")
    actor = _integer(transition.get("actor"))
    global_index = _integer(transition.get("global_decision_index"))
    source_sha256 = transition.get("source_hash")
    state_before = transition.get("state_before")
    feature_row = (
        _integer(state_before.get("feature_row"))
        if isinstance(state_before, Mapping)
        else None
    )
    representation = (
        state_before.get("representation")
        if isinstance(state_before, Mapping)
        else None
    )
    asset = event.observation_asset
    if (
        source_event_integrity_failures(event)
        or not isinstance(decision_id, str)
        or not decision_id
        or not isinstance(source_game_id, str)
        or not source_game_id
        or source_game_id != event.source_game_id
        or actor not in (0, 1)
        or actor != event.actor_seat
        or global_index is None
        or global_index < 0
        or not _is_sha256(source_sha256)
        or asset is None
        or asset.source_sha256 != source_sha256
        or (
            event.source_format == "refgame"
            and event.source_artifact_sha256 != source_sha256
        )
        or feature_row is None
        or feature_row < 0
        or not isinstance(representation, str)
        or not representation
        or not event.event_id
        or not asset.observation_id
        or not _is_sha256(event.source_row_sha256)
    ):
        return None

    return _seal_transition_target(AdmissionTransitionTarget(
        admitted_ordinal=admitted_ordinal,
        local_row=local_row,
        decision_id=decision_id,
        source_game_id=source_game_id,
        actor_seat=actor,
        global_decision_index=global_index,
        source_sha256=source_sha256,
        state_before_feature_row=feature_row,
        state_before_representation=representation,
        source_event_id=event.event_id,
        observation_id=asset.observation_id,
        source_row_sha256=event.source_row_sha256,
        transition_sha256=sha256_json(transition),
        target_commitment_sha256="",
    ))


def _registry_row_payload(
    row: AdmissionTransitionRowCommitment,
) -> Mapping[str, Any]:
    return {
        "admitted_ordinal": row.admitted_ordinal,
        "local_row": row.local_row,
        "source_event_id": row.source_event_id,
        "observation_id": row.observation_id,
        "source_row_sha256": row.source_row_sha256,
        "transition_sha256": row.transition_sha256,
        "target": (
            _transition_target_payload(row.target, include_commitment=True)
            if row.target is not None
            else None
        ),
    }


def _derive_registry_targets(
    rows: Sequence[AdmissionTransitionRowCommitment],
) -> tuple[dict[str, AdmissionTransitionTarget], tuple[str, ...], int]:
    targets: dict[str, AdmissionTransitionTarget] = {}
    duplicate_ids: set[str] = set()
    invalid = 0
    for row in rows:
        target = row.target
        if target is None:
            invalid += 1
            continue
        if target.decision_id in duplicate_ids:
            invalid += 1
            continue
        if target.decision_id in targets:
            targets.pop(target.decision_id)
            duplicate_ids.add(target.decision_id)
            invalid += 2
            continue
        targets[target.decision_id] = target

    return targets, tuple(sorted(duplicate_ids)), invalid


def _registry_body(
    *,
    admission_identity: str,
    binding_id: str,
    rows: Sequence[AdmissionTransitionRowCommitment],
) -> Mapping[str, Any]:
    targets, duplicate_ids, invalid = _derive_registry_targets(rows)

    return {
        "schema_version": ADMISSION_TRANSITION_REGISTRY_SCHEMA_VERSION,
        "admission_identity": admission_identity,
        "binding_id": binding_id,
        "row_count": len(rows),
        "valid_target_count": len(targets),
        "invalid_target_count": invalid,
        "duplicate_decision_ids": list(duplicate_ids),
        "rows": [_registry_row_payload(row) for row in rows],
    }


def _registry_builder_seal(
    body: Mapping[str, Any],
    registry_sha256: str,
) -> str:
    sealed_payload = canonical_json_bytes({
        "body": body,
        "registry_sha256": registry_sha256,
    })

    return hmac.new(
        _ADMISSION_TRANSITION_REGISTRY_SEAL_KEY,
        sealed_payload,
        hashlib.sha256,
    ).hexdigest()


def _register_builder_owned_registry(
    registry: AdmissionTransitionRegistry,
) -> None:
    registry_id = id(registry)
    receipt = _AdmissionTransitionRegistryReceipt(
        admission_identity=registry.admission_identity,
        binding_id=registry.binding_id,
        row_count=registry.row_count,
        valid_target_count=registry.valid_target_count,
        invalid_target_count=registry.invalid_target_count,
        duplicate_decision_ids=registry.duplicate_decision_ids,
        registry_sha256=registry.registry_sha256,
        row_commitments=registry._row_commitments,
        targets=registry._targets,
        target_commitments=MappingProxyType({
            decision_id: target.target_commitment_sha256
            for decision_id, target in registry._targets.items()
        }),
        builder_seal=registry._builder_seal,
    )
    _BUILDER_OWNED_REGISTRIES[registry_id] = registry
    _BUILDER_REGISTRY_RECEIPTS[registry_id] = receipt
    weakref.finalize(
        registry,
        _BUILDER_REGISTRY_RECEIPTS.pop,
        registry_id,
        None,
    )


def _registry_matches_builder_receipt(
    registry: AdmissionTransitionRegistry,
    receipt: _AdmissionTransitionRegistryReceipt,
) -> bool:
    return bool(
        registry.admission_identity == receipt.admission_identity
        and registry.binding_id == receipt.binding_id
        and registry.row_count == receipt.row_count
        and registry.valid_target_count == receipt.valid_target_count
        and registry.invalid_target_count == receipt.invalid_target_count
        and registry.duplicate_decision_ids == receipt.duplicate_decision_ids
        and registry.registry_sha256 == receipt.registry_sha256
        and registry._row_commitments is receipt.row_commitments
        and registry._targets is receipt.targets
        and registry._builder_seal == receipt.builder_seal
    )


def _registry_structural_integrity_failures(
    registry: Any,
) -> tuple[str, ...]:
    """Recompute the full deterministic registry and target commitments."""

    if type(registry) is not AdmissionTransitionRegistry:
        return ("registry_type_invalid",)
    failures: list[str] = []
    try:
        rows = registry._row_commitments
        if type(rows) is not tuple:
            return ("registry_rows_not_immutable_tuple",)
        if not isinstance(registry.admission_identity, str):
            failures.append("registry_admission_identity_invalid")
        if not isinstance(registry.binding_id, str) or not registry.binding_id:
            failures.append("registry_binding_id_invalid")
        previous_local_row = -1
        for expected_ordinal, row in enumerate(rows):
            if type(row) is not AdmissionTransitionRowCommitment:
                failures.append("registry_row_type_invalid")
                continue
            if row.admitted_ordinal != expected_ordinal:
                failures.append("registry_admitted_ordinal_invalid")
            if (
                _integer(row.local_row) is None
                or row.local_row < 0
                or row.local_row <= previous_local_row
            ):
                failures.append("registry_local_rows_not_strictly_increasing")
            previous_local_row = row.local_row
            if not isinstance(row.source_event_id, str):
                failures.append("registry_source_event_id_invalid")
            if row.observation_id is not None and not isinstance(
                row.observation_id,
                str,
            ):
                failures.append("registry_observation_id_invalid")
            if not isinstance(row.source_row_sha256, str):
                failures.append("registry_source_row_sha256_invalid")
            if not _is_sha256(row.transition_sha256):
                failures.append("registry_transition_sha256_invalid")
            target = row.target
            if target is None:
                continue
            if type(target) is not AdmissionTransitionTarget:
                failures.append("registry_target_type_invalid")
                continue
            if (
                target.admitted_ordinal != row.admitted_ordinal
                or target.local_row != row.local_row
                or target.source_event_id != row.source_event_id
                or target.observation_id != row.observation_id
                or target.source_row_sha256 != row.source_row_sha256
                or target.transition_sha256 != row.transition_sha256
            ):
                failures.append("registry_target_row_binding_invalid")
            expected_target_commitment = sha256_json(
                _transition_target_payload(
                    target,
                    include_commitment=False,
                )
            )
            if target.target_commitment_sha256 != expected_target_commitment:
                failures.append("registry_target_commitment_invalid")
        derived_targets, duplicate_ids, invalid = _derive_registry_targets(rows)
        body = _registry_body(
            admission_identity=registry.admission_identity,
            binding_id=registry.binding_id,
            rows=rows,
        )
        expected_registry_sha256 = sha256_json(body)
        if registry.row_count != len(rows):
            failures.append("registry_row_count_invalid")
        if registry.valid_target_count != len(derived_targets):
            failures.append("registry_valid_target_count_invalid")
        if registry.invalid_target_count != invalid:
            failures.append("registry_invalid_target_count_invalid")
        if registry.duplicate_decision_ids != duplicate_ids:
            failures.append("registry_duplicate_decision_ids_invalid")
        if registry.registry_sha256 != expected_registry_sha256:
            failures.append("registry_sha256_invalid")
        if type(registry._targets) is not _MAPPING_PROXY_TYPE:
            failures.append("registry_targets_not_immutable_mapping")
        elif (
            set(registry._targets) != set(derived_targets)
            or any(
                registry._targets[key] != target
                for key, target in derived_targets.items()
            )
        ):
            failures.append("registry_target_map_invalid")
        expected_builder_seal = _registry_builder_seal(
            body,
            expected_registry_sha256,
        )
        if (
            not isinstance(registry._builder_seal, str)
            or not hmac.compare_digest(
                registry._builder_seal,
                expected_builder_seal,
            )
        ):
            failures.append("registry_builder_seal_invalid")
    except (AttributeError, KeyError, TypeError, ValueError):
        failures.append("registry_commitment_structure_invalid")

    return tuple(dict.fromkeys(failures))


def admission_transition_registry_integrity_failures(
    registry: Any,
) -> tuple[str, ...]:
    """Verify the builder receipt before any proof lookup uses the registry."""

    if type(registry) is not AdmissionTransitionRegistry:
        return ("registry_type_invalid",)
    receipt = _BUILDER_REGISTRY_RECEIPTS.get(id(registry))
    builder_owned = (
        _BUILDER_OWNED_REGISTRIES.get(id(registry)) is registry
        and isinstance(receipt, _AdmissionTransitionRegistryReceipt)
    )
    if (
        builder_owned
        and receipt is not None
        and _registry_matches_builder_receipt(registry, receipt)
    ):
        return ()
    failures = [
        (
            "registry_builder_receipt_invalid"
            if builder_owned
            else "registry_builder_instance_invalid"
        )
    ]
    failures.extend(_registry_structural_integrity_failures(registry))

    return tuple(dict.fromkeys(failures))


def _registry_target_integrity_failures(
    registry: AdmissionTransitionRegistry,
    target: AdmissionTransitionTarget,
) -> tuple[str, ...]:
    receipt = _BUILDER_REGISTRY_RECEIPTS.get(id(registry))
    if not isinstance(receipt, _AdmissionTransitionRegistryReceipt):
        return ("registry_target_receipt_missing",)
    failures: list[str] = []
    expected_commitment = sha256_json(
        _transition_target_payload(target, include_commitment=False)
    )
    if target.target_commitment_sha256 != expected_commitment:
        failures.append("registry_target_commitment_invalid")
    if receipt.target_commitments.get(target.decision_id) != expected_commitment:
        failures.append("registry_target_receipt_commitment_invalid")
    ordinal = _integer(target.admitted_ordinal)
    if ordinal is None or ordinal < 0 or ordinal >= len(registry._row_commitments):
        failures.append("registry_target_ordinal_invalid")
    else:
        row = registry._row_commitments[ordinal]
        if (
            row.target is not target
            or row.admitted_ordinal != target.admitted_ordinal
            or row.local_row != target.local_row
            or row.source_event_id != target.source_event_id
            or row.observation_id != target.observation_id
            or row.source_row_sha256 != target.source_row_sha256
            or row.transition_sha256 != target.transition_sha256
        ):
            failures.append("registry_target_row_binding_invalid")

    return tuple(dict.fromkeys(failures))


def _registry_row_for_local_row(
    registry: AdmissionTransitionRegistry,
    local_row: int,
) -> AdmissionTransitionRowCommitment | None:
    rows = registry._row_commitments
    lower = 0
    upper = len(rows)
    while lower < upper:
        middle = (lower + upper) // 2
        candidate = rows[middle].local_row
        if candidate < local_row:
            lower = middle + 1
        else:
            upper = middle
    if lower >= len(rows) or rows[lower].local_row != local_row:
        return None

    return rows[lower]


def _registry_current_target(
    event: SourceEvent,
    registry: AdmissionTransitionRegistry,
) -> AdmissionTransitionTarget | None:
    local_row = _integer(event.local_row)
    transition = event.transition
    decision_id = transition.get("decision_id")
    source_game_id = transition.get("canonical_game_id")
    actor_seat = _integer(transition.get("actor"))
    global_index = _integer(transition.get("global_decision_index"))
    source_sha256 = transition.get("source_hash")
    state_before = transition.get("state_before")
    feature_row = (
        _integer(state_before.get("feature_row"))
        if isinstance(state_before, Mapping)
        else None
    )
    representation = (
        state_before.get("representation")
        if isinstance(state_before, Mapping)
        else None
    )
    observation_id = (
        event.observation_asset.observation_id
        if event.observation_asset is not None
        else None
    )
    if (
        local_row is None
        or local_row < 0
        or source_event_integrity_failures(event)
    ):
        return None
    row = _registry_row_for_local_row(registry, local_row)
    if row is None or row.target is None:
        return None
    target = row.target
    if (
        _registry_target_integrity_failures(registry, target)
        or row.admitted_ordinal != target.admitted_ordinal
        or row.local_row != event.local_row
        or row.source_event_id != event.event_id
        or row.observation_id != observation_id
        or row.source_row_sha256 != event.source_row_sha256
        or row.transition_sha256 != sha256_json(transition)
        or target.decision_id != decision_id
        or target.source_game_id != event.source_game_id
        or target.source_game_id != source_game_id
        or target.actor_seat != event.actor_seat
        or target.actor_seat != actor_seat
        or target.global_decision_index != global_index
        or target.source_sha256 != source_sha256
        or target.state_before_feature_row != feature_row
        or target.state_before_representation != representation
        or target.source_event_id != event.event_id
        or target.observation_id != observation_id
        or target.source_row_sha256 != event.source_row_sha256
        or target.transition_sha256 != sha256_json(transition)
    ):
        return None

    return target


def build_admission_transition_registry(
    events: Iterable[SourceEvent],
    *,
    admission_identity: str,
    binding_id: str,
    expected_rows: int | None = None,
) -> AdmissionTransitionRegistry:
    """Build the immutable target registry from one admitted binding."""

    if (
        expected_rows is not None
        and (_integer(expected_rows) is None or expected_rows < 0)
    ):
        raise CausalAlignmentError("registry expected_rows is invalid")
    rows: list[AdmissionTransitionRowCommitment] = []
    previous_local_row = -1
    for admitted_ordinal, event in enumerate(events):
        if (
            event.admission_identity != admission_identity
            or event.binding_id != binding_id
        ):
            raise CausalAlignmentError(
                "refgame registry event differs from its admission binding"
            )
        local_row = _integer(event.local_row)
        if (
            local_row is None
            or local_row < 0
            or local_row <= previous_local_row
        ):
            raise CausalAlignmentError(
                "registry local rows must be unique and strictly increasing"
            )
        previous_local_row = local_row
        transition_sha256 = sha256_json(event.transition)
        observation_id = (
            event.observation_asset.observation_id
            if event.observation_asset is not None
            else None
        )
        rows.append(AdmissionTransitionRowCommitment(
            admitted_ordinal=admitted_ordinal,
            local_row=local_row,
            source_event_id=event.event_id,
            observation_id=observation_id,
            source_row_sha256=event.source_row_sha256,
            transition_sha256=transition_sha256,
            target=_registry_target(
                event,
                admitted_ordinal=admitted_ordinal,
                local_row=local_row,
            ),
        ))
    if expected_rows is not None and len(rows) != expected_rows:
        raise CausalAlignmentError(
            "refgame registry does not cover the exact admitted binding"
        )
    targets, duplicate_ids, invalid = _derive_registry_targets(rows)
    body = _registry_body(
        admission_identity=admission_identity,
        binding_id=binding_id,
        rows=rows,
    )
    registry_sha256 = sha256_json(body)
    builder_seal = _registry_builder_seal(body, registry_sha256)

    registry = AdmissionTransitionRegistry(
        admission_identity=admission_identity,
        binding_id=binding_id,
        row_count=len(rows),
        valid_target_count=len(targets),
        invalid_target_count=invalid,
        duplicate_decision_ids=duplicate_ids,
        registry_sha256=registry_sha256,
        _row_commitments=tuple(rows),
        _targets=MappingProxyType(dict(targets)),
        _builder_seal=builder_seal,
    )
    if _registry_structural_integrity_failures(registry):
        raise CausalAlignmentError("builder produced an invalid transition registry")
    _register_builder_owned_registry(registry)

    return registry


def _resulting_transition_id(
    *,
    event: SourceEvent,
    admission_raw_sha256: str,
    result_raw_sha256: str,
    pre_action_event_id: str,
    action_event_ids: Sequence[str],
    result_event_id: str,
    pre_action_observation_id: str,
    result_observation_id: str | None,
    result_status: Any,
    result_reward: Any,
    result_outcome: Any,
    terminal: bool,
    next_same_actor_state: Any,
    transition_horizon_logical_decisions: int | None,
    rule_id: str,
    registry_sha256: str | None = None,
    registry_current_target: AdmissionTransitionTarget | None = None,
    registry_target: AdmissionTransitionTarget | None = None,
) -> str:
    if not _is_sha256(admission_raw_sha256) or not _is_sha256(result_raw_sha256):
        raise CausalAlignmentError("resulting transition source hashes are invalid")
    if registry_sha256 is not None and not _is_sha256(registry_sha256):
        raise CausalAlignmentError("resulting transition registry hash is invalid")
    if (registry_current_target is None) != (registry_target is None):
        raise CausalAlignmentError(
            "resulting transition registry link is incomplete"
        )
    for bound_target in (registry_current_target, registry_target):
        if bound_target is None:
            continue
        expected_target_commitment = sha256_json(
            _transition_target_payload(
                bound_target,
                include_commitment=False,
            )
        )
        if (
            registry_sha256 is None
            or not _is_sha256(bound_target.target_commitment_sha256)
            or bound_target.target_commitment_sha256
            != expected_target_commitment
        ):
            raise CausalAlignmentError(
                "resulting transition registry target is invalid"
            )
    payload = {
        "schema_version": RESULTING_TRANSITION_OWNER_SCHEMA_VERSION,
        "admission_identity": event.admission_identity,
        "admission_raw_sha256": admission_raw_sha256,
        "result_raw_sha256": result_raw_sha256,
        "source_id": event.source_id,
        "source_format": event.source_format,
        "source_version": event.source_version,
        "source_game_id": event.source_game_id,
        "actor_seat": event.actor_seat,
        "pre_action_event_id": pre_action_event_id,
        "action_event_ids": list(action_event_ids),
        "result_event_id": result_event_id,
        "pre_action_observation_id": pre_action_observation_id,
        "result_observation_id": result_observation_id,
        "result_status": result_status,
        "result_reward": result_reward,
        "result_outcome": result_outcome,
        "terminal": terminal,
        "next_same_actor_state": next_same_actor_state,
        "transition_horizon_logical_decisions": transition_horizon_logical_decisions,
        "causal_aligner_rule_id": rule_id,
        "causal_aligner_version": CAUSAL_ALIGNER_VERSION,
        "admission_transition_registry_sha256": registry_sha256,
        "admission_transition_registry_current_target": (
            _transition_target_payload(
                registry_current_target,
                include_commitment=True,
            )
            if registry_current_target is not None
            else None
        ),
        "admission_transition_registry_next_target": (
            _transition_target_payload(
                registry_target,
                include_commitment=True,
            )
            if registry_target is not None
            else None
        ),
    }

    return identity("transition1", payload)


def explicit_multi_event_chain_id(evidence_without_chain_id: Mapping[str, Any]) -> str:
    """Return the only accepted content identity for explicit-chain evidence."""

    return sha256_json(evidence_without_chain_id)


def _explicit_multi_event_evidence(
    raw_evidence: Mapping[str, Any] | None,
    *,
    pre_action_event: SourceEvent,
    action_events: Sequence[SourceEvent],
    resulting_event: SourceEvent,
    rule_id: str,
    source_scope: str,
) -> tuple[bool, str | None]:
    if not isinstance(raw_evidence, Mapping):
        return False, None
    events = (pre_action_event, *action_events, resulting_event)
    expected_without_chain = {
        "schema_version": EXPLICIT_MULTI_EVENT_EVIDENCE_SCHEMA_VERSION,
        "rule_id": rule_id,
        "source_scope": source_scope,
        "source_game_id": pre_action_event.source_game_id,
        "actor_seat": pre_action_event.actor_seat,
        "stage_count": len(action_events),
        "event_positions": [event.source_event_position for event in events],
        "request_event_ids": [
            event.event_id
            for event in (pre_action_event, *action_events[:-1])
        ],
        "action_event_ids": [event.event_id for event in action_events],
        "resulting_event_id": resulting_event.event_id,
    }
    chain_id = explicit_multi_event_chain_id(expected_without_chain)
    expected = {**expected_without_chain, "chain_id": chain_id}
    if dict(raw_evidence) != expected:
        return False, None

    return True, str(chain_id)


def _unique_failures(values: Sequence[ValidationFailure]) -> tuple[ValidationFailure, ...]:
    return tuple(dict.fromkeys(values))


def _causal_index(event: SourceEvent, fallback: int) -> int:
    value = _integer(event.transition.get("global_decision_index"))

    return value if value is not None else fallback


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()


@dataclass(frozen=True)
class BoundKaggleEpisode:
    path: Path
    sha256: str
    data: Mapping[str, Any]


class KaggleEpisodeRepository:
    """Hash-bind raw episodes while retaining at most one decoded episode."""

    def __init__(self, repository_root: str | Path) -> None:
        self.repository_root = Path(repository_root).resolve()
        self._last: BoundKaggleEpisode | None = None
        self._observation_hashes: dict[tuple[int, int], str] = {}

    def resolve(self, source_path: str) -> Path:
        path = Path(source_path)
        if not path.is_absolute():
            path = self.repository_root / path

        return path.resolve()

    def load(self, source_path: str) -> BoundKaggleEpisode:
        path = self.resolve(source_path)
        if self._last is not None and self._last.path == path:
            return self._last
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = _sha256_path(path)
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, Mapping):
            raise CausalAlignmentError("Kaggle episode root must be an object")
        bound = BoundKaggleEpisode(path=path, sha256=digest, data=data)
        self._last = bound
        self._observation_hashes = {}

        return bound

    def observation_hash(self, step: int, seat: int) -> str | None:
        return self._observation_hashes.get((step, seat))

    def register_observation_hash(self, step: int, seat: int, content_sha256: str) -> None:
        key = (step, seat)
        prior = self._observation_hashes.get(key)
        if prior is not None and prior != content_sha256:
            raise CausalAlignmentError("one raw Kaggle observation has conflicting hashes")
        self._observation_hashes[key] = content_sha256


def _project_observation(value: Any) -> tuple[Mapping[str, Any] | None, ValidationFailure | None]:
    try:
        projected = project_actor_observation(value)
    except ValueError:
        return None, ValidationFailure.PRIVILEGED_OBSERVATION

    return projected, None


def _raw_kaggle_event(
    anchor: SourceEvent,
    *,
    episode: BoundKaggleEpisode,
    step: int,
    seat: int,
    agent: Mapping[str, Any],
    role: str,
    source_contract: KaggleReplaySourceContract,
    source_content_sha256: str | None = None,
) -> tuple[SourceEvent | None, ValidationFailure | None]:
    source_observation = agent.get("observation")
    projected, failure = _project_observation(source_observation)
    if projected is None:
        return None, failure
    identity_observation = (
        projected
        if source_contract.actor_visible_content_identity
        else source_observation
    )
    raw_content_sha256 = source_content_sha256 or sha256_json(identity_observation)
    select = projected.get("select")
    legal = select.get("option") if isinstance(select, Mapping) else None
    selected = _selected_values(agent.get("action"))
    raw_record = dict(agent)
    raw_record["observation"] = identity_observation
    raw_record.update({
        "schema_version": source_contract.source_schema,
        "source_kind": "kaggle_episode_replay",
        "source_path": str(episode.path),
        "source_artifact_sha256": episode.sha256,
        "game_id": anchor.source_native_game_id,
        "player_id": seat,
        "step": step,
    })
    raw_transition: dict[str, Any] = {
        "schema_version": EXPECTED_ACTION_SCHEMA_VERSION,
        "source_schema": source_contract.source_schema,
        "source_id": anchor.source_id,
        "source_game_id": anchor.source_native_game_id,
        "canonical_game_id": anchor.source_game_id,
        "source_hash": episode.sha256,
        "actor": seat,
        "global_decision_index": step,
        "decision_id": (
            f"{anchor.source_game_id}:raw_step_{step:06d}:s{seat}:{role}"
        ),
        "state_before": {
            "feature_row": step,
            "representation": "raw_kaggle_actor_observation.v1",
            "source_observation": {
                "source_sha256": episode.sha256,
                "source_path": str(episode.path),
                "source_line": step,
                "content_sha256": raw_content_sha256,
                "source_field": "raw_episode_agent_observation",
            },
        },
        "information_mode": InformationMode.ACTOR_VISIBLE.value,
        "terminal": anchor.transition.get("terminal") if role == "result" else False,
        "terminal_actor_relative_result": anchor.transition.get(
            "terminal_actor_relative_result"
        ),
        "next_same_actor_state": (
            anchor.transition.get("next_same_actor_state")
            if role == "result"
            else None
        ),
        "transition_horizon_logical_decisions": (
            anchor.transition.get("transition_horizon_logical_decisions")
            if role == "result"
            else None
        ),
    }
    if selected is not None:
        raw_transition["chosen_action"] = {
            "indices": list(selected),
            "index_space": "option_position",
        }
    if isinstance(legal, list) and all(isinstance(option, Mapping) for option in legal):
        raw_transition["legal_action_set"] = [dict(option) for option in legal]
    raw_event = source_event_from_admission(
        admission_identity=anchor.admission_identity,
        binding_id=anchor.binding_id,
        local_row=anchor.local_row,
        global_row=anchor.global_row,
        transition=raw_transition,
        source_record=raw_record,
        verify_source_content=True,
    )
    raw_asset = None
    if raw_event.observation_asset is not None:
        raw_asset = replace(
            raw_event.observation_asset,
            asset_origin="raw_kaggle_replay_repair_v1",
            reusable_admission_asset=False,
            observation_id="",
        )
        raw_asset = replace(
            raw_asset,
            observation_id=compute_observation_asset_id(raw_asset),
        )
    raw_event = replace(
        raw_event,
        source_format="kaggle_episode_replay",
        observation_asset=raw_asset,
        source_corrupt=raw_event.source_corrupt or anchor.source_corrupt,
        event_id="",
    )
    raw_event = replace(raw_event, event_id=compute_source_event_id(raw_event))
    if source_event_integrity_failures(raw_event):
        return None, ValidationFailure.SOURCE_CORRUPT

    return raw_event, None


def _failed_alignment(
    event: SourceEvent,
    *,
    rule_id: str,
    source_scope: str,
    failures: Sequence[ValidationFailure],
    notes: Sequence[str] = (),
) -> AlignedDecision:
    proof = AlignmentProof(
        rule_id=rule_id,
        source_scope=source_scope,
        pre_action_event_id=event.event_id,
        action_event_ids=(event.event_id,),
        resulting_event_id=None,
        pre_action_observation_id=event.observation_asset.observation_id if event.observation_asset else None,
        legal_observation_id=None,
        resulting_observation_id=None,
        action_equality_proven=False,
        actor_seat_proven=False,
        event_order_proven=False,
        resulting_transition_proven=False,
        source_hash_bound=False,
        deterministic_repair_attempted=REPAIR_ATTEMPT_NOTE in notes,
        deterministic_repair_applied=False,
        notes=tuple(notes),
    )

    return _seal_aligned_decision(AlignedDecision(
        pre_action_event=event,
        action_events=(event,),
        resulting_event=None,
        stage_selected_choices=(event.stored_selected_choices,),
        stage_index_spaces=(event.stored_index_space,),
        causal_decision_index=_causal_index(event, event.source_event_position),
        proof=proof,
        failures=_unique_failures(failures),
    ))


def _kaggle_anchor_failures(event: SourceEvent) -> tuple[ValidationFailure, ...]:
    failures: list[ValidationFailure] = []
    if source_event_integrity_failures(event):
        failures.append(ValidationFailure.SOURCE_CORRUPT)
    if (
        event.source_id != KAGGLE_ANCHOR_SOURCE_ID
        or event.source_format != KAGGLE_ANCHOR_SOURCE_FORMAT
        or event.source_version != KAGGLE_ANCHOR_SOURCE_VERSION
        or event.action_schema_version != EXPECTED_ACTION_SCHEMA_VERSION
        or event.observation_schema_version != KAGGLE_ANCHOR_SOURCE_VERSION
    ):
        failures.append(ValidationFailure.SCHEMA_MISMATCH)
    if event.source_corrupt or event.observation_asset is None or event.observation is None:
        failures.append(ValidationFailure.SOURCE_CORRUPT)
    if (
        event.information_mode != InformationMode.ACTOR_VISIBLE.value
        or (
            event.observation_asset is not None
            and event.observation_asset.information_mode
            != InformationMode.ACTOR_VISIBLE.value
        )
    ):
        failures.append(ValidationFailure.PRIVILEGED_OBSERVATION)
    if (
        event.actor_seat not in (0, 1)
        or event.controlled_player != event.actor_seat
    ):
        failures.append(ValidationFailure.SEAT_IDENTITY)
    if (
        event.stored_selected_choices is None
        or event.stored_index_space != "option_position"
        or not _is_sha256(event.source_row_sha256)
        or event.observation_asset is None
        or not _is_sha256(event.observation_asset.source_sha256)
        or not _is_sha256(event.observation_asset.source_content_sha256)
    ):
        failures.append(ValidationFailure.SOURCE_CORRUPT)
    record = event.source_record
    if (
        record.get("schema_version") != KAGGLE_ANCHOR_SOURCE_VERSION
        or record.get("source_kind") != "kaggle_replay"
        or record.get("game_id") != event.source_native_game_id
        or _integer(record.get("player_id")) != event.actor_seat
        or _integer(record.get("step")) != event.source_event_position
        or not isinstance(record.get("source_path"), str)
        or not record.get("source_path")
    ):
        failures.append(ValidationFailure.SOURCE_CORRUPT)

    return _unique_failures(failures)


def _kaggle_episode_final_outcome(
    data: Mapping[str, Any],
    *,
    seat: int,
) -> tuple[str, float, float] | None:
    steps = data.get("steps")
    statuses = data.get("statuses")
    rewards = data.get("rewards")
    if (
        not isinstance(steps, list)
        or not steps
        or not isinstance(statuses, list)
        or len(statuses) != 2
        or not isinstance(rewards, list)
        or len(rewards) != 2
        or seat not in (0, 1)
        or not isinstance(steps[-1], list)
        or len(steps[-1]) != 2
        or not isinstance(steps[-1][seat], Mapping)
    ):
        return None
    final_status = statuses[seat]
    final_reward = _finite_number(rewards[seat])
    final_agent = steps[-1][seat]
    final_agent_reward = _finite_number(final_agent.get("reward"))
    if (
        final_status != "DONE"
        or final_agent.get("status") != final_status
        or final_reward is None
        or final_agent_reward is None
        or final_agent_reward != final_reward
        or final_reward not in (-1.0, 0.0, 1.0)
    ):
        return None
    outcome = (final_reward + 1.0) / 2.0

    return final_status, final_reward, outcome


def _transition_envelope(
    transition: Mapping[str, Any],
) -> tuple[bool, float, Mapping[str, Any] | None, int | None] | None:
    terminal = transition.get("terminal")
    outcome = _finite_number(transition.get("terminal_actor_relative_result"))
    next_state = transition.get("next_same_actor_state")
    horizon_value = transition.get("transition_horizon_logical_decisions")
    horizon = _integer(horizon_value)
    if not isinstance(terminal, bool) or outcome is None:
        return None
    if terminal:
        if next_state is not None or horizon_value is not None:
            return None

        return terminal, outcome, None, None
    if _strict_next_state(next_state) is None or horizon is None or horizon <= 0:
        return None

    return terminal, outcome, dict(next_state), horizon


def _registry_proves_next_state(
    event: SourceEvent,
    registry: AdmissionTransitionRegistry | None,
    *,
    next_state: Mapping[str, Any],
    horizon: int,
) -> tuple[AdmissionTransitionTarget, AdmissionTransitionTarget] | None:
    parsed_next = _strict_next_state(next_state)
    current_global = _integer(event.transition.get("global_decision_index"))
    source_sha256 = event.transition.get("source_hash")
    if parsed_next is None or registry is None:
        return None
    if admission_transition_registry_integrity_failures(registry):
        return None
    if (
        registry.admission_identity != event.admission_identity
        or registry.binding_id != event.binding_id
        or not _is_sha256(registry.registry_sha256)
        or current_global is None
        or not _is_sha256(source_sha256)
    ):
        return None
    current_target = _registry_current_target(event, registry)
    if current_target is None:
        return None
    decision_id, feature_row, representation = parsed_next
    target = registry.target(decision_id)
    if (
        target is None
        or _registry_target_integrity_failures(registry, target)
        or target.source_game_id != event.source_game_id
        or target.actor_seat != event.actor_seat
        or target.source_sha256 != source_sha256
        or target.state_before_feature_row != feature_row
        or target.state_before_representation != representation
        or target.global_decision_index - current_global != horizon
    ):
        return None

    return current_target, target


def align_kaggle_event(
    event: SourceEvent,
    repository: KaggleEpisodeRepository,
    transition_registry: AdmissionTransitionRegistry | None = None,
) -> AlignedDecision:
    """Apply the exact hash-bound Kaggle prior-state alignment rule."""

    failures = list(_kaggle_anchor_failures(event))
    source_path = event.source_record.get("source_path")
    if not isinstance(source_path, str) or not source_path:
        return _failed_alignment(
            event,
            rule_id=KAGGLE_ALIGNMENT_RULE_ID,
            source_scope=KAGGLE_SOURCE_SCOPE,
            failures=(*failures, ValidationFailure.CAUSAL_ALIGNMENT),
            notes=(REPAIR_ATTEMPT_NOTE, "raw_episode_path_missing"),
        )
    try:
        episode = repository.load(source_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return _failed_alignment(
            event,
            rule_id=KAGGLE_ALIGNMENT_RULE_ID,
            source_scope=KAGGLE_SOURCE_SCOPE,
            failures=(*failures, ValidationFailure.CAUSAL_ALIGNMENT),
            notes=(REPAIR_ATTEMPT_NOTE, "raw_episode_unavailable_or_corrupt"),
        )
    data = episode.data
    source_contract = kaggle_source_contract_from_metadata(data)
    rule_id = (
        source_contract.alignment_rule_id
        if source_contract is not None
        else KAGGLE_ALIGNMENT_RULE_ID
    )
    source_scope = (
        source_contract.source_scope
        if source_contract is not None
        else KAGGLE_SOURCE_SCOPE
    )
    if event.observation_asset is not None:
        try:
            repository.register_observation_hash(
                event.source_event_position,
                event.actor_seat if event.actor_seat is not None else -1,
                event.observation_asset.source_content_sha256,
            )
        except CausalAlignmentError:
            failures.append(ValidationFailure.SOURCE_CORRUPT)
    metadata_ok = (
        source_contract is not None
        and isinstance(data.get("id"), str)
        and data.get("id") == event.source_native_game_id
    )
    if not metadata_ok:
        failures.append(ValidationFailure.SCHEMA_MISMATCH)
    if source_contract is None:
        return _failed_alignment(
            event,
            rule_id=rule_id,
            source_scope=source_scope,
            failures=failures,
            notes=(REPAIR_ATTEMPT_NOTE, "replay_source_contract_unregistered"),
        )
    step = _integer(event.source_record.get("step"))
    seat = event.actor_seat
    steps = data.get("steps")
    if step is None or step < 1 or seat not in (0, 1) or not isinstance(steps, list) or step >= len(steps):
        failures.append(ValidationFailure.CAUSAL_ALIGNMENT)

        return _failed_alignment(
            event,
            rule_id=rule_id,
            source_scope=source_scope,
            failures=failures,
            notes=(REPAIR_ATTEMPT_NOTE, "source_coordinates_invalid"),
        )
    prior_step = steps[step - 1]
    current_step = steps[step]
    if not isinstance(prior_step, list) or not isinstance(current_step, list):
        failures.append(ValidationFailure.CAUSAL_ALIGNMENT)

        return _failed_alignment(
            event,
            rule_id=rule_id,
            source_scope=source_scope,
            failures=failures,
            notes=(REPAIR_ATTEMPT_NOTE, "episode_step_shape_invalid"),
        )
    if seat >= len(prior_step) or seat >= len(current_step):
        failures.append(ValidationFailure.CAUSAL_ALIGNMENT)

        return _failed_alignment(
            event,
            rule_id=rule_id,
            source_scope=source_scope,
            failures=failures,
            notes=(REPAIR_ATTEMPT_NOTE, "seat_missing_from_episode_step"),
        )
    prior_agent = prior_step[seat]
    current_agent = current_step[seat]
    if not isinstance(prior_agent, Mapping) or not isinstance(current_agent, Mapping):
        failures.append(ValidationFailure.CAUSAL_ALIGNMENT)

        return _failed_alignment(
            event,
            rule_id=rule_id,
            source_scope=source_scope,
            failures=failures,
            notes=(REPAIR_ATTEMPT_NOTE, "agent_step_shape_invalid"),
        )
    prior_observation = prior_agent.get("observation")
    prior_select = prior_observation.get("select") if isinstance(prior_observation, Mapping) else None
    if prior_agent.get("status") != "ACTIVE" or not (
        isinstance(prior_select, Mapping) and isinstance(prior_select.get("option"), list)
    ):
        failures.append(ValidationFailure.CAUSAL_ALIGNMENT)
    raw_action = _selected_values(current_agent.get("action"))
    if raw_action is None or raw_action != event.stored_selected_choices:
        failures.append(ValidationFailure.CAUSAL_ALIGNMENT)
    result_observation = current_agent.get("observation")
    comparable_result_observation = result_observation
    if source_contract.actor_visible_content_identity:
        comparable_result_observation, comparable_failure = _project_observation(
            result_observation
        )
        if comparable_failure is not None:
            failures.append(comparable_failure)
    stored_result = event.source_record.get("state_json")
    if not isinstance(comparable_result_observation, Mapping) or not isinstance(stored_result, Mapping):
        failures.append(ValidationFailure.CAUSAL_ALIGNMENT)
    elif sha256_json(comparable_result_observation) != sha256_json(stored_result):
        failures.append(ValidationFailure.CAUSAL_ALIGNMENT)
    final_outcome = _kaggle_episode_final_outcome(data, seat=seat)
    envelope = _transition_envelope(event.transition)
    current_status = current_agent.get("status")
    current_reward = _finite_number(current_agent.get("reward"))
    terminal: bool | None = None
    terminal_outcome: float | None = None
    next_state: Mapping[str, Any] | None = None
    horizon: int | None = None
    registry_current_target: AdmissionTransitionTarget | None = None
    registry_target: AdmissionTransitionTarget | None = None
    if final_outcome is None or envelope is None or current_reward is None:
        failures.extend((
            ValidationFailure.SOURCE_CORRUPT,
            ValidationFailure.RESULTING_TRANSITION,
        ))
    else:
        _, final_reward, expected_outcome = final_outcome
        terminal, terminal_outcome, next_state, horizon = envelope
        status_relation_exact = bool(
            (
                terminal
                and current_status in {"INACTIVE", "DONE"}
                and (
                    (current_status == "DONE" and current_reward == final_reward)
                    or (current_status == "INACTIVE" and current_reward == 0.0)
                )
            )
            or (
                not terminal
                and current_status in {"ACTIVE", "INACTIVE"}
                and current_reward == 0.0
            )
        )
        if terminal_outcome != expected_outcome or not status_relation_exact:
            failures.append(ValidationFailure.RESULTING_TRANSITION)
        if not terminal and next_state is not None and horizon is not None:
            registry_link = _registry_proves_next_state(
                event,
                transition_registry,
                next_state=next_state,
                horizon=horizon,
            )
            if registry_link is None:
                failures.append(ValidationFailure.RESULTING_TRANSITION)
            else:
                registry_current_target, registry_target = registry_link
    pre_event, pre_failure = _raw_kaggle_event(
        event,
        episode=episode,
        step=step - 1,
        seat=seat,
        agent=prior_agent,
        role="pre_action",
        source_contract=source_contract,
        source_content_sha256=repository.observation_hash(step - 1, seat),
    )
    result_event, result_failure = _raw_kaggle_event(
        event,
        episode=episode,
        step=step,
        seat=seat,
        agent=current_agent,
        role="result",
        source_contract=source_contract,
        source_content_sha256=(
            event.observation_asset.source_content_sha256
            if event.observation_asset is not None
            else repository.observation_hash(step, seat)
        ),
    )
    if pre_event is not None and pre_event.observation_asset is not None:
        repository.register_observation_hash(
            step - 1,
            seat,
            pre_event.observation_asset.source_content_sha256,
        )
    if pre_failure is not None:
        failures.append(pre_failure)
    if result_failure is not None:
        failures.append(result_failure)
    if pre_event is None or result_event is None:
        failures.append(ValidationFailure.CAUSAL_ALIGNMENT)

        return _failed_alignment(
            event,
            rule_id=rule_id,
            source_scope=source_scope,
            failures=failures,
            notes=(REPAIR_ATTEMPT_NOTE, "deployable_observation_projection_failed"),
        )
    actor_seat_proven = pre_event.controlled_player == seat and result_event.controlled_player == seat
    if not actor_seat_proven:
        failures.append(ValidationFailure.SEAT_IDENTITY)
    old_legal = tuple(event.legacy_legal_actions)
    repaired_legal = tuple(pre_event.legacy_legal_actions)
    legacy_stale = sha256_json(old_legal) != sha256_json(repaired_legal)
    action_equality_proven = (
        raw_action is not None
        and raw_action == event.stored_selected_choices
    )
    event_order_proven = (
        pre_event.source_event_position + 1 == result_event.source_event_position
    )
    result_observation_proven = (
        isinstance(comparable_result_observation, Mapping)
        and isinstance(stored_result, Mapping)
        and sha256_json(comparable_result_observation) == sha256_json(stored_result)
    )
    envelope_proven = bool(
        envelope is not None
        and final_outcome is not None
        and terminal_outcome == final_outcome[2]
        and current_reward is not None
        and (
            (
                terminal is True
                and current_status in {"INACTIVE", "DONE"}
                and (
                    (current_status == "DONE" and current_reward == final_outcome[1])
                    or (current_status == "INACTIVE" and current_reward == 0.0)
                )
            )
            or (
                terminal is False
                and current_status in {"ACTIVE", "INACTIVE"}
                and current_reward == 0.0
                and registry_target is not None
            )
        )
    )
    resulting_transition_proven = bool(
        result_observation_proven
        and envelope_proven
        and not failures
    )
    admission_raw_sha256 = (
        event.observation_asset.source_sha256
        if event.observation_asset is not None
        else ""
    )
    result_transition_id = None
    if (
        resulting_transition_proven
        and _is_sha256(admission_raw_sha256)
        and _is_sha256(episode.sha256)
        and terminal is not None
        and terminal_outcome is not None
    ):
        result_transition_id = _resulting_transition_id(
            event=event,
            admission_raw_sha256=admission_raw_sha256,
            result_raw_sha256=episode.sha256,
            pre_action_event_id=pre_event.event_id,
            action_event_ids=(event.event_id,),
            result_event_id=result_event.event_id,
            pre_action_observation_id=pre_event.observation_asset.observation_id,
            result_observation_id=result_event.observation_asset.observation_id,
            result_status=current_status,
            result_reward=current_agent.get("reward"),
            result_outcome={
                "terminal_actor_relative_result": terminal_outcome,
                "raw_final_status": final_outcome[0],
                "raw_final_reward": final_outcome[1],
                "raw_final_actor_relative_outcome": final_outcome[2],
            },
            terminal=terminal,
            next_same_actor_state=next_state,
            transition_horizon_logical_decisions=horizon,
            rule_id=rule_id,
            registry_sha256=(
                transition_registry.registry_sha256
                if transition_registry is not None
                else None
            ),
            registry_current_target=registry_current_target,
            registry_target=registry_target,
        )
    repair_succeeded = not failures and all((
        action_equality_proven,
        actor_seat_proven,
        event_order_proven,
        resulting_transition_proven,
    ))
    notes = [
        REPAIR_ATTEMPT_NOTE,
        "prior_step_select_bound",
        "current_step_action_and_result_bound",
        f"registered_rule_id={rule_id}",
    ]
    if transition_registry is not None:
        notes.append(
            f"admission_transition_registry_sha256={transition_registry.registry_sha256}"
        )
    if registry_current_target is not None:
        notes.append(
            "admission_transition_current_target_commitment_sha256="
            f"{registry_current_target.target_commitment_sha256}"
        )
    if registry_target is not None:
        notes.append(
            "admission_transition_target_commitment_sha256="
            f"{registry_target.target_commitment_sha256}"
        )
    if repair_succeeded:
        notes.append(REPAIR_SUCCEEDED_NOTE)
    if legacy_stale:
        notes.append("admission_v0_legal_set_was_post_action")
    proof = AlignmentProof(
        rule_id=rule_id,
        source_scope=source_scope,
        pre_action_event_id=pre_event.event_id,
        action_event_ids=(event.event_id,),
        resulting_event_id=result_transition_id,
        pre_action_observation_id=pre_event.observation_asset.observation_id,
        legal_observation_id=pre_event.observation_asset.observation_id,
        resulting_observation_id=result_event.observation_asset.observation_id,
        action_equality_proven=action_equality_proven,
        actor_seat_proven=actor_seat_proven,
        event_order_proven=event_order_proven,
        resulting_transition_proven=resulting_transition_proven,
        source_hash_bound=bool(
            _is_sha256(episode.sha256)
            and _is_sha256(admission_raw_sha256)
            and ValidationFailure.SOURCE_CORRUPT not in failures
            and ValidationFailure.SCHEMA_MISMATCH not in failures
            and ValidationFailure.PRIVILEGED_OBSERVATION not in failures
        ),
        deterministic_repair_attempted=True,
        deterministic_repair_applied=repair_succeeded,
        notes=tuple(notes),
    )

    return _seal_aligned_decision(AlignedDecision(
        pre_action_event=pre_event,
        action_events=(event,),
        resulting_event=result_event,
        stage_selected_choices=(raw_action,),
        stage_index_spaces=("option_position",),
        causal_decision_index=_causal_index(event, step),
        proof=proof,
        failures=_unique_failures(failures),
    ))


def _transition_choices(transition: Mapping[str, Any]) -> tuple[int, ...] | None:
    chosen = transition.get("chosen_action")
    if not isinstance(chosen, Mapping):
        return None

    return _selected_values(chosen.get("indices"))


def align_refgame_event(
    event: SourceEvent,
    transition_registry: AdmissionTransitionRegistry | None = None,
) -> AlignedDecision:
    """Bind a normalized refgame row's contemporaneous request and transition."""

    failures: list[ValidationFailure] = []
    if (
        event.source_corrupt
        or event.observation_asset is None
        or event.observation is None
        or source_event_integrity_failures(event)
    ):
        failures.append(ValidationFailure.SOURCE_CORRUPT)
    actor_seat_proven = event.actor_seat in (0, 1) and event.controlled_player == event.actor_seat
    if not actor_seat_proven:
        failures.append(ValidationFailure.SEAT_IDENTITY)
    transition_choices = _transition_choices(event.transition)
    action_equality = transition_choices is not None and transition_choices == event.stored_selected_choices
    if not action_equality:
        failures.append(ValidationFailure.CAUSAL_ALIGNMENT)
    envelope = _transition_envelope(event.transition)
    terminal: bool | None = None
    terminal_result: float | None = None
    next_state: Mapping[str, Any] | None = None
    horizon: int | None = None
    registry_current_target: AdmissionTransitionTarget | None = None
    registry_target: AdmissionTransitionTarget | None = None
    if envelope is None:
        failures.append(ValidationFailure.SOURCE_CORRUPT)
        failures.append(ValidationFailure.RESULTING_TRANSITION)
    else:
        terminal, terminal_result, next_state, horizon = envelope
        if not terminal and next_state is not None and horizon is not None:
            registry_link = _registry_proves_next_state(
                event,
                transition_registry,
                next_state=next_state,
                horizon=horizon,
            )
            if registry_link is None:
                failures.append(ValidationFailure.RESULTING_TRANSITION)
            else:
                registry_current_target, registry_target = registry_link
    source_hash_bound = bool(
        event.observation_asset is not None
        and _is_sha256(event.observation_asset.source_sha256)
        and _is_sha256(event.observation_asset.source_content_sha256)
        and _is_sha256(event.source_row_sha256)
        and _is_sha256(event.source_artifact_sha256)
        and _is_sha256(event.transition.get("source_hash"))
        and event.observation_asset.source_sha256
        == event.source_artifact_sha256
        == event.transition.get("source_hash")
    )
    if not source_hash_bound:
        failures.append(ValidationFailure.SOURCE_CORRUPT)
    observation_id = event.observation_asset.observation_id if event.observation_asset else None
    result_event_id = (
        event.event_id
        if terminal is True
        else registry_target.source_event_id if registry_target is not None else None
    )
    result_observation_id = (
        observation_id
        if terminal is True
        else registry_target.observation_id if registry_target is not None else None
    )
    event_order_proven = bool(
        event.source_event_position >= 0
        and (
            terminal is True
            or registry_target is not None
        )
    )
    resulting_transition_proven = bool(
        envelope is not None
        and (terminal is True or registry_target is not None)
        and source_hash_bound
        and action_equality
        and actor_seat_proven
        and event_order_proven
        and not failures
        and observation_id is not None
        and result_event_id is not None
    )
    result_transition_id = None
    if (
        resulting_transition_proven
        and event.observation_asset is not None
        and terminal is not None
        and terminal_result is not None
        and result_event_id is not None
    ):
        result_transition_id = _resulting_transition_id(
            event=event,
            admission_raw_sha256=event.observation_asset.source_sha256,
            result_raw_sha256=event.observation_asset.source_sha256,
            pre_action_event_id=event.event_id,
            action_event_ids=(event.event_id,),
            result_event_id=result_event_id,
            pre_action_observation_id=event.observation_asset.observation_id,
            result_observation_id=result_observation_id,
            result_status=event.transition.get("state_after_status"),
            result_reward={
                "immediate_reward": event.transition.get("immediate_reward"),
                "immediate_reward_status": event.transition.get(
                    "immediate_reward_status"
                ),
            },
            result_outcome=terminal_result,
            terminal=terminal,
            next_same_actor_state=next_state,
            transition_horizon_logical_decisions=horizon,
            rule_id=REFGAME_ALIGNMENT_RULE_ID,
            registry_sha256=(
                transition_registry.registry_sha256
                if transition_registry is not None
                else None
            ),
            registry_current_target=registry_current_target,
            registry_target=registry_target,
        )
    notes = ["same_source_event_contains_request_legal_set_and_choice"]
    if transition_registry is not None:
        notes.append(
            f"admission_transition_registry_sha256={transition_registry.registry_sha256}"
        )
    if registry_current_target is not None:
        notes.append(
            "admission_transition_current_target_commitment_sha256="
            f"{registry_current_target.target_commitment_sha256}"
        )
    if registry_target is not None:
        notes.append(
            "admission_transition_target_commitment_sha256="
            f"{registry_target.target_commitment_sha256}"
        )
    proof = AlignmentProof(
        rule_id=REFGAME_ALIGNMENT_RULE_ID,
        source_scope=f"refgame source_schema={event.source_version}",
        pre_action_event_id=event.event_id,
        action_event_ids=(event.event_id,),
        resulting_event_id=result_transition_id if resulting_transition_proven else None,
        pre_action_observation_id=observation_id,
        legal_observation_id=observation_id,
        resulting_observation_id=result_observation_id,
        action_equality_proven=action_equality,
        actor_seat_proven=actor_seat_proven,
        event_order_proven=event_order_proven,
        resulting_transition_proven=resulting_transition_proven,
        source_hash_bound=source_hash_bound,
        deterministic_repair_attempted=False,
        deterministic_repair_applied=False,
        notes=tuple(notes),
    )

    return _seal_aligned_decision(AlignedDecision(
        pre_action_event=event,
        action_events=(event,),
        resulting_event=None,
        stage_selected_choices=(event.stored_selected_choices,),
        stage_index_spaces=(event.stored_index_space,),
        causal_decision_index=_causal_index(event, event.source_event_position),
        proof=proof,
        failures=_unique_failures(failures),
    ))


def align_explicit_multi_event_action(
    pre_action_event: SourceEvent,
    action_events: Sequence[SourceEvent],
    resulting_event: SourceEvent,
    *,
    rule_id: str,
    source_scope: str,
    causal_decision_index: int,
    evidence: Mapping[str, Any] | None = None,
    transition_registry: AdmissionTransitionRegistry | None = None,
) -> AlignedDecision:
    """Bind an explicitly documented multi-event action without adjacency guesses.

    This adapter is intentionally generic: a source adapter must already have
    identified the exact event sequence under a versioned source convention.
    Merely neighboring rows are not sufficient input evidence.
    """

    if not rule_id or not source_scope or not action_events:
        raise CausalAlignmentError("explicit multi-event alignment metadata is incomplete")
    events = (pre_action_event, *action_events, resulting_event)
    failures: list[ValidationFailure] = []
    if any(source_event_integrity_failures(event) for event in events):
        failures.append(ValidationFailure.SOURCE_CORRUPT)
    admission_identities = {event.admission_identity for event in events}
    games = {event.source_game_id for event in events}
    seats = {event.actor_seat for event in events}
    source_ids = {event.source_id for event in events}
    source_formats = {event.source_format for event in events}
    source_versions = {event.source_version for event in events}
    if (
        len(admission_identities) != 1
        or len(games) != 1
        or len(source_ids) != 1
        or len(source_formats) != 1
        or len(source_versions) != 1
    ):
        failures.append(ValidationFailure.CAUSAL_ALIGNMENT)
    if len(seats) != 1 or pre_action_event.actor_seat not in (0, 1):
        failures.append(ValidationFailure.SEAT_IDENTITY)
    source_tuple = (
        rule_id,
        source_scope,
        pre_action_event.source_id,
        pre_action_event.source_format,
        pre_action_event.source_version,
    )
    rule_registered = source_tuple in EXPLICIT_MULTI_EVENT_RULE_REGISTRY
    if not rule_registered:
        failures.append(ValidationFailure.CAUSAL_ALIGNMENT)
    positions = [event.source_event_position for event in events]
    event_order_proven = all(
        right == left + 1 for left, right in zip(positions, positions[1:])
    )
    if not event_order_proven:
        failures.append(ValidationFailure.CAUSAL_ALIGNMENT)
    request_events = (pre_action_event, *action_events[:-1])
    request_selects = tuple(
        event.observation.get("select")
        if isinstance(event.observation, Mapping)
        else None
        for event in request_events
    )
    requests_complete = all(
        isinstance(select, Mapping)
        and isinstance(select.get("option"), list)
        for select in request_selects
    )
    if not requests_complete:
        failures.append(ValidationFailure.CAUSAL_ALIGNMENT)
    selected = tuple(event.stored_selected_choices for event in action_events)
    transition_selected = tuple(
        _transition_choices(event.transition)
        for event in action_events
    )
    action_equality_proven = all(
        values is not None and values == transition_values
        for values, transition_values in zip(selected, transition_selected)
    )
    if not action_equality_proven:
        failures.append(ValidationFailure.UNRESOLVED_ACTION)
    actor_seat_proven = all(
        event.controlled_player == pre_action_event.actor_seat for event in events
    )
    if not actor_seat_proven:
        failures.append(ValidationFailure.SEAT_IDENTITY)
    source_hashes = {
        event.observation_asset.source_sha256
        for event in events
        if event.observation_asset is not None
    }
    artifact_hashes = {event.source_artifact_sha256 for event in events}
    artifact_paths = {event.source_artifact_path for event in events}
    source_hash_bound = bool(
        len(source_hashes) == 1
        and len(artifact_hashes) == 1
        and len(artifact_paths) == 1
        and None not in artifact_paths
        and all(event.observation_asset is not None for event in events)
        and all(_is_sha256(value) for value in source_hashes)
        and all(_is_sha256(value) for value in artifact_hashes)
        and source_hashes == artifact_hashes
        and all(_is_sha256(event.source_row_sha256) for event in events)
        and all(
            event.observation_asset is not None
            and _is_sha256(event.observation_asset.source_content_sha256)
            for event in events
        )
    )
    if not source_hash_bound:
        failures.append(ValidationFailure.SOURCE_CORRUPT)
    evidence_proven, chain_id = _explicit_multi_event_evidence(
        evidence,
        pre_action_event=pre_action_event,
        action_events=action_events,
        resulting_event=resulting_event,
        rule_id=rule_id,
        source_scope=source_scope,
    )
    if not evidence_proven:
        failures.append(ValidationFailure.CAUSAL_ALIGNMENT)
    if pre_action_event.observation_asset is None or resulting_event.observation_asset is None:
        failures.append(ValidationFailure.SOURCE_CORRUPT)
    pre_observation_id = (
        pre_action_event.observation_asset.observation_id
        if pre_action_event.observation_asset is not None
        else None
    )
    result_observation_id = (
        resulting_event.observation_asset.observation_id
        if resulting_event.observation_asset is not None
        else None
    )
    result_envelope = _transition_envelope(resulting_event.transition)
    result_terminal: bool | None = None
    result_outcome: float | None = None
    result_next_state: Mapping[str, Any] | None = None
    result_horizon: int | None = None
    result_registry_current_target: AdmissionTransitionTarget | None = None
    result_registry_target: AdmissionTransitionTarget | None = None
    if result_envelope is not None:
        (
            result_terminal,
            result_outcome,
            result_next_state,
            result_horizon,
        ) = result_envelope
        if (
            result_terminal is False
            and result_next_state is not None
            and result_horizon is not None
        ):
            registry_link = _registry_proves_next_state(
                resulting_event,
                transition_registry,
                next_state=result_next_state,
                horizon=result_horizon,
            )
            if registry_link is None:
                failures.append(ValidationFailure.RESULTING_TRANSITION)
            else:
                (
                    result_registry_current_target,
                    result_registry_target,
                ) = registry_link
    else:
        failures.append(ValidationFailure.RESULTING_TRANSITION)
    resulting_transition_proven = bool(
        result_observation_id is not None
        and pre_observation_id is not None
        and evidence_proven
        and source_hash_bound
        and event_order_proven
        and action_equality_proven
        and actor_seat_proven
        and rule_registered
        and result_envelope is not None
        and (result_terminal is True or result_registry_target is not None)
        and not failures
    )
    resulting_transition_id = None
    if (
        resulting_transition_proven
        and result_terminal is not None
        and result_outcome is not None
        and pre_action_event.observation_asset is not None
        and resulting_event.observation_asset is not None
    ):
        resulting_transition_id = _resulting_transition_id(
            event=pre_action_event,
            admission_raw_sha256=pre_action_event.observation_asset.source_sha256,
            result_raw_sha256=resulting_event.observation_asset.source_sha256,
            pre_action_event_id=pre_action_event.event_id,
            action_event_ids=tuple(event.event_id for event in action_events),
            result_event_id=resulting_event.event_id,
            pre_action_observation_id=pre_action_event.observation_asset.observation_id,
            result_observation_id=resulting_event.observation_asset.observation_id,
            result_status=resulting_event.transition.get("state_after_status"),
            result_reward={
                "immediate_reward": resulting_event.transition.get("immediate_reward"),
                "immediate_reward_status": resulting_event.transition.get(
                    "immediate_reward_status"
                ),
            },
            result_outcome=result_outcome,
            terminal=result_terminal,
            next_same_actor_state=result_next_state,
            transition_horizon_logical_decisions=result_horizon,
            rule_id=rule_id,
            registry_sha256=(
                transition_registry.registry_sha256
                if transition_registry is not None
                else None
            ),
            registry_current_target=result_registry_current_target,
            registry_target=result_registry_target,
        )
    registry_note = (
        f"admission_transition_registry_sha256={transition_registry.registry_sha256}"
        if transition_registry is not None
        else "admission_transition_registry_not_used"
    )
    registry_target_note = (
        "admission_transition_target_commitment_sha256="
        f"{result_registry_target.target_commitment_sha256}"
        if result_registry_target is not None
        else "admission_transition_target_not_used"
    )
    registry_current_target_note = (
        "admission_transition_current_target_commitment_sha256="
        f"{result_registry_current_target.target_commitment_sha256}"
        if result_registry_current_target is not None
        else "admission_transition_current_target_not_used"
    )
    proof = AlignmentProof(
        rule_id=rule_id,
        source_scope=source_scope,
        pre_action_event_id=pre_action_event.event_id,
        action_event_ids=tuple(event.event_id for event in action_events),
        resulting_event_id=resulting_transition_id,
        pre_action_observation_id=pre_observation_id,
        legal_observation_id=pre_observation_id,
        resulting_observation_id=result_observation_id,
        action_equality_proven=action_equality_proven,
        actor_seat_proven=actor_seat_proven,
        event_order_proven=event_order_proven,
        resulting_transition_proven=resulting_transition_proven,
        source_hash_bound=source_hash_bound,
        deterministic_repair_attempted=False,
        deterministic_repair_applied=False,
        notes=(
            "explicit_source_convention_required",
            "no_adjacency_inference",
            f"explicit_rule_registry_sha256={EXPLICIT_MULTI_EVENT_RULE_REGISTRY_SHA256}",
            (
                "explicit_rule_registered"
                if rule_registered
                else "explicit_rule_unregistered"
            ),
            (
                f"evidence_schema={EXPLICIT_MULTI_EVENT_EVIDENCE_SCHEMA_VERSION}"
                if evidence_proven
                else "explicit_multi_event_evidence_unproven"
            ),
            f"stage_request_action_pairs_bound={len(action_events)}",
            registry_note,
            registry_current_target_note,
            registry_target_note,
        ),
    )

    return _seal_aligned_decision(AlignedDecision(
        pre_action_event=pre_action_event,
        action_events=tuple(action_events),
        resulting_event=resulting_event,
        stage_selected_choices=selected,
        stage_index_spaces=tuple(event.stored_index_space for event in action_events),
        causal_decision_index=causal_decision_index,
        proof=proof,
        failures=_unique_failures(failures),
    ))


def align_decision(
    event: SourceEvent,
    *,
    kaggle_repository: KaggleEpisodeRepository | None = None,
    transition_registry: AdmissionTransitionRegistry | None = None,
) -> AlignedDecision:
    if event.source_format == "kaggle_replay":
        if kaggle_repository is None:
            raise CausalAlignmentError("Kaggle alignment requires an explicit episode repository")

        return align_kaggle_event(
            event,
            kaggle_repository,
            transition_registry,
        )
    if event.source_format == "refgame":
        return align_refgame_event(event, transition_registry)

    return _failed_alignment(
        event,
        rule_id=CAUSAL_ALIGNER_VERSION,
        source_scope=f"unsupported source_format={event.source_format}",
        failures=(ValidationFailure.CAUSAL_ALIGNMENT,),
        notes=("source_format_not_supported",),
    )


__all__ = [
    "ADMISSION_TRANSITION_REGISTRY_SCHEMA_VERSION",
    "ALIGNED_DECISION_RECEIPT_SCHEMA_VERSION",
    "CAUSAL_ALIGNER_VERSION",
    "EXPECTED_ACTION_SCHEMA_VERSION",
    "EXPLICIT_MULTI_EVENT_RULE_REGISTRY",
    "EXPLICIT_MULTI_EVENT_RULE_REGISTRY_SHA256",
    "EXPLICIT_MULTI_EVENT_EVIDENCE_SCHEMA_VERSION",
    "KAGGLE_ANCHOR_SOURCE_FORMAT",
    "KAGGLE_ANCHOR_SOURCE_ID",
    "KAGGLE_ANCHOR_SOURCE_VERSION",
    "KAGGLE_ALIGNMENT_RULE_ID",
    "KAGGLE_SOURCE_SCOPE",
    "REPAIR_ATTEMPT_NOTE",
    "REPAIR_SUCCEEDED_NOTE",
    "REFGAME_ALIGNMENT_RULE_ID",
    "RESULTING_TRANSITION_OWNER_SCHEMA_VERSION",
    "AdmissionTransitionRegistry",
    "AdmissionTransitionRowCommitment",
    "AdmissionTransitionTarget",
    "BoundKaggleEpisode",
    "CausalAlignmentError",
    "KaggleEpisodeRepository",
    "align_decision",
    "align_explicit_multi_event_action",
    "align_kaggle_event",
    "align_refgame_event",
    "aligned_decision_integrity_failures",
    "admission_transition_registry_integrity_failures",
    "build_admission_transition_registry",
    "explicit_multi_event_chain_id",
]
