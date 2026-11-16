"""Authoritative combined legality validation and decision construction."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import asdict, replace
from typing import Any, Mapping, Sequence

from deepmew.replay.contract.causal_alignment import (
    CAUSAL_ALIGNER_VERSION,
    aligned_decision_integrity_failures,
)
from deepmew.replay.contract.chosen_actions import reconstruct_chosen_action
from deepmew.replay.contract.identities import (
    canonical_decision_id,
    canonical_json_bytes,
    identity,
    is_lowercase_sha256,
)
from deepmew.replay.contract.legal_actions import (
    REFGAME_CORE_SEMANTIC_FIELDS,
    REFGAME_DISPLAY_ONLY_SEMANTIC_FIELDS,
    LegalActionContractError,
    StaleLegalSetError,
    build_cabt_legal_action_set,
    build_multi_event_cabt_legal_action_set,
    build_semantic_legal_action_set,
    exact_json_value_equal,
)
from deepmew.replay.contract.kaggle_sources import KAGGLE_REPLAY_SOURCE_SCHEMAS
from deepmew.replay.contract.schemas import (
    CHOSEN_ACTION_SCHEMA_VERSION,
    CRITIC_ELIGIBILITY_VERSION,
    DECISION_RECORD_SCHEMA_VERSION,
    VALIDATOR_VERSION,
    AlignedDecision,
    CanonicalChosenAction,
    CanonicalDecisionImplementationHashes,
    CanonicalDecisionRecord,
    CanonicalLegalActionSet,
    CriticEligibilityPolicy,
    ValidationFailure,
    ValidationOutcome,
    ValidationResult,
)
from deepmew.replay.contract.source_events import source_event_integrity_failures
from deepmew.replay.information_boundary import InformationMode, assert_deployable_observation


EXPECTED_TRANSITION_SCHEMA = "deepmew.actor_transition.v1"
SUPPORTED_SOURCE_VERSIONS_BY_FORMAT = {
    "kaggle_replay": frozenset({"deepmew.replay_action_row.v1"}),
    "kaggle_episode_replay": KAGGLE_REPLAY_SOURCE_SCHEMAS,
    "refgame": frozenset({
        "deepmew.refgame_decision_row.rebuilt.v1",
        "deepmew.refgame_decision_row.rebuilt.v2",
        "deepmew.refgames_factory.decision.v0",
    }),
}


class CanonicalDecisionValidationHardStop(RuntimeError):
    """The caller requested semantics outside the frozen V1 validator."""

FAILURE_PRECEDENCE = (
    ValidationFailure.SOURCE_CORRUPT,
    ValidationFailure.SCHEMA_MISMATCH,
    ValidationFailure.PRIVILEGED_OBSERVATION,
    ValidationFailure.CAUSAL_ALIGNMENT,
    ValidationFailure.RESULTING_TRANSITION,
    ValidationFailure.SEAT_IDENTITY,
    ValidationFailure.STALE_LEGAL_SET,
    ValidationFailure.ACTION_TEMPLATE_NOT_LEGAL,
    ValidationFailure.MISSING_REQUIRED_CHOICE,
    ValidationFailure.TOO_FEW_CHOICES,
    ValidationFailure.TOO_MANY_CHOICES,
    ValidationFailure.CHOICE_NOT_LEGAL,
    ValidationFailure.DUPLICATE_FORBIDDEN_CHOICE,
    ValidationFailure.CHOICE_ORDER,
    ValidationFailure.UNRESOLVED_ACTION,
    ValidationFailure.UNKNOWN,
)

OUTCOME_BY_FAILURE = {
    ValidationFailure.MISSING_REQUIRED_CHOICE: ValidationOutcome.REJECTED_MISSING_REQUIRED_CHOICE,
    ValidationFailure.TOO_FEW_CHOICES: ValidationOutcome.REJECTED_TOO_FEW_CHOICES,
    ValidationFailure.TOO_MANY_CHOICES: ValidationOutcome.REJECTED_TOO_MANY_CHOICES,
    ValidationFailure.CHOICE_NOT_LEGAL: ValidationOutcome.REJECTED_CHOICE_NOT_LEGAL,
    ValidationFailure.CHOICE_ORDER: ValidationOutcome.REJECTED_CHOICE_ORDER,
    ValidationFailure.DUPLICATE_FORBIDDEN_CHOICE: ValidationOutcome.REJECTED_DUPLICATE_FORBIDDEN_CHOICE,
    ValidationFailure.ACTION_TEMPLATE_NOT_LEGAL: ValidationOutcome.REJECTED_ACTION_TEMPLATE_NOT_LEGAL,
    ValidationFailure.STALE_LEGAL_SET: ValidationOutcome.REJECTED_STALE_LEGAL_SET,
    ValidationFailure.CAUSAL_ALIGNMENT: ValidationOutcome.REJECTED_CAUSAL_ALIGNMENT,
    ValidationFailure.RESULTING_TRANSITION: ValidationOutcome.REJECTED_CAUSAL_ALIGNMENT,
    ValidationFailure.SEAT_IDENTITY: ValidationOutcome.REJECTED_SEAT_IDENTITY,
    ValidationFailure.SCHEMA_MISMATCH: ValidationOutcome.REJECTED_SCHEMA_MISMATCH,
    ValidationFailure.PRIVILEGED_OBSERVATION: ValidationOutcome.REJECTED_PRIVILEGED_OBSERVATION,
    ValidationFailure.SOURCE_CORRUPT: ValidationOutcome.REJECTED_SOURCE_CORRUPT,
    ValidationFailure.UNRESOLVED_ACTION: ValidationOutcome.UNKNOWN_FAIL_CLOSED,
    ValidationFailure.UNKNOWN: ValidationOutcome.UNKNOWN_FAIL_CLOSED,
}

ACTION_SEMANTIC_FAILURES = frozenset({
    ValidationFailure.ACTION_TEMPLATE_NOT_LEGAL,
    ValidationFailure.MISSING_REQUIRED_CHOICE,
    ValidationFailure.TOO_FEW_CHOICES,
    ValidationFailure.TOO_MANY_CHOICES,
    ValidationFailure.CHOICE_NOT_LEGAL,
    ValidationFailure.DUPLICATE_FORBIDDEN_CHOICE,
    ValidationFailure.CHOICE_ORDER,
    ValidationFailure.UNRESOLVED_ACTION,
})

CANONICAL_ID_OUTCOMES = frozenset({
    ValidationOutcome.VALID_ACTOR_AND_CRITIC,
    ValidationOutcome.VALID_CRITIC_ONLY_FORCED,
})


def _unique_failures(values: Sequence[ValidationFailure]) -> tuple[ValidationFailure, ...]:
    present = set(values)

    return tuple(failure for failure in FAILURE_PRECEDENCE if failure in present)


def _contract_values_match(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    try:
        left_fields = asdict(left)
        right_fields = asdict(right)
    except Exception:
        return False

    return exact_json_value_equal(left_fields, right_fields)


def _validate_implementation_hashes(
    implementation_hashes: CanonicalDecisionImplementationHashes,
) -> None:
    if type(implementation_hashes) is not CanonicalDecisionImplementationHashes:
        raise CanonicalDecisionValidationHardStop(
            "canonical decision implementation hashes must use the typed V1 input"
        )
    for name in (
        "causal_aligner_implementation_sha256",
        "legality_validator_implementation_sha256",
        "action_schema_implementation_sha256",
    ):
        if not is_lowercase_sha256(getattr(implementation_hashes, name)):
            raise CanonicalDecisionValidationHardStop(
                f"{name} must be an exact lowercase SHA-256"
            )


def _invalid_legal_set(
    aligned: AlignedDecision,
    failure: ValidationFailure,
) -> CanonicalLegalActionSet:
    observation_id = aligned.proof.pre_action_observation_id or "unresolved-observation"

    return CanonicalLegalActionSet(
        observation_id=observation_id,
        action_family="UNRESOLVED",
        action_template="UNRESOLVED",
        stages=(),
        complete_action_count=None,
        legal_action_set_id=identity("legal1", {
            "observation_id": observation_id,
            "failure": failure.value,
        }),
        source_select_type=None,
        source_select_context=None,
        failures=(failure,),
    )


def reconstruct_legal_actions(aligned: AlignedDecision) -> CanonicalLegalActionSet:
    """Reconstruct the only legal set downstream consumers may use."""

    event = aligned.pre_action_event
    observation_id = aligned.proof.pre_action_observation_id
    if observation_id is None or event.observation is None:
        return _invalid_legal_set(aligned, ValidationFailure.SOURCE_CORRUPT)
    select = event.observation.get("select")
    try:
        if len(aligned.action_events) > 1:
            request_events = (event, *aligned.action_events[:-1])
            requests = []
            for request_event in request_events:
                if request_event.observation_asset is None:
                    raise LegalActionContractError(
                        "multi-event request observation asset is absent"
                    )
                request_select = (
                    request_event.observation.get("select")
                    if isinstance(request_event.observation, Mapping)
                    else None
                )
                if not isinstance(request_select, Mapping):
                    raise LegalActionContractError(
                        "multi-event request CABT select is absent"
                    )
                requests.append((
                    request_event.observation_asset.observation_id,
                    request_select,
                ))

            return build_multi_event_cabt_legal_action_set(
                observation_id,
                requests,
                aligned.stage_selected_choices,
                stage_index_spaces=aligned.stage_index_spaces,
            )
        if event.source_format == "kaggle_episode_replay":
            if not isinstance(select, Mapping):
                raise LegalActionContractError("aligned Kaggle pre-action select is absent")

            return build_cabt_legal_action_set(observation_id, select)
        if event.source_format == "kaggle_replay":
            alignment_failure = next(
                (
                    failure
                    for failure in FAILURE_PRECEDENCE
                    if failure in aligned.failures
                ),
                ValidationFailure.CAUSAL_ALIGNMENT,
            )

            return _invalid_legal_set(aligned, alignment_failure)
        source_legal = event.source_record.get("legal_set")
        if not isinstance(source_legal, list):
            raise LegalActionContractError("refgame semantic legal_set is absent")
        action_mask = event.source_record.get("action_mask")
        if not isinstance(action_mask, Mapping):
            raise LegalActionContractError("refgame action_mask is absent")

        return build_semantic_legal_action_set(
            observation_id,
            source_legal,
            action_mask,
            actor_observation=event.observation,
            select=select if isinstance(select, Mapping) else None,
        )
    except StaleLegalSetError:
        return _invalid_legal_set(aligned, ValidationFailure.STALE_LEGAL_SET)
    except LegalActionContractError:
        return _invalid_legal_set(aligned, ValidationFailure.SOURCE_CORRUPT)


def reconstruct_selected_action(
    aligned: AlignedDecision,
    legal_actions: CanonicalLegalActionSet,
) -> CanonicalChosenAction:
    event = aligned.pre_action_event
    if not legal_actions.stages:
        return CanonicalChosenAction(
            action_template=legal_actions.action_template,
            stages=(),
            chosen_action_id=None,
            complete=False,
            failures=legal_actions.failures or (ValidationFailure.UNRESOLVED_ACTION,),
        )
    request_events = (
        (event, *aligned.action_events[:-1])
        if len(aligned.action_events) > 1
        else (event,)
    )
    legacy_stage_options = tuple(
        tuple(request_event.legacy_legal_actions)
        for request_event in request_events
    )

    return reconstruct_chosen_action(
        legal_actions,
        aligned.stage_selected_choices,
        stage_index_spaces=aligned.stage_index_spaces,
        legacy_stage_options=legacy_stage_options,
    )


def _require_authoritative_reconstruction(
    aligned: AlignedDecision,
    legal_actions: CanonicalLegalActionSet,
    chosen: CanonicalChosenAction,
) -> tuple[CanonicalLegalActionSet, CanonicalChosenAction]:
    authoritative_legal_actions = reconstruct_legal_actions(aligned)
    if not _contract_values_match(legal_actions, authoritative_legal_actions):
        raise CanonicalDecisionValidationHardStop(
            "caller-supplied legal action set differs from authoritative reconstruction"
        )
    authoritative_chosen = reconstruct_selected_action(
        aligned,
        authoritative_legal_actions,
    )
    if not _contract_values_match(chosen, authoritative_chosen):
        raise CanonicalDecisionValidationHardStop(
            "caller-supplied chosen action differs from authoritative reconstruction"
        )

    return authoritative_legal_actions, authoritative_chosen


def _source_semantics_match(
    aligned: AlignedDecision,
    legal_actions: CanonicalLegalActionSet,
    chosen: CanonicalChosenAction,
) -> bool:
    event = aligned.pre_action_event
    if event.source_format in {"kaggle_episode_replay", "kaggle_replay"}:
        return True
    if legal_actions.failures or not legal_actions.stages:
        return True
    raw_chosen = event.source_record.get("chosen_action")
    if not isinstance(raw_chosen, Mapping):
        return False
    raw_semantics = raw_chosen.get("semantics")
    if raw_semantics is None:
        return True
    if not chosen.complete:
        return True
    if not isinstance(raw_semantics, list) or len(chosen.stages) != 1:
        return False
    option_by_id = {
        option.option_id: option.payload
        for option in legal_actions.stages[0].options
    }
    selected_payloads = [
        option_by_id.get(option_id)
        for option_id in chosen.stages[0].selected_option_ids
    ]
    core_fields = frozenset(REFGAME_CORE_SEMANTIC_FIELDS)
    allowed_fields = core_fields | REFGAME_DISPLAY_ONLY_SEMANTIC_FIELDS
    projected_semantics = []
    for raw_semantic in raw_semantics:
        if not isinstance(raw_semantic, Mapping):
            return False
        source_fields = frozenset(raw_semantic)
        if core_fields - source_fields or source_fields - allowed_fields:
            return False
        projected_semantics.append({
            field: raw_semantic[field]
            for field in REFGAME_CORE_SEMANTIC_FIELDS
        })

    return exact_json_value_equal(selected_payloads, projected_semantics)


def _validate_critic_policy(policy: CriticEligibilityPolicy) -> None:
    frozen_policy = CriticEligibilityPolicy()
    if type(policy) is not CriticEligibilityPolicy:
        raise CanonicalDecisionValidationHardStop(
            "critic-only eligibility policy must use the typed V1 input"
        )
    if policy.version != CRITIC_ELIGIBILITY_VERSION:
        raise CanonicalDecisionValidationHardStop(
            "critic-only eligibility policy version is not supported by V1"
        )
    if policy != frozen_policy:
        raise CanonicalDecisionValidationHardStop(
            "unresolved-action critic-only eligibility is disabled in V1"
        )


def _base_failures(
    aligned: AlignedDecision,
    legal_actions: CanonicalLegalActionSet,
    chosen: CanonicalChosenAction,
) -> tuple[ValidationFailure, ...]:
    failures = list(aligned.failures)
    if aligned_decision_integrity_failures(aligned):
        failures.append(ValidationFailure.CAUSAL_ALIGNMENT)
    failures.extend(legal_actions.failures)
    failures.extend(chosen.failures)
    event = aligned.pre_action_event
    proof = aligned.proof
    evidence_events = [event, *aligned.action_events]
    if aligned.resulting_event is not None:
        evidence_events.append(aligned.resulting_event)
    for evidence_event in evidence_events:
        try:
            integrity_failures = source_event_integrity_failures(evidence_event)
        except (TypeError, ValueError):
            integrity_failures = ("source_event_integrity_check_failed",)
        if integrity_failures:
            failures.append(ValidationFailure.SOURCE_CORRUPT)
    if event.source_corrupt:
        failures.append(ValidationFailure.SOURCE_CORRUPT)
    supported_versions = SUPPORTED_SOURCE_VERSIONS_BY_FORMAT.get(event.source_format)
    if (
        event.action_schema_version != EXPECTED_TRANSITION_SCHEMA
        or supported_versions is None
        or event.source_version not in supported_versions
        or event.information_mode != InformationMode.ACTOR_VISIBLE.value
    ):
        failures.append(ValidationFailure.SCHEMA_MISMATCH)
    try:
        assert_deployable_observation(
            event.observation,
            expected_mode=InformationMode.ACTOR_VISIBLE.value,
        )
    except ValueError:
        failures.append(ValidationFailure.PRIVILEGED_OBSERVATION)
    if not proof.actor_seat_proven or event.actor_seat != event.controlled_player:
        failures.append(ValidationFailure.SEAT_IDENTITY)
    asset = event.observation_asset
    if asset is None:
        failures.append(ValidationFailure.SOURCE_CORRUPT)
    else:
        if (
            asset.admission_identity != event.admission_identity
            or asset.source_game_id != event.source_game_id
        ):
            failures.append(ValidationFailure.SOURCE_CORRUPT)
        if asset.actor_seat != event.actor_seat:
            failures.append(ValidationFailure.SEAT_IDENTITY)
        if asset.observation_id != proof.pre_action_observation_id:
            failures.append(ValidationFailure.CAUSAL_ALIGNMENT)
    if proof.pre_action_event_id != event.event_id:
        failures.append(ValidationFailure.CAUSAL_ALIGNMENT)
    if proof.action_event_ids != tuple(
        action_event.event_id for action_event in aligned.action_events
    ):
        failures.append(ValidationFailure.CAUSAL_ALIGNMENT)
    for action_event in aligned.action_events:
        if (
            action_event.admission_identity != event.admission_identity
            or action_event.source_game_id != event.source_game_id
        ):
            failures.append(ValidationFailure.CAUSAL_ALIGNMENT)
        if action_event.actor_seat != event.actor_seat:
            failures.append(ValidationFailure.SEAT_IDENTITY)
    result_event = aligned.resulting_event
    if result_event is not None:
        result_asset = result_event.observation_asset
        if (
            result_event.admission_identity != event.admission_identity
            or result_event.source_game_id != event.source_game_id
        ):
            failures.append(ValidationFailure.CAUSAL_ALIGNMENT)
        if result_event.actor_seat != event.actor_seat:
            failures.append(ValidationFailure.SEAT_IDENTITY)
        if (
            result_asset is None
            or result_asset.observation_id != proof.resulting_observation_id
        ):
            failures.append(ValidationFailure.RESULTING_TRANSITION)
    if not (
        proof.action_equality_proven
        and proof.event_order_proven
        and proof.source_hash_bound
        and proof.pre_action_observation_id
    ):
        failures.append(ValidationFailure.CAUSAL_ALIGNMENT)
    if not proof.resulting_transition_proven or not proof.resulting_event_id:
        failures.append(ValidationFailure.RESULTING_TRANSITION)
    if legal_actions.observation_id != proof.pre_action_observation_id:
        failures.append(ValidationFailure.STALE_LEGAL_SET)
    if proof.legal_observation_id != proof.pre_action_observation_id:
        failures.append(ValidationFailure.STALE_LEGAL_SET)
    if (
        legal_actions.complete_action_count is None
        or legal_actions.complete_action_count <= 0
    ):
        failures.append(ValidationFailure.UNRESOLVED_ACTION)
    if not _source_semantics_match(aligned, legal_actions, chosen):
        failures.append(ValidationFailure.SOURCE_CORRUPT)

    return _unique_failures(failures)


def _validate_authoritative_decision(
    aligned: AlignedDecision,
    legal_actions: CanonicalLegalActionSet,
    chosen: CanonicalChosenAction,
    policy: CriticEligibilityPolicy,
) -> ValidationResult:
    failures = _base_failures(aligned, legal_actions, chosen)
    forced = not failures and legal_actions.complete_action_count == 1
    if not failures:
        outcome = (
            ValidationOutcome.VALID_CRITIC_ONLY_FORCED
            if forced
            else ValidationOutcome.VALID_ACTOR_AND_CRITIC
        )
        return ValidationResult(
            outcome=outcome,
            first_failure=None,
            failures=(),
            actor_loss_weight=0.0 if forced else 1.0,
            critic_eligible=True,
            forced_decision=forced,
            canonical_decision_id=None,
            resulting_transition_id=aligned.proof.resulting_event_id,
            diagnostics={
                "complete_legal_action_count": legal_actions.complete_action_count,
                "alignment_rule_id": aligned.proof.rule_id,
                "critic_policy_version": policy.version,
            },
        )
    first = failures[0]
    critic_eligible = False
    outcome = OUTCOME_BY_FAILURE.get(first, ValidationOutcome.UNKNOWN_FAIL_CLOSED)

    return ValidationResult(
        outcome=outcome,
        first_failure=first,
        failures=failures,
        actor_loss_weight=0.0,
        critic_eligible=critic_eligible,
        forced_decision=False,
        canonical_decision_id=None,
        resulting_transition_id=aligned.proof.resulting_event_id,
        diagnostics={
            "complete_legal_action_count": legal_actions.complete_action_count,
            "alignment_rule_id": aligned.proof.rule_id,
            "critic_policy_version": policy.version,
            "unresolved_critic_enabled": False,
        },
    )


def validate_decision(
    aligned: AlignedDecision,
    legal_actions: CanonicalLegalActionSet,
    chosen: CanonicalChosenAction,
    *,
    critic_policy: CriticEligibilityPolicy | None = None,
) -> ValidationResult:
    """Validate only the exact legal/chosen reconstruction owned by V1."""

    policy = CriticEligibilityPolicy() if critic_policy is None else critic_policy
    _validate_critic_policy(policy)
    authoritative_legal_actions, authoritative_chosen = (
        _require_authoritative_reconstruction(aligned, legal_actions, chosen)
    )

    return _validate_authoritative_decision(
        aligned,
        authoritative_legal_actions,
        authoritative_chosen,
        policy,
    )


def _build_authoritative_decision_record(
    aligned: AlignedDecision,
    legal_actions: CanonicalLegalActionSet,
    chosen: CanonicalChosenAction,
    validation: ValidationResult,
    implementation_hashes: CanonicalDecisionImplementationHashes,
) -> CanonicalDecisionRecord:
    event = aligned.pre_action_event
    observation_id = aligned.proof.pre_action_observation_id or "unresolved-observation"
    actor_seat = event.actor_seat if event.actor_seat is not None else -1
    decision_id = None
    if validation.outcome in CANONICAL_ID_OUTCOMES:
        if chosen.chosen_action_id is None or validation.resulting_transition_id is None:
            raise CanonicalDecisionValidationHardStop(
                "positive validation lacks a chosen action or resulting transition identity"
            )
        decision_id = canonical_decision_id(
            schema_version=DECISION_RECORD_SCHEMA_VERSION,
            admission_identity=event.admission_identity,
            source_game_id=event.source_game_id,
            actor_seat=actor_seat,
            causal_decision_index=aligned.causal_decision_index,
            pre_action_observation_id=observation_id,
            legal_action_set_id=legal_actions.legal_action_set_id,
            chosen_action_id=chosen.chosen_action_id,
            resulting_transition_id=validation.resulting_transition_id,
            causal_aligner_version=CAUSAL_ALIGNER_VERSION,
            causal_aligner_implementation_sha256=(
                implementation_hashes.causal_aligner_implementation_sha256
            ),
            legality_validator_version=VALIDATOR_VERSION,
            legality_validator_implementation_sha256=(
                implementation_hashes.legality_validator_implementation_sha256
            ),
            action_schema_version=CHOSEN_ACTION_SCHEMA_VERSION,
            action_schema_implementation_sha256=(
                implementation_hashes.action_schema_implementation_sha256
            ),
        )
    bound_validation = replace(validation, canonical_decision_id=decision_id)

    return CanonicalDecisionRecord(
        admission_identity=event.admission_identity,
        source_game_id=event.source_game_id,
        actor_seat=actor_seat,
        causal_decision_index=aligned.causal_decision_index,
        pre_action_observation_id=observation_id,
        legal_action_set_id=legal_actions.legal_action_set_id,
        chosen_action_id=chosen.chosen_action_id,
        resulting_transition_id=validation.resulting_transition_id,
        causal_aligner_version=CAUSAL_ALIGNER_VERSION,
        causal_aligner_implementation_sha256=(
            implementation_hashes.causal_aligner_implementation_sha256
        ),
        legality_validator_version=VALIDATOR_VERSION,
        legality_validator_implementation_sha256=(
            implementation_hashes.legality_validator_implementation_sha256
        ),
        action_schema_version=CHOSEN_ACTION_SCHEMA_VERSION,
        action_schema_implementation_sha256=(
            implementation_hashes.action_schema_implementation_sha256
        ),
        validation=bound_validation,
        source_lineage={
            "binding_id": event.binding_id,
            "local_row": event.local_row,
            "global_row": event.global_row,
            "source_id": event.source_id,
            "source_format": event.source_format,
            "source_version": event.source_version,
            "source_row_sha256": event.source_row_sha256,
            "source_artifact_sha256": event.source_artifact_sha256,
            "alignment_rule_id": aligned.proof.rule_id,
            "deterministic_repair_attempted": aligned.proof.deterministic_repair_attempted,
            "deterministic_repair_applied": aligned.proof.deterministic_repair_applied,
        },
    )


def build_decision_record(
    aligned: AlignedDecision,
    legal_actions: CanonicalLegalActionSet,
    chosen: CanonicalChosenAction,
    validation: ValidationResult,
    *,
    implementation_hashes: CanonicalDecisionImplementationHashes,
) -> CanonicalDecisionRecord:
    """Revalidate all evidence before binding a canonical decision record."""

    _validate_implementation_hashes(implementation_hashes)
    authoritative_validation = validate_decision(aligned, legal_actions, chosen)
    if not _contract_values_match(validation, authoritative_validation):
        raise CanonicalDecisionValidationHardStop(
            "caller-supplied validation result differs from exact revalidation"
        )

    return _build_authoritative_decision_record(
        aligned,
        legal_actions,
        chosen,
        authoritative_validation,
        implementation_hashes,
    )


def canonicalize_decision(
    aligned: AlignedDecision,
    *,
    implementation_hashes: CanonicalDecisionImplementationHashes,
    critic_policy: CriticEligibilityPolicy | None = None,
) -> tuple[CanonicalDecisionRecord, CanonicalLegalActionSet, CanonicalChosenAction]:
    _validate_implementation_hashes(implementation_hashes)
    policy = CriticEligibilityPolicy() if critic_policy is None else critic_policy
    _validate_critic_policy(policy)
    legal_actions = reconstruct_legal_actions(aligned)
    chosen = reconstruct_selected_action(aligned, legal_actions)
    validation = _validate_authoritative_decision(
        aligned,
        legal_actions,
        chosen,
        policy,
    )
    record = _build_authoritative_decision_record(
        aligned,
        legal_actions,
        chosen,
        validation,
        implementation_hashes,
    )

    return record, legal_actions, chosen


_VALIDATION_ENVELOPE_CREATION_TOKEN = object()
_VALIDATION_ENVELOPE_HMAC_KEY = secrets.token_bytes(32)


class ValidatedCanonicalDecisionEnvelope:
    """Ephemeral builder-sealed evidence for one storage-bound decision."""

    __slots__ = (
        "_aligned",
        "_chosen",
        "_critic_policy",
        "_implementation_hashes",
        "_legal_actions",
        "_record",
        "_seal",
    )

    def __init__(
        self,
        *,
        _creation_token: object,
        aligned: AlignedDecision,
        record: CanonicalDecisionRecord,
        legal_actions: CanonicalLegalActionSet,
        chosen: CanonicalChosenAction,
        implementation_hashes: CanonicalDecisionImplementationHashes,
        critic_policy: CriticEligibilityPolicy,
    ) -> None:
        if _creation_token is not _VALIDATION_ENVELOPE_CREATION_TOKEN:
            raise CanonicalDecisionValidationHardStop(
                "validated decision envelopes must be created by the canonical builder"
            )
        self._aligned = aligned
        self._record = record
        self._legal_actions = legal_actions
        self._chosen = chosen
        self._implementation_hashes = implementation_hashes
        self._critic_policy = critic_policy
        self._seal = _validation_envelope_seal(self)

    @property
    def aligned(self) -> AlignedDecision:

        return self._aligned

    @property
    def record(self) -> CanonicalDecisionRecord:

        return self._record

    @property
    def legal_actions(self) -> CanonicalLegalActionSet:

        return self._legal_actions

    @property
    def chosen(self) -> CanonicalChosenAction:

        return self._chosen

    @property
    def implementation_hashes(self) -> CanonicalDecisionImplementationHashes:

        return self._implementation_hashes

    @property
    def critic_policy(self) -> CriticEligibilityPolicy:

        return self._critic_policy

    def __repr__(self) -> str:

        return "<ValidatedCanonicalDecisionEnvelope ephemeral>"


def _validation_envelope_payload(
    envelope: ValidatedCanonicalDecisionEnvelope,
) -> dict[str, Any]:

    return {
        "aligned": envelope.aligned,
        "record": envelope.record,
        "legal_actions": envelope.legal_actions,
        "chosen": envelope.chosen,
        "implementation_hashes": envelope.implementation_hashes,
        "critic_policy": envelope.critic_policy,
    }


def _validation_envelope_seal(
    envelope: ValidatedCanonicalDecisionEnvelope,
) -> str:
    payload = canonical_json_bytes(_validation_envelope_payload(envelope))

    return hmac.new(
        _VALIDATION_ENVELOPE_HMAC_KEY,
        payload,
        hashlib.sha256,
    ).hexdigest()


def build_validated_decision_envelope(
    aligned: AlignedDecision,
    *,
    implementation_hashes: CanonicalDecisionImplementationHashes,
    critic_policy: CriticEligibilityPolicy | None = None,
) -> ValidatedCanonicalDecisionEnvelope:
    """Canonicalize evidence and issue an ephemeral storage-bound seal."""

    policy = CriticEligibilityPolicy() if critic_policy is None else critic_policy
    record, legal_actions, chosen = canonicalize_decision(
        aligned,
        implementation_hashes=implementation_hashes,
        critic_policy=policy,
    )

    return ValidatedCanonicalDecisionEnvelope(
        _creation_token=_VALIDATION_ENVELOPE_CREATION_TOKEN,
        aligned=aligned,
        record=record,
        legal_actions=legal_actions,
        chosen=chosen,
        implementation_hashes=implementation_hashes,
        critic_policy=policy,
    )


def revalidate_decision_envelope(
    envelope: ValidatedCanonicalDecisionEnvelope,
) -> CanonicalDecisionRecord:
    """Replay canonicalization and exact-compare a builder-sealed envelope."""

    if type(envelope) is not ValidatedCanonicalDecisionEnvelope:
        raise CanonicalDecisionValidationHardStop(
            "positive storage input is not a builder-sealed decision envelope"
        )
    try:
        current_seal = _validation_envelope_seal(envelope)
    except Exception as exc:
        raise CanonicalDecisionValidationHardStop(
            "validated decision envelope evidence cannot be re-encoded"
        ) from exc
    if not hmac.compare_digest(envelope._seal, current_seal):
        raise CanonicalDecisionValidationHardStop(
            "validated decision envelope evidence changed after validation"
        )
    record, legal_actions, chosen = canonicalize_decision(
        envelope.aligned,
        implementation_hashes=envelope.implementation_hashes,
        critic_policy=envelope.critic_policy,
    )
    if not (
        _contract_values_match(record, envelope.record)
        and _contract_values_match(legal_actions, envelope.legal_actions)
        and _contract_values_match(chosen, envelope.chosen)
    ):
        raise CanonicalDecisionValidationHardStop(
            "validated decision envelope disagrees with independent replay"
        )

    return record


__all__ = [
    "ACTION_SEMANTIC_FAILURES",
    "CanonicalDecisionValidationHardStop",
    "EXPECTED_TRANSITION_SCHEMA",
    "FAILURE_PRECEDENCE",
    "SUPPORTED_SOURCE_VERSIONS_BY_FORMAT",
    "ValidatedCanonicalDecisionEnvelope",
    "build_validated_decision_envelope",
    "canonicalize_decision",
    "build_decision_record",
    "reconstruct_legal_actions",
    "reconstruct_selected_action",
    "revalidate_decision_envelope",
    "validate_decision",
]
