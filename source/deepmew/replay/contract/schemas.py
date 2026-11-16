"""Versioned schemas for DeepMew's canonical replay decision boundary.

These types are research data contracts.  They do not implement gameplay and
must not be imported by protected competition runtime surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


CONTRACT_SCHEMA_VERSION = "deepmew.canonical_decision_contract.v1"
SOURCE_EVENT_SCHEMA_VERSION = "deepmew.canonical_source_event.v1"
OBSERVATION_ASSET_SCHEMA_VERSION = "deepmew.canonical_observation_asset.v1"
LEGAL_ACTION_SCHEMA_VERSION = "deepmew.canonical_legal_action_set.v1"
CHOSEN_ACTION_SCHEMA_VERSION = "deepmew.canonical_chosen_action.v1"
DECISION_RECORD_SCHEMA_VERSION = "deepmew.canonical_decision_record.v1"
VALIDATOR_VERSION = "deepmew.canonical_decision_validator.v1"
CRITIC_ELIGIBILITY_VERSION = "deepmew.critic_only_eligibility.v1"


class ValidationOutcome(str, Enum):
    VALID_ACTOR_AND_CRITIC = "VALID_ACTOR_AND_CRITIC"
    VALID_CRITIC_ONLY_FORCED = "VALID_CRITIC_ONLY_FORCED"
    VALID_CRITIC_ONLY_UNRESOLVED_ACTION = "VALID_CRITIC_ONLY_UNRESOLVED_ACTION"
    REJECTED_MISSING_REQUIRED_CHOICE = "REJECTED_MISSING_REQUIRED_CHOICE"
    REJECTED_TOO_FEW_CHOICES = "REJECTED_TOO_FEW_CHOICES"
    REJECTED_TOO_MANY_CHOICES = "REJECTED_TOO_MANY_CHOICES"
    REJECTED_CHOICE_NOT_LEGAL = "REJECTED_CHOICE_NOT_LEGAL"
    REJECTED_CHOICE_ORDER = "REJECTED_CHOICE_ORDER"
    REJECTED_DUPLICATE_FORBIDDEN_CHOICE = "REJECTED_DUPLICATE_FORBIDDEN_CHOICE"
    REJECTED_ACTION_TEMPLATE_NOT_LEGAL = "REJECTED_ACTION_TEMPLATE_NOT_LEGAL"
    REJECTED_STALE_LEGAL_SET = "REJECTED_STALE_LEGAL_SET"
    REJECTED_CAUSAL_ALIGNMENT = "REJECTED_CAUSAL_ALIGNMENT"
    REJECTED_SEAT_IDENTITY = "REJECTED_SEAT_IDENTITY"
    REJECTED_SCHEMA_MISMATCH = "REJECTED_SCHEMA_MISMATCH"
    REJECTED_PRIVILEGED_OBSERVATION = "REJECTED_PRIVILEGED_OBSERVATION"
    REJECTED_SOURCE_CORRUPT = "REJECTED_SOURCE_CORRUPT"
    UNKNOWN_FAIL_CLOSED = "UNKNOWN_FAIL_CLOSED"


class ValidationFailure(str, Enum):
    MISSING_REQUIRED_CHOICE = "missing_required_choice"
    TOO_FEW_CHOICES = "too_few_choices"
    TOO_MANY_CHOICES = "too_many_choices"
    CHOICE_NOT_LEGAL = "choice_not_legal"
    CHOICE_ORDER = "choice_order_invalid_or_unresolved"
    DUPLICATE_FORBIDDEN_CHOICE = "duplicate_forbidden_choice"
    ACTION_TEMPLATE_NOT_LEGAL = "action_template_not_legal"
    STALE_LEGAL_SET = "stale_legal_set"
    CAUSAL_ALIGNMENT = "causal_alignment_unproven"
    SEAT_IDENTITY = "seat_identity_mismatch"
    SCHEMA_MISMATCH = "schema_mismatch"
    PRIVILEGED_OBSERVATION = "privileged_observation"
    SOURCE_CORRUPT = "source_corrupt"
    UNRESOLVED_ACTION = "unresolved_action_encoding"
    RESULTING_TRANSITION = "resulting_transition_unproven"
    UNKNOWN = "unknown_fail_closed"


class OrderingSignificance(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    ORDERED = "ordered"
    UNORDERED = "unordered"
    UNKNOWN = "unknown_fail_closed"


class ChoicePresence(str, Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"


@dataclass(frozen=True)
class ObservationAssetReference:
    admission_identity: str
    source_sha256: str
    source_path: str
    source_line: int
    source_content_sha256: str
    content_sha256: str
    source_game_id: str
    source_event_position: int
    actor_seat: int
    information_mode: str
    observation_schema_version: str
    visibility_schema_version: str
    asset_origin: str
    reusable_admission_asset: bool
    feature_version: str | None = None
    tensor_sha256: str | None = None
    observation_id: str = ""


@dataclass(frozen=True)
class SourceEvent:
    admission_identity: str
    binding_id: str
    local_row: int
    global_row: int
    source_id: str
    source_format: str
    source_version: str
    source_game_id: str
    source_event_position: int
    actor_seat: int | None
    controlled_player: int | None
    observation_asset: ObservationAssetReference | None
    observation: Mapping[str, Any] | None
    stored_selected_choices: tuple[int, ...] | None
    stored_index_space: str | None
    legacy_legal_actions: tuple[Mapping[str, Any], ...]
    transition: Mapping[str, Any]
    information_mode: str
    action_schema_version: str
    observation_schema_version: str
    source_row_sha256: str
    source_native_game_id: str = ""
    source_record: Mapping[str, Any] = field(default_factory=dict)
    source_artifact_path: str | None = None
    source_artifact_sha256: str | None = None
    source_corrupt: bool = False
    event_id: str = ""


@dataclass(frozen=True)
class AlignmentProof:
    rule_id: str
    source_scope: str
    pre_action_event_id: str
    action_event_ids: tuple[str, ...]
    resulting_event_id: str | None
    pre_action_observation_id: str | None
    legal_observation_id: str | None
    resulting_observation_id: str | None
    action_equality_proven: bool
    actor_seat_proven: bool
    event_order_proven: bool
    resulting_transition_proven: bool
    source_hash_bound: bool
    deterministic_repair_attempted: bool
    deterministic_repair_applied: bool
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AlignedDecision:
    pre_action_event: SourceEvent
    action_events: tuple[SourceEvent, ...]
    resulting_event: SourceEvent | None
    stage_selected_choices: tuple[tuple[int, ...] | None, ...]
    stage_index_spaces: tuple[str | None, ...]
    causal_decision_index: int
    proof: AlignmentProof
    failures: tuple[ValidationFailure, ...] = ()


@dataclass(frozen=True)
class CanonicalLegalOption:
    stage_id: str
    position: int
    option_id: str
    payload_sha256: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class ConditionalOptionSet:
    prior_stage_option_ids: tuple[tuple[str, ...], ...]
    option_ids: tuple[str, ...]


@dataclass(frozen=True)
class ChoiceStageContract:
    stage_id: str
    stage_index: int
    stage_name: str
    presence: ChoicePresence
    minimum_count: int
    maximum_count: int
    repetition_allowed: bool
    ordering_significance: OrderingSignificance
    options: tuple[CanonicalLegalOption, ...]
    conditional_options: tuple[ConditionalOptionSet, ...] = ()


@dataclass(frozen=True)
class CanonicalLegalActionSet:
    observation_id: str
    action_family: str
    action_template: str
    stages: tuple[ChoiceStageContract, ...]
    complete_action_count: int | None
    legal_action_set_id: str
    source_select_type: int | str | None
    source_select_context: int | str | None
    failures: tuple[ValidationFailure, ...] = ()


@dataclass(frozen=True)
class CanonicalChosenStage:
    stage_id: str
    selected_option_ids: tuple[str, ...]
    source_selected_values: tuple[int, ...]
    omitted_optional: bool


@dataclass(frozen=True)
class CanonicalChosenAction:
    action_template: str
    stages: tuple[CanonicalChosenStage, ...]
    chosen_action_id: str | None
    complete: bool
    failures: tuple[ValidationFailure, ...] = ()


@dataclass(frozen=True)
class CriticEligibilityPolicy:
    version: str = CRITIC_ELIGIBILITY_VERSION
    require_actor_visible_observation: bool = True
    require_causal_trajectory_position: bool = True
    require_outcome_or_return: bool = True
    allow_unresolved_action_for_critic: bool = False
    allow_unresolved_action_for_actor_loss: bool = False


@dataclass(frozen=True)
class CanonicalDecisionImplementationHashes:
    """Exact implementations that own canonical decision semantics."""

    causal_aligner_implementation_sha256: str
    legality_validator_implementation_sha256: str
    action_schema_implementation_sha256: str


@dataclass(frozen=True)
class ValidationResult:
    outcome: ValidationOutcome
    first_failure: ValidationFailure | None
    failures: tuple[ValidationFailure, ...]
    actor_loss_weight: float
    critic_eligible: bool
    forced_decision: bool
    canonical_decision_id: str | None
    resulting_transition_id: str | None
    validator_version: str = VALIDATOR_VERSION
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalDecisionRecord:
    admission_identity: str
    source_game_id: str
    actor_seat: int
    causal_decision_index: int
    pre_action_observation_id: str
    legal_action_set_id: str
    chosen_action_id: str | None
    resulting_transition_id: str | None
    causal_aligner_version: str
    causal_aligner_implementation_sha256: str
    legality_validator_version: str
    legality_validator_implementation_sha256: str
    action_schema_version: str
    action_schema_implementation_sha256: str
    validation: ValidationResult
    source_lineage: Mapping[str, Any]
    schema_version: str = DECISION_RECORD_SCHEMA_VERSION


__all__ = [
    "CHOSEN_ACTION_SCHEMA_VERSION",
    "CONTRACT_SCHEMA_VERSION",
    "CRITIC_ELIGIBILITY_VERSION",
    "DECISION_RECORD_SCHEMA_VERSION",
    "LEGAL_ACTION_SCHEMA_VERSION",
    "OBSERVATION_ASSET_SCHEMA_VERSION",
    "SOURCE_EVENT_SCHEMA_VERSION",
    "VALIDATOR_VERSION",
    "AlignedDecision",
    "AlignmentProof",
    "CanonicalChosenAction",
    "CanonicalChosenStage",
    "CanonicalDecisionImplementationHashes",
    "CanonicalDecisionRecord",
    "CanonicalLegalActionSet",
    "CanonicalLegalOption",
    "ChoicePresence",
    "ChoiceStageContract",
    "ConditionalOptionSet",
    "CriticEligibilityPolicy",
    "ObservationAssetReference",
    "OrderingSignificance",
    "SourceEvent",
    "ValidationFailure",
    "ValidationOutcome",
    "ValidationResult",
]
