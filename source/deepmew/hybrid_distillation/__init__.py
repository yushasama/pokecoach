"""Research-only contracts for state-conditioned multi-parent distillation."""

from .contracts import (
    COMPLETE_ACTION_DISTRIBUTION_SCHEMA_VERSION,
    HYBRID_EXAMPLE_SCHEMA_VERSION,
    HybridContractError,
    assert_student_input_safe,
    canonical_complete_action_ids,
    mix_complete_action_distributions,
    one_stage_complete_action_ids,
    semantic_legal_action_ids,
    validate_complete_action_distribution,
    validate_game_group_splits,
    validate_parent_arbitration,
)
from .corpus import (
    COLLECTOR_CAPABILITY_SCHEMA_VERSION,
    REQUIRED_COMPOUND_ACTION_FIELDS,
    audit_collector_capability,
)

__all__ = [
    "COMPLETE_ACTION_DISTRIBUTION_SCHEMA_VERSION",
    "COLLECTOR_CAPABILITY_SCHEMA_VERSION",
    "HYBRID_EXAMPLE_SCHEMA_VERSION",
    "HybridContractError",
    "REQUIRED_COMPOUND_ACTION_FIELDS",
    "assert_student_input_safe",
    "audit_collector_capability",
    "canonical_complete_action_ids",
    "mix_complete_action_distributions",
    "one_stage_complete_action_ids",
    "semantic_legal_action_ids",
    "validate_complete_action_distribution",
    "validate_game_group_splits",
    "validate_parent_arbitration",
]
