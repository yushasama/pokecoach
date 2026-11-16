"""Canonical replay-decision semantics owned by :mod:`deepmew.replay`.

All serious research data consumers must use this contract or an explicitly
versioned successor.  Admission V0 remains immutable structural evidence.
"""

from deepmew.replay.contract.schemas import (
    CONTRACT_SCHEMA_VERSION,
    AlignedDecision,
    CanonicalChosenAction,
    CanonicalDecisionImplementationHashes,
    CanonicalDecisionRecord,
    CanonicalLegalActionSet,
    CriticEligibilityPolicy,
    SourceEvent,
    ValidationFailure,
    ValidationOutcome,
    ValidationResult,
)
from deepmew.replay.contract.causal_alignment import (
    CAUSAL_ALIGNER_VERSION,
    KaggleEpisodeRepository,
    align_decision,
)
from deepmew.replay.contract.validation import (
    CanonicalDecisionValidationHardStop,
    canonicalize_decision,
)


CANONICAL_DECISION_CONTRACT_V1 = CONTRACT_SCHEMA_VERSION


__all__ = [
    "CANONICAL_DECISION_CONTRACT_V1",
    "AlignedDecision",
    "CAUSAL_ALIGNER_VERSION",
    "CanonicalChosenAction",
    "CanonicalDecisionImplementationHashes",
    "CanonicalDecisionRecord",
    "CanonicalDecisionValidationHardStop",
    "CanonicalLegalActionSet",
    "CriticEligibilityPolicy",
    "KaggleEpisodeRepository",
    "SourceEvent",
    "ValidationFailure",
    "ValidationOutcome",
    "ValidationResult",
    "align_decision",
    "canonicalize_decision",
]
