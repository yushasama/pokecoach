"""One NumPy-only owner for snapshot-BC row validation and exclusion rules."""

from __future__ import annotations

import numpy as np

from deepmew_ex.data import Decision

EXCLUSION_PRECEDENCE = ('forced', 'zero_actor_weight', 'optional_empty', 'multi_select', 'unsupported_selection')


def validate_decision(decision: Decision) -> None:
    options, values = decision.option_features, decision.value_features
    if options.ndim != 2 or options.shape[1] != 304 or values.shape != (144,) or len(options) == 0:
        raise ValueError(f'invalid_feature_shape: {decision.row_id}')
    if options.dtype != np.float32 or values.dtype != np.float32 or decision.selection.dtype.kind not in 'iu' or decision.selection.ndim != 1:
        raise ValueError(f'invalid_feature_or_selection_dtype: {decision.row_id}')
    if not np.isfinite(options).all() or not np.isfinite(values).all() or not np.isfinite(decision.actor_loss_weight):
        raise ValueError(f'nonfinite_actor_data: {decision.row_id}')
    if decision.actor_loss_weight < 0 or decision.complete_legal_action_count < 1 or not 0 <= decision.min_count <= decision.max_count:
        raise ValueError(f'invalid_legal_bounds_or_weight: {decision.row_id}')
    if decision.min_count == decision.max_count == 1 and decision.complete_legal_action_count != len(options):
        raise ValueError(f'incomplete_single_selection_options: {decision.row_id}')
    # Unsupported prompts can have combinatorial action counts and repeated selections.
    # V1 has no ordering/repetition contract, so do not invent one from candidate count.
    if any(not 0 <= int(target) < len(options) for target in decision.selection):
        raise ValueError(f'invalid_target_index: {decision.row_id}')


def exclusion_reason(decision: Decision) -> str | None:
    if decision.complete_legal_action_count == 1:
        return 'forced'
    if decision.actor_loss_weight == 0:
        return 'zero_actor_weight'
    if decision.min_count == 0:
        return 'optional_empty'
    if decision.max_count != 1 or decision.min_count != 1:
        return 'multi_select'
    if len(decision.selection) != 1:
        return 'unsupported_selection'
    return None
