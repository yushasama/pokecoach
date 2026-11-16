"""Fail-closed data contracts for research-only hybrid distillation.

The contract deliberately operates on complete compound actions. Parent
arbitration happens once per select prompt, before any token of a multi-select
action is consumed, so a child target can never switch experts midway through
one compound action.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from deepmew.replay.information_boundary import (
    InformationMode,
    assert_deployable_observation,
)
from deepmew.replay.contract.legal_actions import allowed_option_ids
from deepmew.replay.contract.schemas import (
    CanonicalLegalActionSet,
    ChoiceStageContract,
    OrderingSignificance,
)


COMPLETE_ACTION_DISTRIBUTION_SCHEMA_VERSION = (
    "deepmew.hybrid.complete_action_distribution.v1"
)
HYBRID_EXAMPLE_SCHEMA_VERSION = "deepmew.hybrid.distillation_example.v1"
DEFAULT_PROBABILITY_TOLERANCE = 1e-6
DEFAULT_MAX_EXACT_SUPPORT = 100_000

# These surfaces may remain in audit provenance, but they are never valid
# inputs to a deployable student/router.
_FORBIDDEN_STUDENT_KEY_FRAGMENTS = (
    "actor_policy",
    "branch_q",
    "checkpoint",
    "competence_bucket",
    "controlled_stratum",
    "deck_hash",
    "deck_id",
    "hidden",
    "member_id",
    "opponent_policy",
    "oracle",
    "outcome",
    "parent_alias",
    "parent_id",
    "policy_id",
    "result",
    "search_begin_input",
    "specialization",
    "teacher",
    "teacher_id",
    "true_private",
)


class HybridContractError(ValueError):
    """Raised when a hybrid artifact would violate the frozen contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _semantic_payload(action: Mapping[str, Any]) -> dict[str, Any]:
    # `index` is presentation order, not action meaning. Source/target indices
    # remain because they distinguish concrete cards or board slots.
    return {str(key): value for key, value in action.items() if key != "index"}


def semantic_legal_action_ids(legal_actions: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Return deterministic IDs for one state's semantic legal-action table."""

    if not legal_actions:
        raise HybridContractError("a legal-action table must be non-empty")
    occurrences: dict[str, int] = {}
    action_ids: list[str] = []
    for action in legal_actions:
        if not isinstance(action, Mapping):
            raise HybridContractError("each legal action must be a mapping")
        payload = _canonical_json(_semantic_payload(action))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        occurrence = occurrences.get(digest, 0)
        occurrences[digest] = occurrence + 1
        action_ids.append(f"la1-{digest}:{occurrence}")

    return tuple(action_ids)


def _require_action_group_id(action_group_id: str) -> str:
    normalized = str(action_group_id).strip()
    if not normalized:
        raise HybridContractError("complete-action support requires a durable action_group_id")

    return normalized


def one_stage_complete_action_ids(
    legal_action_ids: Sequence[str],
    min_count: int,
    max_count: int,
    *,
    action_group_id: str,
    max_support: int = DEFAULT_MAX_EXACT_SUPPORT,
) -> tuple[str, ...]:
    """Enumerate the explicit one-stage special case.

    This helper must not be used for a staged or conditionally staged CABT
    action. Those actions require :func:`canonical_complete_action_ids`.
    """

    group_id = _require_action_group_id(action_group_id)
    option_count = len(legal_action_ids)
    if option_count == 0:
        raise HybridContractError("compound-action support requires legal actions")
    if min_count < 0 or max_count < min_count or max_count > option_count:
        raise HybridContractError("invalid min_count/max_count for legal-action table")
    support_size = sum(
        math.factorial(option_count) // math.factorial(option_count - count)
        for count in range(min_count, max_count + 1)
    )
    if support_size > max_support:
        raise HybridContractError(
            f"exact compound-action support {support_size} exceeds max_support={max_support}"
        )

    support: list[str] = []
    for count in range(min_count, max_count + 1):
        for selection in itertools.permutations(legal_action_ids, count):
            payload = {
                "action_group_id": group_id,
                "stages": [{"stage_id": "one-stage-special-case", "selected_option_ids": list(selection)}],
            }
            digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
            support.append(f"ca1-{digest}")

    return tuple(support)


def _canonical_stage_selections(
    stage: ChoiceStageContract,
    allowed_ids: frozenset[str],
) -> tuple[tuple[str, ...], ...]:
    if stage.ordering_significance == OrderingSignificance.UNKNOWN:
        raise HybridContractError(f"stage {stage.stage_id} has unresolved ordering significance")
    option_ids = tuple(
        option.option_id
        for option in stage.options
        if option.option_id in allowed_ids
    )
    selections: list[tuple[str, ...]] = []
    for selected_count in range(stage.minimum_count, stage.maximum_count + 1):
        if stage.repetition_allowed:
            if stage.ordering_significance == OrderingSignificance.ORDERED:
                iterator = itertools.product(option_ids, repeat=selected_count)
            else:
                iterator = itertools.combinations_with_replacement(option_ids, selected_count)
        elif stage.ordering_significance == OrderingSignificance.ORDERED:
            iterator = itertools.permutations(option_ids, selected_count)
        else:
            iterator = itertools.combinations(option_ids, selected_count)
        selections.extend(iterator)

    return tuple(selections)


def canonical_complete_action_ids(
    legal_actions: CanonicalLegalActionSet,
    *,
    action_group_id: str,
    max_support: int = DEFAULT_MAX_EXACT_SUPPORT,
) -> tuple[str, ...]:
    """Enumerate an exact canonical multi-stage complete-action support."""

    group_id = _require_action_group_id(action_group_id)
    if legal_actions.failures or not legal_actions.stages:
        raise HybridContractError("canonical legal action set is unresolved or empty")
    if legal_actions.complete_action_count is None:
        raise HybridContractError("canonical complete_action_count is unresolved")
    if legal_actions.complete_action_count > max_support:
        raise HybridContractError(
            f"exact compound-action support {legal_actions.complete_action_count} "
            f"exceeds max_support={max_support}"
        )
    expected_stage_indices = tuple(range(len(legal_actions.stages)))
    observed_stage_indices = tuple(stage.stage_index for stage in legal_actions.stages)
    if observed_stage_indices != expected_stage_indices:
        raise HybridContractError("canonical stages are not durably ordered")

    support: list[str] = []

    def visit(stage_index: int, prior: tuple[tuple[str, ...], ...]) -> None:
        if stage_index == len(legal_actions.stages):
            payload = {
                "action_group_id": group_id,
                "legal_action_set_id": legal_actions.legal_action_set_id,
                "stages": [
                    {
                        "stage_id": stage.stage_id,
                        "selected_option_ids": list(selected),
                    }
                    for stage, selected in zip(legal_actions.stages, prior, strict=True)
                ],
            }
            digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
            support.append(f"ca1-{digest}")

            return
        stage = legal_actions.stages[stage_index]
        allowed = allowed_option_ids(stage, prior)
        if allowed is None:
            raise HybridContractError(
                f"stage {stage.stage_id} has an absent or ambiguous conditional path"
            )
        for selected in _canonical_stage_selections(stage, allowed):
            visit(stage_index + 1, prior + (selected,))

    visit(0, ())
    if len(support) != legal_actions.complete_action_count:
        raise HybridContractError(
            "enumerated complete-action support disagrees with canonical complete_action_count"
        )

    return tuple(support)


def validate_complete_action_distribution(
    distribution: Mapping[str, Any],
    expected_support: Sequence[str],
    *,
    action_group_id: str,
    legal_action_set_id: str,
    tolerance: float = DEFAULT_PROBABILITY_TOLERANCE,
) -> dict[str, float]:
    """Require a normalized probability for every complete legal action."""

    if distribution.get("schema_version") != COMPLETE_ACTION_DISTRIBUTION_SCHEMA_VERSION:
        raise HybridContractError("unknown complete-action distribution schema")
    if distribution.get("support_mode") != "exact_enumeration":
        raise HybridContractError("complete-action distribution is not exact enumeration")
    if distribution.get("support_coverage") != 1.0:
        raise HybridContractError("complete-action distribution support coverage is not 1.0")
    if distribution.get("action_group_id") != _require_action_group_id(action_group_id):
        raise HybridContractError("complete-action distribution changed action_group_id")
    if distribution.get("legal_action_set_id") != legal_action_set_id or not legal_action_set_id:
        raise HybridContractError("complete-action distribution changed legal_action_set_id")
    probabilities = distribution.get("probabilities")
    if not isinstance(probabilities, Mapping):
        raise HybridContractError("complete-action probabilities must be a mapping")
    expected = set(expected_support)
    observed = {str(action_id) for action_id in probabilities}
    if observed != expected:
        missing = sorted(expected.difference(observed))
        extra = sorted(observed.difference(expected))
        raise HybridContractError(
            f"complete-action support mismatch: missing={missing[:3]}, extra={extra[:3]}"
        )
    normalized: dict[str, float] = {}
    for action_id, raw_probability in probabilities.items():
        try:
            probability = float(raw_probability)
        except (TypeError, ValueError) as exc:
            raise HybridContractError("complete-action probability is not numeric") from exc
        if not math.isfinite(probability) or probability < 0.0:
            raise HybridContractError("complete-action probability must be finite and non-negative")
        normalized[str(action_id)] = probability
    if not math.isclose(sum(normalized.values()), 1.0, abs_tol=tolerance, rel_tol=0.0):
        raise HybridContractError("complete-action probabilities do not sum to 1")

    return normalized


def validate_parent_arbitration(
    arbitration: Mapping[str, Any],
    parent_ids: Sequence[str],
    *,
    tolerance: float = DEFAULT_PROBABILITY_TOLERANCE,
) -> dict[str, float]:
    """Validate once-per-compound-action state-conditioned parent weights."""

    if arbitration.get("unit") != "complete_compound_action":
        raise HybridContractError("parent arbitration must be fixed for a complete compound action")
    if arbitration.get("timing") != "once_before_first_action_token":
        raise HybridContractError("parent arbitration timing permits mid-action switching")
    mode = arbitration.get("mode")
    if mode not in {"single_parent", "soft_parent_mixture"}:
        raise HybridContractError("unknown parent arbitration mode")
    raw_weights = arbitration.get("parent_weights")
    if not isinstance(raw_weights, Mapping):
        raise HybridContractError("parent arbitration weights must be a mapping")
    expected = set(parent_ids)
    observed = {str(parent_id) for parent_id in raw_weights}
    if observed != expected:
        raise HybridContractError("parent arbitration weights do not cover the frozen parents")
    weights = {str(parent_id): float(weight) for parent_id, weight in raw_weights.items()}
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights.values()):
        raise HybridContractError("parent arbitration weights must be finite and non-negative")
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=tolerance, rel_tol=0.0):
        raise HybridContractError("parent arbitration weights do not sum to 1")
    positive = sum(weight > tolerance for weight in weights.values())
    if mode == "single_parent" and positive != 1:
        raise HybridContractError("single-parent arbitration is not one-hot")
    if mode == "soft_parent_mixture" and positive < 2:
        raise HybridContractError("near-tie arbitration must retain at least two parents")

    return weights


def mix_complete_action_distributions(
    parent_probabilities: Mapping[str, Mapping[str, float]],
    parent_weights: Mapping[str, float],
) -> dict[str, float]:
    """Mix parents at complete-action level, never token-by-token."""

    if set(parent_probabilities) != set(parent_weights):
        raise HybridContractError("parent distributions and weights use different parents")
    supports = [set(distribution) for distribution in parent_probabilities.values()]
    if not supports or any(support != supports[0] for support in supports[1:]):
        raise HybridContractError("parent complete-action supports are not identical")
    mixed = {action_id: 0.0 for action_id in supports[0]}
    for parent_id, distribution in parent_probabilities.items():
        weight = float(parent_weights[parent_id])
        for action_id, probability in distribution.items():
            mixed[action_id] += weight * float(probability)

    return mixed


def _walk_keys(value: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], str]]:
    keys: list[tuple[tuple[str, ...], str]] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).lower()
            keys.append((path, key))
            keys.extend(_walk_keys(child, path + (key,)))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            keys.extend(_walk_keys(child, path))

    return keys


def assert_student_input_safe(student_input: Mapping[str, Any]) -> None:
    """Reject identities, outcomes, oracle fields, and non-deployable observations."""

    observation = student_input.get("actor_visible_observation")
    try:
        assert_deployable_observation(
            observation,
            expected_mode=InformationMode.ACTOR_VISIBLE.value,
        )
    except ValueError as exc:
        raise HybridContractError("student observation is not actor-visible") from exc
    for path, key in _walk_keys(student_input):
        if any(fragment in key for fragment in _FORBIDDEN_STUDENT_KEY_FRAGMENTS):
            location = ".".join(path + (key,))
            raise HybridContractError(f"forbidden student-input key: {location}")


def validate_game_group_splits(
    rows: Sequence[Mapping[str, Any]],
    *,
    forbidden_seeds: Sequence[int] = (),
) -> dict[str, str]:
    """Require one split per complete game and exclude frozen evaluation seeds."""

    game_splits: dict[str, str] = {}
    forbidden = {int(seed) for seed in forbidden_seeds}
    seen_rows: set[str] = set()
    for row in rows:
        row_id = str(row.get("row_id") or "")
        game_id = str(row.get("game_id") or "")
        split = str(row.get("split") or "")
        if not row_id or row_id in seen_rows:
            raise HybridContractError("hybrid rows require unique non-empty row_id values")
        if not game_id or split not in {"train", "calibration", "test"}:
            raise HybridContractError("hybrid row has invalid game_id or split")
        seed = int(row.get("seed"))
        if seed in forbidden:
            raise HybridContractError(f"hybrid row uses forbidden evaluation seed {seed}")
        previous = game_splits.setdefault(game_id, split)
        if previous != split:
            raise HybridContractError(f"game {game_id} crosses split boundaries")
        seen_rows.add(row_id)

    return game_splits
