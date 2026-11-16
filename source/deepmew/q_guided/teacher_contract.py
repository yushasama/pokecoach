"""Aggregate-only contract for privileged simulator-teacher supervision.

Private simulator state is deliberately absent from every type in this module.
The branch executor supplies only actor-visible roots and scalar outcomes after
its ephemeral hidden state has been consumed inside the simulator process.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np

from deepmew.replay.information_boundary import assert_deployable_observation


PRIVILEGED_SIMULATOR_TEACHER = "privileged_simulator_teacher"
ACTOR_VISIBLE = "actor_visible"
ORACLE_WORLD_SUPERVISION = "oracle_world_supervision"
DEFAULT_DOWNSIDE_QUANTILES = (0.10, 0.25)
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_MIN_WORLD_SAMPLES = 2
DEFAULT_MIN_PAIRED_SAMPLES = 2

# This is an exclusive persistence allowlist. Internal game/split/signature and
# world-sample identifiers never enter a shard.
SAFE_SHARD_FIELDS = frozenset({
    "actor_visible_root_observation",
    "actor_visible_legal_options",
    "selected_root_action",
    "aggregate_q_mean",
    "q_variance",
    "q_standard_error",
    "downside_quantiles",
    "terminal_return_frequency",
    "rollout_count",
    "continuation_policy_identity",
    "opponent_strength_bucket",
    "action_ranking_confidence",
    "v_checkpoint_sha256",
    "v_calibrator_sha256",
    "config_sha256",
    "label_information_mode",
})

STUDENT_INPUT_FIELDS = (
    "actor_visible_root_observation",
    "actor_visible_legal_options",
)

_FORBIDDEN_ACTOR_PAYLOAD_KEYS = frozenset({
    "searchbegininput",
    "visualizerpayload",
    "opponenthiddenhand",
    "opponenthand",
    "hiddendeckorder",
    "truedeckorder",
    "deckorder",
    "hiddenprizeidentities",
    "pendinghiddensearchstate",
    "rngstate",
    "completeenginestate",
    "privatestate",
    "privatestatetensor",
})


ActionId = str | int


@dataclass(frozen=True)
class BranchRollout:
    """One scalar branch result after all private simulator state is gone."""

    action_id: ActionId
    seed: int
    value: float
    terminal: bool


@dataclass(frozen=True)
class RootTeacherEvidence:
    """In-memory evidence for one compatible hidden-world sample.

    ``world_sample_group`` is an encounter ordinal assigned independently of
    hidden content. It is used only for paired aggregation and never serialized.
    ``root_id``, ``game_id``, ``split``, and ``actor_visible_signature`` are
    likewise validation-only fields excluded from the safe shard.
    """

    root_id: str
    game_id: str
    split: str
    actor_visible_signature: str
    actor_visible_root_observation: Mapping[str, Any]
    actor_visible_legal_options: Sequence[Any]
    legal_action_ids: Sequence[ActionId]
    behavior_action_id: ActionId
    world_sample_group: int
    continuation_policy_identity: str
    opponent_strength_bucket: str
    v_checkpoint_sha256: str
    v_calibrator_sha256: str
    config_sha256: str
    rollouts: Sequence[BranchRollout]
    teacher_information_mode: str = PRIVILEGED_SIMULATOR_TEACHER
    student_information_mode: str = ACTOR_VISIBLE


@dataclass(frozen=True)
class TeacherAggregation:
    """In-memory diagnostics plus the sole persistence-approved label, if safe."""

    status: str
    quarantine_reason: str | None
    action_statistics: Mapping[str, Mapping[str, Any]]
    material_advantage: Mapping[str, float]
    paired_confidence_intervals: Mapping[str, Mapping[str, Any]]
    action_ranking_confidence: float
    hidden_world_sensitivity: Mapping[str, Any]
    safe_shard: Mapping[str, Any]
    training_label: Mapping[str, Any] | None
    policy_control_eligible: bool


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("payload must be finite and JSON-serializable") from exc

    return text.encode("utf-8")


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical_json_bytes(value).decode("utf-8"))


def _normalized_key(key: str) -> str:
    return "".join(character.lower() for character in key if character.isalnum())


def _reject_forbidden_actor_keys(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string JSON key")
            if _normalized_key(key) in _FORBIDDEN_ACTOR_PAYLOAD_KEYS:
                raise ValueError(f"{path} contains forbidden private field {key}")
            _reject_forbidden_actor_keys(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden_actor_keys(child, f"{path}[{index}]")


def _validate_sha256(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ValueError(f"{name} must be a 64-character SHA-256")


def _action_key(action_id: ActionId) -> str:
    if isinstance(action_id, bool) or not isinstance(action_id, (str, int)):
        raise TypeError("action_id must be a string or integer actor-visible identifier")
    if isinstance(action_id, str) and not action_id:
        raise ValueError("action_id cannot be empty")

    return str(action_id)


def _field(row: RootTeacherEvidence | Mapping[str, Any], name: str) -> Any:
    if isinstance(row, RootTeacherEvidence):
        return getattr(row, name)
    if isinstance(row, Mapping):
        if name not in row:
            raise ValueError(f"split audit row is missing {name}")

        return row[name]
    raise TypeError("split audit rows must be RootTeacherEvidence or mappings")


def assert_game_disjoint_splits(
    rows: Sequence[RootTeacherEvidence | Mapping[str, Any]],
) -> None:
    """Reject any complete game assigned to more than one data split."""
    game_split: dict[str, str] = {}
    for row in rows:
        game_id = _field(row, "game_id")
        split = _field(row, "split")
        if not isinstance(game_id, str) or not game_id:
            raise ValueError("game_id must be a non-empty validation-only string")
        if not isinstance(split, str) or not split:
            raise ValueError("split must be a non-empty string")
        previous = game_split.setdefault(game_id, split)
        if previous != split:
            raise ValueError(f"game {game_id} overlaps splits {previous} and {split}")


def assert_no_actor_visible_signature_overlap(
    rows: Sequence[RootTeacherEvidence | Mapping[str, Any]],
) -> None:
    """Reject actor-visible state-action signatures shared across splits."""
    signature_split: dict[str, str] = {}
    for row in rows:
        signature = _field(row, "actor_visible_signature")
        split = _field(row, "split")
        if not isinstance(signature, str) or not signature:
            raise ValueError("actor_visible_signature must be a non-empty validation-only string")
        if not isinstance(split, str) or not split:
            raise ValueError("split must be a non-empty string")
        previous = signature_split.setdefault(signature, split)
        if previous != split:
            raise ValueError(
                f"actor-visible signature {signature} overlaps splits {previous} and {split}"
            )


def _validate_common_evidence(evidence: Sequence[RootTeacherEvidence]) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if not evidence:
        raise ValueError("teacher aggregation requires at least one world sample")
    if any(not isinstance(row, RootTeacherEvidence) for row in evidence):
        raise TypeError("teacher aggregation accepts only RootTeacherEvidence rows")

    assert_game_disjoint_splits(evidence)
    assert_no_actor_visible_signature_overlap(evidence)
    reference = evidence[0]
    if reference.teacher_information_mode != PRIVILEGED_SIMULATOR_TEACHER:
        raise ValueError("teacher evidence requires privileged_simulator_teacher mode")
    if reference.student_information_mode != ACTOR_VISIBLE:
        raise ValueError("student information mode must remain actor_visible")
    assert_deployable_observation(reference.actor_visible_root_observation)
    _reject_forbidden_actor_keys(reference.actor_visible_root_observation)
    _reject_forbidden_actor_keys(reference.actor_visible_legal_options)
    _canonical_json_bytes(reference.actor_visible_root_observation)
    _canonical_json_bytes(reference.actor_visible_legal_options)

    legal_keys = tuple(_action_key(action_id) for action_id in reference.legal_action_ids)
    if not legal_keys or len(legal_keys) != len(set(legal_keys)):
        raise ValueError("legal_action_ids must be non-empty and unique")
    if len(reference.actor_visible_legal_options) != len(legal_keys):
        raise ValueError("actor-visible legal options and action IDs must have equal coverage")
    behavior_key = _action_key(reference.behavior_action_id)
    if behavior_key not in legal_keys:
        raise ValueError("behavior action is not actor-visible legal")

    for name in ("v_checkpoint_sha256", "v_calibrator_sha256", "config_sha256"):
        _validate_sha256(getattr(reference, name), name)
    for name in ("root_id", "game_id", "split", "actor_visible_signature"):
        value = getattr(reference, name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty validation-only string")
    for name in ("continuation_policy_identity", "opponent_strength_bucket"):
        value = getattr(reference, name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")

    reference_values = {
        "split": reference.split,
        "actor_visible_signature": reference.actor_visible_signature,
        "observation": _canonical_json_bytes(reference.actor_visible_root_observation),
        "legal_options": _canonical_json_bytes(reference.actor_visible_legal_options),
        "legal_keys": legal_keys,
        "behavior_key": behavior_key,
        "continuation_policy_identity": reference.continuation_policy_identity,
        "opponent_strength_bucket": reference.opponent_strength_bucket,
        "v_checkpoint_sha256": reference.v_checkpoint_sha256.lower(),
        "v_calibrator_sha256": reference.v_calibrator_sha256.lower(),
        "config_sha256": reference.config_sha256.lower(),
    }
    world_groups = []
    common_seeds: tuple[int, ...] | None = None
    for row in evidence:
        if row.teacher_information_mode != PRIVILEGED_SIMULATOR_TEACHER:
            raise ValueError("teacher evidence requires privileged_simulator_teacher mode")
        if row.student_information_mode != ACTOR_VISIBLE:
            raise ValueError("student information mode must remain actor_visible")
        if isinstance(row.world_sample_group, bool) or not isinstance(row.world_sample_group, int):
            raise TypeError("world_sample_group must be a non-negative encounter ordinal")
        if row.world_sample_group < 0:
            raise ValueError("world_sample_group must be a non-negative encounter ordinal")
        world_groups.append(row.world_sample_group)
        for name in ("v_checkpoint_sha256", "v_calibrator_sha256", "config_sha256"):
            _validate_sha256(getattr(row, name), name)
        row_values = {
            "split": row.split,
            "actor_visible_signature": row.actor_visible_signature,
            "observation": _canonical_json_bytes(row.actor_visible_root_observation),
            "legal_options": _canonical_json_bytes(row.actor_visible_legal_options),
            "legal_keys": tuple(_action_key(action_id) for action_id in row.legal_action_ids),
            "behavior_key": _action_key(row.behavior_action_id),
            "continuation_policy_identity": row.continuation_policy_identity,
            "opponent_strength_bucket": row.opponent_strength_bucket,
            "v_checkpoint_sha256": row.v_checkpoint_sha256.lower(),
            "v_calibrator_sha256": row.v_calibrator_sha256.lower(),
            "config_sha256": row.config_sha256.lower(),
        }
        if row_values != reference_values:
            raise ValueError("world samples do not share one byte-identical actor-visible root contract")
        assert_deployable_observation(row.actor_visible_root_observation)
        _reject_forbidden_actor_keys(row.actor_visible_root_observation)
        _reject_forbidden_actor_keys(row.actor_visible_legal_options)
        seeds = _validate_full_sibling_coverage(row, legal_keys)
        if common_seeds is None:
            common_seeds = seeds
        elif seeds != common_seeds:
            raise ValueError("world samples must use identical paired seed sets")
    if len(world_groups) != len(set(world_groups)):
        raise ValueError("world_sample_group encounter ordinals must be unique")

    return legal_keys, common_seeds or ()


def _validate_full_sibling_coverage(
    evidence: RootTeacherEvidence,
    legal_keys: tuple[str, ...],
) -> tuple[int, ...]:
    if not evidence.rollouts:
        raise ValueError("world sample contains no branch rollouts")
    values_by_action: dict[str, dict[int, BranchRollout]] = defaultdict(dict)
    for rollout in evidence.rollouts:
        if not isinstance(rollout, BranchRollout):
            raise TypeError("rollouts must contain BranchRollout rows")
        action_key = _action_key(rollout.action_id)
        if action_key not in legal_keys:
            raise ValueError("rollout action is not in the actor-visible legal set")
        if isinstance(rollout.seed, bool) or not isinstance(rollout.seed, int) or rollout.seed < 0:
            raise ValueError("rollout seed must be a non-negative integer")
        if rollout.seed in values_by_action[action_key]:
            raise ValueError("duplicate action/seed rollout")
        if isinstance(rollout.value, bool) or not isinstance(rollout.value, (int, float)):
            raise TypeError("rollout value must be numeric")
        value = float(rollout.value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("rollout value must be finite and in [0, 1]")
        if not isinstance(rollout.terminal, bool):
            raise TypeError("rollout terminal marker must be bool")
        if rollout.terminal and value not in {0.0, 0.5, 1.0}:
            raise ValueError("terminal rollout value must be exactly 0.0, 0.5, or 1.0")
        values_by_action[action_key][rollout.seed] = rollout
    if set(values_by_action) != set(legal_keys):
        raise ValueError("branch evidence does not cover every actor-visible sibling action")
    seed_sets = {tuple(sorted(rows)) for rows in values_by_action.values()}
    if len(seed_sets) != 1:
        raise ValueError("sibling actions must use identical paired seeds")
    seeds = next(iter(seed_sets))
    if not seeds:
        raise ValueError("paired sibling seed set cannot be empty")

    return seeds


def _quantile_name(quantile: float) -> str:
    percentage = f"{100.0 * quantile:g}".replace(".", "_")

    return f"p{percentage}"


def _validate_aggregation_settings(
    confidence_level: float,
    downside_quantiles: Sequence[float],
    min_world_samples: int,
    min_paired_samples: int,
) -> tuple[float, tuple[float, ...]]:
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
        or not 0.0 < float(confidence_level) < 1.0
    ):
        raise ValueError("confidence_level must be in (0, 1)")
    quantiles = tuple(float(value) for value in downside_quantiles)
    if (
        not quantiles
        or len(quantiles) != len(set(quantiles))
        or any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in quantiles)
    ):
        raise ValueError("downside_quantiles must be unique finite values in (0, 1)")
    if isinstance(min_world_samples, bool) or not isinstance(min_world_samples, int) or min_world_samples < 2:
        raise ValueError("min_world_samples must be at least 2")
    if isinstance(min_paired_samples, bool) or not isinstance(min_paired_samples, int) or min_paired_samples < 2:
        raise ValueError("min_paired_samples must be at least 2")

    return float(confidence_level), quantiles


def _rollout_matrix(
    evidence: Sequence[RootTeacherEvidence],
    legal_keys: tuple[str, ...],
    seeds: tuple[int, ...],
) -> dict[str, list[float]]:
    matrix = {action_key: [] for action_key in legal_keys}
    for row in sorted(evidence, key=lambda item: item.world_sample_group):
        indexed = {
            (_action_key(rollout.action_id), rollout.seed): float(rollout.value)
            for rollout in row.rollouts
        }
        for seed in seeds:
            for action_key in legal_keys:
                matrix[action_key].append(indexed[(action_key, seed)])

    return matrix


def _action_statistics(
    matrix: Mapping[str, Sequence[float]],
    quantiles: tuple[float, ...],
) -> dict[str, dict[str, Any]]:
    statistics = {}
    for action_key, raw_values in matrix.items():
        values = np.asarray(raw_values, dtype=np.float64)
        variance = float(np.var(values, ddof=1)) if len(values) > 1 else 0.0
        statistics[action_key] = {
            "mean": float(np.mean(values)),
            "variance": variance,
            "standard_error": float(math.sqrt(variance / len(values))),
            "downside_quantiles": {
                _quantile_name(quantile): float(np.quantile(values, quantile, method="linear"))
                for quantile in quantiles
            },
            "rollout_count": int(len(values)),
        }

    return statistics


def _terminal_frequencies(
    evidence: Sequence[RootTeacherEvidence],
    legal_keys: tuple[str, ...],
) -> dict[str, float]:
    terminal_by_action: dict[str, list[bool]] = {action_key: [] for action_key in legal_keys}
    for row in evidence:
        for rollout in row.rollouts:
            terminal_by_action[_action_key(rollout.action_id)].append(rollout.terminal)

    return {
        action_key: float(sum(values) / len(values))
        for action_key, values in terminal_by_action.items()
    }


def _paired_advantage_intervals(
    matrix: Mapping[str, Sequence[float]],
    behavior_key: str,
    confidence_level: float,
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    behavior = np.asarray(matrix[behavior_key], dtype=np.float64)
    critical = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    advantages = {}
    intervals = {}
    for action_key, raw_values in matrix.items():
        differences = np.asarray(raw_values, dtype=np.float64) - behavior
        mean = float(np.mean(differences))
        if len(differences) > 1:
            standard_error = float(np.std(differences, ddof=1) / math.sqrt(len(differences)))
            low = mean - critical * standard_error
            high = mean + critical * standard_error
        else:
            standard_error = None
            low = None
            high = None
        advantages[action_key] = mean
        intervals[action_key] = {
            "mean": mean,
            "standard_error": standard_error,
            "low": low,
            "high": high,
            "paired_samples": int(len(differences)),
            "confidence_level": confidence_level,
        }

    return advantages, intervals


def _ranking_confidence(
    matrix: Mapping[str, Sequence[float]],
    action_statistics: Mapping[str, Mapping[str, Any]],
) -> float:
    ranked = sorted(
        action_statistics,
        key=lambda action_key: (-float(action_statistics[action_key]["mean"]), action_key),
    )
    if len(ranked) == 1:
        return 1.0
    best = np.asarray(matrix[ranked[0]], dtype=np.float64)
    second = np.asarray(matrix[ranked[1]], dtype=np.float64)
    differences = best - second

    return float(np.mean((differences > 0.0).astype(np.float64) + 0.5 * (differences == 0.0)))


def _hidden_world_sensitivity(
    evidence: Sequence[RootTeacherEvidence],
    legal_keys: tuple[str, ...],
    global_best: str,
) -> dict[str, Any]:
    world_means: dict[str, list[float]] = {action_key: [] for action_key in legal_keys}
    global_best_agreement = []
    for row in sorted(evidence, key=lambda item: item.world_sample_group):
        values: dict[str, list[float]] = {action_key: [] for action_key in legal_keys}
        for rollout in row.rollouts:
            values[_action_key(rollout.action_id)].append(float(rollout.value))
        means = {action_key: float(np.mean(action_values)) for action_key, action_values in values.items()}
        for action_key, mean in means.items():
            world_means[action_key].append(mean)
        best_mean = max(means.values())
        global_best_agreement.append(float(means[global_best] == best_mean))
    action_sensitivity = {}
    for action_key, raw_values in world_means.items():
        values = np.asarray(raw_values, dtype=np.float64)
        variance = float(np.var(values, ddof=1)) if len(values) > 1 else 0.0
        action_sensitivity[action_key] = {
            "world_mean_range": float(np.max(values) - np.min(values)),
            "world_mean_variance": variance,
            "world_mean_standard_deviation": float(math.sqrt(variance)),
        }

    return {
        "world_sample_count": len(evidence),
        "single_world_evidence": len(evidence) == 1,
        "global_best_action_world_agreement": float(np.mean(global_best_agreement)),
        "actions": action_sensitivity,
    }


def _safe_shard(
    reference: RootTeacherEvidence,
    action_statistics: Mapping[str, Mapping[str, Any]],
    terminal_frequencies: Mapping[str, float],
    action_ranking_confidence: float,
) -> dict[str, Any]:
    row = {
        "actor_visible_root_observation": _json_copy(reference.actor_visible_root_observation),
        "actor_visible_legal_options": _json_copy(reference.actor_visible_legal_options),
        "selected_root_action": _json_copy(reference.behavior_action_id),
        "aggregate_q_mean": {
            action_key: float(values["mean"]) for action_key, values in action_statistics.items()
        },
        "q_variance": {
            action_key: float(values["variance"]) for action_key, values in action_statistics.items()
        },
        "q_standard_error": {
            action_key: float(values["standard_error"])
            for action_key, values in action_statistics.items()
        },
        "downside_quantiles": {
            action_key: dict(values["downside_quantiles"])
            for action_key, values in action_statistics.items()
        },
        "terminal_return_frequency": dict(terminal_frequencies),
        "rollout_count": {
            action_key: int(values["rollout_count"])
            for action_key, values in action_statistics.items()
        },
        "continuation_policy_identity": reference.continuation_policy_identity,
        "opponent_strength_bucket": reference.opponent_strength_bucket,
        "action_ranking_confidence": float(action_ranking_confidence),
        "v_checkpoint_sha256": reference.v_checkpoint_sha256.lower(),
        "v_calibrator_sha256": reference.v_calibrator_sha256.lower(),
        "config_sha256": reference.config_sha256.lower(),
        "label_information_mode": PRIVILEGED_SIMULATOR_TEACHER,
    }
    validate_safe_shard(row)

    return row


def aggregate_teacher_target(
    evidence: Sequence[RootTeacherEvidence],
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    downside_quantiles: Sequence[float] = DEFAULT_DOWNSIDE_QUANTILES,
    min_world_samples: int = DEFAULT_MIN_WORLD_SAMPLES,
    min_paired_samples: int = DEFAULT_MIN_PAIRED_SAMPLES,
) -> TeacherAggregation:
    """Aggregate paired sibling outcomes without persisting hidden-world identity.

    A single compatible world remains oracle-world evidence in quarantine. It
    receives diagnostics but no training label. Structural violations such as
    incomplete siblings, unpaired seeds, leakage, or split overlap fail closed.
    """
    confidence_level, quantiles = _validate_aggregation_settings(
        confidence_level,
        downside_quantiles,
        min_world_samples,
        min_paired_samples,
    )
    rows = tuple(evidence)
    legal_keys, seeds = _validate_common_evidence(rows)
    matrix = _rollout_matrix(rows, legal_keys, seeds)
    statistics = _action_statistics(matrix, quantiles)
    terminal_frequencies = _terminal_frequencies(rows, legal_keys)
    behavior_key = _action_key(rows[0].behavior_action_id)
    material_advantage, paired_intervals = _paired_advantage_intervals(
        matrix,
        behavior_key,
        confidence_level,
    )
    ranking_confidence = _ranking_confidence(matrix, statistics)
    global_best = min(
        legal_keys,
        key=lambda action_key: (-float(statistics[action_key]["mean"]), action_key),
    )
    sensitivity = _hidden_world_sensitivity(rows, legal_keys, global_best)
    paired_samples = len(next(iter(matrix.values())))
    quarantine_reasons = []
    if len(rows) < min_world_samples:
        quarantine_reasons.append(
            f"single/exact-world evidence: {len(rows)} world samples < {min_world_samples}"
        )
    if paired_samples < min_paired_samples:
        quarantine_reasons.append(
            f"paired evidence: {paired_samples} samples < {min_paired_samples}"
        )
    safe_shard = _safe_shard(
        rows[0],
        statistics,
        terminal_frequencies,
        ranking_confidence,
    )
    if quarantine_reasons:
        training_label = None
        status = "quarantine"
        quarantine_reason = "; ".join(quarantine_reasons)
    else:
        training_label = safe_shard
        status = "eligible_aggregate_teacher_target"
        quarantine_reason = None

    return TeacherAggregation(
        status=status,
        quarantine_reason=quarantine_reason,
        action_statistics=statistics,
        material_advantage=material_advantage,
        paired_confidence_intervals=paired_intervals,
        action_ranking_confidence=ranking_confidence,
        hidden_world_sensitivity=sensitivity,
        safe_shard=safe_shard,
        training_label=training_label,
        # Aggregate target eligibility is not a policy-control decision. That
        # requires the separate equal-budget gameplay and promotion gates.
        policy_control_eligible=False,
    )


def _assert_finite_tree(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite_tree(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_finite_tree(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite float")


def validate_safe_shard(row: Mapping[str, Any]) -> None:
    """Fail closed unless a persisted row uses exactly the approved fields."""
    if not isinstance(row, Mapping):
        raise TypeError("safe shard row must be a mapping")
    fields = set(row)
    if fields != SAFE_SHARD_FIELDS:
        missing = sorted(SAFE_SHARD_FIELDS - fields)
        unexpected = sorted(fields - SAFE_SHARD_FIELDS)
        raise ValueError(f"safe shard field inventory mismatch: missing={missing}, unexpected={unexpected}")
    if row["label_information_mode"] != PRIVILEGED_SIMULATOR_TEACHER:
        raise ValueError("safe shard must identify privileged_simulator_teacher label generation")
    assert_deployable_observation(row["actor_visible_root_observation"])
    _reject_forbidden_actor_keys(row["actor_visible_root_observation"])
    _reject_forbidden_actor_keys(row["actor_visible_legal_options"])
    for name in ("v_checkpoint_sha256", "v_calibrator_sha256", "config_sha256"):
        _validate_sha256(row[name], name)
    confidence = row["action_ranking_confidence"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise ValueError("action_ranking_confidence must be finite and in [0, 1]")
    action_fields = (
        "aggregate_q_mean",
        "q_variance",
        "q_standard_error",
        "downside_quantiles",
        "terminal_return_frequency",
        "rollout_count",
    )
    action_keys = [set(row[name]) for name in action_fields if isinstance(row[name], Mapping)]
    if len(action_keys) != len(action_fields) or not action_keys[0] or any(keys != action_keys[0] for keys in action_keys):
        raise ValueError("safe shard action aggregates must have identical non-empty coverage")
    _assert_finite_tree(row)
    _canonical_json_bytes(row)


def student_input_bytes(row: Mapping[str, Any]) -> bytes:
    """Canonical student input bytes, intentionally ignoring teacher targets."""
    if not isinstance(row, Mapping):
        raise TypeError("student row must be a mapping")
    missing = [field for field in STUDENT_INPUT_FIELDS if field not in row]
    if missing:
        raise ValueError(f"student row is missing actor-visible fields {missing}")
    payload = {field: row[field] for field in STUDENT_INPUT_FIELDS}
    assert_deployable_observation(payload["actor_visible_root_observation"])
    _reject_forbidden_actor_keys(payload)

    return _canonical_json_bytes(payload)


def assert_student_input_byte_identity(
    row_with_teacher_payload: Mapping[str, Any],
    row_without_teacher_payload: Mapping[str, Any],
) -> None:
    """Prove teacher payload presence cannot change the student's input bytes."""
    with_teacher = student_input_bytes(row_with_teacher_payload)
    without_teacher = student_input_bytes(row_without_teacher_payload)
    if with_teacher != without_teacher:
        raise ValueError("student input changed when privileged teacher payload was deleted")
