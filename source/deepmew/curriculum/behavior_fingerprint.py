"""Diagnostic-only behavioral fingerprints from public/action-summary rows.

The extractor is deliberately a projection, not a row copier.  It accepts a
small set of versioned replay/refgame contracts and retains only public state
aggregates plus legal/chosen action *types*.  Card identities, deck/prize/hand
contents, search state, and raw observations are never read into the projected
observation or returned in the fingerprint.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from deepmew.replay.prospective_trace import (
    PUBLIC_PROJECTION_VERSION,
    PUBLIC_ROW_SCHEMA_VERSION,
    assert_public_trace_row_safe,
)


FINGERPRINT_SCHEMA_VERSION = "deepmew.behavior_fingerprint.v2"
DISTANCE_SCHEMA_VERSION = "deepmew.behavior_fingerprint_distance.v1"
PROJECTION_SCHEMA_VERSION = "deepmew.behavior_public_projection.v1"
CLASSIFICATION = "diagnostic_only"

REPLAY_DECISION_SCHEMA_VERSION = "deepmew.replay_decision_row.v1"
REFGAME_SCHEMA_VERSIONS: frozenset[str] = frozenset({
    "deepmew.refgame_decision_row.rebuilt.v1",
    "deepmew.refgame_decision_row.rebuilt.v2",
})
SUPPORTED_ROW_SCHEMA_VERSIONS: frozenset[str] = frozenset({
    PUBLIC_ROW_SCHEMA_VERSION,
    REPLAY_DECISION_SCHEMA_VERSION,
    *REFGAME_SCHEMA_VERSIONS,
})

# Families are semantic types, not card/deck identities.  Every family rate is
# bounded in [0, 1], so the unweighted diagnostic distance has a stable scale.
ACTION_FAMILIES: tuple[str, ...] = (
    "attack",
    "attach",
    "evolve",
    "play",
    "retreat",
    "end",
    "affirmative",
    "negative",
    "numeric",
    "other",
)

PUBLIC_STATE_FEATURES: tuple[str, ...] = (
    "own_active_present_rate",
    "opponent_active_present_rate",
    "own_bench_fill_rate",
    "opponent_bench_fill_rate",
    "energy_already_attached_rate",
    "supporter_already_played_rate",
    "retreated_already_rate",
    "stadium_present_rate",
)

FEATURE_NAMES: tuple[str, ...] = (
    *(f"chosen_{family}_rate" for family in ACTION_FAMILIES),
    *(f"legal_{family}_available_rate" for family in ACTION_FAMILIES),
    *(f"{family}_conversion_rate" for family in ACTION_FAMILIES),
    "chosen_family_entropy",
    "legal_family_concentration",
    "decision_turn_progress",
    "attack_turn_progress",
    "control_pass_rate",
    "turn_end_rate",
    *PUBLIC_STATE_FEATURES,
)

_ACTION_FAMILY_BY_TYPE: dict[str, str] = {
    "ATTACK": "attack",
    "ATTACH": "attach",
    "EVOLVE": "evolve",
    "PLAY": "play",
    "RETREAT": "retreat",
    "END": "end",
    "END_TURN": "end",
    "YES": "affirmative",
    "NO": "negative",
    "NUMBER": "numeric",
}


@dataclass(frozen=True)
class BehaviorObservation:
    """Identity-free, card-free projection of one decision row."""

    row_id: str
    game_id: str
    source_schema_version: str
    policy_id: str | None
    policy_identity_kind: str | None
    feature_policy_id: str | None
    turn_progress: float | None
    chosen_families: frozenset[str] | None
    legal_families: frozenset[str] | None
    control_passed: bool | None
    ended_turn: bool | None
    public_signals: tuple[tuple[str, float | None], ...] | None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _as_non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None

    return parsed if parsed >= 0 else None


def _normalized_string(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None

    return value.strip()


def _policy_identity(row: dict[str, Any], schema_version: str) -> tuple[str | None, str | None]:
    if schema_version in REFGAME_SCHEMA_VERSIONS:
        candidates = (("policy_id", row.get("policy_id")),)
    elif schema_version == REPLAY_DECISION_SCHEMA_VERSION:
        candidates = (("agent_label", row.get("agent_label")),)
    else:
        policy = _as_dict(row.get("actor_policy"))
        candidates = (
            ("league_member_id", policy.get("league_member_id")),
            ("artifact_sha256", policy.get("artifact_sha256")),
            ("checkpoint_sha256", policy.get("checkpoint_sha256")),
            ("policy_id", policy.get("policy_id")),
            ("policy_spec", policy.get("policy_spec")),
        )
    for kind, value in candidates:
        normalized = _normalized_string(value)
        if normalized is not None:
            return normalized, kind

    return None, None


def _feature_policy_id(row: dict[str, Any], schema_version: str) -> str | None:
    if schema_version == PUBLIC_ROW_SCHEMA_VERSION:
        value = _as_dict(row.get("actor_policy")).get("feature_policy_id")
    else:
        value = row.get("feature_policy_id")

    return _normalized_string(value)


def _row_id(row: dict[str, Any], schema_version: str, game_id: str) -> str:
    for field in ("decision_id", "frame_id"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    decision_index = _as_non_negative_int(row.get("decision_index"))
    if decision_index is None:
        raise ValueError(f"{schema_version}: row is missing a stable decision identity")

    return f"{game_id}:d{decision_index:05d}"


def _action_family(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().upper()

    return _ACTION_FAMILY_BY_TYPE.get(normalized, "other")


def _families_from_types(values: Any) -> frozenset[str] | None:
    if not isinstance(values, list):
        return None
    families = {
        family
        for family in (_action_family(value) for value in values)
        if family is not None
    }

    return frozenset(families)


def _families_from_semantics(container: Any, field: str) -> frozenset[str] | None:
    mapping = _as_dict(container)
    if field not in mapping or not isinstance(mapping.get(field), list):
        return None
    families = set()
    for value in mapping[field]:
        semantic = _as_dict(value)
        family = _action_family(semantic.get("option_type"))
        if family is not None:
            families.add(family)

    return frozenset(families)


def _public_card_zone_present(value: Any) -> float | None:
    if isinstance(value, (list, dict)):
        return float(bool(value))

    return None


def _bench_fill(player: dict[str, Any]) -> float | None:
    bench = player.get("bench")
    bench_max = _as_non_negative_int(player.get("benchMax"))
    if not isinstance(bench, list) or bench_max is None or bench_max <= 0:
        return None

    return min(len(bench) / bench_max, 1.0)


def _public_state_signals(observation: Any, actor_seat: Any) -> tuple[tuple[str, float | None], ...] | None:
    if not isinstance(observation, dict):
        return None
    current = _as_dict(observation.get("current"))
    players = _as_list(current.get("players"))
    seat = _as_non_negative_int(actor_seat)
    if seat not in (0, 1) or len(players) != 2:
        return tuple((name, None) for name in PUBLIC_STATE_FEATURES)
    own = _as_dict(players[seat])
    opponent = _as_dict(players[1 - seat])
    stadium = current.get("stadium")
    stadium_present = float(bool(stadium)) if isinstance(stadium, (list, dict)) else None
    values: dict[str, float | None] = {
        "own_active_present_rate": _public_card_zone_present(own.get("active")),
        "opponent_active_present_rate": _public_card_zone_present(opponent.get("active")),
        "own_bench_fill_rate": _bench_fill(own),
        "opponent_bench_fill_rate": _bench_fill(opponent),
        "energy_already_attached_rate": _bool_float(current.get("energyAttached")),
        "supporter_already_played_rate": _bool_float(current.get("supporterPlayed")),
        "retreated_already_rate": _bool_float(current.get("retreated")),
        "stadium_present_rate": stadium_present,
    }

    return tuple((name, values[name]) for name in PUBLIC_STATE_FEATURES)


def _bool_float(value: Any) -> float | None:
    parsed = _as_bool(value)

    return None if parsed is None else float(parsed)


def project_behavior_row(row: dict[str, Any]) -> BehaviorObservation:
    """Project one supported row without retaining identity-bearing game state."""
    if not isinstance(row, dict):
        raise ValueError("behavior fingerprint input row must be a JSON object")
    schema_version = str(row.get("schema_version") or "")
    if schema_version not in SUPPORTED_ROW_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported behavior fingerprint row schema: {schema_version!r}")
    game_id = row.get("game_id")
    if not isinstance(game_id, str) or not game_id.strip():
        raise ValueError(f"{schema_version}: row is missing game_id")
    game_id = game_id.strip()
    turn = _as_non_negative_int(row.get("turn"))
    turn_progress = None if turn is None else turn / (turn + 1.0)

    if schema_version == PUBLIC_ROW_SCHEMA_VERSION:
        if row.get("information_mode") != "public_belief":
            raise ValueError("prospective public fingerprint projection requires public_belief information_mode")
        assert_public_trace_row_safe(row)
        chosen_family = _action_family(row.get("action_type"))
        chosen = None if row.get("action_type") in (None, "") else frozenset({chosen_family or "other"})
        legal = None
        control_passed = _as_bool(row.get("control_passed"))
        ended_turn = _as_bool(row.get("ended_turn"))
        public_signals = _public_state_signals(row.get("public_observation_before"), row.get("actor_seat"))
    elif schema_version in REFGAME_SCHEMA_VERSIONS:
        if schema_version.endswith(".v2") and row.get("information_mode") != "public_belief":
            raise ValueError("refgame v2 fingerprint projection requires public_belief information_mode")
        chosen = _families_from_semantics(row.get("chosen_action"), "semantics")
        legal_container = {"semantics": row.get("legal_set")} if "legal_set" in row else {}
        legal = _families_from_semantics(legal_container, "semantics")
        control_passed = None
        ended_turn = None
        public_signals = None
    else:
        chosen = _families_from_types(row.get("chosen_option_types"))
        legal = _families_from_types(row.get("legal_option_types"))
        control_passed = None
        ended_turn = None
        public_signals = None

    policy_id, policy_identity_kind = _policy_identity(row, schema_version)

    return BehaviorObservation(
        row_id=_row_id(row, schema_version, game_id),
        game_id=game_id,
        source_schema_version=schema_version,
        policy_id=policy_id,
        policy_identity_kind=policy_identity_kind,
        feature_policy_id=_feature_policy_id(row, schema_version),
        turn_progress=turn_progress,
        chosen_families=chosen,
        legal_families=legal,
        control_passed=control_passed,
        ended_turn=ended_turn,
        public_signals=public_signals,
    )


def _feature_description(name: str) -> str:
    if name.startswith("chosen_") and name.endswith("_rate"):
        return "fraction of action-summary rows containing this chosen semantic family"
    if name.startswith("legal_") and name.endswith("_available_rate"):
        return "fraction of legal-summary rows where this semantic family was available"
    if name.endswith("_conversion_rate"):
        return "fraction of eligible rows choosing this family when that family was legal"
    descriptions = {
        "chosen_family_entropy": "normalized entropy of observed chosen semantic families",
        "legal_family_concentration": "mean reciprocal count of distinct legal semantic families",
        "decision_turn_progress": "mean bounded turn transform turn/(turn+1) over decisions",
        "attack_turn_progress": "mean bounded turn transform over chosen-attack decisions",
        "control_pass_rate": "fraction of public transitions that pass control to the other seat",
        "turn_end_rate": "fraction of public transitions that advance the engine turn",
        "own_active_present_rate": "fraction of public observations with an occupied actor active zone",
        "opponent_active_present_rate": "fraction of public observations with an occupied opponent active zone",
        "own_bench_fill_rate": "mean public actor bench occupancy divided by observed bench capacity",
        "opponent_bench_fill_rate": "mean public opponent bench occupancy divided by observed bench capacity",
        "energy_already_attached_rate": "fraction of public observations after the turn energy attachment",
        "supporter_already_played_rate": "fraction of public observations after the turn supporter play",
        "retreated_already_rate": "fraction of public observations after the turn retreat",
        "stadium_present_rate": "fraction of public observations with a stadium in play",
    }

    return descriptions[name]


def _unavailable_feature(name: str, availability: str, reason: str) -> dict[str, Any]:
    return {
        "availability": availability,
        "bounded": [0.0, 1.0],
        "description": _feature_description(name),
        "n_observations": 0,
        "reason": reason,
        "uncertainty": {
            "availability": availability,
            "method": "cluster_bootstrap_by_game",
            "reason": "point estimate unavailable",
        },
        "unit": "fraction",
        "value": None,
    }


def _mean_feature(name: str, values: list[float], *, instrumented: bool, empty_reason: str) -> dict[str, Any]:
    if not instrumented:
        return _unavailable_feature(name, "not_instrumented", "required source field is absent")
    if not values:
        return _unavailable_feature(name, "missing", empty_reason)
    value = statistics.fmean(values)

    return {
        "availability": "available",
        "bounded": [0.0, 1.0],
        "description": _feature_description(name),
        "n_observations": len(values),
        "reason": None,
        "uncertainty": {},
        "unit": "fraction",
        "value": float(value),
    }


def _entropy_feature(observations: list[BehaviorObservation]) -> dict[str, Any]:
    name = "chosen_family_entropy"
    instrumented = any(row.chosen_families is not None for row in observations)
    counts: Counter[str] = Counter()
    for row in observations:
        if row.chosen_families is not None:
            counts.update(row.chosen_families)
    total = sum(counts.values())
    if not instrumented:
        return _unavailable_feature(name, "not_instrumented", "chosen action summaries are absent")
    if total == 0:
        return _unavailable_feature(name, "missing", "no chosen semantic family was observed")
    probabilities = [count / total for count in counts.values()]
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    normalized = entropy / math.log(len(ACTION_FAMILIES))
    feature = _mean_feature(name, [normalized], instrumented=True, empty_reason="no chosen families")
    feature["n_observations"] = total

    return feature


def _compute_features(observations: list[BehaviorObservation]) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    chosen_instrumented = any(row.chosen_families is not None for row in observations)
    legal_instrumented = any(row.legal_families is not None for row in observations)
    for family in ACTION_FAMILIES:
        chosen_name = f"chosen_{family}_rate"
        chosen_values = [
            float(family in row.chosen_families)
            for row in observations
            if row.chosen_families is not None
        ]
        features[chosen_name] = _mean_feature(
            chosen_name,
            chosen_values,
            instrumented=chosen_instrumented,
            empty_reason="no action-summary rows",
        )

        legal_name = f"legal_{family}_available_rate"
        legal_values = [
            float(family in row.legal_families)
            for row in observations
            if row.legal_families is not None
        ]
        features[legal_name] = _mean_feature(
            legal_name,
            legal_values,
            instrumented=legal_instrumented,
            empty_reason="no legal-summary rows",
        )

        conversion_name = f"{family}_conversion_rate"
        conversion_rows = [
            row
            for row in observations
            if row.chosen_families is not None and row.legal_families is not None
        ]
        eligible = [row for row in conversion_rows if family in row.legal_families]
        conversion_values = [float(family in row.chosen_families) for row in eligible]
        features[conversion_name] = _mean_feature(
            conversion_name,
            conversion_values,
            instrumented=bool(conversion_rows),
            empty_reason="semantic family was never legal in jointly instrumented rows",
        )

    features["chosen_family_entropy"] = _entropy_feature(observations)
    concentration_values = [
        1.0 / len(row.legal_families)
        for row in observations
        if row.legal_families is not None and row.legal_families
    ]
    features["legal_family_concentration"] = _mean_feature(
        "legal_family_concentration",
        concentration_values,
        instrumented=legal_instrumented,
        empty_reason="no non-empty legal semantic family sets",
    )
    turn_values = [row.turn_progress for row in observations if row.turn_progress is not None]
    features["decision_turn_progress"] = _mean_feature(
        "decision_turn_progress",
        [float(value) for value in turn_values],
        instrumented=bool(turn_values),
        empty_reason="turn is missing",
    )
    attack_turn_values = [
        float(row.turn_progress)
        for row in observations
        if row.turn_progress is not None
        and row.chosen_families is not None
        and "attack" in row.chosen_families
    ]
    features["attack_turn_progress"] = _mean_feature(
        "attack_turn_progress",
        attack_turn_values,
        instrumented=chosen_instrumented and bool(turn_values),
        empty_reason="no chosen attack with an instrumented turn",
    )
    control_values = [float(row.control_passed) for row in observations if row.control_passed is not None]
    features["control_pass_rate"] = _mean_feature(
        "control_pass_rate",
        control_values,
        instrumented=bool(control_values),
        empty_reason="control transition is missing",
    )
    end_values = [float(row.ended_turn) for row in observations if row.ended_turn is not None]
    features["turn_end_rate"] = _mean_feature(
        "turn_end_rate",
        end_values,
        instrumented=bool(end_values),
        empty_reason="turn transition is missing",
    )
    for name in PUBLIC_STATE_FEATURES:
        rows_with_public_state = [row for row in observations if row.public_signals is not None]
        values = [
            float(signal_map[name])
            for row in rows_with_public_state
            for signal_map in [dict(row.public_signals or ())]
            if signal_map.get(name) is not None
        ]
        features[name] = _mean_feature(
            name,
            values,
            instrumented=bool(rows_with_public_state),
            empty_reason="public observation lacks the required aggregate",
        )

    return {name: features[name] for name in FEATURE_NAMES}


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower

    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _available_vector(features: dict[str, dict[str, Any]]) -> dict[str, float]:
    vector: dict[str, float] = {}
    for name in FEATURE_NAMES:
        feature = _as_dict(features.get(name))
        if feature.get("availability") != "available":
            continue
        value = feature.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"available behavior feature {name!r} must have a numeric value")
        parsed = float(value)
        if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
            raise ValueError(f"available behavior feature {name!r} must be finite and unit bounded")
        vector[name] = parsed

    return vector


def _vector_distance(left: dict[str, float], right: dict[str, float]) -> tuple[float | None, int]:
    common = sorted(set(left).intersection(right))
    if not common:
        return None, 0
    squared = [(left[name] - right[name]) ** 2 for name in common]

    return math.sqrt(statistics.fmean(squared)), len(common)


def _attach_bootstrap(
    features: dict[str, dict[str, Any]],
    observations: list[BehaviorObservation],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    by_game: dict[str, list[BehaviorObservation]] = defaultdict(list)
    for row in observations:
        by_game[row.game_id].append(row)
    game_ids = sorted(by_game)
    disabled = bootstrap_samples == 0
    insufficient_games = len(game_ids) < 2
    insufficient_samples = bootstrap_samples < 2
    if disabled or insufficient_games or insufficient_samples:
        if disabled:
            availability = "not_instrumented"
            reason = "bootstrap disabled"
        else:
            availability = "missing"
            reason = "bootstrap requires at least two games and two resamples"
        for feature in features.values():
            if feature.get("availability") == "available":
                feature["uncertainty"] = {
                    "availability": availability,
                    "method": "cluster_bootstrap_by_game",
                    "reason": reason,
                }

        return {
            "availability": availability,
            "bootstrap_samples": bootstrap_samples,
            "games": len(game_ids),
            "method": "cluster_bootstrap_by_game",
            "reason": reason,
        }

    rng = random.Random(bootstrap_seed)
    replicate_features: list[dict[str, dict[str, Any]]] = []
    for _ in range(bootstrap_samples):
        sampled: list[BehaviorObservation] = []
        for _game in game_ids:
            sampled.extend(by_game[game_ids[rng.randrange(len(game_ids))]])
        replicate_features.append(_compute_features(sampled))

    for name, feature in features.items():
        if feature.get("availability") != "available":
            continue
        values = [
            float(replicate[name]["value"])
            for replicate in replicate_features
            if replicate[name].get("availability") == "available"
        ]
        if len(values) < 2:
            feature["uncertainty"] = {
                "availability": "missing",
                "method": "cluster_bootstrap_by_game",
                "reason": "fewer than two finite bootstrap estimates",
            }
            continue
        low = _quantile(values, 0.025)
        high = _quantile(values, 0.975)
        feature["uncertainty"] = {
            "availability": "available",
            "bootstrap_samples": len(values),
            "ci95": [low, high],
            "method": "cluster_bootstrap_by_game",
            "standard_error": statistics.stdev(values),
            "stability": {
                "definition": "one_minus_ci95_width_on_unit_interval",
                "value": max(0.0, 1.0 - (high - low)),
            },
        }

    full_vector = _available_vector(features)
    similarities: list[float] = []
    common_counts: list[int] = []
    for replicate in replicate_features:
        distance, common_count = _vector_distance(full_vector, _available_vector(replicate))
        if distance is not None:
            similarities.append(max(0.0, 1.0 - distance))
            common_counts.append(common_count)
    if not similarities:
        return {
            "availability": "missing",
            "bootstrap_samples": bootstrap_samples,
            "games": len(game_ids),
            "method": "cluster_bootstrap_by_game",
            "reason": "no common finite fingerprint features across resamples",
        }

    return {
        "availability": "available",
        "bootstrap_samples": len(similarities),
        "games": len(game_ids),
        "mean_similarity_to_full": statistics.fmean(similarities),
        "method": "cluster_bootstrap_by_game",
        "minimum_common_features": min(common_counts),
        "p05_similarity_to_full": _quantile(similarities, 0.05),
        "similarity_definition": "one_minus_rms_distance_on_common_unit_interval_features",
    }


def build_behavior_fingerprint(
    rows: Iterable[dict[str, Any]],
    *,
    subject_id: str,
    source_policy_id: str | None = None,
    bootstrap_samples: int = 200,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Build a deterministic diagnostic fingerprint for one explicit policy."""
    if not isinstance(subject_id, str) or not subject_id.strip():
        raise ValueError("subject_id must be a non-empty string")
    if source_policy_id is not None and (not isinstance(source_policy_id, str) or not source_policy_id.strip()):
        raise ValueError("source_policy_id must be null or a non-empty string")
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be a non-negative integer")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise ValueError("bootstrap_seed must be an integer")
    projected = [project_behavior_row(row) for row in rows]
    if not projected:
        raise ValueError("cannot build a behavior fingerprint from zero rows")

    if source_policy_id is not None:
        source_policy_id = source_policy_id.strip()
        selected = [row for row in projected if row.policy_id == source_policy_id]
        excluded = len(projected) - len(selected)
        if not selected:
            raise ValueError(f"no rows matched source_policy_id {source_policy_id!r}")
        policy_identity_availability = "available"
    else:
        identities = {
            (row.policy_identity_kind, row.policy_id)
            for row in projected
            if row.policy_id is not None
        }
        unknown_identities = sum(row.policy_id is None for row in projected)
        if len(identities) > 1 or (identities and unknown_identities):
            raise ValueError("input contains mixed or partially unknown policy identities; set source_policy_id")
        selected = projected
        excluded = 0
        if identities:
            source_policy_identity_kind, source_policy_id = next(iter(identities))
            policy_identity_availability = "available"
        else:
            source_policy_identity_kind = None
            policy_identity_availability = "not_instrumented"

    if source_policy_id is not None:
        selected_identity_kinds = {
            row.policy_identity_kind
            for row in selected
            if row.policy_id == source_policy_id and row.policy_identity_kind is not None
        }
        if len(selected_identity_kinds) > 1:
            raise ValueError(
                f"source_policy_id {source_policy_id!r} is ambiguous across identity kinds"
            )
        source_policy_identity_kind = (
            next(iter(selected_identity_kinds)) if selected_identity_kinds else None
        )

    feature_ids = {row.feature_policy_id for row in selected if row.feature_policy_id is not None}
    unknown_feature_ids = sum(row.feature_policy_id is None for row in selected)
    if len(feature_ids) == 1 and unknown_feature_ids == 0:
        source_feature_policy_id = next(iter(feature_ids))
        feature_identity_availability = "available"
    elif not feature_ids:
        source_feature_policy_id = None
        feature_identity_availability = "not_instrumented"
    else:
        source_feature_policy_id = None
        feature_identity_availability = "mixed_or_partial"

    row_ids = [row.row_id for row in selected]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("duplicate stable decision identity in behavior fingerprint input")
    schema_counts = Counter(row.source_schema_version for row in selected)
    features = _compute_features(selected)
    bootstrap = _attach_bootstrap(
        features,
        selected,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )

    return {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "projection_schema_version": PROJECTION_SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "subject_id": subject_id.strip(),
        "source_policy_identity": {
            "availability": policy_identity_availability,
            "identity_kind": source_policy_identity_kind,
            "policy_id": source_policy_id,
        },
        "source_feature_policy_identity": {
            "availability": feature_identity_availability,
            "feature_policy_id": source_feature_policy_id,
            "observed_feature_policy_ids": sorted(feature_ids),
        },
        "coverage": {
            "accepted_rows": len(selected),
            "excluded_policy_rows": excluded,
            "games": len({row.game_id for row in selected}),
            "source_schema_counts": dict(sorted(schema_counts.items())),
        },
        "feature_order": list(FEATURE_NAMES),
        "features": features,
        "bootstrap_stability": bootstrap,
        "usage_boundary": {
            "allowed": ["reporting", "clustering_diagnostics", "duplicate_diagnostics"],
            "forbidden_without_equal_budget_evidence": ["selection", "pfsp_weighting", "training", "promotion"],
        },
    }


def compare_behavior_fingerprints(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Return an unweighted diagnostic distance over common bounded features."""
    for label, fingerprint in (("left", left), ("right", right)):
        if fingerprint.get("schema_version") != FINGERPRINT_SCHEMA_VERSION:
            raise ValueError(f"{label} fingerprint has unsupported schema_version")
        if fingerprint.get("classification") != CLASSIFICATION:
            raise ValueError(f"{label} fingerprint is not diagnostic_only")
    left_vector = _available_vector(_as_dict(left.get("features")))
    right_vector = _available_vector(_as_dict(right.get("features")))
    common = sorted(set(left_vector).intersection(right_vector))
    available_union = sorted(set(left_vector).union(right_vector))
    distance, common_count = _vector_distance(left_vector, right_vector)
    if distance is None:
        availability = "missing"
        similarity = None
        per_feature_delta: dict[str, float] = {}
        reason = "no common available bounded features"
    else:
        availability = "available"
        similarity = max(0.0, 1.0 - distance)
        per_feature_delta = {name: right_vector[name] - left_vector[name] for name in common}
        reason = None

    return {
        "schema_version": DISTANCE_SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "availability": availability,
        "left_subject_id": left.get("subject_id"),
        "right_subject_id": right.get("subject_id"),
        "common_features": common,
        "common_feature_count": common_count,
        "coverage_fraction_of_declared_features": common_count / len(FEATURE_NAMES),
        "coverage_fraction_of_available_union": (
            common_count / len(available_union) if available_union else 0.0
        ),
        "per_feature_delta_right_minus_left": per_feature_delta,
        "reason": reason,
        "root_mean_square_distance": distance,
        "similarity": similarity,
        "similarity_definition": "one_minus_rms_distance_on_common_unit_interval_features",
        "usage_boundary": "diagnostic_only; not a selection, PFSP, training, or promotion score",
    }
