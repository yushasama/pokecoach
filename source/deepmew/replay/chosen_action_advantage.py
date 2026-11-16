"""Build chosen-action advantage rows from pre/post refgame observations.

Stage 1 intentionally scores only the logged chosen action. It does not branch
counterfactual legal actions and does not change AWR training.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from deepmew.policy.attacks import attack_pressure
from deepmew.replay.information_boundary import InformationMode, SCHEMA_VERSION as INFORMATION_BOUNDARY_SCHEMA_VERSION
from deepmew.state_machine.transition import analyze_board
from deepmew.trace.record import board_summary, summarize_options


SCHEMA_VERSION = "chosen_action_advantage.v1"
TRANSITION_SCHEMA_VERSION = "deepmew.chosen_action_transition.v1"
VALUE_SCALE_WIN_PROBABILITY = "win_probability"
VALUE_SCALE_SIGNED_VALUE = "signed_value"
VALUE_SCALE_LOGIT = "logit"
VALUE_SCALE_UNKNOWN = "unknown"
VALUE_ACTIVATION_SIGMOID = "sigmoid"
SUPPORTED_ADVANTAGE_VALUE_SCALES = {VALUE_SCALE_WIN_PROBABILITY, VALUE_SCALE_SIGNED_VALUE}


class ValueScorer(Protocol):
    value_scale: str
    value_activation_used: str | None

    def __call__(self, observation: dict[str, Any]) -> float:
        """Return value for observation.current.yourIndex on ``value_scale``."""


@dataclass(frozen=True)
class AdvantageBuildResult:
    rows: list[dict[str, Any]]
    summary: dict[str, Any]
    report: str


@dataclass(frozen=True)
class ValueMetadata:
    value_scale: str
    terminal_override_scale: str
    value_activation_used: str | None


@dataclass
class _Stats:
    source_decisions: int = 0
    with_post_observation: int = 0
    missing_post_observation: int = 0
    fresh_transition_rows: int = 0
    fresh_missing_post_observation: int = 0
    valid_perspective: int = 0
    unsafe_perspective: int = 0
    terminal_transitions: int = 0
    mismatch_errors: int = 0
    value_rows: int = 0
    dry_run_rows: int = 0
    rows_with_v_before: int = 0
    rows_with_v_after: int = 0
    rows_with_advantage: int = 0
    invalid_chosen_index_rows: int = 0
    chosen_action_mismatch_rows: int = 0
    legal_options_preserved_rows: int = 0
    legal_options_not_preserved_rows: int = 0
    pre_post_equal_rows: int = 0
    pre_post_hash_equal_rows: int = 0
    pre_post_same_object_rows: int = 0
    terminal_result_mismatch_rows: int = 0
    warnings: Counter = field(default_factory=Counter)
    source_schemas: Counter = field(default_factory=Counter)
    transition_schemas: Counter = field(default_factory=Counter)
    by_action_type: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    by_phase: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_int(value: Any, default: int | None = 0) -> int | None:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _current(obs: dict[str, Any] | None) -> dict[str, Any]:
    return _as_dict(_as_dict(obs).get("current"))


def _your_index(obs: dict[str, Any] | None) -> int | None:
    return _as_int(_current(obs).get("yourIndex"), None)


def _turn(obs: dict[str, Any] | None) -> int | None:
    return _as_int(_current(obs).get("turn"), None)


def _result(obs: dict[str, Any] | None) -> int | None:
    return _as_int(_current(obs).get("result"), None)


def _is_terminal_result(result: int | None) -> bool:
    return result is not None and result != -1


def _terminal_payoff(result: int | None, actor: int | None, value_scale: str) -> float | None:
    if result is None or actor is None or result == -1:
        return None
    if value_scale not in SUPPORTED_ADVANTAGE_VALUE_SCALES:
        return None
    if result == actor:
        return 1.0
    if result in (0, 1):
        return 0.0 if value_scale == VALUE_SCALE_WIN_PROBABILITY else -1.0
    return 0.5 if value_scale == VALUE_SCALE_WIN_PROBABILITY else 0.0


def _terminal_override_scale(value_scale: str) -> str:
    if value_scale in SUPPORTED_ADVANTAGE_VALUE_SCALES:
        return value_scale
    return VALUE_SCALE_UNKNOWN


def _resolve_value_metadata(
    scorer: ValueScorer | None,
    *,
    value_scale: str | None,
    value_activation_used: str | None,
) -> ValueMetadata:
    if value_scale is not None:
        scale = str(value_scale)
    elif scorer is not None and getattr(scorer, "value_scale", None):
        scale = str(getattr(scorer, "value_scale"))
    else:
        scale = VALUE_SCALE_UNKNOWN
    if value_activation_used is not None:
        activation = str(value_activation_used)
    elif scorer is not None and getattr(scorer, "value_activation_used", None):
        activation = str(getattr(scorer, "value_activation_used"))
    else:
        activation = None

    return ValueMetadata(
        value_scale=scale,
        terminal_override_scale=_terminal_override_scale(scale),
        value_activation_used=activation,
    )


def _expected_value_range(value_scale: str) -> tuple[float, float] | None:
    if value_scale == VALUE_SCALE_WIN_PROBABILITY:
        return 0.0, 1.0
    if value_scale == VALUE_SCALE_SIGNED_VALUE:
        return -1.0, 1.0
    return None


def _value_in_expected_range(value: float, value_scale: str) -> bool:
    expected = _expected_value_range(value_scale)
    if expected is None:
        return False
    lower, upper = expected
    return lower <= value <= upper


def _stable_observation_hash(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _decision_id(row: dict[str, Any]) -> str:
    if row.get("frame_id"):
        return str(row["frame_id"])
    game_id = str(row.get("game_id") or "unknown_game")
    decision_index = _as_int(row.get("decision_index"), _as_int(row.get("step"), 0))
    return f"{game_id}:d{int(decision_index or 0):04d}"


def _legal_options(row: dict[str, Any], raw_observation: dict[str, Any] | None) -> tuple[list[Any], str]:
    legal_set = row.get("legal_set")
    if isinstance(legal_set, list):
        return legal_set, "legal_set"
    options = _as_dict(_as_dict(raw_observation).get("select")).get("option")
    if isinstance(options, list):
        return options, "raw_observation.select.option"
    return [], "missing"


def _normalize_indices(value: Any) -> tuple[list[int], list[str]]:
    warnings: list[str] = []
    indices: list[int] = []
    for item in _as_list(value):
        if isinstance(item, bool) or not isinstance(item, int):
            warnings.append("chosen_index_not_int")
            continue
        indices.append(int(item))

    return indices, warnings


def _chosen_indices(row: dict[str, Any]) -> list[int]:
    chosen_action = _as_dict(row.get("chosen_action"))
    chosen = chosen_action.get("indices")
    if chosen is None:
        chosen = row.get("chosen_indices")
    indices, _warnings = _normalize_indices(chosen)

    return indices


def _chosen_action_type(row: dict[str, Any], legal_options: list[Any], chosen_indices: list[int]) -> str:
    chosen_action = _as_dict(row.get("chosen_action"))
    semantics = _as_list(chosen_action.get("semantics"))
    first = _as_dict(semantics[0]) if semantics else {}
    if first.get("option_type"):
        return str(first["option_type"])
    if chosen_indices:
        index = chosen_indices[0]
        if 0 <= index < len(legal_options):
            option = _as_dict(legal_options[index])
            if option.get("option_type"):
                return str(option["option_type"])
            if option.get("type") is not None:
                return str(option["type"])
    return "unknown"


def _phase(row: dict[str, Any], raw_observation: dict[str, Any] | None, legal_options: list[Any]) -> str:
    if legal_options:
        first = _as_dict(legal_options[0])
        if first.get("select_context"):
            return str(first["select_context"])
    select = _as_dict(_as_dict(raw_observation).get("select"))
    context = select.get("context")
    return str(context) if context is not None else "unknown"


def _final_result(row: dict[str, Any]) -> int | None:
    outcome = _as_dict(row.get("outcome"))
    value = outcome.get("result", row.get("final_result", row.get("result")))
    return _as_int(value, None)


def _result_reason(row: dict[str, Any]) -> str | None:
    outcome = _as_dict(row.get("outcome"))
    value = outcome.get("result_reason", row.get("final_result_reason", row.get("result_reason")))
    return str(value) if value is not None else None


def _safe_board_summary(obs: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(obs, dict):
        return None, "missing_observation"
    try:
        select = _as_dict(obs.get("select"))
        legal = summarize_options(select, obs)
        return board_summary(obs, legal), None
    except Exception as exc:  # pragma: no cover - defensive against card-table failures
        return None, f"{type(exc).__name__}: {exc}"


def _own_player(obs: dict[str, Any] | None) -> dict[str, Any]:
    current = _current(obs)
    players = _as_list(current.get("players"))
    seat = _your_index(obs)
    if seat is None or not (0 <= seat < len(players)):
        return {}
    return _as_dict(players[seat])


def _active(player: dict[str, Any]) -> dict[str, Any]:
    active = _as_list(player.get("active"))
    return _as_dict(active[0]) if active else {}


def _energy_units(pokemon: dict[str, Any]) -> int:
    return len(_as_list(pokemon.get("energies"))) + len(_as_list(pokemon.get("energyCards")))


def _energy_metrics(obs: dict[str, Any] | None) -> dict[str, int]:
    player = _own_player(obs)
    active = _active(player)
    bench = [_as_dict(item) for item in _as_list(player.get("bench")) if item]
    in_play = ([active] if active else []) + bench
    bench_energy = sum(_energy_units(item) for item in bench)
    total_energy = sum(_energy_units(item) for item in in_play)
    max_energy = max((_energy_units(item) for item in in_play), default=0)
    return {
        "board_width": len(in_play),
        "bench_energy": bench_energy,
        "total_energy": total_energy,
        "max_energy": max_energy,
    }


def _attack_pressure_metrics(obs: dict[str, Any] | None) -> dict[str, int]:
    if not isinstance(obs, dict):
        return {"ko": 0, "meaningful": 0, "weak": 0, "unknown": 0}
    select = _as_dict(obs.get("select"))
    metrics = {"ko": 0, "meaningful": 0, "weak": 0, "unknown": 0}
    for option in _as_list(select.get("option")):
        pressure = attack_pressure(_as_dict(option), obs, use_card_semantics=True)
        if pressure not in metrics:
            pressure = "unknown"
        metrics[pressure] += 1

    return metrics


def _tempo_diagnostics(raw_observation: dict[str, Any] | None, post_observation: dict[str, Any] | None, *, same_actor: bool) -> dict[str, Any]:
    if not same_actor or not isinstance(raw_observation, dict) or not isinstance(post_observation, dict):
        return {
            "available": False,
            "unavailable": [
                "delta_legal_attack_available",
                "delta_best_shortfall",
                "delta_energy_attached",
                "delta_board_width",
                "delta_total_energy",
                "delta_max_energy",
                "delta_bench_energy",
                "delta_visible_ko_attack_pressure",
                "delta_meaningful_attack_pressure",
                "delta_weak_attack_pressure",
            ],
        }
    before = analyze_board(raw_observation)
    after = analyze_board(post_observation)
    before_energy = _energy_metrics(raw_observation)
    after_energy = _energy_metrics(post_observation)
    before_pressure = _attack_pressure_metrics(raw_observation)
    after_pressure = _attack_pressure_metrics(post_observation)
    diagnostics: dict[str, Any] = {
        "available": True,
        "legal_attack_available_before": before.legal_attack_available,
        "legal_attack_available_after": after.legal_attack_available,
        "delta_legal_attack_available": int(after.legal_attack_available) - int(before.legal_attack_available),
        "energy_attached_before": before.energy_attached,
        "energy_attached_after": after.energy_attached,
        "delta_energy_attached": int(after.energy_attached) - int(before.energy_attached),
        "best_shortfall_before": before.best_shortfall,
        "best_shortfall_after": after.best_shortfall,
        "delta_best_shortfall": None,
    }
    if before.best_shortfall is not None and after.best_shortfall is not None:
        diagnostics["delta_best_shortfall"] = after.best_shortfall - before.best_shortfall
    for key, before_value in before_energy.items():
        after_value = after_energy[key]
        diagnostics[f"{key}_before"] = before_value
        diagnostics[f"{key}_after"] = after_value
        diagnostics[f"delta_{key}"] = after_value - before_value
    for key, before_value in before_pressure.items():
        after_value = after_pressure[key]
        diagnostics[f"{key}_attack_pressure_before"] = before_value
        diagnostics[f"{key}_attack_pressure_after"] = after_value
        diagnostics[f"delta_{key}_attack_pressure"] = after_value - before_value
    return diagnostics


def _score_value(
    scorer: ValueScorer | None,
    obs: dict[str, Any],
    warnings: list[str],
    *,
    metadata: ValueMetadata,
) -> float | None:
    if scorer is None:
        return None
    if metadata.value_scale not in SUPPORTED_ADVANTAGE_VALUE_SCALES:
        warnings.append(f"unsupported_value_scale:{metadata.value_scale}")
        return None
    try:
        value = float(scorer(obs))
    except Exception as exc:
        warnings.append(f"value_scorer_error:{type(exc).__name__}")
        return None
    if not math.isfinite(value):
        warnings.append("value_scorer_nonfinite")
        return None
    if not _value_in_expected_range(value, metadata.value_scale):
        warnings.append(f"value_outside_{metadata.value_scale}_scale")
        return None
    return value


def _observation_integrity(raw_observation: dict[str, Any] | None, post_observation: dict[str, Any] | None) -> dict[str, Any]:
    raw_hash = _stable_observation_hash(raw_observation)
    post_hash = _stable_observation_hash(post_observation)
    both = isinstance(raw_observation, dict) and isinstance(post_observation, dict)

    return {
        "raw_observation_present": isinstance(raw_observation, dict),
        "post_observation_present": isinstance(post_observation, dict),
        "raw_post_same_object": bool(both and raw_observation is post_observation),
        "raw_post_equal": bool(both and raw_observation == post_observation),
        "raw_observation_sha256": raw_hash,
        "post_observation_sha256": post_hash,
        "raw_post_hash_equal": bool(raw_hash is not None and raw_hash == post_hash),
    }


def _action_integrity(
    row: dict[str, Any],
    legal_options: list[Any],
    legal_options_source: str,
    chosen_indices: list[int],
    warnings: list[str],
) -> dict[str, Any]:
    chosen_action = _as_dict(row.get("chosen_action"))
    row_indices, row_index_warnings = _normalize_indices(row.get("chosen_indices"))
    action_indices, action_index_warnings = _normalize_indices(chosen_action.get("indices"))
    warnings.extend(row_index_warnings)
    warnings.extend(action_index_warnings)
    invalid_indices = [index for index in chosen_indices if index < 0 or index >= len(legal_options)]
    if invalid_indices:
        warnings.append("chosen_index_out_of_range")
    if chosen_indices and not legal_options:
        warnings.append("chosen_indices_without_legal_options")
    if row_indices and action_indices and row_indices != action_indices:
        warnings.append("chosen_action_indices_mismatch")
    semantics = _as_list(chosen_action.get("semantics"))
    semantics_match = None
    if semantics and len(semantics) == len(chosen_indices):
        mismatches = []
        for index, semantic in zip(chosen_indices, semantics, strict=False):
            if 0 <= index < len(legal_options) and _as_dict(semantic) != _as_dict(legal_options[index]):
                mismatches.append(index)
        semantics_match = not mismatches
        if mismatches:
            warnings.append("chosen_action_semantics_mismatch")
    legal_set = row.get("legal_set")
    if isinstance(legal_set, list):
        preserved = legal_options == legal_set
    elif legal_options_source == "raw_observation.select.option":
        raw_options = _as_list(_as_dict(_as_dict(row.get("raw_observation")).get("select")).get("option"))
        preserved = legal_options == raw_options
    else:
        preserved = None
    if preserved is False:
        warnings.append("legal_options_not_preserved_exactly")

    return {
        "legal_option_count": len(legal_options),
        "legal_options_source": legal_options_source,
        "legal_options_preserved_exact": preserved,
        "chosen_indices_valid": not invalid_indices,
        "invalid_chosen_indices": invalid_indices,
        "chosen_action_indices_match": False if row_indices and action_indices and row_indices != action_indices else True,
        "chosen_action_semantics_match": semantics_match,
    }


def build_advantage_row(
    row: dict[str, Any],
    *,
    source: str,
    scorer: ValueScorer | None = None,
    value_scale: str | None = None,
    value_activation_used: str | None = None,
) -> dict[str, Any]:
    metadata = _resolve_value_metadata(
        scorer,
        value_scale=value_scale,
        value_activation_used=value_activation_used,
    )
    raw_observation = row.get("raw_observation") if isinstance(row.get("raw_observation"), dict) else None
    post_observation = row.get("post_observation") if isinstance(row.get("post_observation"), dict) else None
    actor = _as_int(row.get("seat", row.get("player_id")), _your_index(raw_observation))
    raw_actor = _your_index(raw_observation)
    post_actor = _your_index(post_observation)
    post_result = _result(post_observation)
    is_terminal = _is_terminal_result(post_result)
    terminal_payoff = _terminal_payoff(post_result, actor, metadata.value_scale)
    legal_options, legal_options_source = _legal_options(row, raw_observation)
    chosen_indices = _chosen_indices(row)
    action_type = _chosen_action_type(row, legal_options, chosen_indices)
    phase = _phase(row, raw_observation, legal_options)
    warnings: list[str] = []
    action_integrity = _action_integrity(row, legal_options, legal_options_source, chosen_indices, warnings)
    observation_integrity = _observation_integrity(raw_observation, post_observation)

    if raw_observation is None:
        warnings.append("missing_raw_observation")
    if post_observation is None:
        warnings.append("missing_post_observation")
        if row.get("transition_schema_version") == TRANSITION_SCHEMA_VERSION:
            warnings.append("fresh_row_missing_post_observation")
    if row.get("schema_version") is None:
        warnings.append("missing_source_schema_version")
    if actor is None:
        warnings.append("missing_actor")
    if raw_actor is None:
        warnings.append("missing_raw_your_index")
    elif actor is not None and raw_actor != actor:
        warnings.append("raw_observation_not_actor_perspective")
    final_result = _final_result(row)
    if is_terminal and final_result is not None and post_result != final_result:
        warnings.append("terminal_post_result_final_result_mismatch")
    if scorer is not None and metadata.value_scale == VALUE_SCALE_UNKNOWN:
        warnings.append("unknown_value_scale")

    raw_perspective_ok = actor is not None and raw_actor == actor
    same_actor = bool(raw_perspective_ok and post_actor == actor and not is_terminal)
    if is_terminal:
        perspective_status = "terminal"
    elif same_actor:
        perspective_status = "same_actor"
    elif post_observation is None:
        perspective_status = "unavailable"
    else:
        perspective_status = "unavailable"
        warnings.append("post_observation_not_actor_perspective")

    v_before: float | None = None
    v_after: float | None = None
    advantage: float | None = None
    advantage_source = "unavailable"
    if raw_perspective_ok and isinstance(raw_observation, dict):
        v_before = _score_value(scorer, raw_observation, warnings, metadata=metadata)
    if is_terminal:
        if v_before is not None and terminal_payoff is not None:
            advantage = terminal_payoff - v_before
            advantage_source = "terminal_override"
    elif same_actor and isinstance(post_observation, dict):
        v_after = _score_value(scorer, post_observation, warnings, metadata=metadata)
        if v_before is not None and v_after is not None:
            advantage = v_after - v_before
            advantage_source = "value_delta"

    before_summary, before_summary_error = _safe_board_summary(raw_observation)
    after_summary, after_summary_error = _safe_board_summary(post_observation)
    if before_summary_error:
        warnings.append(f"board_summary_before:{before_summary_error}")
    if after_summary_error and post_observation is not None:
        warnings.append(f"board_summary_after:{after_summary_error}")

    return {
        "schema_version": SCHEMA_VERSION,
        "information_boundary_schema_version": row.get("information_boundary_schema_version", INFORMATION_BOUNDARY_SCHEMA_VERSION),
        "information_mode": row.get("information_mode", InformationMode.PUBLIC_BELIEF.value),
        "information_mode_inferred": row.get("information_mode") is None,
        "source": row.get("source_shard") or source,
        "source_dataset": row.get("source_dataset"),
        "source_shard": row.get("source_shard"),
        "source_schema_version": row.get("schema_version"),
        "transition_schema_version": row.get("transition_schema_version"),
        "value_scale": metadata.value_scale,
        "terminal_override_scale": metadata.terminal_override_scale,
        "value_activation_used": metadata.value_activation_used,
        "game_id": row.get("game_id"),
        "episode_id": row.get("episode_id"),
        "frame_id": row.get("frame_id"),
        "decision_id": _decision_id(row),
        "decision_index": row.get("decision_index", row.get("step")),
        "actor": actor,
        "raw_actor": raw_actor,
        "post_actor": post_actor,
        "turn": row.get("turn", _turn(raw_observation)),
        "post_turn": row.get("post_turn", _turn(post_observation)),
        "phase": phase,
        "seed": row.get("seed"),
        "raw_observation": raw_observation,
        "post_observation": post_observation,
        "legal_options": legal_options,
        "legal_options_source": legal_options_source,
        "chosen_indices": chosen_indices,
        "chosen_action": row.get("chosen_action"),
        "chosen_action_type": action_type,
        "action_integrity": action_integrity,
        "observation_integrity": observation_integrity,
        "final_result": final_result,
        "post_result": post_result,
        "terminal_result_for_actor": terminal_payoff,
        "result_reason": _result_reason(row),
        "board_summary_before": before_summary,
        "board_summary_after": after_summary,
        "v_before": v_before,
        "v_after": v_after,
        "advantage": advantage,
        "advantage_source": advantage_source,
        "perspective_status": perspective_status,
        "tempo_diagnostics": _tempo_diagnostics(raw_observation, post_observation, same_actor=same_actor),
        "warnings": warnings,
    }


def _distribution(values: Iterable[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return {"count": 0, "min": None, "max": None, "mean": None, "p50": None}
    count = len(ordered)
    return {
        "count": count,
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / count,
        "p50": ordered[count // 2],
    }


def _value_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def _warning_count(summary: dict[str, Any], warning: str) -> int:
    return int(_as_dict(summary.get("warnings")).get(warning, 0))


def _suspicious_patterns(rows: list[dict[str, Any]], summary: dict[str, Any]) -> list[str]:
    patterns: list[str] = []
    advantages = [float(row["advantage"]) for row in rows if isinstance(row.get("advantage"), (int, float))]
    if advantages and all(abs(value) <= 1e-12 for value in advantages):
        patterns.append("all_advantages_are_zero")
    if len(advantages) >= 3:
        positives = sum(1 for value in advantages if value > 0)
        negatives = sum(1 for value in advantages if value < 0)
        if max(positives, negatives) / len(advantages) >= 0.95:
            patterns.append("nearly_all_advantages_have_same_sign")
    if _warning_count(summary, "value_outside_win_probability_scale") or _warning_count(summary, "value_outside_signed_value_scale"):
        patterns.append("value_outside_expected_scale")
    if _warning_count(summary, "fresh_row_missing_post_observation"):
        patterns.append("post_observation_missing_on_fresh_rows")
    fresh_rows = int(summary.get("fresh_transition_rows") or 0)
    if fresh_rows and int(summary.get("fresh_perspective_unavailable_or_unsafe") or 0) / fresh_rows > 0.5:
        patterns.append("perspective_status_unavailable_for_most_fresh_rows")
    rows_with_post = int(summary.get("with_post_observation") or 0)
    if rows_with_post and int(summary.get("pre_post_equal_rows") or 0) / rows_with_post > 0.5:
        patterns.append("post_observation_equals_raw_observation_for_most_rows")
    if summary.get("mixed_old_and_fresh_rows"):
        patterns.append("old_and_fresh_schemas_mixed_in_source")
    scale = summary.get("value_scale")
    terminal_scale = summary.get("terminal_override_scale")
    if scale in SUPPORTED_ADVANTAGE_VALUE_SCALES and terminal_scale != scale:
        patterns.append("terminal_override_scale_differs_from_value_scale")
    if _warning_count(summary, "chosen_index_out_of_range"):
        patterns.append("invalid_chosen_indices_present")
    if _warning_count(summary, "chosen_action_semantics_mismatch") or _warning_count(summary, "chosen_action_indices_mismatch"):
        patterns.append("chosen_action_mismatch_present")

    return patterns


def _compact_example(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_id": row.get("decision_id"),
        "game_id": row.get("game_id"),
        "frame_id": row.get("frame_id"),
        "actor": row.get("actor"),
        "turn": row.get("turn"),
        "phase": row.get("phase"),
        "chosen_action_type": row.get("chosen_action_type"),
        "chosen_indices": row.get("chosen_indices"),
        "legal_option_count": _as_dict(row.get("action_integrity")).get("legal_option_count"),
        "v_before": row.get("v_before"),
        "v_after": row.get("v_after"),
        "advantage": row.get("advantage"),
        "advantage_source": row.get("advantage_source"),
        "perspective_status": row.get("perspective_status"),
        "board_summary_before": row.get("board_summary_before"),
        "board_summary_after": row.get("board_summary_after"),
        "warnings": row.get("warnings"),
    }


def summarize_advantage_rows(rows: list[dict[str, Any]], stats: _Stats) -> dict[str, Any]:
    v_before = [row["v_before"] for row in rows if isinstance(row.get("v_before"), (int, float))]
    v_after = [row["v_after"] for row in rows if isinstance(row.get("v_after"), (int, float))]
    advantages = [row["advantage"] for row in rows if isinstance(row.get("advantage"), (int, float))]
    positives = sorted((row for row in rows if isinstance(row.get("advantage"), (int, float))), key=lambda item: item["advantage"], reverse=True)[:5]
    negatives = sorted((row for row in rows if isinstance(row.get("advantage"), (int, float))), key=lambda item: item["advantage"])[:5]
    near_zero = sorted((row for row in rows if isinstance(row.get("advantage"), (int, float))), key=lambda item: abs(item["advantage"]))[:5]
    terminals = [row for row in rows if row.get("advantage_source") == "terminal_override"][:5]
    unsafe = [row for row in rows if row.get("perspective_status") == "unavailable"][:5]
    missing_post = [row for row in rows if "missing_post_observation" in _as_list(row.get("warnings"))][:5]
    fresh_unsafe = sum(
        1
        for row in rows
        if row.get("transition_schema_version") == TRANSITION_SCHEMA_VERSION
        and row.get("perspective_status") == "unavailable"
    )
    value_scales = _value_counts(rows, "value_scale")
    terminal_scales = _value_counts(rows, "terminal_override_scale")
    value_activations = _value_counts(rows, "value_activation_used")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "source_decisions_read": stats.source_decisions,
        "rows_emitted": len(rows),
        "with_post_observation": stats.with_post_observation,
        "missing_post_observation": stats.missing_post_observation,
        "fresh_transition_rows": stats.fresh_transition_rows,
        "old_format_rows": len(rows) - stats.fresh_transition_rows,
        "fresh_missing_post_observation": stats.fresh_missing_post_observation,
        "valid_perspective_handling": stats.valid_perspective,
        "perspective_unavailable_or_unsafe": stats.unsafe_perspective,
        "fresh_perspective_unavailable_or_unsafe": fresh_unsafe,
        "terminal_transitions": stats.terminal_transitions,
        "mismatch_errors_count": stats.mismatch_errors,
        "dry_run_rows": stats.dry_run_rows,
        "value_rows": stats.value_rows,
        "rows_with_v_before": stats.rows_with_v_before,
        "rows_with_v_after": stats.rows_with_v_after,
        "rows_with_advantage": stats.rows_with_advantage,
        "invalid_chosen_index_rows": stats.invalid_chosen_index_rows,
        "chosen_action_mismatch_rows": stats.chosen_action_mismatch_rows,
        "legal_options_preserved_exact_rows": stats.legal_options_preserved_rows,
        "legal_options_not_preserved_exact_rows": stats.legal_options_not_preserved_rows,
        "pre_post_equal_rows": stats.pre_post_equal_rows,
        "pre_post_hash_equal_rows": stats.pre_post_hash_equal_rows,
        "pre_post_same_object_rows": stats.pre_post_same_object_rows,
        "terminal_result_mismatch_rows": stats.terminal_result_mismatch_rows,
        "source_schema_versions": dict(sorted(stats.source_schemas.items())),
        "transition_schema_versions": dict(sorted(stats.transition_schemas.items())),
        "mixed_old_and_fresh_rows": bool(stats.fresh_transition_rows and len(rows) != stats.fresh_transition_rows),
        "value_scale": next(iter(value_scales)) if len(value_scales) == 1 else "mixed",
        "terminal_override_scale": next(iter(terminal_scales)) if len(terminal_scales) == 1 else "mixed",
        "value_activation_used": next(iter(value_activations)) if len(value_activations) == 1 else "mixed",
        "value_scales": value_scales,
        "terminal_override_scales": terminal_scales,
        "value_activation_used_counts": value_activations,
        "v_before_distribution": _distribution(v_before),
        "v_after_distribution": _distribution(v_after),
        "advantage_distribution": _distribution(advantages),
        "warnings": dict(sorted(stats.warnings.items())),
        "advantage_by_action_type": {key: _distribution(values) for key, values in sorted(stats.by_action_type.items())},
        "advantage_by_phase": {key: _distribution(values) for key, values in sorted(stats.by_phase.items())},
        "top_positive_advantage_examples": [_compact_example(row) for row in positives],
        "top_negative_advantage_examples": [_compact_example(row) for row in negatives],
        "near_zero_advantage_examples": [_compact_example(row) for row in near_zero],
        "terminal_override_examples": [_compact_example(row) for row in terminals],
        "unsafe_or_unavailable_perspective_examples": [_compact_example(row) for row in unsafe],
        "missing_post_observation_examples": [_compact_example(row) for row in missing_post],
        "sample_rows": [_compact_example(row) for row in rows[:5]],
    }
    summary["suspicious_patterns"] = _suspicious_patterns(rows, summary)

    return summary


def _observe_stats(source_row: dict[str, Any], advantage_row: dict[str, Any], stats: _Stats, scorer: ValueScorer | None) -> None:
    stats.source_decisions += 1
    transition_schema = source_row.get("transition_schema_version")
    source_schema = source_row.get("schema_version")
    stats.source_schemas[str(source_schema)] += 1
    stats.transition_schemas[str(transition_schema)] += 1
    if transition_schema == TRANSITION_SCHEMA_VERSION:
        stats.fresh_transition_rows += 1
    if isinstance(source_row.get("post_observation"), dict):
        stats.with_post_observation += 1
    else:
        stats.missing_post_observation += 1
        if transition_schema == TRANSITION_SCHEMA_VERSION:
            stats.fresh_missing_post_observation += 1
    if advantage_row["perspective_status"] in {"same_actor", "terminal"}:
        stats.valid_perspective += 1
    else:
        stats.unsafe_perspective += 1
    if advantage_row["perspective_status"] == "terminal":
        stats.terminal_transitions += 1
    if scorer is None:
        stats.dry_run_rows += 1
    if advantage_row["v_before"] is not None:
        stats.rows_with_v_before += 1
    if advantage_row["v_after"] is not None:
        stats.rows_with_v_after += 1
    if advantage_row["advantage"] is not None:
        stats.value_rows += 1
        stats.rows_with_advantage += 1
        stats.by_action_type[str(advantage_row["chosen_action_type"])].append(float(advantage_row["advantage"]))
        stats.by_phase[str(advantage_row["phase"])].append(float(advantage_row["advantage"]))
    action_integrity = _as_dict(advantage_row.get("action_integrity"))
    if action_integrity.get("chosen_indices_valid") is False:
        stats.invalid_chosen_index_rows += 1
    if action_integrity.get("chosen_action_indices_match") is False or action_integrity.get("chosen_action_semantics_match") is False:
        stats.chosen_action_mismatch_rows += 1
    if action_integrity.get("legal_options_preserved_exact") is True:
        stats.legal_options_preserved_rows += 1
    elif action_integrity.get("legal_options_preserved_exact") is False:
        stats.legal_options_not_preserved_rows += 1
    observation_integrity = _as_dict(advantage_row.get("observation_integrity"))
    if observation_integrity.get("raw_post_equal"):
        stats.pre_post_equal_rows += 1
    if observation_integrity.get("raw_post_hash_equal"):
        stats.pre_post_hash_equal_rows += 1
    if observation_integrity.get("raw_post_same_object"):
        stats.pre_post_same_object_rows += 1
    if "terminal_post_result_final_result_mismatch" in advantage_row["warnings"]:
        stats.terminal_result_mismatch_rows += 1
    if "raw_observation_not_actor_perspective" in advantage_row["warnings"]:
        stats.mismatch_errors += 1
    for warning in advantage_row["warnings"]:
        stats.warnings[str(warning)] += 1


def build_advantage_rows(
    source_rows: Iterable[dict[str, Any]],
    *,
    source: str,
    scorer: ValueScorer | None = None,
    value_scale: str | None = None,
    value_activation_used: str | None = None,
) -> AdvantageBuildResult:
    stats = _Stats()
    rows: list[dict[str, Any]] = []
    for source_row in source_rows:
        advantage_row = build_advantage_row(
            source_row,
            source=source,
            scorer=scorer,
            value_scale=value_scale,
            value_activation_used=value_activation_used,
        )
        rows.append(advantage_row)
        _observe_stats(source_row, advantage_row, stats, scorer)
    summary = summarize_advantage_rows(rows, stats)
    report = render_report(summary)
    return AdvantageBuildResult(rows=rows, summary=summary, report=report)


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Chosen Action Advantage Stage 1.5 Validation",
        "",
        f"schema_version={summary['schema_version']}",
        f"source_decisions_read={summary['source_decisions_read']} rows_emitted={summary['rows_emitted']}",
        f"with_post_observation={summary['with_post_observation']} missing_post_observation={summary['missing_post_observation']}",
        f"fresh_transition_rows={summary['fresh_transition_rows']} old_format_rows={summary['old_format_rows']} "
        f"fresh_missing_post_observation={summary['fresh_missing_post_observation']}",
        f"valid_perspective_handling={summary['valid_perspective_handling']} "
        f"perspective_unavailable_or_unsafe={summary['perspective_unavailable_or_unsafe']}",
        f"fresh_perspective_unavailable_or_unsafe={summary['fresh_perspective_unavailable_or_unsafe']}",
        f"terminal_transitions={summary['terminal_transitions']} mismatch_errors_count={summary['mismatch_errors_count']}",
        f"value_rows={summary['value_rows']} rows_with_v_before={summary['rows_with_v_before']} "
        f"rows_with_v_after={summary['rows_with_v_after']} dry_run_rows={summary['dry_run_rows']}",
        "",
        "## Value Scale",
        "",
        f"value_scale={summary['value_scale']}",
        f"terminal_override_scale={summary['terminal_override_scale']}",
        f"value_activation_used={summary['value_activation_used']}",
        f"value_scales={summary['value_scales']}",
        f"terminal_override_scales={summary['terminal_override_scales']}",
        f"value_activation_used_counts={summary['value_activation_used_counts']}",
        "",
        "## Perspective Audit",
        "",
        "- `raw_observation.current.yourIndex` is treated as the acting player for a captured decision.",
        "- `post_observation` is safe for value-delta scoring only when its `current.yourIndex` equals the original actor.",
        "- `current.result` is a global winner seat (`0` or `1`), with other terminal codes treated as draw-like.",
        "- If `post_observation` flips to the opponent and is not terminal, Stage 1 marks advantage unavailable rather than complementing a value that may include the next actor's private observation.",
        "- Terminal overrides are actor-relative and use the same scale recorded in `terminal_override_scale`.",
        "",
        "## Integrity",
        "",
        f"invalid_chosen_index_rows={summary['invalid_chosen_index_rows']}",
        f"chosen_action_mismatch_rows={summary['chosen_action_mismatch_rows']}",
        f"legal_options_preserved_exact_rows={summary['legal_options_preserved_exact_rows']} "
        f"legal_options_not_preserved_exact_rows={summary['legal_options_not_preserved_exact_rows']}",
        f"pre_post_equal_rows={summary['pre_post_equal_rows']} "
        f"pre_post_hash_equal_rows={summary['pre_post_hash_equal_rows']} "
        f"pre_post_same_object_rows={summary['pre_post_same_object_rows']}",
        f"terminal_result_mismatch_rows={summary['terminal_result_mismatch_rows']}",
        f"source_schema_versions={summary['source_schema_versions']}",
        f"transition_schema_versions={summary['transition_schema_versions']}",
        "",
        "## Distributions",
        "",
        f"v_before={summary['v_before_distribution']}",
        f"v_after={summary['v_after_distribution']}",
        f"advantage={summary['advantage_distribution']}",
        "",
        "## Grouped Advantage",
        "",
        f"by_action_type={summary['advantage_by_action_type']}",
        f"by_phase={summary['advantage_by_phase']}",
        "",
        "## Examples",
        "",
        "top_positive=" + json.dumps(summary["top_positive_advantage_examples"], sort_keys=True),
        "top_negative=" + json.dumps(summary["top_negative_advantage_examples"], sort_keys=True),
        "near_zero=" + json.dumps(summary["near_zero_advantage_examples"], sort_keys=True),
        "terminal_override=" + json.dumps(summary["terminal_override_examples"], sort_keys=True),
        "unsafe_or_unavailable_perspective=" + json.dumps(summary["unsafe_or_unavailable_perspective_examples"], sort_keys=True),
        "missing_post_observation=" + json.dumps(summary["missing_post_observation_examples"], sort_keys=True),
        "sample_rows=" + json.dumps(summary["sample_rows"], sort_keys=True),
        "",
        "## Suspicious Patterns",
        "",
        json.dumps(summary["suspicious_patterns"], indent=2, sort_keys=True),
        "",
        "## Warnings",
        "",
        json.dumps(summary["warnings"], indent=2, sort_keys=True),
    ]
    return "\n".join(lines)


def iter_refgame_rows(refgames_dir: str | Path) -> Iterable[dict[str, Any]]:
    shard_dir = Path(refgames_dir) / "shards"
    for shard in sorted(shard_dir.glob("decisions_shard_*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def iter_jsonl_rows(paths: Iterable[str | Path]) -> Iterable[dict[str, Any]]:
    for path_like in paths:
        path = Path(path_like)
        opener: Callable[..., Any] = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
