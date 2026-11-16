"""Audit Stage 2A chosen-action labels without changing trainer behavior."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from deepmew.artifacts.writers import write_json


SCHEMA_VERSION = "deepmew.chosen_action_advantage.signal_probe_data.v1"
FROZEN_BETA = 2.0
FROZEN_WEIGHT_MIN = 0.1
FROZEN_WEIGHT_MAX = 3.0


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)

    return number if math.isfinite(number) else None


def _counts(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value or "not_found") for value in values).items()))


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end

    return ranks


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_norm = sum((value - left_mean) ** 2 for value in left)
    right_norm = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_norm * right_norm)

    return numerator / denominator if denominator > 0.0 else None


def _ess(weights: list[float]) -> float | None:
    denominator = sum(weight * weight for weight in weights)

    return (sum(weights) ** 2) / denominator if denominator > 0.0 else None


def _sign(value: float, tolerance: float = 1e-12) -> str:
    if value > tolerance:
        return "positive"
    if value < -tolerance:
        return "negative"

    return "zero"


def _return_advantages(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_game[str(row.get("game_id") or "not_found")].append(row)

    comparisons: list[dict[str, Any]] = []
    for game_rows in by_game.values():
        game_rows.sort(key=lambda row: (_as_int(row.get("decision_index")) or 0, str(row.get("decision_id") or "")))
        for root_index, root in enumerate(game_rows):
            root_v = _finite(root.get("v_before"))
            immediate = _finite(root.get("advantage"))
            if root_v is None:
                continue
            control_pass_index = None
            for candidate_index in range(root_index, len(game_rows)):
                candidate = game_rows[candidate_index]
                actor = _as_int(candidate.get("actor"))
                post_actor = _as_int(candidate.get("post_actor"))
                if actor is not None and post_actor is not None and actor != post_actor:
                    control_pass_index = candidate_index
                    break
                if str(candidate.get("perspective_status")) == "terminal":
                    break
            if control_pass_index is None or control_pass_index + 1 >= len(game_rows):
                continue
            return_row = game_rows[control_pass_index + 1]
            return_v = _finite(return_row.get("v_before"))
            if return_v is None or _as_int(return_row.get("actor")) != _as_int(root.get("actor")):
                continue
            comparisons.append({
                "decision_id": root.get("decision_id"),
                "action_type": root.get("chosen_action_type"),
                "phase": root.get("phase"),
                "a_immediate": immediate,
                "a_return": return_v - root_v,
            })

    return comparisons


def audit(rows: list[dict[str, Any]], *, source: str) -> dict[str, Any]:
    unavailable = [row for row in rows if str(row.get("perspective_status")) == "unavailable"]
    eligible = [row for row in rows if _finite(row.get("advantage")) is not None]
    terminal = [row for row in eligible if str(row.get("perspective_status")) == "terminal"]
    weights_raw = [
        min(FROZEN_WEIGHT_MAX, max(FROZEN_WEIGHT_MIN, 1.0 + FROZEN_BETA * float(row["advantage"])))
        for row in eligible
    ]
    mean_weight = sum(weights_raw) / len(weights_raw) if weights_raw else 1.0
    weights = [weight / mean_weight for weight in weights_raw]
    terminal_ids = {str(row.get("decision_id")) for row in terminal}
    terminal_weights = [
        weight for row, weight in zip(eligible, weights)
        if str(row.get("decision_id")) in terminal_ids
    ]
    nonterminal_weights = [
        weight for row, weight in zip(eligible, weights)
        if str(row.get("decision_id")) not in terminal_ids
    ]
    returns = _return_advantages(rows)
    paired = [row for row in returns if row["a_immediate"] is not None]
    immediate_values = [float(row["a_immediate"]) for row in paired]
    return_values = [float(row["a_return"]) for row in paired]
    disagreements = [row for row in paired if _sign(float(row["a_immediate"])) != _sign(float(row["a_return"]))]

    return {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "rows": len(rows),
        "information_modes": _counts(row.get("information_mode") for row in rows),
        "probe_1_unavailable_perspective": {
            "count": len(unavailable),
            "fraction": len(unavailable) / len(rows) if rows else None,
            "action_types": _counts(row.get("chosen_action_type") for row in unavailable),
            "phases": _counts(row.get("phase") for row in unavailable),
            "actor_seat_transitions": _counts(
                f"{row.get('actor')}->{row.get('post_actor')}" for row in unavailable
            ),
            "control_pass_count": sum(
                _as_int(row.get("actor")) != _as_int(row.get("post_actor")) for row in unavailable
            ),
            "turn_changed_count": sum(
                _as_int(row.get("turn")) != _as_int(row.get("post_turn")) for row in unavailable
            ),
            "attack_end_count": sum(
                str(row.get("chosen_action_type")) in {"ATTACK", "END"} for row in unavailable
            ),
        },
        "probe_2_transition_class": {
            "deterministic": 0,
            "stochastic": 0,
            "unknown": len(rows),
            "reason": "no validated transition-effect or consumed-RNG field is persisted",
        },
        "probe_3_horizons": {
            "immediate_coverage": len(eligible),
            "turn_coverage": 0,
            "return_coverage": len(returns),
            "immediate_return_paired": len(paired),
            "sign_agreement": (
                sum(_sign(a) == _sign(b) for a, b in zip(immediate_values, return_values)) / len(paired)
                if paired else None
            ),
            "pearson": _correlation(immediate_values, return_values),
            "spearman": _correlation(_rank(immediate_values), _rank(return_values)) if paired else None,
            "disagreement_count": len(disagreements),
            "disagreement_by_action_type": _counts(row.get("action_type") for row in disagreements),
            "disagreement_by_phase": _counts(row.get("phase") for row in disagreements),
            "largest_disagreements": sorted(
                disagreements,
                key=lambda row: abs(float(row["a_return"]) - float(row["a_immediate"])),
                reverse=True,
            )[:20],
        },
        "probe_5_weight_concentration": {
            "formula": "clip(1 + 2 * advantage, 0.1, 3.0), then eligible-mean normalize",
            "eligible": len(eligible),
            "positive_zero_negative": _counts(_sign(float(row["advantage"])) for row in eligible),
            "clipped_min_fraction": (
                sum(weight == FROZEN_WEIGHT_MIN for weight in weights_raw) / len(weights_raw)
                if weights_raw else None
            ),
            "clipped_max_fraction": (
                sum(weight == FROZEN_WEIGHT_MAX for weight in weights_raw) / len(weights_raw)
                if weights_raw else None
            ),
            "ess": _ess(weights),
            "terminal_rows": len(terminal),
            "terminal_weight_mass_fraction": (
                sum(terminal_weights) / sum(weights) if weights else None
            ),
            "nonterminal_ess": _ess(nonterminal_weights),
            "eligible_by_action_type": _counts(row.get("chosen_action_type") for row in eligible),
            "eligible_by_phase": _counts(row.get("phase") for row in eligible),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    sidecar = Path(args.sidecar)
    rows = [json.loads(line) for line in sidecar.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = audit(rows, source=str(sidecar))
    write_json(Path(args.out), result)
    print(json.dumps(result, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
