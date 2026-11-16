from __future__ import annotations

from typing import Any

from cg.api import OptionType
from deepmew.metrics.model import PolicyMetrics


USEFUL = {"ATTACK", "PLAY", "ATTACH", "EVOLVE", "SKILL", "ABILITY"}
RISKY = {"DISCARD", "RETREAT"}


def _option_type_name(option: dict[str, Any]) -> str:
    value = option.get("type")
    try:
        return OptionType(value).name
    except (TypeError, ValueError):
        return f"UNKNOWN_{value}"


def analyze_decision(obs: dict[str, Any], selection: list[int]) -> dict[str, Any]:
    select = obs.get("select")
    if select is None:
        return {
            "chosen_types": [],
            "available_types": [],
            "end_with_useful": False,
            "attack_skipped": False,
            "unknown_action": False,
            "risky_over_safer": False,
        }

    options = select.get("option") or []
    available_types = [_option_type_name(option) for option in options]
    chosen_types = [available_types[index] for index in selection if 0 <= index < len(available_types)]
    useful_available = any(option_type in USEFUL for option_type in available_types)
    safer_useful_available = any(option_type in USEFUL for option_type in available_types)

    return {
        "chosen_types": chosen_types,
        "available_types": available_types,
        "end_with_useful": "END" in chosen_types and useful_available,
        "attack_skipped": "ATTACK" in available_types and "ATTACK" not in chosen_types,
        "unknown_action": any(option_type.startswith("UNKNOWN_") for option_type in chosen_types),
        "risky_over_safer": any(option_type in RISKY for option_type in chosen_types) and safer_useful_available,
    }


def update_metrics(metrics: PolicyMetrics, obs: dict[str, Any], selection: list[int]) -> dict[str, Any]:
    analysis = analyze_decision(obs, selection)
    if obs.get("select") is not None:
        metrics.decisions += 1
    if analysis["end_with_useful"]:
        metrics.end_with_useful_options += 1
    if analysis["attack_skipped"]:
        metrics.attack_available_but_not_picked += 1
    if analysis["unknown_action"]:
        metrics.unknown_action += 1
    if analysis["risky_over_safer"]:
        metrics.risky_over_safer += 1
    return analysis
