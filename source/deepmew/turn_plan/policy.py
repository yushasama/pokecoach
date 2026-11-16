from __future__ import annotations

from typing import Any

from cg.api import SelectContext, SelectType
from deepmew.policy import choose_action
from deepmew.policy.enums import enum_name
from deepmew.turn_plan.planner import plan_first_action
from deepmew.turn_plan.selection import choose_plan_selection_action


def _is_main_selection(obs: dict[str, Any]) -> bool:
    select = obs.get("select") or {}
    return enum_name(SelectType, select.get("type")) == "MAIN" and enum_name(SelectContext, select.get("context")) == "MAIN"


def choose_v4_action(obs: dict[str, Any]) -> list[int]:
    """Choose an experimental V4 turn-plan action, falling back to default V3 scoring."""

    if not _is_main_selection(obs):
        selected = choose_plan_selection_action(obs)
        if selected is not None:
            return selected
        return choose_action(obs)

    decision = plan_first_action(obs, use_card_semantics=True)
    if decision.fallback:
        return choose_action(obs)
    return list(decision.selected_indices)
