"""Projected option effects for board plans."""

from __future__ import annotations

from typing import Any

from cg.api import AreaType, OptionType

from deepmew.state_machine.state import AttachmentEffect, BoardPlan


def _enum_name(enum_cls: Any, value: Any) -> str:
    if value is None:
        return "None"
    try:
        return enum_cls(value).name
    except (TypeError, ValueError):
        return f"UNKNOWN_{value}"


def attachment_effect(plan: BoardPlan, option: dict[str, Any]) -> AttachmentEffect:
    if option.get("type") != OptionType.ATTACH.value:
        return AttachmentEffect(False, False, False, False, False, None, None)

    area = _enum_name(AreaType, option.get("inPlayArea"))
    index = int(option.get("inPlayIndex") or 0)
    target = next((item for item in plan.readiness if item.area == area and item.index == index), None)
    if target is None or target.best_shortfall is None:
        return AttachmentEffect(False, False, False, area == "ACTIVE", False, None, None)

    before = target.best_shortfall
    after = max(0, before - 1)
    return AttachmentEffect(
        known_target=True,
        improves_readiness=after < before,
        enables_attack=before == 1,
        target_is_active=area == "ACTIVE",
        target_is_closest=area == plan.closest_area and index == plan.closest_index,
        before_shortfall=before,
        after_shortfall=after,
    )
