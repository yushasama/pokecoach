from __future__ import annotations

from typing import Any

from cg.api import OptionType


USEFUL_ACTIONS = {
    "ABILITY",
    "ATTACH",
    "EVOLVE",
    "PLAY",
    "SKILL",
    "ATTACK",
}

RISKY_ACTIONS = {
    "DISCARD",
    "RETREAT",
    "SWITCH",
    "TO_DECK",
    "TO_DECK_BOTTOM",
}

KNOWN_OPTION_TYPES = {member.value for member in OptionType}


def enum_name(enum_cls: Any, value: Any) -> str:
    if value is None:
        return "None"
    try:
        return enum_cls(value).name
    except (TypeError, ValueError):
        return f"UNKNOWN_{value}"


def option_type(option: dict[str, Any]) -> str:
    return enum_name(OptionType, option.get("type"))
