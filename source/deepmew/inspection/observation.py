from __future__ import annotations

from collections import Counter
from enum import IntEnum
from typing import Any

from cg.api import AreaType, LogType, OptionType, SelectContext, SelectType, SpecialConditionType


def enum_name(enum_cls: type[IntEnum], value: Any) -> str:
    if value is None:
        return "None"
    try:
        member = enum_cls(value)
    except ValueError:
        return str(value)
    return member.name


def option_summary(option: dict[str, Any]) -> str:
    parts = [f"type={enum_name(OptionType, option.get('type'))}"]
    for key in (
        "number",
        "area",
        "index",
        "playerIndex",
        "toolIndex",
        "energyIndex",
        "count",
        "inPlayArea",
        "inPlayIndex",
        "attackId",
        "cardId",
        "serial",
        "specialConditionType",
    ):
        value = option.get(key)
        if value is None:
            continue
        if key in ("area", "inPlayArea"):
            value = enum_name(AreaType, value)
        elif key == "specialConditionType":
            value = enum_name(SpecialConditionType, value)
        parts.append(f"{key}={value}")
    return ", ".join(parts)


def describe_observation(obs: dict[str, Any], step: int, limit_options: int = 12) -> list[str]:
    lines = [f"step={step}"]
    current = obs.get("current")
    if current is None:
        lines.append("  current=None")
    else:
        players = current.get("players") or []
        lines.append(
            "  current: "
            f"turn={current.get('turn')} "
            f"turnActionCount={current.get('turnActionCount')} "
            f"yourIndex={current.get('yourIndex')} "
            f"firstPlayer={current.get('firstPlayer')} "
            f"result={current.get('result')}"
        )
        for index, player in enumerate(players):
            active = player.get("active") or []
            active_ids = [None if card is None else card.get("id") for card in active]
            lines.append(
                f"  player[{index}]: "
                f"active={active_ids} bench={len(player.get('bench') or [])}/"
                f"{player.get('benchMax')} deck={player.get('deckCount')} "
                f"handCount={player.get('handCount')} prize={len(player.get('prize') or [])} "
                f"discard={len(player.get('discard') or [])}"
            )

    select = obs.get("select")
    if select is None:
        lines.append("  select=None (initial deck request)")
    else:
        options = select.get("option") or []
        type_counts = Counter(enum_name(OptionType, option.get("type")) for option in options)
        lines.append(
            "  select: "
            f"type={enum_name(SelectType, select.get('type'))} "
            f"context={enum_name(SelectContext, select.get('context'))} "
            f"min={select.get('minCount')} max={select.get('maxCount')} "
            f"options={len(options)} "
            f"deckVisible={select.get('deck') is not None} "
            f"remainDamageCounter={select.get('remainDamageCounter')} "
            f"remainEnergyCost={select.get('remainEnergyCost')}"
        )
        lines.append(f"  option_types: {dict(type_counts)}")
        for index, option in enumerate(options[:limit_options]):
            lines.append(f"    option[{index}]: {option_summary(option)}")
        if len(options) > limit_options:
            lines.append(f"    ... {len(options) - limit_options} more option(s)")

    logs = obs.get("logs") or []
    if logs:
        rendered = []
        for log in logs[-5:]:
            rendered.append(
                enum_name(LogType, log.get("type"))
                + "("
                + ", ".join(f"{k}={v}" for k, v in log.items() if k != "type" and v is not None)
                + ")"
            )
        lines.append(f"  recent_logs: {'; '.join(rendered)}")
    else:
        lines.append("  recent_logs: []")

    sbi = obs.get("search_begin_input")
    lines.append(f"  search_begin_input_len={0 if sbi is None else len(sbi)}")
    return lines


def print_observation(obs: dict[str, Any], step: int, limit_options: int = 12) -> None:
    for line in describe_observation(obs, step, limit_options):
        print(line)
