from __future__ import annotations

from typing import Any


def selection_bounds(select: dict[str, Any], option_count: int) -> tuple[int, int]:
    min_count = int(select.get("minCount") or 0)
    max_count = int(select.get("maxCount") or 0)
    min_count = max(0, min(min_count, option_count))
    max_count = max(min_count, min(max_count, option_count))
    return min_count, max_count


def valid_selection(selection: list[int], select: dict[str, Any]) -> bool:
    options = select.get("option") or []
    min_count, max_count = selection_bounds(select, len(options))
    return (
        isinstance(selection, list)
        and min_count <= len(selection) <= max_count
        and len(set(selection)) == len(selection)
        and all(isinstance(index, int) and 0 <= index < len(options) for index in selection)
    )


def guard_selection(selection: list[int], select: dict[str, Any]) -> list[int]:
    if valid_selection(selection, select):
        return selection
    return safe_fallback(select)


def safe_fallback(select: dict[str, Any]) -> list[int]:
    options = select.get("option") or []
    min_count, _ = selection_bounds(select, len(options))
    if min_count == 0:
        return []
    return list(range(min_count))
