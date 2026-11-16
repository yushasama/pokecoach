from __future__ import annotations

from typing import Any


def get_field(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def card_id(card: Any) -> int | None:
    if card is None:
        return None
    value = get_field(card, "id", get_field(card, "cardId"))
    return int(value) if value is not None else None
