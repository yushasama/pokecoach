"""Cached engine card and attack tables."""

from __future__ import annotations

from typing import Any

from cg.api import all_attack, all_card_data


_ATTACKS_BY_ID: dict[int, Any] | None = None
_CARDS_BY_ID: dict[int, Any] | None = None


def _load_card_tables() -> tuple[dict[int, Any], dict[int, Any]]:
    global _ATTACKS_BY_ID, _CARDS_BY_ID

    if _ATTACKS_BY_ID is None:
        _ATTACKS_BY_ID = {attack.attackId: attack for attack in all_attack()}
    if _CARDS_BY_ID is None:
        _CARDS_BY_ID = {card.cardId: card for card in all_card_data()}

    return _CARDS_BY_ID, _ATTACKS_BY_ID
