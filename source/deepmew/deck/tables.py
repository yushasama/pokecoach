from __future__ import annotations

from typing import Any

from cg.api import CardData, CardType, EnergyType, all_attack, all_card_data


TRAINER_TYPES = {CardType.ITEM, CardType.TOOL, CardType.SUPPORTER, CardType.STADIUM}
ENERGY_TYPES = {CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY}


def card_table() -> dict[int, CardData]:
    return {card.cardId: card for card in all_card_data()}


def attack_table() -> dict[int, Any]:
    return {attack.attackId: attack for attack in all_attack()}


def type_name(card: CardData) -> str:
    try:
        return CardType(card.cardType).name
    except (TypeError, ValueError):
        return f"UNKNOWN_{card.cardType}"


def energy_name(value: Any) -> str:
    try:
        return EnergyType(value).name
    except (TypeError, ValueError):
        return f"UNKNOWN_{value}"
