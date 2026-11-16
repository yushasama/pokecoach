from __future__ import annotations

from typing import Any

from deepmew.card_semantics.accessors import card_id, get_field


def own_player(obs: dict[str, Any]) -> Any | None:
    current = obs.get("current") or {}
    players = get_field(current, "players", []) or []
    your_index = int(get_field(current, "yourIndex", 0) or 0)
    if your_index >= len(players):
        return None
    return players[your_index]


def visible_own_cards(player: Any) -> list[Any]:
    cards: list[Any] = []
    for zone_name in ("hand", "discard", "prize"):
        zone = get_field(player, zone_name) or []
        cards.extend(card for card in zone if card is not None)
    for area_name in ("active", "bench"):
        for pokemon in get_field(player, area_name) or []:
            if pokemon is None:
                continue
            cards.append(pokemon)
            cards.extend(card for card in (get_field(pokemon, "energyCards") or []) if card is not None)
            cards.extend(card for card in (get_field(pokemon, "tools") or []) if card is not None)
            cards.extend(card for card in (get_field(pokemon, "preEvolution") or []) if card is not None)
    return cards


def visible_card_ids(player: Any) -> list[int | None]:
    return [card_id(card) for card in visible_own_cards(player)]
