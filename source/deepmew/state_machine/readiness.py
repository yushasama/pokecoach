"""Visible Pokemon readiness extraction."""

from __future__ import annotations

from typing import Any

from deepmew.state_machine.state import PokemonReadiness
from deepmew.state_machine.tables import _load_card_tables


def _energy_count(pokemon: dict[str, Any]) -> int:
    energies = pokemon.get("energies")
    if isinstance(energies, list):
        return len(energies)

    energy_cards = pokemon.get("energyCards")
    if isinstance(energy_cards, list):
        return len(energy_cards)

    return 0


def _readiness_for_pokemon(area: str, index: int, pokemon: dict[str, Any]) -> PokemonReadiness:
    cards_by_id, attacks_by_id = _load_card_tables()
    card_id = int(pokemon.get("id") or 0)
    card_data = cards_by_id.get(card_id)
    attached = _energy_count(pokemon)
    best_shortfall: int | None = None
    best_damage: int | None = None

    for attack_id in getattr(card_data, "attacks", []) or []:
        attack = attacks_by_id.get(int(attack_id))
        if attack is None:
            continue
        shortfall = max(0, len(attack.energies or []) - attached)
        if best_shortfall is None or shortfall < best_shortfall:
            best_shortfall = shortfall
            best_damage = attack.damage
        elif shortfall == best_shortfall and best_damage is not None:
            best_damage = max(best_damage, attack.damage)

    return PokemonReadiness(
        area=area,
        index=index,
        card_id=card_id,
        attached_energy=attached,
        best_shortfall=best_shortfall,
        best_damage=best_damage,
        can_attack_by_energy=best_shortfall == 0,
    )


def _own_player(current: dict[str, Any]) -> dict[str, Any] | None:
    your_index = int(current.get("yourIndex") or 0)
    players = current.get("players") or []
    if your_index >= len(players):
        return None
    return players[your_index] or {}


def _own_pokemon(current: dict[str, Any]) -> list[PokemonReadiness]:
    player = _own_player(current)
    if player is None:
        return []
    readiness: list[PokemonReadiness] = []

    active = player.get("active") or []
    if active and active[0] is not None:
        readiness.append(_readiness_for_pokemon("ACTIVE", 0, active[0]))

    for index, pokemon in enumerate(player.get("bench") or []):
        if pokemon is not None:
            readiness.append(_readiness_for_pokemon("BENCH", index, pokemon))

    return readiness
