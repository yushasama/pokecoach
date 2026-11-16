"""Engine-free board-state reader for K4.1 labelers.

Parses the acting seat's view out of a raw ``state_json`` (the §2.16
``raw_observation`` kept by ``deepmew.replay_action_row.v1``). Prizes are read as
*remaining* counts (the prize pile length), so a lower count is closer to winning.
Hidden/None fields degrade to conservative defaults; nothing here imports the
battle engine, so the snapshot is reproducible offline.
"""

from __future__ import annotations

from typing import Any

from deepmew.imitation_fine.schemas import BoardSnapshot


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _active0(player: dict[str, Any]) -> dict[str, Any]:
    active = _as_list(player.get("active"))
    return _as_dict(active[0]) if active else {}


def _energy_count(pokemon: dict[str, Any]) -> int:
    energies = pokemon.get("energies")
    if isinstance(energies, list):
        return len(energies)
    energy_cards = pokemon.get("energyCards")
    if isinstance(energy_cards, list):
        return len(energy_cards)
    return _int(energies, 0)


def _hp(pokemon: dict[str, Any]) -> tuple[int, int]:
    """Return (current_hp, max_hp). ``damage`` is honored when ``hp`` is absent."""
    max_hp = _int(pokemon.get("maxHp"), 0)
    if pokemon.get("hp") is not None:
        return _int(pokemon.get("hp"), max_hp), max_hp
    damage = _int(pokemon.get("damage"), 0)
    return max(0, max_hp - damage), max_hp


def read_board(state_json: Any) -> BoardSnapshot:
    """Read the acting seat's board snapshot from a raw observation dict."""
    state = _as_dict(state_json)
    current = _as_dict(state.get("current"))
    players = _as_list(current.get("players"))
    your_index = _int(current.get("yourIndex"), 0)
    if len(players) < 2 or your_index not in (0, 1):
        return BoardSnapshot(
            turn=_int(current.get("turn"), 0),
            my_prizes_remaining=6, opp_prizes_remaining=6,
            my_active_id=None, my_active_hp=0, my_active_max_hp=0, my_active_energy=0, my_bench_count=0,
            opp_active_id=None, opp_active_hp=0, opp_active_max_hp=0, opp_bench_count=0,
            my_hand_count=0, my_deck_count=60, valid=False,
        )

    me = _as_dict(players[your_index])
    opp = _as_dict(players[1 - your_index])
    my_active = _active0(me)
    opp_active = _active0(opp)
    my_hp, my_max = _hp(my_active)
    opp_hp, opp_max = _hp(opp_active)

    return BoardSnapshot(
        turn=_int(current.get("turn"), 0),
        my_prizes_remaining=len(_as_list(me.get("prize"))),
        opp_prizes_remaining=len(_as_list(opp.get("prize"))),
        my_active_id=_int(my_active.get("id"), 0) or None,
        my_active_hp=my_hp,
        my_active_max_hp=my_max,
        my_active_energy=_energy_count(my_active),
        my_bench_count=len(_as_list(me.get("bench"))),
        opp_active_id=_int(opp_active.get("id"), 0) or None,
        opp_active_hp=opp_hp,
        opp_active_max_hp=opp_max,
        opp_bench_count=len(_as_list(opp.get("bench"))),
        my_hand_count=_int(me.get("handCount"), len(_as_list(me.get("hand")))),
        my_deck_count=_int(me.get("deckCount"), 60),
        valid=True,
    )
