"""Visible board metrics for Card Function Discovery v0.

The reader is engine-free: it inspects raw replay observations already captured
in `deepmew.replay_action_row.v1`. Unknown or hidden fields degrade to counts
and conservative booleans; no missing state is fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from deepmew.deck_profiles.current_water_mega_abomasnow import (
    KYOGRE_CARD_ID,
    MEGA_ABOMASNOW_CARD_ID,
    MEGA_LINE_IDS,
    SNOVER_CARD_ID,
)
from deepmew.options.identity import card_name_for_id, resolve_option_identity

KNOWN_ATTACKER_IDS = frozenset({KYOGRE_CARD_ID, SNOVER_CARD_ID, MEGA_ABOMASNOW_CARD_ID})


@dataclass(frozen=True)
class BoardMetrics:
    valid: bool
    player_index: int | None
    turn: int | None
    active_id: int | None
    active_name: str
    active_serial: int | None
    active_hp: int
    active_energy_count: int
    active_attack_available: bool | None
    bench_count: int
    basic_count_in_play: int
    evolution_line_count: int
    main_attacker_ready: bool
    backup_attacker_ready: bool
    backup_attacker_count: int
    active_is_only_threat: bool
    hand_count: int
    deck_count: int
    discard_count: int
    prize_count: int
    supporter_played: bool
    energy_attached: bool
    stadium_played: bool
    legal_action_count: int | None
    opponent_active_id: int | None = None
    opponent_active_hp: int = 0
    in_play_ids: frozenset[int] = field(default_factory=frozenset)
    in_play_serials: frozenset[int] = field(default_factory=frozenset)
    discard_ids: frozenset[int] = field(default_factory=frozenset)
    discard_serials: frozenset[int] = field(default_factory=frozenset)
    visible_card_ids: frozenset[int] = field(default_factory=frozenset)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _card_id(value: Any) -> int | None:
    if not isinstance(value, dict):
        return None
    return _int_or_none(value.get("id", value.get("cardId", value.get("cardID"))))


def _serial(value: Any) -> int | None:
    if not isinstance(value, dict):
        return None
    return _int_or_none(value.get("serial"))


def _card_name(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    name = value.get("name") or card_name_for_id(_card_id(value))
    return str(name) if name else ""


def _energy_count(pokemon: dict[str, Any]) -> int:
    energy_cards = pokemon.get("energyCards")
    if isinstance(energy_cards, list):
        return len(energy_cards)
    energies = pokemon.get("energies")
    if isinstance(energies, list):
        return len(energies)
    return _int(energies, 0)


def _zone_cards(player: dict[str, Any], zone: str) -> list[dict[str, Any]]:
    return [item for item in _as_list(player.get(zone)) if isinstance(item, dict)]


def _active(player: dict[str, Any]) -> dict[str, Any]:
    active = _zone_cards(player, "active")
    return active[0] if active else {}


def _pokemon_in_play(player: dict[str, Any]) -> list[dict[str, Any]]:
    active = _zone_cards(player, "active")
    return active[:1] + _zone_cards(player, "bench")


def _is_basic(pokemon: dict[str, Any]) -> bool:
    if not pokemon:
        return False
    pre_evolution = pokemon.get("preEvolution")
    return not isinstance(pre_evolution, list) or len(pre_evolution) == 0


def _is_evolved(pokemon: dict[str, Any]) -> bool:
    pre_evolution = pokemon.get("preEvolution")
    return isinstance(pre_evolution, list) and len(pre_evolution) > 0


def _ready_backup(pokemon: dict[str, Any]) -> bool:
    card_id = _card_id(pokemon)
    energy = _energy_count(pokemon)
    max_hp = _int(pokemon.get("maxHp"), _int(pokemon.get("hp"), 0))
    if card_id in KNOWN_ATTACKER_IDS:
        return energy >= 1
    return energy >= 2 or (max_hp >= 130 and energy >= 1)


def _all_card_ids(player: dict[str, Any]) -> set[int]:
    ids: set[int] = set()
    for zone in ("active", "bench", "hand", "deck", "discard", "prize"):
        for card in _zone_cards(player, zone):
            card_id = _card_id(card)
            if card_id is not None:
                ids.add(card_id)
    return ids


def _ids(cards: list[dict[str, Any]]) -> frozenset[int]:
    return frozenset(card_id for card_id in (_card_id(card) for card in cards) if card_id is not None)


def _serials(cards: list[dict[str, Any]]) -> frozenset[int]:
    return frozenset(serial for serial in (_serial(card) for card in cards) if serial is not None)


def _players(state_json: Any) -> tuple[dict[str, Any], list[Any]]:
    state = _as_dict(state_json)
    current = _as_dict(state.get("current"))
    players = _as_list(current.get("players"))
    return current, players


def _legal_options_for_actor(state_json: Any, player_index: int | None) -> list[Any] | None:
    if player_index not in (0, 1):
        return None
    state = _as_dict(state_json)
    current = _as_dict(state.get("current"))
    if _int_or_none(current.get("yourIndex")) != player_index:
        return None
    select = _as_dict(state.get("select"))
    options = select.get("option")
    return options if isinstance(options, list) else None


def attack_available(state_json: Any, player_index: int | None) -> bool | None:
    options = _legal_options_for_actor(state_json, player_index)
    if options is None:
        return None
    for option in options:
        identity = resolve_option_identity(_as_dict(state_json), _as_dict(option))
        if identity.option_type == "ATTACK":
            return True
    return False


def legal_action_count(state_json: Any, player_index: int | None) -> int | None:
    options = _legal_options_for_actor(state_json, player_index)
    return None if options is None else len(options)


def read_board_metrics(state_json: Any, player_index: int | None) -> BoardMetrics:
    current, players = _players(state_json)
    if player_index not in (0, 1) or len(players) <= player_index:
        return BoardMetrics(
            valid=False,
            player_index=player_index,
            turn=_int_or_none(current.get("turn")),
            active_id=None,
            active_name="",
            active_serial=None,
            active_hp=0,
            active_energy_count=0,
            active_attack_available=None,
            bench_count=0,
            basic_count_in_play=0,
            evolution_line_count=0,
            main_attacker_ready=False,
            backup_attacker_ready=False,
            backup_attacker_count=0,
            active_is_only_threat=False,
            hand_count=0,
            deck_count=0,
            discard_count=0,
            prize_count=0,
            supporter_played=False,
            energy_attached=False,
            stadium_played=False,
            legal_action_count=None,
        )

    me = _as_dict(players[player_index])
    opp = _as_dict(players[1 - player_index]) if len(players) > 1 else {}
    active = _active(me)
    bench = _zone_cards(me, "bench")
    in_play = _pokemon_in_play(me)
    discard = _zone_cards(me, "discard")
    atk_available = attack_available(state_json, player_index)
    active_energy = _energy_count(active)
    active_id = _card_id(active)
    backup_count = sum(1 for pokemon in bench if _ready_backup(pokemon))
    active_threat = bool(active_id is not None and (active_energy > 0 or atk_available is True))
    visible_ids = _all_card_ids(me)
    visible_ids.update(_all_card_ids(opp))

    opp_active = _active(opp)
    return BoardMetrics(
        valid=True,
        player_index=player_index,
        turn=_int_or_none(current.get("turn")),
        active_id=active_id,
        active_name=_card_name(active),
        active_serial=_serial(active),
        active_hp=_int(active.get("hp"), _int(active.get("maxHp"), 0)),
        active_energy_count=active_energy,
        active_attack_available=atk_available,
        bench_count=len(bench),
        basic_count_in_play=sum(1 for pokemon in in_play if _is_basic(pokemon)),
        evolution_line_count=sum(1 for pokemon in in_play if _is_evolved(pokemon)),
        main_attacker_ready=bool(atk_available is True and active_id is not None),
        backup_attacker_ready=backup_count > 0,
        backup_attacker_count=backup_count,
        active_is_only_threat=active_threat and backup_count == 0,
        hand_count=_int(me.get("handCount"), len(_zone_cards(me, "hand"))),
        deck_count=_int(me.get("deckCount"), len(_zone_cards(me, "deck"))),
        discard_count=len(discard),
        prize_count=len(_as_list(me.get("prize"))),
        supporter_played=bool(current.get("supporterPlayed")),
        energy_attached=bool(current.get("energyAttached")),
        stadium_played=bool(current.get("stadiumPlayed")),
        legal_action_count=legal_action_count(state_json, player_index),
        opponent_active_id=_card_id(opp_active),
        opponent_active_hp=_int(opp_active.get("hp"), _int(opp_active.get("maxHp"), 0)),
        in_play_ids=_ids(in_play),
        in_play_serials=_serials(in_play),
        discard_ids=_ids(discard),
        discard_serials=_serials(discard),
        visible_card_ids=frozenset(visible_ids),
    )


def deck_context_from_state(state_json: Any, player_index: int | None, deck_hash: str | None = None) -> str:
    """Best-effort deck context label from visible ids and known current-deck line."""
    metrics = read_board_metrics(state_json, player_index)
    if metrics.visible_card_ids & (MEGA_LINE_IDS | {KYOGRE_CARD_ID}):
        return "current_water_mega_abomasnow"
    if deck_hash:
        return f"deck_hash:{deck_hash}"
    return "unknown"
