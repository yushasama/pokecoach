from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

from cg.api import all_attack, all_card_data


BASIC_WATER_ENERGY = 3
ENERGY_SYMBOL_TO_ID = {
    "G": 1,
    "R": 2,
    "W": BASIC_WATER_ENERGY,
    "L": 4,
    "P": 5,
    "F": 6,
    "D": 7,
    "M": 8,
}

SURFING_BEACH = 1262
MEGA_ABOMASNOW = 723
WATER_ENERGY_TYPE = 3

SECOND_ORDER_STATE_SCHEMA = (
    "own_discard_basic_water_norm",
    "own_discard_total_norm",
    "own_deck_count_norm",
    "own_deck_low12_flag",
    "stadium_present_flag",
    "surfing_beach_in_play_flag",
    "surfing_beach_available_flag",
    "surfing_beach_beneficial_proxy_flag",
    "own_active_water_flag",
    "own_bench_water_count_norm",
    "mega_abomasnow_in_play_flag",
    "mega_abomasnow_active_flag",
    "own_prizes_remaining_norm",
    "opp_prizes_remaining_norm",
)

SECOND_ORDER_OPTION_SCHEMA = (
    "attack_discard_pile_scaling_flag",
    "attack_self_mill_scaling_flag",
    "attack_hidden_dependent_flag",
    "attack_projected_resource_count_norm",
    "attack_projected_base_damage_norm",
    "attack_projected_weakness_damage_norm",
    "attack_zero_damage_whiff_flag",
    "attack_projected_self_mill_count_norm",
    "attack_projected_deck_after_mill_norm",
    "attack_low_deck_self_mill_flag",
)


@dataclass(frozen=True)
class AttackProjectionRule:
    attack_id: int
    attack_name: str
    kind: str
    resource_card_id: int
    damage_per_resource: int
    mill_count: int = 0
    hidden_dependent: bool = False


@dataclass(frozen=True)
class AttackProjection:
    attack_id: int
    attack_name: str
    kind: str
    resource_count: int
    projected_base_damage: int
    projected_weakness_damage: int
    projected_self_mill_count: int
    projected_deck_after_mill: int
    hidden_dependent: bool
    zero_damage_whiff: bool
    low_deck_self_mill: bool


@dataclass(frozen=True)
class SecondOrderState:
    own_discard_basic_water_count: int
    own_discard_total_count: int
    own_deck_count: int
    own_deck_low12: bool
    stadium_card_id: int | None
    surfing_beach_in_play: bool
    surfing_beach_available: bool
    surfing_beach_beneficial_proxy: bool
    own_active_water: bool
    own_bench_water_count: int
    mega_abomasnow_in_play: bool
    mega_abomasnow_active: bool
    own_prizes_remaining: int
    opp_prizes_remaining: int

    def vector(self) -> list[float]:
        return [
            min(self.own_discard_basic_water_count, 16) / 16.0,
            min(self.own_discard_total_count, 40) / 40.0,
            min(self.own_deck_count, 60) / 60.0,
            1.0 if self.own_deck_low12 else 0.0,
            1.0 if self.stadium_card_id is not None else 0.0,
            1.0 if self.surfing_beach_in_play else 0.0,
            1.0 if self.surfing_beach_available else 0.0,
            1.0 if self.surfing_beach_beneficial_proxy else 0.0,
            1.0 if self.own_active_water else 0.0,
            min(self.own_bench_water_count, 5) / 5.0,
            1.0 if self.mega_abomasnow_in_play else 0.0,
            1.0 if self.mega_abomasnow_active else 0.0,
            min(self.own_prizes_remaining, 6) / 6.0,
            min(self.opp_prizes_remaining, 6) / 6.0,
        ]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _card_id(card: Any) -> int | None:
    if isinstance(card, dict):
        value = card.get("id")
    else:
        value = getattr(card, "id", None)
    if value is None:
        return None

    return _as_int(value)


def _option_attack_id(option: dict[str, Any]) -> int | None:
    value = option.get("attack_id", option.get("attackId"))
    if value is None:
        return None

    return _as_int(value)


def _player(current: dict[str, Any], seat: int) -> dict[str, Any]:
    players = _as_list(current.get("players"))
    if seat < 0 or seat >= len(players):
        return {}

    return _as_dict(players[seat])


def _active(player: dict[str, Any]) -> dict[str, Any]:
    active = _as_list(player.get("active"))
    if active and isinstance(active[0], dict):
        return active[0]

    return {}


def _bench(player: dict[str, Any]) -> list[dict[str, Any]]:
    return [_as_dict(card) for card in _as_list(player.get("bench")) if isinstance(card, dict)]


@lru_cache(maxsize=1)
def _water_pokemon_ids() -> frozenset[int]:
    return frozenset(
        int(card.cardId)
        for card in all_card_data()
        if int(card.cardType) == 0 and int(card.energyType) == WATER_ENERGY_TYPE
    )


@lru_cache(maxsize=1)
def _weakness_by_card_id() -> dict[int, int | None]:
    return {int(card.cardId): card.weakness for card in all_card_data()}


def _is_water_pokemon(pokemon: dict[str, Any]) -> bool:
    energies = _as_list(pokemon.get("energies"))
    if WATER_ENERGY_TYPE in [_as_int(value, -1) for value in energies]:
        return True
    energy_type = pokemon.get("energyType")
    if energy_type is not None and _as_int(energy_type, -1) == WATER_ENERGY_TYPE:
        return True
    card_id = _card_id(pokemon)

    return card_id in _water_pokemon_ids()


def _prizes_remaining(player: dict[str, Any]) -> int:
    return len(_as_list(player.get("prize")))


def _stadium_card_id(current: dict[str, Any]) -> int | None:
    stadium = current.get("stadium")
    if isinstance(stadium, dict):
        return _card_id(stadium)
    for card in _as_list(stadium):
        card_id = _card_id(card)
        if card_id is not None:
            return card_id

    return None


def _surfing_beach_option_index(select: dict[str, Any]) -> int | None:
    for index, option in enumerate(_as_list(select.get("option"))):
        option_dict = _as_dict(option)
        option_type = option_dict.get("type", option_dict.get("option_type"))
        area = option_dict.get("area", option_dict.get("source_area"))
        if str(option_type) in {"10", "ABILITY"} and str(area) in {"7", "STADIUM"}:
            return index

    return None


def read_second_order_state(obs: dict[str, Any]) -> SecondOrderState:
    current = _as_dict(obs.get("current"))
    select = _as_dict(obs.get("select"))
    seat = _as_int(current.get("yourIndex"))
    own = _player(current, seat)
    opp = _player(current, 1 - seat if seat in (0, 1) else 1)
    discard = _as_list(own.get("discard"))
    own_discard_basic_water = sum(1 for card in discard if _card_id(card) == BASIC_WATER_ENERGY)
    own_deck_count = _as_int(own.get("deckCount"))
    own_active = _active(own)
    own_bench = _bench(own)
    own_active_water = _is_water_pokemon(own_active)
    own_bench_water_count = sum(1 for pokemon in own_bench if _is_water_pokemon(pokemon))
    stadium_id = _stadium_card_id(current)
    surfing_in_play = stadium_id == SURFING_BEACH
    surfing_available = surfing_in_play and _surfing_beach_option_index(select) is not None
    mega_in_play = _card_id(own_active) == MEGA_ABOMASNOW or any(_card_id(card) == MEGA_ABOMASNOW for card in own_bench)

    return SecondOrderState(
        own_discard_basic_water_count=own_discard_basic_water,
        own_discard_total_count=len(discard),
        own_deck_count=own_deck_count,
        own_deck_low12=own_deck_count <= 12,
        stadium_card_id=stadium_id,
        surfing_beach_in_play=surfing_in_play,
        surfing_beach_available=surfing_available,
        surfing_beach_beneficial_proxy=surfing_available and own_active_water and own_bench_water_count > 0,
        own_active_water=own_active_water,
        own_bench_water_count=own_bench_water_count,
        mega_abomasnow_in_play=mega_in_play,
        mega_abomasnow_active=_card_id(own_active) == MEGA_ABOMASNOW,
        own_prizes_remaining=_prizes_remaining(own),
        opp_prizes_remaining=_prizes_remaining(opp),
    )


def _weakness_factor(obs: dict[str, Any], resource_card_id: int) -> int:
    current = _as_dict(obs.get("current"))
    seat = _as_int(current.get("yourIndex"))
    opp = _player(current, 1 - seat if seat in (0, 1) else 1)
    defender = _active(opp)
    weakness = defender.get("weakness")
    if weakness is None:
        card_id = _card_id(defender)
        weakness = _weakness_by_card_id().get(card_id) if card_id is not None else None
    if weakness is None:
        return 1
    if resource_card_id == BASIC_WATER_ENERGY and _as_int(weakness, -1) == WATER_ENERGY_TYPE:
        return 2

    return 1


@lru_cache(maxsize=1)
def build_attack_projection_rules() -> dict[int, AttackProjectionRule]:
    rules: dict[int, AttackProjectionRule] = {}
    discard_pattern = re.compile(
        r"does\s+(\d+)\s+damage\s+for\s+each\s+Basic\s+\{([A-Z])\}\s+Energy\s+card\s+in\s+your\s+discard\s+pile",
        re.IGNORECASE,
    )
    self_mill_pattern = re.compile(
        r"Discard\s+the\s+top\s+(\d+)\s+cards\s+of\s+your\s+deck.*does\s+(\d+)\s+damage\s+for\s+each\s+Basic\s+\{([A-Z])\}\s+Energy\s+card",
        re.IGNORECASE,
    )
    for attack in all_attack():
        text = str(attack.text or "")
        discard_match = discard_pattern.search(text)
        if discard_match:
            damage, symbol = discard_match.groups()
            resource_card_id = ENERGY_SYMBOL_TO_ID.get(symbol.upper())
            if resource_card_id is not None:
                rules[int(attack.attackId)] = AttackProjectionRule(
                    attack_id=int(attack.attackId),
                    attack_name=str(attack.name),
                    kind="discard_pile_scaling",
                    resource_card_id=resource_card_id,
                    damage_per_resource=int(damage),
                    hidden_dependent=False,
                )
            continue
        self_mill_match = self_mill_pattern.search(text)
        if self_mill_match:
            mill_count, damage, symbol = self_mill_match.groups()
            resource_card_id = ENERGY_SYMBOL_TO_ID.get(symbol.upper())
            if resource_card_id is not None:
                rules[int(attack.attackId)] = AttackProjectionRule(
                    attack_id=int(attack.attackId),
                    attack_name=str(attack.name),
                    kind="self_mill_top_deck_scaling",
                    resource_card_id=resource_card_id,
                    damage_per_resource=int(damage),
                    mill_count=int(mill_count),
                    hidden_dependent=True,
                )

    return rules


def _count_discard_resource(obs: dict[str, Any], resource_card_id: int) -> int:
    current = _as_dict(obs.get("current"))
    seat = _as_int(current.get("yourIndex"))
    own = _player(current, seat)

    return sum(1 for card in _as_list(own.get("discard")) if _card_id(card) == resource_card_id)


def project_attack(obs: dict[str, Any], attack_id: int, rules: dict[int, AttackProjectionRule] | None = None) -> AttackProjection | None:
    rule = (rules or build_attack_projection_rules()).get(attack_id)
    if rule is None:
        return None
    state = read_second_order_state(obs)
    weakness = _weakness_factor(obs, rule.resource_card_id)
    if rule.kind == "discard_pile_scaling":
        resource_count = _count_discard_resource(obs, rule.resource_card_id)
        base_damage = resource_count * rule.damage_per_resource

        return AttackProjection(
            attack_id=rule.attack_id,
            attack_name=rule.attack_name,
            kind=rule.kind,
            resource_count=resource_count,
            projected_base_damage=base_damage,
            projected_weakness_damage=base_damage * weakness,
            projected_self_mill_count=0,
            projected_deck_after_mill=state.own_deck_count,
            hidden_dependent=rule.hidden_dependent,
            zero_damage_whiff=base_damage == 0,
            low_deck_self_mill=False,
        )
    if rule.kind == "self_mill_top_deck_scaling":
        mill_count = min(rule.mill_count, state.own_deck_count)

        return AttackProjection(
            attack_id=rule.attack_id,
            attack_name=rule.attack_name,
            kind=rule.kind,
            resource_count=0,
            projected_base_damage=0,
            projected_weakness_damage=0,
            projected_self_mill_count=mill_count,
            projected_deck_after_mill=max(0, state.own_deck_count - mill_count),
            hidden_dependent=rule.hidden_dependent,
            zero_damage_whiff=False,
            low_deck_self_mill=state.own_deck_count <= 12,
        )

    return None


def option_second_order_vector(obs: dict[str, Any], option: dict[str, Any]) -> list[float]:
    attack_id = _option_attack_id(option)
    projection = project_attack(obs, attack_id) if attack_id is not None else None
    if projection is None:
        return [0.0] * len(SECOND_ORDER_OPTION_SCHEMA)

    return [
        1.0 if projection.kind == "discard_pile_scaling" else 0.0,
        1.0 if projection.kind == "self_mill_top_deck_scaling" else 0.0,
        1.0 if projection.hidden_dependent else 0.0,
        min(projection.resource_count, 16) / 16.0,
        min(projection.projected_base_damage, 400) / 400.0,
        min(projection.projected_weakness_damage, 800) / 800.0,
        1.0 if projection.zero_damage_whiff else 0.0,
        min(projection.projected_self_mill_count, 6) / 6.0,
        min(projection.projected_deck_after_mill, 60) / 60.0,
        1.0 if projection.low_deck_self_mill else 0.0,
    ]


def augment_option_features(
    obs: dict[str, Any],
    legal_set: Iterable[dict[str, Any]],
    base_features: list[list[float]],
) -> list[list[float]]:
    state_vector = read_second_order_state(obs).vector()
    augmented: list[list[float]] = []
    for features, option in zip(base_features, legal_set):
        option_vector = option_second_order_vector(obs, option)
        augmented.append([float(value) for value in features] + state_vector + option_vector)

    return augmented
