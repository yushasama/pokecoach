"""Base observation encoder: raw game state -> model-ready feature vector.

The base encoder reads only PUBLIC, current-observation fields (the acting
seat's own board plus the opponent's publicly visible board summary and the
legal-option histogram). It never reads opponent hand identities, deck order,
prize contents, future draws, or ground-truth deck/archetype. RL0 uses this
encoder alone; optional `FeatureModule`s append to the declared schema without
touching the `EnvAdapter`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from cg.api import OptionType, SelectContext


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
    cards = pokemon.get("energyCards")
    if isinstance(cards, list):
        return len(cards)
    return _int(energies, 0)


def _hp_ratio(pokemon: dict[str, Any]) -> float:
    max_hp = _int(pokemon.get("maxHp"), 0)
    if max_hp <= 0:
        return 0.0
    if pokemon.get("hp") is not None:
        cur = _int(pokemon.get("hp"), max_hp)
    else:
        cur = max(0, max_hp - _int(pokemon.get("damage"), 0))
    return max(0.0, min(1.0, cur / max_hp))


_OPTION_TYPES: tuple[OptionType, ...] = tuple(OptionType)
_SELECT_CONTEXTS: tuple[SelectContext, ...] = tuple(SelectContext)


def _option_type_name(option: dict[str, Any]) -> str:
    try:
        return OptionType(option.get("type")).name
    except (TypeError, ValueError):
        return "UNKNOWN"


@dataclass(frozen=True)
class ObservationSchema:
    """Ordered, named feature schema. Feature modules extend `names`."""

    names: tuple[str, ...]

    @property
    def dim(self) -> int:
        return len(self.names)

    def extended(self, extra: Sequence[str]) -> "ObservationSchema":
        return ObservationSchema(names=self.names + tuple(extra))


def _base_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    names += [f"ctx_{ctx.name}" for ctx in _SELECT_CONTEXTS]
    names += [f"opt_{opt.name}" for opt in _OPTION_TYPES]
    names += [
        "n_options_norm",
        "min_count_norm",
        "max_count_norm",
        "turn_norm",
        "my_prizes_remaining_norm",
        "opp_prizes_remaining_norm",
        "my_active_hp_ratio",
        "opp_active_hp_ratio",
        "my_active_energy_norm",
        "my_bench_norm",
        "opp_bench_norm",
        "my_hand_norm",
        "my_deck_norm",
        "flag_supporter_played",
        "flag_stadium_played",
        "flag_energy_attached",
        "flag_retreated",
        "is_first_player",
    ]
    return tuple(names)


BASE_SCHEMA = ObservationSchema(names=_base_feature_names())


@dataclass(frozen=True)
class ObservationEncoder:
    """Public, current-observation feature encoder (RL0 base).

    `feature_modules` are optional declared add-ons (disabled in RL0). Enabling
    one extends `schema` only; the `EnvAdapter` is unaffected.
    """

    feature_modules: tuple[Any, ...] = field(default_factory=tuple)

    @property
    def schema(self) -> ObservationSchema:
        schema = BASE_SCHEMA
        for module in self.feature_modules:
            schema = schema.extended(module.spec.output_schema)
        return schema

    def encode_base(self, obs: dict[str, Any]) -> list[float]:
        current = _as_dict(obs.get("current"))
        select = _as_dict(obs.get("select"))
        players = _as_list(current.get("players"))
        your_index = _int(current.get("yourIndex"), 0)
        options = _as_list(select.get("option"))

        ctx_value = select.get("context")
        ctx_vec = [1.0 if ctx.value == ctx_value else 0.0 for ctx in _SELECT_CONTEXTS]

        opt_counts = {opt.name: 0 for opt in _OPTION_TYPES}
        for option in options:
            name = _option_type_name(_as_dict(option))
            if name in opt_counts:
                opt_counts[name] += 1
        denom = max(1, len(options))
        opt_vec = [opt_counts[opt.name] / denom for opt in _OPTION_TYPES]

        min_count = _int(select.get("minCount"), 0)
        max_count = _int(select.get("maxCount"), 0)

        if len(players) >= 2 and your_index in (0, 1):
            me = _as_dict(players[your_index])
            opp = _as_dict(players[1 - your_index])
            my_active = _active0(me)
            opp_active = _active0(opp)
            my_prizes = len(_as_list(me.get("prize")))
            opp_prizes = len(_as_list(opp.get("prize")))
            my_active_hp = _hp_ratio(my_active)
            opp_active_hp = _hp_ratio(opp_active)
            my_energy = _energy_count(my_active)
            my_bench = len(_as_list(me.get("bench")))
            opp_bench = len(_as_list(opp.get("bench")))
            my_hand = _int(me.get("handCount"), len(_as_list(me.get("hand"))))
            my_deck = _int(me.get("deckCount"), 60)
        else:
            my_prizes = opp_prizes = 6
            my_active_hp = opp_active_hp = 0.0
            my_energy = my_bench = opp_bench = 0
            my_hand = 0
            my_deck = 60

        first_player = _int(current.get("firstPlayer"), -1)
        scalars = [
            min(len(options), 30) / 30.0,
            min(min_count, 10) / 10.0,
            min(max_count, 10) / 10.0,
            min(_int(current.get("turn"), 0), 40) / 40.0,
            my_prizes / 6.0,
            opp_prizes / 6.0,
            my_active_hp,
            opp_active_hp,
            min(my_energy, 6) / 6.0,
            min(my_bench, 5) / 5.0,
            min(opp_bench, 5) / 5.0,
            min(my_hand, 12) / 12.0,
            min(my_deck, 60) / 60.0,
            1.0 if current.get("supporterPlayed") else 0.0,
            1.0 if current.get("stadiumPlayed") else 0.0,
            1.0 if current.get("energyAttached") else 0.0,
            1.0 if current.get("retreated") else 0.0,
            1.0 if first_player == your_index else 0.0,
        ]
        return ctx_vec + opt_vec + scalars

    def encode(self, obs: dict[str, Any], context: dict[str, Any] | None = None) -> list[float]:
        vector = self.encode_base(obs)
        for module in self.feature_modules:
            vector.extend(module.compute(obs, context or {}))
        return vector
