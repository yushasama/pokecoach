"""Futile-attack guard: stop repeating attacks that provably or observably deal 0 damage.

Motivating loss (ladder episode 84541591, 2026-07-06): our Mega Abomasnow ex
attacked an opposing Crustle (card 345, [Ability] Mysterious Rock Inn: "Prevent
all damage done to this Pokemon by attacks from your opponent's Pokemon {ex}")
for 0 damage on ~13 consecutive turns, repeatedly choosing Hammer-lanche (which
discards the top 6 cards of OUR deck) and milling ourselves toward deck-out,
while a non-ex Kyogre sat on our bench with RETREAT legal the whole time.

This module is deliberately self-contained (stdlib only, no deepmew/cg imports)
so it can be copied verbatim into a submission bundle as `futility_guard.py`.
All engine enum values are inlined as ints with safe fallbacks (INVARIANT #6:
tolerate unknown enum members; never crash - INVARIANT #1).

The guard post-processes an already-chosen selection. On any internal error it
returns the original choice unchanged. It only ever swaps to another option
index that exists in the same legal-options list, so downstream validity
guards (`valid_selection`) still apply.

Rules (MAIN context, single-index choices only):
1. If the chosen option is an ATTACK classified futile (expected 0 damage):
   a. prefer a non-futile ATTACK (non-self-mill first),
   b. else RETREAT if a non-ex benched Pokemon exists (and remember to prefer
      promoting a non-ex, max-energy Pokemon at the follow-up switch select),
   c. else if the futile attack self-mills, prefer a non-self-mill futile
      attack, then END.
2. Futility classification:
   - static wall: defender's card id has an "immune to damage from Pokemon ex"
     ability and our attacker is a Pokemon ex (no expiry; ability is permanent),
   - static Riptide: attack 1042 with zero visible Basic Water Energy in our
     own discard pile scales to exactly 0,
   - dynamic: we already used this attack id against this same defender serial
     and observed an HP_CHANGE of exactly 0 (expires after `expire_turns` of
     our turns so temporary protection effects get re-probed).
"""

from __future__ import annotations

from typing import Any

# --- engine constants (inlined; see cg/api.py) -----------------------------
_CTX_MAIN = 0
_CTX_SWITCH = 3
_CTX_TO_ACTIVE = 4
_OPT_ATTACH = 8
_OPT_RETREAT = 12
_OPT_ATTACK = 13
_OPT_END = 14
_LOG_TURN_START = 2
_LOG_ATTACK = 15
_LOG_HP_CHANGE = 16
_AREA_ACTIVE = 4
_AREA_BENCH = 5

# --- static card knowledge (regenerable; see scripts/build_futility_tables.py
# and tests/test_futility_guard.py which cross-check EN_Card_Data.csv) -------

# Pokemon whose printed Ability prevents ALL damage from the opponent's
# Pokemon ex: Sylveon 330 (Safeguard), Crustle 345 (Mysterious Rock Inn).
# (Farigiraf ex 83 blocks only Basic Pokemon ex and is intentionally excluded:
# we do not track our attacker's stage here.)
EX_WALL_CARD_IDS = frozenset({330, 345})

# Every card id whose Rule column marks it as a Pokemon ex / Mega Pokemon ex.
EX_CARD_IDS = frozenset({
    24, 29, 30, 37, 40, 44, 46, 52, 63, 75, 79, 80, 83, 84, 96, 99,
    107, 108, 117, 121, 125, 130, 138, 139, 140, 141, 150, 153, 154, 161, 176, 179,
    184, 189, 190, 193, 198, 205, 207, 210, 223, 229, 231, 232, 236, 239, 241, 243,
    244, 246, 248, 249, 259, 269, 272, 283, 293, 299, 302, 306, 313, 316, 320, 326,
    328, 329, 331, 336, 337, 340, 357, 369, 372, 381, 389, 404, 407, 424, 431, 447,
    455, 458, 471, 481, 509, 515, 525, 527, 547, 561, 573, 583, 598, 618, 631, 641,
    648, 652, 662, 678, 687, 695, 723, 737, 747, 754, 756, 766, 772, 781, 790, 795,
    806, 813, 828, 835, 849, 861, 868, 886, 896, 904, 911, 919, 928, 932, 939, 944,
    951, 954, 957, 962, 968, 969, 975, 979, 984, 988, 990, 993, 997, 1002, 1006, 1022,
    1026, 1031, 1040, 1056, 1062, 1064, 1071,
})

# Attacks that discard cards from OUR OWN deck as a cost/effect (self-mill):
# 1046 Hammer-lanche (Mega Abomasnow ex): discard top 6 of own deck.
SELF_MILL_ATTACK_IDS = frozenset({1046})

# Riptide (Kyogre 721, attack 1042): 20x per Basic Water Energy in own discard.
RIPTIDE_ATTACK_ID = 1042
BASIC_WATER_ENERGY_ID = 3


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _card_id(value: Any) -> int | None:
    raw = _as_dict(value).get("id")
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _card_serial(value: Any) -> int | None:
    raw = _as_dict(value).get("serial")
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _energy_count(pokemon: dict[str, Any]) -> int:
    energies = pokemon.get("energies")
    if isinstance(energies, list):
        return len(energies)
    cards = pokemon.get("energyCards")
    if isinstance(cards, list):
        return len(cards)
    return _as_int(energies, 0)


class FutilityGuard:
    """Per-game stateful guard. One instance per process is fine: state resets
    automatically when the observed turn counter goes backwards (new game)."""

    def __init__(self, expire_turns: int = 4) -> None:
        self.expire_turns = int(expire_turns)
        self._reset()

    def _reset(self) -> None:
        # (attack_id, defender_serial) -> turn the 0-damage hit was observed on
        self._futile_observed: dict[tuple[int, int], int] = {}
        self._last_turn = -1
        self._pending_promote = False

    # ------------------------------------------------------------------ API
    def process(self, obs: Any, select: Any, chosen: list[int]) -> list[int]:
        """Observe the latest logs and, if the chosen option is a futile attack,
        swap to a better legal option. Never raises; falls back to `chosen`."""
        try:
            obs = _as_dict(obs)
            select = _as_dict(select)
            self._observe(obs)
            return self._adjust(obs, select, list(chosen))
        except Exception:
            return chosen

    # ------------------------------------------------------------- observing
    def _players(self, obs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], int] | None:
        current = _as_dict(obs.get("current"))
        players = _as_list(current.get("players"))
        my_index = _as_int(current.get("yourIndex"), -1)
        if my_index not in (0, 1) or len(players) < 2:
            return None
        return _as_dict(players[my_index]), _as_dict(players[1 - my_index]), my_index

    def _observe(self, obs: dict[str, Any]) -> None:
        current = _as_dict(obs.get("current"))
        turn = _as_int(current.get("turn"), -1)
        if turn >= 0:
            if turn < self._last_turn:
                self._reset()
            self._last_turn = max(self._last_turn, turn)

        sides = self._players(obs)
        if sides is None:
            return
        _me, _opp, my_index = sides
        opp_index = 1 - my_index

        pending_attack: int | None = None
        for event in _as_list(obs.get("logs")):
            event = _as_dict(event)
            etype = _as_int(event.get("type"), -1)
            if etype == _LOG_ATTACK:
                if _as_int(event.get("playerIndex"), -1) == my_index:
                    pending_attack = _as_int(event.get("attackId"), -1)
                else:
                    pending_attack = None
            elif etype == _LOG_HP_CHANGE and pending_attack is not None and pending_attack >= 0:
                if _as_int(event.get("playerIndex"), -1) != opp_index:
                    continue
                if event.get("putDamageCounter"):
                    continue
                serial = _as_int(event.get("serial"), -1)
                value = _as_int(event.get("value"), 0)
                if serial >= 0 and value == 0:
                    self._futile_observed[(pending_attack, serial)] = self._last_turn
                elif serial >= 0 and value < 0:
                    self._futile_observed.pop((pending_attack, serial), None)
                pending_attack = None

    # ------------------------------------------------------------- adjusting
    def _defender(self, obs: dict[str, Any]) -> tuple[int | None, int | None]:
        sides = self._players(obs)
        if sides is None:
            return None, None
        _me, opp, _my_index = sides
        active = _as_list(opp.get("active"))
        if not active:
            return None, None
        return _card_id(active[0]), _card_serial(active[0])

    def _my_active_id(self, obs: dict[str, Any]) -> int | None:
        sides = self._players(obs)
        if sides is None:
            return None
        me, _opp, _my_index = sides
        active = _as_list(me.get("active"))
        return _card_id(active[0]) if active else None

    def _water_in_own_discard(self, obs: dict[str, Any]) -> int:
        sides = self._players(obs)
        if sides is None:
            return 0
        me, _opp, _my_index = sides
        return sum(1 for card in _as_list(me.get("discard")) if _card_id(card) == BASIC_WATER_ENERGY_ID)

    def _attack_is_futile(self, option: dict[str, Any], obs: dict[str, Any]) -> bool:
        attack_id = _as_int(option.get("attackId"), -1)
        if attack_id < 0:
            return False
        defender_id, defender_serial = self._defender(obs)
        # static: permanent ex-immunity wall vs our ex attacker
        if defender_id in EX_WALL_CARD_IDS and self._my_active_id(obs) in EX_CARD_IDS:
            return True
        # static: Riptide scales to exactly 0 with an empty own Water discard
        if attack_id == RIPTIDE_ATTACK_ID and self._water_in_own_discard(obs) == 0:
            return True
        # dynamic: observed 0-damage hit on this same defender, not yet expired
        if defender_serial is not None:
            observed = self._futile_observed.get((attack_id, defender_serial))
            if observed is not None and (self._last_turn - observed) <= self.expire_turns:
                return True
        return False

    def _non_ex_bench_exists(self, obs: dict[str, Any]) -> bool:
        sides = self._players(obs)
        if sides is None:
            return False
        me, _opp, _my_index = sides
        return any(_card_id(mon) not in EX_CARD_IDS for mon in _as_list(me.get("bench")))

    def _pokemon_at(self, me: dict[str, Any], area: int, index: int) -> dict[str, Any]:
        if area == _AREA_ACTIVE:
            zone = _as_list(me.get("active"))
        elif area == _AREA_BENCH:
            zone = _as_list(me.get("bench"))
        else:
            return {}
        return _as_dict(zone[index]) if 0 <= index < len(zone) else {}

    def _ex_wall_energy_routing(self, obs: dict[str, Any], options: list[Any], chosen: list[int]) -> list[int]:
        """Vs a confirmed ex-immunity wall (345/330) our ex attacker deals 0, so
        energy on the ex is wasted. If the chosen action ends the turn or attaches
        to an ex, redirect the attach onto a NON-EX Pokemon (our only real answer,
        e.g. Hariyama), preferring the one already closest to attacking (most
        energy). Gated to ex-walls only, so normal energy sequencing is untouched.
        Never fabricates an option: only picks among the legal ATTACH options."""
        index = chosen[0]
        option = _as_dict(options[index])
        otype = _as_int(option.get("type"), -1)
        if otype not in (_OPT_ATTACH, _OPT_END):
            return chosen
        sides = self._players(obs)
        if sides is None:
            return chosen
        me, _opp, my_index = sides

        non_ex_attaches: list[tuple[int, int]] = []  # (option_index, target energy count)
        chosen_targets_ex = False
        for i, raw in enumerate(options):
            opt = _as_dict(raw)
            if _as_int(opt.get("type"), -1) != _OPT_ATTACH:
                continue
            area = _as_int(opt.get("inPlayArea"), -1)
            if area not in (_AREA_ACTIVE, _AREA_BENCH):
                continue
            if _as_int(opt.get("playerIndex"), my_index) != my_index:
                continue
            target = self._pokemon_at(me, area, _as_int(opt.get("inPlayIndex"), -1))
            if _card_id(target) in EX_CARD_IDS:
                if i == index:
                    chosen_targets_ex = True
                continue
            non_ex_attaches.append((i, _energy_count(target)))

        if not non_ex_attaches:
            return chosen
        # Only redirect when the turn is otherwise wasted vs the wall.
        if otype == _OPT_END or (otype == _OPT_ATTACH and chosen_targets_ex):
            best_index = max(non_ex_attaches, key=lambda pair: pair[1])[0]
            return [best_index]
        return chosen

    def _adjust(self, obs: dict[str, Any], select: dict[str, Any], chosen: list[int]) -> list[int]:
        if len(chosen) != 1:
            return chosen
        context = _as_int(select.get("context"), -1)
        options = _as_list(select.get("option"))
        index = chosen[0]
        if not (0 <= index < len(options)):
            return chosen

        if context in (_CTX_SWITCH, _CTX_TO_ACTIVE):
            return self._adjust_promotion(obs, options, chosen)
        if context != _CTX_MAIN:
            return chosen

        # Ex-wall energy routing: vs 345/330, fuel our non-ex answer instead of
        # wasting the turn (END) or feeding the useless ex. Fires only vs walls.
        defender_id, _defender_serial = self._defender(obs)
        if defender_id in EX_WALL_CARD_IDS:
            routed = self._ex_wall_energy_routing(obs, options, chosen)
            if routed != chosen:
                return routed

        option = _as_dict(options[index])
        if _as_int(option.get("type"), -1) != _OPT_ATTACK:
            return chosen
        if not self._attack_is_futile(option, obs):
            return chosen

        attacks = [
            (i, _as_dict(opt)) for i, opt in enumerate(options)
            if _as_int(_as_dict(opt).get("type"), -1) == _OPT_ATTACK
        ]
        useful = [(i, opt) for i, opt in attacks if i != index and not self._attack_is_futile(opt, obs)]
        if useful:
            non_mill = [i for i, opt in useful if _as_int(opt.get("attackId"), -1) not in SELF_MILL_ATTACK_IDS]
            return [non_mill[0] if non_mill else useful[0][0]]

        retreat = next((i for i, opt in enumerate(options) if _as_int(_as_dict(opt).get("type"), -1) == _OPT_RETREAT), None)
        if retreat is not None and self._non_ex_bench_exists(obs):
            self._pending_promote = True
            return [retreat]

        if _as_int(option.get("attackId"), -1) in SELF_MILL_ATTACK_IDS:
            other_futile = [
                i for i, opt in attacks
                if i != index and _as_int(opt.get("attackId"), -1) not in SELF_MILL_ATTACK_IDS
            ]
            if other_futile:
                return [other_futile[0]]
            end = next((i for i, opt in enumerate(options) if _as_int(_as_dict(opt).get("type"), -1) == _OPT_END), None)
            if end is not None:
                return [end]
        return chosen

    def _adjust_promotion(self, obs: dict[str, Any], options: list[Any], chosen: list[int]) -> list[int]:
        if not self._pending_promote:
            return chosen
        self._pending_promote = False
        sides = self._players(obs)
        if sides is None:
            return chosen
        me, _opp, my_index = sides
        bench = _as_list(me.get("bench"))

        best_index = None
        best_key: tuple[int, int] | None = None
        for i, raw in enumerate(options):
            opt = _as_dict(raw)
            if _as_int(opt.get("area"), -1) != _AREA_BENCH:
                continue
            if _as_int(opt.get("playerIndex"), my_index) != my_index:
                continue
            bench_index = _as_int(opt.get("index"), -1)
            if not (0 <= bench_index < len(bench)):
                continue
            mon = _as_dict(bench[bench_index])
            non_ex = 1 if _card_id(mon) not in EX_CARD_IDS else 0
            key = (non_ex, _energy_count(mon))
            if best_key is None or key > best_key:
                best_key = key
                best_index = i
        if best_index is not None and best_key is not None and best_key[0] == 1:
            return [best_index]
        return chosen
