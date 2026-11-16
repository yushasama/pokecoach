from __future__ import annotations

from typing import Any

from cg.api import OptionType, SelectType, all_attack
from deepmew.card_semantics import effective_attack_semantics
from deepmew.policy.enums import enum_name


_ATTACK_DAMAGE: dict[int, int] | None = None


def reset_attack_damage_cache() -> None:
    global _ATTACK_DAMAGE
    _ATTACK_DAMAGE = None


def attack_damage(attack_id: Any) -> int | None:
    global _ATTACK_DAMAGE
    if attack_id is None:
        return None
    if _ATTACK_DAMAGE is None:
        _ATTACK_DAMAGE = {attack.attackId: attack.damage for attack in all_attack()}
    return _ATTACK_DAMAGE.get(int(attack_id))


def attack_damage_for_option(option: dict[str, Any], obs: dict[str, Any], use_card_semantics: bool) -> int | None:
    if use_card_semantics:
        semantics = effective_attack_semantics(option, obs)
        if semantics is not None:
            return semantics.effective_damage
    return attack_damage(option.get("attackId"))


def attack_base_score(damage: int | None) -> tuple[int, str]:
    if damage is None:
        return 58, "attack-unknown-damage"
    if damage > 0:
        return 45 + min(damage, 400) // 5, "positive-damage-attack"
    return 42, "zero-damage-attack"


def opponent_active_hp(obs: dict[str, Any]) -> int | None:
    current = obs.get("current") or {}
    players = current.get("players") or []
    your_index = int(current.get("yourIndex") or 0)
    opponent_index = 1 - your_index
    if opponent_index >= len(players):
        return None
    active = (players[opponent_index] or {}).get("active") or []
    if not active or active[0] is None:
        return None
    hp = active[0].get("hp")
    return int(hp) if hp is not None else None


def has_main_setup_option(obs: dict[str, Any]) -> bool:
    select = obs.get("select") or {}
    if enum_name(SelectType, select.get("type")) != "MAIN":
        return False
    return any(
        enum_name(OptionType, option.get("type")) in {"PLAY", "EVOLVE", "ATTACH"}
        for option in select.get("option") or []
    )


def attack_pressure(option: dict[str, Any], obs: dict[str, Any], use_card_semantics: bool) -> str:
    damage = attack_damage_for_option(option, obs, use_card_semantics)
    if damage is None:
        return "unknown"
    opponent_hp = opponent_active_hp(obs)
    if opponent_hp is not None and opponent_hp > 0:
        if damage >= opponent_hp:
            return "ko"
        if damage * 2 >= opponent_hp:
            return "meaningful"
        return "weak"
    if damage >= 100:
        return "meaningful"
    if damage <= 30:
        return "weak"
    return "unknown"
