"""Greedy baseline agent (``greedy`` arena base).

A deliberately dumb, transparent policy: on a MAIN turn it takes the single
highest-value immediate action by a fixed priority ladder (take a lethal KO >
fuel/develop the board > attack for chip damage > pass); on every other select
context it plays the first legal option. No ML, no search, no card semantics,
no replay data -- the intended "one-shot heuristic" baseline to put a real
number on the board and to serve as a floor opponent in the arena.

INVARIANT #1: every return is routed through ``guard_selection`` so an
unexpected select shape yields a legal fallback instead of a crash.
INVARIANT #6: option/context classification goes through ``enum_name`` which
returns an ``UNKNOWN_*`` sentinel for members added mid-competition, so unknown
types fall into the lowest priority tier rather than raising.
"""

from __future__ import annotations

from typing import Any

from cg.api import AreaType, SelectType
from deepmew.policy.attacks import attack_damage, opponent_active_hp
from deepmew.policy.enums import enum_name, option_type
from deepmew.policy.selection import guard_selection, selection_bounds, safe_fallback


# Fixed priority ladder (higher wins). Deliberately simple and readable.
_TIER_LETHAL_ATTACK = 60
_TIER_ATTACH = 50
_TIER_EVOLVE = 40
_TIER_PLAY = 30
_TIER_ABILITY = 20
_TIER_CHIP_ATTACK = 10
_TIER_ZERO_ATTACK = 0
_TIER_END = -10
_TIER_RISKY = -20
_TIER_OTHER = -30

_RISKY = {"DISCARD", "RETREAT", "SWITCH", "TO_DECK", "TO_DECK_BOTTOM"}


def _attach_target_is_active(option: dict[str, Any]) -> bool:
    try:
        return AreaType(option.get("inPlayArea")).name == "ACTIVE"
    except (TypeError, ValueError):
        return False


def _score_main_option(option: dict[str, Any], opp_hp: int | None) -> tuple[int, int]:
    """Return (tier, tiebreak); higher is better. Pure printed-damage greedy."""
    otype = option_type(option)

    if otype == "ATTACK":
        damage = attack_damage(option.get("attackId"))
        if damage is None:
            return (_TIER_CHIP_ATTACK, 0)
        if damage <= 0:
            return (_TIER_ZERO_ATTACK, 0)
        if opp_hp is not None and opp_hp > 0 and damage >= opp_hp:
            return (_TIER_LETHAL_ATTACK, min(damage, 400))
        return (_TIER_CHIP_ATTACK, min(damage, 400))
    if otype == "ATTACH":
        return (_TIER_ATTACH, 1 if _attach_target_is_active(option) else 0)
    if otype == "EVOLVE":
        return (_TIER_EVOLVE, 0)
    if otype == "PLAY":
        return (_TIER_PLAY, 0)
    if otype in {"ABILITY", "SKILL"}:
        return (_TIER_ABILITY, 0)
    if otype == "END":
        return (_TIER_END, 0)
    if otype in _RISKY:
        return (_TIER_RISKY, 0)
    return (_TIER_OTHER, 0)


def choose_greedy_action(obs: dict[str, Any]) -> list[int]:
    """Greedy single-step selection; always returns a guarded legal selection."""
    select = obs.get("select") or {}
    options = select.get("option") or []
    min_count, max_count = selection_bounds(select, len(options))
    if max_count == 0:
        return guard_selection([], select)

    select_type = enum_name(SelectType, select.get("type"))
    # Greedy reasoning only applies to a MAIN turn where we pick one action.
    if select_type != "MAIN" or min_count > 1:
        return guard_selection(list(range(max_count)), select)

    opp_hp = opponent_active_hp(obs)
    best_index = 0
    best_key = _score_main_option(options[0], opp_hp)
    for index in range(1, len(options)):
        key = _score_main_option(options[index], opp_hp)
        if key > best_key:
            best_key = key
            best_index = index

    if min_count <= 1 <= max_count:
        return guard_selection([best_index], select)
    return safe_fallback(select)
