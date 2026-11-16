from __future__ import annotations

from collections import Counter
from typing import Any

from cg.api import OptionType
from deepmew.card_semantics.accessors import card_id, get_field
from deepmew.card_semantics.deck_prior import load_deck_prior
from deepmew.card_semantics.models import AttackSemantics
from deepmew.card_semantics.visibility import own_player, visible_card_ids
from deepmew.deck_profiles import BASIC_WATER_ENERGY_ID, HAMMER_LANCHE_ATTACK_ID, RIPTIDE_ATTACK_ID


def visible_basic_water_in_discard(obs: dict[str, Any]) -> int:
    player = own_player(obs)
    if player is None:
        return 0
    return sum(1 for card in (get_field(player, "discard") or []) if card_id(card) == BASIC_WATER_ENERGY_ID)


def _hammer_lanche_damage(obs: dict[str, Any]) -> AttackSemantics:
    prior = load_deck_prior()
    total_cards = sum(prior.values()) or 60
    total_water = prior.get(BASIC_WATER_ENERGY_ID, 35)
    assumptions = ["uses deck.csv Basic Water density prior", "floors expected hits for conservatism"]

    player = own_player(obs)
    deck_count = get_field(player, "deckCount") if player is not None else None
    if player is not None and deck_count:
        visible = Counter(visible_card_ids(player))
        visible_water = visible.get(BASIC_WATER_ENERGY_ID, 0)
        water_remaining = max(0, total_water - visible_water)
        remaining_cards = max(1, int(deck_count))
        assumptions = (
            "uses deck.csv prior adjusted by visible own Basic Water cards",
            "does not know exact deck order",
            "floors expected hits for conservatism",
        )
    else:
        water_remaining = total_water
        remaining_cards = total_cards

    cards_seen = min(6, remaining_cards)
    expected_hits = int(cards_seen * water_remaining / max(1, remaining_cards))
    return AttackSemantics(
        attack_id=HAMMER_LANCHE_ATTACK_ID,
        effective_damage=expected_hits * 100,
        reason="hammer-lanche-water-density",
        assumptions=tuple(assumptions),
    )


def _riptide_damage(obs: dict[str, Any]) -> AttackSemantics:
    water_count = visible_basic_water_in_discard(obs)
    return AttackSemantics(
        attack_id=RIPTIDE_ATTACK_ID,
        effective_damage=water_count * 20,
        reason="riptide-visible-water-discard",
        assumptions=("counts visible Basic Water Energy in own discard",),
    )


def effective_attack_semantics(option: dict[str, Any], obs: dict[str, Any]) -> AttackSemantics | None:
    if option.get("type") != OptionType.ATTACK.value:
        return None
    attack_id = option.get("attackId")
    if attack_id is None:
        return None
    attack_id = int(attack_id)
    if attack_id == HAMMER_LANCHE_ATTACK_ID:
        return _hammer_lanche_damage(obs)
    if attack_id == RIPTIDE_ATTACK_ID:
        return _riptide_damage(obs)
    return None
