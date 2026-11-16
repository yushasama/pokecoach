"""Deterministic card-role classification for K4.1 goal_of_play labeling.

Roles are derived from engine card metadata (``cardType``/stage/ex flags) plus a
small name-token table for supporter/item sub-roles (gust vs draw vs disruption).
Card *text* is not available in the engine schema, so sub-roles are an explicit
name heuristic, not a parsed effect. Built lazily and cached; offline/eval-only.
"""

from __future__ import annotations

from typing import Any

from deepmew.options.identity import card_name_for_id

# Role families.
POKEMON_BASIC = "pokemon_basic"
POKEMON_EVO = "pokemon_evo"
SUPPORTER_DRAW = "supporter_draw"
SUPPORTER_GUST = "supporter_gust"
SUPPORTER_DISRUPT = "supporter_disrupt"
ITEM_CONSISTENCY = "item_consistency"
ITEM_OTHER = "item_other"
TOOL = "tool"
ENERGY = "energy"
STADIUM = "stadium"
UNKNOWN_ROLE = "unknown"

# cg.api card types observed: 0 Pokemon, 1 Item, 2 Tool, 3 Supporter, 4 Stadium,
# 5 Energy. Unknown values degrade to UNKNOWN_ROLE.
_TYPE_POKEMON = 0
_TYPE_ITEM = 1
_TYPE_TOOL = 2
_TYPE_SUPPORTER = 3
_TYPE_STADIUM = 4
_TYPE_ENERGY = 5

# Name tokens (lowercased) that mark a card's strategic sub-role. Match is a
# substring test against the normalized card name.
_GUST_TOKENS = ("boss's orders", "boss’s orders", "counter catcher", "prime catcher")
_DISRUPT_TOKENS = ("iono", "judge", "roxanne", "team rocket's", "team rocket’s", "crispin")
_DRAW_SEARCH_TOKENS = (
    "professor", "research", "arven", "cyrano", "poffin", "pad", "ball",
    "stretcher", "sacred ash", "lana", "determination", "nemona", "hilda",
    "earthen vessel", "energy search", "energy retrieval", "buddy",
)

_ROLE_CACHE: dict[int, str] = {}
_META_CACHE: dict[int, Any] | None = None


def _normalize(name: str | None) -> str:
    return (name or "").strip().lower().replace("’", "'")


def _meta_by_id() -> dict[int, Any]:
    global _META_CACHE
    if _META_CACHE is None:
        from cg.api import all_card_data

        meta: dict[int, Any] = {}
        for card in all_card_data():
            cid = getattr(card, "cardId", None)
            if cid is None and isinstance(card, dict):
                cid = card.get("cardId", card.get("id"))
            if cid is not None:
                meta[int(cid)] = card
        _META_CACHE = meta
    return _META_CACHE


def _get(card: Any, key: str, default: Any = None) -> Any:
    if isinstance(card, dict):
        return card.get(key, default)
    return getattr(card, key, default)


def _supporter_role(name: str) -> str:
    if any(tok in name for tok in _GUST_TOKENS):
        return SUPPORTER_GUST
    if any(tok in name for tok in _DISRUPT_TOKENS):
        return SUPPORTER_DISRUPT
    return SUPPORTER_DRAW


def _item_role(name: str) -> str:
    if any(tok in name for tok in _GUST_TOKENS):
        return SUPPORTER_GUST
    if any(tok in name for tok in _DRAW_SEARCH_TOKENS):
        return ITEM_CONSISTENCY
    return ITEM_OTHER


def card_role(card_id: int | None) -> str:
    """Classify a card id into a strategic role family (cached)."""
    if card_id is None:
        return UNKNOWN_ROLE
    cached = _ROLE_CACHE.get(card_id)
    if cached is not None:
        return cached

    card = _meta_by_id().get(card_id)
    if card is None:
        role = UNKNOWN_ROLE
    else:
        card_type = _get(card, "cardType")
        name = _normalize(card_name_for_id(card_id))
        if card_type == _TYPE_POKEMON:
            role = POKEMON_BASIC if _get(card, "basic") else POKEMON_EVO
        elif card_type == _TYPE_SUPPORTER:
            role = _supporter_role(name)
        elif card_type == _TYPE_ITEM:
            role = _item_role(name)
        elif card_type == _TYPE_TOOL:
            role = TOOL
        elif card_type == _TYPE_STADIUM:
            role = STADIUM
        elif card_type == _TYPE_ENERGY:
            role = ENERGY
        else:
            role = UNKNOWN_ROLE
    _ROLE_CACHE[card_id] = role
    return role
