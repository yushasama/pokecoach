"""Resolve visible identities for engine legal options."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cg.api import AreaType, OptionType, all_card_data


@dataclass(frozen=True)
class OptionIdentity:
    option_type: str
    resolved: bool
    reason: str

    card_id: int | None = None
    card_name: str | None = None
    source_area: str | None = None
    source_index: int | None = None

    attack_id: int | None = None

    target_area: str | None = None
    target_index: int | None = None
    target_card_id: int | None = None
    target_card_name: str | None = None


_CARD_NAMES_BY_ID: dict[int, str] | None = None


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _enum_name(enum_cls: Any, value: Any) -> str:
    if value is None:
        return "UNKNOWN"
    try:
        return enum_cls(value).name
    except (TypeError, ValueError):
        return f"UNKNOWN_{value}"


def _area_name(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return AreaType(value).name
    except (TypeError, ValueError):
        return str(value).upper()


def _card_names_by_id() -> dict[int, str]:
    global _CARD_NAMES_BY_ID
    if _CARD_NAMES_BY_ID is None:
        names: dict[int, str] = {}
        for card in all_card_data():
            card_id = _int_or_none(_get(card, "cardId", _get(card, "id")))
            name = _get(card, "name", _get(card, "cardName"))
            if card_id is not None and name:
                names[card_id] = str(name)
        _CARD_NAMES_BY_ID = names
    return _CARD_NAMES_BY_ID


def _card_name(card_id: int | None) -> str | None:
    if card_id is None:
        return None
    return _card_names_by_id().get(card_id)


def card_name_for_id(card_id: int | None) -> str | None:
    """Public visible card-name lookup by card id."""

    return _card_name(card_id)


def _card_id(value: Any) -> int | None:
    if value is None:
        return None
    return _int_or_none(_get(value, "id", _get(value, "cardId", _get(value, "cardID"))))


def _current(obs: dict[str, Any]) -> dict[str, Any]:
    current = obs.get("current") if isinstance(obs, dict) else None
    return current if isinstance(current, dict) else {}


def _own_player(obs: dict[str, Any]) -> Any | None:
    current = _current(obs)
    players = current.get("players") or []
    your_index = _int_or_none(current.get("yourIndex")) or 0
    if not isinstance(players, list) or your_index < 0 or your_index >= len(players):
        return None
    return players[your_index]


def _own_hand(obs: dict[str, Any]) -> list[Any] | None:
    player = _own_player(obs)
    if player is None:
        return None
    hand = _get(player, "hand")
    return hand if isinstance(hand, list) else None


def _own_pokemon(obs: dict[str, Any], area: str | None, index: int | None) -> Any | None:
    if area is None or index is None:
        return None
    player = _own_player(obs)
    if player is None:
        return None
    if area == "ACTIVE":
        active = _get(player, "active") or []
        return active[index] if isinstance(active, list) and 0 <= index < len(active) else None
    if area == "BENCH":
        bench = _get(player, "bench") or []
        return bench[index] if isinstance(bench, list) and 0 <= index < len(bench) else None
    return None


def _identity_from_hand(option_type: str, obs: dict[str, Any], option: dict[str, Any]) -> OptionIdentity:
    source_index = _int_or_none(option.get("index"))
    if source_index is None:
        return OptionIdentity(option_type=option_type, resolved=False, reason="missing-hand-index")
    hand = _own_hand(obs)
    if hand is None:
        return OptionIdentity(
            option_type=option_type,
            resolved=False,
            reason="missing-own-hand",
            source_area="HAND",
            source_index=source_index,
        )
    if source_index < 0 or source_index >= len(hand):
        return OptionIdentity(
            option_type=option_type,
            resolved=False,
            reason="hand-index-out-of-range",
            source_area="HAND",
            source_index=source_index,
        )
    card_id = _card_id(hand[source_index])
    if card_id is None:
        return OptionIdentity(
            option_type=option_type,
            resolved=False,
            reason="hand-card-hidden-or-unknown",
            source_area="HAND",
            source_index=source_index,
        )
    return OptionIdentity(
        option_type=option_type,
        resolved=True,
        reason="resolved-from-own-hand",
        card_id=card_id,
        card_name=_card_name(card_id),
        source_area="HAND",
        source_index=source_index,
    )


def _target_fields(obs: dict[str, Any], option: dict[str, Any]) -> dict[str, Any]:
    target_area = _area_name(option.get("inPlayArea", option.get("targetArea")))
    target_index = _int_or_none(option.get("inPlayIndex", option.get("targetIndex")))
    if target_area == "ACTIVE" and target_index is None:
        target_index = 0
    target = _own_pokemon(obs, target_area, target_index)
    target_card_id = _card_id(target)
    return {
        "target_area": target_area,
        "target_index": target_index,
        "target_card_id": target_card_id,
        "target_card_name": _card_name(target_card_id),
    }


def resolve_option_identity(obs: dict[str, Any], option: dict[str, Any]) -> OptionIdentity:
    """Resolve visible card/target identity for one legal option.

    PLAY and EVOLVE source cards are resolved only from the current player's visible hand.
    Hidden or malformed shapes return unresolved identities instead of raising.
    """

    if not isinstance(obs, dict) or not isinstance(option, dict):
        return OptionIdentity(option_type="UNKNOWN", resolved=False, reason="unknown-shape")

    option_type = _enum_name(OptionType, option.get("type"))

    if option_type == "PLAY":
        return _identity_from_hand(option_type, obs, option)

    if option_type == "ATTACK":
        attack_id = _int_or_none(option.get("attackId", option.get("attackID")))
        return OptionIdentity(
            option_type=option_type,
            resolved=attack_id is not None,
            reason="attack-id-present" if attack_id is not None else "missing-attack-id",
            attack_id=attack_id,
        )

    if option_type == "ATTACH":
        target = _target_fields(obs, option)
        target_known = target["target_area"] is not None and target["target_index"] is not None
        return OptionIdentity(
            option_type=option_type,
            resolved=target_known,
            reason="visible-target" if target["target_card_id"] is not None else "target-only" if target_known else "missing-target",
            **target,
        )

    if option_type == "EVOLVE":
        source = _identity_from_hand(option_type, obs, option)
        target = _target_fields(obs, option)
        option_card_id = _int_or_none(option.get("cardId", option.get("cardID")))
        card_id = source.card_id if source.card_id is not None else option_card_id
        resolved = source.resolved or option_card_id is not None or target["target_card_id"] is not None
        reason = source.reason if source.resolved else "option-card-id" if option_card_id is not None else source.reason
        return OptionIdentity(
            option_type=option_type,
            resolved=resolved,
            reason=reason,
            card_id=card_id,
            card_name=_card_name(card_id),
            source_area=source.source_area,
            source_index=source.source_index,
            **target,
        )

    return OptionIdentity(option_type=option_type, resolved=False, reason="unsupported-option-type")
