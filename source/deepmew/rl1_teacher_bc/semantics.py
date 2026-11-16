from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Any

from cg.api import AreaType, OptionType, SelectContext, SelectType
from deepmew.replay.contract.legal_actions import build_refgame_semantic_legal_options


REFERENCE_POLICIES = ("reference_abomasnow", "reference_iono", "reference_dragapult", "reference_lucario")
HASH_BINS = 16
_SELECT_TYPES = tuple(member.name for member in SelectType)
_SELECT_CONTEXTS = tuple(member.name for member in SelectContext)
_OPTION_TYPES = tuple(member.name for member in OptionType)
_AREA_TYPES = tuple(member.name for member in AreaType)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _one_hot(value: Any, names: tuple[str, ...]) -> list[float]:
    text = str(value or "")

    return [1.0 if text == name else 0.0 for name in names]


def _hash_features(value: Any, bins: int = HASH_BINS) -> list[float]:
    if value is None or value == "":
        return [0.0] * bins
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()

    return [digest[index] / 255.0 for index in range(bins)]


def legal_action_table(obs: dict[str, Any], select: dict[str, Any]) -> list[dict[str, Any]]:
    """Compatibility wrapper around the canonical refgame crosswalk."""

    return list(build_refgame_semantic_legal_options(obs, select=select))


@lru_cache(maxsize=100_000)
def _cached_option_features(
    select_type: str,
    select_context: str,
    option_type: str,
    source_area: str,
    source_card_id: str,
    target_area: str,
    target_card_id: str,
    target_index: str,
    attack_id: str,
    semantic_key: str,
) -> tuple[float, ...]:
    vector: list[float] = []
    vector.extend(_one_hot(select_type, _SELECT_TYPES))
    vector.extend(_one_hot(select_context, _SELECT_CONTEXTS))
    vector.extend(_one_hot(option_type, _OPTION_TYPES))
    vector.extend(_one_hot(source_area, _AREA_TYPES))
    vector.extend(_one_hot(target_area, _AREA_TYPES))
    vector.extend(
        [
            1.0 if source_card_id else 0.0,
            1.0 if target_card_id else 0.0,
            1.0 if attack_id else 0.0,
            1.0 if semantic_key else 0.0,
        ]
    )
    vector.extend(_hash_features(source_card_id))
    vector.extend(_hash_features(target_card_id))
    vector.extend(_hash_features(attack_id))
    vector.extend(_hash_features(semantic_key))
    index = _as_int(target_index, 0)
    vector.append(max(0.0, min(1.0, index / 5.0)))

    return tuple(vector)


def build_option_features(option: dict[str, Any]) -> list[float]:
    vector = _cached_option_features(
        str(option.get("select_type") or ""),
        str(option.get("select_context") or ""),
        str(option.get("option_type") or ""),
        str(option.get("source_area") or ""),
        str(option.get("source_card_id") or ""),
        str(option.get("target_area") or ""),
        str(option.get("target_card_id") or ""),
        str(option.get("target_index") or ""),
        str(option.get("attack_id") or ""),
        str(option.get("semantic_key") or ""),
    )

    return list(vector)
