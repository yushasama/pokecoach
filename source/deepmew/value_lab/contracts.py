"""Shared integrity contracts for value/Q split and cache construction."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


STATE_ACTION_SIGNATURE_VERSION = "deepmew.canonical_x_effective_action_signature.v2"
ROW_EXCLUSION_SCHEMA_VERSION = "deepmew.value_q_row_exclusion_manifest.v1"
SPLIT_SCHEMA_VERSION = "deepmew.value_q_split_manifest.v2"


def _without_empty_fields(value: Any) -> Any:
    """Canonicalize JSON-like option data while dropping absent-value spelling."""
    if isinstance(value, dict):
        normalized = {
            str(key): _without_empty_fields(item)
            for key, item in value.items()
            if item is not None and item != ""
        }

        return {key: item for key, item in normalized.items() if item not in ({}, [])}
    if isinstance(value, list):

        return [_without_empty_fields(item) for item in value]

    return value


def _normalized_option(option: dict[str, Any]) -> dict[str, Any]:
    """Normalize one selected legal option without its enumeration position."""
    normalized = _without_empty_fields(option)
    if "semantic_key" in normalized or "option_type" in normalized:
        normalized.pop("index", None)
        normalized.pop("option_position", None)

    return normalized


def normalized_chosen_action(transition: dict[str, Any]) -> dict[str, Any]:
    """Return effective selected-option identity, independent of list ordering."""
    chosen = transition.get("chosen_action")
    if not isinstance(chosen, dict):
        raise ValueError("transition chosen_action must be an object")
    indices = chosen.get("indices")
    legal = transition.get("legal_action_set")
    if not isinstance(indices, list) or not isinstance(legal, list):
        raise ValueError("transition action contract requires indices and legal_action_set")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise ValueError("chosen action indices must be integers")
    if not indices:
        if chosen.get("allows_empty_choice") is not True:
            raise ValueError("empty chosen action is not explicitly allowed")

        return {"selected_options": [], "allows_empty_choice": True}

    selected: list[dict[str, Any]] = []
    if chosen.get("index_space") == "option_position":
        for index in indices:
            if not 0 <= index < len(legal) or not isinstance(legal[index], dict):
                raise ValueError("chosen positional index is outside legal_action_set")
            selected.append(_normalized_option(legal[index]))
    else:
        legal_by_index: dict[int, dict[str, Any]] = {}
        for position, option in enumerate(legal):
            if not isinstance(option, dict):
                raise ValueError("legal action options must be objects")
            option_index = option.get("index", position)
            if isinstance(option_index, bool) or not isinstance(option_index, int):
                raise ValueError("legal action option index must be an integer")
            if option_index in legal_by_index:
                raise ValueError("legal action option indices must be unique")
            legal_by_index[option_index] = option
        for index in indices:
            if index not in legal_by_index:
                raise ValueError("chosen semantic index is outside legal_action_set")
            selected.append(_normalized_option(legal_by_index[index]))
    selected.sort(key=lambda option: json.dumps(option, sort_keys=True, separators=(",", ":")))

    return {"selected_options": selected}


def canonical_state_action_signature(features: np.ndarray, transition: dict[str, Any]) -> bytes:
    """BLAKE2b-128 over the exact canonical model row plus normalized action."""
    array = np.ascontiguousarray(features, dtype=np.float32)
    action = json.dumps(
        normalized_chosen_action(transition),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    digest = hashlib.blake2b(digest_size=16)
    digest.update(array.tobytes(order="C"))
    digest.update(b"\0")
    digest.update(action)

    return digest.digest()
