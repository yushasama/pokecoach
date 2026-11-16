"""Content-addressed identities for canonical decision semantics."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping as MappingABC
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


CANONICAL_DECISION_ID_PREFIX = "decision1"


def canonical_payload(value: Any) -> Any:
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, dict):
        return {
            str(key): canonical_payload(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [canonical_payload(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value):
        return {
            field.name: canonical_payload(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, MappingABC):
        return {
            str(key): canonical_payload(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Path):
        return value.as_posix()

    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        canonical_payload(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def identity(prefix: str, value: Any) -> str:
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("identity prefix must be a non-empty string")

    return f"{prefix}-{sha256_json(value)}"


def implementation_sha256(*paths: str | Path) -> str:
    digest = hashlib.sha256()
    for raw_path in sorted(Path(path).resolve(strict=True) for path in paths):
        digest.update(raw_path.name.encode("utf-8"))
        with raw_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)

    return digest.hexdigest()


def is_lowercase_sha256(value: Any) -> bool:
    """Return whether *value* is an exact lowercase SHA-256 string."""

    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def canonical_decision_id(
    *,
    schema_version: str,
    admission_identity: str,
    source_game_id: str,
    actor_seat: int,
    causal_decision_index: int,
    pre_action_observation_id: str,
    legal_action_set_id: str,
    chosen_action_id: str,
    resulting_transition_id: str,
    causal_aligner_version: str,
    causal_aligner_implementation_sha256: str,
    legality_validator_version: str,
    legality_validator_implementation_sha256: str,
    action_schema_version: str,
    action_schema_implementation_sha256: str,
) -> str:
    """Build the frozen V1 semantic decision identity."""

    payload = {
        "schema_version": schema_version,
        "admission_identity": admission_identity,
        "source_game_id": source_game_id,
        "actor_seat": actor_seat,
        "causal_decision_index": causal_decision_index,
        "pre_action_observation_id": pre_action_observation_id,
        "legal_action_set_id": legal_action_set_id,
        "chosen_action_id": chosen_action_id,
        "resulting_transition_id": resulting_transition_id,
        "causal_aligner_version": causal_aligner_version,
        "causal_aligner_implementation_sha256": causal_aligner_implementation_sha256,
        "legality_validator_version": legality_validator_version,
        "legality_validator_implementation_sha256": (
            legality_validator_implementation_sha256
        ),
        "action_schema_version": action_schema_version,
        "action_schema_implementation_sha256": action_schema_implementation_sha256,
    }

    return identity(CANONICAL_DECISION_ID_PREFIX, payload)


__all__ = [
    "CANONICAL_DECISION_ID_PREFIX",
    "canonical_decision_id",
    "canonical_json_bytes",
    "canonical_payload",
    "identity",
    "implementation_sha256",
    "is_lowercase_sha256",
    "sha256_json",
]
