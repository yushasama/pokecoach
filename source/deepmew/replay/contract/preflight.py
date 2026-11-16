"""Fail-closed execution gate for legacy serious replay consumers.

Importing a historical research script remains safe. Its executable entry
point must call require_accepted_canonical_decision_receipt before it reads
data, creates output, launches a worker, probes, or trains.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from deepmew.replay.contract import CANONICAL_DECISION_CONTRACT_V1
from deepmew.replay.contract.identities import implementation_sha256, sha256_json
from deepmew.replay.contract.schemas import VALIDATOR_VERSION, ValidationOutcome


CANONICAL_DECISION_CONSUMER_RECEIPT_VERSION = (
    "deepmew.canonical_decision_consumer_receipt.v1"
)
CANONICAL_CONTRACT_OWNER = "deepmew.replay.contract"
CONTRACT_OWNER_ALLOWLIST = frozenset({CANONICAL_CONTRACT_OWNER})
SERIOUS_CONSUMER_ACCEPTED_OUTCOMES = (
    ValidationOutcome.VALID_ACTOR_AND_CRITIC.value,
)
_VALIDATOR_IMPLEMENTATION_FILES = (
    "schemas.py",
    "identities.py",
    "source_events.py",
    "causal_alignment.py",
    "legal_actions.py",
    "chosen_actions.py",
    "validation.py",
)


class LegacyDecisionConsumerDeprecationHardStop(RuntimeError):
    """A legacy serious consumer lacks current accepted contract evidence."""


@dataclass(frozen=True)
class AcceptedCanonicalDecisionReceipt:
    path: Path
    manifest_sha256: str
    validator_implementation_sha256: str
    payload: Mapping[str, Any]


def canonical_validator_implementation_sha256() -> str:
    """Hash the current validator and its contract-semantic dependencies."""

    package_root = Path(__file__).resolve().parent

    return implementation_sha256(
        *(package_root / name for name in _VALIDATOR_IMPLEMENTATION_FILES)
    )


def _hard_stop(consumer: str, reason: str) -> LegacyDecisionConsumerDeprecationHardStop:
    name = consumer if isinstance(consumer, str) and consumer else "unknown_consumer"

    return LegacyDecisionConsumerDeprecationHardStop(
        f"{name} is deprecated fail-closed until it receives a complete "
        f"{CANONICAL_DECISION_CONTRACT_V1} receipt: {reason}"
    )


def _validate_sha256(value: Any, *, consumer: str, field: str) -> str:
    if not isinstance(value, str):
        raise _hard_stop(consumer, f"{field} is missing")
    normalized = value.lower()
    if (
        value != normalized
        or len(normalized) != 64
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise _hard_stop(consumer, f"{field} is not a canonical SHA-256")

    return normalized


def _read_receipt(
    receipt_path: str | Path | None,
    *,
    consumer: str,
) -> tuple[Path, dict[str, Any]]:
    if receipt_path is None or not str(receipt_path).strip():
        raise _hard_stop(consumer, "receipt path is missing")
    path = Path(receipt_path).expanduser().resolve()
    if not path.is_file():
        raise _hard_stop(consumer, "receipt file is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _hard_stop(consumer, "receipt is unreadable or corrupt") from exc
    if not isinstance(payload, dict):
        raise _hard_stop(consumer, "receipt root is not an object")

    return path, payload


def require_accepted_canonical_decision_receipt(
    receipt_path: str | Path | None,
    *,
    consumer: str,
) -> AcceptedCanonicalDecisionReceipt:
    """Verify current positive contract evidence before legacy execution."""

    if CANONICAL_CONTRACT_OWNER not in CONTRACT_OWNER_ALLOWLIST:
        raise _hard_stop(consumer, "canonical contract owner is not allowlisted")
    path, payload = _read_receipt(receipt_path, consumer=consumer)
    if payload.get("schema_version") != CANONICAL_DECISION_CONSUMER_RECEIPT_VERSION:
        raise _hard_stop(consumer, "receipt schema is missing or old")
    if payload.get("contract_version") != CANONICAL_DECISION_CONTRACT_V1:
        raise _hard_stop(consumer, "contract version is missing or old")
    if payload.get("validator_version") != VALIDATOR_VERSION:
        raise _hard_stop(consumer, "validator version is missing or old")
    implementation_sha256 = _validate_sha256(
        payload.get("validator_implementation_sha256"),
        consumer=consumer,
        field="validator_implementation_sha256",
    )
    try:
        current_implementation_sha256 = canonical_validator_implementation_sha256()
    except OSError as exc:
        raise _hard_stop(
            consumer,
            "current validator implementation cannot be hashed",
        ) from exc
    if implementation_sha256 != current_implementation_sha256:
        raise _hard_stop(consumer, "validator implementation hash is missing or old")
    if payload.get("status") != "complete":
        raise _hard_stop(consumer, "receipt status is not complete")
    accepted_outcomes = payload.get("accepted_outcomes")
    if (
        not isinstance(accepted_outcomes, list)
        or tuple(accepted_outcomes) != SERIOUS_CONSUMER_ACCEPTED_OUTCOMES
    ):
        raise _hard_stop(
            consumer,
            "accepted outcomes are missing, unresolved, critic-only, or unknown",
        )
    provided_manifest_sha256 = _validate_sha256(
        payload.get("manifest_sha256"),
        consumer=consumer,
        field="manifest_sha256",
    )
    unsigned_payload = dict(payload)
    unsigned_payload.pop("manifest_sha256", None)
    if sha256_json(unsigned_payload) != provided_manifest_sha256:
        raise _hard_stop(consumer, "manifest hash does not bind the receipt")

    return AcceptedCanonicalDecisionReceipt(
        path=path,
        manifest_sha256=provided_manifest_sha256,
        validator_implementation_sha256=implementation_sha256,
        payload=payload,
    )


__all__ = [
    "CANONICAL_CONTRACT_OWNER",
    "CANONICAL_DECISION_CONSUMER_RECEIPT_VERSION",
    "CONTRACT_OWNER_ALLOWLIST",
    "SERIOUS_CONSUMER_ACCEPTED_OUTCOMES",
    "AcceptedCanonicalDecisionReceipt",
    "LegacyDecisionConsumerDeprecationHardStop",
    "canonical_validator_implementation_sha256",
    "require_accepted_canonical_decision_receipt",
]
