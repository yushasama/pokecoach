"""Explicit raw Kaggle replay source contracts admitted by the V1 owner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class KaggleReplaySourceContract:
    """Exact versioned metadata and identity rules for one replay source."""

    module_version: str
    source_schema: str
    alignment_rule_id: str
    source_scope: str
    actor_visible_content_identity: bool


def _contract(module_version: str) -> KaggleReplaySourceContract:
    source_schema = f"schema1:cabt1.0.0:module{module_version}"

    return KaggleReplaySourceContract(
        module_version=module_version,
        source_schema=source_schema,
        alignment_rule_id=(
            "KAGGLE_SCHEMA1_CABT_1_0_0_MODULE_"
            f"{module_version.replace('.', '_')}_CAUSAL_ALIGNMENT_V1"
        ),
        source_scope=(
            "kaggle_episode_replay schema=1 environment=cabt "
            "environment_version=1.0.0 "
            f"module_version={module_version}"
        ),
        actor_visible_content_identity=module_version == "1.32.0",
    )


KAGGLE_REPLAY_SOURCE_CONTRACTS = tuple(
    _contract(version) for version in ("1.30.2", "1.32.0")
)
KAGGLE_REPLAY_SOURCE_CONTRACT_BY_MODULE = {
    contract.module_version: contract
    for contract in KAGGLE_REPLAY_SOURCE_CONTRACTS
}
KAGGLE_REPLAY_SOURCE_CONTRACT_BY_SCHEMA = {
    contract.source_schema: contract
    for contract in KAGGLE_REPLAY_SOURCE_CONTRACTS
}
KAGGLE_REPLAY_SOURCE_SCHEMAS = frozenset(
    KAGGLE_REPLAY_SOURCE_CONTRACT_BY_SCHEMA
)


def kaggle_source_contract_from_metadata(
    data: Mapping[str, Any],
) -> KaggleReplaySourceContract | None:
    """Return the exact registered contract for replay metadata, if any."""

    if (
        data.get("schema_version") != 1
        or data.get("name") != "cabt"
        or data.get("version") != "1.0.0"
    ):
        return None
    module_version = data.get("module_version")
    if not isinstance(module_version, str):
        return None

    return KAGGLE_REPLAY_SOURCE_CONTRACT_BY_MODULE.get(module_version)


__all__ = [
    "KAGGLE_REPLAY_SOURCE_CONTRACTS",
    "KAGGLE_REPLAY_SOURCE_CONTRACT_BY_MODULE",
    "KAGGLE_REPLAY_SOURCE_CONTRACT_BY_SCHEMA",
    "KAGGLE_REPLAY_SOURCE_SCHEMAS",
    "KaggleReplaySourceContract",
    "kaggle_source_contract_from_metadata",
]
