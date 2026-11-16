"""Explicit recorded-request compatibility for the inspected CABT 1.30.1 source.

This admits a parser candidate, not engine equivalence or training membership.
Every observation, prior-request action and terminal outcome still passes the
existing converter. Native version and the distinct alignment identity survive.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

from deepmew_ex.datasets.v2.contract import BoundContract, SourceContract, load_contract
from deepmew_ex.registry.store import canonical, digest, file_hash

SPECIFICATION_SHA256 = 'cb66b2e69b0debb7e99ec0eb6269018abba57f8849ba1ef28d2819dcfac0482e'


@dataclass(frozen=True)
class RecordedLegacyContract:
    module_version: str = '1.30.1'
    source_schema: str = 'schema1:cabt1.0.0:module1.30.1'
    alignment_rule_id: str = 'KAGGLE_SCHEMA1_CABT_1_0_0_MODULE_1_30_1_CAUSAL_ALIGNMENT_EX_RECOVERY_V1'


def recorded_legacy_contract() -> BoundContract:
    base = load_contract()

    def source_contract(value: Mapping[str, object]) -> SourceContract | None:
        if value.get('module_version') != '1.30.1':
            return base.source_contract(value)
        if type(value.get('schema_version')) is not int or value['schema_version'] != 1 or value.get('name') != 'cabt' or value.get('version') != '1.0.0':
            return None
        if digest(canonical(value.get('specification'))) != SPECIFICATION_SHA256:
            return None
        return RecordedLegacyContract()

    return replace(base, source_contract=source_contract, hashes={**base.hashes, 'recorded_legacy_contract': file_hash(Path(__file__)), 'recorded_legacy_specification': SPECIFICATION_SHA256})
