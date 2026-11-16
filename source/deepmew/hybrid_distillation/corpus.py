"""Admission gates for prospective hybrid-distillation source rows."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


COLLECTOR_CAPABILITY_SCHEMA_VERSION = "deepmew.hybrid.collector_capability.v1"
REQUIRED_COMPOUND_ACTION_FIELDS = (
    "compound_action_group_id",
    "compound_action_stage_index",
    "compound_action_stage_count",
    "compound_action_complete",
    "canonical_legal_action_set",
)


def audit_collector_capability(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Prove a collector can support complete canonical parent distributions.

    A per-prompt decision ID is not a compound-action identity. Likewise,
    `response_span_id` groups an opponent response and is deliberately not
    accepted as a substitute for the conditional selection chain.
    """

    missing = Counter()
    invalid = Counter()
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        for field in REQUIRED_COMPOUND_ACTION_FIELDS:
            if field not in row:
                missing[field] += 1
        group_id = row.get("compound_action_group_id")
        if group_id is not None and (not isinstance(group_id, str) or not group_id.strip()):
            invalid["compound_action_group_id"] += 1
        elif isinstance(group_id, str):
            groups.setdefault(group_id, []).append(row)
        stage_index = row.get("compound_action_stage_index")
        stage_count = row.get("compound_action_stage_count")
        if stage_index is not None and (
            isinstance(stage_index, bool) or not isinstance(stage_index, int) or stage_index < 0
        ):
            invalid["compound_action_stage_index"] += 1
        if stage_count is not None and (
            isinstance(stage_count, bool) or not isinstance(stage_count, int) or stage_count < 1
        ):
            invalid["compound_action_stage_count"] += 1
        if isinstance(stage_index, int) and isinstance(stage_count, int) and stage_index >= stage_count:
            invalid["compound_action_stage_bounds"] += 1
        complete = row.get("compound_action_complete")
        if complete is not None and complete is not True:
            invalid["compound_action_complete"] += 1
        canonical = row.get("canonical_legal_action_set")
        if canonical is not None:
            if not isinstance(canonical, Mapping):
                invalid["canonical_legal_action_set"] += 1
            else:
                stages = canonical.get("stages")
                action_count = canonical.get("complete_action_count")
                if not isinstance(canonical.get("legal_action_set_id"), str):
                    invalid["canonical_legal_action_set_id"] += 1
                if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)) or not stages:
                    invalid["canonical_legal_action_stages"] += 1
                if (
                    isinstance(action_count, bool)
                    or not isinstance(action_count, int)
                    or action_count < 1
                ):
                    invalid["canonical_complete_action_count"] += 1
                if canonical.get("failures") not in (None, [], ()):
                    invalid["canonical_legal_action_failures"] += 1

    for group_rows in groups.values():
        stage_counts = {row.get("compound_action_stage_count") for row in group_rows}
        stage_indices = {row.get("compound_action_stage_index") for row in group_rows}
        if len(stage_counts) != 1 or not all(isinstance(value, int) for value in stage_counts):
            invalid["compound_action_group_stage_count"] += 1
            continue
        stage_count = int(next(iter(stage_counts)))
        if stage_indices != set(range(stage_count)) or len(group_rows) != stage_count:
            invalid["compound_action_group_stage_coverage"] += 1
        legal_set_ids = {
            row.get("canonical_legal_action_set", {}).get("legal_action_set_id")
            for row in group_rows
            if isinstance(row.get("canonical_legal_action_set"), Mapping)
        }
        if len(legal_set_ids) != 1:
            invalid["compound_action_group_legal_set_identity"] += 1

    passed = bool(rows) and not missing and not invalid

    return {
        "schema_version": COLLECTOR_CAPABILITY_SCHEMA_VERSION,
        "status": "PASS" if passed else "BLOCKED_MISSING_COMPOUND_ACTION_CONTRACT",
        "passed": passed,
        "rows_scanned": len(rows),
        "missing_field_counts": dict(sorted(missing.items())),
        "invalid_field_counts": dict(sorted(invalid.items())),
        "non_substitutable_existing_fields": {
            "decision_id": "one live select prompt only",
            "response_span_id": "opponent response span, not a compound-action boundary",
        },
        "required_semantics": {
            "compound_action_group_id": "durable across every conditional stage",
            "canonical_legal_action_set": (
                "complete conditional option tree, including counterfactual sibling paths, "
                "or a proof that the action has exactly one stage"
            ),
        },
    }


__all__ = [
    "COLLECTOR_CAPABILITY_SCHEMA_VERSION",
    "REQUIRED_COMPOUND_ACTION_FIELDS",
    "audit_collector_capability",
]
