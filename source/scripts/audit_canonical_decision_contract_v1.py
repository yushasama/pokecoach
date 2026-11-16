"""Run the resumable read-only Admission V0 semantic audit."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import time
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.replay.contract import CANONICAL_DECISION_CONTRACT_V1
from deepmew.replay.contract.causal_alignment import (
    CAUSAL_ALIGNER_VERSION,
    KAGGLE_ALIGNMENT_RULE_ID,
    KAGGLE_SOURCE_SCOPE,
    KaggleEpisodeRepository,
    align_decision,
    build_admission_transition_registry,
)
from deepmew.replay.contract.diagnostics import (
    AUDIT_DIAGNOSTICS_SCHEMA_VERSION,
    AuditDiagnostics,
)
from deepmew.replay.contract.identities import (
    canonical_json_bytes,
    identity,
    implementation_sha256,
    sha256_json,
)
from deepmew.replay.contract.schemas import (
    VALIDATOR_VERSION,
    CanonicalDecisionImplementationHashes,
    ValidationOutcome,
)
from deepmew.replay.contract.source_events import source_event_from_admission
from deepmew.replay.contract.validation import canonicalize_decision


AUDIT_SCHEMA_VERSION = "deepmew.full_canonical_decision_audit.v3"
AUDIT_PART_SCHEMA_VERSION = "deepmew.full_canonical_decision_audit_part.v3"
AUDIT_RECEIPT_SCHEMA_VERSION = "deepmew.full_canonical_decision_audit_receipt.v3"
AUDIT_IDENTITY_SCHEMA_VERSION = "deepmew.full_canonical_decision_audit_identity.v3"
BINDING_REGISTRY_SCHEMA_VERSION = "deepmew.full_canonical_decision_binding_registry.v1"
PART_REGISTRY_SCHEMA_VERSION = "deepmew.full_canonical_decision_part_registry.v1"
SEMANTIC_DEPENDENCY_SCHEMA_VERSION = "deepmew.canonical_decision_semantic_dependencies.v1"
RAW_EPISODE_REFERENCE_INVENTORY_SCHEMA_VERSION = (
    "deepmew.kaggle_raw_episode_reference_inventory.v1"
)
RESUME_POLICY = "fresh_exact_recompute"
MAX_RAW_EPISODE_REFERENCE_SAMPLES = 5
GLOBAL_CANDIDATE_ROW_CONVENTION = (
    "deepmew.admission_global_candidate_row.binding_cumulative_start_plus_admitted_ordinal.v2"
)
REPAIR_RULE_REGISTRY_ID = (
    "CENTRALIZED_CAUSAL_REPLAY_CONTRACT_AND_DERIVED_V1_KAGGLE_FORENSICS_20260714"
)
REPAIR_RULE_REGISTRY_PATH = (
    ROOT
    / ".research"
    / "centralized_causal_replay_contract_and_derived_v1_20260714"
    / "DETERMINISTIC_REPAIR_RULES.json"
)
KAGGLE_REGISTERED_SOURCE_SCOPE = {
    "source_format": "kaggle_episode_replay",
    "replay_schema_version": 1,
    "environment_name": "cabt",
    "environment_version": "1.0.0",
    "module_version": "1.30.2",
}

DEFAULT_ADMISSION = ROOT / ".research" / "terminal_league_starmie_grimmsnarl_v0" / (
    "e1_starmie_complete_enhanced_generation_20260714"
) / "production_admission"
DEFAULT_OUTPUT = ROOT / ".research" / "centralized_causal_replay_contract_and_derived_v1_20260714"

_WORKER_CONNECTION: sqlite3.Connection | None = None
_WORKER_REPOSITORY: KaggleEpisodeRepository | None = None
_WORKER_ADMISSION_IDENTITY = ""
_WORKER_KAGGLE_HASHES: frozenset[str] = frozenset()
_WORKER_CONTRACT_SHA256 = ""
_WORKER_AUDIT_BINDING: dict[str, Any] = {}
_WORKER_IMPLEMENTATION_HASHES: CanonicalDecisionImplementationHashes | None = None

ACTION_SCHEMA_IMPLEMENTATION_PATHS = (
    "deepmew/replay/contract/schemas.py",
    "deepmew/replay/contract/legal_actions.py",
    "deepmew/replay/contract/chosen_actions.py",
)
IMPLEMENTATION_HASH_MANIFEST_SCHEMA_VERSION = (
    "deepmew.canonical_decision_implementation_hashes.v1"
)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _ordered_file_set_sha256(root: Path, relative_paths: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for relative_path in relative_paths:
        normalized = Path(relative_path).as_posix()
        path = (root / relative_path).resolve(strict=True)
        digest.update(normalized.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    return digest.hexdigest()


def _implementation_hash_manifest(root: Path = ROOT) -> dict[str, Any]:
    causal_path = root / "deepmew" / "replay" / "contract" / "causal_alignment.py"
    validator_path = root / "deepmew" / "replay" / "contract" / "validation.py"
    payload = {
        "schema_version": IMPLEMENTATION_HASH_MANIFEST_SCHEMA_VERSION,
        "causal_aligner": {
            "path": causal_path.relative_to(root).as_posix(),
            "sha256": _sha256_path(causal_path),
        },
        "legality_validator": {
            "path": validator_path.relative_to(root).as_posix(),
            "sha256": _sha256_path(validator_path),
        },
        "action_schema": {
            "ordered_paths": list(ACTION_SCHEMA_IMPLEMENTATION_PATHS),
            "sha256": _ordered_file_set_sha256(
                root,
                ACTION_SCHEMA_IMPLEMENTATION_PATHS,
            ),
            "hash_formula": "sha256(path_utf8+NUL+raw_bytes+NUL for ordered_paths)",
        },
    }
    payload["manifest_sha256"] = sha256_json(payload)

    return payload


def _typed_implementation_hashes(
    manifest: Mapping[str, Any],
) -> CanonicalDecisionImplementationHashes:
    return CanonicalDecisionImplementationHashes(
        causal_aligner_implementation_sha256=_require_sha256(
            manifest["causal_aligner"]["sha256"],
            "causal_aligner_implementation_sha256",
        ),
        legality_validator_implementation_sha256=_require_sha256(
            manifest["legality_validator"]["sha256"],
            "legality_validator_implementation_sha256",
        ),
        action_schema_implementation_sha256=_require_sha256(
            manifest["action_schema"]["sha256"],
            "action_schema_implementation_sha256",
        ),
    )


def _require_sha256(value: Any, field: str) -> str:
    normalized = str(value or "").lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")

    return normalized


def _repair_rule_registry_manifest(
    path: Path = REPAIR_RULE_REGISTRY_PATH,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    resolved_root = root.resolve(strict=True)
    raw = resolved.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("deterministic repair registry root must be an object")
    if payload.get("registry_id") != REPAIR_RULE_REGISTRY_ID:
        raise ValueError("deterministic repair registry ID differs from the frozen ID")
    rules = payload.get("rules")
    matching = [
        rule
        for rule in rules if isinstance(rule, Mapping)
        and rule.get("rule_id") == KAGGLE_ALIGNMENT_RULE_ID
    ] if isinstance(rules, list) else []
    if len(matching) != 1:
        raise ValueError("frozen Kaggle causal rule is absent or duplicated")
    rule = matching[0]
    expected_code_scope = (
        "kaggle_episode_replay schema=1 environment=cabt "
        "environment_version=1.0.0 module_version=1.30.2"
    )
    if (
        rule.get("rule_version") != 1
        or rule.get("source_scope") != KAGGLE_REGISTERED_SOURCE_SCOPE
        or KAGGLE_SOURCE_SCOPE != expected_code_scope
    ):
        raise ValueError("code and frozen Kaggle causal-rule scope disagree")
    manifest = {
        "path": resolved.relative_to(resolved_root).as_posix(),
        "bytes": len(raw),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "canonical_payload_sha256": sha256_json(payload),
        "registry_id": REPAIR_RULE_REGISTRY_ID,
        "kaggle_causal_rule_id": KAGGLE_ALIGNMENT_RULE_ID,
        "kaggle_causal_rule_sha256": sha256_json(rule),
    }
    manifest["manifest_sha256"] = sha256_json(manifest)

    return manifest


def _semantic_dependency_manifest(root: Path = ROOT) -> dict[str, Any]:
    contract_dir = root / "deepmew" / "replay" / "contract"
    paths = sorted(
        [
            *contract_dir.glob("*.py"),
            root / "deepmew" / "replay" / "information_boundary.py",
            REPAIR_RULE_REGISTRY_PATH,
        ],
        key=lambda path: path.relative_to(root).as_posix(),
    )
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_path(path),
        }
        for path in paths
    ]
    payload: dict[str, Any] = {
        "schema_version": SEMANTIC_DEPENDENCY_SCHEMA_VERSION,
        "files": entries,
    }
    payload["semantic_dependency_sha256"] = sha256_json(payload)

    return payload


def _binding_registry_payload(
    bindings: Iterable[tuple[str, int]],
    *,
    actual_transition_count: int,
) -> dict[str, Any]:
    entries = []
    row_start = 0
    for binding_id, rows in bindings:
        entries.append({
            "binding_id": binding_id,
            "expected_rows": int(rows),
            "global_candidate_row_start": row_start,
        })
        row_start += int(rows)
    payload: dict[str, Any] = {
        "schema_version": BINDING_REGISTRY_SCHEMA_VERSION,
        "global_candidate_row_convention": GLOBAL_CANDIDATE_ROW_CONVENTION,
        "actual_transition_count": int(actual_transition_count),
        "bindings": entries,
    }
    payload["binding_registry_sha256"] = sha256_json(payload)

    return payload


def _assert_completed_transition_binding_coverage(
    bindings: Iterable[tuple[str, int]],
    *,
    actual_binding_ids: tuple[str, ...],
    expected_candidates: int,
) -> None:
    """Match table membership to nonempty completions while retaining zero rows."""

    frozen = tuple(bindings)
    identities = tuple(binding_id for binding_id, _ in frozen)
    if len(set(identities)) != len(identities):
        raise ValueError("Admission V0 completed transition bindings are duplicated")
    if any(rows < 0 for _, rows in frozen):
        raise ValueError("Admission V0 completed transition rows are negative")
    if sum(rows for _, rows in frozen) != expected_candidates:
        raise ValueError("Admission V0 binding rows do not cover the exact receipt")
    expected_nonempty_ids = tuple(
        binding_id
        for binding_id, rows in frozen
        if rows > 0
    )
    if actual_binding_ids != expected_nonempty_ids:
        raise ValueError(
            "Admission V0 nonempty transition binding set differs from completed inputs"
        )


def _global_candidate_row(binding_row_start: int, admitted_ordinal: int) -> int:
    if binding_row_start < 0 or admitted_ordinal < 0:
        raise ValueError("Admission-global candidate coordinates must be non-negative")

    return binding_row_start + admitted_ordinal


def _build_audit_binding(
    *,
    contract_sha256: str,
    semantic_dependency_sha256: str,
    auditor_implementation_sha256: str,
    admission_manifest_sha256: str,
    admission_receipt_sha256: str,
    admission_identity: str,
    admission_database_expected_sha256: str,
    admission_database_actual_before_sha256: str,
    admission_database_expected_bytes: int,
    admission_binding_set_sha256: str,
    binding_registry_sha256: str,
    repair_rule_registry_file_sha256: str,
    repair_rule_registry_payload_sha256: str,
    repair_rule_registry_manifest_sha256: str,
    repair_rule_registry_id: str,
    causal_aligner_implementation_sha256: str,
    legality_validator_implementation_sha256: str,
    action_schema_implementation_sha256: str,
    implementation_hash_manifest_sha256: str,
    expected_candidates: int,
) -> dict[str, Any]:
    payload = {
        "audit_identity_schema_version": AUDIT_IDENTITY_SCHEMA_VERSION,
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "audit_part_schema_version": AUDIT_PART_SCHEMA_VERSION,
        "contract_version": CANONICAL_DECISION_CONTRACT_V1,
        "contract_implementation_sha256": _require_sha256(
            contract_sha256,
            "contract_implementation_sha256",
        ),
        "semantic_dependency_sha256": _require_sha256(
            semantic_dependency_sha256,
            "semantic_dependency_sha256",
        ),
        "auditor_implementation_sha256": _require_sha256(
            auditor_implementation_sha256,
            "auditor_implementation_sha256",
        ),
        "admission_manifest_sha256": _require_sha256(
            admission_manifest_sha256,
            "admission_manifest_sha256",
        ),
        "admission_receipt_sha256": _require_sha256(
            admission_receipt_sha256,
            "admission_receipt_sha256",
        ),
        "admission_identity": _require_sha256(admission_identity, "admission_identity"),
        "admission_database_expected_sha256": _require_sha256(
            admission_database_expected_sha256,
            "admission_database_expected_sha256",
        ),
        "admission_database_actual_before_sha256": _require_sha256(
            admission_database_actual_before_sha256,
            "admission_database_actual_before_sha256",
        ),
        "admission_database_expected_bytes": int(admission_database_expected_bytes),
        "admission_binding_set_sha256": _require_sha256(
            admission_binding_set_sha256,
            "admission_binding_set_sha256",
        ),
        "binding_registry_sha256": _require_sha256(
            binding_registry_sha256,
            "binding_registry_sha256",
        ),
        "repair_rule_registry_file_sha256": _require_sha256(
            repair_rule_registry_file_sha256,
            "repair_rule_registry_file_sha256",
        ),
        "repair_rule_registry_payload_sha256": _require_sha256(
            repair_rule_registry_payload_sha256,
            "repair_rule_registry_payload_sha256",
        ),
        "repair_rule_registry_manifest_sha256": _require_sha256(
            repair_rule_registry_manifest_sha256,
            "repair_rule_registry_manifest_sha256",
        ),
        "repair_rule_registry_id": str(repair_rule_registry_id),
        "causal_aligner_implementation_sha256": _require_sha256(
            causal_aligner_implementation_sha256,
            "causal_aligner_implementation_sha256",
        ),
        "legality_validator_implementation_sha256": _require_sha256(
            legality_validator_implementation_sha256,
            "legality_validator_implementation_sha256",
        ),
        "action_schema_implementation_sha256": _require_sha256(
            action_schema_implementation_sha256,
            "action_schema_implementation_sha256",
        ),
        "implementation_hash_manifest_sha256": _require_sha256(
            implementation_hash_manifest_sha256,
            "implementation_hash_manifest_sha256",
        ),
        "resume_policy": RESUME_POLICY,
        "global_candidate_row_convention": GLOBAL_CANDIDATE_ROW_CONVENTION,
        "expected_candidates": int(expected_candidates),
    }
    if payload["admission_database_expected_bytes"] < 1 or payload["expected_candidates"] < 1:
        raise ValueError("audit identity population and database size must be positive")
    if payload["repair_rule_registry_id"] != REPAIR_RULE_REGISTRY_ID:
        raise ValueError("audit identity repair registry ID differs from the frozen ID")
    if (
        payload["admission_database_actual_before_sha256"]
        != payload["admission_database_expected_sha256"]
    ):
        raise ValueError("actual Admission database SHA-256 differs from the frozen receipt")
    payload["audit_identity"] = identity("fullaudit3", payload)

    return payload


def _counter_total(value: Any, field: str) -> int:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a counter object")
    total = 0
    for key, count in value.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{field}.{key} must be a non-negative integer")
        total += count

    return total


def _nested_total(value: Any, field: str) -> int:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a nested counter object")
    total = 0
    for key, counts in value.items():
        _counter_total(counts, f"{field}.{key}")
        total += int(counts.get("total", 0))

    return total


def _nonzero_counts(values: Mapping[str, Any]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in values.items()
        if int(value) != 0
    }


def _assert_nested_semantic_conservation(
    value: Mapping[str, Any],
    field: str,
    *,
    expected_outcomes: Mapping[str, int],
    expected_choices: Mapping[str, int],
    expected_repair: Mapping[str, int],
) -> None:
    aggregate_outcomes: Counter[str] = Counter()
    aggregate_choices: Counter[str] = Counter()
    aggregate_repair: Counter[str] = Counter()
    allowed_outcomes = {outcome.value for outcome in ValidationOutcome}
    for key, counts in value.items():
        total = int(counts.get("total", 0))
        outcomes = {
            name.removeprefix("outcome:"): int(count)
            for name, count in counts.items()
            if name.startswith("outcome:")
        }
        choices = {
            name.removeprefix("choice:"): int(count)
            for name, count in counts.items()
            if name.startswith("choice:")
        }
        if set(outcomes) - allowed_outcomes:
            raise ValueError(f"{field}.{key} contains an unknown outcome bucket")
        if sum(outcomes.values()) != total:
            raise ValueError(f"{field}.{key} outcome buckets do not conserve total")
        if sum(choices.values()) != total:
            raise ValueError(f"{field}.{key} choice buckets do not conserve total")
        attempted = int(counts.get("repair_attempted", 0))
        succeeded = int(counts.get("repair_succeeded", 0))
        rejected = int(counts.get("repair_rejected", 0))
        deterministic = int(counts.get("deterministic_repair", 0))
        if attempted != succeeded + rejected:
            raise ValueError(f"{field}.{key} repair attempts are not conserved")
        if deterministic != succeeded:
            raise ValueError(f"{field}.{key} deterministic repairs differ from successes")
        if int(counts.get("semantic_mismatch", 0)) > total:
            raise ValueError(f"{field}.{key} semantic mismatches exceed total")
        aggregate_outcomes.update(outcomes)
        aggregate_choices.update(choices)
        aggregate_repair.update({
            "attempted": attempted,
            "succeeded": succeeded,
            "rejected": rejected,
        })
    if _nonzero_counts(aggregate_outcomes) != _nonzero_counts(expected_outcomes):
        raise ValueError(f"{field} outcome buckets differ from global outcomes")
    if _nonzero_counts(aggregate_choices) != _nonzero_counts(expected_choices):
        raise ValueError(f"{field} choice buckets differ from global choices")
    if _nonzero_counts(aggregate_repair) != _nonzero_counts(expected_repair):
        raise ValueError(f"{field} repair buckets differ from global repairs")


def _repair_statistics(
    repair_rules: Mapping[str, Any],
    *,
    verified: bool = True,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, int]] = {}
    signals: dict[str, int] = {}
    for key, raw_count in repair_rules.items():
        count = int(raw_count)
        if count < 0:
            raise ValueError(f"repair_rules.{key} must be non-negative")
        if ":" not in key:
            signals[key] = count
            continue
        rule_id, state = key.rsplit(":", 1)
        if state not in {"attempted", "succeeded", "rejected"}:
            signals[key] = count
            continue
        grouped.setdefault(rule_id, {"attempted": 0, "succeeded": 0, "rejected": 0})[
            state
        ] += count
    rules: dict[str, dict[str, Any]] = {}
    for rule_id, counts in sorted(grouped.items()):
        attempted = counts["attempted"]
        succeeded = counts["succeeded"]
        rejected = counts["rejected"]
        if attempted != succeeded + rejected:
            raise ValueError(f"repair rule {rule_id} attempts are not conserved")
        rules[rule_id] = {
            "attempted": attempted,
            "succeeded": succeeded,
            "rejected": rejected,
            "denominator": attempted,
            "succeeded_proportion": (
                succeeded / attempted if verified and attempted else None
            ),
            "succeeded_percent": (
                100.0 * succeeded / attempted if verified and attempted else None
            ),
            "rejected_proportion": (
                rejected / attempted if verified and attempted else None
            ),
            "rejected_percent": (
                100.0 * rejected / attempted if verified and attempted else None
            ),
            "sampling_uncertainty": (
                "none_full_corpus_exact"
                if verified
                else "not_asserted_verification_blocked"
            ),
        }

    return {"rules": rules, "non_attempt_signals": dict(sorted(signals.items()))}


def _observation_reference_statistics(counts: Mapping[str, Any]) -> dict[str, Any]:
    names = {
        "pre_observation_ineligible": "pre_observation_reference_ineligible",
        "normalized_corrected_kaggle_pre_reused": (
            "corrected_kaggle_pre_reference_hash_present_in_admission"
        ),
        "normalized_corrected_kaggle_pre_new": (
            "corrected_kaggle_pre_reference_hash_absent_from_admission"
        ),
        "normalized_kaggle_result_reused": "kaggle_result_reference_eligible",
        "normalized_kaggle_result_ineligible": "kaggle_result_reference_ineligible",
        "normalized_admission_pre_reused": "admission_pre_reference_eligible",
        "normalized_admission_pre_new": "admission_pre_reference_not_marked_reusable",
    }
    reference_counts = {
        names.get(key, f"unmapped_reference_counter:{key}"): int(value)
        for key, value in sorted(counts.items())
    }

    return {
        "schema_version": "deepmew.normalized_observation_reference_audit.v1",
        "counts": reference_counts,
        "tensor_registry_status": "not_audited_deferred_to_separate_x_registry",
    }


def _assert_diagnostics_conservation(
    diagnostics: Mapping[str, Any],
    expected_rows: int,
) -> None:
    if diagnostics.get("schema_version") != AUDIT_DIAGNOSTICS_SCHEMA_VERSION:
        raise ValueError("audit diagnostics schema is not frozen")
    total_candidates = diagnostics.get("total_candidates")
    if (
        isinstance(total_candidates, bool)
        or not isinstance(total_candidates, int)
        or total_candidates != expected_rows
    ):
        raise ValueError("diagnostic candidate count is not conserved")
    outcomes = diagnostics.get("outcomes")
    if _counter_total(outcomes, "outcomes") != expected_rows:
        raise ValueError("every candidate must receive exactly one outcome")
    allowed_outcomes = {outcome.value for outcome in ValidationOutcome}
    if set(outcomes) - allowed_outcomes:
        raise ValueError("diagnostics contain an unknown validation outcome")
    exception_count = _counter_total(diagnostics.get("exceptions"), "exceptions")
    unknown_count = int(outcomes.get(ValidationOutcome.UNKNOWN_FAIL_CLOSED.value, 0))
    if exception_count > unknown_count:
        raise ValueError("audit exceptions are not conserved inside fail-closed outcomes")
    classified_rows = expected_rows - exception_count
    classified_outcomes = {
        str(key): int(value)
        for key, value in outcomes.items()
    }
    classified_outcomes[ValidationOutcome.UNKNOWN_FAIL_CLOSED.value] = (
        unknown_count - exception_count
    )
    complete_alignment = diagnostics.get("complete_causal_alignment")
    if (
        isinstance(complete_alignment, bool)
        or not isinstance(complete_alignment, int)
        or complete_alignment < 0
        or complete_alignment > classified_rows
    ):
        raise ValueError("complete causal-alignment count is impossible")
    choice_shapes = diagnostics.get("choice_shapes")
    if _counter_total(choice_shapes, "choice_shapes") != classified_rows:
        raise ValueError("choice-shape counts do not conserve classified rows")
    repair_rules = diagnostics.get("repair_rules")
    _counter_total(repair_rules, "repair_rules")
    repair_statistics = _repair_statistics(repair_rules)
    expected_repair = {
        state: sum(
            int(rule[state])
            for rule in repair_statistics["rules"].values()
        )
        for state in ("attempted", "succeeded", "rejected")
    }
    for field in ("action_families", "source_formats", "source_players"):
        nested = diagnostics.get(field)
        if _nested_total(nested, field) != classified_rows:
            raise ValueError(f"{field} totals do not conserve classified rows")
        _assert_nested_semantic_conservation(
            nested,
            field,
            expected_outcomes=classified_outcomes,
            expected_choices=choice_shapes,
            expected_repair=expected_repair,
        )
    games = diagnostics.get("games")
    if not isinstance(games, Mapping):
        raise ValueError("games must be an object")
    game_counts: Counter[str] = Counter()
    for game_id, game in games.items():
        if not isinstance(game, Mapping):
            raise ValueError(f"games.{game_id} must be an object")
        required_counts = (
            "decisions",
            "complete_alignment",
            "valid_actor_and_critic",
            "forced_critic_only",
            "unresolved_critic_only",
            "rejected",
            "ambiguous",
            "deterministic_repair",
            "repair_attempted",
            "repair_rejected",
            "stale_legacy_legal_set",
            "terminal_rows",
        )
        for field in required_counts:
            value = game.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"games.{game_id}.{field} must be a non-negative integer"
                )
        decisions = int(game["decisions"])
        if game["repair_attempted"] != game["deterministic_repair"] + game["repair_rejected"]:
            raise ValueError(f"games.{game_id} repair attempts are not conserved")
        if (
            game["valid_actor_and_critic"]
            + game["forced_critic_only"]
            + game["unresolved_critic_only"]
            + game["rejected"]
            != decisions
        ):
            raise ValueError(f"games.{game_id} outcome buckets do not conserve decisions")
        if game["ambiguous"] > game["rejected"]:
            raise ValueError(f"games.{game_id} ambiguous rows exceed rejected rows")
        if game["complete_alignment"] > decisions:
            raise ValueError(f"games.{game_id} complete alignments exceed decisions")
        if game["terminal_rows"] > decisions:
            raise ValueError(f"games.{game_id} terminal rows exceed decisions")
        for field in ("global_decision_indices", "source_event_positions"):
            values = game.get(field)
            if not isinstance(values, list) or len(values) > decisions:
                raise ValueError(f"games.{game_id}.{field} is not candidate-bounded")
        actor_indices = game.get("actor_indices")
        if (
            not isinstance(actor_indices, Mapping)
            or any(not isinstance(values, list) for values in actor_indices.values())
            or sum(len(values) for values in actor_indices.values()) > decisions
        ):
            raise ValueError(f"games.{game_id}.actor_indices is not candidate-bounded")
        game_counts.update({field: int(game[field]) for field in required_counts})
    if game_counts["decisions"] != classified_rows:
        raise ValueError("game decision counts do not conserve classified rows")
    expected_game_counts = {
        "complete_alignment": int(complete_alignment),
        "valid_actor_and_critic": int(
            classified_outcomes.get(ValidationOutcome.VALID_ACTOR_AND_CRITIC.value, 0)
        ),
        "forced_critic_only": int(
            classified_outcomes.get(ValidationOutcome.VALID_CRITIC_ONLY_FORCED.value, 0)
        ),
        "unresolved_critic_only": int(
            classified_outcomes.get(
                ValidationOutcome.VALID_CRITIC_ONLY_UNRESOLVED_ACTION.value,
                0,
            )
        ),
        "ambiguous": int(
            classified_outcomes.get(ValidationOutcome.UNKNOWN_FAIL_CLOSED.value, 0)
        ),
        "rejected": sum(
            int(count)
            for outcome, count in classified_outcomes.items()
            if outcome.startswith("REJECTED_")
            or outcome == ValidationOutcome.UNKNOWN_FAIL_CLOSED.value
        ),
        "repair_attempted": expected_repair["attempted"],
        "deterministic_repair": expected_repair["succeeded"],
        "repair_rejected": expected_repair["rejected"],
    }
    for field, expected in expected_game_counts.items():
        if game_counts[field] != expected:
            raise ValueError(f"game {field} differs from global diagnostics")


def _completion_status(
    *,
    inspected_candidates: int,
    expected_candidates: int,
    exception_count: int,
    immutable_ok: bool,
    sha_recomputed: bool,
    frozen_inputs_ok: bool,
    raw_sources_verified: bool,
    raw_reference_coverage_complete: bool,
    counters_conserved: bool,
    fresh_parts_verified: bool,
) -> str:
    complete = (
        inspected_candidates == expected_candidates
        and exception_count == 0
        and immutable_ok
        and sha_recomputed
        and frozen_inputs_ok
        and raw_sources_verified
        and raw_reference_coverage_complete
        and counters_conserved
        and fresh_parts_verified
    )

    return "COMPLETE_READ_ONLY_EXACT" if complete else "BLOCKED_FAIL_CLOSED"


def _verify_raw_episode_sources(
    raw_episode_hashes: Mapping[str, str],
    *,
    repository_root: Path = ROOT,
) -> dict[str, Any]:
    entries = []
    failures = []
    for raw_path, raw_digest in sorted(raw_episode_hashes.items()):
        try:
            expected = _require_sha256(raw_digest, f"raw_episode[{raw_path}]")
        except ValueError as exc:
            failures.append(str(exc))
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            path = repository_root / path
        path = path.resolve()
        if not path.is_file():
            failures.append(
                f"raw Kaggle episode is absent at final verification: {path}"
            )
            continue
        try:
            actual = _sha256_path(path)
        except OSError as exc:
            failures.append(f"raw Kaggle episode could not be rehashed: {path}: {exc}")
            continue
        if actual != expected:
            failures.append(f"raw Kaggle episode changed during audit: {path}")
        entries.append({"path": str(path), "sha256": actual})
    if failures:
        raise ValueError("; ".join(failures[:20]))
    payload: dict[str, Any] = {
        "schema_version": "deepmew.kaggle_raw_episode_final_verification.v1",
        "episodes": entries,
    }
    payload["verification_payload_sha256"] = sha256_json(payload)

    return payload


def _verify_admission_database_before(
    database_path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
    skip_database_sha: bool,
) -> tuple[os.stat_result, str]:
    if skip_database_sha:
        raise ValueError(
            "--skip-database-sha blocks before database scan and V2 part acceptance"
        )
    expected_digest = _require_sha256(
        expected_sha256,
        "admission_database_expected_sha256",
    )
    expected_size = int(expected_bytes)
    before = database_path.stat()
    if before.st_size != expected_size:
        raise ValueError("Admission database size differs from the frozen receipt")
    actual_digest = _sha256_path(database_path)
    after = database_path.stat()
    if before.st_size != after.st_size or after.st_size != expected_size:
        raise ValueError("Admission database size changed during startup verification")
    if actual_digest != expected_digest:
        raise ValueError("Admission database SHA-256 differs from the frozen receipt")

    return after, actual_digest


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_bytes(path, canonical_json_bytes(payload) + b"\n")


def _write_fail_closed_verification_artifacts(
    output_dir: Path,
    *,
    audit_binding: Mapping[str, Any],
    stage: str,
    error: BaseException | str,
    inspected_candidates: int | None = None,
) -> dict[str, Any]:
    message = str(error)[:1000]
    failure = {
        "stage": stage,
        "error_type": type(error).__name__ if isinstance(error, BaseException) else "VerificationError",
        "message": message,
    }
    summary = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "BLOCKED_FAIL_CLOSED",
        "audit_identity": audit_binding.get("audit_identity"),
        "resume_policy": RESUME_POLICY,
        "expected_candidates": audit_binding.get("expected_candidates"),
        "inspected_candidates": inspected_candidates,
        "audit_is_full_corpus": False,
        "verification_failure": failure,
        "sampling_uncertainty": "not_applicable_verification_failed",
    }
    summary_path = output_dir / "FULL_CORPUS_SEMANTIC_AUDIT.json"
    _atomic_json(summary_path, summary)
    report_path = output_dir / "FULL_CORPUS_SEMANTIC_AUDIT_REPORT.md"
    report = "\n".join([
        "# Full Admission V0 Semantic Audit",
        "",
        "Status: `BLOCKED_FAIL_CLOSED`",
        "",
        f"Final verification failed at `{stage}`.",
        "No full-corpus, raw-game-completeness, repair-success, normalized-reference, or tensor-cache claim is valid.",
        f"Error: `{failure['error_type']}: {message}`",
        "",
    ])
    _atomic_bytes(report_path, report.encode("utf-8"))
    receipt = {
        "schema_version": AUDIT_RECEIPT_SCHEMA_VERSION,
        "status": "BLOCKED_FAIL_CLOSED",
        "audit_identity": audit_binding.get("audit_identity"),
        "resume_policy": RESUME_POLICY,
        "admission_database_expected_sha256": audit_binding.get(
            "admission_database_expected_sha256"
        ),
        "admission_database_actual_before_sha256": audit_binding.get(
            "admission_database_actual_before_sha256"
        ),
        "expected_candidates": audit_binding.get("expected_candidates"),
        "inspected_candidates": inspected_candidates,
        "audit_summary_sha256": _sha256_path(summary_path),
        "audit_report_sha256": _sha256_path(report_path),
        "verification_failure": failure,
        "admission_v0_immutable": False,
        "forbidden_work_performed": [],
    }
    receipt["receipt_sha256"] = sha256_json(receipt)
    receipt_path = output_dir / "FULL_CORPUS_SEMANTIC_AUDIT_RECEIPT.json"
    _atomic_json(receipt_path, receipt)

    return receipt


def _readonly_connection(database_path: Path) -> sqlite3.Connection:
    uri = f"file:{database_path.resolve().as_posix()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True, timeout=60.0)
    connection.execute("PRAGMA query_only=ON")

    return connection


def _init_worker(
    database_path: str,
    repository_root: str,
    admission_identity: str,
    kaggle_hashes: frozenset[str],
    contract_sha256: str,
    audit_binding: Mapping[str, Any],
) -> None:
    global _WORKER_CONNECTION
    global _WORKER_REPOSITORY
    global _WORKER_ADMISSION_IDENTITY
    global _WORKER_KAGGLE_HASHES
    global _WORKER_CONTRACT_SHA256
    global _WORKER_AUDIT_BINDING
    global _WORKER_IMPLEMENTATION_HASHES
    _WORKER_CONNECTION = _readonly_connection(Path(database_path))
    _WORKER_REPOSITORY = KaggleEpisodeRepository(repository_root)
    _WORKER_ADMISSION_IDENTITY = admission_identity
    _WORKER_KAGGLE_HASHES = kaggle_hashes
    _WORKER_CONTRACT_SHA256 = contract_sha256
    _WORKER_AUDIT_BINDING = dict(audit_binding)
    _WORKER_IMPLEMENTATION_HASHES = CanonicalDecisionImplementationHashes(
        causal_aligner_implementation_sha256=_require_sha256(
            audit_binding.get("causal_aligner_implementation_sha256"),
            "causal_aligner_implementation_sha256",
        ),
        legality_validator_implementation_sha256=_require_sha256(
            audit_binding.get("legality_validator_implementation_sha256"),
            "legality_validator_implementation_sha256",
        ),
        action_schema_implementation_sha256=_require_sha256(
            audit_binding.get("action_schema_implementation_sha256"),
            "action_schema_implementation_sha256",
        ),
    )


def _source_record(
    connection: sqlite3.Connection,
    transition: Mapping[str, Any],
) -> Mapping[str, Any]:
    state_before = transition.get("state_before")
    reference = state_before.get("source_observation") if isinstance(state_before, Mapping) else None
    if not isinstance(reference, Mapping):
        raise ValueError("transition source-observation reference is absent")
    source_sha256 = str(transition.get("source_hash") or "").lower()
    source_line = int(reference["source_line"])
    content_sha256 = str(reference["content_sha256"]).lower()
    row = connection.execute(
        "SELECT compressed_json FROM observations "
        "WHERE source_sha256=? AND source_line=? AND content_sha256=?",
        (source_sha256, source_line, content_sha256),
    ).fetchone()
    if row is None:
        raise ValueError("Admission V0 observation binding is absent")
    value = json.loads(zlib.decompress(row[0]))
    if not isinstance(value, Mapping):
        raise ValueError("Admission V0 source record is not an object")

    return value


def _transition_reference(
    transition: Mapping[str, Any],
) -> tuple[str, int, str]:
    state_before = transition.get("state_before")
    reference = state_before.get("source_observation") if isinstance(state_before, Mapping) else None
    if not isinstance(reference, Mapping):
        raise ValueError("transition source-observation reference is absent")

    return (
        str(transition.get("source_hash") or "").lower(),
        int(reference["source_line"]),
        str(reference["content_sha256"]).lower(),
    )


def _line_ranges(values: Iterable[int]) -> list[tuple[int, int]]:
    ordered = sorted(set(values))
    if not ordered:
        return []
    ranges = []
    start = prior = ordered[0]
    for value in ordered[1:]:
        if value - prior > 16 or value - start >= 2048:
            ranges.append((start, prior))
            start = value
        prior = value
    ranges.append((start, prior))

    return ranges


def _batch_source_records(
    connection: sqlite3.Connection,
    transitions: list[tuple[int, Mapping[str, Any]]],
) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    requested = {_transition_reference(transition) for _, transition in transitions}
    by_source: dict[str, list[int]] = defaultdict(list)
    for source_sha256, source_line, _ in requested:
        by_source[source_sha256].append(source_line)
    compressed: dict[tuple[str, int, str], bytes] = {}
    for source_sha256, lines in by_source.items():
        for first, last in _line_ranges(lines):
            for source_line, content_sha256, blob in connection.execute(
                "SELECT source_line,content_sha256,compressed_json FROM observations "
                "WHERE source_sha256=? AND source_line BETWEEN ? AND ? "
                "ORDER BY source_line,content_sha256",
                (source_sha256, first, last),
            ):
                key = (source_sha256, int(source_line), str(content_sha256))
                if key in requested:
                    compressed[key] = blob
    if set(compressed) != requested:
        raise ValueError("Admission V0 batch observation coverage is incomplete")

    return {
        key: json.loads(zlib.decompress(blob))
        for key, blob in compressed.items()
    }


def _binding_candidates(
    connection: sqlite3.Connection,
    binding_id: str,
    *,
    batch_rows: int = 512,
) -> Iterable[tuple[int, int, Mapping[str, Any], Mapping[str, Any]]]:
    cursor = connection.execute(
        "SELECT local_row,compressed_json FROM transitions "
        "WHERE binding_id=? ORDER BY local_row",
        (binding_id,),
    )
    batch: list[tuple[int, Mapping[str, Any]]] = []
    admitted_ordinal = 0
    for local_row, compressed_transition in cursor:
        transition = json.loads(zlib.decompress(compressed_transition))
        if not isinstance(transition, Mapping):
            raise ValueError("Admission V0 transition is not an object")
        batch.append((int(local_row), transition))
        if len(batch) < batch_rows:
            continue
        source_records = _batch_source_records(connection, batch)
        for row, item in batch:
            yield (
                admitted_ordinal,
                row,
                item,
                source_records[_transition_reference(item)],
            )
            admitted_ordinal += 1
        batch.clear()
    if batch:
        source_records = _batch_source_records(connection, batch)
        for row, item in batch:
            yield (
                admitted_ordinal,
                row,
                item,
                source_records[_transition_reference(item)],
            )
            admitted_ordinal += 1


class RawEpisodeReferenceConflict(ValueError):
    """A raw Kaggle path was bound to incompatible bytes or a declared digest."""


def _bind_raw_episode_reference(
    source_event: Any,
    repository: KaggleEpisodeRepository,
) -> tuple[str, str] | None:
    if source_event.source_format != "kaggle_replay":
        return None
    source_path = source_event.source_record.get("source_path")
    if not isinstance(source_path, str) or not source_path.strip():
        raise ValueError("Kaggle source_event has no raw episode source_path")
    episode = repository.load(source_path)
    canonical_path = str(episode.path.resolve())
    computed_sha256 = _require_sha256(
        episode.sha256,
        "raw Kaggle computed episode SHA-256",
    )
    declared_sha256 = source_event.source_artifact_sha256
    if declared_sha256 is not None:
        declared = _require_sha256(
            declared_sha256,
            "raw Kaggle declared episode SHA-256",
        )
        if declared != computed_sha256:
            raise RawEpisodeReferenceConflict(
                "raw Kaggle episode declared SHA-256 differs from loaded bytes"
            )

    return canonical_path, computed_sha256


def _raw_episode_reference_inventory(
    *,
    candidate_reference_count: int,
    bound_reference_count: int,
    unresolved_reasons: Mapping[str, int],
    unresolved_samples: Iterable[Mapping[str, Any]],
    raw_episode_hashes: Mapping[str, str],
) -> dict[str, Any]:
    unresolved_reference_count = int(candidate_reference_count) - int(
        bound_reference_count
    )

    return {
        "schema_version": RAW_EPISODE_REFERENCE_INVENTORY_SCHEMA_VERSION,
        "candidate_reference_count": int(candidate_reference_count),
        "bound_reference_count": int(bound_reference_count),
        "unresolved_reference_count": unresolved_reference_count,
        "unique_bound_episode_count": len(raw_episode_hashes),
        "unresolved_reasons": {
            str(key): int(value)
            for key, value in sorted(unresolved_reasons.items())
        },
        "unresolved_reference_samples": [dict(sample) for sample in unresolved_samples],
    }


def _assert_raw_episode_reference_conservation(
    inventory: Mapping[str, Any],
    raw_episode_hashes: Mapping[str, Any],
    expected_rows: int,
) -> None:
    if inventory.get("schema_version") != RAW_EPISODE_REFERENCE_INVENTORY_SCHEMA_VERSION:
        raise ValueError("raw Kaggle reference inventory schema is not frozen")
    counts = {}
    for field in (
        "candidate_reference_count",
        "bound_reference_count",
        "unresolved_reference_count",
        "unique_bound_episode_count",
    ):
        value = inventory.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"raw Kaggle reference inventory {field} is invalid")
        counts[field] = value
    if counts["candidate_reference_count"] > expected_rows:
        raise ValueError("raw Kaggle candidate references exceed audit rows")
    if (
        counts["candidate_reference_count"]
        != counts["bound_reference_count"] + counts["unresolved_reference_count"]
    ):
        raise ValueError("raw Kaggle candidate references are not conserved")
    if not isinstance(raw_episode_hashes, Mapping):
        raise ValueError("raw Kaggle episode hash registry must be an object")
    for path, digest in raw_episode_hashes.items():
        if not isinstance(path, str) or not path:
            raise ValueError("raw Kaggle episode registry contains an invalid path")
        _require_sha256(digest, f"raw Kaggle episode registry[{path}]")
    if counts["unique_bound_episode_count"] != len(raw_episode_hashes):
        raise ValueError("raw Kaggle unique episode count differs from its registry")
    if counts["unique_bound_episode_count"] > counts["bound_reference_count"]:
        raise ValueError("raw Kaggle unique episodes exceed bound references")
    if counts["bound_reference_count"] > 0 and not raw_episode_hashes:
        raise ValueError("raw Kaggle bound references have an empty episode registry")
    unresolved_reasons = inventory.get("unresolved_reasons")
    if (
        _counter_total(unresolved_reasons, "raw Kaggle unresolved reasons")
        != counts["unresolved_reference_count"]
    ):
        raise ValueError("raw Kaggle unresolved reasons are not conserved")
    samples = inventory.get("unresolved_reference_samples")
    expected_sample_count = min(
        counts["unresolved_reference_count"],
        MAX_RAW_EPISODE_REFERENCE_SAMPLES,
    )
    if (
        not isinstance(samples, list)
        or len(samples) != expected_sample_count
        or any(not isinstance(sample, Mapping) for sample in samples)
    ):
        raise ValueError("raw Kaggle unresolved reference samples are not conserved")


def _raw_episode_reference_coverage_complete(
    inventory: Mapping[str, Any],
    raw_episode_hashes: Mapping[str, Any],
    expected_rows: int,
) -> bool:
    _assert_raw_episode_reference_conservation(
        inventory,
        raw_episode_hashes,
        expected_rows,
    )

    return bool(
        inventory["candidate_reference_count"] == inventory["bound_reference_count"]
        and inventory["unresolved_reference_count"] == 0
        and (
            inventory["candidate_reference_count"] == 0
            or inventory["unique_bound_episode_count"] > 0
        )
    )


def _bound_raw_episode_identity(aligned: Any) -> tuple[str, str] | None:
    event = aligned.pre_action_event
    if event.source_format != "kaggle_episode_replay":
        return None
    asset = event.observation_asset
    if not aligned.proof.source_hash_bound or asset is None:
        raise ValueError("raw Kaggle episode event is not hash-bound")
    source_path = str(event.source_artifact_path or "")
    artifact_sha256 = _require_sha256(
        event.source_artifact_sha256,
        "raw Kaggle source_artifact_sha256",
    )
    if not source_path or asset.source_path != source_path:
        raise ValueError("raw Kaggle episode asset path differs from its source artifact")
    if str(asset.source_sha256).lower() != artifact_sha256:
        raise ValueError("raw Kaggle episode asset SHA-256 differs from its source artifact")

    return source_path, artifact_sha256


def _binding_transition_registry(
    connection: sqlite3.Connection,
    *,
    binding_id: str,
    binding_row_start: int,
    expected_rows: int,
):
    def events():
        for admitted_ordinal, local_row, transition, source_record in _binding_candidates(
            connection,
            binding_id,
        ):
            yield source_event_from_admission(
                admission_identity=_WORKER_ADMISSION_IDENTITY,
                binding_id=binding_id,
                local_row=local_row,
                global_row=_global_candidate_row(
                    binding_row_start,
                    admitted_ordinal,
                ),
                transition=transition,
                source_record=source_record,
                verify_source_content=False,
            )

    registry = build_admission_transition_registry(
        events(),
        admission_identity=_WORKER_ADMISSION_IDENTITY,
        binding_id=binding_id,
        expected_rows=expected_rows,
    )
    _require_sha256(
        registry.registry_sha256,
        "admission_transition_registry_sha256",
    )

    return registry


def _audit_binding(job: tuple[str, int, int]) -> dict[str, Any]:
    if (
        _WORKER_CONNECTION is None
        or _WORKER_REPOSITORY is None
        or _WORKER_IMPLEMENTATION_HASHES is None
    ):
        raise RuntimeError("audit worker was not initialized")
    binding_id, binding_row_start, expected_rows = job
    transition_registry = _binding_transition_registry(
        _WORKER_CONNECTION,
        binding_id=binding_id,
        binding_row_start=binding_row_start,
        expected_rows=expected_rows,
    )
    diagnostics = AuditDiagnostics()
    raw_episode_hashes: dict[str, str] = {}
    raw_candidate_reference_count = 0
    raw_bound_reference_count = 0
    raw_unresolved_reasons: Counter[str] = Counter()
    raw_unresolved_samples: list[dict[str, Any]] = []
    size_sample_bytes = 0
    size_sample_count = 0
    started = time.perf_counter()
    for admitted_ordinal, local_row, transition, source_record in _binding_candidates(
        _WORKER_CONNECTION,
        binding_id,
    ):
        try:
            global_candidate_row = _global_candidate_row(
                binding_row_start,
                admitted_ordinal,
            )
            source_event = source_event_from_admission(
                admission_identity=_WORKER_ADMISSION_IDENTITY,
                binding_id=binding_id,
                local_row=int(local_row),
                # SourceEvent retains the historical field name.  This value is
                # an Admission-global candidate coordinate, never a tensor-cache row.
                global_row=global_candidate_row,
                transition=transition,
                source_record=source_record,
                # The immutable Admission V0 receipt already proved this exact
                # key/content binding.  The audit reopens that database read-only.
                verify_source_content=False,
            )
            prebound_raw_episode: tuple[str, str] | None = None
            if source_event.source_format == "kaggle_replay":
                raw_candidate_reference_count += 1
                try:
                    prebound_raw_episode = _bind_raw_episode_reference(
                        source_event,
                        _WORKER_REPOSITORY,
                    )
                except RawEpisodeReferenceConflict:
                    raise
                except Exception as raw_exc:
                    raw_unresolved_reasons[type(raw_exc).__name__] += 1
                    if (
                        len(raw_unresolved_samples)
                        < MAX_RAW_EPISODE_REFERENCE_SAMPLES
                    ):
                        raw_unresolved_samples.append({
                            "binding_id": binding_id,
                            "local_row": int(local_row),
                            "source_path": source_event.source_record.get(
                                "source_path"
                            ),
                            "error_type": type(raw_exc).__name__,
                            "message": str(raw_exc)[:400],
                        })
                else:
                    if prebound_raw_episode is None:
                        raise RawEpisodeReferenceConflict(
                            "Kaggle candidate produced no raw episode binding"
                        )
                    raw_path, raw_sha256 = prebound_raw_episode
                    prior = raw_episode_hashes.get(raw_path)
                    if prior is not None and prior != raw_sha256:
                        raise RawEpisodeReferenceConflict(
                            "one Kaggle episode path has conflicting hashes"
                        )
                    raw_episode_hashes[raw_path] = raw_sha256
                    raw_bound_reference_count += 1
            aligned = align_decision(
                source_event,
                kaggle_repository=_WORKER_REPOSITORY,
                transition_registry=transition_registry,
            )
            record, legal_actions, chosen = canonicalize_decision(
                aligned,
                implementation_hashes=_WORKER_IMPLEMENTATION_HASHES,
            )
            raw_episode_identity = _bound_raw_episode_identity(aligned)
            if raw_episode_identity is not None:
                if prebound_raw_episode != raw_episode_identity:
                    raise RawEpisodeReferenceConflict(
                        "repaired Kaggle event differs from its pre-alignment raw binding"
                    )
            corrected_reused = bool(
                aligned.pre_action_event.observation_asset
                and aligned.pre_action_event.observation_asset.source_content_sha256
                in _WORKER_KAGGLE_HASHES
            )
            diagnostics.observe(
                source_event=source_event,
                aligned=aligned,
                record=record,
                legal_actions=legal_actions,
                chosen=chosen,
                corrected_pre_observation_reused=corrected_reused,
            )
            if size_sample_count < 25:
                size_sample_bytes += len(canonical_json_bytes(record))
                size_sample_count += 1
        except RawEpisodeReferenceConflict:
            raise
        except Exception as exc:  # Every candidate receives an explicit fail-closed class.
            diagnostics.observe_exception(
                binding_id=binding_id,
                local_row=int(local_row),
                exception=exc,
            )
    elapsed = time.perf_counter() - started
    diagnostics_payload = diagnostics.to_payload()
    _assert_diagnostics_conservation(diagnostics_payload, expected_rows)
    raw_reference_inventory = _raw_episode_reference_inventory(
        candidate_reference_count=raw_candidate_reference_count,
        bound_reference_count=raw_bound_reference_count,
        unresolved_reasons=raw_unresolved_reasons,
        unresolved_samples=raw_unresolved_samples,
        raw_episode_hashes=raw_episode_hashes,
    )
    _assert_raw_episode_reference_conservation(
        raw_reference_inventory,
        raw_episode_hashes,
        expected_rows,
    )
    body: dict[str, Any] = {
        "schema_version": AUDIT_PART_SCHEMA_VERSION,
        **_WORKER_AUDIT_BINDING,
        "binding_id": binding_id,
        "global_candidate_row_start": binding_row_start,
        "expected_rows": expected_rows,
        "inspected_rows": diagnostics.total_candidates,
        "elapsed_seconds": elapsed,
        "diagnostics": diagnostics_payload,
        "raw_episode_hashes": dict(sorted(raw_episode_hashes.items())),
        "raw_episode_reference_inventory": raw_reference_inventory,
        "admission_transition_registry": dict(transition_registry.summary()),
        "record_size_sample": {
            "count": size_sample_count,
            "canonical_json_bytes": size_sample_bytes,
        },
    }
    body["part_payload_sha256"] = sha256_json(body)

    return body


def _part_path(parts_dir: Path, binding_id: str) -> Path:
    name = hashlib.sha256(binding_id.encode("utf-8")).hexdigest()

    return parts_dir / f"{name}.json.gz"


def _validate_part_payload(
    payload: Mapping[str, Any],
    *,
    binding_id: str,
    binding_row_start: int,
    expected_rows: int,
    audit_binding: Mapping[str, Any],
) -> None:
    body = dict(payload)
    provided = body.pop("part_payload_sha256", None)
    valid = (
        provided == sha256_json(body)
        and body.get("schema_version") == AUDIT_PART_SCHEMA_VERSION
        and all(body.get(key) == value for key, value in audit_binding.items())
        and body.get("binding_id") == binding_id
        and body.get("global_candidate_row_start") == binding_row_start
        and body.get("expected_rows") == expected_rows
        and body.get("inspected_rows") == expected_rows
    )
    if not valid:
        raise ValueError("audit part identity or exact coverage failed")
    diagnostics = body.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("audit part diagnostics are absent")
    _assert_diagnostics_conservation(diagnostics, expected_rows)
    raw_episode_hashes = body.get("raw_episode_hashes")
    raw_reference_inventory = body.get("raw_episode_reference_inventory")
    if not isinstance(raw_reference_inventory, Mapping):
        raise ValueError("audit part raw Kaggle reference inventory is absent")
    _assert_raw_episode_reference_conservation(
        raw_reference_inventory,
        raw_episode_hashes,
        expected_rows,
    )
    transition_registry = body.get("admission_transition_registry")
    if not isinstance(transition_registry, Mapping):
        raise ValueError("audit part transition registry is absent")
    registry_valid = (
        transition_registry.get("schema_version")
        == "deepmew.admission_transition_registry.v1"
        and transition_registry.get("admission_identity")
        == audit_binding.get("admission_identity")
        and transition_registry.get("binding_id") == binding_id
        and transition_registry.get("row_count") == expected_rows
        and int(transition_registry.get("valid_target_count", -1))
        + int(transition_registry.get("invalid_target_count", -1))
        == expected_rows
    )
    if not registry_valid:
        raise ValueError("audit part transition registry identity failed")
    _require_sha256(
        transition_registry.get("registry_sha256"),
        "admission_transition_registry.registry_sha256",
    )


def _read_part(
    path: Path,
    *,
    binding_id: str,
    binding_row_start: int,
    expected_rows: int,
    audit_binding: Mapping[str, Any],
    trusted_payload_sha256: str,
    trusted_file_sha256: str,
) -> dict[str, Any]:
    expected_payload_sha256 = _require_sha256(
        trusted_payload_sha256,
        "trusted part payload SHA-256",
    )
    expected_file_sha256 = _require_sha256(
        trusted_file_sha256,
        "trusted part file SHA-256",
    )
    actual_file_sha256 = _sha256_path(path)
    if actual_file_sha256 != expected_file_sha256:
        raise ValueError(f"fresh audit part file differs from in-process authority: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("part_payload_sha256") != expected_payload_sha256:
        raise ValueError(f"fresh audit part payload differs from in-process authority: {path}")
    _validate_part_payload(
        payload,
        binding_id=binding_id,
        binding_row_start=binding_row_start,
        expected_rows=expected_rows,
        audit_binding=audit_binding,
    )
    payload["part_file_sha256"] = actual_file_sha256

    return payload


def _write_part(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_bytes(path, gzip.compress(canonical_json_bytes(payload), compresslevel=6))


def _part_registry_payload(parts: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    entries = [
        {
            "binding_id": str(part["binding_id"]),
            "global_candidate_row_start": int(part["global_candidate_row_start"]),
            "expected_rows": int(part["expected_rows"]),
            "inspected_rows": int(part["inspected_rows"]),
            "part_payload_sha256": _require_sha256(
                part["part_payload_sha256"],
                "part_payload_sha256",
            ),
            "part_file_sha256": _require_sha256(
                part["part_file_sha256"],
                "part_file_sha256",
            ),
            "admission_transition_registry_sha256": _require_sha256(
                part["admission_transition_registry"]["registry_sha256"],
                "admission_transition_registry_sha256",
            ),
            "raw_candidate_reference_count": int(
                part["raw_episode_reference_inventory"][
                    "candidate_reference_count"
                ]
            ),
            "raw_bound_reference_count": int(
                part["raw_episode_reference_inventory"]["bound_reference_count"]
            ),
            "raw_unresolved_reference_count": int(
                part["raw_episode_reference_inventory"][
                    "unresolved_reference_count"
                ]
            ),
        }
        for part in parts
    ]
    payload: dict[str, Any] = {
        "schema_version": PART_REGISTRY_SCHEMA_VERSION,
        "resume_policy": RESUME_POLICY,
        "parts": entries,
    }
    payload["part_registry_sha256"] = sha256_json(payload)

    return payload


def _add_counter(target: dict[str, int], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def _merge_nested(
    target: dict[str, dict[str, int]],
    source: Mapping[str, Mapping[str, Any]],
) -> None:
    for key, values in source.items():
        current = target.setdefault(key, {})
        _add_counter(current, values)


def _merge_parts(parts: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "schema_version": AUDIT_DIAGNOSTICS_SCHEMA_VERSION,
        "total_candidates": 0,
        "complete_causal_alignment": 0,
        "outcomes": {},
        "failures": {},
        "first_failures": {},
        "action_families": {},
        "source_formats": {},
        "source_players": {},
        "choice_shapes": {},
        "missing_choice_patterns": {},
        "repair_rules": {},
        "observation_assets": {},
        "games": {},
        "diagnostic_samples": {},
        "exceptions": {},
        "raw_episode_hashes": {},
        "raw_episode_reference_inventory": _raw_episode_reference_inventory(
            candidate_reference_count=0,
            bound_reference_count=0,
            unresolved_reasons={},
            unresolved_samples=(),
            raw_episode_hashes={},
        ),
        "record_size_sample": {"count": 0, "canonical_json_bytes": 0},
        "binding_elapsed_seconds": 0.0,
    }
    for part in parts:
        source = part["diagnostics"]
        raw_episode_hashes = part["raw_episode_hashes"]
        raw_reference_inventory = part["raw_episode_reference_inventory"]
        _assert_raw_episode_reference_conservation(
            raw_reference_inventory,
            raw_episode_hashes,
            int(source["total_candidates"]),
        )
        aggregate["total_candidates"] += int(source["total_candidates"])
        aggregate["complete_causal_alignment"] += int(source["complete_causal_alignment"])
        for key in (
            "outcomes",
            "failures",
            "first_failures",
            "choice_shapes",
            "missing_choice_patterns",
            "repair_rules",
            "observation_assets",
            "exceptions",
        ):
            _add_counter(aggregate[key], source[key])
        for key in ("action_families", "source_formats", "source_players"):
            _merge_nested(aggregate[key], source[key])
        for game_id, game in source["games"].items():
            current = aggregate["games"].get(game_id)
            if current is None:
                current = {
                    **game,
                    "actor_indices": {
                        seat: list(values)
                        for seat, values in game["actor_indices"].items()
                    },
                    "global_decision_indices": list(game["global_decision_indices"]),
                    "source_event_positions": list(game["source_event_positions"]),
                }
                aggregate["games"][game_id] = current
                continue
            if (
                current["source_id"] != game["source_id"]
                or current["source_version"] != game["source_version"]
                or current["source_artifact_path"] != game["source_artifact_path"]
            ):
                raise ValueError(f"canonical game identity collision: {game_id}")
            for field in (
                "decisions",
                "complete_alignment",
                "valid_actor_and_critic",
                "forced_critic_only",
                "unresolved_critic_only",
                "rejected",
                "ambiguous",
                "deterministic_repair",
                "repair_attempted",
                "repair_rejected",
                "stale_legacy_legal_set",
                "terminal_rows",
            ):
                current[field] += int(game[field])
            for seat, values in game["actor_indices"].items():
                current["actor_indices"].setdefault(seat, []).extend(values)
            current["global_decision_indices"].extend(game["global_decision_indices"])
            current["source_event_positions"].extend(game["source_event_positions"])
        for key, values in source["diagnostic_samples"].items():
            current = aggregate["diagnostic_samples"].setdefault(key, [])
            current.extend(values[: max(0, 5 - len(current))])
        aggregate_raw_inventory = aggregate["raw_episode_reference_inventory"]
        for field in (
            "candidate_reference_count",
            "bound_reference_count",
            "unresolved_reference_count",
        ):
            aggregate_raw_inventory[field] += int(raw_reference_inventory[field])
        _add_counter(
            aggregate_raw_inventory["unresolved_reasons"],
            raw_reference_inventory["unresolved_reasons"],
        )
        aggregate_raw_samples = aggregate_raw_inventory[
            "unresolved_reference_samples"
        ]
        aggregate_raw_samples.extend(
            raw_reference_inventory["unresolved_reference_samples"][:
                max(
                    0,
                    MAX_RAW_EPISODE_REFERENCE_SAMPLES - len(aggregate_raw_samples),
                )
            ]
        )
        for path, digest in raw_episode_hashes.items():
            prior = aggregate["raw_episode_hashes"].get(path)
            if prior is not None and prior != digest:
                raise ValueError("raw Kaggle episode hash changed across audit parts")
            aggregate["raw_episode_hashes"][path] = digest
        aggregate["record_size_sample"]["count"] += int(part["record_size_sample"]["count"])
        aggregate["record_size_sample"]["canonical_json_bytes"] += int(
            part["record_size_sample"]["canonical_json_bytes"]
        )
        aggregate["binding_elapsed_seconds"] += float(part["elapsed_seconds"])
    aggregate["raw_episode_reference_inventory"][
        "unique_bound_episode_count"
    ] = len(aggregate["raw_episode_hashes"])
    _assert_raw_episode_reference_conservation(
        aggregate["raw_episode_reference_inventory"],
        aggregate["raw_episode_hashes"],
        aggregate["total_candidates"],
    )

    return aggregate


def _index_span_stats(values: Iterable[int]) -> dict[str, int | None]:
    observed = [int(value) for value in values]
    if not observed:
        return {
            "rows": 0,
            "minimum": None,
            "maximum": None,
            "missing_inside_observed_span": 0,
            "duplicates": 0,
        }
    unique = set(observed)

    return {
        "rows": len(observed),
        "minimum": min(unique),
        "maximum": max(unique),
        "missing_inside_observed_span": max(unique) - min(unique) + 1 - len(unique),
        "duplicates": len(observed) - len(unique),
    }


def _candidate_span_stats(game: Mapping[str, Any]) -> dict[str, Any]:
    decisions = int(game["decisions"])
    actor_indices = game["actor_indices"]
    actor_rows = 0
    actor_missing_inside_span = 0
    actor_duplicates = 0
    seats = 0
    for values in actor_indices.values():
        stats = _index_span_stats(values)
        if stats["rows"]:
            seats += 1
        actor_rows += int(stats["rows"])
        actor_missing_inside_span += int(stats["missing_inside_observed_span"])
        actor_duplicates += int(stats["duplicates"])
    global_stats = _index_span_stats(game["global_decision_indices"])
    source_position_stats = _index_span_stats(game["source_event_positions"])
    actor_metadata_missing = max(0, decisions - actor_rows)
    global_metadata_missing = max(0, decisions - int(global_stats["rows"]))
    source_position_metadata_missing = max(0, decisions - int(source_position_stats["rows"]))
    candidate_span_complete = bool(
        decisions > 0
        and actor_metadata_missing == 0
        and global_metadata_missing == 0
        and source_position_metadata_missing == 0
        and actor_missing_inside_span == 0
        and actor_duplicates == 0
        and int(global_stats["missing_inside_observed_span"]) == 0
        and int(global_stats["duplicates"]) == 0
        and source_position_stats["minimum"] is not None
        and int(source_position_stats["minimum"]) >= 0
    )
    actor_safe_candidate_span = bool(
        candidate_span_complete
        and game["complete_alignment"] == decisions
        and game["rejected"] == 0
        and game["unresolved_critic_only"] == 0
        and game["repair_rejected"] == 0
        and decisions == game["valid_actor_and_critic"] + game["forced_critic_only"]
    )

    return {
        "seats_with_actor_indices": seats,
        "actor_index_rows": actor_rows,
        "actor_index_metadata_missing": actor_metadata_missing,
        "missing_actor_decision_indices_inside_observed_span": actor_missing_inside_span,
        "duplicate_actor_decision_indices": actor_duplicates,
        "global_index_rows": int(global_stats["rows"]),
        "global_index_metadata_missing": global_metadata_missing,
        "global_index_minimum": global_stats["minimum"],
        "global_index_maximum": global_stats["maximum"],
        "missing_global_decision_indices_inside_observed_span": int(
            global_stats["missing_inside_observed_span"]
        ),
        "duplicate_global_decision_indices": int(global_stats["duplicates"]),
        "source_event_position_rows": int(source_position_stats["rows"]),
        "source_event_position_metadata_missing": source_position_metadata_missing,
        "source_event_position_minimum": source_position_stats["minimum"],
        "source_event_position_maximum": source_position_stats["maximum"],
        "candidate_span_complete": candidate_span_complete,
        "actor_safe_candidate_span": actor_safe_candidate_span,
        "raw_terminal_row_evidence": bool(game["terminal_rows"] > 0),
        "raw_source_complete": False,
        "raw_source_completeness_status": (
            "not_proven_by_canonical_audit_requires_raw_source_registry"
        ),
    }


def _write_game_stats(path: Path, games: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    counts = Counter()
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "canonical_game_id",
            "source_id",
            "source_version",
            "source_artifact_path",
            "decisions",
            "complete_alignment",
            "valid_actor_and_critic",
            "forced_critic_only",
            "unresolved_critic_only",
            "rejected",
            "ambiguous",
            "repair_attempted",
            "repair_succeeded",
            "repair_rejected",
            "stale_legacy_legal_set",
            "terminal_rows",
            "seats_with_actor_indices",
            "actor_index_rows",
            "actor_index_metadata_missing",
            "missing_actor_decision_indices_inside_observed_span",
            "duplicate_actor_decision_indices",
            "global_index_rows",
            "global_index_metadata_missing",
            "global_index_minimum",
            "global_index_maximum",
            "missing_global_decision_indices_inside_observed_span",
            "duplicate_global_decision_indices",
            "source_event_position_rows",
            "source_event_position_metadata_missing",
            "source_event_position_minimum",
            "source_event_position_maximum",
            "candidate_span_complete",
            "actor_safe_candidate_span",
            "raw_terminal_row_evidence",
            "raw_source_complete",
            "raw_source_completeness_status",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for game_id in sorted(games):
            game = games[game_id]
            span = _candidate_span_stats(game)
            counts["games"] += 1
            counts["candidate_span_complete_games"] += int(span["candidate_span_complete"])
            counts["candidate_span_incomplete_games"] += int(not span["candidate_span_complete"])
            counts["actor_safe_candidate_span_games"] += int(
                span["actor_safe_candidate_span"]
            )
            counts["actor_unsafe_candidate_span_games"] += int(
                not span["actor_safe_candidate_span"]
            )
            counts["raw_source_complete_games"] += 0
            counts["raw_source_completeness_not_proven_games"] += 1
            counts["games_with_terminal_row_evidence"] += int(
                span["raw_terminal_row_evidence"]
            )
            counts["games_with_repair"] += int(game["deterministic_repair"] > 0)
            counts["games_with_rejected_repair"] += int(game["repair_rejected"] > 0)
            counts["games_with_stale_legacy_legal_set"] += int(
                game["stale_legacy_legal_set"] > 0
            )
            counts["games_with_ambiguity"] += int(game["ambiguous"] > 0)
            counts["games_with_candidate_span_discontinuity"] += int(
                not span["candidate_span_complete"]
            )
            for field in (
                "actor_index_metadata_missing",
                "missing_actor_decision_indices_inside_observed_span",
                "duplicate_actor_decision_indices",
                "global_index_metadata_missing",
                "missing_global_decision_indices_inside_observed_span",
                "duplicate_global_decision_indices",
                "source_event_position_metadata_missing",
            ):
                counts[field] += int(span[field])
            writer.writerow({
                "canonical_game_id": game_id,
                "source_id": game["source_id"],
                "source_version": game["source_version"],
                "source_artifact_path": game["source_artifact_path"],
                "decisions": game["decisions"],
                "complete_alignment": game["complete_alignment"],
                "valid_actor_and_critic": game["valid_actor_and_critic"],
                "forced_critic_only": game["forced_critic_only"],
                "unresolved_critic_only": game["unresolved_critic_only"],
                "rejected": game["rejected"],
                "ambiguous": game["ambiguous"],
                "repair_attempted": game["repair_attempted"],
                "repair_succeeded": game["deterministic_repair"],
                "repair_rejected": game["repair_rejected"],
                "stale_legacy_legal_set": game["stale_legacy_legal_set"],
                "terminal_rows": game["terminal_rows"],
                **{
                    key: int(value) if isinstance(value, bool) else value
                    for key, value in span.items()
                },
            })
    os.replace(temporary, path)

    return dict(sorted(counts.items()))


def _exact_proportions(
    values: Mapping[str, int],
    total: int,
    *,
    verified: bool = True,
) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "count": int(value),
            "denominator": total if verified else None,
            "proportion": (float(value) / total if verified and total else None),
            "percent": (
                100.0 * float(value) / total if verified and total else None
            ),
            "sampling_uncertainty": (
                "none_full_corpus_exact"
                if verified
                else "not_asserted_verification_blocked"
            ),
        }
        for key, value in sorted(values.items())
    }


def _report(summary: Mapping[str, Any]) -> str:
    outcomes = summary["outcomes"]
    verified = summary["status"] == "COMPLETE_READ_ONLY_EXACT"
    lines = [
        "# Full Admission V0 Semantic Audit",
        "",
        f"Status: `{summary['status']}`",
        "",
        (
            "Final verification passed; full-corpus proportions below are authoritative."
            if verified
            else "Final verification did not pass; counts below are diagnostic and no full-corpus proportion is asserted."
        ),
        "",
        f"- Inspected: **{summary['total_candidates']:,} / {summary['expected_candidates']:,}** candidates.",
        f"- Complete causal alignment: **{summary['complete_causal_alignment']:,}**.",
        f"- Valid actor and critic: **{outcomes.get('VALID_ACTOR_AND_CRITIC', 0):,}**.",
        f"- Forced critic-only: **{outcomes.get('VALID_CRITIC_ONLY_FORCED', 0):,}**.",
        f"- Unresolved critic-only: **{outcomes.get('VALID_CRITIC_ONLY_UNRESOLVED_ACTION', 0):,}**.",
        f"- Ambiguous fail-closed: **{outcomes.get('UNKNOWN_FAIL_CLOSED', 0):,}**.",
        f"- Unexpected audit exceptions: **{sum(summary['exceptions'].values()):,}**.",
        "",
        "Admission V0 was opened read-only and remains historical structural evidence. "
        "No actor label was accepted from its old Kaggle same-row legal set.",
        "",
        "## Exact outcomes" if verified else "## Diagnostic outcomes (verification blocked)",
        "",
        "| outcome | count | exact % |",
        "|---|---:|---:|",
    ]
    for outcome, details in summary["outcome_proportions"].items():
        percent = details["percent"]
        display = f"{percent:.8f}" if percent is not None else "not asserted"
        lines.append(f"| {outcome} | {details['count']:,} | {display} |")
    lines += [
        "",
        "## Deterministic repair",
        "",
        "Attempts are the denominator for both succeeded and rejected rates.",
        "",
        "| rule | attempted | succeeded | rejected | succeeded % | rejected % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for rule_id, details in summary["repair_statistics"]["rules"].items():
        succeeded_percent = details["succeeded_percent"]
        rejected_percent = details["rejected_percent"]
        lines.append(
            f"| {rule_id} | {details['attempted']:,} | {details['succeeded']:,} | "
            f"{details['rejected']:,} | "
            f"{succeeded_percent if succeeded_percent is not None else 'n/a'} | "
            f"{rejected_percent if rejected_percent is not None else 'n/a'} |"
        )
    lines += [
        "",
        "## Normalized observation references",
        "",
        f"- Reference identity counts: `{summary['normalized_observation_references']}`",
        "- Exact tensor-cache reuse is not inferred here; it is deferred to the separate X registry.",
        "",
        "## Game scope",
        "",
        f"- Game impact: `{summary['game_impact']}`",
        f"- Candidate-span completeness: `{summary['candidate_span_completeness']}`",
        f"- Raw-source completeness: `{summary['raw_source_completeness']}`",
        "- Min/max continuity and terminal-row presence do not prove raw-source completeness.",
        f"- Derived V1 estimate: `{summary['derived_v1_estimate']}`",
        "",
        "## Boundary",
        "",
        "No E1, branch-Q, PPO, evaluation, Modal, paid compute, policy training, "
        "protected runtime change, or Derived V1 production materialization occurred.",
    ]

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission-dir", type=Path, default=DEFAULT_ADMISSION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--skip-database-sha", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    admission_dir = args.admission_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = admission_dir / "admission.sqlite3"
    manifest_path = admission_dir / "admission.manifest.json"
    admission_manifest_sha256 = _sha256_path(manifest_path)
    admission_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        admission_manifest.get("status") != "complete"
        or admission_manifest.get("transition_count") != 7_215_562
        or admission_manifest.get("request_count") != 7_215_562
        or admission_manifest.get("observation_count") != 7_215_562
    ):
        raise SystemExit("Admission V0 exact complete receipt is absent")
    provided_receipt_sha256 = _require_sha256(
        admission_manifest.get("receipt_sha256"),
        "admission receipt_sha256",
    )
    receipt_body = dict(admission_manifest)
    receipt_body.pop("receipt_sha256", None)
    if provided_receipt_sha256 != sha256_json(receipt_body):
        raise SystemExit("Admission V0 receipt self-hash is invalid")
    wal_path = database_path.with_name(database_path.name + "-wal")
    wal_before = wal_path.stat().st_size if wal_path.exists() else 0
    if wal_before != 0:
        raise SystemExit("Admission V0 WAL is non-empty; immutable audit refused")
    try:
        database_before, database_sha256_before = _verify_admission_database_before(
            database_path,
            expected_sha256=admission_manifest["database_sha256"],
            expected_bytes=admission_manifest["database_bytes"],
            skip_database_sha=args.skip_database_sha,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Admission V0 startup verification failed: {exc}") from exc
    contract_files = sorted((ROOT / "deepmew" / "replay" / "contract").glob("*.py"))
    contract_sha256 = implementation_sha256(*contract_files)
    auditor_implementation_sha256 = _sha256_path(Path(__file__).resolve())
    implementation_hash_manifest = _implementation_hash_manifest()
    implementation_hashes = _typed_implementation_hashes(
        implementation_hash_manifest
    )
    try:
        repair_rule_registry_manifest = _repair_rule_registry_manifest()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"deterministic repair registry verification failed: {exc}") from exc
    semantic_dependency_manifest = _semantic_dependency_manifest()
    semantic_dependency_sha256 = semantic_dependency_manifest[
        "semantic_dependency_sha256"
    ]
    connection = _readonly_connection(database_path)
    try:
        bindings = [
            (str(identity), int(rows))
            for identity, rows in connection.execute(
                "SELECT identity,rows FROM completed_inputs "
                "WHERE kind='transition' ORDER BY identity"
            )
        ]
        kaggle_hashes = frozenset(
            str(content)
            for content, in connection.execute(
                "SELECT content_sha256 FROM requests "
                "WHERE lower(source_path) LIKE '%kaggle_ingest%'"
            )
        )
        actual_transition_count = int(
            connection.execute("SELECT COUNT(*) FROM transitions").fetchone()[0]
        )
        actual_binding_ids = tuple(
            str(binding_id)
            for binding_id, in connection.execute(
                "SELECT DISTINCT binding_id FROM transitions ORDER BY binding_id"
            )
        )
    finally:
        connection.close()
    expected_candidates = int(admission_manifest["transition_count"])
    try:
        _assert_completed_transition_binding_coverage(
            bindings,
            actual_binding_ids=actual_binding_ids,
            expected_candidates=expected_candidates,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if actual_transition_count != expected_candidates:
        raise SystemExit("Admission V0 transition table count differs from its receipt")
    binding_registry = _binding_registry_payload(
        bindings,
        actual_transition_count=actual_transition_count,
    )
    binding_registry_path = output_dir / "FULL_AUDIT_BINDING_REGISTRY.json"
    _atomic_json(binding_registry_path, binding_registry)
    semantic_dependency_path = output_dir / "FULL_AUDIT_SEMANTIC_DEPENDENCIES.json"
    _atomic_json(semantic_dependency_path, semantic_dependency_manifest)
    repair_rule_registry_manifest_path = (
        output_dir / "FULL_AUDIT_REPAIR_RULE_REGISTRY_BINDING.json"
    )
    _atomic_json(repair_rule_registry_manifest_path, repair_rule_registry_manifest)
    implementation_hash_manifest_path = (
        output_dir / "FULL_AUDIT_IMPLEMENTATION_HASHES.json"
    )
    _atomic_json(implementation_hash_manifest_path, implementation_hash_manifest)
    audit_binding = _build_audit_binding(
        contract_sha256=contract_sha256,
        semantic_dependency_sha256=semantic_dependency_sha256,
        auditor_implementation_sha256=auditor_implementation_sha256,
        admission_manifest_sha256=admission_manifest_sha256,
        admission_receipt_sha256=provided_receipt_sha256,
        admission_identity=admission_manifest["admission_identity_sha256"],
        admission_database_expected_sha256=admission_manifest["database_sha256"],
        admission_database_actual_before_sha256=database_sha256_before,
        admission_database_expected_bytes=admission_manifest["database_bytes"],
        admission_binding_set_sha256=admission_manifest["binding_set_sha256"],
        binding_registry_sha256=binding_registry["binding_registry_sha256"],
        repair_rule_registry_file_sha256=repair_rule_registry_manifest[
            "file_sha256"
        ],
        repair_rule_registry_payload_sha256=repair_rule_registry_manifest[
            "canonical_payload_sha256"
        ],
        repair_rule_registry_manifest_sha256=repair_rule_registry_manifest[
            "manifest_sha256"
        ],
        repair_rule_registry_id=repair_rule_registry_manifest["registry_id"],
        causal_aligner_implementation_sha256=(
            implementation_hashes.causal_aligner_implementation_sha256
        ),
        legality_validator_implementation_sha256=(
            implementation_hashes.legality_validator_implementation_sha256
        ),
        action_schema_implementation_sha256=(
            implementation_hashes.action_schema_implementation_sha256
        ),
        implementation_hash_manifest_sha256=implementation_hash_manifest[
            "manifest_sha256"
        ],
        expected_candidates=expected_candidates,
    )
    binding_jobs = [
        (
            str(entry["binding_id"]),
            int(entry["global_candidate_row_start"]),
            int(entry["expected_rows"]),
        )
        for entry in binding_registry["bindings"]
    ]
    parts_dir = output_dir / "full_audit_v3_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    ignored_preexisting_part_files = sum(
        1 for path in parts_dir.glob("*.json.gz") if path.is_file()
    )
    completed: dict[str, dict[str, Any]] = {}
    trusted_part_hashes: dict[str, tuple[str, str]] = {}
    pending = list(binding_jobs)
    print(
        json.dumps({
            "event": "audit_start",
            "bindings": len(bindings),
            "resume_policy": RESUME_POLICY,
            "resumed_bindings": 0,
            "preexisting_part_files_ignored": ignored_preexisting_part_files,
            "pending_bindings": len(pending),
            "expected_candidates": expected_candidates,
            "workers": args.workers,
            "kaggle_observation_hash_index": len(kaggle_hashes),
            "audit_identity": audit_binding["audit_identity"],
            "binding_registry_sha256": binding_registry["binding_registry_sha256"],
        }),
        flush=True,
    )
    wall_started = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(
                str(database_path),
                str(ROOT),
                admission_manifest["admission_identity_sha256"],
                kaggle_hashes,
                contract_sha256,
                audit_binding,
            ),
        ) as pool:
            future_jobs = {pool.submit(_audit_binding, job): job for job in pending}
            for future in as_completed(future_jobs):
                binding_id, _, rows = future_jobs[future]
                payload = future.result()
                binding_row_start = future_jobs[future][1]
                _validate_part_payload(
                    payload,
                    binding_id=binding_id,
                    binding_row_start=binding_row_start,
                    expected_rows=rows,
                    audit_binding=audit_binding,
                )
                part_path = _part_path(parts_dir, binding_id)
                _write_part(part_path, payload)
                part_file_sha256 = _sha256_path(part_path)
                part_payload_sha256 = _require_sha256(
                    payload["part_payload_sha256"],
                    "fresh part payload SHA-256",
                )
                trusted_part_hashes[binding_id] = (
                    part_payload_sha256,
                    part_file_sha256,
                )
                completed[binding_id] = {
                    **payload,
                    "part_file_sha256": part_file_sha256,
                }
                if len(completed) % 10 == 0 or len(completed) == len(bindings):
                    inspected = sum(part["inspected_rows"] for part in completed.values())
                    print(
                        json.dumps({
                            "event": "audit_progress",
                            "completed_bindings": len(completed),
                            "total_bindings": len(bindings),
                            "inspected_candidates": inspected,
                            "percent": 100.0 * inspected / expected_candidates,
                        }),
                        flush=True,
                    )
    wall_seconds = time.perf_counter() - wall_started
    try:
        ordered_parts = [
            completed[binding_id]
            for binding_id, binding_row_start, rows in binding_jobs
        ]
        for binding_id, binding_row_start, rows in binding_jobs:
            trusted_payload_sha256, trusted_file_sha256 = trusted_part_hashes[
                binding_id
            ]
            _read_part(
                _part_path(parts_dir, binding_id),
                binding_id=binding_id,
                binding_row_start=binding_row_start,
                expected_rows=rows,
                audit_binding=audit_binding,
                trusted_payload_sha256=trusted_payload_sha256,
                trusted_file_sha256=trusted_file_sha256,
            )
        aggregate = _merge_parts(ordered_parts)
        _assert_diagnostics_conservation(aggregate, expected_candidates)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        receipt = _write_fail_closed_verification_artifacts(
            output_dir,
            audit_binding=audit_binding,
            stage="part_reduction_and_counter_conservation",
            error=exc,
        )
        print(json.dumps({
            "event": "audit_blocked_fail_closed",
            "stage": "part_reduction_and_counter_conservation",
            "receipt_sha256": receipt["receipt_sha256"],
        }), flush=True)

        return 2
    counters_conserved = True
    fresh_parts_verified = True
    observation_reference_statistics = _observation_reference_statistics(
        aggregate["observation_assets"]
    )
    part_registry = _part_registry_payload(ordered_parts)
    part_registry_path = output_dir / "FULL_AUDIT_PART_REGISTRY.json"
    _atomic_json(part_registry_path, part_registry)
    game_stats_path = output_dir / "FULL_AUDIT_GAME_STATS.csv.gz"
    game_impact = _write_game_stats(game_stats_path, aggregate["games"])
    raw_hash_path = output_dir / "KAGGLE_RAW_EPISODE_HASH_REGISTRY.json"
    raw_reference_inventory = aggregate["raw_episode_reference_inventory"]
    raw_reference_coverage_complete = _raw_episode_reference_coverage_complete(
        raw_reference_inventory,
        aggregate["raw_episode_hashes"],
        expected_candidates,
    )
    database_after = database_path.stat()
    wal_after = wal_path.stat().st_size if wal_path.exists() else 0
    sha_recomputed = True
    try:
        database_sha256_after = _sha256_path(database_path)
    except OSError as exc:
        receipt = _write_fail_closed_verification_artifacts(
            output_dir,
            audit_binding=audit_binding,
            stage="admission_database_final_sha256",
            error=exc,
            inspected_candidates=aggregate["total_candidates"],
        )
        print(json.dumps({
            "event": "audit_blocked_fail_closed",
            "stage": "admission_database_final_sha256",
            "receipt_sha256": receipt["receipt_sha256"],
        }), flush=True)

        return 2
    raw_source_verification_error = None
    try:
        raw_source_verification = _verify_raw_episode_sources(
            aggregate["raw_episode_hashes"],
        )
        raw_rehash_verified = (
            len(raw_source_verification["episodes"])
            == len(aggregate["raw_episode_hashes"])
        )
    except (OSError, ValueError) as exc:
        raw_rehash_verified = False
        raw_source_verification_error = {
            "error_type": type(exc).__name__,
            "message": str(exc)[:1000],
        }
        raw_source_verification = {
            "schema_version": "deepmew.kaggle_raw_episode_final_verification.v1",
            "episodes": [
                {"path": path, "expected_sha256": digest, "verified": False}
                for path, digest in sorted(aggregate["raw_episode_hashes"].items())
            ],
            "verification_failure": raw_source_verification_error,
        }
        raw_source_verification["verification_payload_sha256"] = sha256_json(
            raw_source_verification
        )
    if raw_rehash_verified and not raw_reference_coverage_complete:
        raw_source_verification_error = {
            "error_type": "RawEpisodeReferenceCoverageError",
            "message": (
                "raw Kaggle candidate references are empty, incomplete, or unresolved"
            ),
        }
    raw_sources_verified = bool(
        raw_rehash_verified and raw_reference_coverage_complete
    )
    raw_hash_payload = {
        "schema_version": "deepmew.kaggle_raw_episode_hash_registry.v2",
        "contract_version": CANONICAL_DECISION_CONTRACT_V1,
        "audit_identity": audit_binding["audit_identity"],
        "resume_policy": RESUME_POLICY,
        "reference_inventory": raw_reference_inventory,
        "reference_coverage_complete": raw_reference_coverage_complete,
        "episodes": raw_source_verification["episodes"],
        "final_verification_sha256": raw_source_verification[
            "verification_payload_sha256"
        ],
        "verification_failure": raw_source_verification_error,
    }
    raw_hash_payload["registry_sha256"] = sha256_json(raw_hash_payload)
    _atomic_json(raw_hash_path, raw_hash_payload)
    try:
        auditor_implementation_sha256_after = _sha256_path(Path(__file__).resolve())
        semantic_dependency_sha256_after = _semantic_dependency_manifest()[
            "semantic_dependency_sha256"
        ]
        repair_rule_registry_manifest_after = _repair_rule_registry_manifest()
        implementation_hash_manifest_after = _implementation_hash_manifest()
        admission_manifest_sha256_after = _sha256_path(manifest_path)
    except (OSError, ValueError) as exc:
        receipt = _write_fail_closed_verification_artifacts(
            output_dir,
            audit_binding=audit_binding,
            stage="frozen_input_final_sha256",
            error=exc,
            inspected_candidates=aggregate["total_candidates"],
        )
        print(json.dumps({
            "event": "audit_blocked_fail_closed",
            "stage": "frozen_input_final_sha256",
            "receipt_sha256": receipt["receipt_sha256"],
        }), flush=True)

        return 2
    frozen_inputs_ok = (
        auditor_implementation_sha256_after == auditor_implementation_sha256
        and semantic_dependency_sha256_after == semantic_dependency_sha256
        and repair_rule_registry_manifest_after == repair_rule_registry_manifest
        and implementation_hash_manifest_after == implementation_hash_manifest
        and admission_manifest_sha256_after == admission_manifest_sha256
    )
    outcomes = aggregate["outcomes"]
    positive_rows = sum(
        outcomes.get(name, 0)
        for name in (
            "VALID_ACTOR_AND_CRITIC",
            "VALID_CRITIC_ONLY_FORCED",
            "VALID_CRITIC_ONLY_UNRESOLVED_ACTION",
        )
    )
    size_sample = aggregate["record_size_sample"]
    average_record_bytes = (
        size_sample["canonical_json_bytes"] / size_sample["count"]
        if size_sample["count"]
        else 0.0
    )
    derived_estimate = {
        "positive_rows": positive_rows,
        "canonical_record_average_json_bytes_sampled": average_record_bytes,
        "canonical_record_sample_count": size_sample["count"],
        "uncompressed_record_bytes": int(positive_rows * average_record_bytes),
        "estimated_local_materialization_seconds_at_audit_rate": (
            1.25 * expected_candidates / (expected_candidates / wall_seconds)
            if wall_seconds > 0
            else None
        ),
        "paid_cost_usd": 0.0,
        "estimate_class": "planning_estimate_not_production_materialization",
        "tensor_registry_status": "not_audited_deferred_to_separate_x_registry",
    }
    exception_count = sum(aggregate["exceptions"].values())
    immutable_ok = (
        database_before.st_size == database_after.st_size == admission_manifest["database_bytes"]
        and database_sha256_before == admission_manifest["database_sha256"]
        and database_sha256_after == admission_manifest["database_sha256"]
        and wal_before == wal_after == 0
        and sha_recomputed
    )
    status = _completion_status(
        inspected_candidates=aggregate["total_candidates"],
        expected_candidates=expected_candidates,
        exception_count=exception_count,
        immutable_ok=immutable_ok,
        sha_recomputed=sha_recomputed,
        frozen_inputs_ok=frozen_inputs_ok,
        raw_sources_verified=raw_sources_verified,
        raw_reference_coverage_complete=raw_reference_coverage_complete,
        counters_conserved=counters_conserved,
        fresh_parts_verified=fresh_parts_verified,
    )
    verification_complete = status == "COMPLETE_READ_ONLY_EXACT"
    repair_statistics = _repair_statistics(
        aggregate["repair_rules"],
        verified=verification_complete,
    )
    summary = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": status,
        "contract_version": CANONICAL_DECISION_CONTRACT_V1,
        "contract_implementation_sha256": contract_sha256,
        "semantic_dependency_sha256": semantic_dependency_sha256,
        "auditor_implementation_sha256": auditor_implementation_sha256,
        "audit_identity": audit_binding["audit_identity"],
        "resume_policy": RESUME_POLICY,
        "global_candidate_row_convention": GLOBAL_CANDIDATE_ROW_CONVENTION,
        "admission_manifest_sha256": admission_manifest_sha256,
        "binding_registry_sha256": binding_registry["binding_registry_sha256"],
        "repair_rule_registry": repair_rule_registry_manifest,
        "implementation_hashes": implementation_hash_manifest,
        "part_registry_sha256": part_registry["part_registry_sha256"],
        "validator_version": VALIDATOR_VERSION,
        "causal_aligner_version": CAUSAL_ALIGNER_VERSION,
        "admission_identity": admission_manifest["admission_identity_sha256"],
        "admission_receipt_sha256": admission_manifest["receipt_sha256"],
        "admission_database_sha256_before": database_sha256_before,
        "admission_database_sha256": database_sha256_after,
        "expected_candidates": expected_candidates,
        "total_candidates": aggregate["total_candidates"],
        "complete_causal_alignment": aggregate["complete_causal_alignment"],
        "outcomes": dict(sorted(outcomes.items())),
        "outcome_proportions": _exact_proportions(
            outcomes,
            expected_candidates,
            verified=verification_complete,
        ),
        "failures": dict(sorted(aggregate["failures"].items())),
        "first_failures": dict(sorted(aggregate["first_failures"].items())),
        "choice_shapes": dict(sorted(aggregate["choice_shapes"].items())),
        "repair_statistics": repair_statistics,
        "normalized_observation_references": observation_reference_statistics,
        "action_families": dict(sorted(aggregate["action_families"].items())),
        "source_formats": dict(sorted(aggregate["source_formats"].items())),
        "source_players": dict(sorted(aggregate["source_players"].items())),
        "missing_choice_patterns": dict(
            sorted(aggregate["missing_choice_patterns"].items())
        ),
        "exceptions": dict(sorted(aggregate["exceptions"].items())),
        "diagnostic_samples": aggregate["diagnostic_samples"],
        "game_impact": game_impact,
        "candidate_span_completeness": {
            "scope": "admitted_candidate_span_only_not_raw_source_completeness",
            "complete_games": game_impact.get("candidate_span_complete_games", 0),
            "incomplete_games": game_impact.get("candidate_span_incomplete_games", 0),
            "actor_index_metadata_missing": game_impact.get(
                "actor_index_metadata_missing", 0
            ),
            "missing_actor_decision_indices_inside_observed_span": game_impact.get(
                "missing_actor_decision_indices_inside_observed_span", 0
            ),
            "duplicate_actor_decision_indices": game_impact.get(
                "duplicate_actor_decision_indices", 0
            ),
            "global_index_metadata_missing": game_impact.get(
                "global_index_metadata_missing", 0
            ),
            "missing_global_decision_indices_inside_observed_span": game_impact.get(
                "missing_global_decision_indices_inside_observed_span", 0
            ),
            "duplicate_global_decision_indices": game_impact.get(
                "duplicate_global_decision_indices", 0
            ),
        },
        "raw_source_completeness": {
            "status": "not_proven_requires_separate_raw_source_registry",
            "complete_games": 0,
            "not_proven_games": game_impact.get(
                "raw_source_completeness_not_proven_games", 0
            ),
            "games_with_terminal_row_evidence": game_impact.get(
                "games_with_terminal_row_evidence", 0
            ),
            "terminal_row_evidence_is_not_completeness_proof": True,
        },
        "raw_episode_reference_coverage": {
            **raw_reference_inventory,
            "reference_coverage_complete": raw_reference_coverage_complete,
            "every_unique_bound_episode_final_rehashed": raw_rehash_verified,
            "scope": "referenced_kaggle_raw_episode_inventory_not_game_completeness",
        },
        "derived_v1_estimate": derived_estimate,
        "execution": {
            "workers": args.workers,
            "wall_seconds_current_invocation": wall_seconds,
            "sum_binding_seconds": aggregate["binding_elapsed_seconds"],
            "rows_per_wall_second_current_invocation": (
                sum(job[2] for job in pending) / wall_seconds if wall_seconds > 0 else None
            ),
            "resume_policy": RESUME_POLICY,
            "resumed_bindings": 0,
            "preexisting_part_files_ignored": ignored_preexisting_part_files,
            "executed_bindings": len(pending),
            "part_count": len(ordered_parts),
        },
        "immutable_admission_proof": {
            "database_bytes_before": database_before.st_size,
            "database_bytes_after": database_after.st_size,
            "database_sha256_expected": admission_manifest["database_sha256"],
            "database_sha256_before": database_sha256_before,
            "database_sha256_after": database_sha256_after,
            "sha_recomputed": sha_recomputed,
            "wal_bytes_before": wal_before,
            "wal_bytes_after": wal_after,
            "immutable_ok": immutable_ok,
        },
        "frozen_input_proof": {
            "auditor_sha256_before": auditor_implementation_sha256,
            "auditor_sha256_after": auditor_implementation_sha256_after,
            "semantic_dependency_sha256_before": semantic_dependency_sha256,
            "semantic_dependency_sha256_after": semantic_dependency_sha256_after,
            "repair_rule_registry_manifest_sha256_before": (
                repair_rule_registry_manifest["manifest_sha256"]
            ),
            "repair_rule_registry_manifest_sha256_after": (
                repair_rule_registry_manifest_after["manifest_sha256"]
            ),
            "implementation_hash_manifest_sha256_before": (
                implementation_hash_manifest["manifest_sha256"]
            ),
            "implementation_hash_manifest_sha256_after": (
                implementation_hash_manifest_after["manifest_sha256"]
            ),
            "admission_manifest_sha256_before": admission_manifest_sha256,
            "admission_manifest_sha256_after": admission_manifest_sha256_after,
            "frozen_inputs_ok": frozen_inputs_ok,
        },
        "integrity": {
            "actual_transition_count": actual_transition_count,
            "binding_count": len(bindings),
            "counters_conserved": counters_conserved,
            "fresh_parts_verified": fresh_parts_verified,
            "raw_sources_verified": raw_sources_verified,
            "raw_reference_coverage_complete": raw_reference_coverage_complete,
            "raw_candidate_reference_count": raw_reference_inventory[
                "candidate_reference_count"
            ],
            "raw_bound_reference_count": raw_reference_inventory[
                "bound_reference_count"
            ],
            "raw_unresolved_reference_count": raw_reference_inventory[
                "unresolved_reference_count"
            ],
            "raw_unique_bound_episode_count": raw_reference_inventory[
                "unique_bound_episode_count"
            ],
            "raw_sources_verification_sha256": raw_source_verification[
                "verification_payload_sha256"
            ],
            "raw_sources_verification_failure": raw_source_verification_error,
        },
        "artifacts": {
            "binding_registry": binding_registry_path.name,
            "game_stats": game_stats_path.name,
            "part_registry": part_registry_path.name,
            "raw_episode_hash_registry": raw_hash_path.name,
            "semantic_dependencies": semantic_dependency_path.name,
            "repair_rule_registry_binding": (
                repair_rule_registry_manifest_path.name
            ),
            "implementation_hashes": implementation_hash_manifest_path.name,
            "parts_directory": parts_dir.name,
        },
        "critic_only_unresolved_policy": "disabled_by_default_fail_closed_v1",
        "audit_is_full_corpus": verification_complete,
        "candidate_iteration_covered_full_admission_population": (
            aggregate["total_candidates"] == expected_candidates
        ),
        "sampling_uncertainty": (
            "none_full_corpus_exact"
            if verification_complete
            else "not_asserted_verification_blocked"
        ),
    }
    summary_path = output_dir / "FULL_CORPUS_SEMANTIC_AUDIT.json"
    _atomic_json(summary_path, summary)
    report_path = output_dir / "FULL_CORPUS_SEMANTIC_AUDIT_REPORT.md"
    _atomic_bytes(report_path, _report(summary).encode("utf-8"))
    receipt = {
        "schema_version": AUDIT_RECEIPT_SCHEMA_VERSION,
        "status": status,
        "contract_version": CANONICAL_DECISION_CONTRACT_V1,
        "contract_implementation_sha256": contract_sha256,
        "semantic_dependency_sha256": semantic_dependency_sha256,
        "auditor_implementation_sha256": auditor_implementation_sha256,
        "audit_identity": audit_binding["audit_identity"],
        "resume_policy": RESUME_POLICY,
        "global_candidate_row_convention": GLOBAL_CANDIDATE_ROW_CONVENTION,
        "admission_manifest_sha256": admission_manifest_sha256,
        "admission_identity": admission_manifest["admission_identity_sha256"],
        "admission_receipt_sha256": admission_manifest["receipt_sha256"],
        "admission_database_expected_sha256": admission_manifest["database_sha256"],
        "admission_database_actual_before_sha256": database_sha256_before,
        "admission_database_sha256": database_sha256_after,
        "expected_candidates": expected_candidates,
        "inspected_candidates": aggregate["total_candidates"],
        "audit_summary_sha256": _sha256_path(summary_path),
        "audit_report_sha256": _sha256_path(report_path),
        "game_stats_sha256": _sha256_path(game_stats_path),
        "raw_episode_hash_registry_sha256": _sha256_path(raw_hash_path),
        "raw_sources_verification_sha256": raw_source_verification[
            "verification_payload_sha256"
        ],
        "raw_sources_verification_failure": raw_source_verification_error,
        "binding_registry_sha256": binding_registry["binding_registry_sha256"],
        "binding_registry_file_sha256": _sha256_path(binding_registry_path),
        "repair_rule_registry_file_sha256": repair_rule_registry_manifest[
            "file_sha256"
        ],
        "repair_rule_registry_payload_sha256": repair_rule_registry_manifest[
            "canonical_payload_sha256"
        ],
        "repair_rule_registry_manifest_sha256": repair_rule_registry_manifest[
            "manifest_sha256"
        ],
        "repair_rule_registry_id": repair_rule_registry_manifest["registry_id"],
        "repair_rule_registry_binding_file_sha256": _sha256_path(
            repair_rule_registry_manifest_path
        ),
        "causal_aligner_implementation_sha256": (
            implementation_hashes.causal_aligner_implementation_sha256
        ),
        "legality_validator_implementation_sha256": (
            implementation_hashes.legality_validator_implementation_sha256
        ),
        "action_schema_implementation_sha256": (
            implementation_hashes.action_schema_implementation_sha256
        ),
        "implementation_hash_manifest_sha256": implementation_hash_manifest[
            "manifest_sha256"
        ],
        "implementation_hash_manifest_file_sha256": _sha256_path(
            implementation_hash_manifest_path
        ),
        "part_registry_sha256": part_registry["part_registry_sha256"],
        "part_registry_file_sha256": _sha256_path(part_registry_path),
        "semantic_dependencies_file_sha256": _sha256_path(semantic_dependency_path),
        "part_count": len(ordered_parts),
        "sha_recomputed": sha_recomputed,
        "frozen_inputs_ok": frozen_inputs_ok,
        "counters_conserved": counters_conserved,
        "fresh_parts_verified": fresh_parts_verified,
        "raw_sources_verified": raw_sources_verified,
        "raw_candidate_reference_count": raw_reference_inventory[
            "candidate_reference_count"
        ],
        "raw_bound_reference_count": raw_reference_inventory[
            "bound_reference_count"
        ],
        "raw_unresolved_reference_count": raw_reference_inventory[
            "unresolved_reference_count"
        ],
        "raw_unique_bound_episode_count": raw_reference_inventory[
            "unique_bound_episode_count"
        ],
        "game_completeness_scope": "admitted_candidate_span_only",
        "raw_source_completeness_scope": (
            "referenced_kaggle_raw_episode_inventory_not_game_completeness"
        ),
        "raw_source_completeness_proven": raw_sources_verified,
        "tensor_registry_status": "not_audited_deferred_to_separate_x_registry",
        "admission_v0_immutable": immutable_ok,
        "forbidden_work_performed": [],
    }
    receipt["receipt_sha256"] = sha256_json(receipt)
    receipt_path = output_dir / "FULL_CORPUS_SEMANTIC_AUDIT_RECEIPT.json"
    _atomic_json(receipt_path, receipt)
    print(
        json.dumps({
            "event": "audit_complete",
            "status": status,
            "inspected_candidates": aggregate["total_candidates"],
            "outcomes": outcomes,
            "receipt_sha256": receipt["receipt_sha256"],
        }),
        flush=True,
    )

    return 0 if status == "COMPLETE_READ_ONLY_EXACT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
