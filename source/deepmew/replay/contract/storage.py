"""Append-safe storage for validated canonical replay decisions.

Shard files are immutable, self-hashed objects.  A shard becomes externally
visible only after its catalog transaction commits and the root manifest is
atomically replaced.  Admission V0 and observation assets remain read-only
inputs; this module stores only canonical decision records and their receipts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from deepmew.replay.contract.identities import (
    canonical_decision_id,
    canonical_json_bytes,
    canonical_payload,
    identity,
    is_lowercase_sha256,
    sha256_json,
)
from deepmew.replay.contract.schemas import (
    CHOSEN_ACTION_SCHEMA_VERSION,
    CONTRACT_SCHEMA_VERSION,
    DECISION_RECORD_SCHEMA_VERSION,
    LEGAL_ACTION_SCHEMA_VERSION,
    OBSERVATION_ASSET_SCHEMA_VERSION,
    SOURCE_EVENT_SCHEMA_VERSION,
    VALIDATOR_VERSION,
    CanonicalDecisionRecord,
    ValidationFailure,
    ValidationOutcome,
    ValidationResult,
)
from deepmew.replay.contract.validation import (
    CanonicalDecisionValidationHardStop,
    ValidatedCanonicalDecisionEnvelope,
    revalidate_decision_envelope,
)


STORAGE_SCHEMA_VERSION = "deepmew.canonical_decision_storage.v1"
SHARD_SCHEMA_VERSION = "deepmew.canonical_decision_shard.v1"
RECORD_IDENTITY_VERSION = "deepmew.canonical_decision_storage_record.v1"
ORDERED_ID_HASH_ALGORITHM = "sha256(canonical-json-array-of-record-ids)"

MANIFEST_FILENAME = "manifest.json"
CATALOG_FILENAME = "catalog.sqlite3"
SHARD_DIRECTORY = "shards"

POPULATION_VALID_ACTOR_AND_CRITIC = "valid_actor_and_critic"
POPULATION_FORCED_CRITIC_ONLY = "forced_critic_only"
POPULATION_UNRESOLVED_CRITIC_ONLY = "unresolved_critic_only"
POPULATION_REJECTED = "rejected"
POPULATIONS = (
    POPULATION_VALID_ACTOR_AND_CRITIC,
    POPULATION_FORCED_CRITIC_ONLY,
    POPULATION_UNRESOLVED_CRITIC_ONLY,
    POPULATION_REJECTED,
)
POSITIVE_POPULATIONS = frozenset({
    POPULATION_VALID_ACTOR_AND_CRITIC,
    POPULATION_FORCED_CRITIC_ONLY,
})

CanonicalDecisionAppendInput = (
    CanonicalDecisionRecord | ValidatedCanonicalDecisionEnvelope
)


class CanonicalDecisionStorageHardStop(RuntimeError):
    """Storage identity, validation, coverage, or immutable content drifted."""


@dataclass(frozen=True)
class CanonicalDecisionStorageIdentity:
    """Immutable identities shared by every shard in one audit store."""

    admission_identity: str
    source_corpus_sha256: str
    causal_aligner_version: str
    legality_validator_version: str
    contract_implementation_sha256: str
    causal_aligner_implementation_sha256: str
    legality_validator_implementation_sha256: str
    action_schema_implementation_sha256: str
    action_schema_version: str = CHOSEN_ACTION_SCHEMA_VERSION

    def validate(self) -> None:
        for name in (
            "causal_aligner_version",
            "legality_validator_version",
            "action_schema_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise CanonicalDecisionStorageHardStop(
                    f"storage identity {name} must be a non-empty string"
                )
        _require_sha256(self.admission_identity, "admission_identity")
        _require_sha256(self.source_corpus_sha256, "source_corpus_sha256")
        _require_sha256(
            self.contract_implementation_sha256,
            "contract_implementation_sha256",
        )
        _require_sha256(
            self.causal_aligner_implementation_sha256,
            "causal_aligner_implementation_sha256",
        )
        _require_sha256(
            self.legality_validator_implementation_sha256,
            "legality_validator_implementation_sha256",
        )
        _require_sha256(
            self.action_schema_implementation_sha256,
            "action_schema_implementation_sha256",
        )

    @property
    def sha256(self) -> str:
        self.validate()

        return sha256_json(asdict(self))


@dataclass(frozen=True)
class StoredCanonicalDecision:
    record_id: str
    record_sha256: str
    population: str
    record: CanonicalDecisionRecord


@dataclass(frozen=True)
class ShardCommitResult:
    shard_index: int | None
    committed_record_ids: tuple[str, ...]
    already_present_record_ids: tuple[str, ...]
    shard_sha256: str | None

    @property
    def committed_count(self) -> int:

        return len(self.committed_record_ids)

    @property
    def already_present_count(self) -> int:

        return len(self.already_present_record_ids)


def storage_record_id(record: CanonicalDecisionRecord) -> str:
    """Return the stable source-candidate identity used for exact resume."""

    if not isinstance(record, CanonicalDecisionRecord):
        raise CanonicalDecisionStorageHardStop(
            "storage accepts CanonicalDecisionRecord values only"
        )
    source_identity = {
        "version": RECORD_IDENTITY_VERSION,
        "admission_identity": record.admission_identity,
        "source_game_id": record.source_game_id,
        "actor_seat": record.actor_seat,
        "causal_decision_index": record.causal_decision_index,
        "source_lineage": record.source_lineage,
    }

    return identity("canonical-decision-record", source_identity)


def ordered_record_ids_sha256(record_ids: Iterable[str]) -> str:
    """Hash an ordered ID stream as its canonical JSON array without buffering."""

    digest = hashlib.sha256()
    digest.update(b"[")
    for index, record_id in enumerate(record_ids):
        if not isinstance(record_id, str) or not record_id:
            raise CanonicalDecisionStorageHardStop(
                "record IDs must be non-empty strings"
            )
        if index:
            digest.update(b",")
        digest.update(canonical_json_bytes(record_id))
    digest.update(b"]")

    return digest.hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if not is_lowercase_sha256(value):
        raise CanonicalDecisionStorageHardStop(f"{name} must be a lowercase SHA-256")

    return value


def _schema_versions() -> dict[str, str]:

    return {
        "contract": CONTRACT_SCHEMA_VERSION,
        "source_event": SOURCE_EVENT_SCHEMA_VERSION,
        "observation_asset": OBSERVATION_ASSET_SCHEMA_VERSION,
        "legal_action_set": LEGAL_ACTION_SCHEMA_VERSION,
        "chosen_action": CHOSEN_ACTION_SCHEMA_VERSION,
        "decision_record": DECISION_RECORD_SCHEMA_VERSION,
        "validator": VALIDATOR_VERSION,
        "storage": STORAGE_SCHEMA_VERSION,
        "shard": SHARD_SCHEMA_VERSION,
        "record_identity": RECORD_IDENTITY_VERSION,
    }


def _expected_canonical_decision_id(record: CanonicalDecisionRecord) -> str:
    if record.chosen_action_id is None or record.resulting_transition_id is None:
        raise CanonicalDecisionStorageHardStop(
            "positive decision record lacks a chosen action or resulting transition identity"
        )

    return canonical_decision_id(
        schema_version=record.schema_version,
        admission_identity=record.admission_identity,
        source_game_id=record.source_game_id,
        actor_seat=record.actor_seat,
        causal_decision_index=record.causal_decision_index,
        pre_action_observation_id=record.pre_action_observation_id,
        legal_action_set_id=record.legal_action_set_id,
        chosen_action_id=record.chosen_action_id,
        resulting_transition_id=record.resulting_transition_id,
        causal_aligner_version=record.causal_aligner_version,
        causal_aligner_implementation_sha256=(
            record.causal_aligner_implementation_sha256
        ),
        legality_validator_version=record.legality_validator_version,
        legality_validator_implementation_sha256=(
            record.legality_validator_implementation_sha256
        ),
        action_schema_version=record.action_schema_version,
        action_schema_implementation_sha256=(
            record.action_schema_implementation_sha256
        ),
    )


def _population_for_record(record: CanonicalDecisionRecord) -> str:
    validation = record.validation
    if record.schema_version != DECISION_RECORD_SCHEMA_VERSION:
        raise CanonicalDecisionStorageHardStop("decision-record schema mismatch")
    if validation.validator_version != record.legality_validator_version:
        raise CanonicalDecisionStorageHardStop(
            "decision record and validation result disagree on validator version"
        )
    if validation.resulting_transition_id != record.resulting_transition_id:
        raise CanonicalDecisionStorageHardStop(
            "decision record and validation result disagree on transition identity"
        )
    if not isinstance(validation.outcome, ValidationOutcome):
        raise CanonicalDecisionStorageHardStop("validation outcome is not typed")
    if not isinstance(validation.failures, tuple) or any(
        not isinstance(failure, ValidationFailure) for failure in validation.failures
    ):
        raise CanonicalDecisionStorageHardStop("validation failures are not typed")
    if validation.first_failure is not None and validation.first_failure not in validation.failures:
        raise CanonicalDecisionStorageHardStop(
            "first validation failure is absent from the failure sequence"
        )
    if not math.isfinite(validation.actor_loss_weight):
        raise CanonicalDecisionStorageHardStop("actor loss weight is not finite")
    if not 0.0 <= validation.actor_loss_weight <= 1.0:
        raise CanonicalDecisionStorageHardStop("actor loss weight is outside [0,1]")

    if validation.outcome is ValidationOutcome.VALID_ACTOR_AND_CRITIC:
        if (
            validation.actor_loss_weight <= 0.0
            or not validation.critic_eligible
            or validation.forced_decision
            or not validation.canonical_decision_id
            or not record.chosen_action_id
        ):
            raise CanonicalDecisionStorageHardStop(
                "actor-visible record lacks positive actor-and-critic validation"
            )

        return POPULATION_VALID_ACTOR_AND_CRITIC

    if validation.outcome is ValidationOutcome.VALID_CRITIC_ONLY_FORCED:
        if (
            validation.actor_loss_weight != 0.0
            or not validation.critic_eligible
            or not validation.forced_decision
            or not validation.canonical_decision_id
            or not record.chosen_action_id
        ):
            raise CanonicalDecisionStorageHardStop(
                "forced critic-only record has inconsistent validation weights"
            )

        return POPULATION_FORCED_CRITIC_ONLY

    if validation.outcome is ValidationOutcome.VALID_CRITIC_ONLY_UNRESOLVED_ACTION:
        raise CanonicalDecisionStorageHardStop(
            "unresolved critic-only population is disabled in V1"
        )

    if validation.actor_loss_weight != 0.0 or validation.critic_eligible:
        raise CanonicalDecisionStorageHardStop(
            "rejected record cannot carry actor or critic eligibility"
        )
    if validation.canonical_decision_id is not None:
        raise CanonicalDecisionStorageHardStop(
            "rejected or unknown record cannot carry a canonical decision ID"
        )

    return POPULATION_REJECTED


def _validate_record_against_identity(
    record: CanonicalDecisionRecord,
    storage_identity: CanonicalDecisionStorageIdentity,
) -> str:
    _require_sha256(record.admission_identity, "record admission_identity")
    if record.admission_identity != storage_identity.admission_identity:
        raise CanonicalDecisionStorageHardStop(
            "record admission identity conflicts with the storage identity"
        )
    if record.causal_aligner_version != storage_identity.causal_aligner_version:
        raise CanonicalDecisionStorageHardStop(
            "record causal-aligner version conflicts with the storage identity"
        )
    if record.legality_validator_version != storage_identity.legality_validator_version:
        raise CanonicalDecisionStorageHardStop(
            "record legality-validator version conflicts with the storage identity"
        )
    if record.action_schema_version != storage_identity.action_schema_version:
        raise CanonicalDecisionStorageHardStop(
            "record action-schema version conflicts with the storage identity"
        )
    implementation_bindings = (
        (
            "causal_aligner_implementation_sha256",
            record.causal_aligner_implementation_sha256,
            storage_identity.causal_aligner_implementation_sha256,
        ),
        (
            "legality_validator_implementation_sha256",
            record.legality_validator_implementation_sha256,
            storage_identity.legality_validator_implementation_sha256,
        ),
        (
            "action_schema_implementation_sha256",
            record.action_schema_implementation_sha256,
            storage_identity.action_schema_implementation_sha256,
        ),
    )
    for name, record_sha256, storage_sha256 in implementation_bindings:
        _require_sha256(record_sha256, f"record {name}")
        if record_sha256 != storage_sha256:
            raise CanonicalDecisionStorageHardStop(
                f"record {name} conflicts with the storage identity"
            )
    if (
        not isinstance(record.source_game_id, str)
        or not record.source_game_id
        or type(record.actor_seat) is not int
        or record.actor_seat < 0
        or type(record.causal_decision_index) is not int
        or record.causal_decision_index < 0
    ):
        raise CanonicalDecisionStorageHardStop("record source decision identity is invalid")
    for name in ("pre_action_observation_id", "legal_action_set_id"):
        value = getattr(record, name)
        if not isinstance(value, str) or not value:
            raise CanonicalDecisionStorageHardStop(f"record {name} is missing")
    if not isinstance(record.source_lineage, Mapping) or not record.source_lineage:
        raise CanonicalDecisionStorageHardStop("record source lineage is missing")

    population = _population_for_record(record)
    expected_decision_id = (
        _expected_canonical_decision_id(record)
        if population in {
            POPULATION_VALID_ACTOR_AND_CRITIC,
            POPULATION_FORCED_CRITIC_ONLY,
        }
        else None
    )
    if record.validation.canonical_decision_id != expected_decision_id:
        raise CanonicalDecisionStorageHardStop(
            "record canonical decision ID conflicts with its semantic identity"
        )

    return population


def _record_from_payload(value: Any) -> CanonicalDecisionRecord:
    if not isinstance(value, Mapping):
        raise CanonicalDecisionStorageHardStop("stored decision record is not an object")
    expected_record_fields = {
        "admission_identity",
        "source_game_id",
        "actor_seat",
        "causal_decision_index",
        "pre_action_observation_id",
        "legal_action_set_id",
        "chosen_action_id",
        "resulting_transition_id",
        "causal_aligner_version",
        "causal_aligner_implementation_sha256",
        "legality_validator_version",
        "legality_validator_implementation_sha256",
        "action_schema_version",
        "action_schema_implementation_sha256",
        "validation",
        "source_lineage",
        "schema_version",
    }
    if set(value) != expected_record_fields:
        raise CanonicalDecisionStorageHardStop("stored decision-record fields are incomplete")
    validation_value = value["validation"]
    if not isinstance(validation_value, Mapping):
        raise CanonicalDecisionStorageHardStop("stored validation result is not an object")
    expected_validation_fields = {
        "outcome",
        "first_failure",
        "failures",
        "actor_loss_weight",
        "critic_eligible",
        "forced_decision",
        "canonical_decision_id",
        "resulting_transition_id",
        "validator_version",
        "diagnostics",
    }
    if set(validation_value) != expected_validation_fields:
        raise CanonicalDecisionStorageHardStop("stored validation-result fields are incomplete")
    try:
        failures = tuple(
            ValidationFailure(item) for item in validation_value["failures"]
        )
        first_failure_value = validation_value["first_failure"]
        first_failure = (
            ValidationFailure(first_failure_value)
            if first_failure_value is not None
            else None
        )
        validation = ValidationResult(
            outcome=ValidationOutcome(validation_value["outcome"]),
            first_failure=first_failure,
            failures=failures,
            actor_loss_weight=float(validation_value["actor_loss_weight"]),
            critic_eligible=validation_value["critic_eligible"],
            forced_decision=validation_value["forced_decision"],
            canonical_decision_id=validation_value["canonical_decision_id"],
            resulting_transition_id=validation_value["resulting_transition_id"],
            validator_version=validation_value["validator_version"],
            diagnostics=dict(validation_value["diagnostics"]),
        )
        record = CanonicalDecisionRecord(
            admission_identity=value["admission_identity"],
            source_game_id=value["source_game_id"],
            actor_seat=value["actor_seat"],
            causal_decision_index=value["causal_decision_index"],
            pre_action_observation_id=value["pre_action_observation_id"],
            legal_action_set_id=value["legal_action_set_id"],
            chosen_action_id=value["chosen_action_id"],
            resulting_transition_id=value["resulting_transition_id"],
            causal_aligner_version=value["causal_aligner_version"],
            causal_aligner_implementation_sha256=value[
                "causal_aligner_implementation_sha256"
            ],
            legality_validator_version=value["legality_validator_version"],
            legality_validator_implementation_sha256=value[
                "legality_validator_implementation_sha256"
            ],
            action_schema_version=value["action_schema_version"],
            action_schema_implementation_sha256=value[
                "action_schema_implementation_sha256"
            ],
            validation=validation,
            source_lineage=dict(value["source_lineage"]),
            schema_version=value["schema_version"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CanonicalDecisionStorageHardStop(
            "stored decision record cannot be decoded"
        ) from exc

    return record


def _prepare_record(
    record: CanonicalDecisionRecord,
    storage_identity: CanonicalDecisionStorageIdentity,
) -> StoredCanonicalDecision:
    population = _validate_record_against_identity(record, storage_identity)
    record_payload = canonical_payload(record)

    return StoredCanonicalDecision(
        record_id=storage_record_id(record),
        record_sha256=sha256_json(record_payload),
        population=population,
        record=record,
    )


def _prepare_append_input(
    value: CanonicalDecisionAppendInput,
    storage_identity: CanonicalDecisionStorageIdentity,
) -> StoredCanonicalDecision:
    if type(value) is ValidatedCanonicalDecisionEnvelope:
        try:
            record = revalidate_decision_envelope(value)
        except CanonicalDecisionValidationHardStop as exc:
            raise CanonicalDecisionStorageHardStop(
                "validated decision envelope failed independent replay"
            ) from exc

        return _prepare_record(record, storage_identity)

    if type(value) is not CanonicalDecisionRecord:
        raise CanonicalDecisionStorageHardStop(
            "append accepts negative decision records or builder-sealed envelopes only"
        )
    stored = _prepare_record(value, storage_identity)
    if stored.population in POSITIVE_POPULATIONS:
        raise CanonicalDecisionStorageHardStop(
            "record-only positive append is forbidden; a builder-sealed validated "
            "decision envelope is required"
        )

    return stored


def _population_summary(
    stored: Iterable[StoredCanonicalDecision],
) -> dict[str, dict[str, Any]]:
    by_population = {population: [] for population in POPULATIONS}
    for item in stored:
        by_population[item.population].append(item.record_id)

    return {
        population: {
            "count": len(by_population[population]),
            "ordered_record_ids_sha256": ordered_record_ids_sha256(
                by_population[population]
            ),
        }
        for population in POPULATIONS
    }


def _shard_payload(
    *,
    shard_index: int,
    previous_shard_sha256: str | None,
    storage_identity: CanonicalDecisionStorageIdentity,
    records: Sequence[StoredCanonicalDecision],
) -> dict[str, Any]:
    body = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "storage_schema_version": STORAGE_SCHEMA_VERSION,
        "storage_identity_sha256": storage_identity.sha256,
        "shard_index": shard_index,
        "previous_shard_sha256": previous_shard_sha256,
        "record_count": len(records),
        "ordered_record_ids_sha256": ordered_record_ids_sha256(
            item.record_id for item in records
        ),
        "populations": _population_summary(records),
        "records": [
            {
                "record_id": item.record_id,
                "record_sha256": item.record_sha256,
                "population": item.population,
                "record": canonical_payload(item.record),
            }
            for item in records
        ],
    }
    body["shard_payload_sha256"] = sha256_json(body)

    return body


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
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
            try:
                temporary.unlink()
            except OSError:
                pass
        raise


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CanonicalDecisionStorageHardStop(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise CanonicalDecisionStorageHardStop(f"{label} is not an object")

    return value


def _safe_shard_path(root: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise CanonicalDecisionStorageHardStop("shard path is missing")
    relative = Path(relative_path)
    if relative.is_absolute():
        raise CanonicalDecisionStorageHardStop("shard path must be relative")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise CanonicalDecisionStorageHardStop("shard path escapes storage root") from exc

    return resolved


def _decode_shard(
    path: Path,
    *,
    storage_identity: CanonicalDecisionStorageIdentity,
    expected_index: int,
    expected_previous_sha256: str | None,
    expected_entry: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], tuple[StoredCanonicalDecision, ...], str]:
    file_sha256 = _sha256_path(path)
    body = _read_json(path, f"canonical shard {expected_index}")
    provided_payload_sha256 = body.pop("shard_payload_sha256", None)
    if (
        not isinstance(provided_payload_sha256, str)
        or sha256_json(body) != provided_payload_sha256
    ):
        raise CanonicalDecisionStorageHardStop("canonical shard payload hash mismatch")
    body["shard_payload_sha256"] = provided_payload_sha256
    if (
        body.get("schema_version") != SHARD_SCHEMA_VERSION
        or body.get("storage_schema_version") != STORAGE_SCHEMA_VERSION
        or body.get("storage_identity_sha256") != storage_identity.sha256
        or body.get("shard_index") != expected_index
        or body.get("previous_shard_sha256") != expected_previous_sha256
    ):
        raise CanonicalDecisionStorageHardStop(
            "canonical shard schema, identity, index, or chain is inconsistent"
        )
    raw_records = body.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise CanonicalDecisionStorageHardStop("canonical shard is empty or incomplete")
    stored_records = []
    seen_record_ids = set()
    for raw in raw_records:
        if not isinstance(raw, Mapping) or set(raw) != {
            "record_id",
            "record_sha256",
            "population",
            "record",
        }:
            raise CanonicalDecisionStorageHardStop("canonical shard row is incomplete")
        record = _record_from_payload(raw["record"])
        stored = _prepare_record(record, storage_identity)
        if (
            raw["record_id"] != stored.record_id
            or raw["record_sha256"] != stored.record_sha256
            or raw["population"] != stored.population
        ):
            raise CanonicalDecisionStorageHardStop(
                "canonical shard row identity or population conflicts"
            )
        if stored.record_id in seen_record_ids:
            raise CanonicalDecisionStorageHardStop(
                "canonical shard contains a duplicate record ID"
            )
        seen_record_ids.add(stored.record_id)
        stored_records.append(stored)
    records = tuple(stored_records)
    if (
        body.get("record_count") != len(records)
        or body.get("ordered_record_ids_sha256")
        != ordered_record_ids_sha256(item.record_id for item in records)
        or body.get("populations") != _population_summary(records)
    ):
        raise CanonicalDecisionStorageHardStop(
            "canonical shard record coverage is inconsistent"
        )
    if expected_entry is not None:
        expected = {
            "shard_index": expected_index,
            "path": str(path.relative_to(path.parents[1])).replace("\\", "/"),
            "shard_sha256": file_sha256,
            "shard_payload_sha256": provided_payload_sha256,
            "previous_shard_sha256": expected_previous_sha256,
            "record_count": len(records),
            "ordered_record_ids_sha256": body["ordered_record_ids_sha256"],
            "populations": body["populations"],
        }
        if dict(expected_entry) != expected:
            raise CanonicalDecisionStorageHardStop(
                "canonical shard disagrees with its manifest entry"
            )

    return body, records, file_sha256


def _connect_catalog(path: Path, *, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    else:
        connection = sqlite3.connect(path)
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS shards (
                shard_index INTEGER PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                shard_sha256 TEXT NOT NULL UNIQUE,
                shard_payload_sha256 TEXT NOT NULL UNIQUE,
                previous_shard_sha256 TEXT,
                record_count INTEGER NOT NULL,
                ordered_record_ids_sha256 TEXT NOT NULL,
                populations_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS records (
                sequence_index INTEGER PRIMARY KEY,
                record_id TEXT NOT NULL UNIQUE,
                record_sha256 TEXT NOT NULL,
                population TEXT NOT NULL,
                canonical_decision_id TEXT,
                shard_index INTEGER NOT NULL,
                row_offset INTEGER NOT NULL,
                UNIQUE (shard_index, row_offset),
                FOREIGN KEY (shard_index) REFERENCES shards(shard_index)
            );
            """
        )
    quick_check = connection.execute("PRAGMA quick_check").fetchone()
    if quick_check != ("ok",):
        connection.close()
        raise CanonicalDecisionStorageHardStop("canonical storage catalog is corrupt")

    return connection


def _manifest_without_hash(manifest: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(manifest)
    value.pop("manifest_sha256", None)

    return value


def _validate_manifest(
    manifest: Mapping[str, Any],
    storage_identity: CanonicalDecisionStorageIdentity,
) -> None:
    provided_sha256 = manifest.get("manifest_sha256")
    if (
        not isinstance(provided_sha256, str)
        or sha256_json(_manifest_without_hash(manifest)) != provided_sha256
    ):
        raise CanonicalDecisionStorageHardStop("canonical storage manifest hash mismatch")
    if (
        manifest.get("schema_version") != STORAGE_SCHEMA_VERSION
        or manifest.get("storage_identity") != canonical_payload(storage_identity)
        or manifest.get("storage_identity_sha256") != storage_identity.sha256
        or manifest.get("schema_versions") != _schema_versions()
        or manifest.get("ordered_id_hash_algorithm") != ORDERED_ID_HASH_ALGORITHM
    ):
        raise CanonicalDecisionStorageHardStop(
            "canonical storage manifest schema or identity conflicts"
        )
    if manifest.get("status") not in {"in_progress", "complete"}:
        raise CanonicalDecisionStorageHardStop(
            "canonical storage manifest status is invalid"
        )
    shards = manifest.get("shards")
    populations = manifest.get("populations")
    if not isinstance(shards, list) or not isinstance(populations, Mapping):
        raise CanonicalDecisionStorageHardStop(
            "canonical storage manifest inventory is incomplete"
        )
    if set(populations) != set(POPULATIONS):
        raise CanonicalDecisionStorageHardStop(
            "canonical storage manifest populations are incomplete"
        )


class CanonicalDecisionShardStore:
    """Append immutable decision shards and publish a deterministic manifest."""

    def __init__(
        self,
        root: str | Path,
        storage_identity: CanonicalDecisionStorageIdentity,
    ) -> None:
        self.root = Path(root).resolve()
        self.storage_identity = storage_identity
        self.storage_identity.validate()
        self.manifest_path = self.root / MANIFEST_FILENAME
        self.catalog_path = self.root / CATALOG_FILENAME
        self.shard_directory = self.root / SHARD_DIRECTORY
        root_existed = self.root.exists()
        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_directory.mkdir(parents=True, exist_ok=True)
        if root_existed and self.manifest_path.exists() != self.catalog_path.exists():
            raise CanonicalDecisionStorageHardStop(
                "canonical storage manifest/catalog pair is incomplete"
            )
        self.connection = _connect_catalog(self.catalog_path)
        self._bind_catalog_identity()
        if not self.manifest_path.exists():
            if self._catalog_shard_count() or any(self.shard_directory.iterdir()):
                raise CanonicalDecisionStorageHardStop(
                    "canonical storage manifest is missing from nonempty storage"
                )
            self._write_manifest_atomic(self._build_manifest(status="in_progress"))
        self._verify_and_recover()

    def __enter__(self) -> CanonicalDecisionShardStore:

        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def _bind_catalog_identity(self) -> None:
        expected = {
            "schema_version": STORAGE_SCHEMA_VERSION,
            "storage_identity": json.dumps(
                canonical_payload(self.storage_identity),
                sort_keys=True,
                separators=(",", ":"),
            ),
            "storage_identity_sha256": self.storage_identity.sha256,
        }
        existing = dict(self.connection.execute("SELECT key,value FROM meta"))
        if existing and existing != expected:
            raise CanonicalDecisionStorageHardStop(
                "canonical storage catalog identity conflicts"
            )
        if not existing:
            self.connection.executemany(
                "INSERT INTO meta(key,value) VALUES (?,?)",
                expected.items(),
            )
            self.connection.commit()

    def _catalog_shard_count(self) -> int:

        return int(self.connection.execute("SELECT COUNT(*) FROM shards").fetchone()[0])

    def _catalog_record_count(self) -> int:

        return int(self.connection.execute("SELECT COUNT(*) FROM records").fetchone()[0])

    def _catalog_shards(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT shard_index,path,shard_sha256,shard_payload_sha256,"
            "previous_shard_sha256,record_count,ordered_record_ids_sha256,populations_json "
            "FROM shards ORDER BY shard_index"
        )

        return [
            {
                "shard_index": int(row[0]),
                "path": row[1],
                "shard_sha256": row[2],
                "shard_payload_sha256": row[3],
                "previous_shard_sha256": row[4],
                "record_count": int(row[5]),
                "ordered_record_ids_sha256": row[6],
                "populations": json.loads(row[7]),
            }
            for row in rows
        ]

    def _catalog_population_summary(self) -> dict[str, dict[str, Any]]:

        return {
            population: {
                "count": int(
                    self.connection.execute(
                        "SELECT COUNT(*) FROM records WHERE population=?",
                        (population,),
                    ).fetchone()[0]
                ),
                "ordered_record_ids_sha256": ordered_record_ids_sha256(
                    row[0]
                    for row in self.connection.execute(
                        "SELECT record_id FROM records WHERE population=? "
                        "ORDER BY sequence_index",
                        (population,),
                    )
                ),
            }
            for population in POPULATIONS
        }

    def _catalog_ordered_ids_sha256(self) -> str:

        return ordered_record_ids_sha256(
            row[0]
            for row in self.connection.execute(
                "SELECT record_id FROM records ORDER BY sequence_index"
            )
        )

    def _build_manifest(
        self,
        *,
        status: str,
        expected_record_count: int | None = None,
        expected_record_ids_sha256: str | None = None,
    ) -> dict[str, Any]:
        shards = self._catalog_shards()
        manifest = {
            "schema_version": STORAGE_SCHEMA_VERSION,
            "status": status,
            "storage_identity": canonical_payload(self.storage_identity),
            "storage_identity_sha256": self.storage_identity.sha256,
            "schema_versions": _schema_versions(),
            "ordered_id_hash_algorithm": ORDERED_ID_HASH_ALGORITHM,
            "record_count": self._catalog_record_count(),
            "ordered_record_ids_sha256": self._catalog_ordered_ids_sha256(),
            "shard_count": len(shards),
            "populations": self._catalog_population_summary(),
            "shards": shards,
            "expected_record_count": expected_record_count,
            "expected_record_ids_sha256": expected_record_ids_sha256,
        }
        manifest["manifest_sha256"] = sha256_json(manifest)

        return manifest

    def _write_manifest_atomic(self, manifest: Mapping[str, Any]) -> None:
        _atomic_write(
            self.manifest_path,
            canonical_json_bytes(manifest) + b"\n",
        )

    def _actual_shard_indices(self) -> list[int]:
        indices = []
        for path in self.shard_directory.iterdir():
            if path.name.startswith(".") or path.suffix == ".tmp":
                raise CanonicalDecisionStorageHardStop(
                    "canonical storage contains an incomplete temporary shard"
                )
            if not path.is_file():
                raise CanonicalDecisionStorageHardStop(
                    "canonical shard inventory contains a non-file"
                )
            name = path.name
            if (
                not name.startswith("shard_")
                or not name.endswith(".json")
                or not name[6:-5].isdigit()
            ):
                raise CanonicalDecisionStorageHardStop(
                    "canonical shard inventory contains an unexpected file"
                )
            indices.append(int(name[6:-5]))

        return sorted(indices)

    def _verify_catalog_shards(
        self,
        entries: Sequence[Mapping[str, Any]],
    ) -> None:
        expected_sequence = 0
        previous_sha256 = None
        seen_record_ids = set()
        for expected_index, entry in enumerate(entries):
            if entry.get("shard_index") != expected_index:
                raise CanonicalDecisionStorageHardStop(
                    "canonical catalog shard indices are not contiguous"
                )
            path = _safe_shard_path(self.root, entry.get("path"))
            if not path.exists():
                raise CanonicalDecisionStorageHardStop(
                    "canonical catalog references a missing shard"
                )
            _, records, _ = _decode_shard(
                path,
                storage_identity=self.storage_identity,
                expected_index=expected_index,
                expected_previous_sha256=previous_sha256,
                expected_entry=entry,
            )
            catalog_rows = self.connection.execute(
                "SELECT sequence_index,record_id,record_sha256,population,"
                "canonical_decision_id,row_offset FROM records "
                "WHERE shard_index=? ORDER BY row_offset",
                (expected_index,),
            ).fetchall()
            if len(catalog_rows) != len(records):
                raise CanonicalDecisionStorageHardStop(
                    "canonical catalog/shard row counts disagree"
                )
            for row_offset, (stored, catalog_row) in enumerate(
                zip(records, catalog_rows, strict=True)
            ):
                expected_row = (
                    expected_sequence,
                    stored.record_id,
                    stored.record_sha256,
                    stored.population,
                    stored.record.validation.canonical_decision_id,
                    row_offset,
                )
                if catalog_row != expected_row:
                    raise CanonicalDecisionStorageHardStop(
                        "canonical catalog row conflicts with immutable shard content"
                    )
                if stored.record_id in seen_record_ids:
                    raise CanonicalDecisionStorageHardStop(
                        "canonical storage contains a duplicate record ID"
                    )
                seen_record_ids.add(stored.record_id)
                expected_sequence += 1
            previous_sha256 = entry["shard_sha256"]
        if expected_sequence != self._catalog_record_count():
            raise CanonicalDecisionStorageHardStop(
                "canonical catalog contains omitted or unreferenced record IDs"
            )

    def _adopt_orphan_shard(
        self,
        *,
        shard_index: int,
        previous_sha256: str | None,
    ) -> None:
        path = self.shard_directory / f"shard_{shard_index:06d}.json"
        body, records, file_sha256 = _decode_shard(
            path,
            storage_identity=self.storage_identity,
            expected_index=shard_index,
            expected_previous_sha256=previous_sha256,
        )
        self._insert_catalog_shard(
            path=path,
            body=body,
            records=records,
            file_sha256=file_sha256,
        )

    def _verify_and_recover(self) -> None:
        manifest = _read_json(self.manifest_path, "canonical storage manifest")
        _validate_manifest(manifest, self.storage_identity)
        manifest_shards = manifest["shards"]
        catalog_shards = self._catalog_shards()
        actual_indices = self._actual_shard_indices()
        if actual_indices != list(range(len(actual_indices))):
            raise CanonicalDecisionStorageHardStop(
                "canonical shard file indices are not contiguous"
            )
        if len(manifest_shards) > len(catalog_shards):
            raise CanonicalDecisionStorageHardStop(
                "canonical manifest is ahead of its transactional catalog"
            )
        if len(catalog_shards) - len(manifest_shards) > 1:
            raise CanonicalDecisionStorageHardStop(
                "canonical catalog is more than one commit ahead of its manifest"
            )
        if len(actual_indices) - len(catalog_shards) > 1:
            raise CanonicalDecisionStorageHardStop(
                "canonical storage contains multiple unpublished shards"
            )
        if len(actual_indices) < len(catalog_shards):
            raise CanonicalDecisionStorageHardStop(
                "canonical catalog references an incomplete shard inventory"
            )
        if manifest_shards != catalog_shards[: len(manifest_shards)]:
            raise CanonicalDecisionStorageHardStop(
                "canonical manifest and catalog shard inventories conflict"
            )
        self._verify_catalog_shards(catalog_shards)
        if manifest["status"] == "complete" and (
            len(catalog_shards) != len(manifest_shards)
            or len(actual_indices) != len(catalog_shards)
        ):
            raise CanonicalDecisionStorageHardStop(
                "completed canonical storage has unpublished content"
            )
        if len(actual_indices) == len(catalog_shards) + 1:
            previous_sha256 = (
                catalog_shards[-1]["shard_sha256"] if catalog_shards else None
            )
            self._adopt_orphan_shard(
                shard_index=len(catalog_shards),
                previous_sha256=previous_sha256,
            )
            catalog_shards = self._catalog_shards()
            self._verify_catalog_shards(catalog_shards)
        if len(catalog_shards) != len(manifest_shards):
            self._write_manifest_atomic(self._build_manifest(status="in_progress"))
            manifest = _read_json(self.manifest_path, "canonical storage manifest")
            _validate_manifest(manifest, self.storage_identity)
        expected_manifest = self._build_manifest(
            status=manifest["status"],
            expected_record_count=manifest.get("expected_record_count"),
            expected_record_ids_sha256=manifest.get("expected_record_ids_sha256"),
        )
        if manifest != expected_manifest:
            raise CanonicalDecisionStorageHardStop(
                "canonical manifest coverage disagrees with committed shards"
            )
        if manifest["status"] == "complete":
            if (
                manifest["expected_record_count"] != manifest["record_count"]
                or manifest["expected_record_ids_sha256"]
                != manifest["ordered_record_ids_sha256"]
            ):
                raise CanonicalDecisionStorageHardStop(
                    "completed canonical storage lacks exact coverage proof"
                )

    @property
    def status(self) -> str:

        return _read_json(
            self.manifest_path,
            "canonical storage manifest",
        )["status"]

    def _insert_catalog_shard(
        self,
        *,
        path: Path,
        body: Mapping[str, Any],
        records: Sequence[StoredCanonicalDecision],
        file_sha256: str,
    ) -> None:
        shard_index = int(body["shard_index"])
        sequence_start = self._catalog_record_count()
        relative_path = str(path.relative_to(self.root)).replace("\\", "/")
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            self.connection.execute(
                "INSERT INTO shards VALUES (?,?,?,?,?,?,?,?)",
                (
                    shard_index,
                    relative_path,
                    file_sha256,
                    body["shard_payload_sha256"],
                    body["previous_shard_sha256"],
                    len(records),
                    body["ordered_record_ids_sha256"],
                    json.dumps(
                        body["populations"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ),
            )
            self.connection.executemany(
                "INSERT INTO records VALUES (?,?,?,?,?,?,?)",
                [
                    (
                        sequence_start + row_offset,
                        item.record_id,
                        item.record_sha256,
                        item.population,
                        item.record.validation.canonical_decision_id,
                        shard_index,
                        row_offset,
                    )
                    for row_offset, item in enumerate(records)
                ],
            )
            self.connection.commit()
        except sqlite3.Error as exc:
            self.connection.rollback()
            raise CanonicalDecisionStorageHardStop(
                "canonical shard catalog transaction failed"
            ) from exc

    def _existing_record(
        self,
        record_id: str,
    ) -> tuple[str, str] | None:
        row = self.connection.execute(
            "SELECT record_sha256,population FROM records WHERE record_id=?",
            (record_id,),
        ).fetchone()
        if row is None:

            return None

        return str(row[0]), str(row[1])

    def missing_record_ids(
        self,
        records: Iterable[CanonicalDecisionAppendInput],
    ) -> tuple[str, ...]:
        """Return exact missing IDs while rejecting duplicates and conflicts."""

        prepared = [
            _prepare_append_input(record, self.storage_identity)
            for record in records
        ]
        seen = set()
        missing = []
        for item in prepared:
            if item.record_id in seen:
                raise CanonicalDecisionStorageHardStop(
                    "resume input contains a duplicate record ID"
                )
            seen.add(item.record_id)
            existing = self._existing_record(item.record_id)
            if existing is None:
                missing.append(item.record_id)
            elif existing != (item.record_sha256, item.population):
                raise CanonicalDecisionStorageHardStop(
                    "resume record identity conflicts with committed content"
                )

        return tuple(missing)

    def append(
        self,
        records: Sequence[CanonicalDecisionAppendInput],
    ) -> ShardCommitResult:
        """Validate a batch fully, then publish one immutable shard."""

        if self.status != "in_progress":
            raise CanonicalDecisionStorageHardStop(
                "completed canonical storage cannot be appended"
            )
        if not records:
            raise CanonicalDecisionStorageHardStop("cannot append an empty shard")
        prepared = [
            _prepare_append_input(record, self.storage_identity)
            for record in records
        ]
        seen = set()
        new_records = []
        already_present = []
        for item in prepared:
            if item.record_id in seen:
                raise CanonicalDecisionStorageHardStop(
                    "append batch contains a duplicate record ID"
                )
            seen.add(item.record_id)
            existing = self._existing_record(item.record_id)
            if existing is None:
                new_records.append(item)
            elif existing == (item.record_sha256, item.population):
                already_present.append(item.record_id)
            else:
                raise CanonicalDecisionStorageHardStop(
                    "record identity conflicts with committed content"
                )
        if not new_records:

            return ShardCommitResult(
                shard_index=None,
                committed_record_ids=(),
                already_present_record_ids=tuple(already_present),
                shard_sha256=None,
            )
        shard_index = self._catalog_shard_count()
        previous_row = self.connection.execute(
            "SELECT shard_sha256 FROM shards ORDER BY shard_index DESC LIMIT 1"
        ).fetchone()
        previous_sha256 = previous_row[0] if previous_row is not None else None
        body = _shard_payload(
            shard_index=shard_index,
            previous_shard_sha256=previous_sha256,
            storage_identity=self.storage_identity,
            records=new_records,
        )
        path = self.shard_directory / f"shard_{shard_index:06d}.json"
        if path.exists():
            raise CanonicalDecisionStorageHardStop(
                "next canonical shard path already exists"
            )
        _atomic_write(path, canonical_json_bytes(body) + b"\n")
        file_sha256 = _sha256_path(path)
        self._insert_catalog_shard(
            path=path,
            body=body,
            records=new_records,
            file_sha256=file_sha256,
        )
        self._write_manifest_atomic(self._build_manifest(status="in_progress"))

        return ShardCommitResult(
            shard_index=shard_index,
            committed_record_ids=tuple(item.record_id for item in new_records),
            already_present_record_ids=tuple(already_present),
            shard_sha256=file_sha256,
        )

    def finalize(
        self,
        *,
        expected_record_ids: Iterable[str] | None = None,
        expected_record_count: int | None = None,
        expected_record_ids_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Publish a complete receipt only after exact ordered coverage proof."""

        if self.status != "in_progress":
            raise CanonicalDecisionStorageHardStop(
                "canonical storage is already complete"
            )
        if expected_record_ids is not None:
            expected = tuple(expected_record_ids)
            if len(expected) != len(set(expected)):
                raise CanonicalDecisionStorageHardStop(
                    "expected coverage contains duplicate record IDs"
                )
            derived_count = len(expected)
            derived_sha256 = ordered_record_ids_sha256(expected)
            if expected_record_count is not None and expected_record_count != derived_count:
                raise CanonicalDecisionStorageHardStop(
                    "expected count conflicts with expected record IDs"
                )
            if (
                expected_record_ids_sha256 is not None
                and expected_record_ids_sha256 != derived_sha256
            ):
                raise CanonicalDecisionStorageHardStop(
                    "expected hash conflicts with expected record IDs"
                )
            expected_record_count = derived_count
            expected_record_ids_sha256 = derived_sha256
        if expected_record_count is None or expected_record_ids_sha256 is None:
            raise CanonicalDecisionStorageHardStop(
                "finalization requires exact expected record count and ordered ID hash"
            )
        if expected_record_count < 0:
            raise CanonicalDecisionStorageHardStop(
                "expected record count cannot be negative"
            )
        _require_sha256(
            expected_record_ids_sha256,
            "expected_record_ids_sha256",
        )
        actual_count = self._catalog_record_count()
        actual_sha256 = self._catalog_ordered_ids_sha256()
        if (
            actual_count != expected_record_count
            or actual_sha256 != expected_record_ids_sha256
        ):
            raise CanonicalDecisionStorageHardStop(
                "canonical storage coverage has duplicate, omitted, or reordered record IDs"
            )
        manifest = self._build_manifest(
            status="complete",
            expected_record_count=expected_record_count,
            expected_record_ids_sha256=expected_record_ids_sha256,
        )
        self._write_manifest_atomic(manifest)

        return manifest


class CanonicalDecisionStorageIndex:
    """Read only a complete, manifest-bound canonical decision store."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path).resolve(strict=True)
        self.root = self.manifest_path.parent
        manifest = _read_json(self.manifest_path, "canonical storage manifest")
        try:
            storage_identity = CanonicalDecisionStorageIdentity(
                **manifest["storage_identity"]
            )
        except (KeyError, TypeError) as exc:
            raise CanonicalDecisionStorageHardStop(
                "canonical storage identity cannot be decoded"
            ) from exc
        storage_identity.validate()
        _validate_manifest(manifest, storage_identity)
        if manifest["status"] != "complete":
            raise CanonicalDecisionStorageHardStop(
                "canonical decision index requires a complete manifest"
            )
        catalog_path = self.root / CATALOG_FILENAME
        if not catalog_path.exists():
            raise CanonicalDecisionStorageHardStop(
                "canonical decision index catalog is missing"
            )
        connection = _connect_catalog(catalog_path, readonly=True)
        try:
            expected_meta = {
                "schema_version": STORAGE_SCHEMA_VERSION,
                "storage_identity": json.dumps(
                    canonical_payload(storage_identity),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "storage_identity_sha256": storage_identity.sha256,
            }
            if dict(connection.execute("SELECT key,value FROM meta")) != expected_meta:
                raise CanonicalDecisionStorageHardStop(
                    "canonical decision index catalog identity conflicts"
                )
            catalog_entries = [
                {
                    "shard_index": int(row[0]),
                    "path": row[1],
                    "shard_sha256": row[2],
                    "shard_payload_sha256": row[3],
                    "previous_shard_sha256": row[4],
                    "record_count": int(row[5]),
                    "ordered_record_ids_sha256": row[6],
                    "populations": json.loads(row[7]),
                }
                for row in connection.execute(
                    "SELECT shard_index,path,shard_sha256,shard_payload_sha256,"
                    "previous_shard_sha256,record_count,ordered_record_ids_sha256,"
                    "populations_json FROM shards ORDER BY shard_index"
                )
            ]
            if catalog_entries != manifest["shards"]:
                raise CanonicalDecisionStorageHardStop(
                    "canonical decision index manifest/catalog conflict"
                )
            previous_sha256 = None
            sequence_index = 0
            population_ids = {population: [] for population in POPULATIONS}
            all_ids = []
            for shard_index, entry in enumerate(catalog_entries):
                path = _safe_shard_path(self.root, entry["path"])
                _, records, _ = _decode_shard(
                    path,
                    storage_identity=storage_identity,
                    expected_index=shard_index,
                    expected_previous_sha256=previous_sha256,
                    expected_entry=entry,
                )
                catalog_rows = connection.execute(
                    "SELECT sequence_index,record_id,record_sha256,population,"
                    "canonical_decision_id,row_offset FROM records "
                    "WHERE shard_index=? ORDER BY row_offset",
                    (shard_index,),
                ).fetchall()
                if len(catalog_rows) != len(records):
                    raise CanonicalDecisionStorageHardStop(
                        "canonical decision index row coverage conflicts"
                    )
                for row_offset, (stored, catalog_row) in enumerate(
                    zip(records, catalog_rows, strict=True)
                ):
                    expected_row = (
                        sequence_index,
                        stored.record_id,
                        stored.record_sha256,
                        stored.population,
                        stored.record.validation.canonical_decision_id,
                        row_offset,
                    )
                    if catalog_row != expected_row:
                        raise CanonicalDecisionStorageHardStop(
                            "canonical decision index catalog row conflicts"
                        )
                    all_ids.append(stored.record_id)
                    population_ids[stored.population].append(stored.record_id)
                    sequence_index += 1
                previous_sha256 = entry["shard_sha256"]
            populations = {
                population: {
                    "count": len(population_ids[population]),
                    "ordered_record_ids_sha256": ordered_record_ids_sha256(
                        population_ids[population]
                    ),
                }
                for population in POPULATIONS
            }
            if (
                sequence_index != manifest["record_count"]
                or ordered_record_ids_sha256(all_ids)
                != manifest["ordered_record_ids_sha256"]
                or populations != manifest["populations"]
                or manifest["expected_record_count"] != manifest["record_count"]
                or manifest["expected_record_ids_sha256"]
                != manifest["ordered_record_ids_sha256"]
            ):
                raise CanonicalDecisionStorageHardStop(
                    "canonical decision index exact coverage proof failed"
                )
        finally:
            connection.close()
        self.storage_identity = storage_identity
        self.manifest = manifest

    def iter_stored_records(
        self,
        populations: Iterable[str] | None = None,
    ) -> Iterator[StoredCanonicalDecision]:
        selected = set(populations) if populations is not None else set(POPULATIONS)
        if not selected.issubset(POPULATIONS):
            raise CanonicalDecisionStorageHardStop(
                "requested canonical population is unknown"
            )
        previous_sha256 = None
        for shard_index, entry in enumerate(self.manifest["shards"]):
            path = _safe_shard_path(self.root, entry["path"])
            _, records, _ = _decode_shard(
                path,
                storage_identity=self.storage_identity,
                expected_index=shard_index,
                expected_previous_sha256=previous_sha256,
                expected_entry=entry,
            )
            for stored in records:
                if stored.population in selected:
                    yield stored
            previous_sha256 = entry["shard_sha256"]

    def iter_records(
        self,
        populations: Iterable[str] | None = None,
    ) -> Iterator[CanonicalDecisionRecord]:
        for stored in self.iter_stored_records(populations):
            yield stored.record

    def iter_actor_records(self) -> Iterator[CanonicalDecisionRecord]:

        return self.iter_records((POPULATION_VALID_ACTOR_AND_CRITIC,))

    def iter_critic_records(self) -> Iterator[CanonicalDecisionRecord]:

        return self.iter_records(
            (
                POPULATION_VALID_ACTOR_AND_CRITIC,
                POPULATION_FORCED_CRITIC_ONLY,
            )
        )

    def iter_rejected_records(self) -> Iterator[CanonicalDecisionRecord]:

        return self.iter_records((POPULATION_REJECTED,))

    def record_ids(self) -> tuple[str, ...]:

        return tuple(item.record_id for item in self.iter_stored_records())


__all__ = [
    "CATALOG_FILENAME",
    "MANIFEST_FILENAME",
    "ORDERED_ID_HASH_ALGORITHM",
    "POPULATION_FORCED_CRITIC_ONLY",
    "POPULATION_REJECTED",
    "POPULATION_UNRESOLVED_CRITIC_ONLY",
    "POPULATION_VALID_ACTOR_AND_CRITIC",
    "POPULATIONS",
    "RECORD_IDENTITY_VERSION",
    "SHARD_DIRECTORY",
    "SHARD_SCHEMA_VERSION",
    "STORAGE_SCHEMA_VERSION",
    "CanonicalDecisionShardStore",
    "CanonicalDecisionAppendInput",
    "CanonicalDecisionStorageHardStop",
    "CanonicalDecisionStorageIdentity",
    "CanonicalDecisionStorageIndex",
    "ShardCommitResult",
    "StoredCanonicalDecision",
    "ordered_record_ids_sha256",
    "storage_record_id",
]
