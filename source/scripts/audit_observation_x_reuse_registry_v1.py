"""Build a fail-closed registry for exact observation-to-X tensor reuse.

The cache coordinate contract is deliberately local and explicit:

    global X row = cumulative cache-part row_start + transition local row

The semantic candidate coordinate is separate:

    global candidate row = cumulative Admission binding start + admitted ordinal

where admitted ordinal is the zero-based position in ``ORDER BY local_row``;
Admission ``local_row`` values may be sparse and are never candidate ordinals.

``state_before.feature_row`` is a unit-local historical field and is never used
as an X coordinate.  Production runs reopen the immutable Admission database
directly and reconstruct aligned pre-action observations through the frozen
shared causal-alignment contract.  The inline candidate-manifest adapter exists
only for bounded synthetic tests.  Every Admission reference enters a resumable
SQLite ledger, including references that fail closed before yielding an asset;
only verified assets are deduplicated.  Inputs are opened read-only and
hash-verified before use; writes stay under ``output_dir`` as resumable scratch,
compact checkpoints, and final receipts.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import os
import re
import sqlite3
import sys
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from deepmew.replay.canonical_actor_dataset import canonical_actor_feature_observation


CACHE_SCHEMA_VERSION = "deepmew.value_lab_cache.v2"
UNIT_MANIFEST_SCHEMA_VERSION = "deepmew.value_q_conversion_unit.v1"
TRANSITION_SCHEMA_VERSION = "deepmew.actor_transition.v1"
CANDIDATE_MANIFEST_SCHEMA_VERSION = "deepmew.observation_x_reuse_candidates.v1"
CANDIDATE_ROW_SCHEMA_VERSION = "deepmew.observation_x_reuse_candidate.v1"
CHECKPOINT_SCHEMA_VERSION = "deepmew.observation_x_reuse_registry_part.v1"
PROGRESS_SCHEMA_VERSION = "deepmew.observation_x_reuse_registry_progress.v1"
REGISTRY_SCHEMA_VERSION = "deepmew.observation_x_reuse_registry.v1"
RECEIPT_SCHEMA_VERSION = "deepmew.observation_x_reuse_registry_receipt.v1"

X_FILENAME = "X.f32.bin"
FEATURE_DIM = 144
FEATURE_DTYPE = "float32"
FLOAT32_BYTES = np.dtype("<f4").itemsize
X_ROW_BYTES = FEATURE_DIM * FLOAT32_BYTES
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
PART_PATTERN = re.compile(r"^part_(\d+)$")
SOURCE_OBSERVATION_FIELD_BY_SCHEMA: Mapping[str, Mapping[str, str]] = {
    "deepmew.replay_action_row.v1": {"raw_observation": "state_json"},
}

FeatureEncoder = Callable[[dict[str, Any], dict[str, Any]], np.ndarray]
ObservationNormalizer = Callable[[Any], dict[str, Any]]

DEFAULT_ADMISSION_MANIFEST = ROOT / ".research" / "terminal_league_starmie_grimmsnarl_v0" / (
    "e1_starmie_complete_enhanced_generation_20260714"
) / "production_admission" / "admission.manifest.json"
DEFAULT_SEMANTIC_AUDIT_DIR = (
    ROOT / ".research" / "centralized_causal_replay_contract_and_derived_v1_20260714"
)


class ObservationXReuseAuditError(ValueError):
    """Raised when an immutable input or proof contract cannot be verified."""


@dataclass(frozen=True)
class Candidate:
    asset_id: str
    source_game_id: str
    source_event_position: int
    acting_seat: int
    public_visibility_schema: str
    observation_feature_version: str
    normalized_observation_sha256: str
    normalized_observation: dict[str, Any]
    source_observation_sha256_aliases: tuple[str, ...]
    source_record: dict[str, Any]


@dataclass(frozen=True)
class CacheBinding:
    index: int
    binding_id: str
    source_id: str
    unit_id: str
    part: str
    part_number: int
    row_start: int
    rows: int
    transition_path: Path
    transition_sha256: str
    unit_manifest_path: Path
    unit_manifest_sha256: str
    source_sha256_by_path: Mapping[str, str]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_path(path: Path, chunk_size: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise ObservationXReuseAuditError(f"cannot hash input file: {path}") from exc

    return digest.hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ObservationXReuseAuditError(f"{field} must be a SHA-256 hex digest")

    return value.lower()


def _require_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ObservationXReuseAuditError(f"{field} must be an integer >= {minimum}")

    return value


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("rt", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ObservationXReuseAuditError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ObservationXReuseAuditError(f"expected JSON object: {path}")

    return value


def _serialized_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_serialized_json(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _seal_payload(payload: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    sealed = dict(payload)
    sealed[field] = _stable_sha256(payload)

    return sealed


def _verify_sealed_payload(value: Mapping[str, Any], *, field: str, label: str) -> None:
    expected = _require_sha256(value.get(field), field=f"{label} {field}")
    body = {key: item for key, item in value.items() if key != field}
    if _stable_sha256(body) != expected:
        raise ObservationXReuseAuditError(f"{label} self-hash mismatch")


def _path_key(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _paths_equal(left: str | Path, right: str | Path) -> bool:
    return _path_key(left) == _path_key(right)


def _resolve_relative(path_value: str, *, owner: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = owner.parent / path

    return path.resolve()


def _resolve_inside(root: Path, relative_value: Any, *, field: str) -> Path:
    if not isinstance(relative_value, str) or not relative_value:
        raise ObservationXReuseAuditError(f"{field} is missing")
    relative = Path(relative_value)
    if relative.is_absolute():
        candidate = relative.resolve()
    else:
        candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ObservationXReuseAuditError(f"{field} escapes the canonical root") from exc

    return candidate


def _open_jsonl(path: Path):
    if path.suffix.casefold() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")

    return path.open("rt", encoding="utf-8")


def _validate_source_hashes(unit_manifest: Mapping[str, Any]) -> dict[str, str]:
    source_hashes = unit_manifest.get("source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ObservationXReuseAuditError("unit manifest lacks source-file hash bindings")
    normalized: dict[str, str] = {}
    for path_value, sha_value in source_hashes.items():
        if not isinstance(path_value, str) or not Path(path_value).is_absolute():
            raise ObservationXReuseAuditError("unit source path must be absolute")
        key = _path_key(path_value)
        digest = _require_sha256(sha_value, field="unit source SHA")
        if key in normalized and normalized[key] != digest:
            raise ObservationXReuseAuditError("unit source path has conflicting hashes")
        normalized[key] = digest

    return normalized


def _load_cache_bindings(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
) -> tuple[dict[str, Any], Path, str, Path, str, int, tuple[CacheBinding, ...]]:
    expected_manifest_sha256 = _require_sha256(
        expected_manifest_sha256,
        field="expected cache-manifest SHA",
    )
    if not manifest_path.is_file() or _sha256_path(manifest_path) != expected_manifest_sha256:
        raise ObservationXReuseAuditError("cache manifest hash mismatch")
    manifest = _read_json_object(manifest_path)
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ObservationXReuseAuditError("cache manifest schema is not current")
    if manifest.get("status") != "completed":
        raise ObservationXReuseAuditError("cache manifest is not completed")
    if manifest.get("feature_dim") != FEATURE_DIM:
        raise ObservationXReuseAuditError("cache feature dimension is not 144")
    if manifest.get("feature_dtype") != FEATURE_DTYPE:
        raise ObservationXReuseAuditError("cache feature dtype is not float32")
    rows = _require_int(manifest.get("rows"), field="cache rows", minimum=1)

    outputs = manifest.get("output_files")
    if not isinstance(outputs, dict) or not isinstance(outputs.get(X_FILENAME), dict):
        raise ObservationXReuseAuditError("cache manifest lacks X output binding")
    x_receipt = outputs[X_FILENAME]
    x_path = (manifest_path.parent / X_FILENAME).resolve()
    x_sha256 = _require_sha256(x_receipt.get("sha256"), field="cache X SHA")
    x_bytes = _require_int(x_receipt.get("bytes"), field="cache X bytes", minimum=1)
    expected_bytes = rows * X_ROW_BYTES
    if x_bytes != expected_bytes:
        raise ObservationXReuseAuditError("cache X receipt is not exactly 576 bytes per row")
    if not x_path.is_file() or x_path.stat().st_size != expected_bytes:
        raise ObservationXReuseAuditError("cache X size is not exactly 576 bytes per row")
    if _sha256_path(x_path) != x_sha256:
        raise ObservationXReuseAuditError("cache X hash mismatch")

    canonical_root_value = manifest.get("canonical_root")
    if not isinstance(canonical_root_value, str) or not canonical_root_value:
        raise ObservationXReuseAuditError("cache canonical root is missing")
    canonical_root = Path(canonical_root_value).resolve()
    if not canonical_root.is_dir():
        raise ObservationXReuseAuditError("cache canonical root is missing")
    catalog_path_value = manifest.get("catalog_path")
    if not isinstance(catalog_path_value, str) or not catalog_path_value:
        raise ObservationXReuseAuditError("cache catalog path is missing")
    catalog_path = Path(catalog_path_value).resolve()
    catalog_sha256 = _require_sha256(manifest.get("catalog_sha256"), field="cache catalog SHA")
    if not catalog_path.is_file() or _sha256_path(catalog_path) != catalog_sha256:
        raise ObservationXReuseAuditError("cache catalog hash mismatch")
    catalog = _read_json_object(catalog_path)
    if not _paths_equal(catalog.get("out_root", ""), canonical_root):
        raise ObservationXReuseAuditError("catalog out_root differs from cache canonical root")
    catalog_units = catalog.get("included_units")
    if not isinstance(catalog_units, list) or not catalog_units:
        raise ObservationXReuseAuditError("catalog included-unit allowlist is missing")
    if len(catalog_units) != manifest.get("included_units"):
        raise ObservationXReuseAuditError("cache/catalog included-unit count mismatch")

    inputs = manifest.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ObservationXReuseAuditError("cache input inventory is missing")
    if _stable_sha256(inputs) != str(manifest.get("input_bindings_sha256", "")).lower():
        raise ObservationXReuseAuditError("cache input binding digest is not reproducible")
    if len(inputs) != manifest.get("atomic_parts"):
        raise ObservationXReuseAuditError("cache input inventory does not cover atomic parts")

    bindings: list[CacheBinding] = []
    unit_cache: dict[tuple[str, str], tuple[Path, str, dict[str, Any], dict[str, str]]] = {}
    seen_bindings: set[tuple[str, str, str]] = set()
    row_start = 0
    for index, receipt in enumerate(inputs):
        if not isinstance(receipt, dict):
            raise ObservationXReuseAuditError("cache input receipt is not an object")
        source_id = receipt.get("source_id")
        unit_id = receipt.get("unit_id")
        part_name = receipt.get("part")
        if not all(isinstance(value, str) and value for value in (source_id, unit_id, part_name)):
            raise ObservationXReuseAuditError("cache input identity is invalid")
        identity = (source_id, unit_id, part_name)
        if identity in seen_bindings:
            raise ObservationXReuseAuditError("cache input identities are duplicated")
        seen_bindings.add(identity)
        match = PART_PATTERN.fullmatch(part_name)
        if match is None:
            raise ObservationXReuseAuditError("cache part name is invalid")
        part_number = int(match.group(1))
        part_rows = _require_int(receipt.get("rows"), field="cache part rows", minimum=1)
        unit_rank = _require_int(receipt.get("unit_rank"), field="cache unit_rank")
        if unit_rank >= len(catalog_units):
            raise ObservationXReuseAuditError("cache unit_rank is outside catalog allowlist")
        catalog_unit = catalog_units[unit_rank]
        if not isinstance(catalog_unit, dict):
            raise ObservationXReuseAuditError("catalog unit entry is not an object")
        if (catalog_unit.get("source_id"), catalog_unit.get("unit_id")) != (source_id, unit_id):
            raise ObservationXReuseAuditError("cache input identity differs from catalog rank")
        unit_identity = (source_id, unit_id)
        expected_unit_sha = _require_sha256(
            catalog_unit.get("manifest_sha256"),
            field="catalog unit-manifest SHA",
        )
        if str(receipt.get("unit_manifest_sha256", "")).lower() != expected_unit_sha:
            raise ObservationXReuseAuditError("cache/catalog unit-manifest bindings differ")
        if unit_identity not in unit_cache:
            unit_manifest_path = _resolve_inside(
                canonical_root,
                catalog_unit.get("manifest_path"),
                field="catalog unit manifest path",
            )
            if not unit_manifest_path.is_file() or _sha256_path(unit_manifest_path) != expected_unit_sha:
                raise ObservationXReuseAuditError("unit manifest hash mismatch")
            unit_manifest = _read_json_object(unit_manifest_path)
            if unit_manifest.get("schema_version") != UNIT_MANIFEST_SCHEMA_VERSION:
                raise ObservationXReuseAuditError("unit manifest schema is not current")
            if unit_manifest.get("status") != "completed":
                raise ObservationXReuseAuditError("unit manifest is not completed")
            if unit_manifest.get("information_mode") != "actor_visible":
                raise ObservationXReuseAuditError("unit manifest is not actor-visible")
            if (unit_manifest.get("source_id"), unit_manifest.get("unit_id")) != unit_identity:
                raise ObservationXReuseAuditError("unit manifest identity mismatch")
            source_hashes = _validate_source_hashes(unit_manifest)
            unit_cache[unit_identity] = (
                unit_manifest_path,
                expected_unit_sha,
                unit_manifest,
                source_hashes,
            )
        unit_manifest_path, unit_manifest_sha, unit_manifest, source_hashes = unit_cache[unit_identity]
        matching_parts = [
            part for part in unit_manifest.get("parts", [])
            if isinstance(part, dict) and part.get("name") == part_name
        ]
        if len(matching_parts) != 1:
            raise ObservationXReuseAuditError("unit manifest does not uniquely bind cache part")
        part = matching_parts[0]
        if part.get("rows") != part_rows:
            raise ObservationXReuseAuditError("cache/unit part row counts differ")
        transition_name = f"{part_name}.transitions.jsonl.gz"
        transition_sha256 = _require_sha256(
            receipt.get("transitions_sha256"),
            field="cache transition-sidecar SHA",
        )
        files_sha256 = part.get("files_sha256")
        if not isinstance(files_sha256, dict):
            raise ObservationXReuseAuditError("unit part lacks file hashes")
        if str(files_sha256.get(transition_name, "")).lower() != transition_sha256:
            raise ObservationXReuseAuditError("cache/unit transition-sidecar bindings differ")
        transition_path = (unit_manifest_path.parent / transition_name).resolve()
        try:
            transition_path.relative_to(unit_manifest_path.parent.resolve())
        except ValueError as exc:
            raise ObservationXReuseAuditError("transition sidecar escapes unit directory") from exc
        bindings.append(CacheBinding(
            index=index,
            binding_id=f"{source_id}:{unit_id}:{part_name}",
            source_id=source_id,
            unit_id=unit_id,
            part=part_name,
            part_number=part_number,
            row_start=row_start,
            rows=part_rows,
            transition_path=transition_path,
            transition_sha256=transition_sha256,
            unit_manifest_path=unit_manifest_path,
            unit_manifest_sha256=unit_manifest_sha,
            source_sha256_by_path=source_hashes,
        ))
        row_start += part_rows
    if row_start != rows:
        raise ObservationXReuseAuditError("cache bindings do not cover every X row")

    return (
        manifest,
        catalog_path,
        catalog_sha256,
        x_path,
        x_sha256,
        rows,
        tuple(bindings),
    )


def _implementation_sha256() -> str:
    return _sha256_path(Path(__file__).resolve())


def _checkpoint_path(parts_dir: Path, binding: CacheBinding) -> Path:
    suffix = hashlib.sha256(binding.binding_id.encode("utf-8")).hexdigest()[:16]

    return parts_dir / f"part-{binding.index:06d}-{suffix}.json"


def _source_expected_sha(binding: CacheBinding, source_path: Path, transition: Mapping[str, Any]) -> str:
    expected = binding.source_sha256_by_path.get(_path_key(source_path))
    if expected is None:
        raise ObservationXReuseAuditError("source observation path is absent from unit manifest")
    if str(transition.get("source_hash", "")).lower() != expected:
        raise ObservationXReuseAuditError("transition source hash differs from unit manifest")

    return expected


def _strict_card_ids(hand: Any) -> tuple[int, ...]:
    if not isinstance(hand, list):
        raise ValueError("hand is not a list")
    card_ids: list[int] = []
    for entry in hand:
        if isinstance(entry, bool):
            raise ValueError("boolean card ID")
        if isinstance(entry, int):
            card_id = entry
        elif isinstance(entry, dict):
            card_id = entry.get("cardId", entry.get("id"))
        else:
            raise ValueError("invalid hand entry")
        if isinstance(card_id, bool) or not isinstance(card_id, int):
            raise ValueError("invalid card ID")
        card_ids.append(card_id)

    return tuple(card_ids)


def _observation_own_hand(observation: Mapping[str, Any]) -> list[Any]:
    current = observation.get("current")
    if not isinstance(current, dict):
        raise ValueError("observation lacks current")
    your_index = current.get("yourIndex")
    if isinstance(your_index, bool) or not isinstance(your_index, int):
        raise ValueError("observation actor index is invalid")
    players = current.get("players")
    if not isinstance(players, list) or not 0 <= your_index < len(players):
        raise ValueError("observation players do not contain actor")
    actor = players[your_index]
    if not isinstance(actor, dict):
        raise ValueError("observation actor is not an object")
    hand = actor.get("hand")
    _strict_card_ids(hand)

    return json.loads(_canonical_json_bytes(hand))


def _observation_hand_ids(observation: Mapping[str, Any]) -> tuple[int, ...]:
    hand = _observation_own_hand(observation)

    return _strict_card_ids(hand)


def _feature_bytes(
    candidate: Candidate,
    *,
    feature_encoder: FeatureEncoder,
) -> tuple[bytes | None, str | None]:
    try:
        encoded = feature_encoder(candidate.source_record, candidate.normalized_observation)
    except Exception:
        return None, "feature_reencode_failed"
    if not isinstance(encoded, np.ndarray):
        return None, "feature_reencode_not_ndarray"
    if encoded.shape != (FEATURE_DIM,):
        return None, "feature_reencode_shape_mismatch"
    if encoded.dtype != np.dtype(np.float32):
        return None, "feature_reencode_dtype_mismatch"
    if not encoded.flags.c_contiguous:
        return None, "feature_reencode_not_contiguous"
    if not np.isfinite(encoded).all():
        return None, "feature_reencode_nonfinite"
    payload = encoded.astype("<f4", copy=False).tobytes(order="C")
    if len(payload) != X_ROW_BYTES:
        return None, "feature_reencode_byte_length_mismatch"

    return payload, None


class _XRowReader:
    def __init__(self, path: Path, rows: int):
        self.path = path
        self.rows = rows
        self._handle = None
        self._cache: dict[int, tuple[bytes, str, bool]] = {}

    def __enter__(self) -> "_XRowReader":
        self._handle = self.path.open("rb")

        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def read(self, row: int) -> tuple[bytes, str, bool]:
        if row in self._cache:
            return self._cache[row]
        if isinstance(row, bool) or not isinstance(row, int) or not 0 <= row < self.rows:
            raise ObservationXReuseAuditError("resolved X row is outside cache bounds")
        if self._handle is None:
            raise ObservationXReuseAuditError("X reader is not open")
        self._handle.seek(row * X_ROW_BYTES)
        payload = self._handle.read(X_ROW_BYTES)
        if len(payload) != X_ROW_BYTES:
            raise ObservationXReuseAuditError("X row read was not exactly 576 bytes")
        vector = np.frombuffer(payload, dtype="<f4")
        finite = bool(vector.shape == (FEATURE_DIM,) and np.isfinite(vector).all())
        receipt = (payload, hashlib.sha256(payload).hexdigest(), finite)
        self._cache[row] = receipt

        return receipt


def _default_feature_encoder(source_record: dict[str, Any], observation: dict[str, Any]) -> np.ndarray:
    from scripts.build_value_q_canonical import _row_features

    return _row_features(source_record, observation)


@dataclass(frozen=True)
class CandidateStreamBinding:
    binding_id: str
    row_start: int
    rows: int


@dataclass(frozen=True)
class CandidateStream:
    manifest: dict[str, Any]
    manifest_path: Path
    manifest_sha256: str
    candidate_path: Path
    candidate_sha256: str
    candidate_references: int
    admission_receipt_sha256: str
    semantic_audit_receipt_sha256: str
    bindings: tuple[CandidateStreamBinding, ...]


@dataclass(frozen=True)
class ProductionAdmission:
    admission_manifest_path: Path
    admission_manifest_sha256: str
    admission_manifest: dict[str, Any]
    admission_database_path: Path
    admission_database_sha256: str
    admission_identity: str
    admission_receipt_sha256: str
    semantic_audit_receipt_path: Path
    semantic_audit_receipt_file_sha256: str
    semantic_audit_receipt_sha256: str
    semantic_audit_receipt: dict[str, Any]
    binding_registry_path: Path
    binding_registry_file_sha256: str
    binding_registry_sha256: str
    implementation_hash_manifest_path: Path
    implementation_hash_manifest_file_sha256: str
    implementation_hash_manifest_sha256: str
    causal_aligner_implementation_sha256: str
    legality_validator_implementation_sha256: str
    action_schema_implementation_sha256: str
    candidate_references: int
    bindings: tuple[CandidateStreamBinding, ...]
    loader_implementation_sha256: str


def _load_candidate_stream(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_candidate_references: int | None,
) -> CandidateStream:
    expected_sha = _require_sha256(
        expected_manifest_sha256,
        field="expected candidate-manifest SHA",
    )
    if not manifest_path.is_file() or _sha256_path(manifest_path) != expected_sha:
        raise ObservationXReuseAuditError("candidate manifest hash mismatch")
    manifest = _read_json_object(manifest_path)
    if manifest.get("schema_version") != CANDIDATE_MANIFEST_SCHEMA_VERSION:
        raise ObservationXReuseAuditError("candidate manifest schema is not current")
    if manifest.get("status") != "completed":
        raise ObservationXReuseAuditError("candidate manifest is not completed")
    admission_receipt_sha256 = _require_sha256(
        manifest.get("admission_receipt_sha256"),
        field="candidate admission-receipt SHA",
    )
    semantic_audit_receipt_sha256 = _require_sha256(
        manifest.get("semantic_audit_receipt_sha256"),
        field="candidate semantic-audit receipt SHA",
    )
    candidate_references = _require_int(
        manifest.get("admission_candidate_count"),
        field="admission candidate count",
        minimum=1,
    )
    if (
        expected_candidate_references is not None
        and candidate_references != expected_candidate_references
    ):
        raise ObservationXReuseAuditError("candidate count differs from frozen authorization")
    file_binding = manifest.get("candidate_file")
    if not isinstance(file_binding, dict):
        raise ObservationXReuseAuditError("candidate manifest lacks candidate_file binding")
    path_value = file_binding.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ObservationXReuseAuditError("candidate file path is missing")
    candidate_path = _resolve_relative(path_value, owner=manifest_path)
    candidate_sha256 = _require_sha256(
        file_binding.get("sha256"),
        field="candidate file SHA",
    )
    file_rows = _require_int(file_binding.get("rows"), field="candidate file rows", minimum=1)
    if file_rows != candidate_references:
        raise ObservationXReuseAuditError("candidate file does not cover the Admission population")
    if not candidate_path.is_file() or _sha256_path(candidate_path) != candidate_sha256:
        raise ObservationXReuseAuditError("candidate file hash mismatch")

    binding_rows = manifest.get("admission_bindings")
    if not isinstance(binding_rows, list) or not binding_rows:
        raise ObservationXReuseAuditError("candidate manifest lacks Admission binding coverage")
    bindings: list[CandidateStreamBinding] = []
    seen: set[str] = set()
    row_start = 0
    normalized_binding_rows: list[dict[str, Any]] = []
    for item in binding_rows:
        if not isinstance(item, dict):
            raise ObservationXReuseAuditError("Admission binding entry is not an object")
        binding_id = item.get("binding_id")
        if not isinstance(binding_id, str) or not binding_id or binding_id in seen:
            raise ObservationXReuseAuditError("Admission binding IDs are missing or duplicated")
        seen.add(binding_id)
        declared_start = _require_int(
            item.get("global_candidate_row_start"),
            field="Admission binding row start",
        )
        rows = _require_int(item.get("rows"), field="Admission binding rows", minimum=1)
        if declared_start != row_start:
            raise ObservationXReuseAuditError("Admission binding ranges are not contiguous")
        bindings.append(CandidateStreamBinding(binding_id, row_start, rows))
        normalized_binding_rows.append({
            "binding_id": binding_id,
            "global_candidate_row_start": row_start,
            "rows": rows,
        })
        row_start += rows
    if row_start != candidate_references:
        raise ObservationXReuseAuditError("Admission bindings do not conserve candidate references")
    binding_registry_sha256 = _require_sha256(
        manifest.get("admission_binding_registry_sha256"),
        field="Admission binding-registry SHA",
    )
    if _stable_sha256(normalized_binding_rows) != binding_registry_sha256:
        raise ObservationXReuseAuditError("Admission binding-registry digest is not reproducible")

    return CandidateStream(
        manifest=manifest,
        manifest_path=manifest_path.resolve(),
        manifest_sha256=expected_sha,
        candidate_path=candidate_path,
        candidate_sha256=candidate_sha256,
        candidate_references=candidate_references,
        admission_receipt_sha256=admission_receipt_sha256,
        semantic_audit_receipt_sha256=semantic_audit_receipt_sha256,
        bindings=tuple(bindings),
    )


def admission_global_candidate_row(
    binding_row_start: int,
    admitted_ordinal: int,
) -> int:
    """Return cumulative binding start plus ordinal in ``ORDER BY local_row``."""
    if binding_row_start < 0 or admitted_ordinal < 0:
        raise ObservationXReuseAuditError("Admission candidate coordinates must be non-negative")

    return binding_row_start + admitted_ordinal


def admission_transition_reference(
    transition: Mapping[str, Any],
) -> tuple[str, int, str]:
    """Return the exact Admission observation primary key used by the auditor."""
    state_before = transition.get("state_before")
    reference = state_before.get("source_observation") if isinstance(state_before, Mapping) else None
    if not isinstance(reference, Mapping):
        raise ObservationXReuseAuditError("Admission transition observation reference is absent")

    return (
        _require_sha256(transition.get("source_hash"), field="Admission source SHA"),
        _require_int(reference.get("source_line"), field="Admission source line", minimum=1),
        _require_sha256(
            reference.get("content_sha256"),
            field="Admission observation content SHA",
        ),
    )


def _readonly_admission_connection(database_path: Path) -> sqlite3.Connection:
    uri = f"file:{database_path.resolve().as_posix()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True, timeout=60.0)
    connection.execute("PRAGMA query_only=ON")

    return connection


def _candidate_bindings_from_semantic_registry(
    binding_entries: Any,
    *,
    expected_candidates: int,
) -> tuple[CandidateStreamBinding, ...]:
    if not isinstance(binding_entries, list) or not binding_entries:
        raise ObservationXReuseAuditError("semantic binding registry is empty")
    bindings: list[CandidateStreamBinding] = []
    seen_binding_ids: set[str] = set()
    row_start = 0
    for entry in binding_entries:
        if not isinstance(entry, dict):
            raise ObservationXReuseAuditError("semantic binding entry is not an object")
        binding_id = entry.get("binding_id")
        if not isinstance(binding_id, str) or not binding_id:
            raise ObservationXReuseAuditError("semantic binding ID is missing")
        if binding_id in seen_binding_ids:
            raise ObservationXReuseAuditError("semantic binding IDs are duplicated")
        seen_binding_ids.add(binding_id)
        declared_start = _require_int(
            entry.get("global_candidate_row_start"),
            field="semantic binding row start",
        )
        rows = _require_int(
            entry.get("expected_rows"),
            field="semantic binding rows",
        )
        if declared_start != row_start:
            raise ObservationXReuseAuditError("semantic binding ranges are not contiguous")
        bindings.append(CandidateStreamBinding(binding_id, row_start, rows))
        row_start += rows
    if row_start != expected_candidates:
        raise ObservationXReuseAuditError("semantic bindings do not conserve Admission candidates")

    return tuple(bindings)


def _load_production_admission(
    *,
    admission_manifest_path: Path,
    semantic_audit_receipt_path: Path,
    semantic_audit_receipt_file_sha256: str,
    binding_registry_path: Path,
    expected_candidate_references: int | None,
) -> ProductionAdmission:
    from scripts import audit_canonical_decision_contract_v1 as semantic_auditor

    expected_semantic_file_sha256 = _require_sha256(
        semantic_audit_receipt_file_sha256,
        field="expected semantic-audit receipt file SHA",
    )
    if (
        not semantic_audit_receipt_path.is_file()
        or _sha256_path(semantic_audit_receipt_path) != expected_semantic_file_sha256
    ):
        raise ObservationXReuseAuditError("semantic-audit receipt file hash mismatch")
    semantic_receipt = _read_json_object(semantic_audit_receipt_path)
    if semantic_receipt.get("schema_version") != semantic_auditor.AUDIT_RECEIPT_SCHEMA_VERSION:
        raise ObservationXReuseAuditError("semantic-audit receipt schema is not current")
    if semantic_receipt.get("status") != "COMPLETE_READ_ONLY_EXACT":
        raise ObservationXReuseAuditError("semantic audit is not complete read-only exact")
    semantic_receipt_sha256 = _require_sha256(
        semantic_receipt.get("receipt_sha256"),
        field="semantic-audit receipt SHA",
    )
    semantic_body = dict(semantic_receipt)
    semantic_body.pop("receipt_sha256", None)
    if _stable_sha256(semantic_body) != semantic_receipt_sha256:
        raise ObservationXReuseAuditError("semantic-audit receipt self-hash mismatch")
    expected_candidates = _require_int(
        semantic_receipt.get("expected_candidates"),
        field="semantic-audit expected candidates",
        minimum=1,
    )
    inspected_candidates = _require_int(
        semantic_receipt.get("inspected_candidates"),
        field="semantic-audit inspected candidates",
        minimum=1,
    )
    if inspected_candidates != expected_candidates:
        raise ObservationXReuseAuditError("semantic audit did not inspect the frozen population")
    if (
        expected_candidate_references is not None
        and expected_candidates != expected_candidate_references
    ):
        raise ObservationXReuseAuditError("semantic audit differs from frozen candidate authorization")
    required_true = (
        "sha_recomputed",
        "frozen_inputs_ok",
        "counters_conserved",
        "raw_sources_verified",
        "admission_v0_immutable",
    )
    if any(semantic_receipt.get(field) is not True for field in required_true):
        raise ObservationXReuseAuditError("semantic-audit completion proofs are incomplete")
    if semantic_receipt.get("forbidden_work_performed") != []:
        raise ObservationXReuseAuditError("semantic-audit receipt records forbidden work")
    if (
        semantic_receipt.get("global_candidate_row_convention")
        != semantic_auditor.GLOBAL_CANDIDATE_ROW_CONVENTION
    ):
        raise ObservationXReuseAuditError(
            "semantic-audit candidate coordinate convention is not current"
        )

    auditor_path = Path(semantic_auditor.__file__).resolve()
    if _sha256_path(auditor_path) != semantic_receipt.get("auditor_implementation_sha256"):
        raise ObservationXReuseAuditError("semantic auditor implementation drifted after receipt")
    semantic_dependencies = semantic_auditor._semantic_dependency_manifest(ROOT)
    if (
        semantic_dependencies["semantic_dependency_sha256"]
        != semantic_receipt.get("semantic_dependency_sha256")
    ):
        raise ObservationXReuseAuditError("semantic contract dependencies drifted after receipt")
    contract_files = sorted((ROOT / "deepmew" / "replay" / "contract").glob("*.py"))
    current_contract_sha256 = semantic_auditor.implementation_sha256(*contract_files)
    if current_contract_sha256 != semantic_receipt.get("contract_implementation_sha256"):
        raise ObservationXReuseAuditError("canonical contract implementation drifted after receipt")
    implementation_hash_manifest_path = (
        semantic_audit_receipt_path.parent / "FULL_AUDIT_IMPLEMENTATION_HASHES.json"
    )
    implementation_hash_manifest_file_sha256 = _require_sha256(
        semantic_receipt.get("implementation_hash_manifest_file_sha256"),
        field="semantic implementation-hash manifest file SHA",
    )
    if (
        not implementation_hash_manifest_path.is_file()
        or _sha256_path(implementation_hash_manifest_path)
        != implementation_hash_manifest_file_sha256
    ):
        raise ObservationXReuseAuditError(
            "semantic implementation-hash manifest file mismatch"
        )
    implementation_hash_manifest = _read_json_object(
        implementation_hash_manifest_path
    )
    if (
        implementation_hash_manifest.get("schema_version")
        != semantic_auditor.IMPLEMENTATION_HASH_MANIFEST_SCHEMA_VERSION
    ):
        raise ObservationXReuseAuditError(
            "semantic implementation-hash manifest schema is not current"
        )
    implementation_hash_manifest_sha256 = _require_sha256(
        implementation_hash_manifest.get("manifest_sha256"),
        field="semantic implementation-hash manifest SHA",
    )
    implementation_hash_body = dict(implementation_hash_manifest)
    implementation_hash_body.pop("manifest_sha256", None)
    if _stable_sha256(implementation_hash_body) != implementation_hash_manifest_sha256:
        raise ObservationXReuseAuditError(
            "semantic implementation-hash manifest self-hash mismatch"
        )
    if (
        implementation_hash_manifest_sha256
        != semantic_receipt.get("implementation_hash_manifest_sha256")
    ):
        raise ObservationXReuseAuditError(
            "semantic receipt and implementation-hash manifest differ"
        )
    current_implementation_hash_manifest = semantic_auditor._implementation_hash_manifest(
        ROOT
    )
    if implementation_hash_manifest != current_implementation_hash_manifest:
        raise ObservationXReuseAuditError(
            "canonical implementation identities drifted after semantic receipt"
        )
    implementation_hashes = semantic_auditor._typed_implementation_hashes(
        current_implementation_hash_manifest
    )
    for field in (
        "causal_aligner_implementation_sha256",
        "legality_validator_implementation_sha256",
        "action_schema_implementation_sha256",
    ):
        if getattr(implementation_hashes, field) != semantic_receipt.get(field):
            raise ObservationXReuseAuditError(
                f"semantic receipt {field} differs from current canonical implementation"
            )

    admission_manifest_sha256 = _sha256_path(admission_manifest_path)
    if admission_manifest_sha256 != semantic_receipt.get("admission_manifest_sha256"):
        raise ObservationXReuseAuditError("Admission manifest differs from semantic receipt")
    admission_manifest = _read_json_object(admission_manifest_path)
    if (
        admission_manifest.get("schema_version") != "deepmew.e1_replay_admission.v2"
        or admission_manifest.get("status") != "complete"
    ):
        raise ObservationXReuseAuditError("Admission receipt is not complete v2")
    admission_receipt_sha256 = _require_sha256(
        admission_manifest.get("receipt_sha256"),
        field="Admission receipt SHA",
    )
    admission_identity = _require_sha256(
        admission_manifest.get("admission_identity_sha256"),
        field="Admission identity SHA",
    )
    admission_body = dict(admission_manifest)
    admission_body.pop("receipt_sha256", None)
    if _stable_sha256(admission_body) != admission_receipt_sha256:
        raise ObservationXReuseAuditError("Admission receipt self-hash mismatch")
    if admission_receipt_sha256 != semantic_receipt.get("admission_receipt_sha256"):
        raise ObservationXReuseAuditError("Admission receipt differs from semantic audit")
    if admission_identity != semantic_receipt.get("admission_identity"):
        raise ObservationXReuseAuditError("Admission identity differs from semantic audit")
    for field in ("request_count", "observation_count", "transition_count"):
        if admission_manifest.get(field) != expected_candidates:
            raise ObservationXReuseAuditError("Admission tables do not cover semantic candidates")
    database_path_value = admission_manifest.get("database_path")
    if not isinstance(database_path_value, str) or not database_path_value:
        raise ObservationXReuseAuditError("Admission database path is missing")
    database_path = _resolve_relative(database_path_value, owner=admission_manifest_path)
    expected_database_sha256 = _require_sha256(
        admission_manifest.get("database_sha256"),
        field="Admission database SHA",
    )
    expected_database_bytes = _require_int(
        admission_manifest.get("database_bytes"),
        field="Admission database bytes",
        minimum=1,
    )
    if not database_path.is_file() or database_path.stat().st_size != expected_database_bytes:
        raise ObservationXReuseAuditError("Admission database size mismatch")
    wal_path = database_path.with_name(database_path.name + "-wal")
    if wal_path.exists() and wal_path.stat().st_size != 0:
        raise ObservationXReuseAuditError("Admission WAL is non-empty; immutable read refused")
    actual_database_sha256 = _sha256_path(database_path)
    semantic_database_hashes = {
        semantic_receipt.get("admission_database_expected_sha256"),
        semantic_receipt.get("admission_database_actual_before_sha256"),
        semantic_receipt.get("admission_database_sha256"),
    }
    if semantic_database_hashes != {expected_database_sha256}:
        raise ObservationXReuseAuditError("semantic receipt has inconsistent Admission hashes")
    if actual_database_sha256 != expected_database_sha256:
        raise ObservationXReuseAuditError("Admission database hash mismatch")

    expected_binding_file_sha256 = _require_sha256(
        semantic_receipt.get("binding_registry_file_sha256"),
        field="semantic binding-registry file SHA",
    )
    if (
        not binding_registry_path.is_file()
        or _sha256_path(binding_registry_path) != expected_binding_file_sha256
    ):
        raise ObservationXReuseAuditError("semantic binding-registry file hash mismatch")
    registry = _read_json_object(binding_registry_path)
    registry_sha256 = _require_sha256(
        registry.get("binding_registry_sha256"),
        field="semantic binding-registry SHA",
    )
    registry_body = dict(registry)
    registry_body.pop("binding_registry_sha256", None)
    if _stable_sha256(registry_body) != registry_sha256:
        raise ObservationXReuseAuditError("semantic binding-registry self-hash mismatch")
    if registry_sha256 != semantic_receipt.get("binding_registry_sha256"):
        raise ObservationXReuseAuditError("semantic receipt and binding registry differ")
    if (
        registry.get("schema_version") != semantic_auditor.BINDING_REGISTRY_SCHEMA_VERSION
        or registry.get("global_candidate_row_convention")
        != semantic_auditor.GLOBAL_CANDIDATE_ROW_CONVENTION
        or registry.get("actual_transition_count") != expected_candidates
    ):
        raise ObservationXReuseAuditError("semantic binding-registry contract mismatch")
    bindings = _candidate_bindings_from_semantic_registry(
        registry.get("bindings"),
        expected_candidates=expected_candidates,
    )

    admission_connection = _readonly_admission_connection(database_path)
    try:
        completed_inputs = tuple(
            (str(identity), int(rows))
            for identity, rows in admission_connection.execute(
                "SELECT identity,rows FROM completed_inputs "
                "WHERE kind='transition' ORDER BY identity"
            )
        )
        actual_transition_count = int(admission_connection.execute(
            "SELECT COUNT(*) FROM transitions"
        ).fetchone()[0])
    finally:
        admission_connection.close()
    if completed_inputs != tuple((binding.binding_id, binding.rows) for binding in bindings):
        raise ObservationXReuseAuditError("Admission completed inputs differ from semantic bindings")
    if actual_transition_count != expected_candidates:
        raise ObservationXReuseAuditError("Admission transition count differs from semantic receipt")

    return ProductionAdmission(
        admission_manifest_path=admission_manifest_path.resolve(),
        admission_manifest_sha256=admission_manifest_sha256,
        admission_manifest=admission_manifest,
        admission_database_path=database_path,
        admission_database_sha256=actual_database_sha256,
        admission_identity=admission_identity,
        admission_receipt_sha256=admission_receipt_sha256,
        semantic_audit_receipt_path=semantic_audit_receipt_path.resolve(),
        semantic_audit_receipt_file_sha256=expected_semantic_file_sha256,
        semantic_audit_receipt_sha256=semantic_receipt_sha256,
        semantic_audit_receipt=semantic_receipt,
        binding_registry_path=binding_registry_path.resolve(),
        binding_registry_file_sha256=expected_binding_file_sha256,
        binding_registry_sha256=registry_sha256,
        implementation_hash_manifest_path=implementation_hash_manifest_path.resolve(),
        implementation_hash_manifest_file_sha256=(
            implementation_hash_manifest_file_sha256
        ),
        implementation_hash_manifest_sha256=implementation_hash_manifest_sha256,
        causal_aligner_implementation_sha256=(
            implementation_hashes.causal_aligner_implementation_sha256
        ),
        legality_validator_implementation_sha256=(
            implementation_hashes.legality_validator_implementation_sha256
        ),
        action_schema_implementation_sha256=(
            implementation_hashes.action_schema_implementation_sha256
        ),
        candidate_references=expected_candidates,
        bindings=bindings,
        loader_implementation_sha256=_implementation_sha256(),
    )


def _work_connection(path: Path, *, run_id: str) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS assets (
            asset_id TEXT PRIMARY KEY,
            asset_payload_sha256 TEXT NOT NULL,
            source_game_id TEXT NOT NULL,
            source_event_position INTEGER NOT NULL,
            acting_seat INTEGER NOT NULL,
            public_visibility_schema TEXT NOT NULL,
            observation_feature_version TEXT NOT NULL,
            normalized_observation_sha256 TEXT NOT NULL,
            normalized_observation_blob BLOB NOT NULL,
            source_record_blob BLOB NOT NULL,
            precheck_reasons_json TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS asset_aliases (
            asset_id TEXT NOT NULL,
            alias_sha256 TEXT NOT NULL,
            PRIMARY KEY (asset_id, alias_sha256),
            FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS asset_alias_sha_idx ON asset_aliases(alias_sha256);
        CREATE TABLE IF NOT EXISTS candidate_references (
            global_candidate_row INTEGER PRIMARY KEY,
            candidate_reference_id TEXT NOT NULL UNIQUE,
            admission_binding_id TEXT NOT NULL,
            admission_local_row INTEGER NOT NULL,
            asset_id TEXT,
            reference_failure_code TEXT,
            reference_sha256 TEXT NOT NULL,
            FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
        );
        CREATE INDEX IF NOT EXISTS candidate_reference_asset_idx
            ON candidate_references(asset_id);
        CREATE UNIQUE INDEX IF NOT EXISTS candidate_reference_admission_coordinate_idx
            ON candidate_references(admission_binding_id,admission_local_row);
        CREATE TABLE IF NOT EXISTS cache_aliases (
            global_row INTEGER PRIMARY KEY,
            binding_id TEXT NOT NULL,
            decision_id TEXT NOT NULL,
            local_row INTEGER NOT NULL,
            row_start INTEGER NOT NULL,
            source_alias_sha256 TEXT NOT NULL,
            source_path TEXT NOT NULL,
            source_line INTEGER NOT NULL,
            source_field TEXT NOT NULL,
            source_schema TEXT,
            source_file_sha256 TEXT NOT NULL,
            declared_state_before_feature_row INTEGER,
            normalized_observation_sha256 TEXT
        );
        CREATE INDEX IF NOT EXISTS cache_alias_sha_idx
            ON cache_aliases(source_alias_sha256);
        CREATE INDEX IF NOT EXISTS cache_alias_source_idx
            ON cache_aliases(source_path, source_line, global_row);
        CREATE TABLE IF NOT EXISTS binding_scans (
            binding_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            transition_sha256 TEXT NOT NULL,
            rows_scanned INTEGER NOT NULL,
            matches INTEGER NOT NULL,
            match_ledger_sha256 TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS source_scans (
            source_path TEXT PRIMARY KEY,
            source_sha256 TEXT NOT NULL,
            matches INTEGER NOT NULL,
            normalization_ledger_sha256 TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS classifications (
            asset_id TEXT PRIMARY KEY,
            classification TEXT NOT NULL,
            resolved_global_row INTEGER,
            tensor_sha256 TEXT,
            candidate_reference_count INTEGER NOT NULL,
            cache_alias_count INTEGER NOT NULL,
            reason_codes_json TEXT NOT NULL,
            payload_blob BLOB NOT NULL,
            FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
        ) WITHOUT ROWID;
        """
    )
    existing = connection.execute(
        "SELECT value FROM meta WHERE key='run_id'"
    ).fetchone()
    if existing is None:
        connection.execute("INSERT INTO meta VALUES ('run_id',?)", (run_id,))
        connection.commit()
    elif existing[0] != run_id:
        connection.close()
        raise ObservationXReuseAuditError(
            "work database belongs to another immutable registry run"
        )

    return connection


def _binding_for_candidate_row(
    bindings: Sequence[CandidateStreamBinding],
    global_candidate_row: int,
    start_index: int,
) -> tuple[CandidateStreamBinding, int]:
    index = start_index
    while index < len(bindings):
        binding = bindings[index]
        if global_candidate_row < binding.row_start + binding.rows:
            return binding, index
        index += 1
    raise ObservationXReuseAuditError("candidate row is outside Admission binding coverage")


def _validated_candidate_reference(
    row: Mapping[str, Any],
    *,
    ordinal: int,
    expected_binding: CandidateStreamBinding,
    normalizer: ObservationNormalizer,
) -> tuple[Candidate, dict[str, Any], str, tuple[str, ...]]:
    if row.get("schema_version") != CANDIDATE_ROW_SCHEMA_VERSION:
        raise ObservationXReuseAuditError(f"candidate line {ordinal + 1} schema is not current")
    global_candidate_row = _require_int(
        row.get("global_candidate_row"),
        field="candidate global row",
    )
    if global_candidate_row != ordinal:
        raise ObservationXReuseAuditError("candidate rows are not exact contiguous Admission coordinates")
    reference_id = row.get("candidate_reference_id")
    if not isinstance(reference_id, str) or not reference_id:
        raise ObservationXReuseAuditError("candidate reference ID is missing")
    admission_binding_id = row.get("admission_binding_id")
    admission_local_row = _require_int(
        row.get("admission_local_row"),
        field="candidate Admission local row",
    )
    if admission_binding_id != expected_binding.binding_id:
        raise ObservationXReuseAuditError("candidate reference differs from Admission binding coverage")

    asset_id = row.get("observation_asset_id")
    if not isinstance(asset_id, str) or not asset_id:
        raise ObservationXReuseAuditError("candidate observation asset ID is missing")
    source_game_id = row.get("source_game_id")
    if not isinstance(source_game_id, str) or not source_game_id:
        raise ObservationXReuseAuditError(f"candidate {asset_id} source_game_id is missing")
    source_event_position = _require_int(
        row.get("source_event_position"),
        field=f"candidate {asset_id} source_event_position",
    )
    acting_seat = _require_int(
        row.get("acting_seat"),
        field=f"candidate {asset_id} acting_seat",
    )
    public_visibility_schema = row.get("public_visibility_schema")
    if not isinstance(public_visibility_schema, str) or not public_visibility_schema:
        raise ObservationXReuseAuditError(
            f"candidate {asset_id} public_visibility_schema is missing"
        )
    observation_feature_version = row.get("observation_feature_version")
    if not isinstance(observation_feature_version, str) or not observation_feature_version:
        raise ObservationXReuseAuditError(
            f"candidate {asset_id} observation_feature_version is missing"
        )
    observation = row.get("normalized_observation")
    if not isinstance(observation, dict):
        raise ObservationXReuseAuditError(
            f"candidate {asset_id} normalized observation is not an object"
        )
    normalized_sha256 = _require_sha256(
        row.get("normalized_observation_sha256"),
        field=f"candidate {asset_id} normalized-observation SHA",
    )
    if _stable_sha256(observation) != normalized_sha256:
        raise ObservationXReuseAuditError(
            f"candidate {asset_id} normalized-observation hash mismatch"
        )
    try:
        projected = normalizer(observation)
    except (TypeError, ValueError) as exc:
        raise ObservationXReuseAuditError(
            f"candidate {asset_id} cannot be normalized"
        ) from exc
    if _canonical_json_bytes(projected) != _canonical_json_bytes(observation):
        raise ObservationXReuseAuditError(f"candidate {asset_id} is not already normalized")
    aliases = row.get("source_observation_sha256_aliases")
    if not isinstance(aliases, list) or not aliases:
        raise ObservationXReuseAuditError(
            f"candidate {asset_id} has no source-observation aliases"
        )
    normalized_aliases = tuple(sorted({
        _require_sha256(alias, field=f"candidate {asset_id} source alias")
        for alias in aliases
    }))
    source_record = row.get("source_record")
    if not isinstance(source_record, dict):
        raise ObservationXReuseAuditError(f"candidate {asset_id} source_record is not an object")
    if not isinstance(source_record.get("own_hand"), list):
        raise ObservationXReuseAuditError(
            f"candidate {asset_id} source_record.own_hand is required"
        )
    precheck_reasons: set[str] = set()
    try:
        if _strict_card_ids(source_record["own_hand"]) != _observation_hand_ids(observation):
            precheck_reasons.add("own_hand_mismatch")
    except ValueError:
        precheck_reasons.add("own_hand_unprovable")
    candidate = Candidate(
        asset_id=asset_id,
        source_game_id=source_game_id,
        source_event_position=source_event_position,
        acting_seat=acting_seat,
        public_visibility_schema=public_visibility_schema,
        observation_feature_version=observation_feature_version,
        normalized_observation_sha256=normalized_sha256,
        normalized_observation=observation,
        source_observation_sha256_aliases=normalized_aliases,
        source_record=source_record,
    )
    asset_payload = {
        "observation_asset_id": asset_id,
        "source_game_id": source_game_id,
        "source_event_position": source_event_position,
        "acting_seat": acting_seat,
        "public_visibility_schema": public_visibility_schema,
        "observation_feature_version": observation_feature_version,
        "normalized_observation_sha256": normalized_sha256,
        "normalized_observation": observation,
        "source_observation_sha256_aliases": list(normalized_aliases),
        "source_record": source_record,
    }
    reference_payload = {
        "global_candidate_row": ordinal,
        "candidate_reference_id": reference_id,
        "admission_binding_id": admission_binding_id,
        "admission_local_row": admission_local_row,
        "observation_asset_id": asset_id,
    }

    return candidate, reference_payload, _stable_sha256(asset_payload), tuple(sorted(precheck_reasons))


def _insert_verified_candidate_reference(
    connection: sqlite3.Connection,
    *,
    candidate: Candidate,
    reference: Mapping[str, Any],
    asset_payload_sha256: str,
    precheck_reasons: Sequence[str],
) -> None:
    observation_blob = sqlite3.Binary(zlib.compress(
        _canonical_json_bytes(candidate.normalized_observation),
        level=3,
    ))
    source_record_blob = sqlite3.Binary(zlib.compress(
        _canonical_json_bytes(candidate.source_record),
        level=3,
    ))
    connection.execute(
        "INSERT OR IGNORE INTO assets VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            candidate.asset_id,
            asset_payload_sha256,
            candidate.source_game_id,
            candidate.source_event_position,
            candidate.acting_seat,
            candidate.public_visibility_schema,
            candidate.observation_feature_version,
            candidate.normalized_observation_sha256,
            observation_blob,
            source_record_blob,
            json.dumps(tuple(sorted(precheck_reasons)), separators=(",", ":")),
        ),
    )
    actual_asset_sha = connection.execute(
        "SELECT asset_payload_sha256 FROM assets WHERE asset_id=?",
        (candidate.asset_id,),
    ).fetchone()[0]
    if actual_asset_sha != asset_payload_sha256:
        raise ObservationXReuseAuditError(
            "one canonical observation asset ID has conflicting immutable payloads"
        )
    connection.executemany(
        "INSERT OR IGNORE INTO asset_aliases VALUES (?,?)",
        (
            (candidate.asset_id, alias)
            for alias in candidate.source_observation_sha256_aliases
        ),
    )
    reference_payload = {
        **dict(reference),
        "reference_failure_code": None,
    }
    try:
        connection.execute(
            "INSERT INTO candidate_references "
            "(global_candidate_row,candidate_reference_id,admission_binding_id,"
            "admission_local_row,asset_id,reference_failure_code,reference_sha256) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                reference["global_candidate_row"],
                reference["candidate_reference_id"],
                reference["admission_binding_id"],
                reference["admission_local_row"],
                candidate.asset_id,
                None,
                _stable_sha256(reference_payload),
            ),
        )
    except sqlite3.IntegrityError as exc:
        raise ObservationXReuseAuditError(
            "candidate references contain a duplicate coordinate or identity"
        ) from exc


def _insert_failed_candidate_reference(
    connection: sqlite3.Connection,
    *,
    global_candidate_row: int,
    candidate_reference_id: str,
    admission_binding_id: str,
    admission_local_row: int,
    failure_code: str,
) -> None:
    if not failure_code:
        raise ObservationXReuseAuditError("failed candidate reference lacks a reason code")
    reference_payload = {
        "global_candidate_row": global_candidate_row,
        "candidate_reference_id": candidate_reference_id,
        "admission_binding_id": admission_binding_id,
        "admission_local_row": admission_local_row,
        "observation_asset_id": None,
        "reference_failure_code": failure_code,
    }
    try:
        connection.execute(
            "INSERT INTO candidate_references "
            "(global_candidate_row,candidate_reference_id,admission_binding_id,"
            "admission_local_row,asset_id,reference_failure_code,reference_sha256) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                global_candidate_row,
                candidate_reference_id,
                admission_binding_id,
                admission_local_row,
                None,
                failure_code,
                _stable_sha256(reference_payload),
            ),
        )
    except sqlite3.IntegrityError as exc:
        raise ObservationXReuseAuditError(
            "candidate references contain a duplicate coordinate or identity"
        ) from exc


def _ingest_candidate_references(
    connection: sqlite3.Connection,
    stream: CandidateStream,
    *,
    normalizer: ObservationNormalizer,
    max_new_references: int | None,
) -> tuple[bool, int, int]:
    existing = int(connection.execute("SELECT COUNT(*) FROM candidate_references").fetchone()[0])
    if existing:
        minimum, maximum = connection.execute(
            "SELECT MIN(global_candidate_row),MAX(global_candidate_row) FROM candidate_references"
        ).fetchone()
        if minimum != 0 or maximum != existing - 1:
            raise ObservationXReuseAuditError("resumed candidate ledger is not contiguous")
    inserted = 0
    binding_index = 0
    seen_lines = 0
    last_local_row_by_binding = {
        str(binding_id): int(local_row)
        for binding_id, local_row in connection.execute(
            "SELECT admission_binding_id,MAX(admission_local_row) "
            "FROM candidate_references GROUP BY admission_binding_id"
        )
    }
    try:
        with _open_jsonl(stream.candidate_path) as handle:
            for ordinal, line in enumerate(handle):
                seen_lines = ordinal + 1
                if ordinal < existing:
                    continue
                if max_new_references is not None and inserted >= max_new_references:
                    connection.commit()
                    return False, existing + inserted, inserted
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ObservationXReuseAuditError(
                        f"invalid candidate JSON at line {ordinal + 1}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ObservationXReuseAuditError(
                        f"candidate line {ordinal + 1} is not an object"
                    )
                binding, binding_index = _binding_for_candidate_row(
                    stream.bindings,
                    ordinal,
                    binding_index,
                )
                candidate, reference, asset_payload_sha256, precheck_reasons = (
                    _validated_candidate_reference(
                        row,
                        ordinal=ordinal,
                        expected_binding=binding,
                        normalizer=normalizer,
                    )
                )
                prior_local_row = last_local_row_by_binding.get(binding.binding_id)
                if (
                    prior_local_row is not None
                    and reference["admission_local_row"] <= prior_local_row
                ):
                    raise ObservationXReuseAuditError(
                        "candidate Admission local rows are not strictly increasing"
                    )
                _insert_verified_candidate_reference(
                    connection,
                    candidate=candidate,
                    reference=reference,
                    asset_payload_sha256=asset_payload_sha256,
                    precheck_reasons=precheck_reasons,
                )
                last_local_row_by_binding[binding.binding_id] = reference[
                    "admission_local_row"
                ]
                inserted += 1
                if inserted % 2048 == 0:
                    connection.commit()
    except OSError as exc:
        raise ObservationXReuseAuditError(
            f"cannot stream candidate file: {stream.candidate_path}"
        ) from exc
    connection.commit()
    if seen_lines != stream.candidate_references:
        raise ObservationXReuseAuditError("candidate file row count differs from Admission receipt")
    total = int(connection.execute("SELECT COUNT(*) FROM candidate_references").fetchone()[0])
    if total != stream.candidate_references:
        raise ObservationXReuseAuditError("candidate reference conservation failed")
    unique_assets = int(connection.execute("SELECT COUNT(*) FROM assets").fetchone()[0])
    references_by_asset = int(connection.execute(
        "SELECT COUNT(*) FROM candidate_references WHERE asset_id IS NOT NULL"
    ).fetchone()[0])
    if unique_assets < 1 or references_by_asset != total:
        raise ObservationXReuseAuditError("candidate assets were deduplicated before conservation")

    return True, total, inserted


def _admission_line_ranges(values: Iterable[int]) -> list[tuple[int, int]]:
    """Mirror the semantic auditor's bounded source-line batching convention."""
    ordered = sorted(set(values))
    if not ordered:
        return []
    ranges: list[tuple[int, int]] = []
    start = prior = ordered[0]
    for value in ordered[1:]:
        if value - prior > 16 or value - start >= 2048:
            ranges.append((start, prior))
            start = value
        prior = value
    ranges.append((start, prior))

    return ranges


def _batch_admission_source_records(
    connection: sqlite3.Connection,
    transitions: Sequence[tuple[int, Mapping[str, Any]]],
) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    requested = {admission_transition_reference(transition) for _, transition in transitions}
    by_source: dict[str, list[int]] = {}
    for source_sha256, source_line, _ in requested:
        by_source.setdefault(source_sha256, []).append(source_line)
    compressed: dict[tuple[str, int, str], bytes] = {}
    for source_sha256, lines in sorted(by_source.items()):
        for first, last in _admission_line_ranges(lines):
            for source_line, content_sha256, blob in connection.execute(
                "SELECT source_line,content_sha256,compressed_json FROM observations "
                "WHERE source_sha256=? AND source_line BETWEEN ? AND ? "
                "ORDER BY source_line,content_sha256",
                (source_sha256, first, last),
            ):
                key = (source_sha256, int(source_line), str(content_sha256).lower())
                if key in requested:
                    compressed[key] = blob
    if set(compressed) != requested:
        raise ObservationXReuseAuditError(
            "Admission batch observation coverage is incomplete"
        )
    records: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for key, blob in compressed.items():
        try:
            value = json.loads(zlib.decompress(blob))
        except (TypeError, zlib.error, json.JSONDecodeError) as exc:
            raise ObservationXReuseAuditError(
                "Admission observation payload is not valid compressed JSON"
            ) from exc
        if not isinstance(value, Mapping):
            raise ObservationXReuseAuditError("Admission observation is not an object")
        records[key] = value

    return records


def _admission_binding_candidates(
    connection: sqlite3.Connection,
    binding_id: str,
    *,
    start_admitted_ordinal: int = 0,
    batch_rows: int = 512,
) -> Iterable[tuple[int, int, Mapping[str, Any], Mapping[str, Any]]]:
    _require_int(start_admitted_ordinal, field="Admission resume admitted ordinal")
    _require_int(batch_rows, field="Admission batch rows", minimum=1)
    cursor = connection.execute(
        "SELECT local_row,compressed_json FROM transitions "
        "WHERE binding_id=? ORDER BY local_row LIMIT -1 OFFSET ?",
        (binding_id, start_admitted_ordinal),
    )
    batch: list[tuple[int, int, Mapping[str, Any]]] = []
    admitted_ordinal = start_admitted_ordinal
    prior_local_row: int | None = None
    if start_admitted_ordinal:
        prior = connection.execute(
            "SELECT local_row FROM transitions WHERE binding_id=? "
            "ORDER BY local_row LIMIT 1 OFFSET ?",
            (binding_id, start_admitted_ordinal - 1),
        ).fetchone()
        if prior is None:
            raise ObservationXReuseAuditError(
                "Admission resume ordinal is outside binding coverage"
            )
        prior_local_row = _require_int(
            prior[0],
            field="Admission prior transition local row",
        )
    for local_row, compressed_transition in cursor:
        normalized_local_row = _require_int(
            local_row,
            field="Admission transition local row",
        )
        if prior_local_row is not None and normalized_local_row <= prior_local_row:
            raise ObservationXReuseAuditError(
                "Admission transition local rows are not strictly increasing"
            )
        prior_local_row = normalized_local_row
        try:
            transition = json.loads(zlib.decompress(compressed_transition))
        except (TypeError, zlib.error, json.JSONDecodeError) as exc:
            raise ObservationXReuseAuditError(
                "Admission transition payload is not valid compressed JSON"
            ) from exc
        if not isinstance(transition, Mapping):
            raise ObservationXReuseAuditError("Admission transition is not an object")
        batch.append((admitted_ordinal, normalized_local_row, transition))
        admitted_ordinal += 1
        if len(batch) < batch_rows:
            continue
        source_records = _batch_admission_source_records(
            connection,
            [(row, item) for _, row, item in batch],
        )
        for ordinal, row, item in batch:
            yield ordinal, row, item, source_records[admission_transition_reference(item)]
        batch.clear()
    if batch:
        source_records = _batch_admission_source_records(
            connection,
            [(row, item) for _, row, item in batch],
        )
        for ordinal, row, item in batch:
            yield ordinal, row, item, source_records[admission_transition_reference(item)]


def _build_production_transition_registry(
    connection: sqlite3.Connection,
    source: ProductionAdmission,
    binding: CandidateStreamBinding,
    *,
    source_event_factory: Callable[..., Any],
    registry_builder: Callable[..., Any],
) -> Any:
    def events() -> Iterable[Any]:
        observed_rows = 0
        for admitted_ordinal, local_row, transition, source_record in (
            _admission_binding_candidates(
                connection,
                binding.binding_id,
            )
        ):
            if admitted_ordinal != observed_rows:
                raise ObservationXReuseAuditError(
                    "Admission admitted ordinals are not exact and contiguous"
                )
            observed_rows += 1
            yield source_event_factory(
                admission_identity=source.admission_identity,
                binding_id=binding.binding_id,
                local_row=local_row,
                global_row=admission_global_candidate_row(
                    binding.row_start,
                    admitted_ordinal,
                ),
                transition=transition,
                source_record=source_record,
                verify_source_content=False,
            )
        if observed_rows != binding.rows:
            raise ObservationXReuseAuditError(
                "Admission binding rows differ from frozen semantic registry"
            )

    registry = registry_builder(
        events(),
        admission_identity=source.admission_identity,
        binding_id=binding.binding_id,
        expected_rows=binding.rows,
    )
    if (
        getattr(registry, "admission_identity", None) != source.admission_identity
        or getattr(registry, "binding_id", None) != binding.binding_id
        or getattr(registry, "row_count", None) != binding.rows
    ):
        raise ObservationXReuseAuditError(
            "Admission transition registry does not cover the frozen binding"
        )
    _require_sha256(
        getattr(registry, "registry_sha256", None),
        field="Admission transition-registry SHA",
    )

    return registry


def _candidate_from_aligned_pre_action(
    aligned: Any,
    *,
    normalizer: ObservationNormalizer,
) -> tuple[Candidate, str, tuple[str, ...]]:
    from deepmew.replay.information_boundary import assert_deployable_observation

    event = aligned.pre_action_event
    asset = event.observation_asset
    proof = aligned.proof
    if event.source_corrupt or asset is None or not isinstance(event.observation, Mapping):
        raise ObservationXReuseAuditError("aligned pre-action observation is absent or corrupt")
    if not (
        proof.action_equality_proven
        and proof.actor_seat_proven
        and proof.event_order_proven
        and proof.source_hash_bound
        and proof.pre_action_event_id == event.event_id
        and proof.pre_action_observation_id == asset.observation_id
    ):
        raise ObservationXReuseAuditError("aligned pre-action causal proof is incomplete")
    if (
        event.actor_seat != event.controlled_player
        or event.actor_seat != asset.actor_seat
        or event.source_game_id != asset.source_game_id
        or event.source_event_position != asset.source_event_position
    ):
        raise ObservationXReuseAuditError("aligned pre-action asset identity is inconsistent")
    assert_deployable_observation(event.observation, expected_mode="actor_visible")
    normalized = normalizer(event.observation)
    if not isinstance(normalized, dict):
        raise ObservationXReuseAuditError("normalized pre-action observation is not an object")
    assert_deployable_observation(normalized, expected_mode="actor_visible")
    asset_id = str(asset.observation_id or "")
    source_game_id = str(asset.source_game_id or "")
    visibility_schema = str(asset.visibility_schema_version or "")
    feature_version = str(asset.feature_version or "")
    if not all((asset_id, source_game_id, visibility_schema, feature_version)):
        raise ObservationXReuseAuditError("aligned pre-action asset metadata is incomplete")
    source_event_position = _require_int(
        asset.source_event_position,
        field="aligned source event position",
    )
    acting_seat = _require_int(asset.actor_seat, field="aligned acting seat")
    aliases = tuple(sorted({
        _require_sha256(
            asset.source_content_sha256,
            field="aligned source-observation content SHA",
        ),
        _require_sha256(
            asset.content_sha256,
            field="aligned canonical-observation content SHA",
        ),
    }))
    own_hand = _observation_own_hand(normalized)
    normalized_sha256 = _stable_sha256(normalized)
    candidate = Candidate(
        asset_id=asset_id,
        source_game_id=source_game_id,
        source_event_position=source_event_position,
        acting_seat=acting_seat,
        public_visibility_schema=visibility_schema,
        observation_feature_version=feature_version,
        normalized_observation_sha256=normalized_sha256,
        normalized_observation=normalized,
        source_observation_sha256_aliases=aliases,
        source_record={"own_hand": own_hand},
    )
    asset_payload = {
        "observation_asset_id": candidate.asset_id,
        "source_game_id": candidate.source_game_id,
        "source_event_position": candidate.source_event_position,
        "acting_seat": candidate.acting_seat,
        "public_visibility_schema": candidate.public_visibility_schema,
        "observation_feature_version": candidate.observation_feature_version,
        "normalized_observation_sha256": candidate.normalized_observation_sha256,
        "normalized_observation": candidate.normalized_observation,
        "source_observation_sha256_aliases": list(candidate.source_observation_sha256_aliases),
        "source_record": candidate.source_record,
    }

    return candidate, _stable_sha256(asset_payload), ()


def _production_reference_id(
    source: ProductionAdmission,
    *,
    binding_id: str,
    local_row: int,
    transition: Mapping[str, Any],
) -> str:
    payload = {
        "admission_receipt_sha256": source.admission_receipt_sha256,
        "binding_id": binding_id,
        "local_row": local_row,
        "transition_sha256": _stable_sha256(transition),
    }

    return f"oxref1-{_stable_sha256(payload)}"


def _ingest_production_admission_references(
    connection: sqlite3.Connection,
    source: ProductionAdmission,
    *,
    normalizer: ObservationNormalizer,
    max_new_references: int | None,
) -> tuple[bool, int, int]:
    from deepmew.replay.contract.causal_alignment import (
        KaggleEpisodeRepository,
        align_decision,
        build_admission_transition_registry,
    )
    from deepmew.replay.contract.source_events import source_event_from_admission

    existing = int(connection.execute("SELECT COUNT(*) FROM candidate_references").fetchone()[0])
    if existing:
        minimum, maximum = connection.execute(
            "SELECT MIN(global_candidate_row),MAX(global_candidate_row) FROM candidate_references"
        ).fetchone()
        if minimum != 0 or maximum != existing - 1:
            raise ObservationXReuseAuditError("resumed candidate ledger is not contiguous")
    if existing > source.candidate_references:
        raise ObservationXReuseAuditError("resumed candidate ledger exceeds Admission receipt")
    inserted = 0
    repository = KaggleEpisodeRepository(ROOT)
    admission_connection = _readonly_admission_connection(source.admission_database_path)
    try:
        for binding in source.bindings:
            binding_end = binding.row_start + binding.rows
            if existing + inserted >= binding_end:
                continue
            if max_new_references is not None and inserted >= max_new_references:
                connection.commit()
                return False, existing + inserted, inserted
            start_admitted_ordinal = max(
                0,
                existing + inserted - binding.row_start,
            )
            transition_registry = _build_production_transition_registry(
                admission_connection,
                source,
                binding,
                source_event_factory=source_event_from_admission,
                registry_builder=build_admission_transition_registry,
            )
            expected_admitted_ordinal = start_admitted_ordinal
            for admitted_ordinal, local_row, transition, source_record in (
                _admission_binding_candidates(
                    admission_connection,
                    binding.binding_id,
                    start_admitted_ordinal=start_admitted_ordinal,
                )
            ):
                if (
                    admitted_ordinal != expected_admitted_ordinal
                    or admitted_ordinal >= binding.rows
                ):
                    raise ObservationXReuseAuditError(
                        "Admission admitted ordinals are not exact and contiguous"
                    )
                if max_new_references is not None and inserted >= max_new_references:
                    connection.commit()
                    return False, existing + inserted, inserted
                global_candidate_row = admission_global_candidate_row(
                    binding.row_start,
                    admitted_ordinal,
                )
                reference_id = _production_reference_id(
                    source,
                    binding_id=binding.binding_id,
                    local_row=local_row,
                    transition=transition,
                )
                try:
                    source_event = source_event_from_admission(
                        admission_identity=source.admission_identity,
                        binding_id=binding.binding_id,
                        local_row=local_row,
                        global_row=global_candidate_row,
                        transition=transition,
                        source_record=source_record,
                        verify_source_content=False,
                    )
                    aligned = align_decision(
                        source_event,
                        kaggle_repository=repository,
                        transition_registry=transition_registry,
                    )
                    candidate, asset_payload_sha256, precheck_reasons = (
                        _candidate_from_aligned_pre_action(
                            aligned,
                            normalizer=normalizer,
                        )
                    )
                except Exception:
                    _insert_failed_candidate_reference(
                        connection,
                        global_candidate_row=global_candidate_row,
                        candidate_reference_id=reference_id,
                        admission_binding_id=binding.binding_id,
                        admission_local_row=local_row,
                        failure_code="canonical_pre_action_observation_unavailable",
                    )
                else:
                    _insert_verified_candidate_reference(
                        connection,
                        candidate=candidate,
                        reference={
                            "global_candidate_row": global_candidate_row,
                            "candidate_reference_id": reference_id,
                            "admission_binding_id": binding.binding_id,
                            "admission_local_row": local_row,
                            "observation_asset_id": candidate.asset_id,
                        },
                        asset_payload_sha256=asset_payload_sha256,
                        precheck_reasons=precheck_reasons,
                    )
                inserted += 1
                expected_admitted_ordinal += 1
                if inserted % 2048 == 0:
                    connection.commit()
            if expected_admitted_ordinal != binding.rows:
                raise ObservationXReuseAuditError(
                    "Admission binding rows differ from frozen semantic registry"
                )
    finally:
        admission_connection.close()
    connection.commit()
    total = int(connection.execute("SELECT COUNT(*) FROM candidate_references").fetchone()[0])
    if total != source.candidate_references:
        raise ObservationXReuseAuditError("candidate reference conservation failed")
    valid_references = int(connection.execute(
        "SELECT COUNT(*) FROM candidate_references WHERE asset_id IS NOT NULL"
    ).fetchone()[0])
    failed_references = int(connection.execute(
        "SELECT COUNT(*) FROM candidate_references WHERE asset_id IS NULL "
        "AND reference_failure_code IS NOT NULL"
    ).fetchone()[0])
    if valid_references + failed_references != total:
        raise ObservationXReuseAuditError("candidate reference disposition is incomplete")

    return True, total, inserted


def _query_matching_aliases(
    connection: sqlite3.Connection,
    aliases: Sequence[str],
) -> set[str]:
    matched: set[str] = set()
    unique = sorted(set(aliases))
    for start in range(0, len(unique), 800):
        chunk = unique[start:start + 800]
        placeholders = ",".join("?" for _ in chunk)
        matched.update(
            str(value)
            for value, in connection.execute(
                f"SELECT DISTINCT alias_sha256 FROM asset_aliases "
                f"WHERE alias_sha256 IN ({placeholders})",
                chunk,
            )
        )

    return matched


def _scan_cache_binding_to_store(
    connection: sqlite3.Connection,
    binding: CacheBinding,
    *,
    run_id: str,
) -> dict[str, Any]:
    if not binding.transition_path.is_file() or _sha256_path(binding.transition_path) != binding.transition_sha256:
        raise ObservationXReuseAuditError("transition sidecar hash mismatch")
    connection.execute("DELETE FROM cache_aliases WHERE binding_id=?", (binding.binding_id,))
    connection.execute("DELETE FROM binding_scans WHERE binding_id=?", (binding.binding_id,))
    pending: list[dict[str, Any]] = []
    matches = 0
    rows_scanned = 0
    ledger = hashlib.sha256()

    def flush_pending() -> None:
        nonlocal matches
        matched_aliases = _query_matching_aliases(
            connection,
            [str(item["source_alias_sha256"]) for item in pending],
        )
        for item in pending:
            if item["source_alias_sha256"] not in matched_aliases:
                continue
            connection.execute(
                "INSERT INTO cache_aliases VALUES (?,?,?,?,?,?,?,?,?,?,?,?,NULL)",
                (
                    item["global_row"],
                    item["binding_id"],
                    item["decision_id"],
                    item["local_row"],
                    item["row_start"],
                    item["source_alias_sha256"],
                    item["source_path"],
                    item["source_line"],
                    item["source_field"],
                    item["source_schema"],
                    item["source_file_sha256"],
                    item["declared_state_before_feature_row"],
                ),
            )
            ledger.update(_canonical_json_bytes(item) + b"\n")
            matches += 1
        pending.clear()

    try:
        with gzip.open(binding.transition_path, "rt", encoding="utf-8") as handle:
            for local_row, line in enumerate(handle):
                rows_scanned += 1
                try:
                    transition = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ObservationXReuseAuditError(
                        f"invalid transition JSON in {binding.binding_id} row {local_row}"
                    ) from exc
                if not isinstance(transition, dict):
                    raise ObservationXReuseAuditError("transition sidecar row is not an object")
                if transition.get("schema_version") != TRANSITION_SCHEMA_VERSION:
                    raise ObservationXReuseAuditError("transition schema is not current")
                if transition.get("information_mode") != "actor_visible":
                    raise ObservationXReuseAuditError("transition is not actor-visible")
                if (
                    transition.get("source_id") != binding.source_id
                    or transition.get("part") != binding.part_number
                    or transition.get("row") != local_row
                ):
                    raise ObservationXReuseAuditError("transition identity differs from cache binding")
                state_before = transition.get("state_before")
                reference = state_before.get("source_observation") if isinstance(state_before, dict) else None
                if not isinstance(reference, dict):
                    raise ObservationXReuseAuditError("transition lacks source-observation reference")
                source_alias_sha256 = _require_sha256(
                    reference.get("content_sha256"),
                    field="transition source-observation SHA",
                )
                source_path_value = reference.get("source_path")
                if not isinstance(source_path_value, str) or not Path(source_path_value).is_absolute():
                    raise ObservationXReuseAuditError("source-observation path must be absolute")
                source_path = Path(source_path_value).resolve()
                source_file_sha256 = _source_expected_sha(binding, source_path, transition)
                decision_id = transition.get("decision_id")
                if not isinstance(decision_id, str) or not decision_id:
                    raise ObservationXReuseAuditError("transition decision ID is missing")
                source_field = reference.get("source_field")
                if not isinstance(source_field, str) or not source_field:
                    raise ObservationXReuseAuditError("source-observation field is missing")
                pending.append({
                    "global_row": binding.row_start + local_row,
                    "binding_id": binding.binding_id,
                    "decision_id": decision_id,
                    "local_row": local_row,
                    "row_start": binding.row_start,
                    "source_alias_sha256": source_alias_sha256,
                    "source_path": str(source_path),
                    "source_line": _require_int(
                        reference.get("source_line"),
                        field="source-observation line",
                        minimum=1,
                    ),
                    "source_field": source_field,
                    "source_schema": transition.get("source_schema"),
                    "source_file_sha256": source_file_sha256,
                    "declared_state_before_feature_row": state_before.get("feature_row"),
                })
                if len(pending) >= 2048:
                    flush_pending()
    except OSError as exc:
        connection.rollback()
        raise ObservationXReuseAuditError(
            f"cannot read transition sidecar: {binding.transition_path}"
        ) from exc
    if pending:
        flush_pending()
    if rows_scanned != binding.rows:
        connection.rollback()
        raise ObservationXReuseAuditError("transition sidecar row count differs from cache binding")
    match_ledger_sha256 = ledger.hexdigest()
    connection.execute(
        "INSERT INTO binding_scans VALUES (?,?,?,?,?,?)",
        (
            binding.binding_id,
            run_id,
            binding.transition_sha256,
            rows_scanned,
            matches,
            match_ledger_sha256,
        ),
    )
    connection.commit()
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "status": "completed",
        "run_id": run_id,
        "binding_id": binding.binding_id,
        "row_start": binding.row_start,
        "rows_scanned": rows_scanned,
        "transition_sidecar_sha256": binding.transition_sha256,
        "matches": matches,
        "match_ledger_sha256": match_ledger_sha256,
        "coordinate_formula": "row_start + local_row",
        "state_before_feature_row_used": False,
    }

    return _seal_payload(payload, field="checkpoint_sha256")


def _resume_cache_binding_scan(
    connection: sqlite3.Connection,
    checkpoint_path: Path,
    binding: CacheBinding,
    *,
    run_id: str,
) -> dict[str, Any]:
    if _sha256_path(binding.transition_path) != binding.transition_sha256:
        raise ObservationXReuseAuditError("transition sidecar hash mismatch during resume")
    checkpoint = _read_json_object(checkpoint_path)
    _verify_sealed_payload(checkpoint, field="checkpoint_sha256", label="cache scan checkpoint")
    if (
        checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or checkpoint.get("status") != "completed"
        or checkpoint.get("run_id") != run_id
        or checkpoint.get("binding_id") != binding.binding_id
        or checkpoint.get("row_start") != binding.row_start
        or checkpoint.get("rows_scanned") != binding.rows
        or checkpoint.get("transition_sidecar_sha256") != binding.transition_sha256
    ):
        raise ObservationXReuseAuditError("cache scan checkpoint binding mismatch")
    row = connection.execute(
        "SELECT run_id,transition_sha256,rows_scanned,matches,match_ledger_sha256 "
        "FROM binding_scans WHERE binding_id=?",
        (binding.binding_id,),
    ).fetchone()
    expected = (
        run_id,
        binding.transition_sha256,
        binding.rows,
        checkpoint["matches"],
        checkpoint["match_ledger_sha256"],
    )
    actual_matches = int(connection.execute(
        "SELECT COUNT(*) FROM cache_aliases WHERE binding_id=?",
        (binding.binding_id,),
    ).fetchone()[0])
    if row != expected or actual_matches != checkpoint["matches"]:
        raise ObservationXReuseAuditError("cache scan checkpoint/work-index mismatch")

    return checkpoint


def _source_scan_digest(
    connection: sqlite3.Connection,
    source_path: str,
) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    for global_row, normalized_sha256 in connection.execute(
        "SELECT global_row,normalized_observation_sha256 FROM cache_aliases "
        "WHERE source_path=? ORDER BY source_line,global_row",
        (source_path,),
    ):
        if normalized_sha256 is None:
            raise ObservationXReuseAuditError("source normalization checkpoint is incomplete")
        digest.update(_canonical_json_bytes({
            "global_row": int(global_row),
            "normalized_observation_sha256": str(normalized_sha256),
        }) + b"\n")
        count += 1

    return count, digest.hexdigest()


def _normalize_matched_sources(
    connection: sqlite3.Connection,
    *,
    normalizer: ObservationNormalizer,
) -> tuple[int, int]:
    source_rows = connection.execute(
        "SELECT source_path,MIN(source_file_sha256),MAX(source_file_sha256),COUNT(*) "
        "FROM cache_aliases GROUP BY source_path ORDER BY source_path"
    ).fetchall()
    resumed = 0
    executed = 0
    for source_path_value, minimum_sha, maximum_sha, expected_matches in source_rows:
        if minimum_sha != maximum_sha:
            raise ObservationXReuseAuditError("cache aliases disagree on source-file hash")
        source_path = Path(str(source_path_value))
        if not source_path.is_file() or _sha256_path(source_path) != minimum_sha:
            raise ObservationXReuseAuditError(f"source file hash mismatch: {source_path}")
        scan = connection.execute(
            "SELECT source_sha256,matches,normalization_ledger_sha256 FROM source_scans "
            "WHERE source_path=?",
            (str(source_path),),
        ).fetchone()
        if scan is not None:
            actual_count, actual_digest = _source_scan_digest(connection, str(source_path))
            if scan != (minimum_sha, actual_count, actual_digest) or actual_count != expected_matches:
                raise ObservationXReuseAuditError("source normalization checkpoint mismatch")
            resumed += 1
            continue

        connection.execute(
            "UPDATE cache_aliases SET normalized_observation_sha256=NULL WHERE source_path=?",
            (str(source_path),),
        )
        cursor = connection.execute(
            "SELECT source_line,global_row,source_field,source_schema,source_alias_sha256 "
            "FROM cache_aliases WHERE source_path=? ORDER BY source_line,global_row",
            (str(source_path),),
        )
        grouped = itertools.groupby(cursor, key=lambda item: int(item[0]))
        try:
            target_line, group = next(grouped)
        except StopIteration:
            raise ObservationXReuseAuditError("source scan has no matched rows")
        group_rows = list(group)
        updates: list[tuple[str, int]] = []
        digest = hashlib.sha256()
        normalized_count = 0
        try:
            with _open_jsonl(source_path) as handle:
                for line_number, line in enumerate(handle, start=1):
                    if line_number < target_line:
                        continue
                    if line_number > target_line:
                        raise ObservationXReuseAuditError(
                            f"source file is missing referenced line {target_line}: {source_path}"
                        )
                    try:
                        source_record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ObservationXReuseAuditError(
                            f"invalid source JSON: {source_path}:{line_number}"
                        ) from exc
                    if not isinstance(source_record, dict):
                        raise ObservationXReuseAuditError("source row is not an object")
                    for _, global_row, source_field, source_schema, source_alias_sha256 in group_rows:
                        adapters = SOURCE_OBSERVATION_FIELD_BY_SCHEMA.get(str(source_schema), {})
                        actual_field = adapters.get(str(source_field), str(source_field))
                        observation = source_record.get(actual_field)
                        if not isinstance(observation, dict):
                            raise ObservationXReuseAuditError(
                                "referenced source observation is not an object"
                            )
                        if _stable_sha256(observation) != source_alias_sha256:
                            raise ObservationXReuseAuditError(
                                "source-observation content hash mismatch"
                            )
                        try:
                            normalized = normalizer(observation)
                        except (TypeError, ValueError) as exc:
                            raise ObservationXReuseAuditError(
                                "source observation normalization failed"
                            ) from exc
                        normalized_sha256 = _stable_sha256(normalized)
                        updates.append((normalized_sha256, int(global_row)))
                        digest.update(_canonical_json_bytes({
                            "global_row": int(global_row),
                            "normalized_observation_sha256": normalized_sha256,
                        }) + b"\n")
                        normalized_count += 1
                    if len(updates) >= 2048:
                        connection.executemany(
                            "UPDATE cache_aliases SET normalized_observation_sha256=? "
                            "WHERE global_row=?",
                            updates,
                        )
                        updates.clear()
                    try:
                        target_line, group = next(grouped)
                        group_rows = list(group)
                    except StopIteration:
                        target_line = -1
                        break
        except OSError as exc:
            connection.rollback()
            raise ObservationXReuseAuditError(f"cannot read source file: {source_path}") from exc
        if target_line != -1:
            connection.rollback()
            raise ObservationXReuseAuditError(
                f"source file is missing referenced line {target_line}: {source_path}"
            )
        if updates:
            connection.executemany(
                "UPDATE cache_aliases SET normalized_observation_sha256=? WHERE global_row=?",
                updates,
            )
        if normalized_count != expected_matches:
            connection.rollback()
            raise ObservationXReuseAuditError("source normalization count mismatch")
        connection.execute(
            "INSERT INTO source_scans VALUES (?,?,?,?)",
            (str(source_path), minimum_sha, normalized_count, digest.hexdigest()),
        )
        connection.commit()
        executed += 1

    return resumed, executed


def _candidate_from_asset_row(row: Sequence[Any]) -> tuple[Candidate, tuple[str, ...]]:
    (
        asset_id,
        source_game_id,
        source_event_position,
        acting_seat,
        public_visibility_schema,
        observation_feature_version,
        normalized_observation_sha256,
        observation_blob,
        source_record_blob,
        precheck_reasons_json,
    ) = row
    observation = json.loads(zlib.decompress(observation_blob))
    source_record = json.loads(zlib.decompress(source_record_blob))

    return Candidate(
        asset_id=str(asset_id),
        source_game_id=str(source_game_id),
        source_event_position=int(source_event_position),
        acting_seat=int(acting_seat),
        public_visibility_schema=str(public_visibility_schema),
        observation_feature_version=str(observation_feature_version),
        normalized_observation_sha256=str(normalized_observation_sha256),
        normalized_observation=observation,
        source_observation_sha256_aliases=(),
        source_record=source_record,
    ), tuple(json.loads(precheck_reasons_json))


def _classify_assets_to_store(
    connection: sqlite3.Connection,
    *,
    x_path: Path,
    x_rows: int,
    feature_encoder: FeatureEncoder,
    max_new_assets: int | None,
) -> tuple[bool, int, int]:
    total_assets = int(connection.execute("SELECT COUNT(*) FROM assets").fetchone()[0])
    already = int(connection.execute("SELECT COUNT(*) FROM classifications").fetchone()[0])
    executed = 0
    with _XRowReader(x_path, x_rows) as x_reader:
        while True:
            rows = connection.execute(
                "SELECT a.asset_id,a.source_game_id,a.source_event_position,a.acting_seat,"
                "a.public_visibility_schema,a.observation_feature_version,"
                "a.normalized_observation_sha256,a.normalized_observation_blob,"
                "a.source_record_blob,a.precheck_reasons_json "
                "FROM assets a LEFT JOIN classifications c ON c.asset_id=a.asset_id "
                "WHERE c.asset_id IS NULL ORDER BY a.asset_id LIMIT 256"
            ).fetchall()
            if not rows:
                break
            for row in rows:
                if max_new_assets is not None and executed >= max_new_assets:
                    connection.commit()
                    return False, already + executed, executed
                candidate, precheck_reasons = _candidate_from_asset_row(row)
                aliases = tuple(
                    str(alias)
                    for alias, in connection.execute(
                        "SELECT alias_sha256 FROM asset_aliases WHERE asset_id=? ORDER BY alias_sha256",
                        (candidate.asset_id,),
                    )
                )
                reference_count = int(connection.execute(
                    "SELECT COUNT(*) FROM candidate_references WHERE asset_id=?",
                    (candidate.asset_id,),
                ).fetchone()[0])
                cache_rows = connection.execute(
                    "SELECT DISTINCT c.global_row,c.binding_id,c.local_row,c.row_start,"
                    "c.source_alias_sha256,c.normalized_observation_sha256 "
                    "FROM cache_aliases c JOIN asset_aliases a "
                    "ON a.alias_sha256=c.source_alias_sha256 "
                    "WHERE a.asset_id=? ORDER BY c.global_row,c.binding_id",
                    (candidate.asset_id,),
                )
                reasons = set(precheck_reasons)
                expected_bytes, feature_error = _feature_bytes(
                    candidate,
                    feature_encoder=feature_encoder,
                )
                if feature_error is not None:
                    reasons.add(feature_error)
                expected_tensor_sha256 = (
                    hashlib.sha256(expected_bytes).hexdigest()
                    if expected_bytes is not None
                    else None
                )
                alias_ledger = hashlib.sha256()
                cache_alias_count = 0
                cache_alias_global_row_min = None
                cache_alias_global_row_max = None
                first_tensor_sha256 = None
                alias_tensor_bytes_disagree = False
                for (
                    global_row,
                    binding_id,
                    local_row,
                    row_start,
                    source_alias_sha256,
                    normalized_observation_sha256,
                ) in cache_rows:
                    cache_alias_count += 1
                    if cache_alias_global_row_min is None:
                        cache_alias_global_row_min = int(global_row)
                    cache_alias_global_row_max = int(global_row)
                    if normalized_observation_sha256 != candidate.normalized_observation_sha256:
                        reasons.add("normalized_observation_mismatch")
                    x_bytes, tensor_sha256, is_finite = x_reader.read(int(global_row))
                    if first_tensor_sha256 is None:
                        first_tensor_sha256 = tensor_sha256
                    elif first_tensor_sha256 != tensor_sha256:
                        alias_tensor_bytes_disagree = True
                    if not is_finite:
                        reasons.add("x_tensor_nonfinite")
                    if expected_bytes is not None and x_bytes != expected_bytes:
                        reasons.add("feature_bytes_mismatch")
                    alias_ledger.update(_canonical_json_bytes({
                        "global_row": int(global_row),
                        "binding_id": str(binding_id),
                        "local_row": int(local_row),
                        "row_start": int(row_start),
                        "source_alias_sha256": str(source_alias_sha256),
                        "normalized_observation_sha256": str(normalized_observation_sha256),
                        "tensor_sha256": tensor_sha256,
                    }) + b"\n")
                if alias_tensor_bytes_disagree:
                    reasons.add("alias_tensor_bytes_disagree")
                if cache_alias_count == 0:
                    reasons.add("no_cache_alias")
                    resolved_global_row = None
                    tensor_sha256 = None
                    if reasons == {"no_cache_alias"}:
                        classification = "NEW_TENSOR_REQUIRED"
                    else:
                        classification = "AMBIGUOUS_FAIL_CLOSED"
                elif reasons:
                    classification = "AMBIGUOUS_FAIL_CLOSED"
                    resolved_global_row = None
                    tensor_sha256 = None
                else:
                    classification = "REUSED_EXACT"
                    resolved_global_row = cache_alias_global_row_min
                    tensor_sha256 = expected_tensor_sha256
                payload = {
                    "schema_version": "deepmew.observation_x_reuse_registry_asset.v1",
                    "observation_asset_id": candidate.asset_id,
                    "source_game_id": candidate.source_game_id,
                    "source_event_position": candidate.source_event_position,
                    "acting_seat": candidate.acting_seat,
                    "public_visibility_schema": candidate.public_visibility_schema,
                    "observation_feature_version": candidate.observation_feature_version,
                    "normalized_observation_sha256": candidate.normalized_observation_sha256,
                    "source_observation_sha256_aliases": list(aliases),
                    "candidate_reference_count": reference_count,
                    "classification": classification,
                    "reason_codes": sorted(reasons),
                    "cache_alias_count": cache_alias_count,
                    "cache_alias_ledger_sha256": alias_ledger.hexdigest(),
                    "cache_alias_global_row_min": cache_alias_global_row_min,
                    "cache_alias_global_row_max": cache_alias_global_row_max,
                    "resolved_global_row": resolved_global_row,
                    "tensor_sha256": tensor_sha256,
                    "expected_tensor_sha256": expected_tensor_sha256,
                }
                connection.execute(
                    "INSERT INTO classifications VALUES (?,?,?,?,?,?,?,?)",
                    (
                        candidate.asset_id,
                        classification,
                        resolved_global_row,
                        tensor_sha256,
                        reference_count,
                        cache_alias_count,
                        json.dumps(sorted(reasons), separators=(",", ":")),
                        sqlite3.Binary(zlib.compress(_canonical_json_bytes(payload), level=3)),
                    ),
                )
                executed += 1
            connection.commit()
    completed = int(connection.execute("SELECT COUNT(*) FROM classifications").fetchone()[0])
    if completed != total_assets:
        raise ObservationXReuseAuditError("asset classification conservation failed")

    return True, completed, executed


def _atomic_write_gzip_jsonl(
    path: Path,
    rows: Iterable[bytes],
) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    count = 0
    try:
        with temporary_path.open("wb") as raw_handle:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_handle,
                compresslevel=6,
                mtime=0,
            ) as handle:
                for payload in rows:
                    handle.write(payload + b"\n")
                    count += 1
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return count, _sha256_path(path)


def _reference_ledger_sha256(connection: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    for ordinal, reference_sha256 in connection.execute(
        "SELECT global_candidate_row,reference_sha256 FROM candidate_references "
        "ORDER BY global_candidate_row"
    ):
        digest.update(_canonical_json_bytes({
            "global_candidate_row": int(ordinal),
            "reference_sha256": str(reference_sha256),
        }) + b"\n")

    return digest.hexdigest()


def _reference_registry_rows(connection: sqlite3.Connection) -> Iterable[bytes]:
    for row in connection.execute(
        "SELECT r.global_candidate_row,r.candidate_reference_id,"
        "r.admission_binding_id,r.admission_local_row,r.asset_id,"
        "r.reference_failure_code,r.reference_sha256,c.classification "
        "FROM candidate_references AS r LEFT JOIN classifications AS c "
        "ON c.asset_id=r.asset_id ORDER BY r.global_candidate_row"
    ):
        (
            global_candidate_row,
            candidate_reference_id,
            binding_id,
            local_row,
            asset_id,
            failure_code,
            reference_sha256,
            classification,
        ) = row
        if classification is None:
            if asset_id is not None or failure_code is None:
                raise ObservationXReuseAuditError(
                    "candidate reference lacks an exact terminal disposition"
                )
            classification = "AMBIGUOUS_FAIL_CLOSED"
        yield _canonical_json_bytes({
            "schema_version": "deepmew.observation_x_reuse_reference.v1",
            "global_candidate_row": int(global_candidate_row),
            "candidate_reference_id": str(candidate_reference_id),
            "admission_binding_id": str(binding_id),
            "admission_local_row": int(local_row),
            "observation_asset_id": str(asset_id) if asset_id is not None else None,
            "classification": str(classification),
            "reference_failure_code": (
                str(failure_code) if failure_code is not None else None
            ),
            "reference_sha256": str(reference_sha256),
        })


def _candidate_source_input_receipts(
    source: CandidateStream | ProductionAdmission,
    *,
    candidate_references: int,
    reference_ledger_sha256: str,
) -> dict[str, Any]:
    if isinstance(source, CandidateStream):
        return {
            "mode": "synthetic_inline_candidate_manifest_test_only",
            "admission_receipt_sha256": source.admission_receipt_sha256,
            "semantic_audit_receipt_sha256": source.semantic_audit_receipt_sha256,
            "candidate_manifest": {
                "path": str(source.manifest_path),
                "sha256": source.manifest_sha256,
            },
            "candidate_reference_file": {
                "path": str(source.candidate_path),
                "sha256": source.candidate_sha256,
                "rows": candidate_references,
                "reference_ledger_sha256": reference_ledger_sha256,
            },
        }

    return {
        "mode": "direct_read_only_immutable_admission_v0",
        "admission_receipt_sha256": source.admission_receipt_sha256,
        "admission_identity_sha256": source.admission_identity,
        "admission_manifest": {
            "path": str(source.admission_manifest_path),
            "sha256": source.admission_manifest_sha256,
        },
        "admission_database": {
            "path": str(source.admission_database_path),
            "sha256": source.admission_database_sha256,
            "access": "sqlite_uri_mode_ro_immutable_query_only",
        },
        "semantic_audit_receipt": {
            "path": str(source.semantic_audit_receipt_path),
            "file_sha256": source.semantic_audit_receipt_file_sha256,
            "receipt_sha256": source.semantic_audit_receipt_sha256,
        },
        "semantic_binding_registry": {
            "path": str(source.binding_registry_path),
            "file_sha256": source.binding_registry_file_sha256,
            "binding_registry_sha256": source.binding_registry_sha256,
        },
        "canonical_implementation_hashes": {
            "path": str(source.implementation_hash_manifest_path),
            "file_sha256": source.implementation_hash_manifest_file_sha256,
            "manifest_sha256": source.implementation_hash_manifest_sha256,
            "causal_aligner_implementation_sha256": (
                source.causal_aligner_implementation_sha256
            ),
            "legality_validator_implementation_sha256": (
                source.legality_validator_implementation_sha256
            ),
            "action_schema_implementation_sha256": (
                source.action_schema_implementation_sha256
            ),
        },
        "candidate_reference_source": {
            "rows": candidate_references,
            "reference_ledger_sha256": reference_ledger_sha256,
            "loader_implementation_sha256": source.loader_implementation_sha256,
            "coordinate_convention": source.semantic_audit_receipt[
                "global_candidate_row_convention"
            ],
        },
    }


def _finalize_streaming_registry(
    connection: sqlite3.Connection,
    *,
    output_path: Path,
    run_identity: Mapping[str, Any],
    candidate_source: CandidateStream | ProductionAdmission,
    cache_manifest_path: Path,
    cache_manifest_sha256: str,
    catalog_path: Path,
    catalog_sha256: str,
    x_path: Path,
    x_sha256: str,
    x_rows: int,
    bindings: Sequence[CacheBinding],
) -> dict[str, Any]:
    candidate_references = int(connection.execute(
        "SELECT COUNT(*) FROM candidate_references"
    ).fetchone()[0])
    unique_assets = int(connection.execute("SELECT COUNT(*) FROM assets").fetchone()[0])
    classification_rows = connection.execute(
        "SELECT classification,COUNT(*),SUM(candidate_reference_count) "
        "FROM classifications GROUP BY classification ORDER BY classification"
    ).fetchall()
    asset_counts = {
        classification: count
        for classification, count, _ in classification_rows
    }
    reference_counts = {
        classification: references
        for classification, _, references in classification_rows
    }
    failed_reference_rows = connection.execute(
        "SELECT reference_failure_code,COUNT(*) FROM candidate_references "
        "WHERE asset_id IS NULL GROUP BY reference_failure_code "
        "ORDER BY reference_failure_code"
    ).fetchall()
    if any(failure_code is None for failure_code, _ in failed_reference_rows):
        raise ObservationXReuseAuditError("assetless candidate reference lacks failure code")
    failed_reference_counts = {
        str(failure_code): int(count)
        for failure_code, count in failed_reference_rows
    }
    failed_references = sum(failed_reference_counts.values())
    for classification in (
        "REUSED_EXACT",
        "NEW_TENSOR_REQUIRED",
        "AMBIGUOUS_FAIL_CLOSED",
    ):
        asset_counts.setdefault(classification, 0)
        reference_counts.setdefault(classification, 0)
    reference_counts["AMBIGUOUS_FAIL_CLOSED"] += failed_references
    if sum(asset_counts.values()) != unique_assets:
        raise ObservationXReuseAuditError("unique asset classification conservation failed")
    if sum(reference_counts.values()) != candidate_references:
        raise ObservationXReuseAuditError("candidate reference classification conservation failed")
    if candidate_references != candidate_source.candidate_references:
        raise ObservationXReuseAuditError("final candidate references differ from Admission receipt")
    reused_x_rows = int(connection.execute(
        "SELECT COUNT(DISTINCT resolved_global_row) FROM classifications "
        "WHERE classification='REUSED_EXACT'"
    ).fetchone()[0])
    reused_tensor_hashes = int(connection.execute(
        "SELECT COUNT(DISTINCT tensor_sha256) FROM classifications "
        "WHERE classification='REUSED_EXACT'"
    ).fetchone()[0])
    reused_cache_aliases = int(connection.execute(
        "SELECT COALESCE(SUM(cache_alias_count),0) FROM classifications "
        "WHERE classification='REUSED_EXACT'"
    ).fetchone()[0])
    ambiguous_cache_aliases = int(connection.execute(
        "SELECT COALESCE(SUM(cache_alias_count),0) FROM classifications "
        "WHERE classification='AMBIGUOUS_FAIL_CLOSED'"
    ).fetchone()[0])
    ambiguous_reason_counts: dict[str, int] = {}
    for reason_codes_json, in connection.execute(
        "SELECT reason_codes_json FROM classifications "
        "WHERE classification='AMBIGUOUS_FAIL_CLOSED'"
    ):
        for reason in json.loads(reason_codes_json):
            ambiguous_reason_counts[str(reason)] = ambiguous_reason_counts.get(str(reason), 0) + 1
    reference_registry_path = output_path / "OBSERVATION_X_REUSE_REGISTRY_V1_REFERENCES.jsonl.gz"
    reference_rows, reference_file_sha256 = _atomic_write_gzip_jsonl(
        reference_registry_path,
        _reference_registry_rows(connection),
    )
    if reference_rows != candidate_references:
        raise ObservationXReuseAuditError("reference registry row count mismatch")
    asset_registry_path = output_path / "OBSERVATION_X_REUSE_REGISTRY_V1_ASSETS.jsonl.gz"
    asset_rows, asset_file_sha256 = _atomic_write_gzip_jsonl(
        asset_registry_path,
        (
            zlib.decompress(payload_blob)
            for payload_blob, in connection.execute(
                "SELECT payload_blob FROM classifications ORDER BY asset_id"
            )
        ),
    )
    if asset_rows != unique_assets:
        raise ObservationXReuseAuditError("asset registry row count mismatch")
    checkpoint_receipts = [
        {
            "binding_id": binding.binding_id,
            "path": str(_checkpoint_path(output_path / "parts", binding).relative_to(output_path)),
            "sha256": _sha256_path(_checkpoint_path(output_path / "parts", binding)),
        }
        for binding in bindings
    ]
    source_receipts = [
        {
            "path": source_path,
            "sha256": source_sha256,
            "matched_cache_rows": matches,
            "normalization_ledger_sha256": ledger_sha256,
        }
        for source_path, source_sha256, matches, ledger_sha256 in connection.execute(
            "SELECT source_path,source_sha256,matches,normalization_ledger_sha256 "
            "FROM source_scans ORDER BY source_path"
        )
    ]
    reference_ledger_sha256 = _reference_ledger_sha256(connection)
    verdict = (
        "FAIL_CLOSED_AMBIGUOUS"
        if asset_counts["AMBIGUOUS_FAIL_CLOSED"] or failed_references
        else "COMPLETE_EXACT"
    )
    conservation = {
        "candidate_references_expected": candidate_source.candidate_references,
        "candidate_references_verified": candidate_references,
        "candidate_references_by_classification": reference_counts,
        "candidate_reference_conservation_exact": (
            candidate_references
            == sum(reference_counts.values())
            == candidate_source.candidate_references
        ),
        "unresolved_candidate_references": failed_references,
        "reference_failure_counts": failed_reference_counts,
        "unique_canonical_observation_assets": unique_assets,
        "assets_by_classification": asset_counts,
        "asset_conservation_exact": unique_assets == sum(asset_counts.values()),
        "deduplication_after_reference_verification": True,
        "candidate_reference_ledger_sha256": reference_ledger_sha256,
        "exact_reused_x_rows": reused_x_rows,
        "exact_reused_tensor_hashes": reused_tensor_hashes,
        "exact_reused_cache_aliases": reused_cache_aliases,
        "ambiguous_or_tampered_cache_aliases": ambiguous_cache_aliases,
        "ambiguous_reason_counts": dict(sorted(ambiguous_reason_counts.items())),
    }
    registry_body = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "status": "completed",
        "verdict": verdict,
        "run_identity": dict(run_identity),
        "coordinate_contract": {
            "admission_global_candidate_row_formula": (
                "admission_binding.row_start + admitted_ordinal_in_order_by_local_row"
            ),
            "x_global_row_formula": "cache_part.row_start + transition.local_row",
            "admission_local_row_used_as_candidate_ordinal": False,
            "state_before_feature_row_used": False,
            "x_row_bytes": X_ROW_BYTES,
        },
        "equivalence_contract": {
            "all_candidate_references_verified_before_asset_deduplication": True,
            "normalized_observation_bytes_exact": True,
            "own_hand_card_ids_order_exact": True,
            "feature_float32_bytes_exact": True,
            "all_x_values_finite": True,
            "identical_alias_resolution": "minimum_global_row",
        },
        "conservation": conservation,
        "asset_registry": {
            "path": asset_registry_path.name,
            "sha256": asset_file_sha256,
            "rows": asset_rows,
        },
        "reference_registry": {
            "path": reference_registry_path.name,
            "sha256": reference_file_sha256,
            "rows": reference_rows,
        },
        "input_receipts": {
            "access_mode": "read_only",
            **_candidate_source_input_receipts(
                candidate_source,
                candidate_references=candidate_references,
                reference_ledger_sha256=reference_ledger_sha256,
            ),
            "cache_manifest": {
                "path": str(cache_manifest_path),
                "sha256": cache_manifest_sha256,
            },
            "catalog": {"path": str(catalog_path), "sha256": catalog_sha256},
            "x": {
                "path": str(x_path),
                "sha256": x_sha256,
                "rows": x_rows,
                "bytes_per_row": X_ROW_BYTES,
            },
            "transition_sidecars": [
                {
                    "binding_id": binding.binding_id,
                    "path": str(binding.transition_path),
                    "sha256": binding.transition_sha256,
                    "row_start": binding.row_start,
                    "rows": binding.rows,
                }
                for binding in bindings
            ],
            "source_files": source_receipts,
            "all_hashes_verified": True,
        },
        "checkpoint_receipts": checkpoint_receipts,
        "work_index": {
            "role": "resumable_non_authoritative_scratch",
            "authoritative_payloads": [
                reference_registry_path.name,
                asset_registry_path.name,
            ],
        },
    }
    registry = _seal_payload(registry_body, field="registry_sha256")
    registry_path = output_path / "OBSERVATION_X_REUSE_REGISTRY_V1.json"
    _atomic_write_json(registry_path, registry)
    receipt_body = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "completed",
        "verdict": verdict,
        "run_id": run_identity["run_id"],
        "registry_path": registry_path.name,
        "registry_file_sha256": _sha256_path(registry_path),
        "registry_sha256": registry["registry_sha256"],
        "asset_registry_file_sha256": asset_file_sha256,
        "reference_registry_file_sha256": reference_file_sha256,
        "conservation": conservation,
        "inputs_opened_read_only": True,
        "all_required_input_hashes_verified": True,
    }
    receipt = _seal_payload(receipt_body, field="receipt_sha256")
    receipt_path = output_path / "OBSERVATION_X_REUSE_REGISTRY_V1_RECEIPT.json"
    _atomic_write_json(receipt_path, receipt)

    return {
        "status": "completed",
        "verdict": verdict,
        "run_id": run_identity["run_id"],
        "candidate_references": candidate_references,
        "unique_canonical_observation_assets": unique_assets,
        "asset_counts": asset_counts,
        "candidate_reference_counts": reference_counts,
        "registry_path": str(registry_path),
        "asset_registry_path": str(asset_registry_path),
        "reference_registry_path": str(reference_registry_path),
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt["receipt_sha256"],
    }


def _write_stream_progress(
    output_path: Path,
    *,
    run_identity: Mapping[str, Any],
    stage: str,
    completed: int,
    total: int,
) -> Path:
    payload = _seal_payload({
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "status": "in_progress",
        "run_identity": dict(run_identity),
        "stage": stage,
        "completed": completed,
        "total": total,
        "pending": total - completed,
        "inputs_opened_read_only": True,
    }, field="progress_sha256")
    path = output_path / "OBSERVATION_X_REUSE_REGISTRY_V1_PROGRESS.json"
    _atomic_write_json(path, payload)

    return path


def run_registry_audit(
    *,
    cache_manifest_path: str | Path,
    cache_manifest_sha256: str,
    output_dir: str | Path,
    candidate_manifest_path: str | Path | None = None,
    candidate_manifest_sha256: str | None = None,
    admission_manifest_path: str | Path | None = None,
    semantic_audit_receipt_path: str | Path | None = None,
    semantic_audit_receipt_file_sha256: str | None = None,
    binding_registry_path: str | Path | None = None,
    feature_encoder: FeatureEncoder | None = None,
    feature_encoder_id: str = "scripts.build_value_q_canonical._row_features",
    normalizer: ObservationNormalizer = canonical_actor_feature_observation,
    normalizer_id: str = "deepmew.canonical_actor_feature_observation.v1",
    expected_candidate_references: int | None = None,
    max_new_candidate_references: int | None = None,
    max_new_bindings: int | None = None,
    max_new_assets: int | None = None,
) -> dict[str, Any]:
    """Run the bounded-memory, SQLite-checkpointed registry audit."""
    for value, field in (
        (max_new_candidate_references, "max_new_candidate_references"),
        (max_new_bindings, "max_new_bindings"),
        (max_new_assets, "max_new_assets"),
    ):
        if value is not None:
            _require_int(value, field=field, minimum=1)
    synthetic_requested = any(
        value is not None
        for value in (candidate_manifest_path, candidate_manifest_sha256)
    )
    production_values = (
        admission_manifest_path,
        semantic_audit_receipt_path,
        semantic_audit_receipt_file_sha256,
        binding_registry_path,
    )
    production_requested = any(value is not None for value in production_values)
    if synthetic_requested:
        if candidate_manifest_path is None or candidate_manifest_sha256 is None:
            raise ObservationXReuseAuditError(
                "synthetic candidate-manifest path and SHA must be supplied together"
            )
        if production_requested:
            raise ObservationXReuseAuditError(
                "synthetic candidate manifest cannot be mixed with direct Admission inputs"
            )
        candidate_source: CandidateStream | ProductionAdmission = _load_candidate_stream(
            Path(candidate_manifest_path),
            expected_manifest_sha256=candidate_manifest_sha256,
            expected_candidate_references=expected_candidate_references,
        )
        candidate_source_mode = "synthetic_inline_candidate_manifest_test_only"
    else:
        if any(value is None for value in production_values):
            raise ObservationXReuseAuditError(
                "production requires Admission manifest, semantic receipt and exact file SHA, "
                "and semantic binding registry"
            )
        candidate_source = _load_production_admission(
            admission_manifest_path=Path(admission_manifest_path),
            semantic_audit_receipt_path=Path(semantic_audit_receipt_path),
            semantic_audit_receipt_file_sha256=str(semantic_audit_receipt_file_sha256),
            binding_registry_path=Path(binding_registry_path),
            expected_candidate_references=expected_candidate_references,
        )
        candidate_source_mode = "direct_read_only_immutable_admission_v0"
    (
        cache_manifest,
        catalog_path,
        catalog_sha256,
        x_path,
        x_sha256,
        x_rows,
        bindings,
    ) = _load_cache_bindings(
        Path(cache_manifest_path),
        expected_manifest_sha256=cache_manifest_sha256,
    )
    if not isinstance(feature_encoder_id, str) or not feature_encoder_id:
        raise ObservationXReuseAuditError("feature_encoder_id is required")
    if not isinstance(normalizer_id, str) or not normalizer_id:
        raise ObservationXReuseAuditError("normalizer_id is required")
    encoder = feature_encoder or _default_feature_encoder
    candidate_run_binding: dict[str, Any]
    if isinstance(candidate_source, CandidateStream):
        candidate_run_binding = {
            "candidate_manifest_sha256": candidate_source.manifest_sha256,
            "candidate_file_sha256": candidate_source.candidate_sha256,
            "admission_receipt_sha256": candidate_source.admission_receipt_sha256,
            "semantic_audit_receipt_sha256": (
                candidate_source.semantic_audit_receipt_sha256
            ),
        }
    else:
        candidate_run_binding = {
            "admission_manifest_sha256": candidate_source.admission_manifest_sha256,
            "admission_database_sha256": candidate_source.admission_database_sha256,
            "admission_identity_sha256": candidate_source.admission_identity,
            "admission_receipt_sha256": candidate_source.admission_receipt_sha256,
            "semantic_audit_receipt_file_sha256": (
                candidate_source.semantic_audit_receipt_file_sha256
            ),
            "semantic_audit_receipt_sha256": (
                candidate_source.semantic_audit_receipt_sha256
            ),
            "semantic_binding_registry_file_sha256": (
                candidate_source.binding_registry_file_sha256
            ),
            "semantic_binding_registry_sha256": candidate_source.binding_registry_sha256,
            "admission_loader_implementation_sha256": (
                candidate_source.loader_implementation_sha256
            ),
            "semantic_auditor_implementation_sha256": (
                candidate_source.semantic_audit_receipt["auditor_implementation_sha256"]
            ),
            "semantic_dependency_sha256": (
                candidate_source.semantic_audit_receipt["semantic_dependency_sha256"]
            ),
            "contract_implementation_sha256": (
                candidate_source.semantic_audit_receipt["contract_implementation_sha256"]
            ),
            "canonical_implementation_hash_manifest_file_sha256": (
                candidate_source.implementation_hash_manifest_file_sha256
            ),
            "canonical_implementation_hash_manifest_sha256": (
                candidate_source.implementation_hash_manifest_sha256
            ),
            "causal_aligner_implementation_sha256": (
                candidate_source.causal_aligner_implementation_sha256
            ),
            "legality_validator_implementation_sha256": (
                candidate_source.legality_validator_implementation_sha256
            ),
            "action_schema_implementation_sha256": (
                candidate_source.action_schema_implementation_sha256
            ),
        }
    run_body = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "implementation_sha256": _implementation_sha256(),
        "cache_manifest_sha256": cache_manifest_sha256.lower(),
        "candidate_source_mode": candidate_source_mode,
        **candidate_run_binding,
        "candidate_references": candidate_source.candidate_references,
        "feature_encoder_id": feature_encoder_id,
        "normalizer_id": normalizer_id,
        "feature_dim": FEATURE_DIM,
        "feature_dtype": FEATURE_DTYPE,
        "x_row_bytes": X_ROW_BYTES,
        "admission_candidate_coordinate_contract": (
            "admission_binding.row_start + admitted_ordinal_in_order_by_local_row"
        ),
        "x_coordinate_contract": "cache_part.row_start + transition.local_row",
        "admission_local_row_used_as_candidate_ordinal": False,
        "state_before_feature_row_used": False,
        "aggregation": "sqlite_reference_ledger_then_canonical_asset_deduplication",
    }
    run_identity = {**run_body, "run_id": _stable_sha256(run_body)}
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    connection = _work_connection(
        output_path / "observation_x_reuse_registry_v1.work.sqlite3",
        run_id=run_identity["run_id"],
    )
    try:
        if isinstance(candidate_source, CandidateStream):
            ingested_complete, ingested, inserted = _ingest_candidate_references(
                connection,
                candidate_source,
                normalizer=normalizer,
                max_new_references=max_new_candidate_references,
            )
        else:
            ingested_complete, ingested, inserted = (
                _ingest_production_admission_references(
                    connection,
                    candidate_source,
                    normalizer=normalizer,
                    max_new_references=max_new_candidate_references,
                )
            )
        if not ingested_complete:
            progress_path = _write_stream_progress(
                output_path,
                run_identity=run_identity,
                stage="candidate_reference_ingest",
                completed=ingested,
                total=candidate_source.candidate_references,
            )
            return {
                "status": "in_progress",
                "stage": "candidate_reference_ingest",
                "run_id": run_identity["run_id"],
                "candidate_references_completed": ingested,
                "candidate_references_inserted": inserted,
                "progress_path": str(progress_path),
            }

        parts_dir = output_path / "parts"
        parts_dir.mkdir(parents=True, exist_ok=True)
        completed_bindings = 0
        resumed_bindings = 0
        executed_bindings = 0
        for binding in bindings:
            checkpoint_path = _checkpoint_path(parts_dir, binding)
            if checkpoint_path.is_file():
                _resume_cache_binding_scan(
                    connection,
                    checkpoint_path,
                    binding,
                    run_id=run_identity["run_id"],
                )
                resumed_bindings += 1
            elif max_new_bindings is not None and executed_bindings >= max_new_bindings:
                break
            else:
                checkpoint = _scan_cache_binding_to_store(
                    connection,
                    binding,
                    run_id=run_identity["run_id"],
                )
                _atomic_write_json(checkpoint_path, checkpoint)
                executed_bindings += 1
            completed_bindings += 1
        if completed_bindings != len(bindings):
            progress_path = _write_stream_progress(
                output_path,
                run_identity=run_identity,
                stage="cache_binding_scan",
                completed=completed_bindings,
                total=len(bindings),
            )
            return {
                "status": "in_progress",
                "stage": "cache_binding_scan",
                "run_id": run_identity["run_id"],
                "bindings_completed": completed_bindings,
                "bindings_total": len(bindings),
                "resumed_bindings": resumed_bindings,
                "executed_bindings": executed_bindings,
                "progress_path": str(progress_path),
            }

        resumed_sources, executed_sources = _normalize_matched_sources(
            connection,
            normalizer=normalizer,
        )
        classified_complete, classified_assets, executed_assets = _classify_assets_to_store(
            connection,
            x_path=x_path,
            x_rows=x_rows,
            feature_encoder=encoder,
            max_new_assets=max_new_assets,
        )
        if not classified_complete:
            unique_assets = int(connection.execute("SELECT COUNT(*) FROM assets").fetchone()[0])
            progress_path = _write_stream_progress(
                output_path,
                run_identity=run_identity,
                stage="asset_classification",
                completed=classified_assets,
                total=unique_assets,
            )
            return {
                "status": "in_progress",
                "stage": "asset_classification",
                "run_id": run_identity["run_id"],
                "assets_completed": classified_assets,
                "assets_total": unique_assets,
                "assets_executed": executed_assets,
                "progress_path": str(progress_path),
            }
        result = _finalize_streaming_registry(
            connection,
            output_path=output_path,
            run_identity=run_identity,
            candidate_source=candidate_source,
            cache_manifest_path=Path(cache_manifest_path).resolve(),
            cache_manifest_sha256=cache_manifest_sha256.lower(),
            catalog_path=catalog_path,
            catalog_sha256=catalog_sha256,
            x_path=x_path,
            x_sha256=x_sha256,
            x_rows=x_rows,
            bindings=bindings,
        )
        result.update({
            "candidate_references_inserted": inserted,
            "resumed_bindings": resumed_bindings,
            "executed_bindings": executed_bindings,
            "resumed_source_files": resumed_sources,
            "executed_source_files": executed_sources,
            "executed_assets": executed_assets,
        })
        progress_path = output_path / "OBSERVATION_X_REUSE_REGISTRY_V1_PROGRESS.json"
        progress_path.unlink(missing_ok=True)
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.commit()

        return result
    finally:
        connection.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the hash-bound Observation-to-X reuse registry without mutating inputs.",
    )
    parser.add_argument("--cache-manifest", required=True, type=Path)
    parser.add_argument("--cache-manifest-sha256", required=True)
    parser.add_argument(
        "--admission-manifest",
        type=Path,
        default=DEFAULT_ADMISSION_MANIFEST,
    )
    parser.add_argument(
        "--semantic-audit-receipt",
        type=Path,
        default=DEFAULT_SEMANTIC_AUDIT_DIR / "FULL_CORPUS_SEMANTIC_AUDIT_RECEIPT.json",
    )
    parser.add_argument(
        "--semantic-audit-receipt-file-sha256",
        help="Exact externally frozen SHA-256 of the completed semantic-audit receipt file.",
    )
    parser.add_argument(
        "--binding-registry",
        type=Path,
        default=DEFAULT_SEMANTIC_AUDIT_DIR / "FULL_AUDIT_BINDING_REGISTRY.json",
    )
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        help="Synthetic-test-only inline candidate manifest; never a production dependency.",
    )
    parser.add_argument("--candidate-manifest-sha256")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-candidate-references", type=int, default=7_215_562)
    parser.add_argument("--max-new-candidate-references", type=int)
    parser.add_argument("--max-new-bindings", type=int)
    parser.add_argument("--max-new-assets", type=int)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_registry_audit(
            cache_manifest_path=args.cache_manifest,
            cache_manifest_sha256=args.cache_manifest_sha256,
            output_dir=args.output_dir,
            candidate_manifest_path=args.candidate_manifest,
            candidate_manifest_sha256=args.candidate_manifest_sha256,
            admission_manifest_path=(
                None if args.candidate_manifest is not None else args.admission_manifest
            ),
            semantic_audit_receipt_path=(
                None if args.candidate_manifest is not None else args.semantic_audit_receipt
            ),
            semantic_audit_receipt_file_sha256=(
                None
                if args.candidate_manifest is not None
                else args.semantic_audit_receipt_file_sha256
            ),
            binding_registry_path=(
                None if args.candidate_manifest is not None else args.binding_registry
            ),
            expected_candidate_references=args.expected_candidate_references,
            max_new_candidate_references=args.max_new_candidate_references,
            max_new_bindings=args.max_new_bindings,
            max_new_assets=args.max_new_assets,
        )
    except ObservationXReuseAuditError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)

        return 2
    print(json.dumps(result, sort_keys=True))
    if result["status"] == "in_progress":
        return 3
    if result.get("verdict") == "FAIL_CLOSED_AMBIGUOUS":
        return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
