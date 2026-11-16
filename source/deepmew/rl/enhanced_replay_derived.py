"""Content-addressed training-ready representation of admitted replay rows."""

from __future__ import annotations

import hashlib
import gzip
import io
import json
import os
import sqlite3
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch

from deepmew.rl.enhanced_trainer import ReplayTrainerExample


DERIVED_SCHEMA_VERSION = "deepmew.e1_replay_derived.v1"
REFERENCE_ROW_SCHEMA_VERSION = "deepmew.derived_replay_reference_row.v1"
REFERENCE_SHARD_RECEIPT_SCHEMA_VERSION = "deepmew.derived_replay_reference_shard_receipt.v1"
REFERENCE_MANIFEST_SCHEMA_VERSION = "deepmew.derived_replay_reference_manifest.v1"


class DerivedReplayHardStop(RuntimeError):
    """Derived replay identity, content, or row coverage is inconsistent."""


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
    _atomic_bytes(path, _canonical_bytes(payload) + b"\n")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


@dataclass(frozen=True)
class DerivedReplayIdentity:
    admission_receipt_sha256: str
    cache_manifest_sha256: str
    split_manifest_sha256: str
    r7_checkpoint_sha256: str
    v0_checkpoint_sha256: str
    v0_calibrator_sha256: str
    action_schema_identity: str
    observation_schema_identity: str
    encoder_code_sha256: str
    forced_rule: str = "legal_action_count == 1; actor weights 0.0/1.0"

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_bytes(asdict(self))).hexdigest()


@dataclass(frozen=True)
class DerivedReplayReferenceIdentity:
    """Frozen authorities for a tensor-free eligible replay reference view."""

    semantic_audit_receipt_sha256: str
    semantic_audit_identity: str
    admission_identity: str
    semantic_game_stats_sha256: str
    cache_manifest_sha256: str
    cache_x_sha256: str
    game_split_assignment_sha256: str
    canonical_contract_version: str
    causal_aligner_implementation_sha256: str
    legality_validator_implementation_sha256: str
    action_schema_implementation_sha256: str
    feature_converter_set_sha256: str
    regeneration_source_converter_sha256: str
    regeneration_encoder_code_sha256: str
    feature_dim: int

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_bytes(asdict(self))).hexdigest()


class DerivedReplayReferenceBuilder:
    """Commit deterministic reference-only shards without copying observations."""

    REQUIRED_ROW_FIELDS = frozenset({
        "schema_version",
        "stable_row_id",
        "semantic_audit_global_candidate_row",
        "source_game_id",
        "trajectory_id",
        "actor_seat",
        "pre_action_observation_id",
        "source_candidate_cache_coordinate",
        "observation_storage_kind",
        "observation_storage_coordinate",
        "observation_storage_shard",
        "observation_alias_candidate_count",
        "pre_action_observation_content_sha256",
        "legal_action_set_id",
        "complete_chosen_action_id",
        "action_family",
        "action_template",
        "complete_legal_action_count",
        "source_transition_decision_id",
        "source_transition_sha256",
        "transition_reference_id",
        "resulting_transition_id",
        "resulting_transition_proof",
        "terminal_outcome",
        "validation_outcome",
        "forced_state",
        "source_family",
        "source_schema_version",
        "canonical_transition_schema_version",
        "continuity",
    })

    def __init__(
        self,
        output_dir: str | Path,
        identity: DerivedReplayReferenceIdentity,
        *,
        shard_rows: int = 8192,
    ) -> None:
        if isinstance(shard_rows, bool) or not isinstance(shard_rows, int) or shard_rows < 1:
            raise DerivedReplayHardStop("reference shard_rows must be a positive integer")
        self.output_dir = Path(output_dir)
        self.shards_dir = self.output_dir / "shards"
        self.receipts_dir = self.output_dir / "receipts"
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.receipts_dir.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.shard_rows = shard_rows
        self.receipts = self._load_receipts()

    @property
    def committed_rows(self) -> int:
        return sum(int(receipt["row_count"]) for receipt in self.receipts)

    @property
    def final_cache_coordinate(self) -> int | None:
        if not self.receipts:
            return None

        return int(self.receipts[-1]["source_candidate_cache_coordinate_range"]["maximum"])

    def _load_receipts(self) -> list[dict[str, Any]]:
        paths = sorted(self.receipts_dir.glob("shard_*.receipt.json"))
        receipts = []
        for expected_index, path in enumerate(paths):
            receipt = json.loads(path.read_text(encoding="ascii"))
            provided = receipt.pop("receipt_sha256", None)
            actual = hashlib.sha256(_canonical_bytes(receipt)).hexdigest()
            if provided != actual:
                raise DerivedReplayHardStop(f"reference shard receipt hash mismatch: {path}")
            receipt["receipt_sha256"] = provided
            if (
                receipt.get("schema_version") != REFERENCE_SHARD_RECEIPT_SCHEMA_VERSION
                or int(receipt.get("shard_index", -1)) != expected_index
                or receipt.get("identity_sha256") != self.identity.sha256
                or int(receipt.get("configured_shard_rows", -1)) != self.shard_rows
            ):
                raise DerivedReplayHardStop("reference shard receipt identity or order drifted")
            shard = self.output_dir / str(receipt["shard_path"])
            if not shard.is_file() or shard.stat().st_size != int(receipt["shard_bytes"]):
                raise DerivedReplayHardStop("committed reference shard is absent or has a size mismatch")
            receipts.append(receipt)

        return receipts

    def commit(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        regenerated_observations: Sequence[tuple[str, np.ndarray]] = (),
    ) -> dict[str, Any]:
        if not rows or len(rows) > self.shard_rows:
            raise DerivedReplayHardStop("reference commit must contain one bounded nonempty shard")
        normalized = [dict(row) for row in rows]
        for row in normalized:
            if set(row) != self.REQUIRED_ROW_FIELDS:
                missing = sorted(self.REQUIRED_ROW_FIELDS - set(row))
                extra = sorted(set(row) - self.REQUIRED_ROW_FIELDS)
                raise DerivedReplayHardStop(
                    f"reference row field drift; missing={missing[:4]} extra={extra[:4]}"
                )
            if row["schema_version"] != REFERENCE_ROW_SCHEMA_VERSION:
                raise DerivedReplayHardStop("reference row schema drifted")
        stable_ids = [str(row["stable_row_id"]) for row in normalized]
        coordinates = [int(row["source_candidate_cache_coordinate"]) for row in normalized]
        if len(set(stable_ids)) != len(stable_ids):
            raise DerivedReplayHardStop("reference shard contains duplicate stable row IDs")
        if coordinates != sorted(coordinates) or len(set(coordinates)) != len(coordinates):
            raise DerivedReplayHardStop("reference shard cache coordinates are not unique and ordered")
        prior = self.final_cache_coordinate
        if prior is not None and coordinates[0] <= prior:
            raise DerivedReplayHardStop("reference shard overlaps an already committed coordinate")

        shard_index = len(self.receipts)
        shard_name = f"shard_{shard_index:05d}.jsonl.gz"
        shard_path = self.shards_dir / shard_name
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "wb",
                dir=self.shards_dir,
                prefix=f".{shard_name}.",
                suffix=".tmp",
                delete=False,
            ) as raw:
                temporary = Path(raw.name)
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    fileobj=raw,
                    compresslevel=6,
                    mtime=0,
                ) as compressed:
                    for row in normalized:
                        compressed.write(_canonical_bytes(row) + b"\n")
                raw.flush()
                os.fsync(raw.fileno())
            os.replace(temporary, shard_path)
        except Exception:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise

        regenerated_entries = []
        regenerated_path: Path | None = None
        if regenerated_observations:
            observation_ids = [str(item[0]) for item in regenerated_observations]
            if len(set(observation_ids)) != len(observation_ids):
                raise DerivedReplayHardStop("regenerated observation IDs are duplicated in one shard")
            vectors = np.stack(
                [np.asarray(item[1], dtype=np.float32) for item in regenerated_observations],
                axis=0,
            )
            if (
                vectors.ndim != 2
                or vectors.shape[1] != self.identity.feature_dim
                or not np.isfinite(vectors).all()
            ):
                raise DerivedReplayHardStop("regenerated observations are not finite feature vectors")
            regenerated_path = self.shards_dir / f"shard_{shard_index:05d}.regenerated_X.f32.bin"
            _atomic_bytes(regenerated_path, vectors.astype(np.float32, copy=False).tobytes(order="C"))
            regenerated_entries = [
                {
                    "pre_action_observation_id": observation_id,
                    "coordinate": index,
                }
                for index, observation_id in enumerate(observation_ids)
            ]
        for row in normalized:
            storage = str(row["observation_storage_kind"])
            storage_coordinate = int(row["observation_storage_coordinate"])
            storage_shard = row["observation_storage_shard"]
            if storage == "cache":
                if storage_coordinate < 0 or storage_shard is not None:
                    raise DerivedReplayHardStop("cache observation reference is invalid")
            elif storage == "regenerated":
                if (
                    storage_coordinate < 0
                    or isinstance(storage_shard, bool)
                    or not isinstance(storage_shard, int)
                    or storage_shard < 0
                    or storage_shard > shard_index
                ):
                    raise DerivedReplayHardStop("regenerated observation reference is invalid")
            else:
                raise DerivedReplayHardStop("unknown observation storage kind")

        outcomes = Counter(str(row["validation_outcome"]) for row in normalized)
        sources = Counter(str(row["source_family"]) for row in normalized)
        continuity = Counter(str(row["continuity"]) for row in normalized)
        transition_proofs = Counter(
            str(row["resulting_transition_proof"])
            for row in normalized
        )
        action_families = Counter(
            f"{row['action_family']}|{row['action_template']}"
            for row in normalized
        )
        legal_cardinalities = Counter(
            str(int(row["complete_legal_action_count"]))
            for row in normalized
        )
        game_counts: dict[str, Counter[str]] = {}
        for row in normalized:
            game_id = str(row["source_game_id"])
            game_counts.setdefault(game_id, Counter())[str(row["validation_outcome"])] += 1
        forced = sum(bool(row["forced_state"]) for row in normalized)
        reused_coordinates = Counter(
            int(row["observation_storage_coordinate"])
            for row in normalized
            if row["observation_storage_kind"] == "cache"
        )
        receipt: dict[str, Any] = {
            "schema_version": REFERENCE_SHARD_RECEIPT_SCHEMA_VERSION,
            "shard_index": shard_index,
            "configured_shard_rows": self.shard_rows,
            "identity_sha256": self.identity.sha256,
            "semantic_audit_receipt_sha256": self.identity.semantic_audit_receipt_sha256,
            "cache_manifest_sha256": self.identity.cache_manifest_sha256,
            "row_schema_version": REFERENCE_ROW_SCHEMA_VERSION,
            "shard_path": str(shard_path.relative_to(self.output_dir)).replace("\\", "/"),
            "shard_bytes": shard_path.stat().st_size,
            "shard_sha256": _sha256_file(shard_path),
            "row_count": len(normalized),
            "first_stable_row_id": stable_ids[0],
            "final_stable_row_id": stable_ids[-1],
            "stable_row_ids_sha256": hashlib.sha256(_canonical_bytes(stable_ids)).hexdigest(),
            "actor_rows": len(normalized) - forced,
            "forced_rows": forced,
            "validation_outcomes": dict(sorted(outcomes.items())),
            "source_distribution": dict(sorted(sources.items())),
            "continuity_distribution": dict(sorted(continuity.items())),
            "resulting_transition_proof_distribution": dict(
                sorted(transition_proofs.items())
            ),
            "action_family_distribution": dict(sorted(action_families.items())),
            "complete_legal_action_count_distribution": dict(
                sorted(legal_cardinalities.items(), key=lambda item: int(item[0]))
            ),
            "game_semantic_counts": {
                game_id: dict(sorted(counts.items()))
                for game_id, counts in sorted(game_counts.items())
            },
            "source_candidate_cache_coordinate_range": {
                "minimum": coordinates[0],
                "maximum": coordinates[-1],
            },
            "source_candidate_cache_coordinates_sha256": hashlib.sha256(
                _canonical_bytes(coordinates)
            ).hexdigest(),
            "observation_storage": {
                "cache_reuse_rows": sum(
                    row["observation_storage_kind"] == "cache"
                    for row in normalized
                ),
                "cache_reuse_same_candidate_rows": sum(
                    row["observation_storage_kind"] == "cache"
                    and int(row["observation_storage_coordinate"])
                    == int(row["source_candidate_cache_coordinate"])
                    for row in normalized
                ),
                "cache_reuse_alias_rows": sum(
                    row["observation_storage_kind"] == "cache"
                    and int(row["observation_storage_coordinate"])
                    != int(row["source_candidate_cache_coordinate"])
                    for row in normalized
                ),
                "duplicate_cache_coordinate_reference_rows": sum(
                    count - 1 for count in reused_coordinates.values() if count > 1
                ),
                "rows_with_multiple_exact_alias_candidates": sum(
                    int(row["observation_alias_candidate_count"]) > 1
                    for row in normalized
                ),
                "regenerated_reference_rows": sum(
                    row["observation_storage_kind"] == "regenerated"
                    for row in normalized
                ),
                "new_regenerated_assets": len(regenerated_entries),
                "regenerated_assets": regenerated_entries,
                "regenerated_path": (
                    str(regenerated_path.relative_to(self.output_dir)).replace("\\", "/")
                    if regenerated_path is not None
                    else None
                ),
                "regenerated_bytes": regenerated_path.stat().st_size if regenerated_path is not None else 0,
                "regenerated_sha256": _sha256_file(regenerated_path) if regenerated_path is not None else None,
            },
        }
        receipt["receipt_sha256"] = hashlib.sha256(_canonical_bytes(receipt)).hexdigest()
        receipt_path = self.receipts_dir / f"shard_{shard_index:05d}.receipt.json"
        _atomic_json(receipt_path, receipt)
        self.receipts.append(receipt)
        self._write_progress()

        return receipt

    def _write_progress(self) -> None:
        payload = {
            "schema_version": "deepmew.derived_replay_reference_progress.v1",
            "identity_sha256": self.identity.sha256,
            "configured_shard_rows": self.shard_rows,
            "committed_shards": len(self.receipts),
            "committed_rows": self.committed_rows,
            "final_cache_coordinate": self.final_cache_coordinate,
            "receipt_sha256s": [receipt["receipt_sha256"] for receipt in self.receipts],
            "resume_policy": "read_small_receipts_and_continue_after_last_committed_coordinate_without_rereading_shards",
        }
        payload["progress_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
        _atomic_json(self.output_dir / "BUILD_PROGRESS.json", payload)

    def finalize(
        self,
        manifest_path: str | Path,
        *,
        expected_rows: int,
        aggregates: Mapping[str, Any],
        cache_manifest_path: str | Path,
        cache_mapping_report_path: str | Path,
        split_manifest_path: str | Path,
        split_assignments_path: str | Path,
    ) -> dict[str, Any]:
        if self.committed_rows != int(expected_rows):
            raise DerivedReplayHardStop(
                f"reference replay coverage mismatch: {self.committed_rows}/{expected_rows}"
            )
        ordered_receipts = [receipt["receipt_sha256"] for receipt in self.receipts]
        manifest: dict[str, Any] = {
            "schema_version": REFERENCE_MANIFEST_SCHEMA_VERSION,
            "status": "COMPLETE_REFERENCE_ONLY_NO_OBSERVATION_DUPLICATION",
            "identity": asdict(self.identity),
            "identity_sha256": self.identity.sha256,
            "row_schema_version": REFERENCE_ROW_SCHEMA_VERSION,
            "shard_receipt_schema_version": REFERENCE_SHARD_RECEIPT_SCHEMA_VERSION,
            "row_count": self.committed_rows,
            "shard_count": len(self.receipts),
            "configured_shard_rows": self.shard_rows,
            "ordered_shard_receipt_sha256s": ordered_receipts,
            "ordered_shard_identity_sha256": hashlib.sha256(
                _canonical_bytes(ordered_receipts)
            ).hexdigest(),
            "aggregates": dict(aggregates),
            "cache_manifest_path": str(Path(cache_manifest_path).resolve()),
            "cache_mapping_report_path": str(Path(cache_mapping_report_path).resolve()),
            "game_split_manifest_path": str(Path(split_manifest_path).resolve()),
            "game_split_assignments_path": str(Path(split_assignments_path).resolve()),
            "shards_root": str(self.shards_dir.resolve()),
            "receipts_root": str(self.receipts_dir.resolve()),
            "observation_storage": {
                "mode": "immutable_cache_coordinate_reference",
                "tensor_rows_copied": 0,
                "x_artifact_sha256": self.identity.cache_x_sha256,
            },
            "overlay_contract": {
                "core_rows_exclude_experiment_weights": True,
                "allowed_lightweight_overlays": [
                    "actor_loss_weight",
                    "train_validation_holdout_membership",
                    "source_weight",
                    "curriculum_bucket",
                    "future_exploiter_or_failure_slice_membership",
                ],
                "forced_actor_loss_rule": "forced_state=true -> actor_loss_weight=0.0; otherwise 1.0",
            },
            "resume_policy": "receipt_cursor_only_no_completed_shard_reread_no_admission_reread",
        }
        manifest["manifest_sha256"] = hashlib.sha256(_canonical_bytes(manifest)).hexdigest()
        _atomic_json(Path(manifest_path), manifest)

        return manifest


class DerivedReplayReferenceIndex:
    """Consume Derived V1 references while reading X directly from the frozen cache."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path).resolve(strict=True)
        manifest = json.loads(self.manifest_path.read_text(encoding="ascii"))
        provided = manifest.pop("manifest_sha256", None)
        if hashlib.sha256(_canonical_bytes(manifest)).hexdigest() != provided:
            raise DerivedReplayHardStop("reference replay manifest hash mismatch")
        manifest["manifest_sha256"] = provided
        if (
            manifest.get("schema_version") != REFERENCE_MANIFEST_SCHEMA_VERSION
            or manifest.get("status") != "COMPLETE_REFERENCE_ONLY_NO_OBSERVATION_DUPLICATION"
        ):
            raise DerivedReplayHardStop("reference replay manifest is not complete")
        cache_manifest_path = Path(manifest["cache_manifest_path"]).resolve(strict=True)
        identity = manifest["identity"]
        if _sha256_file(cache_manifest_path) != identity["cache_manifest_sha256"]:
            raise DerivedReplayHardStop("reference replay cache manifest hash mismatch")
        cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        x_receipt = cache_manifest.get("output_files", {}).get("X.f32.bin", {})
        if x_receipt.get("sha256") != identity["cache_x_sha256"]:
            raise DerivedReplayHardStop("reference replay X identity differs from cache receipt")
        x_path = cache_manifest_path.parent / "X.f32.bin"
        rows = int(cache_manifest["rows"])
        feature_dim = int(cache_manifest["feature_dim"])
        expected_bytes = rows * feature_dim * np.dtype(np.float32).itemsize
        if not x_path.is_file() or x_path.stat().st_size != expected_bytes:
            raise DerivedReplayHardStop("reference replay X byte extent mismatch")
        self.manifest = manifest
        self.cache_manifest = cache_manifest
        self.x_path = x_path
        self.rows = rows
        self.feature_dim = feature_dim
        assignments_path = Path(manifest["game_split_assignments_path"]).resolve(strict=True)
        with gzip.open(assignments_path, "rt", encoding="ascii") as handle:
            assignments = json.load(handle)
        if not isinstance(assignments, list):
            raise DerivedReplayHardStop("reference replay split assignments are invalid")
        self.game_splits = {
            str(item["canonical_game_id"]): str(item["split"])
            for item in assignments
        }
        if len(self.game_splits) != len(assignments):
            raise DerivedReplayHardStop("reference replay split assignments are duplicated")

    def iter_rows(self, *, split: str | None = None) -> Iterator[dict[str, Any]]:
        count = 0
        receipts_root = Path(self.manifest["receipts_root"])
        for expected_index, expected_receipt_sha in enumerate(
            self.manifest["ordered_shard_receipt_sha256s"]
        ):
            receipt_path = receipts_root / f"shard_{expected_index:05d}.receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="ascii"))
            provided = receipt.pop("receipt_sha256", None)
            if provided != expected_receipt_sha or hashlib.sha256(
                _canonical_bytes(receipt)
            ).hexdigest() != provided:
                raise DerivedReplayHardStop("reference replay shard receipt mismatch")
            shard = self.manifest_path.parent / "deepmew_derived_replay_v1" / receipt["shard_path"]
            if not shard.is_file():
                shard = Path(self.manifest["shards_root"]) / Path(receipt["shard_path"]).name
            if _sha256_file(shard) != receipt["shard_sha256"]:
                raise DerivedReplayHardStop("reference replay shard hash mismatch")
            shard_count = 0
            with gzip.open(shard, "rt", encoding="ascii") as handle:
                for line in handle:
                    row = json.loads(line)
                    if row.get("schema_version") != REFERENCE_ROW_SCHEMA_VERSION:
                        raise DerivedReplayHardStop("reference replay row schema mismatch")
                    shard_count += 1
                    count += 1
                    row_split = self.game_splits.get(str(row.get("source_game_id")))
                    if row_split is None:
                        raise DerivedReplayHardStop("reference replay row has no game-level split")
                    if split is None or row_split == split:
                        yield row
            if shard_count != int(receipt["row_count"]):
                raise DerivedReplayHardStop("reference replay shard row count mismatch")
        if count != int(self.manifest["row_count"]):
            raise DerivedReplayHardStop("reference replay manifest row count mismatch")

    def load_observations(self, cache_coordinates: Sequence[int]) -> np.ndarray:
        coordinates = np.asarray(cache_coordinates, dtype=np.int64)
        if coordinates.ndim != 1:
            raise DerivedReplayHardStop("reference cache coordinates must be one-dimensional")
        if bool(((coordinates < 0) | (coordinates >= self.rows)).any()):
            raise DerivedReplayHardStop("reference cache coordinate is out of bounds")
        tensor = np.memmap(
            self.x_path,
            dtype=np.float32,
            mode="r",
            shape=(self.rows, self.feature_dim),
        )

        return np.asarray(tensor[coordinates])

    def load_reference_observations(
        self,
        references: Sequence[Mapping[str, Any]],
    ) -> np.ndarray:
        result = np.empty((len(references), self.feature_dim), dtype=np.float32)
        cache_positions = []
        cache_coordinates = []
        regenerated: dict[int, list[tuple[int, int]]] = {}
        for position, reference in enumerate(references):
            storage = reference.get("observation_storage_kind")
            coordinate = int(reference.get("observation_storage_coordinate", -1))
            if storage == "cache":
                cache_positions.append(position)
                cache_coordinates.append(coordinate)
            elif storage == "regenerated":
                shard = reference.get("observation_storage_shard")
                if isinstance(shard, bool) or not isinstance(shard, int) or shard < 0:
                    raise DerivedReplayHardStop("regenerated reference shard is invalid")
                regenerated.setdefault(shard, []).append((position, coordinate))
            else:
                raise DerivedReplayHardStop("reference observation storage kind is invalid")
        if cache_positions:
            result[np.asarray(cache_positions)] = self.load_observations(cache_coordinates)
        receipts_root = Path(self.manifest["receipts_root"])
        for shard_index, requests in regenerated.items():
            receipt_path = receipts_root / f"shard_{shard_index:05d}.receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="ascii"))
            storage = receipt.get("observation_storage", {})
            relative = storage.get("regenerated_path")
            rows = int(storage.get("new_regenerated_assets", -1))
            path = self.manifest_path.parent / "deepmew_derived_replay_v1" / str(relative)
            if not path.is_file():
                path = Path(self.manifest["shards_root"]).parent / str(relative)
            expected_bytes = rows * self.feature_dim * np.dtype(np.float32).itemsize
            if (
                rows < 0
                or not path.is_file()
                or path.stat().st_size != expected_bytes
                or _sha256_file(path) != storage.get("regenerated_sha256")
            ):
                raise DerivedReplayHardStop("regenerated observation shard identity mismatch")
            values = np.memmap(
                path,
                dtype=np.float32,
                mode="r",
                shape=(rows, self.feature_dim),
            )
            for position, coordinate in requests:
                if coordinate < 0 or coordinate >= rows:
                    raise DerivedReplayHardStop("regenerated observation coordinate is out of bounds")
                result[position] = values[coordinate]

        return result


def _connect(path: Path, *, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    else:
        connection = sqlite3.connect(path)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
    if not readonly:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS examples (
                global_row INTEGER PRIMARY KEY,
                row_id TEXT NOT NULL UNIQUE,
                option_count INTEGER NOT NULL,
                input_dim INTEGER NOT NULL,
                option_features BLOB NOT NULL,
                selection_json TEXT NOT NULL,
                min_count INTEGER NOT NULL,
                max_count INTEGER NOT NULL,
                v0_probability REAL NOT NULL,
                legal_action_count INTEGER NOT NULL,
                source_id TEXT NOT NULL,
                deck_identity TEXT,
                source_observation_sha256 TEXT NOT NULL
            );
            """
        )

    return connection


class DerivedReplayBuilder:
    def __init__(self, database_path: str | Path, identity: DerivedReplayIdentity) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.connection = _connect(self.database_path)
        expected = {
            "schema_version": DERIVED_SCHEMA_VERSION,
            "identity_sha256": identity.sha256,
            "identity": json.dumps(asdict(identity), sort_keys=True),
        }
        existing = dict(self.connection.execute("SELECT key,value FROM meta"))
        if existing and existing != expected:
            raise DerivedReplayHardStop("derived replay database identity drifted")
        if not existing:
            self.connection.executemany("INSERT INTO meta VALUES (?,?)", expected.items())
            self.connection.commit()

    def add(
        self,
        global_rows: Sequence[int],
        records: Sequence[Any],
        examples: Sequence[ReplayTrainerExample],
    ) -> int:
        if not (len(global_rows) == len(records) == len(examples)):
            raise DerivedReplayHardStop("derived replay batch is misaligned")
        rows = []
        for global_row, record, example in zip(global_rows, records, examples, strict=True):
            example.validate(input_dim=int(example.option_features.shape[1]))
            features = example.option_features.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
            rows.append((
                int(global_row),
                example.row_id,
                int(features.shape[0]),
                int(features.shape[1]),
                sqlite3.Binary(features.tobytes(order="C")),
                json.dumps(list(example.selection), separators=(",", ":")),
                example.min_count,
                example.max_count,
                example.v0_probability,
                int(example.legal_action_count or len(record.legal_set)),
                record.source_id,
                record.deck_identity,
                record.source_observation_sha256,
            ))
        self.connection.executemany(
            "INSERT OR REPLACE INTO examples VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self.connection.commit()

        return len(rows)

    def missing_global_rows(self, global_rows: Sequence[int]) -> tuple[int, ...]:
        """Return rows not yet materialized, preserving the requested order."""

        requested = tuple(int(global_row) for global_row in global_rows)
        if not requested:
            return ()
        placeholders = ",".join("?" for _ in requested)
        existing = {
            int(row[0])
            for row in self.connection.execute(
                f"SELECT global_row FROM examples WHERE global_row IN ({placeholders})",
                requested,
            )
        }

        return tuple(global_row for global_row in requested if global_row not in existing)

    def finalize(self, manifest_path: str | Path, *, expected_rows: int) -> dict[str, Any]:
        rows = int(self.connection.execute("SELECT COUNT(*) FROM examples").fetchone()[0])
        if rows != expected_rows:
            raise DerivedReplayHardStop(f"derived replay coverage mismatch: {rows}/{expected_rows}")
        forced = int(self.connection.execute(
            "SELECT COUNT(*) FROM examples WHERE legal_action_count=1"
        ).fetchone()[0])
        self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.connection.commit()
        receipt = {
            "schema_version": DERIVED_SCHEMA_VERSION,
            "status": "complete",
            "identity": asdict(self.identity),
            "identity_sha256": self.identity.sha256,
            "rows": rows,
            "forced_rows": forced,
            "nonforced_rows": rows - forced,
            "database_path": str(self.database_path.resolve()),
            "database_bytes": self.database_path.stat().st_size,
            "database_sha256": _sha256_file(self.database_path),
        }
        receipt["receipt_sha256"] = hashlib.sha256(_canonical_bytes(receipt)).hexdigest()
        target = Path(manifest_path)
        with tempfile.NamedTemporaryFile("wb", dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(_canonical_bytes(receipt) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)

        return receipt


class DerivedReplayIndex:
    def __init__(self, manifest_path: str | Path) -> None:
        receipt = json.loads(Path(manifest_path).read_text(encoding="ascii"))
        provided = receipt.pop("receipt_sha256")
        if hashlib.sha256(_canonical_bytes(receipt)).hexdigest() != provided:
            raise DerivedReplayHardStop("derived replay receipt hash mismatch")
        database = Path(receipt["database_path"]).resolve(strict=True)
        if _sha256_file(database) != receipt["database_sha256"]:
            raise DerivedReplayHardStop("derived replay database hash mismatch")
        receipt["receipt_sha256"] = provided
        self.receipt = receipt
        self.connection = _connect(database, readonly=True)

    def global_rows(self) -> tuple[int, ...]:
        """Return the exact deterministic row order stored by the derived index."""

        return tuple(
            int(row[0])
            for row in self.connection.execute("SELECT global_row FROM examples ORDER BY global_row")
        )

    def load(self, global_rows: Iterable[int]) -> tuple[ReplayTrainerExample, ...]:
        examples = []
        for global_row in global_rows:
            row = self.connection.execute(
                "SELECT row_id,option_count,input_dim,option_features,selection_json,min_count,max_count,v0_probability,legal_action_count "
                "FROM examples WHERE global_row=?",
                (int(global_row),),
            ).fetchone()
            if row is None:
                raise DerivedReplayHardStop("derived replay row is missing")
            row_id, option_count, input_dim, payload, selection, minimum, maximum, probability, legal_count = row
            array = np.frombuffer(payload, dtype=np.float32).reshape(option_count, input_dim).copy()
            examples.append(ReplayTrainerExample(
                row_id=row_id,
                option_features=torch.from_numpy(array),
                selection=tuple(json.loads(selection)),
                min_count=minimum,
                max_count=maximum,
                v0_probability=probability,
                legal_action_count=legal_count,
            ))

        return tuple(examples)


__all__ = [
    "DerivedReplayBuilder",
    "DerivedReplayHardStop",
    "DerivedReplayIdentity",
    "DerivedReplayIndex",
    "DerivedReplayReferenceBuilder",
    "DerivedReplayReferenceIdentity",
    "DerivedReplayReferenceIndex",
    "REFERENCE_MANIFEST_SCHEMA_VERSION",
    "REFERENCE_ROW_SCHEMA_VERSION",
    "REFERENCE_SHARD_RECEIPT_SCHEMA_VERSION",
]
