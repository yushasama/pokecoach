"""Content-addressed admission index for immutable replay source observations."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sqlite3
import tempfile
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from deepmew.replay.canonical_actor_dataset import TRANSITION_SCHEMA_VERSION
from deepmew.rl.enhanced_replay import EnhancedReplayError, _open_source, _sha256_path


ADMISSION_SCHEMA_VERSION = "deepmew.e1_replay_admission.v2"


class ReplayAdmissionHardStop(RuntimeError):
    """An admission input, indexed row, or content binding drifted."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class AdmissionBinding:
    binding_id: str
    transition_path: str
    transition_sha256: str
    row_start: int
    row_count: int
    source_hashes: Mapping[str, str]

    def validate(self) -> None:
        if not self.binding_id or self.row_start < 0 or self.row_count < 1:
            raise ReplayAdmissionHardStop("admission binding identity or row range is invalid")
        if len(self.transition_sha256) != 64 or not self.source_hashes:
            raise ReplayAdmissionHardStop("admission binding hashes are incomplete")


def bindings_from_materializer(materializer: Any) -> tuple[AdmissionBinding, ...]:
    values = []
    for binding in materializer.bindings:
        source_hashes = {
            str(Path(path).resolve()): str(digest).lower()
            for path, digest in binding.unit_manifest["source_sha256"].items()
        }
        values.append(AdmissionBinding(
            binding_id=f"{binding.source_id}:{binding.unit_id}:{binding.part}",
            transition_path=str(binding.transition_path.resolve()),
            transition_sha256=str(binding.input_receipt["transitions_sha256"]).lower(),
            row_start=int(binding.row_start),
            row_count=int(binding.row_end - binding.row_start),
            source_hashes=source_hashes,
        ))

    return tuple(values)


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS requests (
            source_sha256 TEXT NOT NULL,
            source_path TEXT NOT NULL,
            source_line INTEGER NOT NULL,
            content_sha256 TEXT NOT NULL,
            PRIMARY KEY (source_sha256, source_line, content_sha256)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS observations (
            source_sha256 TEXT NOT NULL,
            source_line INTEGER NOT NULL,
            content_sha256 TEXT NOT NULL,
            compressed_json BLOB NOT NULL,
            PRIMARY KEY (source_sha256, source_line, content_sha256),
            FOREIGN KEY (source_sha256, source_line, content_sha256)
                REFERENCES requests(source_sha256, source_line, content_sha256)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS transitions (
            binding_id TEXT NOT NULL,
            local_row INTEGER NOT NULL,
            compressed_json BLOB NOT NULL,
            PRIMARY KEY (binding_id, local_row)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS completed_inputs (
            kind TEXT NOT NULL,
            identity TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            rows INTEGER NOT NULL,
            elapsed_seconds REAL NOT NULL,
            PRIMARY KEY (kind, identity)
        ) WITHOUT ROWID;
        """
    )

    return connection


class ReplayAdmissionBuilder:
    """Build a resumable exact observation index from accepted transitions."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        corpus_identity: Mapping[str, Any],
        bindings: Sequence[AdmissionBinding],
        split_idx: Sequence[Any],
        is_target_eligible: Sequence[Any],
        train_index: int,
    ) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.bindings = tuple(bindings)
        if not self.bindings:
            raise ReplayAdmissionHardStop("admission requires bindings")
        for binding in self.bindings:
            binding.validate()
        self.split_idx = split_idx
        self.is_target_eligible = is_target_eligible
        self.train_index = train_index
        self.corpus_identity = dict(corpus_identity)
        self.binding_set_sha256 = _sha256_json([asdict(value) for value in self.bindings])
        self.admission_identity_sha256 = _sha256_json({
            "schema_version": ADMISSION_SCHEMA_VERSION,
            "corpus_identity": self.corpus_identity,
            "binding_set_sha256": self.binding_set_sha256,
        })
        self.connection = _connect(self.database_path)
        self._bind_identity()

    def _bind_identity(self) -> None:
        existing = dict(self.connection.execute("SELECT key, value FROM meta"))
        expected = {
            "schema_version": ADMISSION_SCHEMA_VERSION,
            "admission_identity_sha256": self.admission_identity_sha256,
            "binding_set_sha256": self.binding_set_sha256,
            "corpus_identity": json.dumps(self.corpus_identity, sort_keys=True),
        }
        if existing and existing != expected:
            raise ReplayAdmissionHardStop("admission database identity drifted")
        if not existing:
            self.connection.executemany("INSERT INTO meta(key,value) VALUES (?,?)", expected.items())
            self.connection.commit()

    def collect_requests(self) -> dict[str, Any]:
        completed = {
            identity for identity, in self.connection.execute(
                "SELECT identity FROM completed_inputs WHERE kind='transition'"
            )
        }
        added = 0
        started = time.perf_counter()
        for binding in self.bindings:
            if binding.binding_id in completed:
                continue
            path = Path(binding.transition_path)
            if _sha256_path(path) != binding.transition_sha256:
                raise ReplayAdmissionHardStop(f"transition hash mismatch: {path}")
            rows = 0
            request_batch = []
            transition_batch = []
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for local_row, line in enumerate(handle):
                    if local_row >= binding.row_count:
                        raise ReplayAdmissionHardStop("transition sidecar exceeds admitted row count")
                    global_row = binding.row_start + local_row
                    if self.split_idx[global_row] != self.train_index or not self.is_target_eligible[global_row]:
                        continue
                    try:
                        transition = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ReplayAdmissionHardStop(f"invalid transition JSON: {path}:{local_row + 1}") from exc
                    if transition.get("schema_version") != TRANSITION_SCHEMA_VERSION or transition.get("row") != local_row:
                        raise ReplayAdmissionHardStop("transition schema or row identity mismatch during admission")
                    reference = (transition.get("state_before") or {}).get("source_observation")
                    if not isinstance(reference, dict):
                        raise ReplayAdmissionHardStop("transition source-observation reference is missing")
                    source_path = str(Path(reference["source_path"]).resolve())
                    expected_source = binding.source_hashes.get(source_path)
                    if expected_source is None or transition.get("source_hash", "").lower() != expected_source:
                        raise ReplayAdmissionHardStop("transition source hash is outside the admitted binding")
                    source_line = int(reference["source_line"])
                    content_sha256 = str(reference["content_sha256"]).lower()
                    request_batch.append((expected_source, source_path, source_line, content_sha256))
                    transition_batch.append((
                        binding.binding_id,
                        local_row,
                        sqlite3.Binary(zlib.compress(_canonical_bytes(transition), level=3)),
                    ))
                    rows += 1
                    if len(request_batch) >= 4096:
                        self.connection.executemany(
                            "INSERT OR IGNORE INTO requests VALUES (?,?,?,?)",
                            request_batch,
                        )
                        self.connection.executemany(
                            "INSERT OR REPLACE INTO transitions VALUES (?,?,?)",
                            transition_batch,
                        )
                        self.connection.commit()
                        request_batch.clear()
                        transition_batch.clear()
            if request_batch:
                self.connection.executemany("INSERT OR IGNORE INTO requests VALUES (?,?,?,?)", request_batch)
                self.connection.executemany("INSERT OR REPLACE INTO transitions VALUES (?,?,?)", transition_batch)
            elapsed = time.perf_counter() - started
            self.connection.execute(
                "INSERT OR REPLACE INTO completed_inputs VALUES ('transition',?,?,?,?)",
                (binding.binding_id, binding.transition_sha256, rows, elapsed),
            )
            self.connection.commit()
            added += rows

        return {"eligible_transition_requests_added": added, "elapsed_seconds": time.perf_counter() - started}

    def materialize_observations(self) -> dict[str, Any]:
        sources = self.connection.execute(
            "SELECT source_sha256, source_path, COUNT(*) FROM requests GROUP BY source_sha256, source_path"
        ).fetchall()
        completed = {
            identity for identity, in self.connection.execute(
                "SELECT identity FROM completed_inputs WHERE kind='source'"
            )
        }
        inserted = 0
        started = time.perf_counter()
        for source_sha256, source_path_value, request_count in sources:
            if source_sha256 in completed:
                continue
            source_path = Path(source_path_value)
            if _sha256_path(source_path) != source_sha256:
                raise ReplayAdmissionHardStop(f"source hash mismatch: {source_path}")
            requested = {
                int(line): content
                for line, content in self.connection.execute(
                    "SELECT source_line, content_sha256 FROM requests WHERE source_sha256=?",
                    (source_sha256,),
                )
            }
            if len(requested) != request_count:
                raise ReplayAdmissionHardStop("one source line maps to conflicting content identities")
            rows = []
            maximum = max(requested)
            with _open_source(source_path) as handle:
                for line_number, line in enumerate(handle, start=1):
                    if line_number > maximum and len(rows) == request_count:
                        break
                    expected_content = requested.get(line_number)
                    if expected_content is None:
                        continue
                    try:
                        source_row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ReplayAdmissionHardStop(
                            f"invalid source JSON: {source_path}:{line_number}"
                        ) from exc
                    payload = _canonical_bytes(source_row)
                    rows.append((
                        source_sha256,
                        line_number,
                        expected_content,
                        sqlite3.Binary(zlib.compress(payload, level=3)),
                    ))
                    if len(rows) >= 2048:
                        self.connection.executemany("INSERT OR REPLACE INTO observations VALUES (?,?,?,?)", rows)
                        self.connection.commit()
                        inserted += len(rows)
                        rows.clear()
            if rows:
                self.connection.executemany("INSERT OR REPLACE INTO observations VALUES (?,?,?,?)", rows)
                inserted += len(rows)
            observed = self.connection.execute(
                "SELECT COUNT(*) FROM observations WHERE source_sha256=?", (source_sha256,)
            ).fetchone()[0]
            if observed != request_count:
                raise ReplayAdmissionHardStop(
                    f"source admission incomplete: {source_path} {observed}/{request_count}"
                )
            elapsed = time.perf_counter() - started
            self.connection.execute(
                "INSERT OR REPLACE INTO completed_inputs VALUES ('source',?,?,?,?)",
                (source_sha256, source_sha256, observed, elapsed),
            )
            self.connection.commit()

        return {"observations_inserted": inserted, "elapsed_seconds": time.perf_counter() - started}

    def finalize(self, manifest_path: str | Path) -> dict[str, Any]:
        request_count = self.connection.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        observation_count = self.connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        transition_count = self.connection.execute("SELECT COUNT(*) FROM transitions").fetchone()[0]
        if request_count != observation_count or request_count != transition_count:
            raise ReplayAdmissionHardStop(f"admission is incomplete: {observation_count}/{request_count}")
        self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.connection.commit()
        database_sha256 = _sha256_path(self.database_path)
        receipt = {
            "schema_version": ADMISSION_SCHEMA_VERSION,
            "status": "complete",
            "admission_identity_sha256": self.admission_identity_sha256,
            "binding_set_sha256": self.binding_set_sha256,
            "corpus_identity": self.corpus_identity,
            "request_count": request_count,
            "observation_count": observation_count,
            "transition_count": transition_count,
            "database_path": str(self.database_path.resolve()),
            "database_bytes": self.database_path.stat().st_size,
            "database_sha256": database_sha256,
        }
        receipt["receipt_sha256"] = _sha256_json(receipt)
        target = Path(manifest_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("wb", dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(_canonical_bytes(receipt) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)

        return receipt


class AdmittedObservationIndex:
    """Read exact source rows from a complete content-addressed admission."""

    def __init__(self, manifest_path: str | Path) -> None:
        path = Path(manifest_path).resolve(strict=True)
        receipt = json.loads(path.read_text(encoding="ascii"))
        database = Path(receipt["database_path"]).resolve(strict=True)
        if receipt.get("schema_version") != ADMISSION_SCHEMA_VERSION or receipt.get("status") != "complete":
            raise ReplayAdmissionHardStop("admission receipt is not complete")
        if _sha256_path(database) != receipt.get("database_sha256"):
            raise ReplayAdmissionHardStop("admission database hash mismatch")
        expected_receipt_sha = receipt.pop("receipt_sha256")
        if _sha256_json(receipt) != expected_receipt_sha:
            raise ReplayAdmissionHardStop("admission receipt hash mismatch")
        receipt["receipt_sha256"] = expected_receipt_sha
        self.receipt = receipt
        self.connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)

    def read(self, requests: Iterable[tuple[str, int, str]]) -> dict[tuple[str, int], dict[str, Any]]:
        result = {}
        for source_sha256, source_line, content_sha256 in requests:
            row = self.connection.execute(
                "SELECT compressed_json FROM observations WHERE source_sha256=? AND source_line=? AND content_sha256=?",
                (source_sha256, source_line, content_sha256),
            ).fetchone()
            if row is None:
                raise ReplayAdmissionHardStop("requested observation is absent from the admission index")
            value = json.loads(zlib.decompress(row[0]))
            result[(source_sha256, source_line)] = value

        return result

    def read_transitions(
        self,
        binding_id: str,
        local_rows: Iterable[int],
    ) -> dict[int, dict[str, Any]]:
        result = {}
        for local_row in local_rows:
            row = self.connection.execute(
                "SELECT compressed_json FROM transitions WHERE binding_id=? AND local_row=?",
                (binding_id, int(local_row)),
            ).fetchone()
            if row is None:
                raise ReplayAdmissionHardStop("requested transition is absent from the admission index")
            result[int(local_row)] = json.loads(zlib.decompress(row[0]))

        return result


__all__ = [
    "AdmissionBinding",
    "AdmittedObservationIndex",
    "ReplayAdmissionBuilder",
    "ReplayAdmissionHardStop",
    "bindings_from_materializer",
]
