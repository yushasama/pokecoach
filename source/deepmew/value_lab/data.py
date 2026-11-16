"""Catalog-bound float32 cache and sequence batching for the value lab.

The v2 cache is accepted only when it is bound to an accepted Phase-3 split,
its materialized target-row exclusions, and the exact canonical allowlist.
Excluded target rows remain present as temporal context.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:
    import psutil
except ImportError:  # pragma: no cover - production receipts report this explicitly.
    psutil = None

from deepmew.replay.canonical_actor_dataset import (
    FEATURE_BLOCKS,
    FEATURE_DIM,
    SEQUENCE_SCHEMA_VERSION,
    UNIT_MANIFEST_SCHEMA_VERSION,
)
from deepmew.value_lab.contracts import (
    ROW_EXCLUSION_SCHEMA_VERSION,
    SPLIT_SCHEMA_VERSION,
    STATE_ACTION_SIGNATURE_VERSION,
)


CACHE_SCHEMA_VERSION = "deepmew.value_lab_cache.v2"
CATALOG_SCHEMA_VERSION = "deepmew.value_q_dataset_catalog.v1"
SPLIT_NAMES = ("train", "validation", "calibration", "test")
RATIOS = (0.80, 0.06, 0.06, 0.08)
X_FILENAME = "X.f32.bin"
X_STREAM_CHUNK_BYTES = 8 << 20
LEGAL_OFFSET_CONTRACT = "catalog_unit_rank_then_manifest_part_then_row_prefix_sum.v1"
FEATURE_BLOCK_LAYOUT = json.loads(json.dumps(FEATURE_BLOCKS))

META_KEYS = (
    "z",
    "seat",
    "turn",
    "select_context",
    "legal_count",
    "decision_index",
    "source_idx",
    "split_idx",
    "is_target_eligible",
)
BLOCK_KEYS = (
    "row_start",
    "row_len",
    "z",
    "source_idx",
    "era_rank",
    "seat",
    "order_key",
    "policy_gidx",
    "opp_policy_gidx",
    "deck_gidx",
    "opp_deck_gidx",
    "unit_idx",
    "split_idx",
)


def _sha256(path: str | Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)

    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")

    return value


def _same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def _resolve_reference(reference: str, anchor: Path) -> Path:
    path = Path(reference)

    return path if path.is_absolute() else anchor / path


def _load_catalog(canonical_root: str, catalog_path: str) -> tuple[dict, str]:
    catalog = _read_json(catalog_path)
    catalog_sha = _sha256(catalog_path)
    if catalog.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError(f"unexpected catalog schema {catalog.get('schema_version')}")
    if not _same_path(catalog.get("out_root", ""), canonical_root):
        raise ValueError("catalog out_root does not match canonical_root")
    if catalog.get("duplicate_game_seat_blocks_across_units") != 0:
        raise ValueError("catalog contains duplicate game-seat blocks")
    if catalog.get("duplicate_game_content") != 0:
        raise ValueError("catalog contains duplicate game content")
    included = catalog.get("included_units")
    if not isinstance(included, list) or not included:
        raise ValueError("catalog lacks explicit included_units allowlist")
    identities = [(unit.get("source_id"), unit.get("unit_id")) for unit in included]
    if len(set(identities)) != len(identities):
        raise ValueError("catalog included_units identities are not unique")
    expected_units = sum(int(source["units"]) for source in catalog.get("sources", {}).values())
    if len(included) != expected_units:
        raise ValueError(f"catalog includes {len(included)} units, expected {expected_units}")
    totals = catalog.get("totals") or {}
    if int(totals.get("rows_kept", -1)) < 1 or int(totals.get("games_kept", -1)) < 1:
        raise ValueError("catalog totals are missing or empty")
    exclusions = catalog.get("unit_exclusions") or {}
    if not isinstance(exclusions.get("sha256"), str):
        raise ValueError("catalog lacks unit-exclusion binding")

    return catalog, catalog_sha


def _load_split(
    canonical_root: str,
    split_manifest_path: str,
    catalog: dict,
    catalog_sha: str,
) -> tuple[dict, str]:
    split = _read_json(split_manifest_path)
    split_sha = _sha256(split_manifest_path)
    if split.get("schema_version") != SPLIT_SCHEMA_VERSION or split.get("status") != "accepted":
        raise ValueError("split manifest is not an accepted current-schema artifact")
    if not _same_path(split.get("canonical_root", ""), canonical_root):
        raise ValueError("split canonical_root binding mismatch")
    if split.get("catalog_sha256") != catalog_sha:
        raise ValueError("split/catalog hash mismatch")
    if split.get("catalog_exclusions_sha256") != catalog["unit_exclusions"]["sha256"]:
        raise ValueError("split/catalog unit-exclusion binding mismatch")
    if split.get("split_names") != list(SPLIT_NAMES):
        raise ValueError("split names are not the four accepted folds")
    expected_ratios = dict(zip(SPLIT_NAMES, RATIOS))
    if split.get("ratios") != expected_ratios:
        raise ValueError("split ratios do not match the Phase-3 contract")
    if int(split.get("rows", -1)) != int(catalog["totals"]["rows_kept"]):
        raise ValueError("split/catalog row total mismatch")
    if int(split.get("games", -1)) != int(catalog["totals"]["games_kept"]):
        raise ValueError("split/catalog game total mismatch")
    assignment = split.get("assignment")
    if not isinstance(assignment, dict) or len(assignment) != int(split["games"]):
        raise ValueError("split assignment is missing or incomplete")
    unknown = sorted(set(assignment.values()) - set(SPLIT_NAMES))
    if unknown:
        raise ValueError(f"split assignment contains unknown folds: {unknown}")
    assignment_sha = _stable_hash({
        "catalog_sha256": catalog_sha,
        "ratios": RATIOS,
        "assignment": sorted(assignment.items()),
    })
    if split.get("assignment_sha256") != assignment_sha:
        raise ValueError("split assignment hash is not reproducible")
    expected_split_hashes = {
        name: hashlib.sha256(
            "\n".join(sorted(game_id for game_id, fold in assignment.items() if fold == name)).encode()
        ).hexdigest()
        for name in SPLIT_NAMES
    }
    if split.get("split_hashes") != expected_split_hashes:
        raise ValueError("split fold hashes are not reproducible")
    bindings = split.get("unit_bindings")
    if not isinstance(bindings, list) or len(bindings) != len(catalog["included_units"]):
        raise ValueError("split unit bindings do not cover the catalog allowlist")
    for rank, (unit, binding) in enumerate(zip(catalog["included_units"], bindings)):
        expected = (unit["source_id"], unit["unit_id"], unit["manifest_sha256"], rank)
        actual = (
            binding.get("source_id"),
            binding.get("unit_id"),
            binding.get("manifest_sha256"),
            binding.get("unit_rank"),
        )
        if actual != expected:
            raise ValueError(f"split unit binding mismatch at rank {rank}")
    acceptance = split.get("acceptance") or {}
    required_acceptance = {
        "catalog_bound": True,
        "game_grouped": True,
        "chronology_boundary_violations": 0,
        "game_overlap_across_splits": 0,
        "game_content_overlap_across_splits": 0,
        "row_exclusions_materialized": True,
        "state_action_overlap_after_exclusion": 0,
    }
    if any(acceptance.get(key) != value for key, value in required_acceptance.items()):
        raise ValueError("split acceptance receipt is incomplete")
    audit = (split.get("summary") or {}).get("state_action_overlap") or {}
    if audit.get("status") != "pass_with_materialized_row_exclusions":
        raise ValueError("split lacks materialized state-action exclusions")
    if audit.get("signature_version") != STATE_ACTION_SIGNATURE_VERSION:
        raise ValueError("split state-action signature version mismatch")
    if audit.get("signature_implementation_sha256") != split.get("signature_implementation_sha256"):
        raise ValueError("split signature implementation binding mismatch")
    if int(audit.get("cross_split_signature_count_after_exclusion", -1)) != 0:
        raise ValueError("split retains cross-fold state-action overlap")

    return split, split_sha


def _load_row_exclusions(
    split: dict,
    split_manifest_path: str,
    catalog_sha: str,
    *,
    verify_frozen_audit_bytes: bool = True,
) -> tuple[dict, Path, str, dict[tuple[str, str, str], dict]]:
    reference = split.get("row_exclusion_manifest_path")
    expected_sha = split.get("row_exclusion_manifest_sha256")
    if not isinstance(reference, str) or not reference or not isinstance(expected_sha, str):
        raise ValueError("split lacks row-exclusion manifest binding")
    path = _resolve_reference(reference, Path(split_manifest_path).parent)
    if not path.is_file() or _sha256(path) != expected_sha:
        raise ValueError("row-exclusion manifest hash mismatch")
    manifest = _read_json(path)
    if manifest.get("schema_version") != ROW_EXCLUSION_SCHEMA_VERSION or manifest.get("status") != "completed":
        raise ValueError("unexpected row-exclusion manifest schema")
    if manifest.get("catalog_sha256") != catalog_sha:
        raise ValueError("row exclusions are bound to a different catalog")
    if manifest.get("assignment_sha256") != split.get("assignment_sha256"):
        raise ValueError("row exclusions are bound to a different assignment")
    if manifest.get("signature_version") != STATE_ACTION_SIGNATURE_VERSION:
        raise ValueError("row-exclusion signature version mismatch")
    if manifest.get("signature_implementation_sha256") != split.get("signature_implementation_sha256"):
        raise ValueError("row-exclusion implementation binding mismatch")
    audit = split["summary"]["state_action_overlap"]
    signatures = manifest.get("excluded_signatures")
    if not isinstance(signatures, list) or signatures != sorted(set(signatures)):
        raise ValueError("row-exclusion signature inventory is not canonical")
    signature_sha = hashlib.sha256("\n".join(signatures).encode()).hexdigest()
    if manifest.get("excluded_signature_sha256") != signature_sha:
        raise ValueError("row-exclusion signature digest mismatch")
    if sorted(audit.get("cross_split_signatures") or []) != signatures:
        raise ValueError("row-exclusion signatures do not match the split audit")
    if audit.get("excluded_signature_sha256") != signature_sha:
        raise ValueError("split audit signature digest mismatch")
    if int(audit.get("cross_split_signature_count_before_exclusion", -1)) != len(signatures):
        raise ValueError("split audit crossing-signature count mismatch")
    context_rows = int(manifest.get("context_rows", -1))
    eligible_rows = int(manifest.get("eligible_target_rows", -1))
    excluded_rows = int(manifest.get("excluded_target_rows", -1))
    if context_rows != eligible_rows + excluded_rows:
        raise ValueError("row-exclusion target counts do not sum to context rows")
    if int(manifest.get("effective_cross_split_signature_count", -1)) != 0:
        raise ValueError("row-exclusion effective audit is not clean")
    coverage = audit.get("exclusion_coverage") or {}
    if int(coverage.get("affected_rows", -1)) != excluded_rows:
        raise ValueError("split/exclusion affected-row mismatch")
    if int(coverage.get("eligible_target_rows", -1)) != eligible_rows:
        raise ValueError("split/exclusion eligible-row mismatch")
    if int(coverage.get("effective_cross_split_signature_count", -1)) != 0:
        raise ValueError("split exclusion coverage is not clean")
    if coverage.get("row_exclusion_manifest_sha256") != expected_sha:
        raise ValueError("split exclusion coverage manifest hash mismatch")
    for db_field, hash_field in (
        ("source_audit_db_path", "source_audit_db_sha256"),
        ("effective_db_path", "effective_db_sha256"),
    ):
        db_reference = manifest.get(db_field)
        db_sha = manifest.get(hash_field)
        if not isinstance(db_reference, str) or not isinstance(db_sha, str):
            raise ValueError(f"row-exclusion manifest lacks {db_field} binding")
        db_path = _resolve_reference(db_reference, path.parent)
        if not db_path.is_file():
            raise ValueError(f"row-exclusion audit database is missing: {db_field}")
        if verify_frozen_audit_bytes and _sha256(db_path) != db_sha:
            raise ValueError(f"row-exclusion audit database hash mismatch: {db_field}")
    if manifest.get("source_audit_db_sha256") != audit.get("audit_db_sha256"):
        raise ValueError("source state-action database binding mismatch")
    if manifest.get("effective_db_sha256") != coverage.get("effective_db_sha256"):
        raise ValueError("effective state-action database binding mismatch")
    receipts: dict[tuple[str, str, str], dict] = {}
    parts = manifest.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("row-exclusion manifest has no part receipts")
    for receipt in parts:
        key = (receipt.get("source_id"), receipt.get("unit_id"), receipt.get("part"))
        if not all(isinstance(value, str) and value for value in key) or key in receipts:
            raise ValueError("row-exclusion part identities are invalid or duplicated")
        receipts[key] = receipt
    if int(coverage.get("row_mask_parts", -1)) != len(receipts):
        raise ValueError("row-exclusion part coverage count mismatch")

    return manifest, path, expected_sha, receipts


def _load_unit_manifest(canonical_root: str, unit: dict) -> tuple[dict, Path]:
    path = Path(canonical_root) / unit["manifest_path"]
    if not path.is_file() or _sha256(path) != unit["manifest_sha256"]:
        raise ValueError(f"unit manifest hash mismatch: {path}")
    manifest = _read_json(path)
    if manifest.get("schema_version") != UNIT_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unexpected unit manifest schema: {path}")
    if manifest.get("status") != "completed":
        raise ValueError(f"included unit is not completed: {path}")
    if (manifest.get("source_id"), manifest.get("unit_id")) != (unit["source_id"], unit["unit_id"]):
        raise ValueError(f"unit identity mismatch: {path}")
    if manifest.get("source_set_sha256") != unit.get("source_set_sha256"):
        raise ValueError(f"unit source-set binding mismatch: {path}")
    if manifest.get("converter_hash") != unit.get("converter_hash"):
        raise ValueError(f"unit converter binding mismatch: {path}")
    if manifest.get("information_mode") != "actor_visible":
        raise ValueError(f"included unit is not actor-visible: {path}")
    if manifest.get("feature_dim") != FEATURE_DIM or manifest.get("feature_blocks") != FEATURE_BLOCK_LAYOUT:
        raise ValueError(f"unit feature contract mismatch: {path}")
    if manifest.get("sequence_schema") != SEQUENCE_SCHEMA_VERSION:
        raise ValueError(f"unit sequence schema mismatch: {path}")
    parts = manifest.get("parts")
    if not isinstance(parts, list) or len({part.get('name') for part in parts}) != len(parts):
        raise ValueError(f"unit part inventory is missing or duplicated: {path}")

    return manifest, path


def _atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def _atomic_gzip_json(path: Path, value: dict) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with open(tmp, "wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as gzip_handle:
            with io.TextIOWrapper(gzip_handle, encoding="utf-8") as text_handle:
                json.dump(value, text_handle, sort_keys=True, separators=(",", ":"))
    os.replace(tmp, path)


class _Float32StreamWriter:
    """Write exact contiguous float32 bytes while computing their receipt once."""

    def __init__(self, handle):
        self.handle = handle
        self.digest = hashlib.sha256()
        self.bytes_written = 0

    def write(self, array: np.ndarray) -> None:
        contiguous = np.ascontiguousarray(array, dtype=np.float32)
        payload = memoryview(contiguous).cast("B")
        for start in range(0, len(payload), X_STREAM_CHUNK_BYTES):
            chunk = payload[start:start + X_STREAM_CHUNK_BYTES]
            written = self.handle.write(chunk)
            if written != len(chunk):
                raise OSError(f"short cache feature write: {written} of {len(chunk)} bytes")
            self.digest.update(chunk)
            self.bytes_written += written

    @property
    def sha256(self) -> str:
        return self.digest.hexdigest()


class _BuildTelemetry:
    """Bounded process telemetry sampled at deterministic build boundaries."""

    def __init__(self):
        self.started = time.perf_counter()
        self.process = psutil.Process() if psutil is not None else None
        self.samples: list[dict[str, int]] = []
        self.sample()

    def sample(self) -> None:
        if self.process is None:
            return
        memory = self.process.memory_info()
        io_counters = self.process.io_counters()
        self.samples.append({
            "rss_bytes": int(memory.rss),
            "private_bytes": int(getattr(memory, "private", 0)),
            "vms_bytes": int(memory.vms),
            "page_faults": int(getattr(memory, "num_page_faults", 0)),
            "read_bytes": int(io_counters.read_bytes),
            "write_bytes": int(io_counters.write_bytes),
        })

    def receipt(self) -> dict[str, Any]:
        if not self.samples:
            return {
                "instrumentation": "not_instrumented",
                "reason": "psutil is unavailable",
            }
        first = self.samples[0]
        last = self.samples[-1]

        return {
            "instrumentation": "psutil_part_boundaries",
            "samples": len(self.samples),
            "peak_rss_bytes": max(sample["rss_bytes"] for sample in self.samples),
            "peak_private_bytes": max(sample["private_bytes"] for sample in self.samples),
            "peak_vms_bytes": max(sample["vms_bytes"] for sample in self.samples),
            "page_faults_delta": last["page_faults"] - first["page_faults"],
            "process_read_bytes_delta": last["read_bytes"] - first["read_bytes"],
            "process_write_bytes_delta": last["write_bytes"] - first["write_bytes"],
        }


def _global_vocab_index(vocab: dict[str, int], value: Any) -> int:
    key = str(value)
    if key not in vocab:
        vocab[key] = len(vocab)

    return vocab[key]


def _mapped_local_index(value: Any, mapping: dict[int, int], field: str, part_path: Path) -> int:
    index = int(value)
    if index < 0:

        return -1
    if index not in mapping:
        raise ValueError(f"{field} local vocabulary index is invalid: {part_path}")

    return mapping[index]


def _normalized_counts(values: dict | Counter) -> dict[str, int]:
    return {name: int(values.get(name, 0)) for name in SPLIT_NAMES}


def prepare_cache(
    canonical_root: str,
    cache_dir: str,
    *,
    catalog_path: str,
    split_manifest_path: str,
) -> dict:
    """Build one accepted cache without ingesting catalog-excluded physical units."""
    telemetry = _BuildTelemetry()
    build_started = time.perf_counter()
    stage_seconds = Counter()
    binding_started = time.perf_counter()
    catalog, catalog_sha = _load_catalog(canonical_root, catalog_path)
    split, split_sha = _load_split(canonical_root, split_manifest_path, catalog, catalog_sha)
    exclusions, exclusion_path, exclusion_sha, mask_receipts = _load_row_exclusions(
        split,
        split_manifest_path,
        catalog_sha,
        verify_frozen_audit_bytes=False,
    )
    stage_seconds["accepted_binding_validation"] += time.perf_counter() - binding_started
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    x_path = cache / X_FILENAME
    x_tmp = cache / f".{X_FILENAME}.{uuid.uuid4().hex}.tmp"
    expected_rows = int(catalog["totals"]["rows_kept"])
    meta_dtypes = {
        key: np.bool_ if key == "is_target_eligible" else np.float32 if key == "z" else np.int32
        for key in META_KEYS
    }
    meta_arrays = {
        key: np.empty(expected_rows, dtype=dtype)
        for key, dtype in meta_dtypes.items()
    }
    blocks: dict[str, list] = {key: [] for key in BLOCK_KEYS}
    block_keys: list[str] = []
    block_key_set: set[str] = set()
    game_ids: set[str] = set()
    source_ids: list[str] = []
    unit_ids: list[str] = []
    policy_vocab: dict[str, int] = {}
    deck_vocab: dict[str, int] = {}
    assignment = split["assignment"]
    split_index = {name: index for index, name in enumerate(SPLIT_NAMES)}
    context_rows_by_split = Counter()
    eligible_rows_by_split = Counter()
    excluded_rows_by_split = Counter()
    seen_mask_parts: set[tuple[str, str, str]] = set()
    inputs: list[dict] = []
    total_rows = 0
    total_parts = 0
    total_legal_options = 0
    live_verified_source_bytes = 0
    frozen_bound_evidence_bytes = 0
    x_stream_sha = ""
    x_stream_bytes = 0
    try:
        with open(x_tmp, "wb") as x_handle:
            x_writer = _Float32StreamWriter(x_handle)
            for unit_rank, unit in enumerate(catalog["included_units"]):
                manifest_started = time.perf_counter()
                manifest, manifest_path = _load_unit_manifest(canonical_root, unit)
                stage_seconds["unit_manifest_hashing"] += time.perf_counter() - manifest_started
                live_verified_source_bytes += manifest_path.stat().st_size
                source_id = manifest["source_id"]
                if source_id not in source_ids:
                    source_ids.append(source_id)
                source_idx = source_ids.index(source_id)
                unit_idx = len(unit_ids)
                unit_ids.append(f"{source_id}/{manifest['unit_id']}")
                local_policy = manifest.get("vocab_policy") or []
                local_deck = manifest.get("vocab_deck") or []
                policy_map = {
                    index: _global_vocab_index(policy_vocab, name)
                    for index, name in enumerate(local_policy)
                }
                deck_map = {
                    index: _global_vocab_index(deck_vocab, name)
                    for index, name in enumerate(local_deck)
                }
                unit_dir = manifest_path.parent
                for part in manifest["parts"]:
                    total_parts += 1
                    part_name = part["name"]
                    part_key = (source_id, manifest["unit_id"], part_name)
                    receipt = mask_receipts.get(part_key)
                    if receipt is None:
                        raise ValueError(f"row-exclusion mask receipt missing for {part_key}")
                    seen_mask_parts.add(part_key)
                    npz_name = f"{part_name}.npz"
                    games_name = f"{part_name}.games.jsonl.gz"
                    transitions_name = f"{part_name}.transitions.jsonl.gz"
                    sequences_name = f"{part_name}.sequences.jsonl.gz"
                    actions_name = f"{part_name}.actions.jsonl.gz"
                    expected_files = part.get("files_sha256") or {}
                    npz_path = unit_dir / npz_name
                    games_path = unit_dir / games_name
                    transitions_path = unit_dir / transitions_name
                    sequences_path = unit_dir / sequences_name
                    actions_path = unit_dir / actions_name
                    source_hash_started = time.perf_counter()
                    for file_path, file_name in ((npz_path, npz_name), (games_path, games_name)):
                        expected_hash = expected_files.get(file_name)
                        if not isinstance(expected_hash, str) or not file_path.is_file() or _sha256(file_path) != expected_hash:
                            raise ValueError(f"canonical part hash mismatch: {file_path}")
                        live_verified_source_bytes += file_path.stat().st_size
                    stage_seconds["consumed_source_hashing"] += time.perf_counter() - source_hash_started
                    for file_path, file_name in (
                        (transitions_path, transitions_name),
                        (sequences_path, sequences_name),
                        (actions_path, actions_name),
                    ):
                        expected_hash = expected_files.get(file_name)
                        if not isinstance(expected_hash, str) or not file_path.is_file():
                            raise ValueError(f"canonical bound evidence is missing: {file_path}")
                        frozen_bound_evidence_bytes += file_path.stat().st_size
                    if receipt.get("input_npz_sha256") != expected_files[npz_name]:
                        raise ValueError(f"mask/NPZ input binding mismatch: {part_key}")
                    if receipt.get("input_transitions_sha256") != expected_files[transitions_name]:
                        raise ValueError(f"mask/transition input binding mismatch: {part_key}")
                    mask_path = _resolve_reference(receipt["mask_path"], exclusion_path.parent)
                    mask_hash_started = time.perf_counter()
                    if not mask_path.is_file() or _sha256(mask_path) != receipt.get("mask_sha256"):
                        raise ValueError(f"target-row mask hash mismatch: {mask_path}")
                    stage_seconds["target_mask_hashing"] += time.perf_counter() - mask_hash_started
                    live_verified_source_bytes += mask_path.stat().st_size
                    mask_load_started = time.perf_counter()
                    with np.load(mask_path, allow_pickle=False) as mask_data:
                        if mask_data.files != ["is_target_eligible"]:
                            raise ValueError(f"unexpected target-row mask fields: {mask_path}")
                        is_target_eligible = mask_data["is_target_eligible"]
                    stage_seconds["target_mask_load"] += time.perf_counter() - mask_load_started
                    if is_target_eligible.dtype != np.bool_:
                        raise ValueError(f"target-row mask is not boolean: {mask_path}")
                    npz_load_started = time.perf_counter()
                    with np.load(npz_path, allow_pickle=False) as data:
                        required_arrays = {
                            "X", "z", "seat", "turn", "select_context", "legal_count", "decision_index",
                            "policy_idx", "opp_policy_idx", "deck_idx", "opp_deck_idx",
                        }
                        if not required_arrays.issubset(data.files):
                            raise ValueError(f"canonical NPZ lacks required arrays: {npz_path}")
                        x = data["X"]
                        arrays = {key: data[key] for key in required_arrays - {"X"}}
                    stage_seconds["canonical_npz_load"] += time.perf_counter() - npz_load_started
                    n = int(part["rows"])
                    if x.dtype != np.float32 or x.shape != (n, FEATURE_DIM):
                        raise ValueError(f"canonical float32 feature shape mismatch: {npz_path}")
                    if is_target_eligible.shape != (n,) or int(receipt.get("rows", -1)) != n:
                        raise ValueError(f"target-row mask shape mismatch: {mask_path}")
                    if int(receipt.get("excluded_rows", -1)) != int((~is_target_eligible).sum()):
                        raise ValueError(f"target-row mask excluded count mismatch: {mask_path}")
                    for key, array in arrays.items():
                        if array.shape != (n,):
                            raise ValueError(f"canonical metadata shape mismatch for {key}: {npz_path}")
                    legal_count = arrays["legal_count"].astype(np.int64, copy=False)
                    if np.any(legal_count < 0):
                        raise ValueError(f"canonical legal-option count is negative: {npz_path}")
                    part_legal_options = int(legal_count.sum(dtype=np.int64))
                    if part_legal_options != int(part.get("action_options", -1)):
                        raise ValueError(f"canonical legal-option total mismatch: {npz_path}")
                    x_write_started = time.perf_counter()
                    x_writer.write(x)
                    stage_seconds["x_stream_write_and_hash"] += time.perf_counter() - x_write_started
                    row_slice = slice(total_rows, total_rows + n)
                    meta_arrays["z"][row_slice] = arrays["z"]
                    for key in ("seat", "turn", "select_context", "legal_count", "decision_index"):
                        meta_arrays[key][row_slice] = arrays[key]
                    meta_arrays["source_idx"][row_slice] = source_idx
                    meta_arrays["is_target_eligible"][row_slice] = is_target_eligible
                    part_split_idx = np.full(n, -1, dtype=np.int32)
                    expected_start = 0
                    games_started = time.perf_counter()
                    with gzip.open(games_path, "rt", encoding="utf-8") as handle:
                        for line in handle:
                            record = json.loads(line)
                            start = int(record["block_start"])
                            length = int(record["block_len"])
                            end = start + length
                            if start != expected_start or length < 1 or end > n:
                                raise ValueError(f"game blocks do not exactly cover part rows: {games_path}")
                            expected_start = end
                            if record.get("schema_version") != SEQUENCE_SCHEMA_VERSION:
                                raise ValueError(f"unexpected game-block schema: {games_path}")
                            if record.get("information_mode") != "actor_visible":
                                raise ValueError(f"game block is not actor-visible: {games_path}")
                            if record.get("game_completion_status") != "completed" or record.get("safety_status") != "clean":
                                raise ValueError(f"game block is incomplete or unsafe: {games_path}")
                            if (record.get("source_id"), record.get("unit_id")) != (source_id, manifest["unit_id"]):
                                raise ValueError(f"game-block identity mismatch: {games_path}")
                            game_id = record.get("canonical_game_id")
                            fold = assignment.get(game_id)
                            if fold not in split_index:
                                raise ValueError(f"game lacks an accepted split assignment: {game_id}")
                            fold_idx = split_index[fold]
                            part_split_idx[start:end] = fold_idx
                            game_ids.add(game_id)
                            context_rows_by_split[fold] += length
                            eligible = int(is_target_eligible[start:end].sum())
                            eligible_rows_by_split[fold] += eligible
                            excluded_rows_by_split[fold] += length - eligible
                            if not np.all(arrays["z"][start:end] == float(record["z"])):
                                raise ValueError(f"game-block outcome does not match NPZ rows: {games_path}")
                            if not np.all(arrays["seat"][start:end] == int(record["seat"])):
                                raise ValueError(f"game-block seat does not match NPZ rows: {games_path}")
                            block_key = f"{game_id}:s{record['seat']}"
                            if block_key in block_key_set:
                                raise ValueError(f"duplicate game-seat block in cache inputs: {block_key}")
                            block_key_set.add(block_key)
                            block_keys.append(block_key)
                            blocks["row_start"].append(total_rows + start)
                            blocks["row_len"].append(length)
                            blocks["z"].append(record["z"])
                            blocks["source_idx"].append(source_idx)
                            blocks["era_rank"].append(manifest["era_rank"])
                            blocks["seat"].append(record["seat"])
                            order_key = record.get("order_key")
                            blocks["order_key"].append(int(order_key) if isinstance(order_key, int) else -1)
                            blocks["policy_gidx"].append(_mapped_local_index(
                                arrays["policy_idx"][start], policy_map, "policy_idx", npz_path,
                            ))
                            blocks["opp_policy_gidx"].append(_mapped_local_index(
                                arrays["opp_policy_idx"][start], policy_map, "opp_policy_idx", npz_path,
                            ))
                            blocks["deck_gidx"].append(_mapped_local_index(
                                arrays["deck_idx"][start], deck_map, "deck_idx", npz_path,
                            ))
                            blocks["opp_deck_gidx"].append(_mapped_local_index(
                                arrays["opp_deck_idx"][start], deck_map, "opp_deck_idx", npz_path,
                            ))
                            blocks["unit_idx"].append(unit_idx)
                            blocks["split_idx"].append(fold_idx)
                    if expected_start != n or np.any(part_split_idx < 0):
                        raise ValueError(f"game blocks leave uncovered part rows: {games_path}")
                    stage_seconds["game_index_parse_and_transform"] += time.perf_counter() - games_started
                    meta_arrays["split_idx"][row_slice] = part_split_idx
                    total_rows += n
                    total_legal_options += part_legal_options
                    inputs.append({
                        "source_id": source_id,
                        "unit_id": manifest["unit_id"],
                        "unit_rank": unit_rank,
                        "unit_manifest_sha256": unit["manifest_sha256"],
                        "part": part_name,
                        "rows": n,
                        "npz_sha256": expected_files[npz_name],
                        "games_sha256": expected_files[games_name],
                        "transitions_sha256": expected_files[transitions_name],
                        "sequences_sha256": expected_files[sequences_name],
                        "actions_sha256": expected_files[actions_name],
                        "target_mask_sha256": receipt["mask_sha256"],
                        "action_options": part_legal_options,
                        "live_verified": [npz_name, games_name, "target_mask"],
                        "frozen_manifest_bound": [transitions_name, sequences_name, actions_name],
                    })
                    del x, arrays, is_target_eligible, part_split_idx
                    telemetry.sample()
            x_handle.flush()
            os.fsync(x_handle.fileno())
            x_stream_sha = x_writer.sha256
            x_stream_bytes = x_writer.bytes_written
        if seen_mask_parts != set(mask_receipts):
            missing = set(mask_receipts) - seen_mask_parts
            extra = seen_mask_parts - set(mask_receipts)
            raise ValueError(f"row-mask receipt coverage mismatch: missing={len(missing)} extra={len(extra)}")
        if total_rows != expected_rows or total_rows != int(exclusions["context_rows"]):
            raise ValueError("cache context rows do not match catalog/exclusion totals")
        if len(game_ids) != int(catalog["totals"]["games_kept"]):
            raise ValueError("cache game count does not match catalog")
        assignment_ids = set(assignment)
        if game_ids != assignment_ids:
            raise ValueError(
                f"cache/split assignment coverage mismatch: missing={len(game_ids - assignment_ids)} "
                f"extra={len(assignment_ids - game_ids)}"
            )
        actual_eligible = sum(eligible_rows_by_split.values())
        actual_excluded = sum(excluded_rows_by_split.values())
        if actual_eligible != int(exclusions["eligible_target_rows"]):
            raise ValueError("eligible target-row total mismatch")
        if actual_excluded != int(exclusions["excluded_target_rows"]):
            raise ValueError("excluded target-row total mismatch")
        if _normalized_counts(eligible_rows_by_split) != _normalized_counts(exclusions.get("eligible_target_rows_by_split") or {}):
            raise ValueError("eligible target-row split counts mismatch")
        if _normalized_counts(excluded_rows_by_split) != _normalized_counts(exclusions.get("excluded_target_rows_by_split") or {}):
            raise ValueError("excluded target-row split counts mismatch")
        summary_rows = {
            name: sum(int(value) for value in split["summary"]["counts_rows"][name].values())
            for name in SPLIT_NAMES
        }
        if _normalized_counts(context_rows_by_split) != summary_rows:
            raise ValueError("cache context split counts do not match split summary")
        if x_stream_bytes != expected_rows * FEATURE_DIM * np.dtype(np.float32).itemsize:
            raise ValueError("cache feature byte count does not match the exact float32 contract")
        os.replace(x_tmp, x_path)
    finally:
        if x_tmp.exists():
            x_tmp.unlink()

    block_dtypes = {
        "row_start": np.int64,
        "row_len": np.int32,
        "z": np.float32,
        **{key: np.int32 for key in BLOCK_KEYS if key not in {"row_start", "row_len", "z"}},
    }
    block_arrays = {
        key: np.asarray(values, dtype=block_dtypes[key])
        for key, values in blocks.items()
    }
    metadata_started = time.perf_counter()
    _atomic_savez(cache / "meta.npz", meta_arrays)
    _atomic_savez(cache / "blocks.npz", block_arrays)
    keys_payload = {
        "block_keys": block_keys,
        "source_ids": source_ids,
        "unit_ids": unit_ids,
        "policy_vocab": [name for name, _ in sorted(policy_vocab.items(), key=lambda item: item[1])],
        "deck_vocab": [name for name, _ in sorted(deck_vocab.items(), key=lambda item: item[1])],
        "split_names": list(SPLIT_NAMES),
    }
    _atomic_gzip_json(cache / "blocks_keys.json.gz", keys_payload)
    stage_seconds["metadata_and_index_finalize"] += time.perf_counter() - metadata_started
    telemetry.sample()
    output_hash_started = time.perf_counter()
    output_files = {
        X_FILENAME: {"bytes": x_path.stat().st_size, "sha256": x_stream_sha},
    }
    for filename in ("meta.npz", "blocks.npz", "blocks_keys.json.gz"):
        path = cache / filename
        output_files[filename] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    stage_seconds["small_output_receipt_hashing"] += time.perf_counter() - output_hash_started
    telemetry.sample()
    elapsed_seconds = time.perf_counter() - build_started
    output_bytes = sum(int(receipt["bytes"]) for receipt in output_files.values())
    measured_io = telemetry.receipt()
    measured_read_bytes = measured_io.get("process_read_bytes_delta")
    measured_write_bytes = measured_io.get("process_write_bytes_delta")
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "status": "completed",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "canonical_root": str(Path(canonical_root).resolve()),
        "catalog_path": str(Path(catalog_path).resolve()),
        "catalog_sha256": catalog_sha,
        "catalog_exclusions_sha256": catalog["unit_exclusions"]["sha256"],
        "split_manifest_path": str(Path(split_manifest_path).resolve()),
        "split_manifest_sha256": split_sha,
        "assignment_sha256": split["assignment_sha256"],
        "row_exclusion_manifest_path": str(exclusion_path.resolve()),
        "row_exclusion_manifest_sha256": exclusion_sha,
        "state_action_signature_version": STATE_ACTION_SIGNATURE_VERSION,
        "signature_implementation_sha256": split["signature_implementation_sha256"],
        "rows": total_rows,
        "eligible_target_rows": sum(eligible_rows_by_split.values()),
        "excluded_target_rows": sum(excluded_rows_by_split.values()),
        "legal_options": total_legal_options,
        "legal_offset_contract": LEGAL_OFFSET_CONTRACT,
        "context_rows_by_split": _normalized_counts(context_rows_by_split),
        "eligible_target_rows_by_split": _normalized_counts(eligible_rows_by_split),
        "excluded_target_rows_by_split": _normalized_counts(excluded_rows_by_split),
        "games": len(game_ids),
        "blocks": len(block_keys),
        "included_units": len(unit_ids),
        "atomic_parts": total_parts,
        "feature_dim": FEATURE_DIM,
        "feature_dtype": "float32",
        "feature_blocks": FEATURE_BLOCK_LAYOUT,
        "sources": source_ids,
        "split_names": list(SPLIT_NAMES),
        "input_bindings_sha256": _stable_hash(inputs),
        "inputs": inputs,
        "output_files": output_files,
        "frozen_evidence_verification": {
            "mode": "accepted_hash_binding_without_byte_reread",
            "audit_databases": {
                "source_audit_db_sha256": exclusions["source_audit_db_sha256"],
                "effective_db_sha256": exclusions["effective_db_sha256"],
            },
            "part_files": ["transitions", "sequences", "actions"],
            "reason": "accepted Phase-3 evidence is not consumed by the value-cache transform",
        },
        "build_metrics": {
            "elapsed_seconds": elapsed_seconds,
            "rows_per_second": total_rows / elapsed_seconds,
            "stage_seconds": {key: float(value) for key, value in sorted(stage_seconds.items())},
            "live_verified_source_bytes": live_verified_source_bytes,
            "frozen_bound_evidence_bytes": frozen_bound_evidence_bytes,
            "output_bytes": output_bytes,
            "x_bytes": x_stream_bytes,
            "x_stream_sha256": x_stream_sha,
            "process": measured_io,
            "measured_read_mib_per_second": (
                measured_read_bytes / (1 << 20) / elapsed_seconds
                if isinstance(measured_read_bytes, int) else "not_instrumented"
            ),
            "measured_write_mib_per_second": (
                measured_write_bytes / (1 << 20) / elapsed_seconds
                if isinstance(measured_write_bytes, int) else "not_instrumented"
            ),
        },
    }
    manifest_path = cache / "cache_manifest.json"
    tmp_manifest = cache / f".cache_manifest.json.{uuid.uuid4().hex}.tmp"
    tmp_manifest.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp_manifest, manifest_path)

    return manifest


class LabData:
    def __init__(
        self,
        cache_dir: str,
        *,
        split_manifest_path: str,
        load_labels: bool = True,
    ):
        if not isinstance(load_labels, bool):
            raise ValueError("load_labels must be boolean")
        self.has_labels = load_labels
        cache = Path(cache_dir)
        self.manifest = _read_json(cache / "cache_manifest.json")
        if self.manifest.get("schema_version") != CACHE_SCHEMA_VERSION or self.manifest.get("status") != "completed":
            raise ValueError("cache is not an accepted v2 artifact")
        if _sha256(split_manifest_path) != self.manifest.get("split_manifest_sha256"):
            raise ValueError("LabData split manifest does not match the cache binding")
        split = _read_json(split_manifest_path)
        if split.get("schema_version") != SPLIT_SCHEMA_VERSION or split.get("status") != "accepted":
            raise ValueError("LabData requires an accepted current-schema split")
        for cache_field, split_field in (
            ("catalog_sha256", "catalog_sha256"),
            ("assignment_sha256", "assignment_sha256"),
            ("row_exclusion_manifest_sha256", "row_exclusion_manifest_sha256"),
            ("signature_implementation_sha256", "signature_implementation_sha256"),
        ):
            if self.manifest.get(cache_field) != split.get(split_field):
                raise ValueError(f"LabData cache/split binding mismatch: {cache_field}")
        if self.manifest.get("state_action_signature_version") != STATE_ACTION_SIGNATURE_VERSION:
            raise ValueError("LabData cache signature version is stale")
        for filename, receipt in self.manifest.get("output_files", {}).items():
            path = cache / filename
            if not path.is_file() or path.stat().st_size != int(receipt["bytes"]) or _sha256(path) != receipt["sha256"]:
                raise ValueError(f"cache output hash mismatch: {path}")
        required_outputs = {X_FILENAME, "meta.npz", "blocks.npz", "blocks_keys.json.gz"}
        if set(self.manifest.get("output_files", {})) != required_outputs:
            raise ValueError("cache output inventory is incomplete")
        rows = int(self.manifest["rows"])
        dim = int(self.manifest["feature_dim"])
        if (
            self.manifest.get("feature_dtype") != "float32"
            or dim != FEATURE_DIM
            or self.manifest.get("feature_blocks") != FEATURE_BLOCK_LAYOUT
        ):
            raise ValueError("cache feature contract mismatch")
        if (cache / X_FILENAME).stat().st_size != rows * dim * np.dtype(np.float32).itemsize:
            raise ValueError("cache feature file size is inconsistent")
        self.X = np.memmap(cache / X_FILENAME, dtype=np.float32, mode="r", shape=(rows, dim))
        with np.load(cache / "meta.npz", allow_pickle=False) as meta:
            if set(meta.files) != set(META_KEYS):
                raise ValueError("cache metadata inventory is invalid")
            loaded_meta_keys = META_KEYS if load_labels else tuple(key for key in META_KEYS if key != "z")
            self.meta = {key: meta[key] for key in loaded_meta_keys}
        if any(array.shape != (rows,) for array in self.meta.values()):
            raise ValueError("cache metadata inventory or shape mismatch")
        if self.meta["is_target_eligible"].dtype != np.bool_:
            raise ValueError("cache target eligibility is not boolean")
        with np.load(cache / "blocks.npz", allow_pickle=False) as blocks:
            if set(blocks.files) != set(BLOCK_KEYS):
                raise ValueError("cache block inventory is invalid")
            loaded_block_keys = BLOCK_KEYS if load_labels else tuple(key for key in BLOCK_KEYS if key != "z")
            self.blocks = {key: blocks[key] for key in loaded_block_keys}
        n_blocks = int(self.manifest["blocks"])
        if any(array.shape != (n_blocks,) for array in self.blocks.values()):
            raise ValueError("cache block inventory or shape mismatch")
        starts = self.blocks["row_start"].astype(np.int64, copy=False)
        lengths = self.blocks["row_len"].astype(np.int64, copy=False)
        if n_blocks < 1 or starts[0] != 0 or np.any(lengths < 1):
            raise ValueError("cache blocks are empty or invalid")
        if np.any(starts[1:] != starts[:-1] + lengths[:-1]) or starts[-1] + lengths[-1] != rows:
            raise ValueError("cache blocks do not exactly cover context rows")
        with gzip.open(cache / "blocks_keys.json.gz", "rt", encoding="utf-8") as handle:
            keys = json.load(handle)
        self.block_keys = keys["block_keys"]
        self.source_ids = keys["source_ids"]
        self.policy_vocab = keys["policy_vocab"]
        self.deck_vocab = keys["deck_vocab"]
        self.split_names = keys["split_names"]
        if len(self.block_keys) != n_blocks or len(set(self.block_keys)) != n_blocks:
            raise ValueError("cache block keys are missing or duplicated")
        if self.split_names != list(SPLIT_NAMES) or split.get("split_names") != list(SPLIT_NAMES):
            raise ValueError("cache/split fold names mismatch")
        assignment = split.get("assignment") or {}
        unknown_folds = sorted(set(assignment.values()) - set(SPLIT_NAMES))
        if unknown_folds:
            raise ValueError(f"LabData split contains unknown folds: {unknown_folds}")
        block_game_ids = {key.rsplit(":s", 1)[0] for key in self.block_keys}
        if len(block_game_ids) != int(self.manifest["games"]):
            raise ValueError("cache block keys do not match the recorded game count")
        if set(assignment) != block_game_ids:
            raise ValueError("LabData split has missing or extra game assignments")
        index_of = {name: index for index, name in enumerate(SPLIT_NAMES)}
        expected_split_idx = np.asarray([
            index_of[assignment[key.rsplit(":s", 1)[0]]]
            for key in self.block_keys
        ], dtype=np.int32)
        if np.any(self.blocks["split_idx"] != expected_split_idx):
            raise ValueError("cache block split indices do not match the bound assignment")
        self.split_of_block = expected_split_idx
        expected_row_split = np.repeat(expected_split_idx, lengths)
        if not np.array_equal(self.meta["split_idx"], expected_row_split):
            raise ValueError("cache row split indices do not match block boundaries")
        context_counts = np.bincount(expected_row_split, minlength=len(SPLIT_NAMES))
        eligible_counts = np.bincount(
            expected_row_split,
            weights=self.meta["is_target_eligible"].astype(np.int64),
            minlength=len(SPLIT_NAMES),
        ).astype(np.int64)
        expected_context = [int(self.manifest["context_rows_by_split"][name]) for name in SPLIT_NAMES]
        expected_eligible = [int(self.manifest["eligible_target_rows_by_split"][name]) for name in SPLIT_NAMES]
        expected_excluded = [int(self.manifest["excluded_target_rows_by_split"][name]) for name in SPLIT_NAMES]
        if context_counts.tolist() != expected_context or eligible_counts.tolist() != expected_eligible:
            raise ValueError("cache per-fold context/eligible counts are inconsistent")
        if (context_counts - eligible_counts).tolist() != expected_excluded:
            raise ValueError("cache per-fold excluded counts are inconsistent")
        if int(self.meta["is_target_eligible"].sum()) != int(self.manifest["eligible_target_rows"]):
            raise ValueError("cache eligible target-row count mismatch")
        if rows - int(self.meta["is_target_eligible"].sum()) != int(self.manifest["excluded_target_rows"]):
            raise ValueError("cache excluded target-row count mismatch")
        if self.manifest.get("legal_offset_contract") != LEGAL_OFFSET_CONTRACT:
            raise ValueError("cache legal-option offset contract is missing or stale")
        legal_count = self.meta["legal_count"].astype(np.int64, copy=False)
        if np.any(legal_count < 0):
            raise ValueError("cache legal-option count is negative")
        self.legal_offset = np.empty(rows, dtype=np.int64)
        self.legal_offset[0] = 0
        if rows > 1:
            np.cumsum(legal_count[:-1], dtype=np.int64, out=self.legal_offset[1:])
        legal_options = int(self.legal_offset[-1] + legal_count[-1])
        if legal_options != int(self.manifest.get("legal_options", -1)):
            raise ValueError("cache legal-option offsets do not match the recorded total")

    def block_ids(self, split: str) -> np.ndarray:
        if split not in self.split_names:
            raise ValueError(f"unknown split {split}")
        index = self.split_names.index(split)

        return np.nonzero(self.split_of_block == index)[0]

    def batch_arrays(self, block_ids: np.ndarray, max_len: int = 0):
        if not self.has_labels:
            raise RuntimeError("label-bearing batches are unavailable in label-blind mode")
        block_ids = np.asarray(block_ids, dtype=np.int64)
        if block_ids.ndim != 1 or len(block_ids) == 0:
            raise ValueError("batch requires at least one one-dimensional block id")
        if np.any(block_ids < 0) or np.any(block_ids >= len(self.block_keys)):
            raise ValueError("batch block id is outside the block table")
        starts = self.blocks["row_start"][block_ids]
        lengths = self.blocks["row_len"][block_ids]
        if max_len and np.any(lengths > max_len):
            raise ValueError("sequence truncation is forbidden until nonterminal chunk targets are represented")
        horizon = int(lengths.max())
        batch = len(block_ids)
        x = np.zeros((batch, horizon, self.X.shape[1]), dtype=np.float32)
        sequence_mask = np.zeros((batch, horizon), dtype=bool)
        target_mask = np.zeros((batch, horizon), dtype=bool)
        for row, (start, length) in enumerate(zip(starts, lengths)):
            end = start + length
            x[row, :length] = self.X[start:end]
            sequence_mask[row, :length] = True
            target_mask[row, :length] = self.meta["is_target_eligible"][start:end]
        z = self.blocks["z"][block_ids].astype(np.float32)

        return x, sequence_mask, target_mask, z
