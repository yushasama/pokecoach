"""Bounded Phase-4 I/O profiler over accepted canonical cache inputs.

This is instrumentation, not a cache builder.  It reads only parts selected
from the accepted catalog, writes task-owned scratch outputs, validates those
outputs, records raw timings, and removes the scratch files unless requested
otherwise.  It never writes a cache manifest.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import deepmew.value_lab.data as value_data  # noqa: E402


FEATURE_DIM = 144
X_ROW_BYTES = FEATURE_DIM * np.dtype(np.float32).itemsize
PROFILE_SCHEMA_VERSION = "deepmew.value_q_cache_io_profile.v1"
SELECTIONS = {
    "small": (
        "r18_awr",
        "research_r18_awr_iterate_r3_data_shards_decisions_shard_001.jsonl.gz",
        ("part_0000",),
    ),
    "medium": (
        "r18_artifacts",
        "DeepMew_artifacts_datasets_r18_starmie_curriculum_graded_shards_decisions_shard_001.jsonl.gz",
        ("part_0000",),
    ),
    "largest": (
        "unified_r2",
        "DeepMew_artifacts_datasets_unified_r2_refgames_dec061_regen3_clean_shards_decisions_shard_005.jsonl.gz",
        ("part_0012",),
    ),
    "multi_part": (
        "unified_r2",
        "DeepMew_artifacts_datasets_unified_r2_refgames_dec061_regen3_clean_shards_decisions_shard_001.jsonl.gz",
        None,
    ),
}


@dataclass(frozen=True)
class PartInput:
    selection: str
    source_id: str
    unit_id: str
    unit_manifest_path: Path
    unit_manifest_sha256: str
    part: dict[str, Any]
    npz_path: Path
    games_path: Path
    transitions_path: Path
    mask_path: Path
    mask_sha256: str


class ResourceSampler:
    def __init__(self, interval_seconds: float = 0.25):
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "ResourceSampler":
        process = psutil.Process()
        process.cpu_percent(None)
        psutil.cpu_percent(None)

        def sample() -> None:
            while not self._stop.is_set():
                memory = process.memory_info()
                virtual = psutil.virtual_memory()
                swap = psutil.swap_memory()
                io = process.io_counters()
                disks = psutil.disk_io_counters(perdisk=True) or {}
                self.samples.append({
                    "monotonic_seconds": time.perf_counter(),
                    "process_cpu_percent": process.cpu_percent(None),
                    "system_cpu_percent": psutil.cpu_percent(None),
                    "rss_bytes": int(memory.rss),
                    "vms_bytes": int(memory.vms),
                    "private_bytes": int(getattr(memory, "private", 0)),
                    "page_faults": int(getattr(memory, "num_page_faults", 0)),
                    "available_physical_bytes": int(virtual.available),
                    "swap_used_bytes": int(swap.used),
                    "swap_sin_bytes": int(swap.sin),
                    "swap_sout_bytes": int(swap.sout),
                    "process_read_bytes": int(io.read_bytes),
                    "process_write_bytes": int(io.write_bytes),
                    "disks": {
                        name: {
                            "read_bytes": int(counter.read_bytes),
                            "write_bytes": int(counter.write_bytes),
                            "read_time_ms": int(counter.read_time),
                            "write_time_ms": int(counter.write_time),
                        }
                        for name, counter in disks.items()
                    },
                })
                self._stop.wait(self.interval_seconds)

        self._thread = threading.Thread(target=sample, name="phase4-io-sampler", daemon=True)
        self._thread.start()

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {"samples": 0}
        first = self.samples[0]
        last = self.samples[-1]

        return {
            "samples": len(self.samples),
            "process_cpu_percent_mean": _mean(self.samples, "process_cpu_percent"),
            "process_cpu_percent_peak": _max(self.samples, "process_cpu_percent"),
            "system_cpu_percent_mean": _mean(self.samples, "system_cpu_percent"),
            "system_cpu_percent_peak": _max(self.samples, "system_cpu_percent"),
            "rss_bytes_peak": _max(self.samples, "rss_bytes"),
            "private_bytes_peak": _max(self.samples, "private_bytes"),
            "vms_bytes_peak": _max(self.samples, "vms_bytes"),
            "available_physical_bytes_min": _min(self.samples, "available_physical_bytes"),
            "page_faults_delta": last["page_faults"] - first["page_faults"],
            "swap_used_bytes_peak": _max(self.samples, "swap_used_bytes"),
            "swap_sin_bytes_delta": last["swap_sin_bytes"] - first["swap_sin_bytes"],
            "swap_sout_bytes_delta": last["swap_sout_bytes"] - first["swap_sout_bytes"],
            "process_read_bytes_delta": last["process_read_bytes"] - first["process_read_bytes"],
            "process_write_bytes_delta": last["process_write_bytes"] - first["process_write_bytes"],
            "disk_counter_deltas": _disk_deltas(first["disks"], last["disks"]),
        }


def _mean(samples: list[dict[str, Any]], key: str) -> float:
    return sum(float(sample[key]) for sample in samples) / len(samples)


def _max(samples: list[dict[str, Any]], key: str) -> int | float:
    return max(sample[key] for sample in samples)


def _min(samples: list[dict[str, Any]], key: str) -> int | float:
    return min(sample[key] for sample in samples)


def _disk_deltas(first: dict[str, dict[str, int]], last: dict[str, dict[str, int]]) -> dict:
    deltas = {}
    for name in sorted(set(first) & set(last)):
        deltas[name] = {
            key: int(last[name][key]) - int(first[name][key])
            for key in first[name]
        }

    return deltas


def _sha256(path: Path, chunk_size: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)

    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")

    return value


def _resolve(reference: str, anchor: Path) -> Path:
    path = Path(reference)

    return path if path.is_absolute() else anchor / path


def _elapsed(operation) -> tuple[Any, float]:
    started = time.perf_counter()
    value = operation()

    return value, time.perf_counter() - started


def _load_inputs(
    canonical_root: Path,
    catalog_path: Path,
    split_path: Path,
) -> tuple[dict, dict, dict, Path, dict[str, list[PartInput]]]:
    catalog, catalog_sha = value_data._load_catalog(str(canonical_root), str(catalog_path))
    split, _ = value_data._load_split(
        str(canonical_root),
        str(split_path),
        catalog,
        catalog_sha,
    )
    exclusion, exclusion_path, _, receipts = value_data._load_row_exclusions(
        split,
        str(split_path),
        catalog_sha,
        verify_frozen_audit_bytes=False,
    )
    units = {
        (unit["source_id"], unit["unit_id"]): unit
        for unit in catalog["included_units"]
    }
    selections: dict[str, list[PartInput]] = {}
    for selection, (source_id, unit_id, requested_parts) in SELECTIONS.items():
        unit = units[(source_id, unit_id)]
        manifest, manifest_path = value_data._load_unit_manifest(str(canonical_root), unit)
        available = {part["name"]: part for part in manifest["parts"]}
        names = tuple(available) if requested_parts is None else requested_parts
        selected = []
        for part_name in names:
            part = available[part_name]
            receipt = receipts[(source_id, unit_id, part_name)]
            selected.append(PartInput(
                selection=selection,
                source_id=source_id,
                unit_id=unit_id,
                unit_manifest_path=manifest_path,
                unit_manifest_sha256=unit["manifest_sha256"],
                part=part,
                npz_path=manifest_path.parent / f"{part_name}.npz",
                games_path=manifest_path.parent / f"{part_name}.games.jsonl.gz",
                transitions_path=manifest_path.parent / f"{part_name}.transitions.jsonl.gz",
                mask_path=_resolve(receipt["mask_path"], exclusion_path.parent),
                mask_sha256=receipt["mask_sha256"],
            ))
        selections[selection] = selected

    return catalog, split, exclusion, exclusion_path, selections


def _verify_binding(path: Path, expected_sha256: str) -> tuple[int, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"source hash mismatch: {path}")

    return path.stat().st_size, actual


def _profile_audit_databases(exclusion: dict, exclusion_path: Path, *, rehash: bool) -> dict:
    records = []
    started = time.perf_counter()
    for path_field, sha_field in (
        ("source_audit_db_path", "source_audit_db_sha256"),
        ("effective_db_path", "effective_db_sha256"),
    ):
        path = _resolve(exclusion[path_field], exclusion_path.parent)
        item_started = time.perf_counter()
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        actual = _sha256(path) if rehash else exclusion[sha_field]
        if actual != exclusion[sha_field]:
            raise ValueError(f"audit database hash mismatch: {path}")
        records.append({
            "field": path_field,
            "path": str(path),
            "bytes": size,
            "sha256": actual,
            "seconds": time.perf_counter() - item_started,
        })
    seconds = time.perf_counter() - started
    total_bytes = sum(item["bytes"] for item in records)

    return {
        "files": records,
        "bytes": total_bytes,
        "seconds": seconds,
        "mib_per_second": (
            total_bytes / (1 << 20) / seconds if rehash else "not_applicable_no_byte_read"
        ),
        "verification": "live_sha256" if rehash else "accepted_hash_binding_without_byte_reread",
    }


def _profile_selection(
    selection: str,
    parts: list[PartInput],
    assignment: dict[str, str],
    scratch_root: Path,
    *,
    mode: str,
    keep_output: bool,
) -> dict:
    run_dir = scratch_root / f"{selection}-{uuid.uuid4().hex}"
    run_dir.mkdir(parents=True)
    x_path = run_dir / "X.f32.bin"
    meta_path = run_dir / "meta.npz"
    blocks_path = run_dir / "blocks.npz"
    keys_path = run_dir / "blocks_keys.json.gz"
    stage_seconds = {
        "source_hashing": 0.0,
        "mask_load": 0.0,
        "npz_load": 0.0,
        "x_transform": 0.0,
        "x_write": 0.0,
        "games_parse_transform": 0.0,
        "metadata_finalize_write": 0.0,
        "output_receipt_hashing": 0.0,
        "independent_reopen_validate": 0.0,
    }
    source_bytes = 0
    bound_evidence_bytes = 0
    meta_rows: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "z", "seat", "turn", "select_context", "legal_count",
            "decision_index", "split_idx", "is_target_eligible",
        )
    }
    block_starts: list[int] = []
    block_lengths: list[int] = []
    block_z: list[float] = []
    block_seat: list[int] = []
    block_keys: list[str] = []
    split_names = ("train", "validation", "calibration", "test")
    split_index = {name: index for index, name in enumerate(split_names)}
    total_rows = 0
    output_hashes: dict[str, str] = {}
    started = time.perf_counter()
    try:
        with open(x_path, "wb") as x_handle:
            x_writer = value_data._Float32StreamWriter(x_handle) if mode == "optimized" else None
            seen_manifests: set[Path] = set()
            for item in parts:
                hash_started = time.perf_counter()
                if item.unit_manifest_path not in seen_manifests:
                    size, _ = _verify_binding(item.unit_manifest_path, item.unit_manifest_sha256)
                    source_bytes += size
                    seen_manifests.add(item.unit_manifest_path)
                expected = item.part["files_sha256"]
                live_paths = [item.npz_path, item.games_path]
                if mode == "baseline":
                    live_paths.append(item.transitions_path)
                for path in live_paths:
                    size, _ = _verify_binding(path, expected[path.name])
                    source_bytes += size
                if mode == "optimized":
                    if not item.transitions_path.is_file():
                        raise FileNotFoundError(item.transitions_path)
                    bound_evidence_bytes += item.transitions_path.stat().st_size
                size, _ = _verify_binding(item.mask_path, item.mask_sha256)
                source_bytes += size
                stage_seconds["source_hashing"] += time.perf_counter() - hash_started

                mask_started = time.perf_counter()
                with np.load(item.mask_path, allow_pickle=False) as mask_data:
                    eligible = mask_data["is_target_eligible"].copy()
                stage_seconds["mask_load"] += time.perf_counter() - mask_started

                npz_started = time.perf_counter()
                with np.load(item.npz_path, allow_pickle=False) as data:
                    x = data["X"]
                    arrays = {
                        key: data[key]
                        for key in (
                            "z", "seat", "turn", "select_context", "legal_count", "decision_index",
                        )
                    }
                stage_seconds["npz_load"] += time.perf_counter() - npz_started
                rows = int(item.part["rows"])
                if x.dtype != np.float32 or x.shape != (rows, FEATURE_DIM):
                    raise ValueError(f"unexpected feature contract: {item.npz_path}")
                if eligible.dtype != np.bool_ or eligible.shape != (rows,):
                    raise ValueError(f"unexpected mask contract: {item.mask_path}")

                transform_started = time.perf_counter()
                payload = (
                    np.ascontiguousarray(x, dtype=np.float32).tobytes(order="C")
                    if mode == "baseline" else None
                )
                stage_seconds["x_transform"] += time.perf_counter() - transform_started
                write_started = time.perf_counter()
                if mode == "baseline":
                    x_handle.write(payload)
                else:
                    x_writer.write(x)
                stage_seconds["x_write"] += time.perf_counter() - write_started

                games_started = time.perf_counter()
                part_split = np.full(rows, -1, dtype=np.int32)
                expected_start = 0
                with gzip.open(item.games_path, "rt", encoding="utf-8") as handle:
                    for line in handle:
                        record = json.loads(line)
                        block_start = int(record["block_start"])
                        block_length = int(record["block_len"])
                        if block_start != expected_start or block_length < 1:
                            raise ValueError(f"non-contiguous block index: {item.games_path}")
                        expected_start += block_length
                        game_id = record["canonical_game_id"]
                        fold = assignment[game_id]
                        part_split[block_start:expected_start] = split_index[fold]
                        block_starts.append(total_rows + block_start)
                        block_lengths.append(block_length)
                        block_z.append(float(record["z"]))
                        block_seat.append(int(record["seat"]))
                        block_keys.append(f"{game_id}:s{record['seat']}")
                if expected_start != rows or np.any(part_split < 0):
                    raise ValueError(f"block index does not cover part: {item.games_path}")
                stage_seconds["games_parse_transform"] += time.perf_counter() - games_started

                meta_rows["z"].append(arrays["z"].astype(np.float32, copy=False))
                for key in ("seat", "turn", "select_context", "legal_count", "decision_index"):
                    meta_rows[key].append(arrays[key].astype(np.int32, copy=False))
                meta_rows["split_idx"].append(part_split)
                meta_rows["is_target_eligible"].append(eligible)
                total_rows += rows
                del arrays, eligible, part_split, payload, x
            x_handle.flush()
            os.fsync(x_handle.fileno())

        metadata_started = time.perf_counter()
        meta_arrays = {
            key: np.concatenate(values).astype(
                np.bool_ if key == "is_target_eligible" else np.float32 if key == "z" else np.int32,
                copy=False,
            )
            for key, values in meta_rows.items()
        }
        np.savez_compressed(meta_path, **meta_arrays)
        block_arrays = {
            "row_start": np.asarray(block_starts, dtype=np.int64),
            "row_len": np.asarray(block_lengths, dtype=np.int32),
            "z": np.asarray(block_z, dtype=np.float32),
            "seat": np.asarray(block_seat, dtype=np.int32),
        }
        np.savez_compressed(blocks_path, **block_arrays)
        if mode == "baseline":
            with gzip.open(keys_path, "wt", encoding="utf-8") as handle:
                json.dump({"block_keys": block_keys}, handle, sort_keys=True, separators=(",", ":"))
        else:
            value_data._atomic_gzip_json(keys_path, {"block_keys": block_keys})
        stage_seconds["metadata_finalize_write"] += time.perf_counter() - metadata_started

        receipt_started = time.perf_counter()
        if mode == "optimized":
            output_hashes[x_path.name] = x_writer.sha256
        paths_to_hash = (
            (x_path, meta_path, blocks_path, keys_path)
            if mode == "baseline" else (meta_path, blocks_path, keys_path)
        )
        for path in paths_to_hash:
            output_hashes[path.name] = _sha256(path)
        stage_seconds["output_receipt_hashing"] += time.perf_counter() - receipt_started

        reopen_started = time.perf_counter()
        for path in (x_path, meta_path, blocks_path, keys_path):
            if _sha256(path) != output_hashes[path.name]:
                raise ValueError(f"reopen hash mismatch: {path}")
        x_memmap = np.memmap(x_path, dtype=np.float32, mode="r", shape=(total_rows, FEATURE_DIM))
        if not np.isfinite(x_memmap[[0, total_rows - 1]]).all():
            raise ValueError("non-finite endpoint features")
        with np.load(meta_path, allow_pickle=False) as meta:
            if any(meta[key].shape != (total_rows,) for key in meta.files):
                raise ValueError("reopened metadata shape mismatch")
        with np.load(blocks_path, allow_pickle=False) as blocks:
            if int(blocks["row_len"].sum()) != total_rows:
                raise ValueError("reopened blocks do not cover rows")
        with gzip.open(keys_path, "rt", encoding="utf-8") as handle:
            if len(json.load(handle)["block_keys"]) != len(block_keys):
                raise ValueError("reopened block-key count mismatch")
        del x_memmap
        stage_seconds["independent_reopen_validate"] += time.perf_counter() - reopen_started

        output_bytes = sum(path.stat().st_size for path in (x_path, meta_path, blocks_path, keys_path))
        x_bytes = x_path.stat().st_size
        seconds = time.perf_counter() - started

        return {
            "selection": selection,
            "mode": mode,
            "parts": len(parts),
            "rows": total_rows,
            "x_bytes": x_bytes,
            "output_bytes": output_bytes,
            "source_hashed_bytes": source_bytes,
            "frozen_bound_evidence_bytes": bound_evidence_bytes,
            "seconds": seconds,
            "rows_per_second": total_rows / seconds,
            "x_mib_per_second_total": x_bytes / (1 << 20) / seconds,
            "source_hash_mib_per_second": source_bytes / (1 << 20) / stage_seconds["source_hashing"],
            "x_write_mib_per_second": x_bytes / (1 << 20) / stage_seconds["x_write"],
            "stage_seconds": stage_seconds,
            "output_hashes": output_hashes,
            "part_receipts": [
                {
                    "source_id": item.source_id,
                    "unit_id": item.unit_id,
                    "part": item.part["name"],
                    "rows": int(item.part["rows"]),
                    "action_options": int(item.part["action_options"]),
                    "npz_sha256": item.part["files_sha256"][item.npz_path.name],
                    "games_sha256": item.part["files_sha256"][item.games_path.name],
                    "transitions_sha256": item.part["files_sha256"][item.transitions_path.name],
                    "mask_sha256": item.mask_sha256,
                }
                for item in parts
            ],
        }
    finally:
        if not keep_output and run_dir.exists():
            shutil.rmtree(run_dir)


def _write_probe(directory: Path, size_bytes: int) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"phase4-write-probe-{uuid.uuid4().hex}.bin"
    block = memoryview(bytearray(8 << 20))
    remaining = size_bytes
    started = time.perf_counter()
    try:
        with open(path, "wb", buffering=8 << 20) as handle:
            while remaining:
                count = min(remaining, len(block))
                handle.write(block[:count])
                remaining -= count
            handle.flush()
            os.fsync(handle.fileno())
        write_seconds = time.perf_counter() - started
        hash_started = time.perf_counter()
        digest = _sha256(path)
        hash_seconds = time.perf_counter() - hash_started

        return {
            "directory": str(directory.resolve()),
            "bytes": size_bytes,
            "write_seconds": write_seconds,
            "write_mib_per_second": size_bytes / (1 << 20) / write_seconds,
            "reopen_hash_seconds": hash_seconds,
            "reopen_read_mib_per_second": size_bytes / (1 << 20) / hash_seconds,
            "sha256": digest,
        }
    finally:
        if path.exists():
            path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--raw-out", type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "optimized"), default="baseline")
    parser.add_argument("--write-probe-dir", type=Path, action="append", default=[])
    parser.add_argument("--write-probe-mib", type=int, default=512)
    parser.add_argument("--keep-output", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    canonical_root = args.canonical_root.resolve()
    catalog_path = args.catalog.resolve()
    split_path = args.split.resolve()
    scratch_root = args.scratch_root.resolve()
    raw_out = args.raw_out.resolve()
    scratch_root.mkdir(parents=True, exist_ok=True)
    catalog, split, exclusion, exclusion_path, selections = _load_inputs(
        canonical_root,
        catalog_path,
        split_path,
    )
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with ResourceSampler() as sampler:
        audit_databases = _profile_audit_databases(
            exclusion,
            exclusion_path,
            rehash=args.mode == "baseline",
        )
        selection_results = [
            _profile_selection(
                name,
                parts,
                split["assignment"],
                scratch_root,
                mode=args.mode,
                keep_output=args.keep_output,
            )
            for name, parts in selections.items()
        ]
        write_probes = [
            _write_probe(directory.resolve(), args.write_probe_mib << 20)
            for directory in args.write_probe_dir
        ]
    profile = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "mode": args.mode,
        "started_at_utc": started_utc,
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": [os.fspath(Path(os.sys.executable)), *os.sys.argv],
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": psutil.cpu_count(logical=True),
            "physical_cpus": psutil.cpu_count(logical=False),
            "physical_memory_bytes": psutil.virtual_memory().total,
        },
        "bindings": {
            "canonical_root": str(canonical_root),
            "catalog_path": str(catalog_path),
            "catalog_sha256": _sha256(catalog_path),
            "split_path": str(split_path),
            "split_sha256": _sha256(split_path),
            "row_exclusion_manifest_path": str(exclusion_path),
            "row_exclusion_manifest_sha256": _sha256(exclusion_path),
            "catalog_rows": int(catalog["totals"]["rows_kept"]),
            "catalog_games": int(catalog["totals"]["games_kept"]),
            "eligible_target_rows": int(exclusion["eligible_target_rows"]),
            "excluded_target_rows": int(exclusion["excluded_target_rows"]),
        },
        "audit_database_hashing": audit_databases,
        "selections": selection_results,
        "write_probes": write_probes,
        "resources": sampler.summary(),
        "raw_samples": sampler.samples,
    }
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    raw_out.write_text(json.dumps(profile, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "raw_out": str(raw_out),
        "audit_database_seconds": audit_databases["seconds"],
        "selections": [
            {
                "selection": item["selection"],
                "rows": item["rows"],
                "seconds": item["seconds"],
                "rows_per_second": item["rows_per_second"],
            }
            for item in selection_results
        ],
        "write_probes": write_probes,
        "resource_summary": profile["resources"],
    }, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
