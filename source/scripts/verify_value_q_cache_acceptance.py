"""Accept a completed value cache with source parity and loader benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import psutil


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepmew.artifacts.writers import write_json  # noqa: E402
from deepmew.value_lab.data import LabData, X_FILENAME  # noqa: E402
from deepmew.value_lab.mc_screen import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_SEED,
    eligible_row_stream,
    row_stream_sha256,
)


SCHEMA_VERSION = "deepmew.value_q_cache_acceptance.v1"


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


def _sample_part_indices(part_count: int, sample_parts: int, seed: int) -> np.ndarray:
    required = {0, part_count // 2, part_count - 1}
    rng = np.random.default_rng(seed)
    remaining = min(part_count, max(sample_parts, len(required))) - len(required)
    candidates = np.asarray(sorted(set(range(part_count)) - required), dtype=np.int64)
    if remaining > 0:
        required.update(int(value) for value in rng.choice(candidates, size=remaining, replace=False))

    return np.asarray(sorted(required), dtype=np.int64)


def _source_parity(
    data: LabData,
    cache_manifest: dict,
    catalog_path: Path,
    split_path: Path,
    *,
    sample_parts: int,
    seed: int,
) -> dict:
    catalog = _read_json(catalog_path)
    units = {
        (unit["source_id"], unit["unit_id"]): catalog_path.parent / unit["manifest_path"]
        for unit in catalog["included_units"]
    }
    split = _read_json(split_path)
    exclusion_path = _resolve(split["row_exclusion_manifest_path"], split_path.parent)
    exclusions = _read_json(exclusion_path)
    masks = {
        (receipt["source_id"], receipt["unit_id"], receipt["part"]): receipt
        for receipt in exclusions["parts"]
    }
    inputs = cache_manifest["inputs"]
    starts = np.zeros(len(inputs), dtype=np.int64)
    if len(inputs) > 1:
        np.cumsum(
            np.asarray([int(item["rows"]) for item in inputs[:-1]], dtype=np.int64),
            out=starts[1:],
        )
    sampled_part_indices = _sample_part_indices(len(inputs), sample_parts, seed)
    rows_checked = 0
    parts_checked = []
    fields = ("z", "seat", "turn", "select_context", "legal_count", "decision_index")
    started = time.perf_counter()
    for part_index in sampled_part_indices:
        receipt = inputs[int(part_index)]
        identity = (receipt["source_id"], receipt["unit_id"])
        manifest_path = units[identity]
        part_name = receipt["part"]
        npz_path = manifest_path.parent / f"{part_name}.npz"
        mask_receipt = masks[(*identity, part_name)]
        mask_path = _resolve(mask_receipt["mask_path"], exclusion_path.parent)
        with np.load(npz_path, allow_pickle=False) as source:
            rows = int(receipt["rows"])
            local_rows = np.asarray(sorted({0, rows // 2, rows - 1}), dtype=np.int64)
            global_rows = starts[int(part_index)] + local_rows
            if not np.array_equal(data.X[global_rows], source["X"][local_rows]):
                raise ValueError(f"exact float32 source parity failed: {npz_path}")
            for field in fields:
                if not np.array_equal(data.meta[field][global_rows], source[field][local_rows]):
                    raise ValueError(f"metadata source parity failed for {field}: {npz_path}")
        with np.load(mask_path, allow_pickle=False) as mask:
            if not np.array_equal(
                data.meta["is_target_eligible"][global_rows],
                mask["is_target_eligible"][local_rows],
            ):
                raise ValueError(f"target-mask source parity failed: {mask_path}")
        rows_checked += len(local_rows)
        parts_checked.append({
            "part_index": int(part_index),
            "source_id": identity[0],
            "unit_id": identity[1],
            "part": part_name,
            "rows": rows,
            "sampled_local_rows": local_rows.tolist(),
            "npz_sha256": receipt["npz_sha256"],
            "mask_sha256": receipt["target_mask_sha256"],
        })

    return {
        "status": "pass",
        "seconds": time.perf_counter() - started,
        "parts_checked": len(parts_checked),
        "rows_checked": rows_checked,
        "fields_checked": ["X.float32_exact", *fields, "is_target_eligible"],
        "samples": parts_checked,
    }


def _sequential_loader(cache_dir: Path, *, chunk_mib: int) -> dict:
    path = cache_dir / X_FILENAME
    buffer = bytearray(chunk_mib << 20)
    view = memoryview(buffer)
    total = 0
    checksum = 0
    started = time.perf_counter()
    with open(path, "rb", buffering=chunk_mib << 20) as handle:
        while count := handle.readinto(buffer):
            total += count
            checksum = (checksum + view[0] + view[count - 1]) & 0xFFFFFFFF
    seconds = time.perf_counter() - started
    if total != path.stat().st_size:
        raise ValueError("sequential loader byte count mismatch")

    return {
        "status": "pass",
        "bytes": total,
        "seconds": seconds,
        "mib_per_second": total / (1 << 20) / seconds,
        "rows_per_second": (total // (144 * 4)) / seconds,
        "chunk_mib": chunk_mib,
        "endpoint_checksum": checksum,
    }


def _randomized_loader(
    data: LabData,
    *,
    presentations: int,
    batch_size: int,
    seed: int,
) -> dict:
    stream_started = time.perf_counter()
    rows = eligible_row_stream(data, split="train", presentations=presentations, seed=seed)
    stream_seconds = time.perf_counter() - stream_started
    batch_started = time.perf_counter()
    checksum = 0.0
    nonfinite = 0
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        features = np.asarray(data.X[batch_rows], dtype=np.float32)
        nonfinite += int((~np.isfinite(features)).sum())
        checksum += float(features.sum(dtype=np.float64))
    batch_seconds = time.perf_counter() - batch_started
    if nonfinite:
        raise ValueError(f"randomized loader produced {nonfinite} non-finite values")
    feature_bytes = presentations * data.X.shape[1] * np.dtype(np.float32).itemsize

    return {
        "status": "pass",
        "presentations": presentations,
        "batch_size": batch_size,
        "seed": seed,
        "row_stream_sha256": row_stream_sha256(rows),
        "row_stream_seconds": stream_seconds,
        "batch_load_seconds": batch_seconds,
        "rows_per_second": presentations / batch_seconds,
        "logical_feature_mib_per_second": feature_bytes / (1 << 20) / batch_seconds,
        "nonfinite_values": nonfinite,
        "feature_sum_float64": checksum,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-parts", type=int, default=24)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--presentations", type=int, default=65_536)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--sequential-chunk-mib", type=int, default=16)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_parts < 3 or args.presentations < 1 or args.batch_size < 1:
        raise ValueError("sample-parts, presentations, and batch-size are invalid")
    cache_dir = args.cache.resolve()
    catalog_path = args.catalog.resolve()
    split_path = args.split.resolve()
    out_path = args.out.resolve()
    process = psutil.Process()
    start_memory = process.memory_info()
    started = time.perf_counter()
    reopen_started = time.perf_counter()
    data = LabData(str(cache_dir), split_manifest_path=str(split_path))
    reopen_seconds = time.perf_counter() - reopen_started
    manifest_path = cache_dir / "cache_manifest.json"
    source_parity = _source_parity(
        data,
        data.manifest,
        catalog_path,
        split_path,
        sample_parts=args.sample_parts,
        seed=args.seed,
    )
    sequential = _sequential_loader(cache_dir, chunk_mib=args.sequential_chunk_mib)
    randomized = _randomized_loader(
        data,
        presentations=args.presentations,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    end_memory = process.memory_info()
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "accepted",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": [sys.executable, *sys.argv],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "bindings": {
            "cache": str(cache_dir),
            "cache_manifest_sha256": _sha256(manifest_path),
            "catalog_sha256": _sha256(catalog_path),
            "split_sha256": _sha256(split_path),
            "rows": int(data.manifest["rows"]),
            "games": int(data.manifest["games"]),
            "eligible_target_rows": int(data.manifest["eligible_target_rows"]),
            "excluded_target_rows": int(data.manifest["excluded_target_rows"]),
            "legal_options": int(data.manifest["legal_options"]),
            "output_files": data.manifest["output_files"],
        },
        "full_reopen_validation": {
            "status": "pass",
            "seconds": reopen_seconds,
            "contract": "LabData full output hash, split, shape, boundary, mask, and legal-offset validation",
        },
        "source_parity": source_parity,
        "sequential_loader": sequential,
        "randomized_loader": randomized,
        "process": {
            "rss_bytes_start": int(start_memory.rss),
            "rss_bytes_end": int(end_memory.rss),
            "private_bytes_start": int(getattr(start_memory, "private", 0)),
            "private_bytes_end": int(getattr(end_memory, "private", 0)),
            "page_faults_delta": int(
                getattr(end_memory, "num_page_faults", 0)
                - getattr(start_memory, "num_page_faults", 0)
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(out_path, receipt)
    print(json.dumps({
        "status": receipt["status"],
        "out": str(out_path),
        "elapsed_seconds": receipt["elapsed_seconds"],
        "reopen_seconds": reopen_seconds,
        "source_parity": source_parity,
        "sequential_loader": sequential,
        "randomized_loader": randomized,
    }, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
