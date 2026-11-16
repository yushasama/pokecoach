"""Build an exact tri-state control-pass sidecar from canonical sequence rows."""

from __future__ import annotations

import gzip
import io
import json
import os
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

import deepmew.value_lab.data as value_data
from deepmew.artifacts.manifest import build_manifest, read_manifest, write_manifest
from deepmew.artifacts.writers import write_json
from deepmew.value_lab.crossfit import (
    _HashingReader,
    _atomic_gzip_json,
    _atomic_npz,
    _file_receipts,
    _sha256,
    _stable_hash,
    _verify_output_files,
)
from deepmew.value_lab.data import LabData


CONTROL_PASS_MASK_SCHEMA_VERSION = "deepmew.value_control_pass_mask.v1"


def _control_pass_value(record: dict[str, Any]) -> tuple[bool, bool]:
    value = record.get("control_passed")
    if value is None:
        return False, False
    if not isinstance(value, bool):
        raise ValueError("canonical control_passed must be boolean or null")

    return True, value


def _count_receipt(known: np.ndarray, passed: np.ndarray) -> dict[str, Any]:
    rows = int(len(known))
    known_rows = int(known.sum())
    passed_rows = int((known & passed).sum())

    return {
        "rows": rows,
        "known_rows": known_rows,
        "unknown_rows": rows - known_rows,
        "passed_rows": passed_rows,
        "retained_control_rows": known_rows - passed_rows,
        "known_fraction": known_rows / rows if rows else 0.0,
        "passed_fraction_of_known": passed_rows / known_rows if known_rows else None,
    }


def _group_receipts(
    known: np.ndarray,
    passed: np.ndarray,
    keys: np.ndarray,
    names: list[str],
) -> dict[str, dict[str, Any]]:
    result = {}
    for raw_key in np.unique(keys):
        index = int(raw_key)
        name = names[index] if 0 <= index < len(names) else str(index)
        mask = keys == raw_key
        result[name] = _count_receipt(known[mask], passed[mask])

    return result


def run_control_pass_mask_build(
    *,
    cache_dir: Path,
    split_path: Path,
    out_dir: Path,
    command: list[str] | None = None,
) -> dict[str, Any]:
    cache_dir = cache_dir.resolve()
    split_path = split_path.resolve()
    out_dir = out_dir.resolve()
    cache_manifest_path = cache_dir / "cache_manifest.json"
    cache_manifest_sha256 = _sha256(cache_manifest_path)
    split_manifest_sha256 = _sha256(split_path)
    config = {
        "schema_version": CONTROL_PASS_MASK_SCHEMA_VERSION,
        "cache_path": str(cache_dir),
        "cache_manifest_sha256": cache_manifest_sha256,
        "split_path": str(split_path),
        "split_manifest_sha256": split_manifest_sha256,
        "source_field": "canonical sequence row control_passed",
        "unknown_semantics": "preserved_null_never_filled",
        "code_sha256": {
            "control_pass_mask": _sha256(Path(__file__)),
            "data": _sha256(Path(value_data.__file__)),
        },
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "test_targets_consumed": False,
    }
    config["sha256"] = _stable_hash(config)
    if out_dir.exists():
        manifest_path = out_dir / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"refusing incomplete control-pass directory: {out_dir}")
        manifest = read_manifest(manifest_path)
        if manifest.get("inputs", {}).get("config_sha256") != config["sha256"]:
            raise ValueError("existing control-pass configuration differs")
        _verify_output_files(out_dir, manifest["outputs"]["files"])

        return {"action": "reused_verified", "manifest": manifest}
    data = LabData(
        str(cache_dir),
        split_manifest_path=str(split_path),
        load_labels=False,
    )
    cache_manifest = data.manifest
    inputs = cache_manifest.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("accepted cache input inventory is missing")
    rows = int(cache_manifest["rows"])
    known = np.zeros(rows, dtype=np.bool_)
    passed = np.zeros(rows, dtype=np.bool_)
    canonical_root = Path(cache_manifest["canonical_root"])
    sequence_receipts = []
    global_offset = 0
    started = time.perf_counter()
    for part in inputs:
        local_rows = int(part["rows"])
        path = (
            canonical_root
            / part["source_id"]
            / part["unit_id"]
            / f"{part['part']}.sequences.jsonl.gz"
        )
        if not path.is_file():
            raise ValueError(f"canonical sequence sidecar is missing: {path}")
        with open(path, "rb") as raw_handle:
            hashing_reader = _HashingReader(raw_handle)
            with gzip.GzipFile(fileobj=hashing_reader, mode="rb") as zipped:
                with io.TextIOWrapper(zipped, encoding="utf-8") as text_handle:
                    observed = 0
                    for line in text_handle:
                        if observed >= local_rows:
                            raise ValueError(f"sequence sidecar has too many rows: {path}")
                        record = json.loads(line)
                        if int(record.get("row", -1)) != observed:
                            raise ValueError(f"sequence sidecar row alignment mismatch: {path}")
                        is_known, did_pass = _control_pass_value(record)
                        global_row = global_offset + observed
                        known[global_row] = is_known
                        passed[global_row] = did_pass
                        observed += 1
        if observed != local_rows:
            raise ValueError(f"sequence sidecar row count mismatch: {path}")
        observed_sha256 = hashing_reader.digest.hexdigest()
        if observed_sha256 != part.get("sequences_sha256"):
            raise ValueError(f"sequence sidecar hash mismatch: {path}")
        sequence_receipts.append({
            "source_id": part["source_id"],
            "unit_id": part["unit_id"],
            "part": part["part"],
            "rows": local_rows,
            "path": str(path.resolve()),
            "sha256": observed_sha256,
        })
        global_offset += local_rows
    if global_offset != rows or np.any(passed & ~known):
        raise ValueError("control-pass sidecar coverage is inconsistent")
    split_keys = data.meta["split_idx"].astype(np.int32, copy=False)
    source_keys = data.meta["source_idx"].astype(np.int32, copy=False)
    result = {
        "schema_version": CONTROL_PASS_MASK_SCHEMA_VERSION,
        "status": "completed_tri_state",
        "cache_manifest_sha256": cache_manifest_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "rows": rows,
        "parts": len(sequence_receipts),
        "elapsed_seconds": time.perf_counter() - started,
        "overall": _count_receipt(known, passed),
        "by_split": _group_receipts(known, passed, split_keys, data.split_names),
        "by_source": _group_receipts(known, passed, source_keys, data.source_ids),
        "unknown_rows_are_never_filled": True,
        "semantic_caveat": (
            "mask exactly reproduces the accepted nullable canonical field; it does not "
            "reinterpret malformed post-observation ownership"
        ),
        "test_targets_consumed": False,
        "protected_runtime_surfaces_changed": False,
    }
    temp_dir = out_dir.with_name(f".{out_dir.name}.running.{uuid.uuid4().hex}")
    temp_dir.mkdir(parents=True)
    output_paths = []
    config_path = temp_dir / "config.json"
    write_json(config_path, config)
    output_paths.append(config_path)
    mask_path = temp_dir / "control_pass_mask.npz"
    _atomic_npz(
        mask_path,
        row_id=np.arange(rows, dtype=np.int64),
        is_control_pass_known=known,
        control_passed=passed,
    )
    output_paths.append(mask_path)
    result["mask_sha256"] = _sha256(mask_path)
    sequence_path = temp_dir / "sequence_receipts.json.gz"
    _atomic_gzip_json(sequence_path, {"parts": sequence_receipts})
    output_paths.append(sequence_path)
    result["sequence_receipts_sha256"] = _sha256(sequence_path)
    result_path = temp_dir / "result.json"
    write_json(result_path, result)
    output_paths.append(result_path)
    files = _file_receipts(temp_dir, output_paths)
    manifest = build_manifest(
        run_id=out_dir.name,
        kind="value_control_pass_mask",
        command=command or [],
        inputs={"config": config, "config_sha256": config["sha256"]},
        outputs={"files": files},
        summary={
            "status": "completed_tri_state",
            "rows": rows,
            "known_rows": result["overall"]["known_rows"],
            "unknown_rows": result["overall"]["unknown_rows"],
            "passed_rows": result["overall"]["passed_rows"],
            "mask_sha256": result["mask_sha256"],
            "test_targets_consumed": False,
            "protected_runtime_surfaces_changed": False,
        },
    )
    write_manifest(temp_dir / "manifest.json", manifest)
    os.replace(temp_dir, out_dir)

    return {"action": "completed", "manifest": manifest, "result": result}
