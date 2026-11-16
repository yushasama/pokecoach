"""Atomically build or independently verify the split-bound value-lab cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import deepmew.value_lab.data as value_data  # noqa: E402
from deepmew.artifacts.manifest import build_manifest, write_manifest  # noqa: E402
from deepmew.artifacts.run_dir import make_run_id  # noqa: E402
from deepmew.artifacts.writers import write_json  # noqa: E402
from deepmew.value_lab.data import LabData, prepare_cache  # noqa: E402


TRANSFER_SCHEMA_VERSION = "deepmew.value_q_cache_transfer.v1"
COPY_CHUNK_BYTES = 1 << 24
# Reserve room beyond the cache estimate for validation receipts and a first checkpoint.
MIN_SAFETY_MARGIN_BYTES = 1 << 30
# Raw metadata is 33 bytes/row today; 48 leaves headroom without depending on compression.
METADATA_BYTES_PER_ROW_ESTIMATE = 48
FIXED_CACHE_OVERHEAD_BYTES = 1 << 26


def _sha256(path: Path, chunk_size: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)

    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    build = subparsers.add_parser("build", help="Build into a sibling temp dir, verify, then rename atomically.")
    build.add_argument("--canonical-root", required=True)
    build.add_argument("--catalog", required=True)
    build.add_argument("--split", required=True)
    build.add_argument("--out", required=True)
    build.add_argument(
        "--temp-root",
        help="Optional existing-volume build root. Cross-volume builds are verified and copied into final-volume staging.",
    )
    verify = subparsers.add_parser("verify", help="Fully re-hash and validate an existing cache.")
    verify.add_argument("--cache", required=True)
    verify.add_argument("--split", required=True)

    return parser.parse_args()


def validate_cache(cache_dir: Path, split_path: Path) -> dict:
    data = LabData(str(cache_dir), split_manifest_path=str(split_path))

    return data.manifest


def _receipt(
    action: str,
    cache_dir: Path,
    manifest: dict,
    *,
    skipped: bool,
    elapsed_seconds: float | None = None,
    preflight: dict | None = None,
    transfer: dict | None = None,
) -> dict:
    manifest_path = cache_dir / "cache_manifest.json"
    receipt = {
        "action": action,
        "skipped": skipped,
        "cache": str(cache_dir),
        "cache_manifest_sha256": _sha256(manifest_path),
        "schema_version": manifest["schema_version"],
        "status": manifest["status"],
        "rows": manifest["rows"],
        "eligible_target_rows": manifest["eligible_target_rows"],
        "excluded_target_rows": manifest["excluded_target_rows"],
        "games": manifest["games"],
        "blocks": manifest["blocks"],
        "legal_options": manifest.get("legal_options"),
        "split_manifest_sha256": manifest["split_manifest_sha256"],
        "output_files": manifest["output_files"],
        "cache_build_metrics": manifest.get("build_metrics"),
    }
    if elapsed_seconds is not None:
        receipt["elapsed_seconds"] = elapsed_seconds
    if preflight is not None:
        receipt["disk_preflight"] = preflight
    if transfer is not None:
        receipt["transfer"] = transfer

    return receipt


def _nearest_existing_path(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise FileNotFoundError(f"no existing ancestor for disk-space probe: {path}")
        candidate = parent

    return candidate


def _available_bytes(path: Path) -> int:
    return int(shutil.disk_usage(_nearest_existing_path(path)).free)


def _same_volume(left: Path, right: Path) -> bool:
    left_resolved = left.resolve()
    right_resolved = right.resolve()
    return os.stat(_nearest_existing_path(left_resolved)).st_dev == os.stat(
        _nearest_existing_path(right_resolved)
    ).st_dev


def _estimate_cache_bytes(catalog_path: Path) -> dict[str, int]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    rows = int((catalog.get("totals") or {}).get("rows_kept", -1))
    if rows < 1:
        raise ValueError("catalog totals.rows_kept must be positive for disk preflight")
    feature_dim = int(value_data.FEATURE_DIM)
    feature_bytes = rows * feature_dim * 4
    metadata_bytes = rows * METADATA_BYTES_PER_ROW_ESTIMATE + FIXED_CACHE_OVERHEAD_BYTES
    estimated_cache_bytes = feature_bytes + metadata_bytes
    safety_margin_bytes = max(MIN_SAFETY_MARGIN_BYTES, estimated_cache_bytes // 5)

    return {
        "rows": rows,
        "feature_dim": feature_dim,
        "feature_bytes": feature_bytes,
        "metadata_and_receipt_estimate_bytes": metadata_bytes,
        "estimated_cache_bytes": estimated_cache_bytes,
        "safety_margin_bytes": safety_margin_bytes,
        "required_bytes_per_volume": estimated_cache_bytes + safety_margin_bytes,
    }


def _preflight_disk_space(
    catalog_path: Path,
    build_parent: Path,
    destination_parent: Path,
    *,
    same_volume: bool,
) -> dict:
    estimate = _estimate_cache_bytes(catalog_path)
    required = estimate["required_bytes_per_volume"]
    build_free = _available_bytes(build_parent)
    checks = [
        {
            "role": "build_and_destination" if same_volume else "build",
            "path": str(build_parent),
            "free_bytes": build_free,
            "required_bytes": required,
            "passed": build_free >= required,
        }
    ]
    if not same_volume:
        destination_free = _available_bytes(destination_parent)
        checks.append(
            {
                "role": "destination",
                "path": str(destination_parent),
                "free_bytes": destination_free,
                "required_bytes": required,
                "passed": destination_free >= required,
            }
        )
    failed = [check for check in checks if not check["passed"]]
    if failed:
        detail = "; ".join(
            f"{check['role']} free={check['free_bytes']} required={check['required_bytes']} path={check['path']}"
            for check in failed
        )
        raise OSError(f"insufficient disk space for value cache: {detail}")

    return {
        "same_volume": same_volume,
        **estimate,
        "checks": checks,
    }


def _file_inventory(root: Path) -> list[Path]:
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    symlinks = [path for path in paths if path.is_symlink()]
    if symlinks:
        raise RuntimeError(f"refusing to transfer cache symlinks: {symlinks[0]}")

    return paths


def _cache_file_summary(cache_dir: Path) -> tuple[int, int]:
    files = _file_inventory(cache_dir)

    return len(files), sum(path.stat().st_size for path in files)


def _copy_and_verify_tree(source: Path, destination: Path) -> dict[str, dict[str, int | str]]:
    copied: dict[str, dict[str, int | str]] = {}
    for source_path in _file_inventory(source):
        relative = source_path.relative_to(source)
        destination_path = destination / relative
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        source_digest = hashlib.sha256()
        byte_count = 0
        with open(source_path, "rb") as source_handle, open(destination_path, "xb") as destination_handle:
            while block := source_handle.read(COPY_CHUNK_BYTES):
                source_digest.update(block)
                destination_handle.write(block)
                byte_count += len(block)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        source_sha256 = source_digest.hexdigest()
        if destination_path.stat().st_size != byte_count:
            raise IOError(f"copied cache file verification failed: {relative.as_posix()}")
        copied[relative.as_posix()] = {
            "bytes": byte_count,
            "sha256": source_sha256,
        }

    return copied


def _remove_published_build_temp(temp_dir: Path, build_parent: Path, out_name: str) -> None:
    resolved_temp = temp_dir.resolve()
    resolved_parent = build_parent.resolve()
    if resolved_temp.parent != resolved_parent or not resolved_temp.name.startswith(f".{out_name}.building."):
        raise RuntimeError(f"refusing to remove unrecognized build temp: {resolved_temp}")
    shutil.rmtree(resolved_temp)


def _run_manifest(
    *,
    canonical_root: Path,
    catalog_path: Path,
    split_path: Path,
    manifest: dict,
    command: list[str],
    cache_manifest_sha: str,
    elapsed_seconds: float,
    preflight: dict,
    transfer: dict,
    transfer_receipt_sha256: str | None,
) -> dict:
    outputs = {
        "cache_manifest": "cache_manifest.json",
        "cache_manifest_sha256": cache_manifest_sha,
        "cache_files": manifest["output_files"],
    }
    if transfer_receipt_sha256 is not None:
        outputs.update(
            {
                "transfer_receipt": "transfer_receipt.json",
                "transfer_receipt_sha256": transfer_receipt_sha256,
            }
        )

    return build_manifest(
        run_id=make_run_id("value_q_cache"),
        kind="value_q_cache",
        command=command,
        inputs={
            "canonical_root": str(canonical_root),
            "catalog_path": str(catalog_path),
            "catalog_sha256": manifest["catalog_sha256"],
            "split_manifest_path": str(split_path),
            "split_manifest_sha256": manifest["split_manifest_sha256"],
            "row_exclusion_manifest_sha256": manifest["row_exclusion_manifest_sha256"],
            "data_code_sha256": _sha256(Path(value_data.__file__)),
            "cli_code_sha256": _sha256(Path(__file__)),
        },
        outputs=outputs,
        summary={
            "rows": manifest["rows"],
            "eligible_target_rows": manifest["eligible_target_rows"],
            "excluded_target_rows": manifest["excluded_target_rows"],
            "games": manifest["games"],
            "blocks": manifest["blocks"],
            "feature_dim": manifest["feature_dim"],
            "feature_dtype": manifest["feature_dtype"],
            "legal_options": manifest.get("legal_options"),
            "cache_build_metrics": manifest.get("build_metrics"),
            "elapsed_seconds_before_publish": elapsed_seconds,
            "disk_preflight": preflight,
            "transfer": transfer,
        },
    )


def verify_cache(cache_dir: Path, split_path: Path) -> dict:
    cache_dir = cache_dir.resolve()
    split_path = split_path.resolve()
    manifest_path = cache_dir / "cache_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"completed cache manifest not found: {manifest_path}")
    manifest = validate_cache(cache_dir, split_path)

    return _receipt("verified", cache_dir, manifest, skipped=False)


def build_cache(
    canonical_root: Path,
    catalog_path: Path,
    split_path: Path,
    out_dir: Path,
    *,
    command: list[str],
    temp_root: Path | None = None,
) -> dict:
    started = time.perf_counter()
    canonical_root = canonical_root.resolve()
    catalog_path = catalog_path.resolve()
    split_path = split_path.resolve()
    out_dir = out_dir.resolve()
    manifest_path = out_dir / "cache_manifest.json"
    if manifest_path.is_file():
        manifest = validate_cache(out_dir, split_path)
        if Path(manifest["canonical_root"]).resolve() != canonical_root:
            raise ValueError("existing cache canonical-root binding differs from the requested root")
        if Path(manifest["catalog_path"]).resolve() != catalog_path:
            raise ValueError("existing cache catalog binding differs from the requested catalog")

        return _receipt("reused_verified", out_dir, manifest, skipped=True)
    if out_dir.exists():
        names = sorted(path.name for path in out_dir.iterdir())
        preview = ", ".join(names[:10]) if names else "empty directory"
        raise RuntimeError(
            "refusing to replace an incomplete cache destination; preserve or relocate it first: "
            f"{out_dir} ({preview})"
        )
    build_parent = temp_root.resolve() if temp_root is not None else out_dir.parent
    if build_parent == out_dir or out_dir in build_parent.parents:
        raise ValueError("--temp-root must not be the cache destination or a child of it")
    same_volume = _same_volume(build_parent, out_dir.parent)
    preflight = _preflight_disk_space(
        catalog_path,
        build_parent,
        out_dir.parent,
        same_volume=same_volume,
    )
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    build_parent.mkdir(parents=True, exist_ok=True)
    temp_dir = build_parent / f".{out_dir.name}.building.{uuid.uuid4().hex}"
    temp_dir.mkdir()
    manifest = prepare_cache(
        str(canonical_root),
        str(temp_dir),
        catalog_path=str(catalog_path),
        split_manifest_path=str(split_path),
    )
    manifest = validate_cache(temp_dir, split_path)
    cache_manifest_sha = _sha256(temp_dir / "cache_manifest.json")
    if same_volume:
        file_count, byte_count = _cache_file_summary(temp_dir)
        transfer = {
            "mode": "same_volume_atomic_rename",
            "source": str(temp_dir),
            "destination": str(out_dir),
            "copied_file_count": file_count,
            "copied_bytes": byte_count,
            "verification": "independent_cache_validation_before_publish",
        }
        elapsed_before_publish = time.perf_counter() - started
        run_manifest = _run_manifest(
            canonical_root=canonical_root,
            catalog_path=catalog_path,
            split_path=split_path,
            manifest=manifest,
            command=command,
            cache_manifest_sha=cache_manifest_sha,
            elapsed_seconds=elapsed_before_publish,
            preflight=preflight,
            transfer=transfer,
            transfer_receipt_sha256=None,
        )
        write_manifest(temp_dir / "build_run_manifest.json", run_manifest)
        os.replace(temp_dir, out_dir)
    else:
        staging_dir = out_dir.with_name(f".{out_dir.name}.transfer.{uuid.uuid4().hex}")
        staging_dir.mkdir()
        transfer_started = time.perf_counter()
        copied_files = _copy_and_verify_tree(temp_dir, staging_dir)
        transferred_manifest = validate_cache(staging_dir, split_path)
        if transferred_manifest != manifest:
            raise ValueError("cross-volume cache manifest changed during transfer")
        expected_copied_hashes = {
            "cache_manifest.json": cache_manifest_sha,
            **{
                filename: receipt["sha256"]
                for filename, receipt in manifest["output_files"].items()
            },
        }
        if set(copied_files) != set(expected_copied_hashes):
            raise ValueError("cross-volume copied-file inventory is incomplete")
        if any(
            copied_files[filename]["sha256"] != expected_sha
            for filename, expected_sha in expected_copied_hashes.items()
        ):
            raise ValueError("cross-volume source receipt differs from accepted cache receipts")
        if _sha256(staging_dir / "cache_manifest.json") != cache_manifest_sha:
            raise ValueError("cross-volume cache manifest byte hash mismatch")
        transfer_elapsed = time.perf_counter() - transfer_started
        transfer = {
            "schema_version": TRANSFER_SCHEMA_VERSION,
            "status": "completed",
            "mode": "cross_volume_sequential_copy",
            "source": str(temp_dir),
            "destination_staging": str(staging_dir),
            "destination": str(out_dir),
            "copied_file_count": len(copied_files),
            "copied_bytes": sum(int(item["bytes"]) for item in copied_files.values()),
            "elapsed_seconds": transfer_elapsed,
            "verification": (
                "streamed_source_sha256_bound_to_cache_receipts_then_"
                "independent_destination_cache_validation"
            ),
            "copied_files": copied_files,
        }
        transfer_receipt_path = staging_dir / "transfer_receipt.json"
        write_json(transfer_receipt_path, transfer)
        transfer_receipt_sha = _sha256(transfer_receipt_path)
        elapsed_before_publish = time.perf_counter() - started
        run_manifest = _run_manifest(
            canonical_root=canonical_root,
            catalog_path=catalog_path,
            split_path=split_path,
            manifest=manifest,
            command=command,
            cache_manifest_sha=cache_manifest_sha,
            elapsed_seconds=elapsed_before_publish,
            preflight=preflight,
            transfer=transfer,
            transfer_receipt_sha256=transfer_receipt_sha,
        )
        write_manifest(staging_dir / "build_run_manifest.json", run_manifest)
        os.replace(staging_dir, out_dir)
        _remove_published_build_temp(temp_dir, build_parent, out_dir.name)

    elapsed = time.perf_counter() - started

    return _receipt(
        "built_and_verified",
        out_dir,
        manifest,
        skipped=False,
        elapsed_seconds=elapsed,
        preflight=preflight,
        transfer=transfer,
    )


def main() -> int:
    args = parse_args()
    if args.action == "verify":
        receipt = verify_cache(Path(args.cache), Path(args.split))
    else:
        receipt = build_cache(
            Path(args.canonical_root),
            Path(args.catalog),
            Path(args.split),
            Path(args.out),
            command=[sys.executable, *sys.argv],
            temp_root=Path(args.temp_root) if args.temp_root else None,
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
