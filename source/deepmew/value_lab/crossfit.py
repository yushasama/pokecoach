"""Grouped game-level cross-fit production for the selected snapshot value model."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import platform
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import deepmew.value_lab.contracts as value_contracts
import deepmew.value_lab.data as value_data
import deepmew.value_lab.mc_screen as value_mc_screen
import deepmew.value_lab.models as value_models
from deepmew.artifacts.manifest import build_manifest, read_manifest, write_manifest
from deepmew.artifacts.writers import write_json
from deepmew.replay.canonical_actor_dataset import S1_SLICE
from deepmew.value_lab.contracts import (
    STATE_ACTION_SIGNATURE_VERSION,
    canonical_state_action_signature,
)
from deepmew.value_lab.data import LabData
from deepmew.value_lab.feature_contract import representation_manifest
from deepmew.value_lab.mc_screen import (
    SnapshotSpec,
    _atomic_torch_save,
    _json_safe,
    _output_receipt,
    _predict,
    _sha256,
    _stable_hash,
    canonical_game_indices,
    row_stream_receipt,
    row_stream_sha256,
    train_snapshot,
)
from deepmew.value_lab.models import parameter_count


CROSSFIT_SCHEMA_VERSION = "deepmew.value_grouped_crossfit.v1"
FOLD_ASSIGNMENT_SCHEMA_VERSION = "deepmew.value_oof_fold_assignment.v1"
INNER_AUDIT_SCHEMA_VERSION = "deepmew.value_inner_signature_audit.v1"
FOLD_UNIT_SCHEMA_VERSION = "deepmew.value_crossfit_fold_unit.v1"
OOF_PREDICTION_SCHEMA_VERSION = "deepmew.value_oof_predictions.v1"
CROSSFIT_SALT = "deepmew.value_v0_grouped_crossfit.v1"
CROSSFIT_FOLDS = 5
FINAL_SEED = 915_002
DEFAULT_BATCH_SIZE = 4_096
DEFAULT_THREADS = 8
SNAPSHOT_SPEC = SnapshotSpec(
    "s1_v0_snapshot",
    S1_SLICE[0],
    S1_SLICE[1],
    "default",
    "grouped_crossfit_selected_snapshot",
)


class _HashingReader:
    """Read-only wrapper that hashes the exact compressed bytes consumed by gzip."""

    def __init__(self, handle):
        self.handle = handle
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        payload = self.handle.read(size)
        self.digest.update(payload)
        self.bytes_read += len(payload)

        return payload

    def tell(self) -> int:
        return self.handle.tell()


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with open(temporary, "wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _atomic_gzip_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with open(temporary, "wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8") as text_handle:
                json.dump(value, text_handle, sort_keys=True, separators=(",", ":"))
    os.replace(temporary, path)


def _file_receipts(root: Path, paths: list[Path]) -> list[dict[str, Any]]:
    return [_output_receipt(path, root) for path in sorted(paths)]


def _verify_output_files(root: Path, files: list[dict[str, Any]]) -> None:
    if not isinstance(files, list) or not files:
        raise ValueError(f"output inventory is missing: {root}")
    seen = set()
    for receipt in files:
        relative = receipt.get("path")
        if not isinstance(relative, str) or relative in seen:
            raise ValueError(f"output inventory identity is invalid: {root}")
        seen.add(relative)
        path = root / relative
        if not path.is_file() or path.stat().st_size != int(receipt["bytes"]):
            raise ValueError(f"output missing or size-mismatched: {path}")
        if _sha256(path) != receipt["sha256"]:
            raise ValueError(f"output hash mismatch: {path}")


def source_stratified_complete_game_folds(
    data: LabData,
    *,
    outer_split: str = "train",
    folds: int = CROSSFIT_FOLDS,
    salt: str = CROSSFIT_SALT,
) -> dict[str, Any]:
    if outer_split not in data.split_names or folds < 2 or not salt:
        raise ValueError("grouped fold configuration is invalid")
    block_game, game_ids = canonical_game_indices(data)
    game_source = np.full(len(game_ids), -1, dtype=np.int32)
    game_split = np.full(len(game_ids), -1, dtype=np.int32)
    for block_id, game_index in enumerate(block_game):
        source = int(data.blocks["source_idx"][block_id])
        split = int(data.blocks["split_idx"][block_id])
        if game_source[game_index] not in (-1, source) or game_split[game_index] not in (-1, split):
            raise ValueError("canonical game source/split differs across actor blocks")
        game_source[game_index] = source
        game_split[game_index] = split
    split_index = data.split_names.index(outer_split)
    fold_of_game = np.full(len(game_ids), -1, dtype=np.int8)
    source_receipts = []
    for source in np.unique(game_source[game_split == split_index]):
        selected = np.flatnonzero((game_source == source) & (game_split == split_index))
        ordered = sorted(
            (int(index) for index in selected),
            key=lambda index: (
                hashlib.sha256(
                    f"{salt}\0{data.source_ids[int(source)]}\0{game_ids[index]}".encode("utf-8")
                ).digest(),
                game_ids[index],
            ),
        )
        for rank, game_index in enumerate(ordered):
            fold_of_game[game_index] = rank % folds
        source_receipts.append({
            "source_id": data.source_ids[int(source)],
            "games": len(ordered),
            "games_by_fold": [
                sum(1 for index in ordered if int(fold_of_game[index]) == fold)
                for fold in range(folds)
            ],
        })
    selected_games = game_split == split_index
    if np.any(fold_of_game[selected_games] < 0) or np.any(fold_of_game[~selected_games] != -1):
        raise ValueError("fold assignment coverage is invalid")
    fold_of_block = fold_of_game[block_game]
    lengths = data.blocks["row_len"].astype(np.int64, copy=False)
    fold_of_row = np.repeat(fold_of_block, lengths)
    block_of_row = np.repeat(np.arange(len(block_game), dtype=np.int32), lengths)
    if len(fold_of_row) != len(data.meta["split_idx"]):
        raise ValueError("fold assignment does not cover every cache row")
    assignment = {
        "schema_version": FOLD_ASSIGNMENT_SCHEMA_VERSION,
        "salt": salt,
        "folds": folds,
        "outer_split": outer_split,
        "assignment": {
            game_ids[index]: int(fold_of_game[index])
            for index in np.flatnonzero(selected_games)
        },
    }

    return {
        "block_game": block_game,
        "game_ids": game_ids,
        "game_source": game_source,
        "game_split": game_split,
        "fold_of_game": fold_of_game,
        "fold_of_block": fold_of_block,
        "fold_of_row": fold_of_row,
        "block_of_row": block_of_row,
        "source_receipts": source_receipts,
        "assignment": assignment,
        "assignment_sha256": _stable_hash(assignment),
    }


def derive_signature_fold_bits(
    signatures: np.ndarray,
    eligible_rows: np.ndarray,
    row_fold: np.ndarray,
    *,
    total_rows: int,
    folds: int = CROSSFIT_FOLDS,
) -> dict[str, Any]:
    signatures = np.asarray(signatures, dtype="V16")
    eligible_rows = np.asarray(eligible_rows, dtype=np.int64)
    row_fold = np.asarray(row_fold, dtype=np.int8)
    if (
        signatures.ndim != 1
        or eligible_rows.shape != signatures.shape
        or row_fold.shape != signatures.shape
        or len(signatures) == 0
        or np.any(eligible_rows < 0)
        or np.any(eligible_rows >= total_rows)
        or len(np.unique(eligible_rows)) != len(eligible_rows)
        or np.any(row_fold < 0)
        or np.any(row_fold >= folds)
    ):
        raise ValueError("eligible signature arrays are invalid")
    order = np.argsort(signatures, kind="stable")
    sorted_signatures = signatures[order]
    sorted_folds = row_fold[order]
    starts = np.flatnonzero(np.r_[True, sorted_signatures[1:] != sorted_signatures[:-1]])
    fold_bits = np.left_shift(np.uint8(1), sorted_folds.astype(np.uint8))
    group_fold_bits = np.bitwise_or.reduceat(fold_bits, starts)
    lengths = np.diff(np.r_[starts, len(signatures)])
    sorted_group_bits = np.repeat(group_fold_bits, lengths)
    signature_fold_bits = np.zeros(total_rows, dtype=np.uint8)
    signature_fold_bits[eligible_rows[order]] = sorted_group_bits
    crossing_group = (group_fold_bits & (group_fold_bits - 1)) != 0
    crossing_signatures = sorted_signatures[starts[crossing_group]].copy()
    crossing_fold_bits = group_fold_bits[crossing_group].copy()
    crossing_digest = hashlib.sha256(crossing_signatures.tobytes(order="C")).hexdigest()

    return {
        "signature_fold_bits": signature_fold_bits,
        "unique_signatures": int(len(starts)),
        "duplicate_rows_within_outer_train": int(len(signatures) - len(starts)),
        "crossing_signature_count": int(crossing_group.sum()),
        "crossing_signatures": crossing_signatures,
        "crossing_fold_bits": crossing_fold_bits,
        "crossing_signatures_sha256": crossing_digest,
        "crossing_signature_count_by_fold": [
            int(((crossing_fold_bits & (1 << fold)) != 0).sum())
            for fold in range(folds)
        ],
    }


def _catalog_transition_parts(cache_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    canonical_root = Path(cache_manifest["canonical_root"])
    catalog_path = Path(cache_manifest["catalog_path"])
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    units = {
        (unit["source_id"], unit["unit_id"]): unit
        for unit in catalog["included_units"]
    }
    parts = []
    for receipt in cache_manifest.get("inputs", []):
        key = (receipt["source_id"], receipt["unit_id"])
        unit = units.get(key)
        if unit is None:
            raise ValueError(f"cache input unit is absent from the catalog: {key}")
        manifest_path = canonical_root / unit["manifest_path"]
        if not manifest_path.is_file() or _sha256(manifest_path) != unit["manifest_sha256"]:
            raise ValueError(f"canonical unit manifest drifted: {manifest_path}")
        unit_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        part_by_name = {part["name"]: part for part in unit_manifest["parts"]}
        part = part_by_name.get(receipt["part"])
        if part is None or int(part["rows"]) != int(receipt["rows"]):
            raise ValueError(f"cache input part disagrees with unit manifest: {receipt['part']}")
        transition_name = f"{receipt['part']}.transitions.jsonl.gz"
        expected_hash = part["files_sha256"].get(transition_name)
        if expected_hash != receipt["transitions_sha256"]:
            raise ValueError(f"cache transition binding drifted: {transition_name}")
        parts.append({
            **receipt,
            "path": manifest_path.parent / transition_name,
            "expected_sha256": expected_hash,
        })
    if len(parts) != 390:
        raise ValueError(f"expected 390 canonical transition parts, found {len(parts)}")

    return parts


def _scan_effective_signatures(
    data: LabData,
    cache_manifest: dict[str, Any],
    fold_contract: dict[str, Any],
    signature_path: Path,
) -> dict[str, Any]:
    total_rows = len(data.meta["split_idx"])
    temporary = signature_path.with_name(f".{signature_path.name}.{uuid.uuid4().hex}.tmp")
    signatures = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype="V16",
        shape=(total_rows,),
    )
    signatures[:] = np.zeros(1, dtype="V16")[0]
    transition_receipts = []
    global_offset = 0
    started = time.perf_counter()
    block_of_row = fold_contract["block_of_row"]
    for part in _catalog_transition_parts(cache_manifest):
        path = Path(part["path"])
        if not path.is_file():
            raise ValueError(f"canonical transition sidecar is missing: {path}")
        local_rows = int(part["rows"])
        with open(path, "rb") as raw_handle:
            hashing_reader = _HashingReader(raw_handle)
            with gzip.GzipFile(fileobj=hashing_reader, mode="rb") as zipped:
                with io.TextIOWrapper(zipped, encoding="utf-8") as text_handle:
                    observed = 0
                    for line in text_handle:
                        if observed >= local_rows:
                            raise ValueError(f"transition sidecar has too many rows: {path}")
                        record = json.loads(line)
                        global_row = global_offset + observed
                        block_id = int(block_of_row[global_row])
                        block_key = data.block_keys[block_id]
                        game_id, raw_seat = block_key.rsplit(":s", 1)
                        block_local_row = global_row - int(data.blocks["row_start"][block_id])
                        state_before = record.get("state_before") or {}
                        if (
                            int(record.get("row", -1)) != observed
                            or int(state_before.get("feature_row", -1)) != block_local_row
                            or record.get("information_mode") != "actor_visible"
                            or record.get("canonical_game_id") != game_id
                            or int(record.get("actor", -1)) != int(raw_seat)
                            or int(record.get("global_decision_index", -1))
                            != int(data.meta["decision_index"][global_row])
                        ):
                            raise ValueError(f"canonical transition/cache row alignment mismatch: {path}:{observed}")
                        if bool(data.meta["is_target_eligible"][global_row]):
                            signatures[global_row] = np.frombuffer(
                                canonical_state_action_signature(data.X[global_row], record),
                                dtype="V16",
                                count=1,
                            )[0]
                        observed += 1
        if observed != local_rows or hashing_reader.digest.hexdigest() != part["expected_sha256"]:
            raise ValueError(f"transition row count or hash mismatch: {path}")
        transition_receipts.append({
            "source_id": part["source_id"],
            "unit_id": part["unit_id"],
            "part": part["part"],
            "rows": observed,
            "bytes": hashing_reader.bytes_read,
            "sha256": hashing_reader.digest.hexdigest(),
        })
        global_offset += observed
    signatures.flush()
    del signatures
    if global_offset != total_rows:
        raise ValueError(f"transition scan covered {global_offset} rows, expected {total_rows}")
    os.replace(temporary, signature_path)

    return {
        "rows": global_offset,
        "parts": len(transition_receipts),
        "transition_receipts": transition_receipts,
        "elapsed_seconds": time.perf_counter() - started,
    }


def build_inner_fold_audit(
    data: LabData,
    cache_dir: Path,
    out_dir: Path,
    fold_contract: dict[str, Any],
    *,
    config_sha256: str,
) -> dict[str, Any]:
    manifest_path = out_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = read_manifest(manifest_path)
        if manifest.get("inputs", {}).get("config_sha256") != config_sha256:
            raise ValueError("existing inner-fold audit config differs")
        _verify_output_files(out_dir, manifest["outputs"]["files"])

        return manifest
    if out_dir.exists():
        preview = ", ".join(sorted(path.name for path in out_dir.iterdir())[:10]) or "empty"
        raise RuntimeError(f"refusing incomplete inner-fold audit: {out_dir} ({preview})")
    temp_dir = out_dir.with_name(f".{out_dir.name}.running.{uuid.uuid4().hex}")
    temp_dir.mkdir(parents=True)
    cache_manifest = json.loads((cache_dir / "cache_manifest.json").read_text(encoding="utf-8"))
    assignment_path = temp_dir / "assignment.json.gz"
    _atomic_gzip_json(assignment_path, fold_contract["assignment"])
    block_fold_path = temp_dir / "block_fold.i1.npy"
    row_fold_path = temp_dir / "row_fold.i1.npy"
    block_of_row_path = temp_dir / "block_of_row.i4.npy"
    _atomic_npy(block_fold_path, fold_contract["fold_of_block"].astype(np.int8, copy=False))
    _atomic_npy(row_fold_path, fold_contract["fold_of_row"].astype(np.int8, copy=False))
    _atomic_npy(block_of_row_path, fold_contract["block_of_row"].astype(np.int32, copy=False))
    signature_path = temp_dir / "effective_signature.v16.npy"
    scan = _scan_effective_signatures(data, cache_manifest, fold_contract, signature_path)
    all_signatures = np.load(signature_path, mmap_mode="r", allow_pickle=False)
    train_index = data.split_names.index("train")
    train_eligible = np.flatnonzero(
        (data.meta["split_idx"] == train_index) & data.meta["is_target_eligible"]
    ).astype(np.int64, copy=False)
    derived = derive_signature_fold_bits(
        all_signatures[train_eligible],
        train_eligible,
        fold_contract["fold_of_row"][train_eligible],
        total_rows=len(data.meta["split_idx"]),
    )
    del all_signatures
    signature_fold_bits_path = temp_dir / "train_signature_fold_bits.u1.npy"
    _atomic_npy(signature_fold_bits_path, derived["signature_fold_bits"])
    crossing_path = temp_dir / "crossing_signatures.npz"
    _atomic_npz(
        crossing_path,
        signature=derived["crossing_signatures"],
        fold_bits=derived["crossing_fold_bits"],
    )
    mask_paths = []
    fold_receipts = []
    outer_train = data.meta["split_idx"] == train_index
    outer_eligible = outer_train & data.meta["is_target_eligible"]
    for fold in range(CROSSFIT_FOLDS):
        fold_dir = temp_dir / f"fold_{fold}"
        fold_dir.mkdir()
        holdout = outer_eligible & (fold_contract["fold_of_row"] == fold)
        train = (
            outer_eligible
            & (fold_contract["fold_of_row"] != fold)
            & ((derived["signature_fold_bits"] & (1 << fold)) == 0)
        )
        if np.any(train & holdout):
            raise AssertionError("inner train and holdout target masks overlap")
        train_path = fold_dir / "train_target_mask.bool.npy"
        holdout_path = fold_dir / "holdout_target_mask.bool.npy"
        _atomic_npy(train_path, train)
        _atomic_npy(holdout_path, holdout)
        mask_paths.extend([train_path, holdout_path])
        fold_receipts.append({
            "fold": fold,
            "train_targets": int(train.sum()),
            "holdout_targets": int(holdout.sum()),
            "train_games": int(len(set(
                fold_contract["block_game"][np.flatnonzero(
                    (data.blocks["split_idx"] == train_index)
                    & (fold_contract["fold_of_block"] != fold)
                )].tolist()
            ))),
            "holdout_games": int((fold_contract["fold_of_game"] == fold).sum()),
            "crossing_signatures_removed_from_train": derived[
                "crossing_signature_count_by_fold"
            ][fold],
            "effective_signature_overlap": 0,
        })
    audit_path = temp_dir / "audit.json"
    audit = {
        "schema_version": INNER_AUDIT_SCHEMA_VERSION,
        "status": "pass",
        "signature_version": STATE_ACTION_SIGNATURE_VERSION,
        "assignment_sha256": fold_contract["assignment_sha256"],
        "outer_train_games": int((fold_contract["fold_of_game"] >= 0).sum()),
        "outer_train_context_rows": int(outer_train.sum()),
        "outer_train_eligible_targets": int(outer_eligible.sum()),
        "outer_train_excluded_targets": int((outer_train & ~data.meta["is_target_eligible"]).sum()),
        "transition_scan": {
            "rows": scan["rows"],
            "parts": scan["parts"],
            "elapsed_seconds": scan["elapsed_seconds"],
            "compressed_inputs_sha256": _stable_hash(scan["transition_receipts"]),
        },
        "unique_train_effective_signatures": derived["unique_signatures"],
        "duplicate_rows_within_outer_train": derived["duplicate_rows_within_outer_train"],
        "crossing_signature_count": derived["crossing_signature_count"],
        "crossing_signatures_sha256": derived["crossing_signatures_sha256"],
        "crossing_signature_count_by_fold": derived["crossing_signature_count_by_fold"],
        "source_receipts": fold_contract["source_receipts"],
        "folds": fold_receipts,
        "effective_forbidden_overlap": 0,
    }
    write_json(audit_path, _json_safe(audit))
    output_paths = [
        assignment_path,
        block_fold_path,
        row_fold_path,
        block_of_row_path,
        signature_path,
        signature_fold_bits_path,
        crossing_path,
        audit_path,
        *mask_paths,
    ]
    files = _file_receipts(temp_dir, output_paths)
    manifest = build_manifest(
        run_id=out_dir.name,
        kind="value_inner_fold_signature_audit",
        command=[],
        inputs={
            "config_sha256": config_sha256,
            "cache_manifest_sha256": _sha256(cache_dir / "cache_manifest.json"),
            "assignment_sha256": fold_contract["assignment_sha256"],
            "signature_code_sha256": _sha256(Path(value_contracts.__file__)),
        },
        outputs={"files": files},
        summary=audit,
    )
    write_manifest(temp_dir / "manifest.json", manifest)
    os.replace(temp_dir, out_dir)

    return manifest


def deterministic_block_row_stream(
    data: LabData,
    target_mask: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    target_mask = np.asarray(target_mask, dtype=np.bool_)
    if target_mask.shape != data.meta["is_target_eligible"].shape:
        raise ValueError("target mask does not cover cache rows")
    if np.any(target_mask & ~data.meta["is_target_eligible"]):
        raise ValueError("target stream attempts to restore an excluded target")
    starts = data.blocks["row_start"].astype(np.int64, copy=False)
    lengths = data.blocks["row_len"].astype(np.int64, copy=False)
    selected_blocks = np.flatnonzero(np.add.reduceat(target_mask.astype(np.int64), starts) > 0)
    order = np.random.default_rng(seed).permutation(selected_blocks)
    rows = np.empty(int(target_mask.sum()), dtype=np.int64)
    cursor = 0
    for block_id in order:
        start = int(starts[block_id])
        length = int(lengths[block_id])
        selected = start + np.flatnonzero(target_mask[start:start + length])
        rows[cursor:cursor + len(selected)] = selected
        cursor += len(selected)
    if cursor != len(rows) or not bool(target_mask[rows].all()):
        raise AssertionError("deterministic target stream coverage drifted")

    return rows


def _fold_unit_config(
    root_config: dict[str, Any],
    fold: int,
    train_mask_path: Path,
    holdout_mask_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": FOLD_UNIT_SCHEMA_VERSION,
        "fold": fold,
        "root_config_sha256": root_config["sha256"],
        "assignment_sha256": root_config["assignment_sha256"],
        "train_mask_sha256": _sha256(train_mask_path),
        "holdout_mask_sha256": _sha256(holdout_mask_path),
        "seed": root_config["seed"],
        "batch_size": root_config["batch_size"],
        "threads": root_config["threads"],
        "learning_rate": root_config["learning_rate"],
        "weight_decay": root_config["weight_decay"],
        "model": root_config["model"],
        "representation": root_config["representation"],
    }


def _verify_unit(unit_dir: Path, expected_config_sha: str) -> dict[str, Any] | None:
    manifest_path = unit_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = read_manifest(manifest_path)
    if manifest.get("inputs", {}).get("config_sha256") != expected_config_sha:
        raise ValueError(f"completed cross-fit unit config differs: {unit_dir}")
    _verify_output_files(unit_dir, manifest["outputs"]["files"])

    return manifest


def train_fold_unit(
    data: LabData,
    root_config: dict[str, Any],
    audit_dir: Path,
    run_dir: Path,
    fold: int,
    calibration_rows: np.ndarray,
) -> dict[str, Any]:
    unit_dir = run_dir / f"fold_{fold}"
    train_mask_path = audit_dir / f"fold_{fold}" / "train_target_mask.bool.npy"
    holdout_mask_path = audit_dir / f"fold_{fold}" / "holdout_target_mask.bool.npy"
    unit_config = _fold_unit_config(root_config, fold, train_mask_path, holdout_mask_path)
    unit_config_sha = _stable_hash(unit_config)
    completed = _verify_unit(unit_dir, unit_config_sha)
    if completed is not None:
        return completed
    if unit_dir.exists():
        raise RuntimeError(f"refusing incomplete completed-unit path: {unit_dir}")
    temp_dir = unit_dir.with_name(f".{unit_dir.name}.running.{uuid.uuid4().hex}")
    temp_dir.mkdir()
    train_mask = np.load(train_mask_path, mmap_mode="r", allow_pickle=False)
    holdout_mask = np.load(holdout_mask_path, mmap_mode="r", allow_pickle=False)
    row_stream = deterministic_block_row_stream(data, train_mask, seed=root_config["seed"])
    stream_receipt = row_stream_receipt(data, row_stream)
    stream_path = temp_dir / "train_row_stream.i8.npy"
    _atomic_npy(stream_path, row_stream)
    model, train_receipt = train_snapshot(
        data,
        SNAPSHOT_SPEC,
        row_stream,
        seed=root_config["seed"],
        batch_size=root_config["batch_size"],
        learning_rate=root_config["learning_rate"],
        weight_decay=root_config["weight_decay"],
    )
    holdout_context = np.flatnonzero(
        (data.meta["split_idx"] == data.split_names.index("train"))
        & (np.load(audit_dir / "row_fold.i1.npy", mmap_mode="r", allow_pickle=False) == fold)
    ).astype(np.int64, copy=False)
    holdout_predictions, holdout_inference = _predict(
        data,
        model,
        SNAPSHOT_SPEC,
        holdout_context,
        batch_size=root_config["batch_size"],
    )
    calibration_predictions, calibration_inference = _predict(
        data,
        model,
        SNAPSHOT_SPEC,
        calibration_rows,
        batch_size=root_config["batch_size"],
    )
    checkpoint_path = temp_dir / "value.pt"
    checkpoint = {
        "schema_version": FOLD_UNIT_SCHEMA_VERSION,
        "fold": fold,
        "spec": asdict(SNAPSHOT_SPEC),
        "model_config": {
            "input_dim": 49,
            "hidden_dims": [256, 128],
            "dropout": 0.0,
        },
        "parameter_count": parameter_count(model),
        "seed": root_config["seed"],
        "row_stream_sha256": stream_receipt["sha256"],
        "unit_config_sha256": unit_config_sha,
        "root_config_sha256": root_config["sha256"],
        "model_state": model.state_dict(),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": root_config["learning_rate"],
            "weight_decay": root_config["weight_decay"],
            "batch_size": root_config["batch_size"],
        },
        "train": train_receipt,
    }
    _atomic_torch_save(checkpoint_path, checkpoint)
    holdout_path = temp_dir / "heldout_context_predictions.npz"
    _atomic_npz(
        holdout_path,
        row_id=holdout_context,
        p=holdout_predictions,
        is_target_eligible=data.meta["is_target_eligible"][holdout_context],
        effective_metric_eligible=np.asarray(holdout_mask[holdout_context], dtype=np.bool_),
    )
    calibration_path = temp_dir / "calibration_predictions.npz"
    _atomic_npz(calibration_path, row_id=calibration_rows, p=calibration_predictions)
    receipt_path = temp_dir / "receipt.json"
    receipt = {
        "schema_version": FOLD_UNIT_SCHEMA_VERSION,
        "status": "completed",
        "fold": fold,
        "config_sha256": unit_config_sha,
        "train_row_stream": stream_receipt,
        "train": train_receipt,
        "holdout_context_rows": int(len(holdout_context)),
        "holdout_target_rows": int(data.meta["is_target_eligible"][holdout_context].sum()),
        "holdout_effective_metric_rows": int(holdout_mask[holdout_context].sum()),
        "holdout_inference": holdout_inference,
        "calibration_rows": int(len(calibration_rows)),
        "calibration_inference": calibration_inference,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "prediction_sha256": _sha256(holdout_path),
        "calibration_prediction_sha256": _sha256(calibration_path),
        "nonfinite_counts": {
            "train": sum(train_receipt["nonfinite_counts"].values()),
            "holdout": sum(holdout_inference["nonfinite_counts"].values()),
            "calibration": sum(calibration_inference["nonfinite_counts"].values()),
        },
    }
    write_json(receipt_path, _json_safe(receipt))
    output_paths = [stream_path, checkpoint_path, holdout_path, calibration_path, receipt_path]
    files = _file_receipts(temp_dir, output_paths)
    manifest = build_manifest(
        run_id=unit_dir.name,
        kind="value_grouped_crossfit_fold",
        command=[],
        inputs={"config": unit_config, "config_sha256": unit_config_sha},
        outputs={"files": files},
        summary=receipt,
    )
    write_manifest(temp_dir / "manifest.json", manifest)
    os.replace(temp_dir, unit_dir)

    return manifest


def train_final_unit(
    data: LabData,
    root_config: dict[str, Any],
    run_dir: Path,
    calibration_rows: np.ndarray,
) -> dict[str, Any]:
    unit_dir = run_dir / "full_train"
    final_config = {
        "schema_version": FOLD_UNIT_SCHEMA_VERSION,
        "role": "final_full_outer_train",
        "root_config_sha256": root_config["sha256"],
        "seed": root_config["seed"],
        "batch_size": root_config["batch_size"],
        "learning_rate": root_config["learning_rate"],
        "weight_decay": root_config["weight_decay"],
        "model": root_config["model"],
        "representation": root_config["representation"],
    }
    config_sha = _stable_hash(final_config)
    completed = _verify_unit(unit_dir, config_sha)
    if completed is not None:
        return completed
    if unit_dir.exists():
        raise RuntimeError(f"refusing incomplete final-unit path: {unit_dir}")
    temp_dir = unit_dir.with_name(f".{unit_dir.name}.running.{uuid.uuid4().hex}")
    temp_dir.mkdir()
    train_index = data.split_names.index("train")
    target_mask = (data.meta["split_idx"] == train_index) & data.meta["is_target_eligible"]
    row_stream = deterministic_block_row_stream(data, target_mask, seed=root_config["seed"])
    stream_receipt = row_stream_receipt(data, row_stream)
    stream_path = temp_dir / "train_row_stream.i8.npy"
    _atomic_npy(stream_path, row_stream)
    model, train_receipt = train_snapshot(
        data,
        SNAPSHOT_SPEC,
        row_stream,
        seed=root_config["seed"],
        batch_size=root_config["batch_size"],
        learning_rate=root_config["learning_rate"],
        weight_decay=root_config["weight_decay"],
    )
    calibration_predictions, inference = _predict(
        data,
        model,
        SNAPSHOT_SPEC,
        calibration_rows,
        batch_size=root_config["batch_size"],
    )
    checkpoint_path = temp_dir / "value.pt"
    checkpoint = {
        "schema_version": FOLD_UNIT_SCHEMA_VERSION,
        "role": "final_full_outer_train",
        "spec": asdict(SNAPSHOT_SPEC),
        "model_config": {"input_dim": 49, "hidden_dims": [256, 128], "dropout": 0.0},
        "parameter_count": parameter_count(model),
        "seed": root_config["seed"],
        "row_stream_sha256": stream_receipt["sha256"],
        "unit_config_sha256": config_sha,
        "root_config_sha256": root_config["sha256"],
        "model_state": model.state_dict(),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": root_config["learning_rate"],
            "weight_decay": root_config["weight_decay"],
            "batch_size": root_config["batch_size"],
        },
        "train": train_receipt,
    }
    _atomic_torch_save(checkpoint_path, checkpoint)
    prediction_path = temp_dir / "calibration_predictions.npz"
    _atomic_npz(prediction_path, row_id=calibration_rows, p=calibration_predictions)
    receipt_path = temp_dir / "receipt.json"
    receipt = {
        "schema_version": FOLD_UNIT_SCHEMA_VERSION,
        "status": "completed",
        "role": "final_full_outer_train",
        "config_sha256": config_sha,
        "train_row_stream": stream_receipt,
        "train": train_receipt,
        "calibration_rows": int(len(calibration_rows)),
        "calibration_inference": inference,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "calibration_prediction_sha256": _sha256(prediction_path),
        "nonfinite_counts": {
            "train": sum(train_receipt["nonfinite_counts"].values()),
            "calibration": sum(inference["nonfinite_counts"].values()),
        },
    }
    write_json(receipt_path, _json_safe(receipt))
    output_paths = [stream_path, checkpoint_path, prediction_path, receipt_path]
    files = _file_receipts(temp_dir, output_paths)
    manifest = build_manifest(
        run_id=unit_dir.name,
        kind="value_grouped_crossfit_final_model",
        command=[],
        inputs={"config": final_config, "config_sha256": config_sha},
        outputs={"files": files},
        summary=receipt,
    )
    write_manifest(temp_dir / "manifest.json", manifest)
    os.replace(temp_dir, unit_dir)

    return manifest


def assemble_oof_predictions(
    data: LabData,
    run_dir: Path,
    audit_dir: Path,
    fold_manifests: list[dict[str, Any]],
    root_config: dict[str, Any],
) -> dict[str, Any]:
    output_path = run_dir / "oof_predictions.npz"
    receipt_path = run_dir / "oof_receipt.json"
    if output_path.is_file() and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("root_config_sha256") != root_config["sha256"]
            or receipt.get("output_sha256") != _sha256(output_path)
        ):
            raise ValueError("existing OOF assembly receipt differs")

        return receipt
    total_rows = len(data.meta["split_idx"])
    train_index = data.split_names.index("train")
    train_rows = np.flatnonzero(data.meta["split_idx"] == train_index).astype(np.int64, copy=False)
    raw_p = np.full(total_rows, np.nan, dtype=np.float32)
    producing_fold = np.full(total_rows, -1, dtype=np.int8)
    effective_metric = np.zeros(total_rows, dtype=np.bool_)
    for fold, manifest in enumerate(fold_manifests):
        prediction_path = run_dir / f"fold_{fold}" / "heldout_context_predictions.npz"
        with np.load(prediction_path, allow_pickle=False) as arrays:
            row_id = arrays["row_id"].astype(np.int64, copy=False)
            p = arrays["p"].astype(np.float32, copy=False)
            effective = arrays["effective_metric_eligible"].astype(np.bool_, copy=False)
        if len(row_id) != len(p) or np.any(producing_fold[row_id] >= 0):
            raise ValueError("fold prediction rows are duplicated or misaligned")
        raw_p[row_id] = p
        producing_fold[row_id] = fold
        effective_metric[row_id] = effective
    if (
        not np.isfinite(raw_p[train_rows]).all()
        or np.any((raw_p[train_rows] < 0.0) | (raw_p[train_rows] > 1.0))
        or np.any(producing_fold[train_rows] < 0)
        or np.any(np.isfinite(raw_p[data.meta["split_idx"] != train_index]))
    ):
        raise ValueError("OOF prediction coverage or probability range is invalid")
    row_fold = np.load(audit_dir / "row_fold.i1.npy", mmap_mode="r", allow_pickle=False)
    if not np.array_equal(producing_fold[train_rows], row_fold[train_rows]):
        raise ValueError("OOF producing fold disagrees with game assignment")
    block_of_row = np.load(audit_dir / "block_of_row.i4.npy", mmap_mode="r", allow_pickle=False)
    block_game, game_ids = canonical_game_indices(data)
    game_index = block_game[block_of_row[train_rows]]
    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(
            handle,
            row_id=train_rows,
            decision_id=data.meta["decision_index"][train_rows].astype(np.int32, copy=False),
            block_id=block_of_row[train_rows].astype(np.int32, copy=False),
            game_gidx=game_index.astype(np.int32, copy=False),
            fold_id=producing_fold[train_rows],
            raw_probability=raw_p[train_rows],
            outer_target_eligible=data.meta["is_target_eligible"][train_rows],
            effective_metric_eligible=effective_metric[train_rows],
        )
    os.replace(temporary, output_path)
    game_vocab_path = run_dir / "oof_game_vocab.json.gz"
    _atomic_gzip_json(game_vocab_path, {"game_ids": game_ids})
    receipt = {
        "schema_version": OOF_PREDICTION_SCHEMA_VERSION,
        "status": "completed_raw_uncalibrated",
        "root_config_sha256": root_config["sha256"],
        "rows": int(len(train_rows)),
        "eligible_target_rows": int(data.meta["is_target_eligible"][train_rows].sum()),
        "excluded_context_rows": int((~data.meta["is_target_eligible"][train_rows]).sum()),
        "effective_metric_rows": int(effective_metric[train_rows].sum()),
        "fold_counts_context": [int((producing_fold[train_rows] == fold).sum()) for fold in range(CROSSFIT_FOLDS)],
        "fold_counts_targets": [
            int(((producing_fold[train_rows] == fold) & data.meta["is_target_eligible"][train_rows]).sum())
            for fold in range(CROSSFIT_FOLDS)
        ],
        "checkpoint_sha256_by_fold": [manifest["summary"]["checkpoint_sha256"] for manifest in fold_manifests],
        "representation_manifest_sha256": root_config["representation"]["scientific_sha256"],
        "split_manifest_sha256": root_config["split_manifest_sha256"],
        "information_mode": "actor_visible",
        "calibration_status": "raw_not_yet_calibrated",
        "output_sha256": _sha256(output_path),
        "game_vocab_sha256": _sha256(game_vocab_path),
        "nonfinite_predictions": 0,
    }
    write_json(receipt_path, receipt)

    return receipt


def _load_raw_s1_representation(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = representation_manifest("R0_raw_s1", range(49))
    if _stable_hash(payload) != _stable_hash(expected):
        raise ValueError("cross-fit representation is not the frozen raw-S1 manifest")

    return {
        "path": str(path.resolve()),
        "file_sha256": _sha256(path),
        "scientific_sha256": expected["sha256"],
        "indices": expected["indices"],
    }


def run_grouped_crossfit(
    *,
    cache_dir: Path,
    split_path: Path,
    representation_path: Path,
    feature_confirmation_manifest: Path,
    out_dir: Path,
    seed: int = FINAL_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    threads: int = DEFAULT_THREADS,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    stop_after_audit: bool = False,
    command: list[str] | None = None,
) -> dict[str, Any]:
    cache_dir = cache_dir.resolve()
    split_path = split_path.resolve()
    representation_path = representation_path.resolve()
    feature_confirmation_manifest = feature_confirmation_manifest.resolve()
    out_dir = out_dir.resolve()
    if seed < 0 or batch_size < 1 or threads < 1 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("cross-fit training configuration is invalid")
    confirmation = read_manifest(feature_confirmation_manifest)
    winner = confirmation.get("summary", {}).get("gate", {}).get("winner")
    if winner != "R0_raw_s1":
        raise ValueError("serious feature confirmation did not freeze raw S1")
    representation = _load_raw_s1_representation(representation_path)
    cache_manifest_sha = _sha256(cache_dir / "cache_manifest.json")
    split_sha = _sha256(split_path)
    config = {
        "schema_version": CROSSFIT_SCHEMA_VERSION,
        "folds": CROSSFIT_FOLDS,
        "fold_kind": "source_stratified_complete_game_grouped_non_temporal_oof",
        "salt": CROSSFIT_SALT,
        "seed": seed,
        "batch_size": batch_size,
        "threads": threads,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "target": "mc_actor_relative_terminal",
        "information_mode": "actor_visible",
        "model": {
            "name": "v0_snapshot_s1",
            "architecture": [49, 256, 128, 1],
            "parameters": 45_825,
            "normalization": "identity_canonical_float32",
            "checkpoint_selection": "final_update_only",
        },
        "representation": representation,
        "cache_path": str(cache_dir),
        "cache_manifest_sha256": cache_manifest_sha,
        "split_path": str(split_path),
        "split_manifest_sha256": split_sha,
        "feature_confirmation_manifest_path": str(feature_confirmation_manifest),
        "feature_confirmation_manifest_sha256": _sha256(feature_confirmation_manifest),
        "code_sha256": {
            "crossfit": _sha256(Path(__file__)),
            "contracts": _sha256(Path(value_contracts.__file__)),
            "data": _sha256(Path(value_data.__file__)),
            "mc_screen": _sha256(Path(value_mc_screen.__file__)),
            "models": _sha256(Path(value_models.__file__)),
        },
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "device": "cpu",
        },
        "calibration_targets_consumed": False,
        "test_targets_consumed": False,
    }
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    data = LabData(str(cache_dir), split_manifest_path=str(split_path))
    fold_contract = source_stratified_complete_game_folds(data)
    config["assignment_sha256"] = fold_contract["assignment_sha256"]
    config["sha256"] = _stable_hash(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "CROSSFIT_PLAN.json"
    if plan_path.is_file():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError("existing cross-fit plan differs from requested config")
    else:
        if any(out_dir.iterdir()):
            raise RuntimeError(f"cross-fit directory is nonempty without a plan: {out_dir}")
        write_json(plan_path, _json_safe(config))
    completed_manifest_path = out_dir / "manifest.json"
    if completed_manifest_path.is_file():
        completed = read_manifest(completed_manifest_path)
        if completed.get("inputs", {}).get("config_sha256") != config["sha256"]:
            raise ValueError("completed cross-fit config differs")
        _verify_output_files(out_dir, completed["outputs"]["files"])

        return {"action": "reused_verified", "manifest": completed}
    audit_dir = out_dir / "fold_contract"
    audit_manifest = build_inner_fold_audit(
        data,
        cache_dir,
        audit_dir,
        fold_contract,
        config_sha256=config["sha256"],
    )
    if stop_after_audit:
        return {"action": "audit_completed", "manifest": audit_manifest}
    calibration_index = data.split_names.index("calibration")
    calibration_rows = np.flatnonzero(
        (data.meta["split_idx"] == calibration_index) & data.meta["is_target_eligible"]
    ).astype(np.int64, copy=False)
    fold_manifests = []
    for fold in range(CROSSFIT_FOLDS):
        fold_manifests.append(train_fold_unit(
            data,
            config,
            audit_dir,
            out_dir,
            fold,
            calibration_rows,
        ))
        write_json(out_dir / "progress.json", {
            "status": "in_progress",
            "completed_folds": fold + 1,
            "total_folds": CROSSFIT_FOLDS,
            "full_train_completed": False,
        })
    final_manifest = train_final_unit(data, config, out_dir, calibration_rows)
    write_json(out_dir / "progress.json", {
        "status": "assembling_oof",
        "completed_folds": CROSSFIT_FOLDS,
        "total_folds": CROSSFIT_FOLDS,
        "full_train_completed": True,
    })
    oof_receipt = assemble_oof_predictions(
        data,
        out_dir,
        audit_dir,
        fold_manifests,
        config,
    )
    progress_path = out_dir / "progress.json"
    write_json(progress_path, {
        "status": "completed",
        "completed_folds": CROSSFIT_FOLDS,
        "total_folds": CROSSFIT_FOLDS,
        "full_train_completed": True,
        "oof_completed": True,
    })
    root_paths = [
        plan_path,
        progress_path,
        out_dir / "oof_predictions.npz",
        out_dir / "oof_receipt.json",
        out_dir / "oof_game_vocab.json.gz",
        audit_dir / "manifest.json",
        *(out_dir / f"fold_{fold}" / "manifest.json" for fold in range(CROSSFIT_FOLDS)),
        out_dir / "full_train" / "manifest.json",
    ]
    files = _file_receipts(out_dir, root_paths)
    summary = {
        "status": "completed_raw_uncalibrated",
        "fold_kind": config["fold_kind"],
        "folds": CROSSFIT_FOLDS,
        "assignment_sha256": config["assignment_sha256"],
        "inner_audit_manifest_sha256": _sha256(audit_dir / "manifest.json"),
        "effective_forbidden_overlap": audit_manifest["summary"]["effective_forbidden_overlap"],
        "fold_checkpoint_sha256": [manifest["summary"]["checkpoint_sha256"] for manifest in fold_manifests],
        "final_checkpoint_sha256": final_manifest["summary"]["checkpoint_sha256"],
        "oof": oof_receipt,
        "calibration_targets_consumed": False,
        "test_targets_consumed": False,
        "protected_runtime_surfaces_changed": False,
    }
    manifest = build_manifest(
        run_id=out_dir.name,
        kind="value_grouped_crossfit",
        command=command or [],
        inputs={"config": config, "config_sha256": config["sha256"]},
        outputs={"files": files},
        summary=summary,
    )
    write_manifest(completed_manifest_path, manifest)

    return {"action": "completed", "manifest": manifest}
