"""Fail-closed producer and sealer for league AWR training receipts.

The legacy R18 trainer manifest remains valid for historical workflows.  A
future league generation opts into the stricter producer contract explicitly;
this module then verifies every bound artifact before emitting the immutable
``deepmew.awr_train_receipt.v1`` consumed by the generation controller.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import subprocess
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from deepmew.curriculum.league_registry import sha256_path
from deepmew.replay.information_boundary import (
    DEPLOYABLE_MODES,
    SCHEMA_VERSION as INFORMATION_BOUNDARY_SCHEMA_VERSION,
    classify_decision_row_visibility,
)


TRAINER_MANIFEST_SCHEMA_VERSION = "deepmew.r18.awr.v1"
TRAIN_RECEIPT_SCHEMA_VERSION = "deepmew.awr_train_receipt.v1"
DATAGEN_SCHEMA_VERSION = "deepmew.r19.league_datagen.v2"
PUBLIC_INFORMATION_MODE = "public_belief"
INPUT_AUDIT_SCHEMA_VERSION = "deepmew.awr_input_audit.v1"
RUNTIME_AUDIT_SCHEMA_VERSION = "deepmew.awr_runtime_contamination_audit.v1"
REFGAME_ROW_SCHEMA_VERSION = "deepmew.refgame_decision_row.rebuilt.v2"

_PROTECTED_RUNTIME_SURFACES = ("main.py", "deck.csv", "cg", "deepmew/submissions")

_GENERATION_KEYS = (
    "generation_id",
    "snapshot_id",
    "champion_member_id",
    "candidate_deck_sha256",
    "feature_policy_id",
)
_SAFETY_KEYS = ("crashes", "timeouts", "illegal", "incomplete")


def _utc_now() -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0)

    return now.isoformat().replace("+00:00", "Z")


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")

    return value


def _require_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")

    return value.strip()


def _require_sha256(value: Any, *, field: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be a 64-character SHA-256")

    return text


def _require_nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be an explicit non-negative integer")

    return value


def _require_positive_int(value: Any, *, field: str) -> int:
    result = _require_nonnegative_int(value, field=field)
    if result <= 0:
        raise ValueError(f"{field} must be positive")

    return result


def _read_json(path: str | Path, *, field: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{field} must contain a JSON object")

    return resolved, payload


def canonical_config_sha256(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(config),
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def _artifact_receipt(
    path: str | Path,
    *,
    schema_version: str,
    status: str,
    rows: int | None = None,
) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    receipt: dict[str, Any] = {
        "path": str(resolved),
        "sha256": sha256_path(resolved),
        "schema_version": schema_version,
        "status": status,
    }
    if rows is not None:
        receipt["rows"] = _require_positive_int(rows, field=f"{resolved}.rows")

    return receipt


def _verify_artifact_receipt(
    value: Any,
    *,
    field: str,
    expected_sha256: str | None = None,
) -> tuple[Path, Mapping[str, Any]]:
    receipt = _require_mapping(value, field=field)
    path = Path(_require_string(receipt.get("path"), field=f"{field}.path")).resolve(strict=True)
    recorded = _require_sha256(receipt.get("sha256"), field=f"{field}.sha256")
    actual = sha256_path(path)
    if actual != recorded:
        raise ValueError(f"{field} bytes do not match the recorded SHA-256")
    if expected_sha256 is not None and actual != _require_sha256(
        expected_sha256,
        field=f"expected_{field}.sha256",
    ):
        raise ValueError(f"{field} does not match the expected immutable SHA-256")
    _require_string(receipt.get("schema_version"), field=f"{field}.schema_version")
    _require_string(receipt.get("status"), field=f"{field}.status")

    return path, receipt


def _require_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be an explicit boolean")

    return value


def _validate_refgame_row(
    row: Mapping[str, Any],
    *,
    path: Path,
    line_number: int,
    generation_binding: Mapping[str, str],
    init_checkpoint_sha256: str,
) -> tuple[str, str, int]:
    location = f"{path}:{line_number}"
    if row.get("schema_version") != REFGAME_ROW_SCHEMA_VERSION:
        raise ValueError(f"{location}: unexpected refgame row schema")
    if row.get("information_boundary_schema_version") != INFORMATION_BOUNDARY_SCHEMA_VERSION:
        raise ValueError(f"{location}: unexpected information-boundary schema")
    if row.get("information_mode") != PUBLIC_INFORMATION_MODE:
        raise ValueError(f"{location}: strict AWR row is not stamped public_belief")
    visibility = classify_decision_row_visibility(dict(row))
    if visibility not in DEPLOYABLE_MODES:
        raise ValueError(f"{location}: refgame row content classified {visibility}")
    if row.get("our_league_member_id") != generation_binding["champion_member_id"]:
        raise ValueError(f"{location}: learner league-member binding mismatch")
    if row.get("our_feature_policy_id") != generation_binding["feature_policy_id"]:
        raise ValueError(f"{location}: learner feature-policy binding mismatch")
    if row.get("our_policy_artifact_sha256") != init_checkpoint_sha256:
        raise ValueError(f"{location}: learner checkpoint binding mismatch")

    game_id = _require_string(row.get("game_id"), field=f"{location}.game_id")
    decision_id = _require_string(row.get("decision_id"), field=f"{location}.decision_id")
    decision_index = _require_nonnegative_int(
        row.get("decision_index"),
        field=f"{location}.decision_index",
    )
    if decision_id != f"{game_id}:d{decision_index:04d}":
        raise ValueError(f"{location}: decision identity is not canonical")
    seat = _require_nonnegative_int(row.get("seat"), field=f"{location}.seat")
    if seat not in (0, 1):
        raise ValueError(f"{location}: seat must be 0 or 1")
    raw_observation = _require_mapping(
        row.get("raw_observation"),
        field=f"{location}.raw_observation",
    )
    current = _require_mapping(
        raw_observation.get("current"),
        field=f"{location}.raw_observation.current",
    )
    if current.get("yourIndex") != seat:
        raise ValueError(f"{location}: raw observation is not learner-actor perspective")

    legal_set = row.get("legal_set")
    if not isinstance(legal_set, list) or not legal_set:
        raise ValueError(f"{location}: legal_set must be non-empty")
    action_mask = _require_mapping(row.get("action_mask"), field=f"{location}.action_mask")
    legal_count = _require_positive_int(
        action_mask.get("n_options"),
        field=f"{location}.action_mask.n_options",
    )
    if legal_count != len(legal_set):
        raise ValueError(f"{location}: action-mask size does not match legal_set")
    select = _require_mapping(raw_observation.get("select"), field=f"{location}.raw_observation.select")
    options = select.get("option")
    if not isinstance(options, list) or len(options) != legal_count:
        raise ValueError(f"{location}: observation options do not match legal_set")
    chosen = _require_mapping(row.get("chosen_action"), field=f"{location}.chosen_action")
    indices = chosen.get("indices")
    if not isinstance(indices, list) or not indices:
        raise ValueError(f"{location}: chosen action indices must be non-empty")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise ValueError(f"{location}: chosen action indices must be integers")
    if len(set(indices)) != len(indices) or any(index < 0 or index >= legal_count for index in indices):
        raise ValueError(f"{location}: chosen action indices are not a legal unique selection")
    minimum = _require_nonnegative_int(
        action_mask.get("min_count"),
        field=f"{location}.action_mask.min_count",
    )
    maximum = _require_nonnegative_int(
        action_mask.get("max_count"),
        field=f"{location}.action_mask.max_count",
    )
    if not minimum <= len(indices) <= maximum <= legal_count:
        raise ValueError(f"{location}: chosen action count violates the action mask")

    outcome = _require_mapping(row.get("outcome"), field=f"{location}.outcome")
    if _require_bool(outcome.get("crashed"), field=f"{location}.outcome.crashed"):
        raise ValueError(f"{location}: crashed row reached strict AWR input")
    if _require_bool(outcome.get("timed_out"), field=f"{location}.outcome.timed_out"):
        raise ValueError(f"{location}: timed-out row reached strict AWR input")
    result = _require_nonnegative_int(outcome.get("result"), field=f"{location}.outcome.result")
    if result not in (0, 1, 2):
        raise ValueError(f"{location}: terminal result must be 0, 1, or 2")
    won = _require_bool(outcome.get("won"), field=f"{location}.outcome.won")
    lost = _require_bool(outcome.get("lost"), field=f"{location}.outcome.lost")
    draw = _require_bool(outcome.get("draw"), field=f"{location}.outcome.draw")
    expected = (result == seat, result in (0, 1) and result != seat, result == 2)
    if (won, lost, draw) != expected:
        raise ValueError(f"{location}: outcome is inconsistent with result and learner seat")

    return game_id, decision_id, decision_index


def _scan_refgame_shards(
    refgames_dir: str | Path,
    *,
    generation_binding: Mapping[str, str],
    init_checkpoint_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(refgames_dir).resolve(strict=True)
    paths = sorted((root / "shards").glob("decisions_shard_*.jsonl.gz"))
    if not paths:
        raise ValueError(f"strict AWR input has no decision shards under {root}")
    receipts = []
    information_modes: Counter[str] = Counter()
    content_visibility: Counter[str] = Counter()
    row_schemas: Counter[str] = Counter()
    decision_ids: set[str] = set()
    game_indices: dict[str, list[int]] = defaultdict(list)
    for path in paths:
        rows = 0
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"refgame row must be an object: {path}:{line_number}")
                game_id, decision_id, decision_index = _validate_refgame_row(
                    row,
                    path=path,
                    line_number=line_number,
                    generation_binding=generation_binding,
                    init_checkpoint_sha256=init_checkpoint_sha256,
                )
                if decision_id in decision_ids:
                    raise ValueError(f"duplicate strict AWR decision_id: {decision_id}")
                decision_ids.add(decision_id)
                game_indices[game_id].append(decision_index)
                rows += 1
                information_modes[str(row.get("information_mode") or "unattributed")] += 1
                content_visibility[classify_decision_row_visibility(row)] += 1
                row_schemas[str(row.get("schema_version") or "unattributed")] += 1
        receipts.append(_artifact_receipt(
            path,
            schema_version="deepmew.refgame_decisions.jsonl.v1",
            status="consumed",
            rows=rows,
        ))

    for game_id, indices in game_indices.items():
        if sorted(indices) != list(range(len(indices))):
            raise ValueError(f"strict AWR game has non-contiguous decision indices: {game_id}")
    total_rows = sum(receipt["rows"] for receipt in receipts)
    audit = {
        "schema_version": INPUT_AUDIT_SCHEMA_VERSION,
        "status": "completed_passed",
        "rows_scanned": total_rows,
        "games_scanned": len(game_indices),
        "unique_decision_ids": len(decision_ids),
        "duplicate_decision_ids": 0,
        "noncontiguous_games": 0,
        "information_mode_counts": dict(sorted(information_modes.items())),
        "content_visibility_counts": dict(sorted(content_visibility.items())),
        "row_schema_counts": dict(sorted(row_schemas.items())),
        "forbidden_private_rows": 0,
        "invalid_outcome_rows": 0,
        "binding_mismatch_rows": 0,
    }

    return receipts, audit


def _shard_identity(value: Any, *, field: str) -> tuple[str, str, int]:
    receipt = _require_mapping(value, field=field)
    path = str(Path(_require_string(receipt.get("path"), field=f"{field}.path")).resolve())
    digest = _require_sha256(receipt.get("sha256"), field=f"{field}.sha256")
    rows = _require_positive_int(receipt.get("rows"), field=f"{field}.rows")

    return path, digest, rows


def _require_exact_shards(
    actual: Sequence[Mapping[str, Any]],
    expected: Any,
    *,
    field: str,
) -> None:
    if not isinstance(expected, list) or not expected:
        raise ValueError(f"{field} must be a non-empty list")
    actual_identities = sorted(
        _shard_identity(receipt, field=f"actual_shards[{index}]")
        for index, receipt in enumerate(actual)
    )
    expected_identities = sorted(
        _shard_identity(receipt, field=f"{field}[{index}]")
        for index, receipt in enumerate(expected)
    )
    if actual_identities != expected_identities:
        raise ValueError("trainer refgame shards do not exactly match the data manifest")


def _require_safety_clean(manifest: Mapping[str, Any]) -> None:
    totals = _require_mapping(manifest.get("totals"), field="data_manifest.totals")
    for key in _SAFETY_KEYS:
        value = _require_nonnegative_int(totals.get(key), field=f"data_manifest.totals.{key}")
        if value != 0:
            raise ValueError(f"data_manifest.totals.{key} is not zero")
    games = _require_positive_int(totals.get("games"), field="data_manifest.totals.games")
    completed = _require_positive_int(
        totals.get("completed"),
        field="data_manifest.totals.completed",
    )
    _require_positive_int(totals.get("rows"), field="data_manifest.totals.rows")
    if completed != games:
        raise ValueError("data manifest requires completed == games")


def _validate_generation_binding(
    manifest: Mapping[str, Any],
    generation_binding: Mapping[str, str],
    *,
    init_checkpoint_sha256: str,
) -> None:
    binding = _require_mapping(
        manifest.get("generation_binding"),
        field="data_manifest.generation_binding",
    )
    for key in ("generation_id", "snapshot_id", "champion_member_id", "feature_policy_id"):
        if binding.get(key) != generation_binding[key]:
            raise ValueError(f"data manifest generation binding mismatch for {key}")
    if binding.get("champion_artifact_sha256") != init_checkpoint_sha256:
        raise ValueError("data manifest champion checkpoint does not match strict trainer init")
    if binding.get("champion_deck_sha256") != generation_binding["candidate_deck_sha256"]:
        raise ValueError("data manifest champion deck does not match candidate deck")
    if binding.get("verified") is not True:
        raise ValueError("data manifest generation binding was not registry-verified")
    our = _require_mapping(manifest.get("our"), field="data_manifest.our")
    if our.get("league_member_id") != generation_binding["champion_member_id"]:
        raise ValueError("data manifest learner member does not match generation champion")
    policy = _require_mapping(our.get("policy_identity"), field="data_manifest.our.policy_identity")
    if policy.get("artifact_sha256") != init_checkpoint_sha256:
        raise ValueError("data manifest learner policy does not match strict trainer init")
    deck = _require_mapping(our.get("deck_binding"), field="data_manifest.our.deck_binding")
    if deck.get("sha256") != generation_binding["candidate_deck_sha256"]:
        raise ValueError("data manifest learner deck does not match candidate deck")
    feature = _require_mapping(our.get("feature_binding"), field="data_manifest.our.feature_binding")
    if feature.get("feature_policy_id") != generation_binding["feature_policy_id"]:
        raise ValueError("data manifest feature policy does not match strict trainer")


def prepare_strict_training_inputs(
    *,
    generation_binding: Mapping[str, str],
    data_manifest_path: str | Path,
    refgames_dir: str | Path,
    init_checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Validate strict generation inputs before training begins."""
    normalized_binding = {
        key: _require_string(generation_binding.get(key), field=f"generation_binding.{key}")
        for key in _GENERATION_KEYS
    }
    normalized_binding["candidate_deck_sha256"] = _require_sha256(
        normalized_binding["candidate_deck_sha256"],
        field="generation_binding.candidate_deck_sha256",
    )
    data_path, data_manifest = _read_json(data_manifest_path, field="data_manifest")
    if data_manifest.get("schema_version") != DATAGEN_SCHEMA_VERSION:
        raise ValueError(f"strict AWR data manifest must be {DATAGEN_SCHEMA_VERSION}")
    if data_manifest.get("status") != "completed":
        raise ValueError("strict AWR data manifest status must be completed")
    init_receipt = _artifact_receipt(
        init_checkpoint_path,
        schema_version="binary-checkpoint",
        status="trainer_init",
    )
    _validate_generation_binding(
        data_manifest,
        normalized_binding,
        init_checkpoint_sha256=init_receipt["sha256"],
    )
    _require_safety_clean(data_manifest)
    shard_receipts, input_audit = _scan_refgame_shards(
        refgames_dir,
        generation_binding=normalized_binding,
        init_checkpoint_sha256=init_receipt["sha256"],
    )
    _require_exact_shards(
        shard_receipts,
        data_manifest.get("shard_receipts"),
        field="data_manifest.shard_receipts",
    )
    total_rows = sum(receipt["rows"] for receipt in shard_receipts)
    if input_audit["information_mode_counts"] != {PUBLIC_INFORMATION_MODE: total_rows}:
        raise ValueError(
            "strict AWR refgames must contain only public_belief rows; "
            f"found {input_audit['information_mode_counts']}"
        )
    totals = _require_mapping(data_manifest.get("totals"), field="data_manifest.totals")
    identity = _require_mapping(data_manifest.get("identity"), field="data_manifest.identity")
    expected_counts = {
        "rows_scanned": _require_positive_int(totals.get("rows"), field="data_manifest.totals.rows"),
        "games_scanned": _require_positive_int(totals.get("games"), field="data_manifest.totals.games"),
        "unique_decision_ids": _require_positive_int(
            identity.get("unique_decision_ids"),
            field="data_manifest.identity.unique_decision_ids",
        ),
    }
    for key, expected in expected_counts.items():
        if input_audit[key] != expected:
            raise ValueError(f"strict AWR input audit {key} does not match the data manifest")
    if _require_nonnegative_int(
        identity.get("duplicate_decision_ids"),
        field="data_manifest.identity.duplicate_decision_ids",
    ) != 0:
        raise ValueError("strict AWR data manifest reports duplicate decision IDs")

    return {
        "generation_binding": normalized_binding,
        "data_manifest": _artifact_receipt(
            data_path,
            schema_version=DATAGEN_SCHEMA_VERSION,
            status="completed_safety_clean",
        ),
        "refgames_dir": str(Path(refgames_dir).resolve(strict=True)),
        "refgame_shards": shard_receipts,
        "init_checkpoint": init_receipt,
        "information_mode": PUBLIC_INFORMATION_MODE,
        "input_audit": input_audit,
    }


def _validate_training_summary(training: Any, metrics: Any) -> tuple[dict[str, int], dict[str, float]]:
    training_map = _require_mapping(training, field="training")
    normalized_training = {
        "samples": _require_positive_int(training_map.get("samples"), field="training.samples"),
        "batches": _require_positive_int(training_map.get("batches"), field="training.batches"),
    }
    metric_map = _require_mapping(metrics, field="metrics")
    if not metric_map:
        raise ValueError("metrics must be non-empty")
    normalized_metrics: dict[str, float] = {}
    for name, value in metric_map.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"metric {name!r} must be finite")
        normalized_metrics[str(name)] = float(value)

    return normalized_training, normalized_metrics


def build_strict_trainer_manifest(
    *,
    prepared_inputs: Mapping[str, Any],
    output_checkpoint_path: str | Path,
    trainer_code_path: str | Path,
    full_config: Mapping[str, Any],
    training: Mapping[str, Any],
    metrics: Mapping[str, Any],
    data_stats: Mapping[str, Any],
    train_stats: Mapping[str, Any],
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build the future-only raw trainer manifest consumed by the sealer."""
    strict_inputs = _require_mapping(prepared_inputs, field="prepared_inputs")
    config = dict(_require_mapping(full_config, field="full_config"))
    if config.get("chosen_advantage_mode") != "off":
        raise ValueError("strict generation training requires chosen_advantage_mode=off")
    if config.get("allow_privileged_chosen_advantage") is not False:
        raise ValueError("strict generation training forbids privileged chosen advantage")
    seed = _require_nonnegative_int(config.get("seed"), field="full_config.seed")
    normalized_training, normalized_metrics = _validate_training_summary(training, metrics)
    output_receipt = _artifact_receipt(
        output_checkpoint_path,
        schema_version="binary-checkpoint",
        status="trainer_output",
    )
    trainer_code = _artifact_receipt(
        trainer_code_path,
        schema_version="python-source",
        status="executed",
    )

    return {
        "schema_version": TRAINER_MANIFEST_SCHEMA_VERSION,
        "status": "completed",
        "created_at_utc": created_at_utc or _utc_now(),
        "generation_binding": dict(strict_inputs["generation_binding"]),
        "information_mode": strict_inputs["information_mode"],
        "chosen_advantage_mode": config["chosen_advantage_mode"],
        "inputs": {
            "data_manifest": dict(strict_inputs["data_manifest"]),
            "refgames_dir": strict_inputs["refgames_dir"],
            "refgame_shards": [dict(receipt) for receipt in strict_inputs["refgame_shards"]],
            "init_checkpoint": dict(strict_inputs["init_checkpoint"]),
            "input_audit": dict(strict_inputs["input_audit"]),
        },
        "output_checkpoint": output_receipt,
        "trainer": {
            "code": trainer_code,
            "full_config": config,
            "config_sha256": canonical_config_sha256(config),
            "seed": seed,
        },
        "training": normalized_training,
        "metrics": normalized_metrics,
        "data": dict(data_stats),
        "train": dict(train_stats),
        "config": config,
        "out": str(Path(output_checkpoint_path)),
    }


def _runtime_repository_root(trainer_code_path: str | Path) -> Path:
    code = Path(trainer_code_path).resolve(strict=True)
    if code.name != "run_r18_awr_trainer.py" or code.parent.name != "scripts":
        raise ValueError("strict generation receipt requires scripts/run_r18_awr_trainer.py")
    root = code.parent.parent
    if not (root / ".git").exists():
        raise ValueError("strict trainer code is not inside a DeepMew git worktree")

    return root


def _protected_runtime_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for relative in _PROTECTED_RUNTIME_SURFACES:
        path = (root / relative).resolve(strict=True)
        if path.is_file():
            files.append(path)
        else:
            files.extend(sorted(item for item in path.rglob("*") if item.is_file()))

    return files


def audit_runtime_contamination(
    *,
    trainer_code_path: str | Path,
    output_checkpoint_path: str | Path,
    output_checkpoint_sha256: str,
    candidate_member_id: str,
) -> dict[str, Any]:
    """Recompute the protected-surface non-contamination gate."""
    root = _runtime_repository_root(trainer_code_path)
    checkpoint = Path(output_checkpoint_path).resolve(strict=True)
    checkpoint_sha256 = _require_sha256(
        output_checkpoint_sha256,
        field="output_checkpoint_sha256",
    )
    if sha256_path(checkpoint) != checkpoint_sha256:
        raise ValueError("runtime audit checkpoint bytes do not match the registered candidate")
    protected_paths = [(root / relative).resolve(strict=True) for relative in _PROTECTED_RUNTIME_SURFACES]
    if any(checkpoint == path or path in checkpoint.parents for path in protected_paths):
        raise ValueError("candidate checkpoint is inside a protected runtime surface")
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *_PROTECTED_RUNTIME_SURFACES,
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    status_entries = [line for line in status.stdout.splitlines() if line.strip()]
    if status_entries:
        raise ValueError(f"protected runtime surfaces are dirty: {status_entries}")

    protected_files = _protected_runtime_files(root)
    tree = hashlib.sha256()
    reference_hits: list[str] = []
    reference_tokens = (
        checkpoint_sha256,
        _require_string(candidate_member_id, field="candidate_member_id"),
        str(checkpoint),
    )
    for path in protected_files:
        relative = path.relative_to(root).as_posix()
        digest = sha256_path(path)
        tree.update(f"{relative}\0{digest}\n".encode("utf-8"))
        if digest == checkpoint_sha256:
            reference_hits.append(f"{relative}:checkpoint_bytes")
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for token in reference_tokens:
            if token in text:
                reference_hits.append(f"{relative}:candidate_reference")
                break
    if reference_hits:
        raise ValueError(f"candidate contaminated protected runtime surfaces: {reference_hits}")

    return {
        "schema_version": RUNTIME_AUDIT_SCHEMA_VERSION,
        "status": "completed_passed",
        "repository_root": str(root),
        "protected_surfaces": list(_PROTECTED_RUNTIME_SURFACES),
        "protected_file_count": len(protected_files),
        "protected_tree_sha256": tree.hexdigest(),
        "git_status_entries": [],
        "candidate_reference_hits": [],
        "checkpoint_outside_protected_surfaces": True,
        "output_checkpoint_sha256": checkpoint_sha256,
        "candidate_member_id": candidate_member_id,
    }


def _validate_raw_trainer_manifest(
    path: str | Path,
    *,
    generation_id: str,
    snapshot_id: str,
    champion_member_id: str,
    candidate_deck_sha256: str,
    feature_policy_id: str,
    input_data_manifest_sha256: str,
    init_checkpoint_sha256: str,
    output_checkpoint_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved, manifest = _read_json(path, field="trainer_manifest")
    if manifest.get("schema_version") != TRAINER_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"trainer manifest must be {TRAINER_MANIFEST_SCHEMA_VERSION}")
    if manifest.get("status") != "completed":
        raise ValueError("trainer manifest status must be completed")
    expected_binding = {
        "generation_id": generation_id,
        "snapshot_id": snapshot_id,
        "champion_member_id": champion_member_id,
        "candidate_deck_sha256": candidate_deck_sha256,
        "feature_policy_id": feature_policy_id,
    }
    if manifest.get("generation_binding") != expected_binding:
        raise ValueError("trainer generation binding mismatch")
    if manifest.get("information_mode") != PUBLIC_INFORMATION_MODE:
        raise ValueError("trainer manifest must bind public_belief information_mode")
    if manifest.get("chosen_advantage_mode") != "off":
        raise ValueError("trainer manifest must bind chosen_advantage_mode=off")
    inputs = _require_mapping(manifest.get("inputs"), field="trainer_manifest.inputs")
    data_path, _ = _verify_artifact_receipt(
        inputs.get("data_manifest"),
        field="trainer_manifest.inputs.data_manifest",
        expected_sha256=input_data_manifest_sha256,
    )
    init_path, _ = _verify_artifact_receipt(
        inputs.get("init_checkpoint"),
        field="trainer_manifest.inputs.init_checkpoint",
        expected_sha256=init_checkpoint_sha256,
    )
    output_path, _ = _verify_artifact_receipt(
        manifest.get("output_checkpoint"),
        field="trainer_manifest.output_checkpoint",
        expected_sha256=output_checkpoint_sha256,
    )
    prepared = prepare_strict_training_inputs(
        generation_binding=expected_binding,
        data_manifest_path=data_path,
        refgames_dir=inputs.get("refgames_dir"),
        init_checkpoint_path=init_path,
    )
    actual_shards = prepared["refgame_shards"]
    _require_exact_shards(
        actual_shards,
        inputs.get("refgame_shards"),
        field="trainer_manifest.inputs.refgame_shards",
    )
    if inputs.get("input_audit") != prepared["input_audit"]:
        raise ValueError("trainer input audit does not match the revalidated refgame content")
    trainer = _require_mapping(manifest.get("trainer"), field="trainer_manifest.trainer")
    code_path, code_receipt = _verify_artifact_receipt(
        trainer.get("code"),
        field="trainer_manifest.trainer.code",
    )
    _runtime_repository_root(code_path)
    config = _require_mapping(trainer.get("full_config"), field="trainer_manifest.trainer.full_config")
    config_hash = canonical_config_sha256(config)
    if trainer.get("config_sha256") != config_hash:
        raise ValueError("trainer full config hash mismatch")
    seed = _require_nonnegative_int(trainer.get("seed"), field="trainer_manifest.trainer.seed")
    if config.get("seed") != seed:
        raise ValueError("trainer seed does not match full config")
    if config.get("chosen_advantage_mode") != "off":
        raise ValueError("trainer full config does not keep chosen advantage off")
    if config.get("allow_privileged_chosen_advantage") is not False:
        raise ValueError("trainer full config allows privileged chosen advantage")
    training, metrics = _validate_training_summary(
        manifest.get("training"),
        manifest.get("metrics"),
    )

    return manifest, {
        "trainer_manifest": _artifact_receipt(
            resolved,
            schema_version=TRAINER_MANIFEST_SCHEMA_VERSION,
            status="completed",
        ),
        "data_manifest": dict(prepared["data_manifest"]),
        "refgame_shards": actual_shards,
        "input_audit": dict(prepared["input_audit"]),
        "init_checkpoint": _artifact_receipt(
            init_path,
            schema_version="binary-checkpoint",
            status="trainer_init",
        ),
        "output_checkpoint": _artifact_receipt(
            output_path,
            schema_version="binary-checkpoint",
            status="registered_candidate",
        ),
        "trainer_code": dict(code_receipt),
        "full_config": dict(config),
        "config_sha256": config_hash,
        "seed": seed,
        "training": training,
        "metrics": metrics,
    }


def seal_awr_train_receipt(
    *,
    generation_id: str,
    snapshot_id: str,
    champion_member_id: str,
    candidate_member_id: str,
    candidate_deck_sha256: str,
    feature_policy_id: str,
    input_data_manifest_sha256: str,
    init_checkpoint_sha256: str,
    output_checkpoint_sha256: str,
    trainer_manifest_path: str | Path,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Seal a strict raw trainer manifest into the generation receipt."""
    generation_id = _require_string(generation_id, field="generation_id")
    snapshot_id = _require_string(snapshot_id, field="snapshot_id")
    champion_member_id = _require_string(champion_member_id, field="champion_member_id")
    candidate_member_id = _require_string(candidate_member_id, field="candidate_member_id")
    candidate_deck_sha256 = _require_sha256(candidate_deck_sha256, field="candidate_deck_sha256")
    feature_policy_id = _require_string(feature_policy_id, field="feature_policy_id")
    input_data_manifest_sha256 = _require_sha256(
        input_data_manifest_sha256,
        field="input_data_manifest_sha256",
    )
    init_checkpoint_sha256 = _require_sha256(
        init_checkpoint_sha256,
        field="init_checkpoint_sha256",
    )
    output_checkpoint_sha256 = _require_sha256(
        output_checkpoint_sha256,
        field="output_checkpoint_sha256",
    )
    _manifest, validated = _validate_raw_trainer_manifest(
        trainer_manifest_path,
        generation_id=generation_id,
        snapshot_id=snapshot_id,
        champion_member_id=champion_member_id,
        candidate_deck_sha256=candidate_deck_sha256,
        feature_policy_id=feature_policy_id,
        input_data_manifest_sha256=input_data_manifest_sha256,
        init_checkpoint_sha256=init_checkpoint_sha256,
        output_checkpoint_sha256=output_checkpoint_sha256,
    )
    runtime_audit = audit_runtime_contamination(
        trainer_code_path=validated["trainer_code"]["path"],
        output_checkpoint_path=validated["output_checkpoint"]["path"],
        output_checkpoint_sha256=output_checkpoint_sha256,
        candidate_member_id=candidate_member_id,
    )

    return {
        "schema_version": TRAIN_RECEIPT_SCHEMA_VERSION,
        "status": "completed",
        "created_at_utc": created_at_utc or _utc_now(),
        "generation_id": generation_id,
        "snapshot_id": snapshot_id,
        "champion_member_id": champion_member_id,
        "candidate_member_id": candidate_member_id,
        "candidate_deck_sha256": candidate_deck_sha256,
        "feature_policy_id": feature_policy_id,
        "information_mode": PUBLIC_INFORMATION_MODE,
        "chosen_advantage_mode": "off",
        "input_data_manifest_sha256": input_data_manifest_sha256,
        "init_checkpoint_sha256": init_checkpoint_sha256,
        "output_checkpoint_sha256": output_checkpoint_sha256,
        "gates": {
            "leakage_passed": True,
            "runtime_contamination_passed": True,
            "safety_clean": True,
        },
        "training": validated["training"],
        "metrics": validated["metrics"],
        "receipts": {
            "data_manifest": validated["data_manifest"],
            "refgame_shards": validated["refgame_shards"],
            "input_audit": validated["input_audit"],
            "init_checkpoint": validated["init_checkpoint"],
            "output_checkpoint": validated["output_checkpoint"],
            "trainer_manifest": validated["trainer_manifest"],
            "trainer_code": validated["trainer_code"],
            "trainer_code_sha256": validated["trainer_code"]["sha256"],
            "full_config": validated["full_config"],
            "config_sha256": validated["config_sha256"],
            "seed": validated["seed"],
            "gate_evidence": {
                "leakage": validated["input_audit"],
                "runtime_contamination": runtime_audit,
                "safety": validated["data_manifest"],
            },
        },
    }


def _require_recorded_artifact_matches(
    recorded: Any,
    actual: Mapping[str, Any],
    *,
    field: str,
) -> None:
    recorded_map = _require_mapping(recorded, field=field)
    recorded_identity = (
        str(Path(_require_string(recorded_map.get("path"), field=f"{field}.path")).resolve()),
        _require_sha256(recorded_map.get("sha256"), field=f"{field}.sha256"),
    )
    actual_identity = (str(Path(actual["path"]).resolve()), str(actual["sha256"]))
    if recorded_identity != actual_identity:
        raise ValueError(f"{field} does not match its validated artifact")
    if "rows" in actual:
        if _require_positive_int(recorded_map.get("rows"), field=f"{field}.rows") != actual["rows"]:
            raise ValueError(f"{field}.rows does not match its validated artifact")


def validate_awr_train_receipt(payload: Mapping[str, Any]) -> None:
    """Revalidate a sealed receipt and all evidence bytes it names."""
    receipt = _require_mapping(payload, field="train_receipt")
    if receipt.get("schema_version") != TRAIN_RECEIPT_SCHEMA_VERSION:
        raise ValueError(f"train receipt must be {TRAIN_RECEIPT_SCHEMA_VERSION}")
    if receipt.get("status") != "completed":
        raise ValueError("train receipt status must be completed")
    if receipt.get("information_mode") != PUBLIC_INFORMATION_MODE:
        raise ValueError("train receipt information mode must be public_belief")
    if receipt.get("chosen_advantage_mode") != "off":
        raise ValueError("train receipt chosen advantage mode must be off")
    gates = _require_mapping(receipt.get("gates"), field="train_receipt.gates")
    for gate in ("leakage_passed", "runtime_contamination_passed", "safety_clean"):
        if gates.get(gate) is not True:
            raise ValueError(f"train receipt gate {gate} did not pass")
    training, metrics = _validate_training_summary(receipt.get("training"), receipt.get("metrics"))
    receipts = _require_mapping(receipt.get("receipts"), field="train_receipt.receipts")
    _raw_manifest, validated = _validate_raw_trainer_manifest(
        _require_mapping(receipts.get("trainer_manifest"), field="receipts.trainer_manifest").get("path"),
        generation_id=_require_string(receipt.get("generation_id"), field="generation_id"),
        snapshot_id=_require_string(receipt.get("snapshot_id"), field="snapshot_id"),
        champion_member_id=_require_string(
            receipt.get("champion_member_id"),
            field="champion_member_id",
        ),
        candidate_deck_sha256=_require_sha256(
            receipt.get("candidate_deck_sha256"),
            field="candidate_deck_sha256",
        ),
        feature_policy_id=_require_string(receipt.get("feature_policy_id"), field="feature_policy_id"),
        input_data_manifest_sha256=_require_sha256(
            receipt.get("input_data_manifest_sha256"),
            field="input_data_manifest_sha256",
        ),
        init_checkpoint_sha256=_require_sha256(
            receipt.get("init_checkpoint_sha256"),
            field="init_checkpoint_sha256",
        ),
        output_checkpoint_sha256=_require_sha256(
            receipt.get("output_checkpoint_sha256"),
            field="output_checkpoint_sha256",
        ),
    )
    if training != validated["training"] or metrics != validated["metrics"]:
        raise ValueError("sealed training summary does not match the raw trainer manifest")
    for name in (
        "data_manifest",
        "init_checkpoint",
        "output_checkpoint",
        "trainer_manifest",
        "trainer_code",
    ):
        _require_recorded_artifact_matches(
            receipts.get(name),
            validated[name],
            field=f"receipts.{name}",
        )
    recorded_shards = receipts.get("refgame_shards")
    if not isinstance(recorded_shards, list) or not recorded_shards:
        raise ValueError("receipts.refgame_shards must be a non-empty list")
    _require_exact_shards(
        validated["refgame_shards"],
        recorded_shards,
        field="receipts.refgame_shards",
    )
    if receipts.get("input_audit") != validated["input_audit"]:
        raise ValueError("sealed input audit does not match revalidated refgame content")
    if receipts.get("full_config") != validated["full_config"]:
        raise ValueError("sealed full config does not match the raw trainer manifest")
    if receipts.get("config_sha256") != validated["config_sha256"]:
        raise ValueError("sealed config hash does not match the raw trainer manifest")
    if receipts.get("trainer_code_sha256") != validated["trainer_code"]["sha256"]:
        raise ValueError("sealed trainer code hash does not match the raw trainer manifest")
    if receipts.get("seed") != validated["seed"]:
        raise ValueError("sealed seed does not match the raw trainer manifest")
    gate_evidence = _require_mapping(receipts.get("gate_evidence"), field="receipts.gate_evidence")
    recorded_leakage = _require_mapping(
        gate_evidence.get("leakage"),
        field="gate_evidence.leakage",
    )
    if recorded_leakage != validated["input_audit"]:
        raise ValueError("sealed leakage evidence does not match revalidated refgame content")
    recorded_runtime = _require_mapping(
        gate_evidence.get("runtime_contamination"),
        field="gate_evidence.runtime_contamination",
    )
    runtime = audit_runtime_contamination(
        trainer_code_path=validated["trainer_code"]["path"],
        output_checkpoint_path=validated["output_checkpoint"]["path"],
        output_checkpoint_sha256=receipt["output_checkpoint_sha256"],
        candidate_member_id=receipt["candidate_member_id"],
    )
    if recorded_runtime != runtime:
        raise ValueError("sealed runtime-contamination audit does not match live protected surfaces")
    _require_recorded_artifact_matches(
        gate_evidence.get("safety"),
        validated["data_manifest"],
        field="gate_evidence.safety",
    )


def write_awr_train_receipt_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Validate and atomically create a receipt without overwriting evidence."""
    validate_awr_train_receipt(payload)
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") == encoded:
            return
        raise FileExistsError(f"refusing to overwrite existing train receipt: {target}")
    temporary = target.with_name(f".{target.name}.tmp.{uuid.uuid4().hex}")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
