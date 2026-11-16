"""Resumable production miner for aggregate-only privileged branch-Q rows.

The privileged simulator is confined to ``run_privileged_teacher_smoke.py``.
This process receives only the smoke's actor-visible aggregate row and its
boolean/count receipt, then writes manifest-bound immutable training shards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.q_guided.branching import assert_private_content_absent  # noqa: E402
from deepmew.q_guided.teacher_contract import validate_safe_shard  # noqa: E402
from deepmew.replay.contract.preflight import (  # noqa: E402
    require_accepted_canonical_decision_receipt,
)
from deepmew.q_guided.training import (  # noqa: E402
    ELIGIBLE_TEACHER_STATUS,
    TEACHER_MANIFEST_SCHEMA_VERSION,
    ShardBinding,
    load_manifest_bound_shards,
)


SCHEMA_VERSION = "deepmew.q_guided.branch_miner.v1"
PROGRESS_SCHEMA_VERSION = "deepmew.q_guided.branch_miner_progress.v1"
DEFAULT_BATCH_ROWS = 32
DEFAULT_WORLD_SAMPLES = 2
DEFAULT_ROLLOUTS_PER_ACTION = 2
DEFAULT_MAX_USEFUL_SECONDS = 8.0 * 60.0 * 60.0
DEFAULT_WORKER_TIMEOUT_SECONDS = 900
SAFETY_KEYS = ("illegal_actions", "crashes", "timeouts")


class MinerSafetyError(RuntimeError):
    """Fail-closed production-miner error without private simulator content."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, _canonical_bytes(value) + b"\n")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MinerSafetyError(f"could not read miner artifact {path.name}") from error
    if not isinstance(value, dict):
        raise MinerSafetyError(f"miner artifact {path.name} is not an object")

    return value


def _read_one_jsonl(path: Path) -> dict[str, Any]:
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        value = json.loads(lines[0]) if len(lines) == 1 else None
    except (OSError, json.JSONDecodeError) as error:
        raise MinerSafetyError("teacher row is unreadable") from error
    if not isinstance(value, dict):
        raise MinerSafetyError("teacher output must contain exactly one object row")
    validate_safe_shard(value)
    assert_private_content_absent(value)

    return value


def _student_signature(row: dict[str, Any]) -> str:
    return _sha256_bytes(
        _canonical_bytes(
            {
                "actor_visible_root_observation": row["actor_visible_root_observation"],
                "actor_visible_legal_options": row["actor_visible_legal_options"],
            }
        )
    )


def _student_source(row: dict[str, Any]) -> str:
    return _sha256_bytes(_canonical_bytes(row["actor_visible_root_observation"]))


def _row_receipt(row: dict[str, Any], smoke_receipt: dict[str, Any]) -> dict[str, Any]:
    if smoke_receipt.get("status") != "passed_end_to_end_smoke":
        raise MinerSafetyError("teacher smoke did not pass end to end")
    if smoke_receipt.get("aggregation_status") != ELIGIBLE_TEACHER_STATUS:
        raise MinerSafetyError("teacher row is not an eligible aggregate target")
    if smoke_receipt.get("policy_control_eligible") is not False:
        raise MinerSafetyError("teacher receipt crossed the research-only control boundary")
    safety = smoke_receipt.get("safety")
    if not isinstance(safety, dict) or any(safety.get(name) != 0 for name in SAFETY_KEYS):
        raise MinerSafetyError("teacher safety counters are not clean")
    checks = smoke_receipt.get("checks")
    if not isinstance(checks, dict) or not checks or any(value is not True for value in checks.values()):
        raise MinerSafetyError("teacher safety checks are not all true")
    world_samples = smoke_receipt.get("world_sample_count")
    paired_samples = smoke_receipt.get("paired_rollouts_per_action")
    if (
        isinstance(world_samples, bool)
        or not isinstance(world_samples, int)
        or world_samples < 2
        or isinstance(paired_samples, bool)
        or not isinstance(paired_samples, int)
        or paired_samples < 2
    ):
        raise MinerSafetyError("teacher aggregation sample coverage is insufficient")
    row_bytes = _canonical_bytes(row)

    return {
        "status": ELIGIBLE_TEACHER_STATUS,
        "world_sample_count": world_samples,
        "paired_sample_count": paired_samples,
        "row_sha256": _sha256_bytes(row_bytes),
        "actor_visible_signature_sha256": _student_signature(row),
        "actor_visible_source_sha256": _student_source(row),
    }


def _config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "miner_id": args.miner_id,
        "deck0": str(Path(args.deck0).resolve()) if args.deck0 else "default",
        "deck1": str(Path(args.deck1).resolve()) if args.deck1 else "default",
        "v_checkpoint": str(Path(args.v_checkpoint).resolve()) if args.v_checkpoint else "default",
        "v_calibrator_dir": str(Path(args.v_calibrator_dir).resolve()) if args.v_calibrator_dir else "default",
        "policy0": args.policy0,
        "policy1": args.policy1,
        "minimum_source_turn": args.minimum_source_turn,
        "maximum_source_turn": args.maximum_source_turn,
        "source_seed": args.source_seed,
        "branch_seed": args.branch_seed,
        "world_samples": args.world_samples,
        "rollouts_per_action": args.rollouts_per_action,
        "root_action_cap": args.root_action_cap,
        "batch_rows": args.batch_rows,
        "split": args.split,
        "execution_mode": "fork",
    }


def _config_sha256(args: argparse.Namespace) -> str:
    return _sha256_bytes(_canonical_bytes(_config(args)))


def _empty_state(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "status": "running",
        "config_sha256": _config_sha256(args),
        "attempts": 0,
        "eligible_rows": 0,
        "duplicate_rows": 0,
        "safe_no_root_failures": 0,
        "cumulative_useful_seconds": 0.0,
        "v0_validated": False,
        "next_shard_index": 0,
        "shard_count": 0,
        "safety": {name: 0 for name in SAFETY_KEYS},
    }


def _pending_path(run_dir: Path) -> Path:
    return run_dir / "pending.json"


def _load_pending(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = _pending_path(run_dir)
    if not path.is_file():
        return [], []
    value = _read_json(path)
    if set(value) != {"rows", "row_receipts"}:
        raise MinerSafetyError("pending artifact field inventory mismatch")
    rows = value["rows"]
    receipts = value["row_receipts"]
    if not isinstance(rows, list) or not isinstance(receipts, list) or len(rows) != len(receipts):
        raise MinerSafetyError("pending artifact receipt coverage mismatch")
    for row, receipt in zip(rows, receipts, strict=True):
        if not isinstance(row, dict) or not isinstance(receipt, dict):
            raise MinerSafetyError("pending artifact contains an invalid row")
        validate_safe_shard(row)
        assert_private_content_absent(row)
        if receipt.get("row_sha256") != _sha256_bytes(_canonical_bytes(row)):
            raise MinerSafetyError("pending row hash mismatch")

    return rows, receipts


def _existing_bindings(run_dir: Path) -> list[ShardBinding]:
    bindings = []
    for shard in sorted((run_dir / "shards").glob("shard_*.jsonl")):
        manifest = shard.with_name(f"{shard.stem}.manifest.json")
        if not manifest.is_file():
            raise MinerSafetyError("immutable shard is missing its manifest")
        bindings.append(ShardBinding(shard, manifest))
    if bindings:
        load_manifest_bound_shards(bindings)

    return bindings


def _load_or_create_state(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    progress_path = run_dir / "progress.json"
    if args.resume:
        if not progress_path.is_file():
            raise MinerSafetyError("--resume requires progress.json")
        state = _read_json(progress_path)
        if state.get("schema_version") != PROGRESS_SCHEMA_VERSION:
            raise MinerSafetyError("miner progress schema mismatch")
        if state.get("config_sha256") != _config_sha256(args):
            raise MinerSafetyError("miner resume config hash mismatch")
        bindings = _existing_bindings(run_dir)
        rows, _ = _load_pending(run_dir)
        manifest_rows = sum(_read_json(binding.manifest_path)["row_count"] for binding in bindings)
        if state.get("eligible_rows") != manifest_rows + len(rows):
            raise MinerSafetyError("miner resume row count mismatch")
        if state.get("next_shard_index") != len(bindings):
            raise MinerSafetyError("miner resume shard cursor mismatch")

        return state
    if run_dir.exists() and any(run_dir.iterdir()):
        raise MinerSafetyError("new miner run directory must be absent or empty")
    run_dir.mkdir(parents=True, exist_ok=True)
    state = _empty_state(args)
    _atomic_json(run_dir / "config.json", {**_config(args), "config_sha256": state["config_sha256"]})
    _atomic_json(progress_path, state)

    return state


def _write_shard(
    run_dir: Path,
    state: dict[str, Any],
    rows: list[dict[str, Any]],
    receipts: list[dict[str, Any]],
) -> None:
    if not rows or len(rows) != len(receipts):
        raise MinerSafetyError("cannot write an empty or uncovered shard")
    shards = run_dir / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    stem = f"shard_{state['next_shard_index']:05d}"
    shard_path = shards / f"{stem}.jsonl"
    manifest_path = shards / f"{stem}.manifest.json"
    if shard_path.exists() or manifest_path.exists():
        raise MinerSafetyError("immutable shard target already exists")
    shard_bytes = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    manifest = {
        "schema_version": TEACHER_MANIFEST_SCHEMA_VERSION,
        "split": state["split"],
        "shard_sha256": _sha256_bytes(shard_bytes),
        "row_count": len(rows),
        "row_receipts": receipts,
    }
    _atomic_bytes(shard_path, shard_bytes)
    _atomic_json(manifest_path, manifest)
    load_manifest_bound_shards([ShardBinding(shard_path, manifest_path)])
    try:
        shard_path.chmod(0o444)
        manifest_path.chmod(0o444)
    except OSError:
        pass
    state["next_shard_index"] += 1
    state["shard_count"] += 1
    _pending_path(run_dir).unlink(missing_ok=True)


def _smoke_command(args: argparse.Namespace, out_dir: Path, attempt: int, skip_v0: bool) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/run_privileged_teacher_smoke.py"),
        "--canonical-decision-receipt",
        args.canonical_decision_receipt,
        "--out-dir",
        str(out_dir),
        "--execution-mode",
        "fork",
        "--policy0",
        args.policy0,
        "--policy1",
        args.policy1,
        "--source-games",
        str(args.source_games),
        "--source-seed",
        str(args.source_seed + attempt),
        "--minimum-source-turn",
        str(args.minimum_source_turn),
        "--maximum-source-turn",
        str(args.maximum_source_turn),
        "--branch-seed",
        str(args.branch_seed + attempt * 10_000),
        "--world-samples",
        str(args.world_samples),
        "--rollouts-per-action",
        str(args.rollouts_per_action),
        "--root-action-cap",
        str(args.root_action_cap),
        "--worker-timeout-seconds",
        str(args.worker_timeout_seconds),
    ]
    if args.deck0:
        command.extend(("--deck0", args.deck0))
    if args.deck1:
        command.extend(("--deck1", args.deck1))
    if args.v_checkpoint:
        command.extend(("--v-checkpoint", args.v_checkpoint))
    if args.v_calibrator_dir:
        command.extend(("--v-calibrator-dir", args.v_calibrator_dir))
    if skip_v0:
        command.append("--skip-v0-score-after-validated-parent")

    return command


def _run_smoke(args: argparse.Namespace, run_dir: Path, state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    attempt = int(state["attempts"])
    scratch = run_dir / "scratch" / f"attempt_{attempt:08d}"
    scratch.parent.mkdir(parents=True, exist_ok=True)
    failure_path = scratch.with_name(f".{scratch.name}.failure.json")
    worker_path = scratch.with_name(f".{scratch.name}.worker")
    for path in (scratch, failure_path, worker_path):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    command = _smoke_command(args, scratch, attempt, bool(state["v0_validated"]))
    environment = dict(os.environ)
    environment["PYTHONIOENCODING"] = "utf-8"
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            check=False,
            timeout=args.worker_timeout_seconds + 60,
            env=environment,
        )
    except subprocess.TimeoutExpired as error:
        raise MinerSafetyError("teacher invocation timed out") from error
    elapsed = time.perf_counter() - started
    state["attempts"] += 1
    state["cumulative_useful_seconds"] += elapsed
    if completed.returncode != 0:
        category = "teacher invocation failed without a receipt"
        if failure_path.is_file():
            failure = _read_json(failure_path)
            category = str(failure.get("failure_category") or category)
        if "no bounded actor-visible multi-action root was found" in category:
            state["safe_no_root_failures"] += 1
            failure_path.unlink(missing_ok=True)
            if worker_path.is_dir():
                shutil.rmtree(worker_path)

            return None
        raise MinerSafetyError(category)
    if completed.stderr.strip():
        raise MinerSafetyError("teacher invocation emitted stderr")
    row = _read_one_jsonl(scratch / "branch_targets.jsonl")
    receipt = _read_json(scratch / "smoke_receipt.json")
    row_receipt = _row_receipt(row, receipt)
    if not state["v0_validated"]:
        probability = receipt.get("root_v0_probability")
        if not isinstance(probability, (int, float)) or isinstance(probability, bool) or not math.isfinite(probability):
            raise MinerSafetyError("first eligible row did not validate V0 in the miner parent")
        if receipt.get("v0_score_status") != "validated_in_parent":
            raise MinerSafetyError("first eligible row lacks V0 validation provenance")
        state["v0_validated"] = True
    elif receipt.get("v0_score_status") != "delegated_to_validated_miner_parent":
        raise MinerSafetyError("subsequent row did not use the validated V0 delegation")
    shutil.rmtree(scratch)

    return row, row_receipt


def _write_run_manifest(args: argparse.Namespace, run_dir: Path, state: dict[str, Any]) -> None:
    shards = []
    for binding in _existing_bindings(run_dir):
        manifest = _read_json(binding.manifest_path)
        shards.append(
            {
                "shard": str(binding.shard_path.relative_to(run_dir)).replace("\\", "/"),
                "manifest": str(binding.manifest_path.relative_to(run_dir)).replace("\\", "/"),
                "shard_sha256": manifest["shard_sha256"],
                "manifest_sha256": _sha256_file(binding.manifest_path),
                "row_count": manifest["row_count"],
            }
        )
    value = {
        "schema_version": SCHEMA_VERSION,
        "status": state["status"],
        "classification": "quarantine",
        "config_sha256": state["config_sha256"],
        "miner_id": args.miner_id,
        "split": args.split,
        "eligible_rows": state["eligible_rows"],
        "cumulative_useful_seconds": state["cumulative_useful_seconds"],
        "safety": state["safety"],
        "shards": shards,
    }
    _atomic_json(run_dir / "manifest.json", value)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.minimum_source_turn < 1:
        raise MinerSafetyError("minimum source turn must be positive")
    if args.maximum_source_turn < 0 or (
        args.maximum_source_turn and args.maximum_source_turn < args.minimum_source_turn
    ):
        raise MinerSafetyError("maximum source turn must be zero or at least the minimum")
    if args.world_samples < 2 or args.rollouts_per_action < 2:
        raise MinerSafetyError("aggregate teacher coverage requires at least two worlds and paired samples")
    if args.batch_rows < 1 or args.max_rows < 0:
        raise MinerSafetyError("row limits are invalid")
    if not math.isfinite(args.max_useful_seconds) or args.max_useful_seconds <= 0.0:
        raise MinerSafetyError("maximum useful seconds must be finite and positive")
    run_dir = Path(args.run_dir).resolve()
    state = _load_or_create_state(args, run_dir)
    state["split"] = args.split
    pending_rows, pending_receipts = _load_pending(run_dir)
    signatures = {
        receipt["actor_visible_signature_sha256"]
        for binding in _existing_bindings(run_dir)
        for receipt in _read_json(binding.manifest_path)["row_receipts"]
    }
    signatures.update(receipt["actor_visible_signature_sha256"] for receipt in pending_receipts)
    state["status"] = "running"
    _atomic_json(run_dir / "progress.json", state)
    try:
        while state["cumulative_useful_seconds"] < args.max_useful_seconds:
            if args.max_rows and state["eligible_rows"] >= args.max_rows:
                break
            result = _run_smoke(args, run_dir, state)
            if result is not None:
                row, receipt = result
                signature = receipt["actor_visible_signature_sha256"]
                if signature in signatures:
                    state["duplicate_rows"] += 1
                else:
                    signatures.add(signature)
                    pending_rows.append(row)
                    pending_receipts.append(receipt)
                    state["eligible_rows"] += 1
                    _atomic_json(_pending_path(run_dir), {"rows": pending_rows, "row_receipts": pending_receipts})
                    if len(pending_rows) >= args.batch_rows:
                        _write_shard(run_dir, state, pending_rows, pending_receipts)
                        pending_rows, pending_receipts = [], []
            _atomic_json(run_dir / "progress.json", state)
            _write_run_manifest(args, run_dir, state)
        if pending_rows:
            _write_shard(run_dir, state, pending_rows, pending_receipts)
        state["status"] = "completed_useful_floor" if state["cumulative_useful_seconds"] >= args.max_useful_seconds else "completed_row_limit"
        _atomic_json(run_dir / "progress.json", state)
        _write_run_manifest(args, run_dir, state)
    except BaseException:
        state["status"] = "failed_safe"
        _atomic_json(run_dir / "progress.json", state)
        _write_run_manifest(args, run_dir, state)
        raise

    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-decision-receipt", default="")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--miner-id", required=True, choices=("M1", "M2"))
    parser.add_argument("--deck0", default="")
    parser.add_argument("--deck1", default="")
    parser.add_argument("--v-checkpoint", default="")
    parser.add_argument("--v-calibrator-dir", default="")
    parser.add_argument("--policy0", default="v2.4")
    parser.add_argument("--policy1", default="v2.4")
    parser.add_argument("--minimum-source-turn", type=int, required=True)
    parser.add_argument("--maximum-source-turn", type=int, default=0)
    parser.add_argument("--source-games", type=int, default=4)
    parser.add_argument("--source-seed", type=int, required=True)
    parser.add_argument("--branch-seed", type=int, required=True)
    parser.add_argument("--world-samples", type=int, default=DEFAULT_WORLD_SAMPLES)
    parser.add_argument("--rollouts-per-action", type=int, default=DEFAULT_ROLLOUTS_PER_ACTION)
    parser.add_argument("--root-action-cap", type=int, default=6)
    parser.add_argument("--batch-rows", type=int, default=DEFAULT_BATCH_ROWS)
    parser.add_argument("--max-useful-seconds", type=float, default=DEFAULT_MAX_USEFUL_SECONDS)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--worker-timeout-seconds", type=int, default=DEFAULT_WORKER_TIMEOUT_SECONDS)
    parser.add_argument("--split", default="train")
    parser.add_argument("--resume", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    accepted_receipt = require_accepted_canonical_decision_receipt(
        args.canonical_decision_receipt,
        consumer=__name__,
    )
    args.canonical_decision_receipt = str(accepted_receipt.path)
    result = run(args)
    print(json.dumps(result, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
