"""Budget-gated Modal launcher for the privileged-teacher CPU smoke.

This module runs exactly one bounded, CPU-only smoke.  The remote function
returns only the aggregate branch shard and its safety receipt; private
simulator state never crosses the remote function boundary.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import modal


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PROJECT = ".research/modal/q_runtime"
GPU_RUNTIME_PROJECT = ".research/modal/q_gpu_runtime"
TOP_STARYU_DECK = ROOT / ".research/curriculum/decks/top_staryu_windecks_b15853cdfb83.deck.csv"
FROZEN_V0_CHECKPOINT = Path(
    r"D:\DeepMew_artifacts\datasets\value_q_canonical_v1\value_lab_runs"
    r"\phasef4_grouped_crossfit_raw_s1_s915002_v3\full_train\value.pt"
)
FROZEN_V0_CALIBRATOR = (
    ROOT / ".research/value_q_experiment/artifacts/phasef5_grouped_calibration_raw_s1_s915002_v3"
)
R7_CHECKPOINT = Path(
    r"D:\DeepMew_artifacts\checkpoints\r22-staryu-full-league-a4-8c-r7-BEST.pt"
)
R7_CHECKPOINT_SHA256 = "e293433756fc7edad865f887f9e130d5cec1e4d4067da30446ddebf9b714aee2"
FROZEN_V0_CHECKPOINT_SHA256 = "cd1339e14fa943382069128bc9bf8657de9c441286a49bdef0f42450d56eb0a9"
REMOTE_V0_CHECKPOINT = PurePosixPath("/repo/.research/value_q_experiment/frozen_v0/value.pt")
REMOTE_V0_CALIBRATOR = PurePosixPath(
    "/repo/.research/value_q_experiment/artifacts/phasef5_grouped_calibration_raw_s1_s915002_v3"
)
REMOTE_TOP_STARYU_DECK = PurePosixPath(
    "/repo/.research/curriculum/decks/top_staryu_windecks_b15853cdfb83.deck.csv"
)
REMOTE_R7_CHECKPOINT = PurePosixPath("/repo/.research/q_guided_starmie/r7_selected.pt")
REMOTE_OUTPUT = PurePosixPath("/tmp/modal_privileged_teacher_smoke_20260712")
REMOTE_FAILURE_RECEIPT = PurePosixPath(
    "/tmp/.modal_privileged_teacher_smoke_20260712.failure.json"
)
DEFAULT_OUTPUT = ROOT / ".research/q_guided_starmie/modal_privileged_teacher_smoke_20260712"
CONTROL_DIR = ROOT / ".research/q_guided_starmie/modal_first_wave_20260712"
VOLUME_NAME = "deepmew-q-research"
VOLUME_ROOT = PurePosixPath("/modal_q")
RUN_ROOT = VOLUME_ROOT / "runs"
TRANCHE_ROOT = VOLUME_ROOT / "tranches"

CPU_CORES = 4.0
MEMORY_MIB = 8_192
TIMEOUT_SECONDS = 900
FIRST_WAVE_TIMEOUT_SECONDS = 36_000
MINER_USEFUL_SECONDS = 8 * 60 * 60
M3_USEFUL_SECONDS = 8 * 60 * 60
M4_USEFUL_SECONDS = 4 * 60 * 60
MINER_BILLABLE_BOUND_SECONDS = 30_600
M3_BILLABLE_BOUND_SECONDS = 30_600
M4_BILLABLE_BOUND_SECONDS = 16_200
M4_RESUME_MINIMUM_PRIOR_GPU_SECONDS = 14_000
M4_RESUME_BILLABLE_BOUND_SECONDS = 600
M4_RESUME_TIMEOUT_SECONDS = 900
M2_RESUME_MINIMUM_PRIOR_USEFUL_SECONDS = 22_293.8424244019
M2_RESUME_MINIMUM_ELIGIBLE_ROWS = 4_319
M2_RESUME_REQUIRED_CONFIG_SHA256 = "9e8d5d17729dfe66ef434aaa9ca274e545f90caf0d83425e42eb5ae0c0c2fd0f"
M2_RESUME_BILLABLE_BOUND_SECONDS = MINER_BILLABLE_BOUND_SECONDS
VOLUME_COMMIT_INTERVAL_SECONDS = 5 * 60
CPU_USD_PER_CORE_SECOND = 0.0000131
MEMORY_USD_PER_GIB_SECOND = 0.00000222
L4_USD_PER_SECOND = 0.000222
T4_USD_PER_SECOND = 0.000164
MAX_SCHEDULE_USD = 24.0
MIN_RESERVE_USD = 5.0
M2_RESUME_MINIMUM_RESERVE_USD = 0.0  # Narrow 2026-07-13 user authorization for this resume only.
DEFAULT_AVAILABLE_USD = MAX_SCHEDULE_USD
WORST_CASE_USD = TIMEOUT_SECONDS * (
    CPU_CORES * CPU_USD_PER_CORE_SECOND
    + (MEMORY_MIB / 1_024) * MEMORY_USD_PER_GIB_SECOND
)
MINER_WORST_CASE_USD = MINER_BILLABLE_BOUND_SECONDS * (
    CPU_CORES * CPU_USD_PER_CORE_SECOND
    + (MEMORY_MIB / 1_024) * MEMORY_USD_PER_GIB_SECOND
)
M3_WORST_CASE_USD = M3_BILLABLE_BOUND_SECONDS * (
    L4_USD_PER_SECOND
    + CPU_CORES * CPU_USD_PER_CORE_SECOND
    + (16_384 / 1_024) * MEMORY_USD_PER_GIB_SECOND
)
M4_WORST_CASE_USD = M4_BILLABLE_BOUND_SECONDS * (
    T4_USD_PER_SECOND
    + CPU_CORES * CPU_USD_PER_CORE_SECOND
    + (16_384 / 1_024) * MEMORY_USD_PER_GIB_SECOND
)
M4_RESUME_WORST_CASE_USD = M4_RESUME_TIMEOUT_SECONDS * (
    T4_USD_PER_SECOND
    + CPU_CORES * CPU_USD_PER_CORE_SECOND
    + (16_384 / 1_024) * MEMORY_USD_PER_GIB_SECOND
)
M2_RESUME_WORST_CASE_USD = M2_RESUME_BILLABLE_BOUND_SECONDS * (
    CPU_CORES * CPU_USD_PER_CORE_SECOND
    + (MEMORY_MIB / 1_024) * MEMORY_USD_PER_GIB_SECOND
)
TRANCHE_FREEZE_WORST_CASE_USD = 600 * (
    CPU_USD_PER_CORE_SECOND + 2 * MEMORY_USD_PER_GIB_SECOND
)
FIRST_WAVE_WORST_CASE_USD = (
    2 * MINER_WORST_CASE_USD
    + TRANCHE_FREEZE_WORST_CASE_USD
    + M3_WORST_CASE_USD
    + M4_WORST_CASE_USD
)
JOB_WORST_CASE_USD = {
    "m1": MINER_WORST_CASE_USD,
    "m2": MINER_WORST_CASE_USD,
    "m3": M3_WORST_CASE_USD,
    "m4": M4_WORST_CASE_USD,
}
MAX_RETURN_BYTES = 2 * 1_024 * 1_024
SAFE_ARTIFACT_NAMES = ("branch_targets.jsonl", "smoke_receipt.json")
REQUIRED_CHECKS = (
    "exact_branch_reproducibility",
    "siblings_begin_from_same_root",
    "root_simulator_state_unchanged",
    "persisted_private_content_scan",
    "search_begin_input_absent",
    "student_actor_visible_only",
    "execution_mode_active",
    "branch_executor_safety_clean",
    "protected_runtime_submission_unchanged",
    "teacher_memory_released",
    "teacher_process_exited",
    "aggregate_readable_after_teacher_exit",
)

# The command builders depend only on these mappings. If either CLI contract
# changes before integration, adjust the mapping rather than the orchestration.
MINER_CLI_FLAGS = {
    "run_dir": "--run-dir",
    "miner_id": "--miner-id",
    "policy0": "--policy0",
    "policy1": "--policy1",
    "minimum_source_turn": "--minimum-source-turn",
    "maximum_source_turn": "--maximum-source-turn",
    "source_seed": "--source-seed",
    "branch_seed": "--branch-seed",
    "max_useful_seconds": "--max-useful-seconds",
    "deck0": "--deck0",
    "deck1": "--deck1",
    "v_checkpoint": "--v-checkpoint",
    "v_calibrator_dir": "--v-calibrator-dir",
    "resume": "--resume",
}
MINER_SPECS = {
    "m1": {
        "miner_id": "M1",
        "policy0": "v2.4",
        "policy1": "v2.4",
        "minimum_source_turn": 1,
        "maximum_source_turn": 3,
        "source_seed": 7_123_101,
        "branch_seed": 7_124_101,
    },
    "m2": {
        "miner_id": "M2",
        "policy0": "v2.4",
        "policy1": "greedy",
        "minimum_source_turn": 4,
        "maximum_source_turn": 0,
        "source_seed": 7_123_201,
        "branch_seed": 7_124_201,
    },
}
TRAINER_CLI_FLAGS = {
    "mode": "--mode",
    "manifest": "--manifest",
    "run_dir": "--run-dir",
    "minimum_cumulative_seconds": "--minimum-cumulative-seconds",
    "source_policy_checkpoint": "--source-policy-checkpoint",
    "source_policy_sha256": "--source-policy-sha256",
    "v_checkpoint": "--v-checkpoint",
    "v_checkpoint_sha256": "--v-checkpoint-sha256",
    "resume": "--resume",
}
TRAINER_ROLES = {
    "m3": "policy",
    "m4": "specialist-v",
}


def _build_image(runtime_project: str) -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.12")
        .uv_sync(runtime_project, frozen=True)
        .add_local_dir(str(ROOT / "cg"), "/repo/cg", copy=True)
        .add_local_dir(str(ROOT / "deepmew"), "/repo/deepmew", copy=True)
        .add_local_dir(str(ROOT / "scripts"), "/repo/scripts", copy=True)
        .add_local_dir(str(ROOT / "data"), "/repo/data", copy=True)
        .add_local_file(str(ROOT / "main.py"), "/repo/main.py", copy=True)
        .add_local_file(str(ROOT / "deck.csv"), "/repo/deck.csv", copy=True)
        .add_local_file(str(TOP_STARYU_DECK), str(REMOTE_TOP_STARYU_DECK), copy=True)
        .add_local_file(str(FROZEN_V0_CHECKPOINT), str(REMOTE_V0_CHECKPOINT), copy=True)
        .add_local_dir(str(FROZEN_V0_CALIBRATOR), str(REMOTE_V0_CALIBRATOR), copy=True)
        .add_local_file(str(R7_CHECKPOINT), str(REMOTE_R7_CHECKPOINT), copy=True)
    )


image = _build_image(RUNTIME_PROJECT)
gpu_image = _build_image(GPU_RUNTIME_PROJECT)

app = modal.App("deepmew-q-guided")
RESEARCH_VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _read_safe_result(out_dir: Path, *, world_samples: int) -> dict[str, bytes]:
    files = {name: (out_dir / name).read_bytes() for name in SAFE_ARTIFACT_NAMES}
    if sum(len(value) for value in files.values()) > MAX_RETURN_BYTES:
        raise RuntimeError("aggregate smoke artifact exceeded its return bound")
    receipt = json.loads(files["smoke_receipt.json"])
    if receipt.get("status") != "passed_end_to_end_smoke":
        raise RuntimeError("aggregate smoke receipt did not report success")
    if receipt.get("execution_mode") != "fork" or receipt.get("execution_platform") != "linux":
        raise RuntimeError("aggregate smoke receipt did not use Linux fork cloning")
    if receipt.get("world_sample_count") != world_samples:
        raise RuntimeError("aggregate smoke receipt has the wrong world-sample count")
    if world_samples >= 2 and receipt.get("aggregation_status") != "eligible_aggregate_teacher_target":
        raise RuntimeError("multi-world pilot did not produce an eligible aggregate target")
    safety = receipt.get("safety")
    if safety != {"illegal_actions": 0, "crashes": 0, "timeouts": 0}:
        raise RuntimeError("aggregate smoke receipt failed its safety gate")
    checks = receipt.get("checks")
    if not isinstance(checks, dict) or any(checks.get(name) is not True for name in REQUIRED_CHECKS):
        raise RuntimeError("aggregate smoke receipt failed a required check")
    shard_lines = [line for line in files["branch_targets.jsonl"].splitlines() if line.strip()]
    if len(shard_lines) != 1:
        raise RuntimeError("aggregate smoke shard has an invalid row count")
    shard = json.loads(shard_lines[0])
    if "/repo" not in sys.path:
        sys.path.insert(0, "/repo")
    from deepmew.q_guided.branching import assert_private_content_absent
    from deepmew.q_guided.teacher_contract import validate_safe_shard

    assert_private_content_absent(shard)
    validate_safe_shard(shard)

    return files


def _pack_safe_result(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in SAFE_ARTIFACT_NAMES:
            payload = files[name]
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            info.mode = 0o600
            archive.addfile(info, io.BytesIO(payload))
    result = buffer.getvalue()
    if len(result) > MAX_RETURN_BYTES:
        raise RuntimeError("compressed aggregate smoke artifact exceeded its return bound")

    return result


def _safe_subprocess_failure_category(completed: subprocess.CompletedProcess[bytes]) -> str:
    stderr = completed.stderr.decode("utf-8", errors="replace")
    for line in reversed(stderr.splitlines()):
        marker = "SmokeSafetyError: "
        if marker in line:
            category = line.rsplit(marker, 1)[1].strip()
            if category.isascii() and 0 < len(category) <= 512:
                return category
    if "unrecognized arguments:" in stderr:
        return "CLI argument contract failure"
    missing = re.search(
        r"ModuleNotFoundError: No module named ['\"]([A-Za-z0-9_.-]+)['\"]",
        stderr,
    )
    if missing:
        return f"missing module {missing.group(1)}"
    if "ModuleNotFoundError:" in stderr or "ImportError:" in stderr:
        return "module import failure"

    return f"subprocess exit code {completed.returncode}"


@app.function(
    image=image,
    cpu=CPU_CORES,
    memory=MEMORY_MIB,
    timeout=TIMEOUT_SECONDS,
    retries=0,
)
def privileged_teacher_smoke(world_samples: int) -> dict[str, Any]:
    command = [
        sys.executable,
        "-u",
        "/repo/scripts/run_privileged_teacher_smoke.py",
        "--execution-mode",
        "fork",
        "--out-dir",
        str(REMOTE_OUTPUT),
        "--world-samples",
        str(world_samples),
        "--deck0",
        str(REMOTE_TOP_STARYU_DECK),
        "--deck1",
        str(REMOTE_TOP_STARYU_DECK),
        "--v-checkpoint",
        str(REMOTE_V0_CHECKPOINT),
        "--v-calibrator-dir",
        str(REMOTE_V0_CALIBRATOR),
    ]
    completed = subprocess.run(
        command,
        cwd="/repo",
        check=False,
        capture_output=True,
        timeout=TIMEOUT_SECONDS - 30,
    )
    if completed.returncode != 0:
        category = "failure receipt unavailable"
        failure_path = Path(REMOTE_FAILURE_RECEIPT)
        if failure_path.is_file():
            failure_receipt = json.loads(failure_path.read_text(encoding="utf-8"))
            category = str(failure_receipt.get("failure_category") or category)
        else:
            category = _safe_subprocess_failure_category(completed)
        raise RuntimeError(f"privileged-teacher Modal smoke failed: {category}")
    files = _read_safe_result(Path(REMOTE_OUTPUT), world_samples=world_samples)
    archive = _pack_safe_result(files)
    manifest = {
        "schema_version": "deepmew.modal_privileged_teacher_smoke.v1",
        "status": "passed_safe_aggregate_smoke",
        "artifact_names": list(SAFE_ARTIFACT_NAMES),
        "artifact_tar_sha256": _sha256_bytes(archive),
        "artifact_tar_bytes": len(archive),
        "world_sample_count": world_samples,
    }

    return {"manifest": manifest, "artifact_tar_gz": archive}


def _append_cli_value(command: list[str], flag: str, value: Any) -> None:
    command.extend((flag, str(value)))


def _miner_command(miner_id: str, stratum_index: int, max_useful_seconds: int) -> list[str]:
    run_dir = RUN_ROOT / miner_id
    spec = MINER_SPECS[miner_id]
    if stratum_index != (0 if miner_id == "m1" else 1):
        raise ValueError("miner stratum index does not match its fixed first-wave contract")
    command = [sys.executable, "-u", "/repo/scripts/run_q_guided_branch_miner.py"]
    values = {
        "run_dir": run_dir,
        **spec,
        "max_useful_seconds": max_useful_seconds,
        "deck0": REMOTE_TOP_STARYU_DECK,
        "deck1": REMOTE_TOP_STARYU_DECK,
        "v_checkpoint": REMOTE_V0_CHECKPOINT,
        "v_calibrator_dir": REMOTE_V0_CALIBRATOR,
    }
    for name, value in values.items():
        _append_cli_value(command, MINER_CLI_FLAGS[name], value)
    if Path(run_dir / "progress.json").is_file():
        command.append(MINER_CLI_FLAGS["resume"])

    return command


def _trainer_command(
    trainer_id: str,
    max_useful_seconds: int,
    tranche_dir: str,
) -> list[str]:
    run_dir = RUN_ROOT / trainer_id
    command = [sys.executable, "-u", "/repo/scripts/run_q_guided_gpu_train.py"]
    values = {
        "mode": TRAINER_ROLES[trainer_id],
        "manifest": PurePosixPath(tranche_dir) / "tranche.manifest.json",
        "run_dir": run_dir,
        "minimum_cumulative_seconds": max_useful_seconds,
        "source_policy_checkpoint": REMOTE_R7_CHECKPOINT,
        "source_policy_sha256": R7_CHECKPOINT_SHA256,
        "v_checkpoint": REMOTE_V0_CHECKPOINT,
        "v_checkpoint_sha256": FROZEN_V0_CHECKPOINT_SHA256,
    }
    for name, value in values.items():
        _append_cli_value(command, TRAINER_CLI_FLAGS[name], value)
    if Path(run_dir / "training.pt").is_file():
        command.append(TRAINER_CLI_FLAGS["resume"])

    return command


def _read_progress_count(path: Path, names: tuple[str, ...]) -> int | float | None:
    if not path.is_file():
        return None
    try:
        progress = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(progress, dict):
        return None
    for name in names:
        value = progress.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value

    return None


def _read_safe_metrics(path: Path) -> tuple[dict[str, float], dict[str, int]]:
    if not path.is_file():
        return {}, {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}, {}
    metrics = value.get("metrics") if isinstance(value, dict) else None
    safe_metrics = {}
    if isinstance(metrics, dict):
        for name, metric in sorted(metrics.items()):
            if len(safe_metrics) == 64:
                break
            if (
                isinstance(name, str)
                and isinstance(metric, (int, float))
                and not isinstance(metric, bool)
                and math.isfinite(float(metric))
            ):
                safe_metrics[name] = float(metric)
    counters = value.get("safety_counters") if isinstance(value, dict) else None
    safe_counters = {}
    if isinstance(counters, dict):
        for name in ("illegal_actions", "crashes", "timeouts"):
            count = counters.get(name)
            if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                safe_counters[name] = count

    return safe_metrics, safe_counters


def _stop_process(process: subprocess.Popen[Any]) -> None:
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def _safe_process_failure_category(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    except OSError:
        return "trainer error log unreadable"
    if not lines:
        return None
    category = lines[-1]
    if len(category) > 512:
        category = category[-512:]
    if not category.isascii():
        category = category.encode("ascii", errors="replace").decode("ascii")

    return category


def _run_checkpointed_job(
    *,
    job_id: str,
    kind: str,
    command: list[str],
    max_useful_seconds: int,
    billable_bound_seconds: int,
    minimum_prior_useful_seconds: float = 0,
) -> dict[str, Any]:
    if not 0 <= minimum_prior_useful_seconds <= max_useful_seconds:
        raise ValueError("minimum prior useful seconds are outside the cumulative floor")
    maximum_remaining_seconds = max_useful_seconds - minimum_prior_useful_seconds
    if billable_bound_seconds <= maximum_remaining_seconds:
        raise ValueError("billable bound must leave time to verify and commit the final checkpoint")
    run_dir = Path(RUN_ROOT / job_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = Path(VOLUME_ROOT / "diagnostics" / job_id)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    invocation_id = str(time.time_ns())
    for name in ("process_stdout.log", "process_stderr.log"):
        legacy = run_dir / name
        if legacy.is_file():
            os.replace(legacy, diagnostics_dir / f"legacy_{invocation_id}_{name}")
    stdout_path = diagnostics_dir / f"invocation_{invocation_id}_stdout.log"
    stderr_path = diagnostics_dir / f"invocation_{invocation_id}_stderr.log"
    stdout_handle = stdout_path.open("ab")
    stderr_handle = stderr_path.open("ab")
    try:
        process = subprocess.Popen(
            command,
            cwd="/repo",
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
    finally:
        stdout_handle.close()
        stderr_handle.close()
    started = time.monotonic()
    deadline = started + billable_bound_seconds
    next_commit = started + VOLUME_COMMIT_INTERVAL_SECONDS
    commit_count = 0
    stopped_at_budget_bound = False
    while process.poll() is None:
        now = time.monotonic()
        if now >= deadline:
            _stop_process(process)
            stopped_at_budget_bound = True
            break
        if now >= next_commit:
            RESEARCH_VOLUME.commit()
            commit_count += 1
            next_commit = now + VOLUME_COMMIT_INTERVAL_SECONDS
        time.sleep(15)
    return_code = process.wait()
    RESEARCH_VOLUME.commit()
    commit_count += 1
    progress_path = run_dir / "progress.json"
    if kind == "miner":
        required_outputs = (
            progress_path,
            run_dir / "manifest.json",
        )
        has_payload = any((run_dir / "shards").glob("*.jsonl"))
        eligible_rows = _read_progress_count(
            progress_path,
            ("eligible_rows", "aggregate_q_rows", "row_count"),
        )
        cumulative_seconds = _read_progress_count(
            progress_path,
            ("cumulative_useful_seconds", "useful_runtime_seconds"),
        )
        research_output_exists = None
    else:
        required_outputs = (
            run_dir / "training.pt",
            run_dir / "metrics.json",
        )
        has_payload = all(path.is_file() for path in required_outputs)
        eligible_rows = None
        cumulative_seconds = _read_progress_count(
            run_dir / "metrics.json",
            ("cumulative_gpu_seconds", "minimum_cumulative_gpu_seconds"),
        )
        research_output_exists = (
            (run_dir / "research_r4f_q_guided.pt").is_file()
            if job_id == "m3"
            else (run_dir / "specialist_v_calibrator.json").is_file()
        )
    safe_metrics, safety_counters = (
        _read_safe_metrics(run_dir / "metrics.json") if kind == "trainer" else ({}, {})
    )
    outputs_complete = all(path.is_file() for path in required_outputs) and has_payload
    if return_code == 0 and outputs_complete:
        status = "completed_checkpointed"
    elif stopped_at_budget_bound and outputs_complete:
        status = "budget_bound_checkpointed"
    elif outputs_complete:
        status = "failed_checkpointed"
    else:
        status = "failed_missing_required_outputs"
    receipt = {
        "schema_version": "deepmew.modal_first_wave_job.v1",
        "job_id": job_id,
        "kind": kind,
        "status": status,
        "process_return_code": return_code,
        "stopped_at_budget_bound": stopped_at_budget_bound,
        "required_outputs_complete": outputs_complete,
        "volume_commit_count": commit_count,
        "cumulative_useful_seconds": cumulative_seconds,
        "eligible_row_count": eligible_rows,
        "research_output_exists": research_output_exists,
        "metrics": safe_metrics,
        "safety_counters": safety_counters,
        "failure_category": _safe_process_failure_category(stderr_path) if return_code != 0 else None,
    }

    return receipt


@app.function(
    image=image,
    cpu=4.0,
    memory=8_192,
    timeout=FIRST_WAVE_TIMEOUT_SECONDS,
    retries=0,
    volumes={str(VOLUME_ROOT): RESEARCH_VOLUME},
)
def branch_miner(miner_id: str, stratum_index: int) -> dict[str, Any]:
    if (miner_id, stratum_index) not in {("m1", 0), ("m2", 1)}:
        raise ValueError("miner identity and stratum must be the fixed disjoint first-wave pair")
    command = _miner_command(miner_id, stratum_index, MINER_USEFUL_SECONDS)

    return _run_checkpointed_job(
        job_id=miner_id,
        kind="miner",
        command=command,
        max_useful_seconds=MINER_USEFUL_SECONDS,
        billable_bound_seconds=MINER_BILLABLE_BOUND_SECONDS,
    )


def _validate_m2_resume_progress() -> None:
    progress_path = Path(RUN_ROOT / "m2" / "progress.json")
    if not progress_path.is_file():
        raise RuntimeError("M2 resume requires its committed progress checkpoint")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("schema_version") != "deepmew.q_guided.branch_miner_progress.v1":
        raise RuntimeError("M2 resume progress schema mismatch")
    if progress.get("config_sha256") != M2_RESUME_REQUIRED_CONFIG_SHA256:
        raise RuntimeError("M2 resume config hash mismatch")
    cumulative = progress.get("cumulative_useful_seconds")
    eligible_rows = progress.get("eligible_rows")
    if (
        isinstance(cumulative, bool)
        or not isinstance(cumulative, (int, float))
        or cumulative < M2_RESUME_MINIMUM_PRIOR_USEFUL_SECONDS
    ):
        raise RuntimeError("M2 resume cumulative checkpoint is below the verified prior floor")
    if (
        isinstance(eligible_rows, bool)
        or not isinstance(eligible_rows, int)
        or eligible_rows < M2_RESUME_MINIMUM_ELIGIBLE_ROWS
    ):
        raise RuntimeError("M2 resume eligible-row checkpoint is below the verified prior floor")
    if progress.get("safety") != {"illegal_actions": 0, "crashes": 0, "timeouts": 0}:
        raise RuntimeError("M2 resume checkpoint is not safety-clean")
    if progress.get("duplicate_rows") != 0 or progress.get("v0_validated") is not True:
        raise RuntimeError("M2 resume checkpoint failed duplicate-row or V0 provenance gates")


@app.function(
    image=image,
    cpu=4.0,
    memory=8_192,
    timeout=FIRST_WAVE_TIMEOUT_SECONDS,
    retries=0,
    volumes={str(VOLUME_ROOT): RESEARCH_VOLUME},
)
def resume_m2() -> dict[str, Any]:
    _validate_m2_resume_progress()
    command = _miner_command("m2", 1, MINER_USEFUL_SECONDS)

    return _run_checkpointed_job(
        job_id="m2",
        kind="miner",
        command=command,
        max_useful_seconds=MINER_USEFUL_SECONDS,
        billable_bound_seconds=M2_RESUME_BILLABLE_BOUND_SECONDS,
        minimum_prior_useful_seconds=M2_RESUME_MINIMUM_PRIOR_USEFUL_SECONDS,
    )


def _miner_manifest_for_shard(run_dir: Path, shard_path: Path, shard_count: int) -> Path:
    sidecar = shard_path.with_name(f"{shard_path.stem}.manifest.json")
    if sidecar.is_file():
        return sidecar
    run_manifest = run_dir / "manifest.json"
    if shard_count == 1 and run_manifest.is_file():
        return run_manifest

    raise RuntimeError("miner shard lacks an unambiguous verified per-row manifest")


def _read_valid_miner_rows() -> dict[str, list[tuple[bytes, dict[str, Any]]]]:
    if "/repo" not in sys.path:
        sys.path.insert(0, "/repo")
    from deepmew.q_guided.training import ShardBinding, load_manifest_bound_shards

    result = {"m1": [], "m2": []}
    seen = set()
    seen_signatures = set()
    for miner_id in result:
        run_dir = Path(RUN_ROOT / miner_id)
        shard_paths = sorted((run_dir / "shards").glob("*.jsonl"), key=lambda path: path.name)
        for shard_path in shard_paths:
            manifest_path = _miner_manifest_for_shard(run_dir, shard_path, len(shard_paths))
            manifest_before = manifest_path.read_bytes()
            manifest = json.loads(manifest_before)
            shard_before = shard_path.read_bytes()
            dataset = load_manifest_bound_shards([ShardBinding(shard_path, manifest_path)])
            shard_after = shard_path.read_bytes()
            manifest_after = manifest_path.read_bytes()
            if shard_before != shard_after or manifest_before != manifest_after:
                raise RuntimeError("miner shard binding changed during immutable-tranche inspection")
            if _sha256_bytes(shard_after) != manifest.get("shard_sha256"):
                raise RuntimeError("miner shard hash changed after manifest validation")
            receipts = manifest.get("row_receipts")
            if not isinstance(receipts, list) or len(receipts) != len(dataset.rows):
                raise RuntimeError("miner manifest receipt coverage changed after validation")
            for loaded_row, receipt in zip(dataset.rows, receipts, strict=True):
                canonical = _canonical_bytes(loaded_row.row) + b"\n"
                row_sha256 = _sha256_bytes(canonical)
                signature = receipt.get("actor_visible_signature_sha256")
                if row_sha256 in seen or signature in seen_signatures:
                    continue
                seen.add(row_sha256)
                seen_signatures.add(signature)
                result[miner_id].append((canonical, dict(receipt)))

    return result


def _interleave_rows(
    rows_by_miner: dict[str, list[tuple[bytes, dict[str, Any]]]],
    limit: int,
) -> tuple[list[tuple[bytes, dict[str, Any]]], dict[str, int]]:
    selected = []
    selected_counts = {miner_id: 0 for miner_id in rows_by_miner}
    index = 0
    while len(selected) < limit:
        added = False
        for miner_id in ("m1", "m2"):
            rows = rows_by_miner[miner_id]
            if index < len(rows):
                selected.append(rows[index])
                selected_counts[miner_id] += 1
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        index += 1

    return selected, selected_counts


@app.function(
    image=image,
    cpu=1.0,
    memory=2_048,
    timeout=600,
    retries=0,
    volumes={str(VOLUME_ROOT): RESEARCH_VOLUME},
)
def freeze_first_tranche(minimum_rows: int = 64) -> dict[str, Any]:
    if minimum_rows < 64:
        raise ValueError("the first immutable Q tranche requires at least 64 eligible rows")
    rows_by_miner = _read_valid_miner_rows()
    available_counts = {name: len(rows) for name, rows in rows_by_miner.items()}
    if sum(available_counts.values()) < minimum_rows or any(count == 0 for count in available_counts.values()):
        return {
            "schema_version": "deepmew.modal_q_tranche_freeze.v1",
            "status": "not_ready",
            "minimum_rows": minimum_rows,
            "eligible_row_counts": available_counts,
        }
    selected, selected_counts = _interleave_rows(rows_by_miner, minimum_rows)
    payload = b"".join(row for row, _receipt in selected)
    payload_sha256 = _sha256_bytes(payload)
    tranche_dir = Path(TRANCHE_ROOT / f"eligible_{minimum_rows:06d}_{payload_sha256[:16]}")
    shard_path = tranche_dir / "tranche.jsonl"
    manifest_path = tranche_dir / "tranche.manifest.json"
    manifest = {
        "schema_version": "deepmew.q_guided.teacher_shard_manifest.v1",
        "split": "first_wave_training",
        "shard_sha256": payload_sha256,
        "row_count": len(selected),
        "row_receipts": [receipt for _row, receipt in selected],
    }
    if tranche_dir.exists():
        if not shard_path.is_file() or not manifest_path.is_file():
            raise RuntimeError("content-addressed tranche exists without its required files")
        if _sha256_bytes(shard_path.read_bytes()) != payload_sha256:
            raise RuntimeError("content-addressed tranche failed its hash gate")
        if manifest_path.read_bytes() != _canonical_bytes(manifest) + b"\n":
            raise RuntimeError("content-addressed tranche manifest changed after freeze")
    else:
        tranche_dir.mkdir(parents=True, exist_ok=False)
        shard_path.write_bytes(payload)
        manifest_path.write_bytes(_canonical_bytes(manifest) + b"\n")
        os.chmod(shard_path, 0o444)
        os.chmod(manifest_path, 0o444)
        os.chmod(tranche_dir, 0o555)
        RESEARCH_VOLUME.commit()
    from deepmew.q_guided.training import ShardBinding, load_manifest_bound_shards

    verified = load_manifest_bound_shards([ShardBinding(shard_path, manifest_path)])
    if len(verified.rows) != minimum_rows:
        raise RuntimeError("frozen tranche failed its manifest-bound row-count gate")
    receipt = {
        "schema_version": "deepmew.modal_q_tranche_freeze.v1",
        "status": "frozen_immutable",
        "minimum_rows": minimum_rows,
        "eligible_row_counts": available_counts,
        "selected_row_counts": selected_counts,
        "tranche_dir": str(tranche_dir),
        "shard_sha256": payload_sha256,
    }

    return receipt


def _run_gpu_trainer(
    trainer_id: str,
    tranche_dir: str,
    max_useful_seconds: int,
    billable_bound_seconds: int,
    minimum_prior_useful_seconds: int = 0,
) -> dict[str, Any]:
    command = _trainer_command(trainer_id, max_useful_seconds, tranche_dir)

    return _run_checkpointed_job(
        job_id=trainer_id,
        kind="trainer",
        command=command,
        max_useful_seconds=max_useful_seconds,
        billable_bound_seconds=billable_bound_seconds,
        minimum_prior_useful_seconds=minimum_prior_useful_seconds,
    )


@app.function(
    image=gpu_image,
    gpu="L4",
    cpu=4.0,
    memory=16_384,
    timeout=FIRST_WAVE_TIMEOUT_SECONDS,
    retries=0,
    volumes={str(VOLUME_ROOT): RESEARCH_VOLUME},
)
def train_m3(tranche_dir: str) -> dict[str, Any]:
    if not tranche_dir.startswith(f"{TRANCHE_ROOT}/eligible_"):
        raise ValueError("M3 requires a frozen first-wave tranche path")

    return _run_gpu_trainer(
        "m3",
        tranche_dir,
        M3_USEFUL_SECONDS,
        M3_BILLABLE_BOUND_SECONDS,
    )


@app.function(
    image=gpu_image,
    gpu="T4",
    cpu=4.0,
    memory=16_384,
    timeout=FIRST_WAVE_TIMEOUT_SECONDS,
    retries=0,
    volumes={str(VOLUME_ROOT): RESEARCH_VOLUME},
)
def train_m4(tranche_dir: str) -> dict[str, Any]:
    if not tranche_dir.startswith(f"{TRANCHE_ROOT}/eligible_"):
        raise ValueError("M4 requires a frozen first-wave tranche path")

    return _run_gpu_trainer(
        "m4",
        tranche_dir,
        M4_USEFUL_SECONDS,
        M4_BILLABLE_BOUND_SECONDS,
    )


@app.function(
    image=gpu_image,
    gpu="T4",
    cpu=4.0,
    memory=16_384,
    timeout=M4_RESUME_TIMEOUT_SECONDS,
    retries=0,
    volumes={str(VOLUME_ROOT): RESEARCH_VOLUME},
)
def resume_m4(tranche_dir: str) -> dict[str, Any]:
    if not tranche_dir.startswith(f"{TRANCHE_ROOT}/eligible_"):
        raise ValueError("M4 resume requires a frozen first-wave tranche path")

    return _run_gpu_trainer(
        "m4",
        tranche_dir,
        M4_USEFUL_SECONDS,
        M4_RESUME_BILLABLE_BOUND_SECONDS,
        M4_RESUME_MINIMUM_PRIOR_GPU_SECONDS,
    )


def _save_safe_result(result: dict[str, Any], out_dir: Path) -> None:
    if out_dir.exists():
        raise RuntimeError("local Modal smoke output path already exists")
    manifest = result.get("manifest")
    archive = result.get("artifact_tar_gz")
    if not isinstance(manifest, dict) or not isinstance(archive, bytes):
        raise RuntimeError("Modal smoke returned an invalid result envelope")
    if manifest.get("status") != "passed_safe_aggregate_smoke":
        raise RuntimeError("Modal smoke result envelope did not report success")
    if len(archive) > MAX_RETURN_BYTES or _sha256_bytes(archive) != manifest.get("artifact_tar_sha256"):
        raise RuntimeError("Modal smoke result envelope failed its integrity gate")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        members = bundle.getmembers()
        if tuple(member.name for member in members) != SAFE_ARTIFACT_NAMES:
            raise RuntimeError("Modal smoke archive contained an unexpected member")
        payloads = {}
        for member in members:
            if not member.isfile() or member.size > MAX_RETURN_BYTES:
                raise RuntimeError("Modal smoke archive member failed its type or size gate")
            handle = bundle.extractfile(member)
            if handle is None:
                raise RuntimeError("Modal smoke archive member could not be read")
            payloads[member.name] = handle.read()
    out_dir.mkdir(parents=True, exist_ok=False)
    for name in SAFE_ARTIFACT_NAMES:
        (out_dir / name).write_bytes(payloads[name])
    (out_dir / "modal_manifest.json").write_bytes(_canonical_bytes(manifest) + b"\n")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_canonical_bytes(value) + b"\n")


def _scheduled_from_local_receipts(control_dir: Path) -> float:
    total = 0.0
    if not control_dir.is_dir():
        return total
    for path in control_dir.glob("*_launch.json"):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        value = receipt.get("worst_case_usd")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total += float(value)
    tranche_path = control_dir / "tranche_receipt.json"
    if tranche_path.is_file():
        receipt = json.loads(tranche_path.read_text(encoding="utf-8"))
        value = receipt.get("worst_case_usd")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total += float(value)

    return total


def _assert_budget(
    *,
    control_dir: Path,
    proposed_usd: float,
    external_scheduled_usd: float,
    available_usd: float,
    minimum_reserve_usd: float = MIN_RESERVE_USD,
) -> dict[str, float]:
    if (
        proposed_usd < 0
        or external_scheduled_usd < 0
        or available_usd < 0
        or minimum_reserve_usd < 0
    ):
        raise ValueError("budget values must be non-negative")
    local_scheduled = _scheduled_from_local_receipts(control_dir)
    scheduled = external_scheduled_usd + local_scheduled + proposed_usd
    effective_cap = min(available_usd, MAX_SCHEDULE_USD)
    reserve = effective_cap - scheduled
    print(
        f"[budget] external=${external_scheduled_usd:.6f} "
        f"local=${local_scheduled:.6f} proposed=${proposed_usd:.6f} "
        f"scheduled=${scheduled:.6f} cap=${effective_cap:.6f} reserve=${reserve:.6f}"
    )
    if scheduled > effective_cap:
        raise RuntimeError(
            f"schedule gate rejected ${scheduled:.6f}; effective cap is ${effective_cap:.2f}"
        )
    if reserve < minimum_reserve_usd:
        raise RuntimeError(
            f"reserve gate rejected ${reserve:.6f}; minimum reserve is ${minimum_reserve_usd:.2f}"
        )

    return {"scheduled_usd": scheduled, "remaining_reserve_usd": reserve}


def _launch_receipt(
    *,
    job_id: str,
    call: modal.FunctionCall[Any],
    resources: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "deepmew.modal_first_wave_launch.v1",
        "status": "launched_detached",
        "job_id": job_id,
        "call_id": call.object_id,
        "run_dir": str(RUN_ROOT / job_id),
        "resources": resources,
        "worst_case_usd": JOB_WORST_CASE_USD[job_id],
        "hard_total_cap_usd": MAX_SCHEDULE_USD,
        "minimum_uncommitted_reserve_usd": MIN_RESERVE_USD,
        "launched_at": _utc_now(),
        "automatic_retries": 0,
        "resume_owned_by_cli": True,
    }


def _require_not_launched(control_dir: Path, job_ids: tuple[str, ...]) -> None:
    existing = [job_id for job_id in job_ids if (control_dir / f"{job_id}_launch.json").exists()]
    if existing:
        raise RuntimeError(f"detached launch receipt already exists for {','.join(existing)}")


def _launch_miners(
    control_dir: Path,
    external_scheduled_usd: float,
    available_usd: float,
) -> None:
    job_ids = ("m1", "m2")
    _require_not_launched(control_dir, job_ids)
    _assert_budget(
        control_dir=control_dir,
        proposed_usd=2 * MINER_WORST_CASE_USD,
        external_scheduled_usd=external_scheduled_usd,
        available_usd=available_usd,
    )
    specs = (("m1", 0), ("m2", 1))
    for job_id, stratum_index in specs:
        call = branch_miner.spawn(job_id, stratum_index)
        receipt = _launch_receipt(
            job_id=job_id,
            call=call,
            resources={
                "cpu_physical_cores": 4,
                "memory_mib": 8_192,
                "gpu": None,
                "max_useful_seconds": MINER_USEFUL_SECONDS,
                "billable_bound_seconds": MINER_BILLABLE_BOUND_SECONDS,
                "stratum_index": stratum_index,
                "stratum_count": 2,
                "policy0": MINER_SPECS[job_id]["policy0"],
                "policy1": MINER_SPECS[job_id]["policy1"],
                "minimum_source_turn": MINER_SPECS[job_id]["minimum_source_turn"],
                "maximum_source_turn": MINER_SPECS[job_id]["maximum_source_turn"],
            },
        )
        _write_json_once(control_dir / f"{job_id}_launch.json", receipt)
        print(f"[modal] {job_id} detached call_id={call.object_id}")


def _freeze_tranche(control_dir: Path) -> dict[str, Any]:
    receipt = freeze_first_tranche.remote(64)
    if not isinstance(receipt, dict):
        raise RuntimeError("tranche freeze returned an invalid receipt")
    local_receipt = {
        "schema_version": str(receipt.get("schema_version")),
        "status": str(receipt.get("status")),
        "minimum_rows": receipt.get("minimum_rows"),
        "eligible_row_counts": receipt.get("eligible_row_counts"),
        "selected_row_counts": receipt.get("selected_row_counts"),
        "tranche_dir": receipt.get("tranche_dir"),
        "shard_sha256": receipt.get("shard_sha256"),
        "worst_case_usd": TRANCHE_FREEZE_WORST_CASE_USD,
        "hard_total_cap_usd": MAX_SCHEDULE_USD,
        "minimum_uncommitted_reserve_usd": MIN_RESERVE_USD,
        "inspected_at": _utc_now(),
    }
    path = control_dir / "tranche_receipt.json"
    if local_receipt["status"] == "frozen_immutable" and not path.exists():
        _write_json_once(path, local_receipt)

    return local_receipt


def _launch_trainers(
    control_dir: Path,
    external_scheduled_usd: float,
    available_usd: float,
) -> None:
    job_ids = ("m3", "m4")
    _require_not_launched(control_dir, job_ids)
    tranche_already_inspected = (control_dir / "tranche_receipt.json").is_file()
    freeze_cost = 0.0 if tranche_already_inspected else TRANCHE_FREEZE_WORST_CASE_USD
    _assert_budget(
        control_dir=control_dir,
        proposed_usd=freeze_cost + M3_WORST_CASE_USD + M4_WORST_CASE_USD,
        external_scheduled_usd=external_scheduled_usd,
        available_usd=available_usd,
    )
    if tranche_already_inspected:
        tranche = json.loads((control_dir / "tranche_receipt.json").read_text(encoding="utf-8"))
    else:
        tranche = _freeze_tranche(control_dir)
    if tranche.get("status") != "frozen_immutable" or not isinstance(tranche.get("tranche_dir"), str):
        raise RuntimeError("first combined tranche is not ready; trainers were not launched")
    m3_call = train_m3.spawn(str(tranche["tranche_dir"]))
    m3_receipt = _launch_receipt(
        job_id="m3",
        call=m3_call,
        resources={
            "cpu_physical_cores": 4,
            "memory_mib": 16_384,
            "gpu": "L4",
            "max_useful_seconds": M3_USEFUL_SECONDS,
            "billable_bound_seconds": M3_BILLABLE_BOUND_SECONDS,
            "tranche_dir": tranche["tranche_dir"],
        },
    )
    _write_json_once(control_dir / "m3_launch.json", m3_receipt)
    print(f"[modal] m3 detached call_id={m3_call.object_id}")
    m4_call = train_m4.spawn(str(tranche["tranche_dir"]))
    m4_receipt = _launch_receipt(
        job_id="m4",
        call=m4_call,
        resources={
            "cpu_physical_cores": 4,
            "memory_mib": 16_384,
            "gpu": "T4",
            "max_useful_seconds": M4_USEFUL_SECONDS,
            "billable_bound_seconds": M4_BILLABLE_BOUND_SECONDS,
            "tranche_dir": tranche["tranche_dir"],
        },
    )
    _write_json_once(control_dir / "m4_launch.json", m4_receipt)
    print(f"[modal] m4 detached call_id={m4_call.object_id}")


def _launch_m2_resume(
    control_dir: Path,
    external_scheduled_usd: float,
    available_usd: float,
) -> None:
    launch_path = control_dir / "m2_resume_launch.json"
    if launch_path.exists():
        raise RuntimeError("the authorized M2 resume attempt is already recorded")
    completion_path = control_dir / "m2_completion.json"
    if not completion_path.is_file():
        raise RuntimeError("M2 resume requires the original completion receipt")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if (
        completion.get("job_id") != "m2"
        or completion.get("status") != "budget_bound_checkpointed"
        or completion.get("stopped_at_budget_bound") is not True
        or completion.get("required_outputs_complete") is not True
        or completion.get("process_return_code") != -15
    ):
        raise RuntimeError("M2 resume requires the verified budget-bound checkpoint receipt")
    cumulative = completion.get("cumulative_useful_seconds")
    eligible_rows = completion.get("eligible_row_count")
    if (
        isinstance(cumulative, bool)
        or not isinstance(cumulative, (int, float))
        or cumulative < M2_RESUME_MINIMUM_PRIOR_USEFUL_SECONDS
        or isinstance(eligible_rows, bool)
        or not isinstance(eligible_rows, int)
        or eligible_rows < M2_RESUME_MINIMUM_ELIGIBLE_ROWS
    ):
        raise RuntimeError("M2 resume receipt is below its verified checkpoint floors")
    budget = _assert_budget(
        control_dir=control_dir,
        proposed_usd=M2_RESUME_WORST_CASE_USD,
        external_scheduled_usd=external_scheduled_usd,
        available_usd=available_usd,
        minimum_reserve_usd=M2_RESUME_MINIMUM_RESERVE_USD,
    )
    call = resume_m2.spawn()
    receipt = {
        "schema_version": "deepmew.modal_first_wave_launch.v1",
        "status": "launched_detached_resume",
        "job_id": "m2",
        "resume_attempt": 1,
        "call_id": call.object_id,
        "run_dir": str(RUN_ROOT / "m2"),
        "resources": {
            "cpu_physical_cores": 4,
            "memory_mib": 8_192,
            "gpu": None,
            "cumulative_floor_seconds": MINER_USEFUL_SECONDS,
            "verified_minimum_prior_seconds": M2_RESUME_MINIMUM_PRIOR_USEFUL_SECONDS,
            "verified_minimum_prior_rows": M2_RESUME_MINIMUM_ELIGIBLE_ROWS,
            "required_config_sha256": M2_RESUME_REQUIRED_CONFIG_SHA256,
            "billable_bound_seconds": M2_RESUME_BILLABLE_BOUND_SECONDS,
            "function_timeout_seconds": FIRST_WAVE_TIMEOUT_SECONDS,
        },
        "worst_case_usd": M2_RESUME_WORST_CASE_USD,
        "scheduled_usd_after_launch": budget["scheduled_usd"],
        "remaining_reserve_usd": budget["remaining_reserve_usd"],
        "hard_total_cap_usd": MAX_SCHEDULE_USD,
        "minimum_uncommitted_reserve_usd": M2_RESUME_MINIMUM_RESERVE_USD,
        "superseded_standing_reserve_usd": MIN_RESERVE_USD,
        "authorization_scope": "single_m2_checkpoint_resume_under_existing_24_usd_cap",
        "launched_at": _utc_now(),
        "automatic_retries": 0,
        "resume_owned_by_cli": True,
    }
    _write_json_once(launch_path, receipt)
    print(f"[modal] m2 bounded resume detached call_id={call.object_id}")


def _launch_m4_resume(
    control_dir: Path,
    external_scheduled_usd: float,
    available_usd: float,
) -> None:
    first_launch = control_dir / "m4_resume_launch.json"
    second_launch = control_dir / "m4_resume_2_launch.json"
    if second_launch.exists():
        raise RuntimeError("the two authorized bounded M4 resume attempts are exhausted")
    if first_launch.exists():
        prior_completion_path = control_dir / "m4_resume_completion.json"
        if not prior_completion_path.is_file():
            raise RuntimeError("the first M4 resume is not durably completed")
        prior_completion = json.loads(prior_completion_path.read_text(encoding="utf-8"))
        if prior_completion.get("status") != "failed_missing_required_outputs":
            raise RuntimeError("a second M4 resume requires the recorded first-resume failure")
        resume_attempt = 2
        launch_path = second_launch
    else:
        resume_attempt = 1
        launch_path = first_launch
    completion_path = control_dir / "m4_completion.json"
    if not completion_path.is_file():
        raise RuntimeError("M4 resume requires the original completion receipt")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "failed_missing_required_outputs":
        raise RuntimeError("M4 resume requires the recorded finalization failure")
    tranche_path = control_dir / "tranche_receipt.json"
    if not tranche_path.is_file():
        raise RuntimeError("M4 resume requires the frozen tranche receipt")
    tranche = json.loads(tranche_path.read_text(encoding="utf-8"))
    tranche_dir = tranche.get("tranche_dir")
    if tranche.get("status") != "frozen_immutable" or not isinstance(tranche_dir, str):
        raise RuntimeError("M4 resume requires an immutable frozen tranche")
    budget = _assert_budget(
        control_dir=control_dir,
        proposed_usd=M4_RESUME_WORST_CASE_USD,
        external_scheduled_usd=external_scheduled_usd,
        available_usd=available_usd,
    )
    call = resume_m4.spawn(tranche_dir)
    receipt = {
        "schema_version": "deepmew.modal_first_wave_launch.v1",
        "status": "launched_detached_resume",
        "job_id": "m4",
        "resume_attempt": resume_attempt,
        "call_id": call.object_id,
        "run_dir": str(RUN_ROOT / "m4"),
        "resources": {
            "cpu_physical_cores": 4,
            "memory_mib": 16_384,
            "gpu": "T4",
            "cumulative_floor_seconds": M4_USEFUL_SECONDS,
            "verified_minimum_prior_seconds": M4_RESUME_MINIMUM_PRIOR_GPU_SECONDS,
            "billable_bound_seconds": M4_RESUME_BILLABLE_BOUND_SECONDS,
            "function_timeout_seconds": M4_RESUME_TIMEOUT_SECONDS,
            "tranche_dir": tranche_dir,
        },
        "worst_case_usd": M4_RESUME_WORST_CASE_USD,
        "scheduled_usd_after_launch": budget["scheduled_usd"],
        "remaining_reserve_usd": budget["remaining_reserve_usd"],
        "hard_total_cap_usd": MAX_SCHEDULE_USD,
        "minimum_uncommitted_reserve_usd": MIN_RESERVE_USD,
        "launched_at": _utc_now(),
        "automatic_retries": 0,
        "resume_owned_by_cli": True,
    }
    _write_json_once(launch_path, receipt)
    print(f"[modal] m4 bounded resume detached call_id={call.object_id}")


def _safe_completion_receipt(value: Any, job_id: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("job_id") != job_id:
        raise RuntimeError("detached call returned an invalid safe receipt")
    allowed = {
        "schema_version",
        "job_id",
        "kind",
        "status",
        "process_return_code",
        "stopped_at_budget_bound",
        "required_outputs_complete",
        "volume_commit_count",
        "cumulative_useful_seconds",
        "eligible_row_count",
        "research_output_exists",
        "metrics",
        "safety_counters",
        "failure_category",
    }

    return {name: value.get(name) for name in allowed}


def _read_status(control_dir: Path) -> None:
    statuses = []
    for job_id in ("m1", "m2", "m3", "m4"):
        is_m2_resume = job_id == "m2" and (control_dir / "m2_resume_launch.json").is_file()
        path = control_dir / ("m2_resume_launch.json" if is_m2_resume else f"{job_id}_launch.json")
        if not path.is_file():
            statuses.append({"job_id": job_id, "status": "not_launched"})
            continue
        launch = json.loads(path.read_text(encoding="utf-8"))
        call_id = launch.get("call_id")
        if not isinstance(call_id, str):
            statuses.append({"job_id": job_id, "status": "invalid_local_launch_receipt"})
            continue
        call = modal.FunctionCall.from_id(call_id)
        try:
            value = call.get(timeout=0)
        except (modal.exception.TimeoutError, TimeoutError):
            statuses.append({"job_id": job_id, "status": "running", "call_id": call_id})
            continue
        except Exception as error:
            statuses.append(
                {
                    "job_id": job_id,
                    "status": "remote_call_failed",
                    "call_id": call_id,
                    "error_type": type(error).__name__,
                }
            )
            continue
        completion = _safe_completion_receipt(value, job_id)
        completion["call_id"] = call_id
        completion["observed_at"] = _utc_now()
        completion_path = control_dir / (
            "m2_resume_completion.json" if is_m2_resume else f"{job_id}_completion.json"
        )
        if not completion_path.exists():
            _write_json_once(completion_path, completion)
        statuses.append(completion)
    print(json.dumps({"status": "read_only_snapshot", "jobs": statuses}, indent=2, sort_keys=True))


@app.local_entrypoint()
def main(
    mode: str = "smoke",
    max_usd: float = 0.10,
    out_dir: str = str(DEFAULT_OUTPUT),
    world_samples: int = 1,
    control_dir: str = str(CONTROL_DIR),
    external_scheduled_usd: float = 0.0,
    available_usd: float = DEFAULT_AVAILABLE_USD,
) -> None:
    if FIRST_WAVE_WORST_CASE_USD > MAX_SCHEDULE_USD:
        raise RuntimeError("fixed first-wave resource schedule exceeds the hard Modal cap")
    if MAX_SCHEDULE_USD - FIRST_WAVE_WORST_CASE_USD < MIN_RESERVE_USD:
        raise RuntimeError("fixed first-wave resource schedule violates the $5 reserve")
    control = Path(control_dir)
    if mode == "launch-miners":
        _launch_miners(control, external_scheduled_usd, available_usd)

        return
    if mode == "freeze-tranche":
        _assert_budget(
            control_dir=control,
            proposed_usd=TRANCHE_FREEZE_WORST_CASE_USD,
            external_scheduled_usd=external_scheduled_usd,
            available_usd=available_usd,
        )
        receipt = _freeze_tranche(control)
        print(json.dumps(receipt, indent=2, sort_keys=True))

        return
    if mode == "launch-trainers":
        _launch_trainers(control, external_scheduled_usd, available_usd)

        return
    if mode == "resume-m2":
        _launch_m2_resume(control, external_scheduled_usd, available_usd)

        return
    if mode == "resume-m4":
        _launch_m4_resume(control, external_scheduled_usd, available_usd)

        return
    if mode in {"status", "monitor"}:
        _read_status(control)

        return
    if mode != "smoke":
        raise ValueError(
            "--mode must be smoke, launch-miners, freeze-tranche, launch-trainers, "
            "resume-m2, resume-m4, status, or monitor"
        )
    if max_usd < 0:
        raise ValueError("--max-usd must be non-negative")
    if world_samples < 1:
        raise ValueError("--world-samples must be positive")
    print(
        f"[budget] 4 CPU cores + 8 GiB x {TIMEOUT_SECONDS}s "
        f"-> worst-case ${WORST_CASE_USD:.6f} (cap ${max_usd:.6f})"
    )
    if WORST_CASE_USD > max_usd:
        raise RuntimeError(
            f"budget gate rejected smoke: ${WORST_CASE_USD:.6f} exceeds ${max_usd:.6f}"
        )
    result = privileged_teacher_smoke.remote(world_samples)
    target = Path(out_dir)
    _save_safe_result(result, target)
    print(f"[modal] safe aggregate smoke saved to {target}")
