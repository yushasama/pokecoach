"""Run the frozen X1 and X2 first-wave training floors on two L4 containers.

The lanes are independent, equal-dollar, and use the sealed payload's existing
terminal-reward PPO trainer. Automatic retries are disabled. Paid work occurs
only through the explicit local entrypoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from deepmew.rl.league_checkpoint import snapshot_league_checkpoint
from scripts import modal_direct_exploiter as base


GPU_LABEL = "L4"
LANES = ("X1", "X2")
TRAINING_FLOOR_SECONDS = 11_900.0
CHECKPOINT_INTERVAL_SECONDS = 300.0
VOLUME_COMMIT_INTERVAL_SECONDS = 120.0
MILESTONE_FRACTIONS = "0.25,0.5,0.75,1.0"
REMOTE_TRAINING_ROOT = PurePosixPath("/x1x2/training_v2")
REMOTE_RESUME_ROOT = PurePosixPath("/x1x2/training_v1")
LOCAL_RECEIPT_ROOT = base.MISSION / "training"
FUNCTION_TIMEOUT_SECONDS = 14_400

LANE_SPECS: dict[str, dict[str, Any]] = {
    "X1": {
        "campaign": "X1_CAMPAIGN_CONFIG.json",
        "artifact_map": "X1_REMOTE_ARTIFACT_MAP_V2.json",
        "seed": 9_271_301,
        "resume_checkpoint_sha256": "0306dd93bd5b478645729ffd049328b40046c0834d6d7bbfc3787220ceff3397",
        "resume_generation_id": "g00000000000000000010-f9195b466d12439d94505bdec54f37a9",
    },
    "X2": {
        "campaign": "X2_CAMPAIGN_CONFIG.json",
        "artifact_map": "X2_REMOTE_ARTIFACT_MAP_V2.json",
        "seed": 9_271_327,
        "resume_checkpoint_sha256": "1688cd85196066522f1072febe3aeaf6a232ead5c08d09abf71d3a1b208c2061",
        "resume_generation_id": "g00000000000000000010-0514b53219694e0293aab9c2e2376f19",
    },
}


class TrainingLaunchError(RuntimeError):
    """A frozen training identity, safety, or completion gate failed."""


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")

    return hashlib.sha256(payload).hexdigest()


def _training_command(lane: str, run_directory: Path) -> list[str]:
    if lane not in LANE_SPECS:
        raise TrainingLaunchError(f"unknown training lane: {lane}")
    spec = LANE_SPECS[lane]

    return [
        sys.executable,
        str(base.REMOTE_REPO / "scripts" / "run_terminal_league_ppo.py"),
        "--campaign-config",
        str(base.REMOTE_MISSION / str(spec["campaign"])),
        "--run-dir",
        str(run_directory),
        "--init-checkpoint",
        str(base.REMOTE_MISSION / "assets" / "r7.pt"),
        "--artifact-map",
        str(base.REMOTE_MISSION / str(spec["artifact_map"])),
        "--minimum-cumulative-seconds",
        str(TRAINING_FLOOR_SECONDS),
        "--checkpoint-interval-seconds",
        str(CHECKPOINT_INTERVAL_SECONDS),
        "--device",
        "cuda",
        "--precision",
        "fp32",
        "--seed",
        str(spec["seed"]),
        "--rollout-decisions",
        str(base.ROLLOUT_DECISIONS),
        "--learning-rate",
        "0.00003",
        "--weight-decay",
        "0.0",
        "--clip-coef",
        "0.15",
        "--value-coef",
        "0.5",
        "--entropy-coef",
        "0.005",
        "--max-grad-norm",
        "0.5",
        "--update-epochs",
        str(base.UPDATE_EPOCHS),
        "--minibatch-size",
        str(base.MINIBATCH_SIZE),
        "--critic-hidden-dim",
        "128",
        "--profile",
        "--milestone-fractions",
        MILESTONE_FRACTIONS,
        "--resume",
    ]


def _commit_volume_until_stopped(stop_event: threading.Event) -> None:
    while not stop_event.wait(VOLUME_COMMIT_INTERVAL_SECONDS):
        base.OUTPUT_VOLUME.commit()


def _require_completed_summary(
    lane: str,
    run_directory: Path,
    summary: Mapping[str, Any],
) -> None:
    if summary.get("status") != "minimum_training_floor_reached":
        raise TrainingLaunchError(f"{lane} did not reach its frozen training floor")
    progress = summary.get("progress")
    if not isinstance(progress, Mapping):
        raise TrainingLaunchError(f"{lane} summary lacks progress")
    if float(progress.get("cumulative_training_seconds", 0.0)) < TRAINING_FLOOR_SECONDS:
        raise TrainingLaunchError(f"{lane} cumulative training floor is short")
    if summary.get("safety") != {
        "crashes": 0,
        "illegal_actions": 0,
        "timeouts": 0,
    }:
        raise TrainingLaunchError(f"{lane} training was not safety-clean")
    for label in ("025pct", "050pct", "075pct", "100pct"):
        _require_milestone_bundle(lane, run_directory / "milestones" / label)


def _require_milestone_bundle(lane: str, milestone: Path) -> None:
    receipt_path = milestone / "receipt.json"
    if not receipt_path.is_file():
        raise TrainingLaunchError(f"{lane} milestone receipt is missing: {milestone.name}")
    receipt = base._read_json(receipt_path)
    if receipt.get("schema_version") != "deepmew.direct_exploiter_milestone.v2":
        raise TrainingLaunchError(f"{lane} milestone schema is incompatible: {milestone.name}")
    checkpoint = receipt.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("storage_format") != "immutable_generation_v1":
        raise TrainingLaunchError(f"{lane} milestone checkpoint is incompatible: {milestone.name}")
    files = receipt.get("files")
    if not isinstance(files, Mapping):
        raise TrainingLaunchError(f"{lane} milestone file manifest is missing: {milestone.name}")
    required_names = {"actor_latest.pt", "status.json", "league_checkpoint.pt.latest.json"}
    if not required_names.issubset(files):
        raise TrainingLaunchError(f"{lane} milestone required files are missing: {milestone.name}")
    generation_files = [
        name
        for name in files
        if str(name).startswith("league_checkpoint.pt.generations/")
        and str(name).endswith(".pt")
    ]
    if len(generation_files) != 1:
        raise TrainingLaunchError(f"{lane} milestone generation is incomplete: {milestone.name}")
    for name, expected in files.items():
        if not isinstance(name, str) or not isinstance(expected, Mapping):
            raise TrainingLaunchError(f"{lane} milestone file receipt is malformed: {milestone.name}")
        path = milestone / name
        if not path.is_file():
            raise TrainingLaunchError(f"{lane} milestone file is missing: {milestone.name}/{name}")
        if path.stat().st_size != int(expected.get("bytes", -1)):
            raise TrainingLaunchError(f"{lane} milestone file size disagrees: {milestone.name}/{name}")
        if base._sha256_file(path) != expected.get("sha256"):
            raise TrainingLaunchError(f"{lane} milestone file hash disagrees: {milestone.name}/{name}")


def _prepare_resume_checkpoint(lane: str, run_directory: Path) -> dict[str, Any]:
    spec = LANE_SPECS[lane]
    source = Path(str(REMOTE_RESUME_ROOT / lane.lower() / "league_checkpoint.pt"))
    destination = run_directory / "league_checkpoint.pt"
    snapshot = snapshot_league_checkpoint(source, destination)
    if snapshot["storage_format"] != "immutable_generation_v1":
        raise TrainingLaunchError(f"{lane} resume checkpoint storage format is incompatible")
    if snapshot["generation_id"] != spec["resume_generation_id"]:
        raise TrainingLaunchError(f"{lane} resume generation identity disagrees")
    if snapshot["checkpoint_sha256"] != spec["resume_checkpoint_sha256"]:
        raise TrainingLaunchError(f"{lane} resume checkpoint hash disagrees")
    receipt = {
        "schema_version": "deepmew.modal_training_resume_source.v1",
        "lane": lane,
        "source": str(source),
        "destination": str(destination),
        "snapshot": snapshot,
    }
    base._atomic_json(run_directory / "RESUME_SOURCE_RECEIPT.json", receipt)

    return receipt


@base.app.function(
    image=base.image,
    cpu=base.CPU_CORES,
    memory=base.MEMORY_MIB,
    timeout=FUNCTION_TIMEOUT_SECONDS,
    retries=0,
    max_containers=2,
    volumes={str(base.REMOTE_VOLUME_ROOT): base.OUTPUT_VOLUME},
    name="train_direct_exploiter",
)
def train_direct_exploiter(
    lane: str,
    requested_at_epoch_seconds: float,
) -> dict[str, Any]:
    if lane not in LANE_SPECS:
        raise TrainingLaunchError(f"unknown training lane: {lane}")
    verification = base._require_payload_v2(remote=True, verify_imports=True)
    base.OUTPUT_VOLUME.reload()
    run_directory = Path(str(REMOTE_TRAINING_ROOT / lane.lower()))
    if run_directory.exists():
        raise TrainingLaunchError(f"training output already exists: {run_directory}")
    run_directory.mkdir(parents=True, exist_ok=False)
    resume_source = _prepare_resume_checkpoint(lane, run_directory)
    command = _training_command(lane, run_directory)
    command_sha256 = _sha256_json(command)
    stop_event = threading.Event()
    committer = threading.Thread(
        target=_commit_volume_until_stopped,
        args=(stop_event,),
        daemon=True,
    )
    function_started_epoch = time.time()
    function_started = time.perf_counter()
    committer.start()
    return_code = -1
    try:
        with (run_directory / "runner.log").open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as log_stream:
            process = subprocess.Popen(
                command,
                cwd=str(base.REMOTE_REPO),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            if process.stdout is None:
                raise TrainingLaunchError("training runner stdout was not captured")
            for line in process.stdout:
                log_stream.write(line)
                log_stream.flush()
                print(f"[{lane}] {line}", end="", flush=True)
            return_code = process.wait()
    finally:
        stop_event.set()
        committer.join(timeout=VOLUME_COMMIT_INTERVAL_SECONDS + 5.0)
        base.OUTPUT_VOLUME.commit()
    if return_code != 0:
        raise TrainingLaunchError(f"{lane} training runner exited {return_code}")
    summary = base._read_json(run_directory / "summary.json")
    _require_completed_summary(lane, run_directory, summary)
    function_execution_seconds = time.perf_counter() - function_started
    function_ended_epoch = time.time()
    receipt = {
        "schema_version": "deepmew.modal_direct_exploiter_training_receipt.v1",
        "status": "PASS",
        "lane": lane,
        "gpu": GPU_LABEL,
        "gpu_count": 1,
        "container_identity": os.environ.get("MODAL_TASK_ID", ""),
        "requested_at_epoch_seconds": requested_at_epoch_seconds,
        "function_started_epoch_seconds": function_started_epoch,
        "function_ended_epoch_seconds": function_ended_epoch,
        "function_execution_seconds": function_execution_seconds,
        "queue_delay_seconds": max(0.0, function_started_epoch - requested_at_epoch_seconds),
        "resource_rate_usd_per_second": base._resource_rate(GPU_LABEL),
        "measured_resource_seconds_cost_usd": function_execution_seconds
        * base._resource_rate(GPU_LABEL),
        "training_floor_seconds": TRAINING_FLOOR_SECONDS,
        "checkpoint_selection_rule": "final_checkpoint_unless_safety_failure",
        "command_sha256": command_sha256,
        "payload_identity_sha256": verification["payload_identity_sha256"],
        "campaign_config_sha256": base._sha256_file(
            Path(str(base.REMOTE_MISSION / str(LANE_SPECS[lane]["campaign"])))
        ),
        "run_directory": str(run_directory),
        "resume_source": resume_source,
        "summary": summary,
    }
    base._atomic_json(run_directory / "TRAINING_RECEIPT.json", receipt)
    base.OUTPUT_VOLUME.commit()

    return receipt


def _local_receipt_path(lane: str) -> Path:
    return LOCAL_RECEIPT_ROOT / f"{lane}_TRAINING_RECEIPT.json"


@base.app.function(
    image=base.image,
    cpu=1.0,
    memory=2_048,
    timeout=300,
    retries=0,
    max_containers=1,
    volumes={str(base.REMOTE_VOLUME_ROOT): base.OUTPUT_VOLUME},
    name="verify_training_resume_sources_cpu",
)
def verify_training_resume_sources_cpu() -> dict[str, Any]:
    verification = base._require_payload_v2(remote=True, verify_imports=True)
    base.OUTPUT_VOLUME.reload()
    with tempfile.TemporaryDirectory(prefix="deepmew-x1-x2-resume-probe-") as temporary:
        temporary_root = Path(temporary)
        sources = {}
        for lane in LANES:
            lane_directory = temporary_root / lane.lower()
            lane_directory.mkdir(parents=True)
            sources[lane] = _prepare_resume_checkpoint(lane, lane_directory)

    return {
        "schema_version": "deepmew.modal_training_resume_source_probe.v1",
        "status": "PASS",
        "gpu_requested": False,
        "gpu_count": 0,
        "cabt_games": 0,
        "rollouts": 0,
        "optimizer_updates": 0,
        "automatic_retries": 0,
        "payload_identity_sha256": verification["payload_identity_sha256"],
        "sources": sources,
    }


@base.app.local_entrypoint()
def training_main(verify_resume_cpu_only: bool = False) -> None:
    base._require_payload_v2(remote=False, verify_imports=False)
    if verify_resume_cpu_only:
        receipt_path = LOCAL_RECEIPT_ROOT / "X1_X2_RESUME_SOURCE_REMOTE_VERIFICATION_RECEIPT.json"
        if receipt_path.exists():
            raise TrainingLaunchError(f"resume-source verification receipt already exists: {receipt_path}")
        result = verify_training_resume_sources_cpu.remote()
        if result.get("status") != "PASS":
            raise TrainingLaunchError("resume-source CPU verification did not pass")
        base._atomic_json(receipt_path, result)
        print(json.dumps(result, indent=2, sort_keys=True))

        return
    for lane in LANES:
        if _local_receipt_path(lane).exists():
            raise TrainingLaunchError(f"local training receipt already exists for {lane}")
    requested_at = time.time()
    configured = train_direct_exploiter.with_options(
        gpu=base.GPU_REQUESTS[GPU_LABEL],
        timeout=FUNCTION_TIMEOUT_SECONDS,
        retries=0,
        max_containers=2,
    )
    results = list(
        configured.starmap(
            [(lane, requested_at) for lane in LANES],
            order_outputs=True,
            return_exceptions=True,
        )
    )
    failures = [result for result in results if isinstance(result, BaseException)]
    if failures:
        raise TrainingLaunchError(f"training lane failed: {failures[0]}")
    container_ids = {str(result["container_identity"]) for result in results}
    if "" in container_ids or len(container_ids) != 2:
        raise TrainingLaunchError("X1 and X2 did not run in two distinct containers")
    overlap_started = max(float(result["function_started_epoch_seconds"]) for result in results)
    overlap_ended = min(float(result["function_ended_epoch_seconds"]) for result in results)
    if overlap_started >= overlap_ended:
        raise TrainingLaunchError("X1 and X2 did not execute concurrently")
    for result in results:
        base._atomic_json(_local_receipt_path(str(result["lane"])), result)
    aggregate = {
        "schema_version": "deepmew.modal_x1_x2_training_aggregate.v1",
        "status": "PASS",
        "lanes": list(LANES),
        "distinct_containers": 2,
        "concurrent_execution": True,
        "equal_training_floor_seconds": TRAINING_FLOOR_SECONDS,
        "total_measured_resource_seconds_cost_usd": sum(
            float(result["measured_resource_seconds_cost_usd"]) for result in results
        ),
        "receipts": results,
    }
    base._atomic_json(LOCAL_RECEIPT_ROOT / "X1_X2_TRAINING_AGGREGATE.json", aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True))
