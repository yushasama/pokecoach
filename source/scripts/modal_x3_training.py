"""Launch the separately sealed X3 D1 tempo/prize-plan training lane.

Importing this module launches nothing. The local entrypoint verifies the
sealed payload and runs the paid-cap preflight before it can reach a Modal
remote invocation. Automatic retries are disabled.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

import modal

from scripts import modal_x3_payload as payload


ROOT = Path(__file__).resolve().parents[1]
MISSION = payload.X3_MISSION
PAYLOAD_STAGING = payload.STAGING_ROOT
REMOTE_REPO = payload.REMOTE_PAYLOAD_ROOT
REMOTE_MISSION = (
    REMOTE_REPO
    / ".research"
    / "current_rating_bucket_exploiter_league_and_hybrid_v1"
    / "x3"
)
REMOTE_ASSET_ROOT = (
    REMOTE_REPO
    / ".research"
    / "x1_x2_direct_exploiter_first_wave"
    / "assets"
)
RUNTIME_PROJECT = ROOT / ".research" / "modal" / "q_gpu_runtime"
COST_GUARD_NAME = "X3_MODAL_HARDWARE_AND_COST_GUARD.json"
CAMPAIGN_NAME = "X3_CAMPAIGN_CONFIG.json"
ARTIFACT_MAP_NAME = "X3_MODAL_ARTIFACT_MAP.json"
APP_NAME = str(payload.FUNCTION_IDENTITY["modal_app_name"])
FUNCTION_NAME = str(payload.FUNCTION_IDENTITY["modal_function_name"])
VOLUME_NAME = str(payload.FUNCTION_IDENTITY["modal_volume_name"])
GPU_LABEL = "L4"
GPU_REQUEST = "L4"
CPU_CORES = 4.0
MEMORY_MIB = 16 * 1024
TRAINING_FLOOR_SECONDS = 11_900.0
CHECKPOINT_INTERVAL_SECONDS = 300.0
MILESTONE_FRACTIONS = "0.25,0.5,0.75,1.0"
FUNCTION_TIMEOUT_SECONDS = 14_400
VOLUME_COMMIT_INTERVAL_SECONDS = 120.0
REMOTE_VOLUME_ROOT = PurePosixPath("/x3")
REMOTE_TRAINING_ROOT = PurePosixPath("/x3/training_v2")
LOCAL_TRAINING_RECEIPT = MISSION / "X3_TRAINING_RECEIPT_V2.json"
MINIMUM_TRAINING_GUARD_USD = 3.75
EXPECTED_CAMPAIGN_SHA256 = (
    "1aab63dba508727e4ddbb72321e0eab7e72fa1e18b9dc198843280920a930235"
)
EXPECTED_ARTIFACT_SHA256 = {
    "d1_actor.pt": "b402035e53c3dfa87d6b0e5c962bb61269ffbb10a3c814b5269f25524abf3564",
    "e1_actor.pt": "0473d802de80d590eddb45da5289028b428627f23713b482b9c0229fb4b1cd85",
    "r7.pt": "e293433756fc7edad865f887f9e130d5cec1e4d4067da30446ddebf9b714aee2",
    "r12_exploiter.pt": "ffd5dcf3104b015a5e11ed68bc194dccc0bbac072055a12f672f1b9236fddb8a",
}


class X3LaunchError(RuntimeError):
    """An X3 identity, safety, completion, or paid-cap gate failed."""


def _sha256_file(path: Path) -> str:
    return payload.sha256_file(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise X3LaunchError(f"expected JSON object: {path}")

    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            encoding="ascii",
            newline="\n",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(dict(value), stream, indent=2, sort_keys=True, ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _require_payload(*, remote: bool) -> dict[str, Any]:
    root = Path(str(REMOTE_REPO)) if remote else PAYLOAD_STAGING
    manifest = _read_json(root / payload.MANIFEST_CONTROL_NAME)
    try:
        verification = payload.verify_staging_tree(root, manifest)
    except payload.X3PayloadError as exc:
        raise X3LaunchError(str(exc)) from exc
    if manifest.get("status") not in {
        "SEALED_LOCAL_PAID_LAUNCH_BLOCKED_CAP",
        "SEALED_LOCAL_PAID_LAUNCH_READY",
    }:
        raise X3LaunchError("X3 payload manifest is not sealed")
    if manifest.get("function_identity") != payload.FUNCTION_IDENTITY:
        raise X3LaunchError("X3 payload function identity disagrees")
    campaign = root.joinpath(*PurePosixPath(str(REMOTE_MISSION / CAMPAIGN_NAME)).relative_to(REMOTE_REPO).parts)
    if _sha256_file(campaign) != EXPECTED_CAMPAIGN_SHA256:
        raise X3LaunchError("frozen X3 campaign identity drifted")
    for name, expected in EXPECTED_ARTIFACT_SHA256.items():
        artifact = root.joinpath(*PurePosixPath(str(REMOTE_ASSET_ROOT / name)).relative_to(REMOTE_REPO).parts)
        if _sha256_file(artifact) != expected:
            raise X3LaunchError(f"frozen X3 artifact identity drifted: {name}")
    launcher = root / "scripts" / "modal_x3_training.py"
    if _sha256_file(launcher) != _sha256_file(Path(__file__)):
        raise X3LaunchError("X3 function source differs from sealed launcher")

    return {
        "schema_version": "deepmew.x3_modal_payload_verification.v1",
        "status": "PASS",
        "payload_identity_sha256": manifest["payload_identity_sha256"],
        "function_identity": dict(payload.FUNCTION_IDENTITY),
        "staging_verification": verification,
    }


def _cost_guard_path(*, remote: bool) -> Path:
    if remote:
        return Path(str(REMOTE_MISSION / COST_GUARD_NAME))

    return MISSION / COST_GUARD_NAME


def _require_paid_launch_preflight(*, remote: bool) -> dict[str, Any]:
    guard = _read_json(_cost_guard_path(remote=remote))
    cap = float(guard["project_cap_usd"])
    actual = float(guard["authoritative_actual_usage_usd"])
    reserve = float(guard["mandatory_reserve_usd"])
    gross = cap - actual
    launchable = gross - reserve
    scheduled = float(guard.get("externally_scheduled_spend_usd") or 0.0)
    local_receipts = float(guard.get("write_once_local_launch_receipts_usd") or 0.0)
    other_lane_guard = float(guard.get("other_authorized_lane_guard_usd") or 0.0)
    projected = actual + scheduled + local_receipts + reserve + MINIMUM_TRAINING_GUARD_USD + other_lane_guard
    numeric = (cap, actual, reserve, gross, launchable, scheduled, local_receipts, other_lane_guard, projected)
    if not all(math.isfinite(value) and value >= 0.0 for value in numeric):
        raise X3LaunchError("X3 paid-cap inputs must be finite and non-negative")
    if not math.isclose(gross, float(guard["gross_headroom_usd"]), abs_tol=1e-9):
        raise X3LaunchError("X3 gross headroom arithmetic disagrees")
    if not math.isclose(
        launchable,
        float(guard["launchable_headroom_usd"]),
        abs_tol=1e-9,
    ):
        raise X3LaunchError("X3 launchable headroom arithmetic disagrees")
    if guard.get("status") != "PAID_LAUNCH_READY":
        raise X3LaunchError(f"paid launch blocked by cost status: {guard.get('status')}")
    if projected > cap:
        raise X3LaunchError(
            f"paid launch exceeds cap: projected={projected:.8f} cap={cap:.8f}"
        )

    return {
        "schema_version": "deepmew.x3_paid_launch_preflight.v1",
        "status": "PASS",
        "project_cap_usd": cap,
        "authoritative_actual_usage_usd": actual,
        "externally_scheduled_spend_usd": scheduled,
        "write_once_local_launch_receipts_usd": local_receipts,
        "mandatory_reserve_usd": reserve,
        "minimum_training_guard_usd": MINIMUM_TRAINING_GUARD_USD,
        "other_authorized_lane_guard_usd": other_lane_guard,
        "projected_total_usd": projected,
    }


def _training_command(run_directory: Path) -> list[str]:
    return [
        sys.executable,
        str(REMOTE_REPO / "scripts" / "run_terminal_league_ppo.py"),
        "--campaign-config",
        str(REMOTE_MISSION / CAMPAIGN_NAME),
        "--run-dir",
        str(run_directory),
        "--init-checkpoint",
        str(REMOTE_ASSET_ROOT / "r7.pt"),
        "--artifact-map",
        str(REMOTE_MISSION / ARTIFACT_MAP_NAME),
        "--minimum-cumulative-seconds",
        str(TRAINING_FLOOR_SECONDS),
        "--checkpoint-interval-seconds",
        str(CHECKPOINT_INTERVAL_SECONDS),
        "--device",
        "cuda",
        "--precision",
        "fp32",
        "--seed",
        "9271351",
        "--rollout-decisions",
        "128",
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
        "4",
        "--minibatch-size",
        "64",
        "--critic-hidden-dim",
        "128",
        "--profile",
        "--milestone-fractions",
        MILESTONE_FRACTIONS,
    ]


def _sha256_json(value: Any) -> str:
    rendered = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")

    return hashlib.sha256(rendered).hexdigest()


def _require_completed_summary(run_directory: Path, summary: Mapping[str, Any]) -> None:
    if summary.get("status") != "minimum_training_floor_reached":
        raise X3LaunchError("X3 did not reach its frozen training floor")
    progress = summary.get("progress")
    if not isinstance(progress, Mapping):
        raise X3LaunchError("X3 summary lacks progress")
    if float(progress.get("cumulative_training_seconds", 0.0)) < TRAINING_FLOOR_SECONDS:
        raise X3LaunchError("X3 cumulative training floor is short")
    if summary.get("safety") != {
        "crashes": 0,
        "illegal_actions": 0,
        "timeouts": 0,
    }:
        raise X3LaunchError("X3 training was not safety-clean")
    for label in ("025pct", "050pct", "075pct", "100pct"):
        receipt = _read_json(run_directory / "milestones" / label / "receipt.json")
        if receipt.get("schema_version") != "deepmew.direct_exploiter_milestone.v2":
            raise X3LaunchError(f"X3 milestone schema is incompatible: {label}")
        checkpoint = receipt.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise X3LaunchError(f"X3 milestone checkpoint is missing: {label}")
        if checkpoint.get("storage_format") != "immutable_generation_v1":
            raise X3LaunchError(f"X3 milestone checkpoint is incompatible: {label}")


def _build_image() -> modal.Image:
    if not modal.is_local():
        return modal.Image.debian_slim(python_version="3.12")
    image = modal.Image.debian_slim(python_version="3.12").uv_sync(
        str(RUNTIME_PROJECT),
        frozen=True,
    )
    if PAYLOAD_STAGING.is_dir():
        _require_payload(remote=False)
        image = image.add_local_dir(
            str(PAYLOAD_STAGING),
            str(REMOTE_REPO),
            copy=True,
            ignore=(),
        )

    return image.env(
        {
            "PYTHONPATH": f"{REMOTE_REPO}:{REMOTE_REPO / 'scripts'}",
            "DEEPMEW_MODAL_PAYLOAD_ROOT": str(REMOTE_REPO),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )


image = _build_image()
app = modal.App(APP_NAME)
OUTPUT_VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _commit_volume_until_stopped(stop_event: threading.Event) -> None:
    while not stop_event.wait(VOLUME_COMMIT_INTERVAL_SECONDS):
        OUTPUT_VOLUME.commit()


@app.function(
    image=image,
    cpu=CPU_CORES,
    memory=MEMORY_MIB,
    timeout=FUNCTION_TIMEOUT_SECONDS,
    retries=0,
    max_containers=1,
    volumes={str(REMOTE_VOLUME_ROOT): OUTPUT_VOLUME},
    name=FUNCTION_NAME,
)
def train_x3_d1_tempo_prize_plan(requested_at_epoch_seconds: float) -> dict[str, Any]:
    verification = _require_payload(remote=True)
    preflight = _require_paid_launch_preflight(remote=True)
    OUTPUT_VOLUME.reload()
    run_directory = Path(str(REMOTE_TRAINING_ROOT))
    if run_directory.exists():
        raise X3LaunchError(f"X3 training output already exists: {run_directory}")
    run_directory.mkdir(parents=True, exist_ok=False)
    command = _training_command(run_directory)
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
                cwd=str(REMOTE_REPO),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            if process.stdout is None:
                raise X3LaunchError("X3 runner stdout was not captured")
            for line in process.stdout:
                log_stream.write(line)
                log_stream.flush()
                print(f"[X3] {line}", end="", flush=True)
            return_code = process.wait()
    finally:
        stop_event.set()
        committer.join(timeout=VOLUME_COMMIT_INTERVAL_SECONDS + 5.0)
        OUTPUT_VOLUME.commit()
    if return_code != 0:
        raise X3LaunchError(f"X3 training runner exited {return_code}")
    summary = _read_json(run_directory / "summary.json")
    _require_completed_summary(run_directory, summary)
    ended_epoch = time.time()
    receipt = {
        "schema_version": "deepmew.modal_x3_training_receipt.v1",
        "status": "PASS",
        "campaign": "X3_D1_TEMPO_PRIZE_PLAN_EXPLOITER",
        "gpu": GPU_LABEL,
        "gpu_count": 1,
        "container_identity": os.environ.get("MODAL_TASK_ID", ""),
        "requested_at_epoch_seconds": requested_at_epoch_seconds,
        "function_started_epoch_seconds": function_started_epoch,
        "function_ended_epoch_seconds": ended_epoch,
        "function_execution_seconds": time.perf_counter() - function_started,
        "queue_delay_seconds": max(0.0, function_started_epoch - requested_at_epoch_seconds),
        "training_floor_seconds": TRAINING_FLOOR_SECONDS,
        "checkpoint_selection_rule": "final_checkpoint_unless_safety_failure",
        "command_sha256": _sha256_json(command),
        "payload_identity_sha256": verification["payload_identity_sha256"],
        "function_identity": dict(payload.FUNCTION_IDENTITY),
        "paid_launch_preflight": preflight,
        "run_directory": str(run_directory),
        "summary": summary,
    }
    _atomic_json(run_directory / "TRAINING_RECEIPT.json", receipt)
    OUTPUT_VOLUME.commit()

    return receipt


def _invoke_remote_training(requested_at: float) -> dict[str, Any]:
    configured = train_x3_d1_tempo_prize_plan.with_options(
        gpu=GPU_REQUEST,
        timeout=FUNCTION_TIMEOUT_SECONDS,
        retries=0,
        max_containers=1,
    )

    return configured.remote(requested_at)


def _launch_with_invoker(
    invoker: Callable[[float], dict[str, Any]],
) -> dict[str, Any]:
    _require_payload(remote=False)
    _require_paid_launch_preflight(remote=False)
    if LOCAL_TRAINING_RECEIPT.exists():
        raise X3LaunchError(f"local X3 training receipt already exists: {LOCAL_TRAINING_RECEIPT}")
    result = invoker(time.time())
    if result.get("status") != "PASS":
        raise X3LaunchError("X3 remote training did not pass")
    _atomic_json(LOCAL_TRAINING_RECEIPT, result)

    return result


@app.local_entrypoint()
def training_main() -> None:
    result = _launch_with_invoker(_invoke_remote_training)
    print(json.dumps(result, indent=2, sort_keys=True))


@app.local_entrypoint()
def preflight_main() -> None:
    print(json.dumps({"payload": _require_payload(remote=False), "budget": _require_paid_launch_preflight(remote=False)}, indent=2, sort_keys=True))
