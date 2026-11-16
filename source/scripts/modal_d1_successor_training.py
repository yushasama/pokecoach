"""Launch the isolated D1-successor mixed-league training lane."""

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

from scripts import modal_d1_successor_payload as payload


ROOT = Path(__file__).resolve().parents[1]
MISSION = payload.MISSION
PAYLOAD_STAGING = payload.STAGING_ROOT
REMOTE_REPO = payload.REMOTE_PAYLOAD_ROOT
REMOTE_MISSION = REMOTE_REPO / ".research" / "d1_successor_mixed_league_v1"
REMOTE_ASSETS = REMOTE_MISSION / "assets"
RUNTIME_PROJECT = ROOT / ".research" / "modal" / "q_gpu_runtime"
APP_NAME = str(payload.FUNCTION_IDENTITY["modal_app_name"])
FUNCTION_NAME = str(payload.FUNCTION_IDENTITY["modal_function_name"])
VOLUME_NAME = str(payload.FUNCTION_IDENTITY["modal_volume_name"])
GPU_REQUEST = "L4"
CPU_CORES = 4.0
MEMORY_MIB = 16 * 1024
TRAINING_FLOOR_SECONDS = 11_900.0
CHECKPOINT_INTERVAL_SECONDS = 300.0
MILESTONE_FRACTIONS = "0.25,0.5,0.75,1.0"
FUNCTION_TIMEOUT_SECONDS = 14_400
VOLUME_COMMIT_INTERVAL_SECONDS = 120.0
REMOTE_VOLUME_ROOT = PurePosixPath("/d1_successor")
REMOTE_TRAINING_ROOT = PurePosixPath("/d1_successor/training_v1")
LOCAL_TRAINING_RECEIPT = MISSION / "D1_SUCCESSOR_TRAINING_RECEIPT.json"
MAXIMUM_INCREMENTAL_MODAL_GUARD_USD = 4.60
EXPECTED_ARTIFACT_SHA256 = {
    "d1_actor.pt": "b402035e53c3dfa87d6b0e5c962bb61269ffbb10a3c814b5269f25524abf3564",
    "d1_actor_critic.pt": "1121ac8f15d0186022d3bf10b6b809d844f98d5de8b7419c335a34cd451cd1dc",
    "d1_critic_config_receipt.json": "11e89d76492eef1dd78ae66fa0c6cfe3e61de148bb34414b54db8ba3ef6fd4b2",
    "h2_actor.pt": "a2ce7bdf877c49bf12be38394dc3b3d88afa9fb905cbbfdd2b1bf99bec4c209e",
    "e1_actor.pt": "0473d802de80d590eddb45da5289028b428627f23713b482b9c0229fb4b1cd85",
    "r7_actor.pt": "e293433756fc7edad865f887f9e130d5cec1e4d4067da30446ddebf9b714aee2",
    "x2_actor.pt": "adf9c3946e3e75fccdc9a5f34ec98ed49f61b786d303f042c3a4a4b82b600b1a",
    "r12_hardtail_actor.pt": "ffd5dcf3104b015a5e11ed68bc194dccc0bbac072055a12f672f1b9236fddb8a",
}


class D1SuccessorLaunchError(RuntimeError):
    """A D1-successor identity, safety, completion, or paid-cap gate failed."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise D1SuccessorLaunchError(f"expected JSON object: {path}")

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


def _sha256_json(value: Any) -> str:
    rendered = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")

    return hashlib.sha256(rendered).hexdigest()


def _payload_path(root: Path, remote_path: PurePosixPath) -> Path:
    return root.joinpath(*remote_path.relative_to(REMOTE_REPO).parts)


def _require_payload(*, remote: bool) -> dict[str, Any]:
    root = Path(str(REMOTE_REPO)) if remote else PAYLOAD_STAGING
    manifest = _read_json(root / payload.MANIFEST_CONTROL_NAME)
    try:
        verification = payload.verify_staging_tree(root, manifest)
    except payload.D1SuccessorPayloadError as exc:
        raise D1SuccessorLaunchError(str(exc)) from exc
    if manifest.get("status") != "SEALED_PAID_LAUNCH_READY":
        raise D1SuccessorLaunchError("payload is not launch-ready")
    if manifest.get("function_identity") != payload.FUNCTION_IDENTITY:
        raise D1SuccessorLaunchError("function identity disagrees")
    for name, expected_hash in EXPECTED_ARTIFACT_SHA256.items():
        remote_path = REMOTE_ASSETS / name
        if payload.sha256_file(_payload_path(root, remote_path)) != expected_hash:
            raise D1SuccessorLaunchError(f"frozen artifact identity drifted: {name}")
    launcher = root / "scripts" / "modal_d1_successor_training.py"
    if payload.sha256_file(launcher) != payload.sha256_file(Path(__file__)):
        raise D1SuccessorLaunchError("function source differs from sealed launcher")
    runner_root = Path(str(REMOTE_REPO)) if remote else ROOT
    if str(runner_root) not in sys.path:
        sys.path.insert(0, str(runner_root))
    from scripts.run_terminal_league_ppo import preflight_campaign_opponents

    campaign_path = (
        root / ".research" / "d1_successor_mixed_league_v1" / "CAMPAIGN_CONFIG.json"
        if remote
        else MISSION / "CAMPAIGN_CONFIG.json"
    )
    artifact_map_path = (
        root
        / ".research"
        / "d1_successor_mixed_league_v1"
        / "D1_SUCCESSOR_REMOTE_POLICY_ARTIFACT_MAP.json"
        if remote
        else MISSION / "D1_SUCCESSOR_LOCAL_POLICY_ARTIFACT_MAP.json"
    )
    materialization = preflight_campaign_opponents(
        campaign_path,
        artifact_map_path,
        seed=211402701,
    )
    if materialization.get("opponent_count") != 10:
        raise D1SuccessorLaunchError("frozen population did not materialize all ten members")

    return {
        "schema_version": "deepmew.d1_successor_modal_payload_verification.v1",
        "status": "PASS",
        "payload_identity_sha256": manifest["payload_identity_sha256"],
        "function_identity": dict(payload.FUNCTION_IDENTITY),
        "staging_verification": verification,
        "opponent_materialization": materialization,
    }


def _require_paid_launch_preflight(*, remote: bool) -> dict[str, Any]:
    path = (
        REMOTE_MISSION / "D1_SUCCESSOR_MODAL_HARDWARE_AND_COST_GUARD.json"
        if remote
        else MISSION / "D1_SUCCESSOR_MODAL_HARDWARE_AND_COST_GUARD.json"
    )
    guard = _read_json(Path(str(path)))
    cap = float(guard["project_cap_usd"])
    actual = float(guard["authoritative_actual_usage_usd"])
    reserve = float(guard["mandatory_reserve_usd"])
    maximum_guard = float(guard["maximum_incremental_modal_guard_usd"])
    gross = cap - actual
    launchable = gross - reserve
    projected = actual + maximum_guard + reserve
    numeric = (cap, actual, reserve, maximum_guard, gross, launchable, projected)
    if not all(math.isfinite(value) and value >= 0.0 for value in numeric):
        raise D1SuccessorLaunchError("paid-cap inputs must be finite and non-negative")
    if guard.get("status") != "PAID_LAUNCH_READY":
        raise D1SuccessorLaunchError(f"paid launch blocked by status: {guard.get('status')}")
    if not math.isclose(maximum_guard, MAXIMUM_INCREMENTAL_MODAL_GUARD_USD, abs_tol=1e-9):
        raise D1SuccessorLaunchError("incremental guard disagrees with authorization")
    if not math.isclose(gross, float(guard["gross_headroom_usd"]), abs_tol=1e-9):
        raise D1SuccessorLaunchError("gross headroom arithmetic disagrees")
    if not math.isclose(launchable, float(guard["launchable_headroom_usd"]), abs_tol=1e-9):
        raise D1SuccessorLaunchError("launchable headroom arithmetic disagrees")
    if projected > cap:
        raise D1SuccessorLaunchError(
            f"paid launch exceeds cap: projected={projected:.8f} cap={cap:.8f}"
        )
    if int(guard.get("live_active_tasks", -1)) != 0 or int(guard.get("live_active_containers", -1)) != 0:
        raise D1SuccessorLaunchError("sealed prelaunch resource gate is not zero")

    return {
        "schema_version": "deepmew.d1_successor_paid_launch_preflight.v1",
        "status": "PASS",
        "project_cap_usd": cap,
        "authoritative_actual_usage_usd": actual,
        "maximum_incremental_modal_guard_usd": maximum_guard,
        "mandatory_reserve_usd": reserve,
        "projected_total_usd": projected,
        "remaining_margin_usd": cap - projected,
    }


def _training_command(run_directory: Path) -> list[str]:
    return [
        sys.executable,
        str(REMOTE_REPO / "scripts" / "run_terminal_league_ppo.py"),
        "--campaign-config",
        str(REMOTE_MISSION / "CAMPAIGN_CONFIG.json"),
        "--run-dir",
        str(run_directory),
        "--init-checkpoint",
        str(REMOTE_ASSETS / "d1_actor.pt"),
        "--init-critic-checkpoint",
        str(REMOTE_ASSETS / "d1_actor_critic.pt"),
        "--init-critic-config-receipt",
        str(REMOTE_ASSETS / "d1_critic_config_receipt.json"),
        "--artifact-map",
        str(REMOTE_MISSION / "D1_SUCCESSOR_REMOTE_POLICY_ARTIFACT_MAP.json"),
        "--minimum-cumulative-seconds",
        str(TRAINING_FLOOR_SECONDS),
        "--checkpoint-interval-seconds",
        str(CHECKPOINT_INTERVAL_SECONDS),
        "--device",
        "cuda",
        "--precision",
        "fp32",
        "--seed",
        "211402701",
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


def _require_completed_summary(run_directory: Path, summary: Mapping[str, Any]) -> None:
    if summary.get("status") != "minimum_training_floor_reached":
        raise D1SuccessorLaunchError("successor did not reach its frozen training floor")
    progress = summary.get("progress")
    if (
        not isinstance(progress, Mapping)
        or float(progress.get("cumulative_training_seconds", 0.0)) < TRAINING_FLOOR_SECONDS
    ):
        raise D1SuccessorLaunchError("successor cumulative training floor is short")
    if summary.get("safety") != {"crashes": 0, "illegal_actions": 0, "timeouts": 0}:
        raise D1SuccessorLaunchError("successor training was not safety-clean")
    for label in ("025pct", "050pct", "075pct", "100pct"):
        receipt = _read_json(run_directory / "milestones" / label / "receipt.json")
        checkpoint = receipt.get("checkpoint")
        if (
            receipt.get("schema_version") != "deepmew.direct_exploiter_milestone.v2"
            or not isinstance(checkpoint, Mapping)
            or checkpoint.get("storage_format") != "immutable_generation_v1"
        ):
            raise D1SuccessorLaunchError(f"milestone receipt is incompatible: {label}")


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
def train_d1_successor_mixed_league(requested_at_epoch_seconds: float) -> dict[str, Any]:
    verification = _require_payload(remote=True)
    preflight = _require_paid_launch_preflight(remote=True)
    OUTPUT_VOLUME.reload()
    run_directory = Path(str(REMOTE_TRAINING_ROOT))
    if run_directory.exists():
        raise D1SuccessorLaunchError(
            f"training output already exists; automatic rerun is forbidden: {run_directory}"
        )
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
                raise D1SuccessorLaunchError("runner stdout was not captured")
            for line in process.stdout:
                log_stream.write(line)
                log_stream.flush()
                print(f"[D1S] {line}", end="", flush=True)
            return_code = process.wait()
    finally:
        stop_event.set()
        committer.join(timeout=VOLUME_COMMIT_INTERVAL_SECONDS + 5.0)
        OUTPUT_VOLUME.commit()
    if return_code != 0:
        raise D1SuccessorLaunchError(f"training runner exited {return_code}")
    summary = _read_json(run_directory / "summary.json")
    _require_completed_summary(run_directory, summary)
    receipt = {
        "schema_version": "deepmew.modal_d1_successor_training_receipt.v1",
        "status": "PASS",
        "campaign": "D1_SUCCESSOR_MIXED_LEAGUE_V1",
        "gpu": "L4",
        "gpu_count": 1,
        "container_identity": os.environ.get("MODAL_TASK_ID", ""),
        "requested_at_epoch_seconds": requested_at_epoch_seconds,
        "function_started_epoch_seconds": function_started_epoch,
        "function_ended_epoch_seconds": time.time(),
        "function_execution_seconds": time.perf_counter() - function_started,
        "queue_delay_seconds": max(0.0, function_started_epoch - requested_at_epoch_seconds),
        "training_floor_seconds": TRAINING_FLOOR_SECONDS,
        "checkpoint_selection_rule": "final_checkpoint_only_no_qualification_seed_selection",
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
    configured = train_d1_successor_mixed_league.with_options(
        gpu=GPU_REQUEST,
        timeout=FUNCTION_TIMEOUT_SECONDS,
        retries=0,
        max_containers=1,
    )

    return configured.remote(requested_at)


def _launch_with_invoker(invoker: Callable[[float], dict[str, Any]]) -> dict[str, Any]:
    _require_payload(remote=False)
    _require_paid_launch_preflight(remote=False)
    if LOCAL_TRAINING_RECEIPT.exists():
        raise D1SuccessorLaunchError(
            f"local training receipt already exists; automatic duplicate is forbidden: {LOCAL_TRAINING_RECEIPT}"
        )
    result = invoker(time.time())
    if result.get("status") != "PASS":
        raise D1SuccessorLaunchError("remote training did not pass")
    _atomic_json(LOCAL_TRAINING_RECEIPT, result)

    return result


@app.local_entrypoint()
def training_main() -> None:
    print(json.dumps(_launch_with_invoker(_invoke_remote_training), indent=2, sort_keys=True))


@app.local_entrypoint()
def preflight_main() -> None:
    print(
        json.dumps(
            {
                "payload": _require_payload(remote=False),
                "budget": _require_paid_launch_preflight(remote=False),
            },
            indent=2,
            sort_keys=True,
        )
    )
