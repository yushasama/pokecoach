"""Launch the isolated H2 E1-initialized Starmie repair lane."""

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

from scripts import modal_h2_payload as payload


ROOT = Path(__file__).resolve().parents[1]
MISSION = payload.MISSION
PAYLOAD_STAGING = payload.STAGING_ROOT
REMOTE_REPO = payload.REMOTE_PAYLOAD_ROOT
REMOTE_MISSION = REMOTE_REPO / ".research" / "h2_x3_first_wave_v1" / "h2"
REMOTE_BASE_ASSETS = REMOTE_REPO / ".research" / "x1_x2_direct_exploiter_first_wave" / "assets"
REMOTE_H2_ASSETS = REMOTE_REPO / ".research" / "h2_x3_first_wave_v1" / "assets"
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
REMOTE_VOLUME_ROOT = PurePosixPath("/h2")
REMOTE_TRAINING_ROOT = PurePosixPath("/h2/training_v2")
LOCAL_TRAINING_RECEIPT = MISSION / "H2_TRAINING_RECEIPT_V2.json"
MINIMUM_TRAINING_GUARD_USD = 3.75
EXPECTED_ARTIFACT_SHA256 = {
    "e1_actor.pt": "0473d802de80d590eddb45da5289028b428627f23713b482b9c0229fb4b1cd85",
    "e1_actor_critic.pt": "6672b8d7c285d0a829ed7a26a5982250aacd7bc316890c3ff85413768f9bb432",
    "x2_actor.pt": "adf9c3946e3e75fccdc9a5f34ec98ed49f61b786d303f042c3a4a4b82b600b1a",
}


class H2LaunchError(RuntimeError):
    """An H2 identity, safety, completion, or paid-cap gate failed."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise H2LaunchError(f"expected JSON object: {path}")

    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
            encoding="ascii", newline="\n", delete=False,
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
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")

    return hashlib.sha256(rendered).hexdigest()


def _payload_path(root: Path, remote_path: PurePosixPath) -> Path:
    return root.joinpath(*remote_path.relative_to(REMOTE_REPO).parts)


def _require_payload(*, remote: bool) -> dict[str, Any]:
    root = Path(str(REMOTE_REPO)) if remote else PAYLOAD_STAGING
    manifest = _read_json(root / payload.MANIFEST_CONTROL_NAME)
    try:
        verification = payload.verify_staging_tree(root, manifest)
    except payload.H2PayloadError as exc:
        raise H2LaunchError(str(exc)) from exc
    if manifest.get("status") != "SEALED_LOCAL_PAID_LAUNCH_READY":
        raise H2LaunchError("H2 payload is not launch-ready")
    if manifest.get("function_identity") != payload.FUNCTION_IDENTITY:
        raise H2LaunchError("H2 function identity disagrees")
    expected_paths = {
        "e1_actor.pt": REMOTE_BASE_ASSETS / "e1_actor.pt",
        "e1_actor_critic.pt": REMOTE_H2_ASSETS / "e1_actor_critic.pt",
        "x2_actor.pt": REMOTE_H2_ASSETS / "x2_actor.pt",
    }
    for name, expected_hash in EXPECTED_ARTIFACT_SHA256.items():
        if payload.sha256_file(_payload_path(root, expected_paths[name])) != expected_hash:
            raise H2LaunchError(f"frozen H2 artifact identity drifted: {name}")
    launcher = root / "scripts" / "modal_h2_training.py"
    if payload.sha256_file(launcher) != payload.sha256_file(Path(__file__)):
        raise H2LaunchError("H2 function source differs from sealed launcher")
    runner_root = Path(str(REMOTE_REPO)) if remote else ROOT
    if str(runner_root) not in sys.path:
        sys.path.insert(0, str(runner_root))
    from scripts.run_terminal_league_ppo import preflight_campaign_opponents

    campaign_path = (
        root / ".research" / "h2_x3_first_wave_v1" / "h2" / "H2_CAMPAIGN_CONFIG.json"
        if remote
        else MISSION / "H2_CAMPAIGN_CONFIG.json"
    )
    artifact_map_path = (
        root / ".research" / "h2_x3_first_wave_v1" / "h2" / "H2_REMOTE_POLICY_ARTIFACT_MAP.json"
        if remote
        else MISSION / "H2_LOCAL_POLICY_ARTIFACT_MAP.json"
    )
    materialization = preflight_campaign_opponents(
        campaign_path,
        artifact_map_path,
        seed=99661570,
    )

    return {
        "schema_version": "deepmew.h2_modal_payload_verification.v1",
        "status": "PASS",
        "payload_identity_sha256": manifest["payload_identity_sha256"],
        "function_identity": dict(payload.FUNCTION_IDENTITY),
        "staging_verification": verification,
        "opponent_materialization": materialization,
    }


def _require_paid_launch_preflight(*, remote: bool) -> dict[str, Any]:
    path = REMOTE_MISSION / "H2_MODAL_HARDWARE_AND_COST_GUARD.json" if remote else MISSION / "H2_MODAL_HARDWARE_AND_COST_GUARD.json"
    guard = _read_json(Path(str(path)))
    cap = float(guard["project_cap_usd"])
    actual = float(guard["authoritative_actual_usage_usd"])
    scheduled = float(guard.get("externally_scheduled_spend_usd") or 0.0)
    local_receipts = float(guard.get("write_once_local_launch_receipts_usd") or 0.0)
    reserve = float(guard["mandatory_reserve_usd"])
    other_lane_guard = float(guard["other_authorized_lane_guard_usd"])
    gross = cap - actual
    launchable = gross - reserve
    projected = actual + scheduled + local_receipts + reserve + MINIMUM_TRAINING_GUARD_USD + other_lane_guard
    numeric = (cap, actual, scheduled, local_receipts, reserve, other_lane_guard, gross, launchable, projected)
    if not all(math.isfinite(value) and value >= 0.0 for value in numeric):
        raise H2LaunchError("H2 paid-cap inputs must be finite and non-negative")
    if guard.get("status") != "PAID_LAUNCH_READY":
        raise H2LaunchError(f"H2 paid launch blocked by status: {guard.get('status')}")
    if not math.isclose(gross, float(guard["gross_headroom_usd"]), abs_tol=1e-9):
        raise H2LaunchError("H2 gross headroom arithmetic disagrees")
    if not math.isclose(launchable, float(guard["launchable_headroom_usd"]), abs_tol=1e-9):
        raise H2LaunchError("H2 launchable headroom arithmetic disagrees")
    if projected > cap:
        raise H2LaunchError(f"combined paid launch exceeds cap: projected={projected:.8f} cap={cap:.8f}")

    return {
        "schema_version": "deepmew.h2_paid_launch_preflight.v1",
        "status": "PASS",
        "project_cap_usd": cap,
        "authoritative_actual_usage_usd": actual,
        "minimum_training_guard_usd": MINIMUM_TRAINING_GUARD_USD,
        "other_authorized_lane_guard_usd": other_lane_guard,
        "mandatory_reserve_usd": reserve,
        "projected_total_usd": projected,
    }


def _training_command(run_directory: Path) -> list[str]:
    return [
        sys.executable,
        str(REMOTE_REPO / "scripts" / "run_terminal_league_ppo.py"),
        "--campaign-config", str(REMOTE_MISSION / "H2_CAMPAIGN_CONFIG.json"),
        "--run-dir", str(run_directory),
        "--init-checkpoint", str(REMOTE_BASE_ASSETS / "e1_actor.pt"),
        "--init-critic-checkpoint", str(REMOTE_H2_ASSETS / "e1_actor_critic.pt"),
        "--artifact-map", str(REMOTE_MISSION / "H2_REMOTE_POLICY_ARTIFACT_MAP.json"),
        "--minimum-cumulative-seconds", str(TRAINING_FLOOR_SECONDS),
        "--checkpoint-interval-seconds", str(CHECKPOINT_INTERVAL_SECONDS),
        "--device", "cuda",
        "--precision", "fp32",
        "--seed", "99661570",
        "--rollout-decisions", "128",
        "--learning-rate", "0.00003",
        "--weight-decay", "0.0",
        "--clip-coef", "0.15",
        "--value-coef", "0.5",
        "--entropy-coef", "0.005",
        "--max-grad-norm", "0.5",
        "--update-epochs", "4",
        "--minibatch-size", "64",
        "--critic-hidden-dim", "128",
        "--profile",
        "--milestone-fractions", MILESTONE_FRACTIONS,
    ]


def _require_completed_summary(run_directory: Path, summary: Mapping[str, Any]) -> None:
    if summary.get("status") != "minimum_training_floor_reached":
        raise H2LaunchError("H2 did not reach its frozen training floor")
    progress = summary.get("progress")
    if not isinstance(progress, Mapping) or float(progress.get("cumulative_training_seconds", 0.0)) < TRAINING_FLOOR_SECONDS:
        raise H2LaunchError("H2 cumulative training floor is short")
    if summary.get("safety") != {"crashes": 0, "illegal_actions": 0, "timeouts": 0}:
        raise H2LaunchError("H2 training was not safety-clean")
    for label in ("025pct", "050pct", "075pct", "100pct"):
        receipt = _read_json(run_directory / "milestones" / label / "receipt.json")
        checkpoint = receipt.get("checkpoint")
        if receipt.get("schema_version") != "deepmew.direct_exploiter_milestone.v2" or not isinstance(checkpoint, Mapping) or checkpoint.get("storage_format") != "immutable_generation_v1":
            raise H2LaunchError(f"H2 milestone receipt is incompatible: {label}")


def _build_image() -> modal.Image:
    if not modal.is_local():
        return modal.Image.debian_slim(python_version="3.12")
    image = modal.Image.debian_slim(python_version="3.12").uv_sync(str(RUNTIME_PROJECT), frozen=True)
    if PAYLOAD_STAGING.is_dir():
        _require_payload(remote=False)
        image = image.add_local_dir(str(PAYLOAD_STAGING), str(REMOTE_REPO), copy=True, ignore=())

    return image.env({"PYTHONPATH": f"{REMOTE_REPO}:{REMOTE_REPO / 'scripts'}", "DEEPMEW_MODAL_PAYLOAD_ROOT": str(REMOTE_REPO), "PYTHONDONTWRITEBYTECODE": "1"})


image = _build_image()
app = modal.App(APP_NAME)
OUTPUT_VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _commit_volume_until_stopped(stop_event: threading.Event) -> None:
    while not stop_event.wait(VOLUME_COMMIT_INTERVAL_SECONDS):
        OUTPUT_VOLUME.commit()


@app.function(
    image=image, cpu=CPU_CORES, memory=MEMORY_MIB, timeout=FUNCTION_TIMEOUT_SECONDS,
    retries=0, max_containers=1, volumes={str(REMOTE_VOLUME_ROOT): OUTPUT_VOLUME}, name=FUNCTION_NAME,
)
def train_h2_e1_starmie_repair(requested_at_epoch_seconds: float) -> dict[str, Any]:
    verification = _require_payload(remote=True)
    preflight = _require_paid_launch_preflight(remote=True)
    OUTPUT_VOLUME.reload()
    run_directory = Path(str(REMOTE_TRAINING_ROOT))
    if run_directory.exists():
        raise H2LaunchError(f"H2 training output already exists: {run_directory}")
    run_directory.mkdir(parents=True, exist_ok=False)
    command = _training_command(run_directory)
    stop_event = threading.Event()
    committer = threading.Thread(target=_commit_volume_until_stopped, args=(stop_event,), daemon=True)
    function_started_epoch = time.time()
    function_started = time.perf_counter()
    committer.start()
    return_code = -1
    try:
        with (run_directory / "runner.log").open("w", encoding="utf-8", newline="\n") as log_stream:
            process = subprocess.Popen(command, cwd=str(REMOTE_REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
            if process.stdout is None:
                raise H2LaunchError("H2 runner stdout was not captured")
            for line in process.stdout:
                log_stream.write(line)
                log_stream.flush()
                print(f"[H2] {line}", end="", flush=True)
            return_code = process.wait()
    finally:
        stop_event.set()
        committer.join(timeout=VOLUME_COMMIT_INTERVAL_SECONDS + 5.0)
        OUTPUT_VOLUME.commit()
    if return_code != 0:
        raise H2LaunchError(f"H2 training runner exited {return_code}")
    summary = _read_json(run_directory / "summary.json")
    _require_completed_summary(run_directory, summary)
    receipt = {
        "schema_version": "deepmew.modal_h2_training_receipt.v1",
        "status": "PASS",
        "campaign": "H2_E1_STARMIE_REPAIR_V1",
        "gpu": "L4",
        "gpu_count": 1,
        "container_identity": os.environ.get("MODAL_TASK_ID", ""),
        "requested_at_epoch_seconds": requested_at_epoch_seconds,
        "function_started_epoch_seconds": function_started_epoch,
        "function_ended_epoch_seconds": time.time(),
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
    configured = train_h2_e1_starmie_repair.with_options(gpu=GPU_REQUEST, timeout=FUNCTION_TIMEOUT_SECONDS, retries=0, max_containers=1)

    return configured.remote(requested_at)


def _launch_with_invoker(invoker: Callable[[float], dict[str, Any]]) -> dict[str, Any]:
    _require_payload(remote=False)
    _require_paid_launch_preflight(remote=False)
    if LOCAL_TRAINING_RECEIPT.exists():
        raise H2LaunchError(f"local H2 training receipt already exists: {LOCAL_TRAINING_RECEIPT}")
    result = invoker(time.time())
    if result.get("status") != "PASS":
        raise H2LaunchError("H2 remote training did not pass")
    _atomic_json(LOCAL_TRAINING_RECEIPT, result)

    return result


@app.local_entrypoint()
def training_main() -> None:
    print(json.dumps(_launch_with_invoker(_invoke_remote_training), indent=2, sort_keys=True))


@app.local_entrypoint()
def preflight_main() -> None:
    print(json.dumps({"payload": _require_payload(remote=False), "budget": _require_paid_launch_preflight(remote=False)}, indent=2, sort_keys=True))
