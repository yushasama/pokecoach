"""Budget-gated Modal launcher for terminal-league PPO research lanes.

This file only defines the launcher. Importing it does not hydrate resources,
create the named Volume, or invoke a Modal Function. A paid run still requires
an explicit command such as::

    modal run --detach scripts/modal_terminal_league.py --lane starmie --detached-ack
    modal run --detach scripts/modal_terminal_league.py --lane grimmsnarl --detached-ack

The remote subprocess writes only below the mounted research Volume. Source,
configuration, deck manifests, and frozen policy artifacts are image inputs and
therefore read-only at execution time.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import modal


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_terminal_league_ppo.py"
LEDGER_PATH = (
    ROOT
    / ".research"
    / "terminal_league_starmie_grimmsnarl_v0"
    / "MODAL_COST_LEDGER.json"
)
RUNTIME_PROJECT = ROOT / ".research" / "modal" / "q_gpu_runtime"

APP_NAME = "deepmew-terminal-league"
VOLUME_NAME = "deepmew-terminal-league"
REMOTE_VOLUME_ROOT = PurePosixPath("/terminal_league")
REMOTE_RUN_ROOT = REMOTE_VOLUME_ROOT / "runs"
REMOTE_LEDGER = PurePosixPath(
    "/repo/.research/terminal_league_starmie_grimmsnarl_v0/MODAL_COST_LEDGER.json"
)

GPU_TYPE = "T4"
CPU_CORES = 4.0
MEMORY_MIB = 16 * 1024
RETRIES = 3
CHECKPOINT_INTERVAL_SECONDS = 600
CHECKPOINT_STALL_SECONDS = 1_200
VOLUME_COMMIT_INTERVAL_SECONDS = 300
STARMIE_FUNCTION_SECONDS = 6 * 60 * 60 + 30 * 60
GRIMMSNARL_FUNCTION_SECONDS = 4 * 60 * 60 + 30 * 60
PRECISION = "fp32"
DEVICE = "cuda"
USD_PER_HOUR = Decimal("0.906912")
SECONDS_PER_HOUR = Decimal(3600)
SAFETY_HARD_STOP_EXIT_CODE = 86
TRAINING_HARD_STOP_EXIT_CODE = 87
REMOTE_R7_CHECKPOINT = PurePosixPath("/repo/.research/terminal_league_assets/r7.pt")
REMOTE_ARTIFACT_MAP = PurePosixPath(
    "/repo/.research/terminal_league_starmie_grimmsnarl_v0/modal_artifact_map.json"
)

EXPECTED_LEDGER_SCHEMA = "deepmew.terminal_league_cost_ledger.v2"
EXPECTED_MISSION = "terminal-league-starmie-grimmsnarl-v0"
BUDGET_GUARD_SCHEMA = "deepmew.terminal_league_budget_guard.v1"
MONEY_QUANTUM = Decimal("0.000001")


class TerminalLeagueLaunchError(RuntimeError):
    """A local contract or budget gate rejected a terminal-league launch."""


@dataclass(frozen=True)
class LaneSpec:
    name: str
    cumulative_seconds: int

    @property
    def cumulative_hours(self) -> Decimal:
        return Decimal(self.cumulative_seconds) / SECONDS_PER_HOUR

    @property
    def scheduled_cost_usd(self) -> Decimal:
        return self.cumulative_hours * USD_PER_HOUR

    @property
    def retry_loss_reserve_usd(self) -> Decimal:
        retry_seconds = Decimal(RETRIES * CHECKPOINT_STALL_SECONDS)

        return retry_seconds / SECONDS_PER_HOUR * USD_PER_HOUR

    @property
    def guarded_commitment_usd(self) -> Decimal:
        return self.scheduled_cost_usd + self.retry_loss_reserve_usd

    @property
    def remote_run_dir(self) -> PurePosixPath:
        return REMOTE_RUN_ROOT / self.name


LANES: dict[str, LaneSpec] = {
    "starmie": LaneSpec(name="starmie", cumulative_seconds=21_600),
    "grimmsnarl": LaneSpec(name="grimmsnarl", cumulative_seconds=14_400),
}


def _build_image() -> modal.Image:
    source_ignore = (
        ".git",
        "**/__pycache__",
        "**/.pytest_cache",
        "**/*.pyc",
        "**/*.pyo",
        "**/*.log",
    )
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .uv_sync(str(RUNTIME_PROJECT), frozen=True)
        .add_local_dir(str(ROOT / "cg"), "/repo/cg", copy=True, ignore=source_ignore)
        .add_local_dir(
            str(ROOT / "deepmew"),
            "/repo/deepmew",
            copy=True,
            ignore=source_ignore,
        )
        .add_local_dir(
            str(ROOT / "scripts"),
            "/repo/scripts",
            copy=True,
            ignore=source_ignore,
        )
        .add_local_dir(
            str(ROOT / "configs"),
            "/repo/configs",
            copy=True,
            ignore=source_ignore,
        )
        .add_local_file(
            str(ROOT / "data" / "pokemon-tcg-ai-battle" / "EN_Card_Data.csv"),
            "/repo/data/pokemon-tcg-ai-battle/EN_Card_Data.csv",
            copy=True,
        )
        .add_local_file(str(ROOT / "main.py"), "/repo/main.py", copy=True)
        .add_local_file(
            str(
                ROOT
                / ".research"
                / "curriculum"
                / "decks"
                / "top_staryu_windecks_b15853cdfb83.deck.csv"
            ),
            "/repo/.research/curriculum/decks/"
            "top_staryu_windecks_b15853cdfb83.deck.csv",
            copy=True,
        )
        .add_local_file(
            str(
                ROOT
                / ".research"
                / "opponent_decks"
                / "01641f8a-79c0-11f1-bdeb-0242ac130203"
                / "0-Eduardo_Rocha_de_Andrade"
                / "deck.csv"
            ),
            "/repo/.research/opponent_decks/"
            "01641f8a-79c0-11f1-bdeb-0242ac130203/"
            "0-Eduardo_Rocha_de_Andrade/deck.csv",
            copy=True,
        )
        .add_local_file(
            str(
                ROOT
                / ".research"
                / "opponent_decks"
                / "01641f8a-79c0-11f1-bdeb-0242ac130203"
                / "0-Eduardo_Rocha_de_Andrade"
                / "deck_manifest.json"
            ),
            "/repo/.research/opponent_decks/"
            "01641f8a-79c0-11f1-bdeb-0242ac130203/"
            "0-Eduardo_Rocha_de_Andrade/deck_manifest.json",
            copy=True,
        )
        .add_local_dir(
            str(ROOT / ".research" / "submissions" / "20260708-starmie-awr"),
            "/repo/.research/submissions/20260708-starmie-awr",
            copy=True,
        )
        .add_local_file(
            str(
                ROOT
                / ".research"
                / "league_temporal_rl"
                / "r23-next-after-l0-l2-20260710"
                / "r0_train"
                / "selected_ar_bc_r4f_candidate.pt"
            ),
            "/repo/.research/league_temporal_rl/"
            "r23-next-after-l0-l2-20260710/r0_train/"
            "selected_ar_bc_r4f_candidate.pt",
            copy=True,
        )
        .add_local_file(
            str(
                ROOT
                / ".research"
                / "r12"
                / "lane1_r4f_focal_reweight"
                / "selected_ar_bc_r4f_reweighted_candidate.pt"
            ),
            "/repo/.research/r12/lane1_r4f_focal_reweight/"
            "selected_ar_bc_r4f_reweighted_candidate.pt",
            copy=True,
        )
        .add_local_file(
            str(LEDGER_PATH),
            str(REMOTE_LEDGER),
            copy=True,
        )
        .add_local_file(
            str(
                ROOT
                / ".research"
                / "terminal_league_starmie_grimmsnarl_v0"
                / "modal_artifact_map.json"
            ),
            str(REMOTE_ARTIFACT_MAP),
            copy=True,
        )
        .add_local_file(
            str(ROOT / ".research" / "DECK_REGISTRY.json"),
            "/repo/.research/DECK_REGISTRY.json",
            copy=True,
        )
        .add_local_file(
            r"D:\DeepMew_artifacts\checkpoints\r22-staryu-full-league-a4-8c-r7-BEST.pt",
            str(REMOTE_R7_CHECKPOINT),
            copy=True,
        )
    )

    return image


image = _build_image()
app = modal.App(APP_NAME)
TERMINAL_LEAGUE_VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _commit_volume_best_effort(*, context: str) -> bool:
    try:
        TERMINAL_LEAGUE_VOLUME.commit()
    except Exception as exc:
        print(
            f"[volume] commit failed during {context}: {type(exc).__name__}: {exc}",
            flush=True,
        )

        return False

    return True


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _budget_guard_path(lane: LaneSpec) -> Path:
    return Path(str(lane.remote_run_dir / "budget_guard.json"))


def _load_or_create_budget_guard(
    lane: LaneSpec,
    *,
    ledger_sha256: str,
) -> dict[str, Any]:
    path = _budget_guard_path(lane)
    authorized_wall_seconds = lane.cumulative_seconds + RETRIES * CHECKPOINT_STALL_SECONDS
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="ascii"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise TerminalLeagueLaunchError("budget guard is unreadable") from exc
        expected = {
            "schema_version": BUDGET_GUARD_SCHEMA,
            "mission": EXPECTED_MISSION,
            "lane": lane.name,
            "ledger_sha256": ledger_sha256,
            "authorized_wall_seconds": authorized_wall_seconds,
        }
        if not isinstance(payload, dict) or any(
            payload.get(key) != value for key, value in expected.items()
        ):
            raise TerminalLeagueLaunchError(
                "existing lane budget guard is incompatible; refusing to reset its deadline"
            )
        deadline = payload.get("deadline_epoch_seconds")
        if not isinstance(deadline, (int, float)) or deadline <= 0:
            raise TerminalLeagueLaunchError("budget guard deadline is invalid")

        return payload

    created_at = time.time()
    payload = {
        "schema_version": BUDGET_GUARD_SCHEMA,
        "mission": EXPECTED_MISSION,
        "lane": lane.name,
        "ledger_sha256": ledger_sha256,
        "created_at_epoch_seconds": created_at,
        "authorized_wall_seconds": authorized_wall_seconds,
        "deadline_epoch_seconds": created_at + authorized_wall_seconds,
        "guarded_commitment_usd": _money(lane.guarded_commitment_usd),
        "cost_rate_usd_per_hour": _money(USD_PER_HOUR),
    }
    _atomic_write_json(path, payload)

    return payload


def _decimal(value: Any, *, name: str) -> Decimal:
    if isinstance(value, bool):
        raise TerminalLeagueLaunchError(f"{name} must be a non-negative finite number")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise TerminalLeagueLaunchError(
            f"{name} must be a non-negative finite number"
        ) from exc
    if not number.is_finite() or number < 0:
        raise TerminalLeagueLaunchError(f"{name} must be a non-negative finite number")

    return number


def _money(value: Decimal) -> str:
    return format(value.quantize(MONEY_QUANTUM), "f")


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TerminalLeagueLaunchError(f"{name} must be an object")

    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TerminalLeagueLaunchError(f"cost ledger has duplicate key {key!r}")
        result[key] = value

    return result


def _load_ledger(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise TerminalLeagueLaunchError(f"cost ledger is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise TerminalLeagueLaunchError("cost ledger root must be an object")

    return payload


def _assert_equal_decimal(actual: Any, expected: Decimal, *, name: str) -> None:
    value = _decimal(actual, name=name)
    if value != expected:
        raise TerminalLeagueLaunchError(
            f"{name} mismatch: ledger={value} expected={expected}"
        )


def _validate_ledger(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != EXPECTED_LEDGER_SCHEMA:
        raise TerminalLeagueLaunchError("cost ledger schema is not the frozen mission schema")
    if payload.get("mission") != EXPECTED_MISSION:
        raise TerminalLeagueLaunchError("cost ledger belongs to a different mission")
    if payload.get("currency") != "USD":
        raise TerminalLeagueLaunchError("cost ledger currency must be USD")
    resources = _require_mapping(
        payload.get("proven_t4_configuration"),
        name="proven_t4_configuration",
    )
    if resources.get("gpu") != GPU_TYPE:
        raise TerminalLeagueLaunchError("ledger GPU does not match the fixed T4 launcher")
    _assert_equal_decimal(
        resources.get("physical_cpu_cores"),
        Decimal(str(CPU_CORES)),
        name="proven_t4_configuration.physical_cpu_cores",
    )
    _assert_equal_decimal(
        resources.get("memory_gib"),
        Decimal(MEMORY_MIB) / Decimal(1024),
        name="proven_t4_configuration.memory_gib",
    )
    _assert_equal_decimal(
        resources.get("cost_per_hour"),
        USD_PER_HOUR,
        name="proven_t4_configuration.cost_per_hour",
    )
    commitments = _require_mapping(
        payload.get("minimum_training_commitments"),
        name="minimum_training_commitments",
    )
    for lane in LANES.values():
        lane_commitment = _require_mapping(
            commitments.get(lane.name),
            name=f"minimum_training_commitments.{lane.name}",
        )
        _assert_equal_decimal(
            lane_commitment.get("cumulative_gpu_hours"),
            lane.cumulative_hours,
            name=f"minimum_training_commitments.{lane.name}.cumulative_gpu_hours",
        )
        _assert_equal_decimal(
            lane_commitment.get("direct_floor_cost"),
            lane.scheduled_cost_usd,
            name=f"minimum_training_commitments.{lane.name}.direct_floor_cost",
        )
    combined_direct_cost = sum(
        (lane.scheduled_cost_usd for lane in LANES.values()),
        start=Decimal(0),
    )
    combined_retry_reserve = sum(
        (lane.retry_loss_reserve_usd for lane in LANES.values()),
        start=Decimal(0),
    )
    _assert_equal_decimal(
        commitments.get("combined_direct_floor_cost"),
        combined_direct_cost,
        name="minimum_training_commitments.combined_direct_floor_cost",
    )
    _assert_equal_decimal(
        commitments.get("six_twenty_minute_retry_loss_reserve"),
        combined_retry_reserve,
        name="minimum_training_commitments.six_twenty_minute_retry_loss_reserve",
    )
    cap = _decimal(
        payload.get("authorized_total_research_cap"),
        name="authorized_total_research_cap",
    )
    prior = _decimal(
        payload.get("prior_conservative_scheduled_spend"),
        name="prior_conservative_scheduled_spend",
    )
    _assert_equal_decimal(
        payload.get("remaining_authorized_commitment"),
        cap - prior,
        name="remaining_authorized_commitment",
    )
    _assert_equal_decimal(
        commitments.get("remaining_startup_smoke_eval_reserve"),
        cap - prior - combined_direct_cost - combined_retry_reserve,
        name="minimum_training_commitments.remaining_startup_smoke_eval_reserve",
    )
    launches = payload.get("launches")
    if not isinstance(launches, list):
        raise TerminalLeagueLaunchError("cost ledger launches must be a list")


def _recorded_launch_commitment(launch: Any, *, index: int) -> Decimal:
    row = _require_mapping(launch, name=f"launches[{index}]")
    for field in (
        "guarded_commitment_usd",
        "scheduled_cost_usd",
        "worst_case_usd",
    ):
        if field in row:
            return _decimal(row[field], name=f"launches[{index}].{field}")
    lane_name = row.get("lane")
    if lane_name in LANES:
        return LANES[str(lane_name)].guarded_commitment_usd
    raise TerminalLeagueLaunchError(
        f"launches[{index}] needs a scheduled commitment or a known lane"
    )


def _budget_projection(lane: LaneSpec, payload: Mapping[str, Any]) -> dict[str, Decimal]:
    _validate_ledger(payload)
    cap = _decimal(
        payload["authorized_total_research_cap"],
        name="authorized_total_research_cap",
    )
    prior = _decimal(
        payload["prior_conservative_scheduled_spend"],
        name="prior_conservative_scheduled_spend",
    )
    recorded = sum(
        (
            _recorded_launch_commitment(launch, index=index)
            for index, launch in enumerate(payload["launches"])
        ),
        start=Decimal(0),
    )
    total = prior + recorded + lane.guarded_commitment_usd
    remaining = cap - total
    if total > cap:
        raise TerminalLeagueLaunchError(
            "cross-workspace schedule gate rejected "
            f"${_money(total)}; mission cap is ${_money(cap)}"
        )

    return {
        "cap_usd": cap,
        "prior_cross_workspace_usd": prior,
        "recorded_mission_commitment_usd": recorded,
        "scheduled_direct_cost_usd": lane.scheduled_cost_usd,
        "retry_loss_reserve_usd": lane.retry_loss_reserve_usd,
        "guarded_proposed_commitment_usd": lane.guarded_commitment_usd,
        "projected_total_usd": total,
        "remaining_authorized_usd": remaining,
    }


def _ledger_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lane(name: str) -> LaneSpec:
    normalized = name.strip().lower()
    if normalized not in LANES:
        raise TerminalLeagueLaunchError(
            f"--lane must be one of {','.join(sorted(LANES))}"
        )

    return LANES[normalized]


def _runner_command(lane: LaneSpec, *, smoke: bool = False) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "/repo/scripts/run_terminal_league_ppo.py",
        "--lane",
        lane.name,
        "--run-dir",
        str(lane.remote_run_dir),
        "--init-checkpoint",
        str(REMOTE_R7_CHECKPOINT),
        "--artifact-map",
        str(REMOTE_ARTIFACT_MAP),
        "--minimum-cumulative-seconds",
        "0" if smoke else str(lane.cumulative_seconds),
        "--checkpoint-interval-seconds",
        str(CHECKPOINT_INTERVAL_SECONDS),
        "--device",
        DEVICE,
        "--precision",
        PRECISION,
        "--resume",
    ]
    if smoke:
        command.extend(("--max-decisions", "1"))

    return command


def _terminate_process(process: subprocess.Popen[Any]) -> int:
    process.terminate()
    try:
        return process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()

        return process.wait(timeout=30)


def _run_terminal_league(
    lane_name: str,
    expected_ledger_sha256: str,
    smoke: bool = False,
) -> dict[str, Any]:
    lane = _lane(lane_name)
    ledger_path = Path(str(REMOTE_LEDGER))
    actual_ledger_sha256 = _ledger_sha256(ledger_path)
    if actual_ledger_sha256 != expected_ledger_sha256:
        raise TerminalLeagueLaunchError(
            "packaged cost ledger differs from the locally gated ledger"
        )
    payload = _load_ledger(ledger_path)
    _budget_projection(lane, payload)
    TERMINAL_LEAGUE_VOLUME.reload()
    Path(str(lane.remote_run_dir)).mkdir(parents=True, exist_ok=True)
    budget_guard = None
    if not smoke:
        budget_guard = _load_or_create_budget_guard(
            lane,
            ledger_sha256=expected_ledger_sha256,
        )
        TERMINAL_LEAGUE_VOLUME.commit()
        if time.time() >= float(budget_guard["deadline_epoch_seconds"]):
            return {
                "status": "budget_deadline_hard_stop",
                "lane": lane.name,
                "run_dir": str(lane.remote_run_dir),
                "automatic_retry_requested": False,
                "budget_guard": budget_guard,
            }
    command = _runner_command(lane, smoke=smoke)
    checkpoint_ready_path = Path(
        f"{lane.remote_run_dir / 'league_checkpoint.pt'}.latest.json"
    )
    checkpoint_ready_mtime_ns = (
        checkpoint_ready_path.stat().st_mtime_ns
        if checkpoint_ready_path.is_file()
        else None
    )
    process = subprocess.Popen(command, cwd="/repo")
    monitor_started = time.monotonic()
    last_complete_checkpoint = monitor_started
    next_commit = monitor_started + VOLUME_COMMIT_INTERVAL_SECONDS
    periodic_commit_failures = 0
    try:
        while process.poll() is None:
            now = time.monotonic()
            if budget_guard is not None and time.time() >= float(
                budget_guard["deadline_epoch_seconds"]
            ):
                returncode = _terminate_process(process)
                hard_stop_commit_succeeded = _commit_volume_best_effort(
                    context=f"{lane.name} budget deadline"
                )

                return {
                    "status": "budget_deadline_hard_stop",
                    "lane": lane.name,
                    "returncode": returncode,
                    "run_dir": str(lane.remote_run_dir),
                    "automatic_retry_requested": False,
                    "hard_stop_commit_succeeded": hard_stop_commit_succeeded,
                    "periodic_commit_failures": periodic_commit_failures,
                    "budget_guard": budget_guard,
                }
            observed_checkpoint_ready_mtime_ns = (
                checkpoint_ready_path.stat().st_mtime_ns
                if checkpoint_ready_path.is_file()
                else None
            )
            if observed_checkpoint_ready_mtime_ns != checkpoint_ready_mtime_ns:
                if not _commit_volume_best_effort(
                    context=f"{lane.name} complete checkpoint generation"
                ):
                    periodic_commit_failures += 1
                checkpoint_ready_mtime_ns = observed_checkpoint_ready_mtime_ns
                last_complete_checkpoint = now
                next_commit = now + VOLUME_COMMIT_INTERVAL_SECONDS
            elif now - last_complete_checkpoint >= CHECKPOINT_STALL_SECONDS:
                returncode = _terminate_process(process)
                hard_stop_commit_succeeded = _commit_volume_best_effort(
                    context=f"{lane.name} checkpoint stall"
                )

                return {
                    "status": "checkpoint_stall_hard_stop",
                    "lane": lane.name,
                    "returncode": returncode,
                    "run_dir": str(lane.remote_run_dir),
                    "automatic_retry_requested": False,
                    "hard_stop_commit_succeeded": hard_stop_commit_succeeded,
                    "periodic_commit_failures": periodic_commit_failures,
                    "checkpoint_stall_seconds": CHECKPOINT_STALL_SECONDS,
                }
            elif now >= next_commit:
                if not _commit_volume_best_effort(context=f"{lane.name} periodic flush"):
                    periodic_commit_failures += 1
                next_commit = now + VOLUME_COMMIT_INTERVAL_SECONDS
            time.sleep(15)
        returncode = process.wait()
    except BaseException:
        _commit_volume_best_effort(context=f"{lane.name} interrupted parent")
        raise
    if returncode == SAFETY_HARD_STOP_EXIT_CODE:
        hard_stop_commit_succeeded = _commit_volume_best_effort(
            context=f"{lane.name} safety hard stop"
        )
        print(
            f"[safety] {lane.name} returned exit code {SAFETY_HARD_STOP_EXIT_CODE}; "
            "lane is hard-stopped and will not be retried",
            flush=True,
        )

        return {
            "status": "safety_hard_stop",
            "lane": lane.name,
            "returncode": returncode,
            "run_dir": str(lane.remote_run_dir),
            "automatic_retry_requested": False,
            "hard_stop_commit_succeeded": hard_stop_commit_succeeded,
            "periodic_commit_failures": periodic_commit_failures,
        }
    if returncode == TRAINING_HARD_STOP_EXIT_CODE:
        hard_stop_commit_succeeded = _commit_volume_best_effort(
            context=f"{lane.name} training-invariant hard stop"
        )

        return {
            "status": "training_invariant_hard_stop",
            "lane": lane.name,
            "returncode": returncode,
            "run_dir": str(lane.remote_run_dir),
            "automatic_retry_requested": False,
            "hard_stop_commit_succeeded": hard_stop_commit_succeeded,
            "periodic_commit_failures": periodic_commit_failures,
        }
    if returncode != 0:
        _commit_volume_best_effort(context=f"{lane.name} retriable subprocess failure")
        raise subprocess.CalledProcessError(returncode, command)
    TERMINAL_LEAGUE_VOLUME.commit()

    summary_path = Path(str(lane.remote_run_dir / "summary.json"))
    status_path = Path(str(lane.remote_run_dir / "status.json"))
    summary = json.loads(summary_path.read_text(encoding="ascii"))
    status_receipt = json.loads(status_path.read_text(encoding="ascii"))
    if summary.get("safety") != {
        "crashes": 0,
        "illegal_actions": 0,
        "timeouts": 0,
    }:
        raise TerminalLeagueLaunchError("runner summary is not safety-clean")
    progress = summary.get("progress")
    expected_summary_status = (
        "bounded_smoke_complete" if smoke else "minimum_training_floor_reached"
    )
    if summary.get("status") != expected_summary_status:
        raise TerminalLeagueLaunchError(
            f"runner status {summary.get('status')!r} is not {expected_summary_status!r}"
        )
    validation = status_receipt.get("validation")
    if not isinstance(progress, dict) or int(progress.get("update_count") or 0) <= 0:
        raise TerminalLeagueLaunchError("runner did not complete a PPO update")
    if not smoke and float(progress.get("cumulative_training_seconds") or 0.0) < float(
        lane.cumulative_seconds
    ):
        raise TerminalLeagueLaunchError("runner returned before the cumulative floor")
    latest_update = validation.get("latest_update") if isinstance(validation, dict) else None
    if not isinstance(latest_update, dict):
        raise TerminalLeagueLaunchError("runner validation lacks update metrics")
    if float(latest_update.get("actor_parameter_delta_norm") or 0.0) <= 0.0:
        raise TerminalLeagueLaunchError("runner actor did not update")
    if float(latest_update.get("critic_parameter_delta_norm") or 0.0) <= 0.0:
        raise TerminalLeagueLaunchError("runner critic did not update")

    return {
        "status": "smoke_complete" if smoke else "completed_cumulative_floor",
        "lane": lane.name,
        "returncode": returncode,
        "run_dir": str(lane.remote_run_dir),
        "minimum_cumulative_seconds": lane.cumulative_seconds,
        "periodic_commit_failures": periodic_commit_failures,
        "runner_summary": summary,
    }


@app.function(
    image=image,
    gpu=GPU_TYPE,
    cpu=CPU_CORES,
    memory=MEMORY_MIB,
    timeout=STARMIE_FUNCTION_SECONDS,
    retries=RETRIES,
    max_containers=1,
    name="starmie_terminal_league",
    volumes={str(REMOTE_VOLUME_ROOT): TERMINAL_LEAGUE_VOLUME},
)
def run_starmie_terminal_league(
    expected_ledger_sha256: str,
    smoke: bool = False,
) -> dict[str, Any]:
    return _run_terminal_league("starmie", expected_ledger_sha256, smoke)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    cpu=CPU_CORES,
    memory=MEMORY_MIB,
    timeout=GRIMMSNARL_FUNCTION_SECONDS,
    retries=RETRIES,
    max_containers=1,
    name="grimmsnarl_terminal_league",
    volumes={str(REMOTE_VOLUME_ROOT): TERMINAL_LEAGUE_VOLUME},
)
def run_grimmsnarl_terminal_league(
    expected_ledger_sha256: str,
    smoke: bool = False,
) -> dict[str, Any]:
    return _run_terminal_league("grimmsnarl", expected_ledger_sha256, smoke)


def _print_schedule(selected: LaneSpec, projection: Mapping[str, Decimal]) -> None:
    for lane in LANES.values():
        print(
            f"[cost] lane={lane.name} cumulative_seconds={lane.cumulative_seconds} "
            f"hours={lane.cumulative_hours} rate_usd_per_hour=${_money(USD_PER_HOUR)} "
            f"scheduled_direct_usd=${_money(lane.scheduled_cost_usd)}"
        )
    print(
        f"[budget] selected={selected.name} "
        f"retry_reserve_usd=${_money(projection['retry_loss_reserve_usd'])} "
        f"guarded_proposed_usd=${_money(projection['guarded_proposed_commitment_usd'])} "
        f"prior_cross_workspace_usd=${_money(projection['prior_cross_workspace_usd'])} "
        f"recorded_mission_usd=${_money(projection['recorded_mission_commitment_usd'])} "
        f"projected_total_usd=${_money(projection['projected_total_usd'])} "
        f"cap_usd=${_money(projection['cap_usd'])} "
        f"remaining_usd=${_money(projection['remaining_authorized_usd'])}"
    )


@app.local_entrypoint()
def main(lane: str, smoke: bool = False, detached_ack: bool = False) -> None:
    selected = _lane(lane)
    if not smoke and not detached_ack:
        raise TerminalLeagueLaunchError(
            "long runs require both Modal --detach and script --detached-ack"
        )
    ledger = _load_ledger(LEDGER_PATH)
    projection = _budget_projection(selected, ledger)
    _print_schedule(selected, projection)
    if not RUNNER.is_file():
        raise TerminalLeagueLaunchError(f"terminal-league runner is missing: {RUNNER}")
    ledger_sha256 = _ledger_sha256(LEDGER_PATH)
    print(
        f"[modal] volume={VOLUME_NAME} lane={selected.name} "
        f"run_dir={selected.remote_run_dir} retries={RETRIES} "
        f"checkpoint_interval_seconds={CHECKPOINT_INTERVAL_SECONDS} precision={PRECISION} "
        f"checkpoint_stall_seconds={CHECKPOINT_STALL_SECONDS} smoke={smoke}",
        flush=True,
    )
    lane_function = (
        run_starmie_terminal_league
        if selected.name == "starmie"
        else run_grimmsnarl_terminal_league
    )
    function = (
        lane_function.with_options(timeout=900, retries=0)
        if smoke
        else lane_function
    )
    result = function.remote(ledger_sha256, smoke)
    print(json.dumps(result, indent=2, sort_keys=True))
