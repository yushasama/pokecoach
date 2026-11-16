"""Run the complete frozen H2/X3 local qualification program.

This orchestration is evaluation-only.  It creates exact two-member payoff
rosters, runs the preregistered fresh-seed/both-seat schedules, and runs the
complete shared reference panels.  Every payoff game also emits aggregate
public telemetry so target-edge and tempo/prize-plan evidence come from the
same games rather than a duplicate trace campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
PAYOFF_RUNNER = ROOT / "scripts" / "run_league_payoff_matrix.py"
PANEL_RUNNER = ROOT / "scripts" / "run_league_shared_panel.py"
POOLS = ROOT / ".research" / "runs" / "bucket_pool" / "20260707" / "20260707T141406Z-bucket_pool-1c29c3dd" / "bucket_pools.json"

STARMIE_DECK = ROOT / ".research" / "curriculum" / "decks" / "top_staryu_windecks_b15853cdfb83.deck.csv"
CINDERACE_DECK = ROOT / ".research" / "post_e1_request_a_m0_m1_v1" / "corpus" / "decks" / "6644aa1457640c4c3d0c132d2b5e493ec808f97162a8ed59ec15448732b9f007.deck.csv"
CRUSTLE_DECK = ROOT / ".research" / "post_e1_request_a_m0_m1_v1" / "corpus" / "decks" / "7a41b8db186a8f5056b573182dcd412c4efc97f8aef1c3678e7fdf9624b8b248.deck.csv"

E1 = ROOT / ".research" / "terminal_league_starmie_grimmsnarl_v0" / "e1_starmie_complete_enhanced_generation_20260714" / "production_e1_20260716" / "stage_05_terminal_ppo.pt.actor.pt"
D1 = ROOT / ".research" / "terminal_league_starmie_grimmsnarl_v0" / "control_closeout_starmie_20260713" / "remote" / "final_actor.pt"
R7 = Path(r"D:\DeepMew_artifacts\checkpoints\r22-staryu-full-league-a4-8c-r7-BEST.pt")
X2 = Path(r"D:\DeepMew_artifacts\checkpoints\x1-x2-direct-exploiter-first-wave-x2-final-actor.pt")
R12 = ROOT / ".research" / "r12" / "lane1_r4f_focal_reweight" / "selected_ar_bc_r4f_reweighted_candidate.pt"

H2_REPAIR_SEEDS = (85735189, 2305321, 936238739)
X3_TARGET_SEEDS = (694951281, 1608927555, 141030956)
H2_CONTROL_SEEDS = {"e1": 193126748, "d1": 355689814, "r7": 1307189798}
X3_CONTROL_SEEDS = {
    "e1": 1664026143,
    "r7": 2032395630,
    "x2": 428796647,
    "r12": 1589034843,
    "h2": 615776325,
}
H2_PANEL_SEED = 745968345
X3_PANEL_SEED = 863749190


@dataclass(frozen=True)
class Participant:
    label: str
    checkpoint: Path
    deck: Path
    feature_policy_id: str
    role: str


@dataclass(frozen=True)
class QualificationTask:
    name: str
    kind: str
    command: tuple[str, ...]
    output_dir: Path
    planned_games: int


@dataclass(frozen=True)
class QualificationCursor:
    completed_games: int
    complete: bool
    progress_path: Path | None
    results_path: Path | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")

    return payload


def _task_cursor(task: QualificationTask) -> QualificationCursor:
    progress_path = task.output_dir / "progress.json"
    results_path = task.output_dir / "results.json"
    if not progress_path.is_file():
        if results_path.exists():
            raise ValueError(f"task has results without durable progress: {task.name}")

        return QualificationCursor(0, False, None, None)

    progress = _read_json(progress_path)
    jobs = progress.get("jobs")
    if not isinstance(jobs, dict):
        raise ValueError(f"task progress jobs are not an object: {task.name}")
    if task.kind == "payoff":
        completed_games = len(jobs)
        for job_key, row in jobs.items():
            if not isinstance(row, dict) or row.get("job_key") != job_key:
                raise ValueError(f"payoff durable job identity mismatch: {task.name}:{job_key}")
            if any(bool(row.get(field)) for field in ("crashed", "timed_out", "illegal", "incomplete")):
                raise ValueError(f"payoff durable safety failure blocks resume: {task.name}:{job_key}")
    elif task.kind == "shared_panel":
        completed_games = 0
        seen_game_keys: set[str] = set()
        for job_key, row in jobs.items():
            if not isinstance(row, dict) or row.get("job_key") != job_key:
                raise ValueError(f"panel durable job identity mismatch: {task.name}:{job_key}")
            planned = int(row.get("planned_games", -1))
            completed = int(row.get("completed", -1))
            games = row.get("games")
            if planned < 0 or completed != planned or not isinstance(games, list) or len(games) != completed:
                raise ValueError(f"panel durable job is not atomic and complete: {task.name}:{job_key}")
            if any(int(row.get(field, 0)) for field in ("crashes", "timeouts", "illegal")):
                raise ValueError(f"panel durable safety failure blocks resume: {task.name}:{job_key}")
            for game in games:
                if not isinstance(game, dict) or not isinstance(game.get("game_key"), str):
                    raise ValueError(f"panel durable game identity is invalid: {task.name}:{job_key}")
                if game["game_key"] in seen_game_keys:
                    raise ValueError(f"panel durable duplicate game key: {task.name}:{game['game_key']}")
                seen_game_keys.add(game["game_key"])
            completed_games += completed
    else:
        raise ValueError(f"unknown qualification task kind: {task.kind}")
    if completed_games > task.planned_games:
        raise ValueError(f"durable cursor exceeds frozen task budget: {task.name}")

    complete = progress.get("status") == "completed" and completed_games == task.planned_games
    if complete:
        if not results_path.is_file() or _read_json(results_path).get("status") != "completed":
            raise ValueError(f"complete task lacks a completed results receipt: {task.name}")
    elif results_path.is_file() and _read_json(results_path).get("status") == "completed":
        raise ValueError(f"completed results disagree with durable cursor: {task.name}")

    return QualificationCursor(
        completed_games=completed_games,
        complete=complete,
        progress_path=progress_path,
        results_path=results_path if results_path.is_file() else None,
    )


def _fault_handled_command(command: tuple[str, ...]) -> tuple[str, ...]:
    if not command or Path(command[0]).resolve() != Path(PYTHON).resolve():
        raise ValueError("qualification child command does not use the frozen Python executable")

    return (command[0], "-X", "faulthandler", *command[1:])


def _next_execution_manifest_path(out_root: Path, *, resume: bool) -> Path:
    if not resume:
        return out_root / "QUALIFICATION_EXECUTION_MANIFEST.json"
    base = out_root / "QUALIFICATION_RESUME_EXECUTION_MANIFEST.json"
    if not base.exists():
        return base
    index = 2
    while True:
        candidate = out_root / f"QUALIFICATION_RESUME_EXECUTION_MANIFEST_{index}.json"
        if not candidate.exists():
            return candidate
        index += 1


def _member_id(participant: Participant) -> str:
    identity = f"{participant.label}|{_sha256(participant.checkpoint)}|{_sha256(participant.deck)}"

    return "lmhx-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]


def _write_roster(path: Path, left: Participant, right: Participant) -> None:
    members = []
    for participant in (left, right):
        members.append({
            "league_member_id": _member_id(participant),
            "label": participant.label,
            "spec": f"checkpoint:{participant.checkpoint}",
            "deck_binding": f"deck:{participant.deck}",
            "feature_policy_id": participant.feature_policy_id,
            "role": participant.role,
        })
    payload = {
        "schema_version": "deepmew.league_payoff_roster.v1",
        "members": members,
    }
    if path.is_file():
        if _read_json(path) != payload:
            raise ValueError(f"frozen qualification roster mismatch; refusing overwrite: {path}")

        return
    _atomic_json(path, payload)


def _payoff_task(
    *,
    out_root: Path,
    name: str,
    left: Participant,
    right: Participant,
    seed: int,
    games_per_ordered_pair: int,
    resume: bool,
) -> QualificationTask:
    roster = out_root / "rosters" / f"{name}.json"
    output = out_root / "payoff" / name
    _write_roster(roster, left, right)
    command = (
        PYTHON,
        str(PAYOFF_RUNNER),
        "--roster", str(roster),
        "--games", str(games_per_ordered_pair),
        "--seed", str(seed),
        "--capture-public-telemetry",
        "--out-dir", str(output),
        "--run-id", name,
        "--resume" if resume else "--no-resume",
    )

    return QualificationTask(name, "payoff", command, output, games_per_ordered_pair * 2)


def _panel_task(
    *,
    out_root: Path,
    name: str,
    champion: Participant,
    candidate: Participant,
    games_per_cell: int,
    seed: int,
    resume: bool,
) -> QualificationTask:
    output = out_root / "shared_panel" / name
    command = [
        PYTHON,
        str(PANEL_RUNNER),
        "--pools", str(POOLS),
        "--champion", f"{champion.label}=checkpoint:{champion.checkpoint}",
        "--candidate", f"{candidate.label}=checkpoint:{candidate.checkpoint}",
        "--deck-binding", f"deck:{candidate.deck}",
        "--stage", "600-800",
        "--top-n", "8",
        "--carry-n", "4",
        "--pilots", "greedy", "v2.4", "exploiter",
        "--exploiter-top", "4",
        "--exploiter-checkpoint", str(R12),
        "--games", str(games_per_cell),
        "--seeds", str(seed),
        "--out-dir", str(output),
        "--run-id", name,
        "--resume" if resume else "--no-resume",
    ]
    if champion.feature_policy_id == candidate.feature_policy_id:
        command.extend(("--feature-policy-id", champion.feature_policy_id))
    else:
        command.extend((
            "--participant-feature-policy-id", f"{champion.label}={champion.feature_policy_id}",
            "--participant-feature-policy-id", f"{candidate.label}={candidate.feature_policy_id}",
        ))

    return QualificationTask(name, "shared_panel", tuple(command), output, 20 * 2 * games_per_cell)


def build_tasks(
    *,
    h2_checkpoint: Path,
    x3_checkpoint: Path,
    out_root: Path,
    resume: bool,
) -> list[QualificationTask]:
    h2 = Participant("H2", h2_checkpoint, STARMIE_DECK, "starmie", "candidate")
    x3 = Participant("X3", x3_checkpoint, CINDERACE_DECK, "starmie", "candidate")
    e1 = Participant("E1", E1, STARMIE_DECK, "starmie", "champion")
    d1 = Participant("D1", D1, STARMIE_DECK, "starmie", "champion")
    r7 = Participant("r7", R7, STARMIE_DECK, "starmie", "historical")
    x2 = Participant("X2", X2, CRUSTLE_DECK, "bucket_exploiter", "historical")
    r12 = Participant("r12", R12, CINDERACE_DECK, "bucket_exploiter", "champion")
    tasks: list[QualificationTask] = []

    for index, seed in enumerate(H2_REPAIR_SEEDS, start=1):
        tasks.append(_payoff_task(
            out_root=out_root,
            name=f"h2_x2_vs_e1_seed_{index}",
            left=x2,
            right=e1,
            seed=seed,
            games_per_ordered_pair=256,
            resume=resume,
        ))
        tasks.append(_payoff_task(
            out_root=out_root,
            name=f"h2_x2_vs_h2_seed_{index}",
            left=x2,
            right=h2,
            seed=seed,
            games_per_ordered_pair=256,
            resume=resume,
        ))
    for label, opponent in (("e1", e1), ("d1", d1), ("r7", r7)):
        tasks.append(_payoff_task(
            out_root=out_root,
            name=f"h2_control_{label}",
            left=h2,
            right=opponent,
            seed=H2_CONTROL_SEEDS[label],
            games_per_ordered_pair=128,
            resume=resume,
        ))
    for label, opponent in (("d1", d1), ("r7", r7)):
        tasks.append(_payoff_task(
            out_root=out_root,
            name=f"h2_baseline_e1_vs_{label}",
            left=e1,
            right=opponent,
            seed=H2_CONTROL_SEEDS[label],
            games_per_ordered_pair=128,
            resume=resume,
        ))
    tasks.append(_panel_task(
        out_root=out_root,
        name="h2_complete_reference_panel",
        champion=e1,
        candidate=h2,
        games_per_cell=128,
        seed=H2_PANEL_SEED,
        resume=resume,
    ))

    for index, seed in enumerate(X3_TARGET_SEEDS, start=1):
        tasks.append(_payoff_task(
            out_root=out_root,
            name=f"x3_vs_d1_seed_{index}",
            left=x3,
            right=d1,
            seed=seed,
            games_per_ordered_pair=256,
            resume=resume,
        ))
        tasks.append(_payoff_task(
            out_root=out_root,
            name=f"r12_vs_d1_seed_{index}",
            left=r12,
            right=d1,
            seed=seed,
            games_per_ordered_pair=256,
            resume=resume,
        ))
    for label, opponent in (("e1", e1), ("r7", r7), ("x2", x2), ("r12", r12), ("h2", h2)):
        tasks.append(_payoff_task(
            out_root=out_root,
            name=f"x3_control_{label}",
            left=x3,
            right=opponent,
            seed=X3_CONTROL_SEEDS[label],
            games_per_ordered_pair=128,
            resume=resume,
        ))
    tasks.append(_panel_task(
        out_root=out_root,
        name="x3_complete_reference_panel",
        champion=r12,
        candidate=x3,
        games_per_cell=64,
        seed=X3_PANEL_SEED,
        resume=resume,
    ))

    return tasks


def _run_task(
    task: QualificationTask,
    log_root: Path,
    process_receipt_root: Path,
    stop_event: threading.Event,
) -> dict[str, Any]:
    if stop_event.is_set():
        return {"name": task.name, "status": "cancelled_before_start"}
    log_root.mkdir(parents=True, exist_ok=True)
    process_receipt_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{task.name}.log"
    process_receipt_path = process_receipt_root / f"{task.name}.json"
    started = _utc_now()
    command = _fault_handled_command(task.command)
    with log_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({
            "event": "qualification_child_starting",
            "recorded_at_utc": started,
            "task": task.name,
            "command": list(command),
        }, sort_keys=True) + "\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        process_receipt = {
            "schema_version": "deepmew.h2_x3_qualification_child_process.v1",
            "task": task.name,
            "kind": task.kind,
            "pid": process.pid,
            "started_at_utc": started,
            "command": list(command),
            "log": str(log_path),
            "output_dir": str(task.output_dir),
            "status": "RUNNING",
        }
        _atomic_json(process_receipt_path, process_receipt)
        stream.write(json.dumps({
            "event": "qualification_child_started",
            "recorded_at_utc": _utc_now(),
            "task": task.name,
            "pid": process.pid,
        }, sort_keys=True) + "\n")
        stream.flush()
        while process.poll() is None:
            if stop_event.wait(1.0):
                process.terminate()
                try:
                    process.wait(timeout=15.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                process_receipt.update({
                    "status": "TERMINATED_UNDER_JOINT_HARD_STOP",
                    "ended_at_utc": _utc_now(),
                    "return_code": process.returncode,
                })
                _atomic_json(process_receipt_path, process_receipt)
                return {
                    "name": task.name,
                    "status": "terminated_under_joint_hard_stop",
                    "return_code": process.returncode,
                    "log": str(log_path),
                }
        return_code = process.returncode
        process_receipt.update({
            "status": "PASS" if return_code == 0 else "FAILED",
            "ended_at_utc": _utc_now(),
            "return_code": return_code,
        })
        _atomic_json(process_receipt_path, process_receipt)
    if return_code != 0:
        stop_event.set()
        raise RuntimeError(f"qualification task {task.name} failed with exit code {return_code}; log={log_path}")

    return {
        "name": task.name,
        "kind": task.kind,
        "status": "PASS",
        "started_at_utc": started,
        "ended_at_utc": _utc_now(),
        "planned_games": task.planned_games,
        "output_dir": str(task.output_dir),
        "log": str(log_path),
    }


def run_program(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root).resolve()
    h2_checkpoint = Path(args.h2_checkpoint).resolve()
    x3_checkpoint = Path(args.x3_checkpoint).resolve()
    required = [h2_checkpoint, x3_checkpoint, E1, D1, R7, X2, R12, STARMIE_DECK, CINDERACE_DECK, CRUSTLE_DECK, POOLS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"qualification inputs are missing: {missing}")
    if out_root.exists() and not args.resume:
        raise FileExistsError(f"qualification output exists; use --resume: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(
        h2_checkpoint=h2_checkpoint,
        x3_checkpoint=x3_checkpoint,
        out_root=out_root,
        resume=args.resume,
    )
    cursors = {task.name: _task_cursor(task) for task in tasks}
    if args.resume and (out_root / "H2_X3_PARTIAL_QUALIFICATION_HARD_STOP_RECEIPT.json").is_file():
        if args.max_workers != 1:
            raise ValueError("access-violation resume requires --max-workers 1 for serialized fresh processes")
    manifest_path = _next_execution_manifest_path(out_root, resume=args.resume)
    preexisting_results = [
        {
            "name": task.name,
            "kind": task.kind,
            "status": "SKIPPED_COMPLETE",
            "planned_games": task.planned_games,
            "completed_games_before_resume": cursors[task.name].completed_games,
            "output_dir": str(task.output_dir),
            "progress": str(cursors[task.name].progress_path),
            "results": str(cursors[task.name].results_path),
            "child_process_launched": False,
        }
        for task in tasks
        if cursors[task.name].complete
    ]
    incomplete_tasks = [task for task in tasks if not cursors[task.name].complete]
    manifest = {
        "schema_version": "deepmew.h2_x3_frozen_qualification_execution.v2",
        "status": "RUNNING",
        "created_at_utc": _utc_now(),
        "manifest_path": str(manifest_path),
        "preserved_original_hard_stop_manifest": str(out_root / "QUALIFICATION_EXECUTION_MANIFEST.json") if args.resume else None,
        "maximum_local_evaluator_processes": args.max_workers,
        "fresh_process_per_incomplete_task": True,
        "python_fault_handler_enabled": True,
        "paid_compute": False,
        "capture_public_telemetry": True,
        "checkpoint_receipts": {
            "h2": {"path": str(h2_checkpoint), "sha256": _sha256(h2_checkpoint)},
            "x3": {"path": str(x3_checkpoint), "sha256": _sha256(x3_checkpoint)},
        },
        "tasks": [
            {
                **asdict(task),
                "command": list(task.command),
                "output_dir": str(task.output_dir),
            }
            for task in tasks
        ],
        "planned_games": sum(task.planned_games for task in tasks),
        "completed_games_before_resume": sum(cursor.completed_games for cursor in cursors.values()),
        "skipped_complete_task_count": len(preexisting_results),
        "incomplete_task_count": len(incomplete_tasks),
        "results": preexisting_results,
    }
    _atomic_json(manifest_path, manifest)
    stop_event = threading.Event()
    futures: dict[Future[dict[str, Any]], QualificationTask] = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for task in incomplete_tasks:
            futures[executor.submit(
                _run_task,
                task,
                out_root / "logs",
                out_root / "process_receipts",
                stop_event,
            )] = task
        done, pending = wait(futures, return_when=FIRST_EXCEPTION)
        failure = next((future.exception() for future in done if future.exception() is not None), None)
        if failure is not None:
            stop_event.set()
            wait(pending)
            manifest["status"] = "HARD_STOPPED"
            manifest["error"] = f"{type(failure).__name__}: {failure}"
            manifest["results"] = preexisting_results + [
                future.result()
                for future in futures
                if future.done() and future.exception() is None
            ]
            _atomic_json(manifest_path, manifest)
            raise failure
        wait(pending)
    manifest["status"] = "PASS"
    manifest["completed_at_utc"] = _utc_now()
    manifest["results"] = preexisting_results + [future.result() for future in futures]
    manifest["completed_games_after_resume"] = sum(task.planned_games for task in tasks)
    _atomic_json(manifest_path, manifest)

    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h2-checkpoint", required=True)
    parser.add_argument("--x3-checkpoint", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--max-workers", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_program(args)
    print(json.dumps({
        "status": result["status"],
        "planned_games": result["planned_games"],
        "task_count": len(result["tasks"]),
        "out_root": str(Path(args.out_root).resolve()),
    }, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
