"""Run the frozen 14,336-game D1-successor local qualification panel.

The panel is evaluation-only.  Every bounded task runs in a fresh Python
process and tasks are serialized because the prior H2/X3 qualification hit an
unattributed native access violation.  Durable task cursors make an explicitly
authorized resume exact; this launcher never retries a failed task itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scripts import run_h2_x3_frozen_qualification as harness


ROOT = Path(__file__).resolve().parents[1]
MISSION = ROOT / ".research" / "d1_successor_mixed_league_v1"
PANEL_MANIFEST = MISSION / "QUALIFICATION_PANEL_MANIFEST.json"

STARMIE_DECK = ROOT / ".research" / "curriculum" / "decks" / "top_staryu_windecks_b15853cdfb83.deck.csv"
CINDERACE_DECK = ROOT / ".research" / "post_e1_request_a_m0_m1_v1" / "corpus" / "decks" / "6644aa1457640c4c3d0c132d2b5e493ec808f97162a8ed59ec15448732b9f007.deck.csv"
CRUSTLE_DECK = ROOT / ".research" / "post_e1_request_a_m0_m1_v1" / "corpus" / "decks" / "7a41b8db186a8f5056b573182dcd412c4efc97f8aef1c3678e7fdf9624b8b248.deck.csv"

D1 = ROOT / ".research" / "terminal_league_starmie_grimmsnarl_v0" / "control_closeout_starmie_20260713" / "remote" / "final_actor.pt"
H2 = Path(r"D:\DeepMew_artifacts\checkpoints\h2-x3-first-wave-h2-final-actor.pt")
E1 = ROOT / ".research" / "terminal_league_starmie_grimmsnarl_v0" / "e1_starmie_complete_enhanced_generation_20260714" / "production_e1_20260716" / "stage_05_terminal_ppo.pt.actor.pt"
R7 = Path(r"D:\DeepMew_artifacts\checkpoints\r22-staryu-full-league-a4-8c-r7-BEST.pt")
X2 = Path(r"D:\DeepMew_artifacts\checkpoints\x1-x2-direct-exploiter-first-wave-x2-final-actor.pt")
R12 = ROOT / ".research" / "r12" / "lane1_r4f_focal_reweight" / "selected_ar_bc_r4f_reweighted_candidate.pt"

PRIMARY_SEEDS = (211402771, 211402772, 211402773)
CONTROL_SEEDS = {
    "h2": (211402781, 211402782, 211402783),
    "e1": (211402791, 211402792, 211402793),
    "r7": (211402801, 211402802, 211402803),
    "x2": (211402811, 211402812, 211402813),
    "r12_hardtail": (211402821, 211402822, 211402823),
}
REFERENCE_SEED = 211402831
FROZEN_TOTAL_GAMES = 14_336


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _payoff_task(
    *,
    out_root: Path,
    name: str,
    left: harness.Participant,
    right: harness.Participant,
    seed: int,
    games_per_ordered_pair: int,
    resume: bool,
) -> harness.QualificationTask:
    roster_namespace = "rosters_resume_v2" if resume else "rosters"
    roster = out_root / roster_namespace / f"{name}.json"
    output = out_root / "payoff" / name
    harness._write_roster(roster, left, right)
    command = (
        harness.PYTHON,
        str(harness.PAYOFF_RUNNER),
        "--roster", str(roster),
        "--games", str(games_per_ordered_pair),
        "--seed", str(seed),
        "--capture-public-telemetry",
        "--out-dir", str(output),
        "--run-id", name,
        "--resume" if resume else "--no-resume",
    )

    return harness.QualificationTask(name, "payoff", command, output, games_per_ordered_pair * 2)


def build_tasks(
    *,
    successor_checkpoint: Path,
    out_root: Path,
    resume: bool,
) -> list[harness.QualificationTask]:
    successor = harness.Participant("SUCCESSOR", successor_checkpoint, STARMIE_DECK, "starmie", "candidate")
    d1 = harness.Participant("D1", D1, STARMIE_DECK, "starmie", "champion")
    opponents = {
        "h2": harness.Participant("H2", H2, STARMIE_DECK, "starmie", "candidate"),
        "e1": harness.Participant("E1", E1, STARMIE_DECK, "starmie", "historical"),
        "r7": harness.Participant("r7", R7, STARMIE_DECK, "starmie", "historical"),
        "x2": harness.Participant("X2", X2, CRUSTLE_DECK, "bucket_exploiter", "historical"),
        "r12_hardtail": harness.Participant("R12_CINDERACE_HARDTAIL", R12, CINDERACE_DECK, "bucket_exploiter", "champion"),
    }
    tasks: list[harness.QualificationTask] = []

    for batch, seed in enumerate(PRIMARY_SEEDS, start=1):
        tasks.append(_payoff_task(
            out_root=out_root,
            name=f"successor_vs_d1_batch_{batch}",
            left=successor,
            right=d1,
            seed=seed,
            games_per_ordered_pair=256,
            resume=resume,
        ))
    for opponent_id, seeds in CONTROL_SEEDS.items():
        opponent = opponents[opponent_id]
        for batch, seed in enumerate(seeds, start=1):
            tasks.append(_payoff_task(
                out_root=out_root,
                name=f"successor_vs_{opponent_id}_batch_{batch}",
                left=successor,
                right=opponent,
                seed=seed,
                games_per_ordered_pair=128,
                resume=resume,
            ))
            tasks.append(_payoff_task(
                out_root=out_root,
                name=f"d1_vs_{opponent_id}_batch_{batch}",
                left=d1,
                right=opponent,
                seed=seed,
                games_per_ordered_pair=128,
                resume=resume,
            ))
    tasks.append(harness._panel_task(
        out_root=out_root,
        name="complete_reference_shared_panel",
        champion=d1,
        candidate=successor,
        games_per_cell=128,
        seed=REFERENCE_SEED,
        resume=resume,
    ))
    planned_games = sum(task.planned_games for task in tasks)
    if planned_games != FROZEN_TOTAL_GAMES:
        raise ValueError(f"qualification schedule changed: {planned_games} != {FROZEN_TOTAL_GAMES}")

    return tasks


def _manifest_task(task: harness.QualificationTask) -> dict[str, Any]:
    payload = asdict(task)
    payload["command"] = list(task.command)
    payload["output_dir"] = str(task.output_dir)

    return payload


def run_program(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root).resolve()
    successor_checkpoint = Path(args.successor_checkpoint).resolve()
    required = [
        successor_checkpoint,
        PANEL_MANIFEST,
        harness.POOLS,
        STARMIE_DECK,
        CINDERACE_DECK,
        CRUSTLE_DECK,
        D1,
        H2,
        E1,
        R7,
        X2,
        R12,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"qualification inputs are missing: {missing}")
    if out_root.exists() and not args.resume:
        raise FileExistsError(f"qualification output exists; use --resume only with explicit authorization: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(successor_checkpoint=successor_checkpoint, out_root=out_root, resume=args.resume)
    cursors = {task.name: harness._task_cursor(task) for task in tasks}
    completed = [task for task in tasks if cursors[task.name].complete]
    incomplete = [task for task in tasks if not cursors[task.name].complete]
    manifest_path = harness._next_execution_manifest_path(out_root, resume=args.resume)
    manifest: dict[str, Any] = {
        "schema_version": "deepmew.d1_successor_frozen_qualification_execution.v1",
        "status": "RUNNING",
        "created_at_utc": harness._utc_now(),
        "authorization": "AUTHORIZE_D1_SUCCESSOR_MIXED_LEAGUE_V1_SINGLE_L4_TRAINING_TO_EXACT_11900_USEFUL_SECOND_FLOOR_AND_FULL_14336_GAME_LOCAL_QUALIFICATION",
        "frozen_panel_manifest": {"path": str(PANEL_MANIFEST), "sha256": _sha256(PANEL_MANIFEST)},
        "successor_checkpoint": {"path": str(successor_checkpoint), "sha256": _sha256(successor_checkpoint)},
        "planned_games": FROZEN_TOTAL_GAMES,
        "task_count": len(tasks),
        "maximum_local_evaluator_processes": 1,
        "fresh_process_per_bounded_task": True,
        "serialized_reason": "prior local qualification native access violation 0xC0000005 was not positively attributable; single-process containment previously completed 16,384 games safely",
        "automatic_retry_after_failure": False,
        "paid_compute": False,
        "checkpoint_selection_uses_qualification_seeds": False,
        "completed_games_before_resume": sum(cursor.completed_games for cursor in cursors.values()),
        "skipped_complete_task_count": len(completed),
        "incomplete_task_count": len(incomplete),
        "tasks": [_manifest_task(task) for task in tasks],
        "results": [
            {
                "name": task.name,
                "status": "SKIPPED_COMPLETE",
                "planned_games": task.planned_games,
                "completed_games_before_resume": cursors[task.name].completed_games,
                "child_process_launched": False,
            }
            for task in completed
        ],
    }
    harness._atomic_json(manifest_path, manifest)

    stop_event = harness.threading.Event()
    try:
        for task in incomplete:
            result = harness._run_task(
                task,
                out_root / "logs",
                out_root / "process_receipts",
                stop_event,
            )
            manifest["results"].append(result)
            manifest["completed_games"] = sum(
                harness._task_cursor(candidate).completed_games
                for candidate in tasks
            )
            harness._atomic_json(manifest_path, manifest)
    except BaseException as error:
        manifest["status"] = "HARD_STOPPED_NO_AUTOMATIC_RERUN"
        manifest["failed_at_utc"] = harness._utc_now()
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["completed_games"] = sum(
            harness._task_cursor(candidate).completed_games
            for candidate in tasks
        )
        harness._atomic_json(manifest_path, manifest)
        raise

    final_cursors = {task.name: harness._task_cursor(task) for task in tasks}
    final_games = sum(cursor.completed_games for cursor in final_cursors.values())
    if final_games != FROZEN_TOTAL_GAMES or not all(cursor.complete for cursor in final_cursors.values()):
        raise RuntimeError(f"qualification did not reach exact frozen completion: {final_games}/{FROZEN_TOTAL_GAMES}")
    manifest["status"] = "PASS"
    manifest["completed_at_utc"] = harness._utc_now()
    manifest["completed_games"] = final_games
    harness._atomic_json(manifest_path, manifest)

    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--successor-checkpoint", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_program(args)
    print(json.dumps({
        "status": result["status"],
        "planned_games": result["planned_games"],
        "completed_games": result["completed_games"],
        "task_count": result["task_count"],
    }, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
