from __future__ import annotations

import argparse
import json
import subprocess
import sys
from json import JSONDecodeError
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_r12_deck_choice_panel.py"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _creation_flags() -> int:
    return 0


def _read_summary(out_dir: Path) -> dict[str, object]:
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        return {
            "rows": 0,
            "expected": "?",
            "crashes": "?",
            "timeouts": "?",
            "complete": False,
        }
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except JSONDecodeError:
        return {
            "rows": "?",
            "expected": "?",
            "crashes": "?",
            "timeouts": "?",
            "complete": False,
        }

    return {
        "rows": payload.get("counts", {}).get("matchup_rows", 0),
        "expected": payload.get("counts", {}).get("expected_matchup_rows", "?"),
        "crashes": payload.get("safety", {}).get("crashes", "?"),
        "timeouts": payload.get("safety", {}).get("timeouts", "?"),
        "complete": payload.get("counts", {}).get("complete", False),
    }


def _wait_for_processes(processes: list[tuple[int, subprocess.Popen[bytes], Path]], poll_seconds: int) -> list[int]:
    exit_codes: dict[int, int] = {}
    while len(exit_codes) < len(processes):
        for shard_index, proc, out_dir in processes:
            code = proc.poll()
            if code is not None and shard_index not in exit_codes:
                exit_codes[shard_index] = int(code)
                (out_dir / "exitcode.txt").write_text(f"{code}\n", encoding="utf-8")
        progress: list[str] = []
        for shard_index, proc, out_dir in processes:
            summary = _read_summary(out_dir)
            status = "done" if shard_index in exit_codes else "running"
            progress.append(
                "shard_{index:02d} {status} rows={rows}/{expected} safety={crashes}/{timeouts}".format(
                    index=shard_index,
                    status=status,
                    rows=summary["rows"],
                    expected=summary["expected"],
                    crashes=summary["crashes"],
                    timeouts=summary["timeouts"],
                )
            )
        print(" | ".join(progress), flush=True)
        if len(exit_codes) < len(processes):
            import time

            time.sleep(max(1, int(poll_seconds)))

    return [exit_codes[index] for index, _, _ in processes]


def launch(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[int]]:
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    launches: list[dict[str, object]] = []
    processes: list[tuple[int, subprocess.Popen[bytes], Path]] = []
    pilot_decks = list(args.pilot_deck or ["reference_abomasnow", "all"])
    opponent_decks = list(args.opponent_deck or [])
    for shard_index in range(int(args.shards)):
        out_dir = out_root / f"shard_{shard_index:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "run.log"
        command = [
            sys.executable,
            str(RUNNER),
            "--deck-zoo",
            str(args.deck_zoo),
            "--out-dir",
            str(out_dir),
            "--games",
            str(int(args.games)),
            "--seed",
            str(int(args.seed)),
            "--opponent-policy",
            str(args.opponent_policy),
            "--pilot-shard-index",
            str(shard_index),
            "--pilot-shard-count",
            str(int(args.shards)),
            "--resume",
        ]
        if args.opponent_deck_zoo:
            command.extend(["--opponent-deck-zoo", str(args.opponent_deck_zoo)])
        for deck_hash in pilot_decks:
            command.extend(["--pilot-deck", str(deck_hash)])
        for deck_hash in opponent_decks:
            command.extend(["--opponent-deck", str(deck_hash)])
        if args.limit_opponents is not None:
            command.extend(["--limit-opponents", str(int(args.limit_opponents))])
        if not args.stop_on_safety:
            command.append("--no-stop-on-safety")
        if args.dry_run:
            payload = {
                "shard_index": shard_index,
                "out_dir": str(out_dir),
                "log_path": str(log_path),
                "command": command,
                "pid": None,
            }
            launches.append(payload)
            continue
        with log_path.open("ab") as log_handle:
            proc = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                creationflags=_creation_flags(),
            )
        payload = {
            "shard_index": shard_index,
            "out_dir": str(out_dir),
            "log_path": str(log_path),
            "command": command,
            "pid": proc.pid,
        }
        (out_dir / "pid.txt").write_text(f"{proc.pid}\n", encoding="utf-8")
        (out_dir / "command.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        launches.append(payload)
        processes.append((shard_index, proc, out_dir))
    manifest = {
        "created_at_utc": _now(),
        "deck_zoo": str(args.deck_zoo),
        "opponent_deck_zoo": str(args.opponent_deck_zoo or args.deck_zoo),
        "out_root": str(out_root),
        "games": int(args.games),
        "seed": int(args.seed),
        "shards": int(args.shards),
        "pilot_decks": pilot_decks,
        "opponent_decks": opponent_decks or ["all"],
        "opponent_policy": str(args.opponent_policy),
        "dry_run": bool(args.dry_run),
        "launches": launches,
    }
    (out_root / "launch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    exit_codes: list[int] = []
    if args.wait and not args.dry_run:
        exit_codes = _wait_for_processes(processes, int(args.poll_seconds))

    return launches, exit_codes


def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch sharded R12 all-deck deck-choice panels.")
    parser.add_argument("--deck-zoo", required=True)
    parser.add_argument("--opponent-deck-zoo")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--pilot-deck", action="append")
    parser.add_argument("--opponent-deck", action="append")
    parser.add_argument("--opponent-policy", default="first-legal")
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--limit-opponents", type=int)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-safety", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    launches, exit_codes = launch(args)
    print(json.dumps({"launches": launches, "exit_codes": exit_codes}, indent=2))
    if exit_codes and any(code != 0 for code in exit_codes):
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
