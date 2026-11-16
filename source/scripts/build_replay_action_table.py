"""Build a raw replay-action table (Bible §2.16 aligned) from existing trace frames.

Flattens `deepmew.decision_frame.v1` `frames.jsonl` artifacts into versioned
`deepmew.replay_action_row.v1` rows that keep raw `state_json` and raw
`legal_options_json`. No games are replayed and no engine instrumentation is
added.

CSV is opt-in (`--csv`): rows embed full raw observations, so a CSV mirror is
large and impractical; JSONL + manifest + report are the primary outputs.

Examples
--------
    python scripts/build_replay_action_table.py --frames .research/runs/trace/<date>/<run_id>
    python scripts/build_replay_action_table.py --all --source-root .research/runs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.artifacts.run_dir import create_run_dir  # noqa: E402
from deepmew.replay.action_ingest import ReplayActionIngestError  # noqa: E402
from deepmew.replay.action_table import build_action_table  # noqa: E402
from deepmew.replay.decision_ingest import FRAMES_FILENAME  # noqa: E402


DEFAULT_SOURCE_ROOT = ".research/runs"
DEFAULT_OUT_ROOT = ".research/runs"
ROW_GATE = 10000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a raw replay-action table from trace frames.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--frames", help="A frames.jsonl file or its run directory.")
    group.add_argument(
        "--all",
        action="store_true",
        help=f"Ingest every run dir with {FRAMES_FILENAME} under --source-root.",
    )
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT, help="Root to scan with --all.")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="Run-directory root for the action table.")
    parser.add_argument("--row-gate", type=int, default=ROW_GATE, help="Minimum row count for the T4.1 gate.")
    parser.add_argument("--csv", action="store_true", help="Also write a (large) CSV mirror; default off.")
    return parser.parse_args(argv)


def _discover_frames(source_root: str | Path) -> list[Path]:
    return sorted(Path(source_root).glob(f"**/{FRAMES_FILENAME}"))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = "python scripts/build_replay_action_table.py " + " ".join(sys.argv[1:])

    if args.all:
        frame_paths = _discover_frames(args.source_root)
        if not frame_paths:
            print(f"no runs with {FRAMES_FILENAME} found under {args.source_root}", file=sys.stderr)
            return 1
        suffix = f"all-{len(frame_paths)}"
    else:
        frame_paths = [Path(args.frames)]
        suffix = Path(args.frames).name or "frames"

    print(f"building action table from {len(frame_paths)} frames source(s)...")
    try:
        out_run = create_run_dir(kind="replay_action_table", out_root=args.out_root, suffix=suffix)
        export = build_action_table(
            list(frame_paths),
            out_run.path,
            run_id=out_run.run_id,
            command=command,
            row_gate=args.row_gate,
            write_csv=args.csv,
        )
    except ReplayActionIngestError as exc:
        print(f"ingest failed: {exc}", file=sys.stderr)
        return 1

    s = export.summary
    print(
        " ".join(
            [
                f"rows={s['row_count']}",
                f"frames={s['frame_count']}",
                f"malformed={s['malformed_count']}",
                f"missing_state={s['missing_state_count']}",
                f"missing_legal={s['missing_legal_options_count']}",
                f"games={s['game_count']}",
                f"runs={len(s['source_run_ids'])}",
                f"gate({s['row_gate']})_met={s['row_gate_met']}",
                f"bible_2_16_strict={s['bible_2_16_strict']}",
            ]
        )
    )
    print(f"out_dir={export.out_dir}")
    print(f"jsonl={export.jsonl_path}")
    if export.csv_path is not None:
        print(f"csv={export.csv_path}")
    print(f"manifest={export.manifest_path}")
    print(f"report={export.report_path}")
    if not s["row_gate_met"]:
        deficit = max(0, s["row_gate"] - s["row_count"])
        print(
            f"NOTE: row gate not met (deficit={deficit}). Generate more trace frames, e.g.: "
            "python scripts\\arena.py matrix --agents v2.4 first-legal random --games 50 --seed 20240801 --trace",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
