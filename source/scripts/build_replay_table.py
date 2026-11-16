"""Build a normalized replay table from V3.7.1 opponent-mixture artifacts.

Examples
--------
Ingest one run directory::

    python scripts/build_replay_table.py \
        --run-dir .research/runs/opponent_mixtures/20260622/<run_id>

Ingest every opponent-mixture run found under the source root::

    python scripts/build_replay_table.py --all

Output is written under ``.research/runs/replay_table/<date>/<run_id>/`` as a
JSONL table, an optional CSV mirror, and a self-describing manifest.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.artifacts.run_dir import create_run_dir  # noqa: E402
from deepmew.replay.ingest import (  # noqa: E402
    RESULTS_FILENAME,
    ReplayIngestError,
    ingest_run_dir,
)
from deepmew.replay.table import write_replay_table  # noqa: E402


DEFAULT_SOURCE_ROOT = ".research/runs/opponent_mixtures"
DEFAULT_OUT_ROOT = ".research/runs"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a replay table from opponent-mixture artifacts.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", help="Single opponent-mixture run directory to ingest.")
    group.add_argument(
        "--all",
        action="store_true",
        help=f"Ingest every run dir with {RESULTS_FILENAME} under --source-root.",
    )
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT, help="Root to scan with --all.")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="Run-directory root for the replay table.")
    parser.add_argument("--no-csv", action="store_true", help="Skip the CSV mirror; write JSONL + manifest only.")
    return parser.parse_args(argv)


def _discover_run_dirs(source_root: str | Path) -> list[Path]:
    root = Path(source_root)
    results = sorted(root.glob(f"**/{RESULTS_FILENAME}"))
    return [path.parent for path in results]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = "python scripts/build_replay_table.py " + " ".join(sys.argv[1:])

    if args.all:
        run_dirs = _discover_run_dirs(args.source_root)
        if not run_dirs:
            print(f"no runs with {RESULTS_FILENAME} found under {args.source_root}", file=sys.stderr)
            return 1
        suffix = f"all-{len(run_dirs)}"
    else:
        run_dirs = [Path(args.run_dir)]
        suffix = Path(args.run_dir).name

    rows: list[dict[str, object]] = []
    source_run_ids: list[str] = []
    for run_dir in run_dirs:
        try:
            ingest = ingest_run_dir(run_dir)
        except ReplayIngestError as exc:
            print(f"ingest failed for {run_dir}: {exc}", file=sys.stderr)
            return 1
        rows.extend(ingest.rows)
        source_run_ids.append(ingest.source_run_id)
        print(f"ingested {ingest.row_count} rows from {ingest.source_run_id}")

    out_run = create_run_dir(kind="replay_table", out_root=args.out_root, suffix=suffix)
    export = write_replay_table(
        rows,
        out_run.path,
        run_id=out_run.run_id,
        command=command,
        source_run_ids=source_run_ids,
        write_csv=not args.no_csv,
    )

    summary = export.summary
    print(
        " ".join(
            [
                f"rows={summary['row_count']}",
                f"completed={summary['completed']}",
                f"crashes={summary['crashes']}",
                f"timeouts={summary['timeouts']}",
                f"mixtures={len(summary['coverage']['mixtures'])}",
                f"policies={len(summary['coverage']['policies'])}",
                f"decks={len(summary['coverage']['deck_hashes'])}",
            ]
        )
    )
    print(f"out_dir={export.out_dir}")
    print(f"jsonl={export.jsonl_path}")
    if export.csv_path is not None:
        print(f"csv={export.csv_path}")
    print(f"manifest={export.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
