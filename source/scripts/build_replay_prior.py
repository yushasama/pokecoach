"""Build an offline replay-prior artifact from decision replay tables.

Reads `deepmew.replay_decision_row.v1` rows and emits a versioned
`deepmew.replay_prior.v1` count-table prior (smoothed win advantage over the
select-context baseline). Offline only: no gameplay code, no online behavior, no
value model. The artifact is never loaded by the eval/runtime submission path.

Examples
--------
    python scripts/build_replay_prior.py --tables .research/runs/replay_decision_table/<date>/<run_id>
    python scripts/build_replay_prior.py --all --source-root .research/runs/replay_decision_table
    python scripts/build_replay_prior.py --tables <dir> --agent v2.4
    python scripts/build_replay_prior.py --tables <dir> --min-count 30 --alpha 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.artifacts.run_dir import create_run_dir  # noqa: E402
from deepmew.replay.prior_build import (  # noqa: E402
    DECISION_TABLE_FILENAME,
    ReplayPriorBuildError,
    build_replay_prior,
    read_decision_rows,
)
from deepmew.replay.prior_table import write_replay_prior  # noqa: E402


DEFAULT_SOURCE_ROOT = ".research/runs/replay_decision_table"
DEFAULT_OUT_ROOT = ".research/runs"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an offline replay-prior artifact.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tables", help="A replay_decision_table.jsonl file or its run directory.")
    group.add_argument(
        "--all",
        action="store_true",
        help=f"Ingest every run dir with {DECISION_TABLE_FILENAME} under --source-root.",
    )
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT, help="Root to scan with --all.")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="Run-directory root for the prior.")
    parser.add_argument("--agent", help="Keep only rows whose agent_label equals this value.")
    parser.add_argument("--min-count", type=int, default=30, help="Minimum cell count to keep an entry.")
    parser.add_argument("--alpha", type=float, default=50.0, help="Smoothing strength toward the context baseline.")
    return parser.parse_args(argv)


def _discover_tables(source_root: str | Path) -> list[Path]:
    return sorted(Path(source_root).glob(f"**/{DECISION_TABLE_FILENAME}"))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = "python scripts/build_replay_prior.py " + " ".join(sys.argv[1:])

    if args.all:
        table_paths: list[Path] = _discover_tables(args.source_root)
        if not table_paths:
            print(f"no {DECISION_TABLE_FILENAME} found under {args.source_root}", file=sys.stderr)
            return 1
        suffix = f"all-{len(table_paths)}"
    else:
        table_paths = [Path(args.tables)]
        suffix = Path(args.tables).name or "prior"

    try:
        rows, resolved_paths = read_decision_rows(list(table_paths))
        result = build_replay_prior(
            rows,
            min_count=args.min_count,
            alpha=args.alpha,
            agent_filter=args.agent,
            source_table_paths=resolved_paths,
        )
    except ReplayPriorBuildError as exc:
        print(f"build failed: {exc}", file=sys.stderr)
        return 1

    out_run = create_run_dir(kind="replay_prior", out_root=args.out_root, suffix=suffix)
    export = write_replay_prior(result, out_run.path, run_id=out_run.run_id, command=command)

    s = export.summary
    print(
        " ".join(
            [
                f"rows_total={s['rows_total']}",
                f"rows_used={s['rows_used']}",
                f"rows_filtered_out={s['rows_filtered_out']}",
                f"entries={s['entry_count']}",
                f"omitted_low_count={s['omitted_low_count']}",
                f"global_win_rate={s['global_win_rate']:.3f}",
            ]
        )
    )
    warn_on = [name for name, value in s["warnings"].items() if value]
    print(f"warnings={warn_on if warn_on else 'none'}")
    print(f"out_dir={export.out_dir}")
    print(f"prior={export.prior_path}")
    print(f"manifest={export.manifest_path}")
    print(f"report={export.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
