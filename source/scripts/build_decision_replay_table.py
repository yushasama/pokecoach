"""Build a normalized per-decision replay table from existing trace frames.

Flattens `deepmew.decision_frame.v1` `frames.jsonl` artifacts into versioned
`deepmew.replay_decision_row.v1` rows. No games are replayed and no engine
instrumentation is added.

Examples
--------
Ingest one trace run (a frames.jsonl file or its run directory)::

    python scripts/build_decision_replay_table.py --frames .research/runs/trace/<date>/<run_id>

Ingest every trace run with a frames.jsonl under the source root::

    python scripts/build_decision_replay_table.py --all --source-root .research/runs

Only keep frames with enriched option identity (skip pre-TD.0.10 frames)::

    python scripts/build_decision_replay_table.py --frames <dir> --require-resolved

Skip the CSV mirror::

    python scripts/build_decision_replay_table.py --frames <dir> --no-csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.artifacts.run_dir import create_run_dir  # noqa: E402
from deepmew.replay.decision_ingest import (  # noqa: E402
    FRAMES_FILENAME,
    ReplayDecisionIngestError,
    ingest_frames,
)
from deepmew.replay.decision_table import write_decision_replay_table  # noqa: E402


DEFAULT_SOURCE_ROOT = ".research/runs"
DEFAULT_OUT_ROOT = ".research/runs"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a per-decision replay table from trace frames.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--frames", help="A frames.jsonl file or its run directory.")
    group.add_argument(
        "--all",
        action="store_true",
        help=f"Ingest every run dir with {FRAMES_FILENAME} under --source-root.",
    )
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT, help="Root to scan with --all.")
    parser.add_argument("--kinds", nargs="+", help="Restrict --all discovery to these run-kind directories.")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="Run-directory root for the decision table.")
    parser.add_argument(
        "--require-resolved",
        action="store_true",
        help="Skip pre-TD.0.10 frames that lack option-identity enrichment.",
    )
    parser.add_argument("--no-csv", action="store_true", help="Skip the CSV mirror; write JSONL + manifest only.")
    return parser.parse_args(argv)


def _discover_frames(source_root: str | Path, kinds: list[str] | None) -> list[Path]:
    root = Path(source_root)
    found = sorted(root.glob(f"**/{FRAMES_FILENAME}"))
    if kinds:
        kept = []
        for path in found:
            parts = path.resolve().parts
            if "runs" in parts:
                runs_index = parts.index("runs")
                kind = parts[runs_index + 1] if runs_index + 1 < len(parts) else None
                if kind in kinds:
                    kept.append(path)
        return kept
    return found


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = "python scripts/build_decision_replay_table.py " + " ".join(sys.argv[1:])

    if args.all:
        frame_paths = _discover_frames(args.source_root, args.kinds)
        if not frame_paths:
            print(f"no runs with {FRAMES_FILENAME} found under {args.source_root}", file=sys.stderr)
            return 1
        suffix = f"all-{len(frame_paths)}"
    else:
        frame_paths = [Path(args.frames)]
        suffix = Path(args.frames).name or "frames"

    rows: list[dict[str, object]] = []
    source_run_ids: list[str] = []
    source_paths: list[str] = []
    frame_count = 0
    skipped_unresolved = 0
    pre_enrichment_frames = 0
    for frames_path in frame_paths:
        try:
            ingest = ingest_frames(frames_path, require_resolved=args.require_resolved)
        except ReplayDecisionIngestError as exc:
            print(f"ingest failed for {frames_path}: {exc}", file=sys.stderr)
            return 1
        rows.extend(ingest.rows)
        source_run_ids.append(ingest.source_run_id)
        source_paths.append(ingest.source_path)
        frame_count += ingest.frame_count
        skipped_unresolved += ingest.skipped_unresolved
        pre_enrichment_frames += ingest.pre_enrichment_frames
        print(
            f"ingested {ingest.row_count}/{ingest.frame_count} frames from {ingest.source_run_id} "
            f"(skipped_unresolved={ingest.skipped_unresolved}, pre_enrichment={ingest.pre_enrichment_frames})"
        )

    out_run = create_run_dir(kind="replay_decision_table", out_root=args.out_root, suffix=suffix)
    export = write_decision_replay_table(
        rows,
        out_run.path,
        run_id=out_run.run_id,
        command=command,
        source_run_ids=source_run_ids,
        source_paths=source_paths,
        frame_count=frame_count,
        skipped_unresolved=skipped_unresolved,
        pre_enrichment_frames=pre_enrichment_frames,
        write_csv=not args.no_csv,
    )

    summary = export.summary
    print(
        " ".join(
            [
                f"rows={summary['row_count']}",
                f"frames={summary['frame_count']}",
                f"resolved={summary['resolved_chosen']}",
                f"unresolved={summary['unresolved_chosen']}",
                f"crashes={summary['crashes']}",
                f"timeouts={summary['timeouts']}",
                f"draws={summary['draws']}",
                f"agents={len(summary['coverage']['agents'])}",
                f"opponents={len(summary['coverage']['opponents'])}",
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
