from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.arena import (  # noqa: E402
    BASE_AGENTS,
    format_matrix_report,
    format_pair_report,
    run_matrix,
    run_pair,
    write_summary,
)
from deepmew.artifacts.manifest import build_manifest, write_manifest  # noqa: E402
from deepmew.artifacts.run_dir import create_run_dir  # noqa: E402
from deepmew.artifacts.writers import write_markdown  # noqa: E402
from deepmew.trace.record import TraceRecorder  # noqa: E402
from deepmew.trace.report import build_trace_report  # noqa: E402


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--games", type=int, default=20, help="Games per pair, with seats swapped automatically.")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out-dir", default=".research/arena_runs")
    parser.add_argument("--no-write", action="store_true", help="Print only; do not write JSON/Markdown artifacts.")
    parser.add_argument("--trace", action="store_true", help="Emit deterministic decision-frame traces.")
    parser.add_argument("--trace-kind", default="trace", help="Trace run kind under .research/runs.")
    parser.add_argument("--trace-out-root", default=".research/runs", help="Trace generated-run root.")
    parser.add_argument("--trace-sample-limit", type=int, help="Maximum decision frames to record.")


def main_cli() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "DeepMew local arena. Agent specs support ablations, e.g. "
            "'v2.4[first=yes]' or 'v2.4[attach_bonus=20,attach_target=active]'."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pair_parser = subparsers.add_parser("pair", help="Run one agent matchup.")
    pair_parser.add_argument("--agent-a", required=True, choices=None)
    pair_parser.add_argument("--agent-b", required=True, choices=None)
    _add_common_args(pair_parser)

    matrix_parser = subparsers.add_parser("matrix", help="Run a matchup matrix.")
    matrix_parser.add_argument("--agents", nargs="+", default=list(BASE_AGENTS))
    matrix_parser.add_argument("--include-self", action="store_true")
    _add_common_args(matrix_parser)

    args = parser.parse_args()
    trace_recorder = None
    trace_run_dir = None
    if args.trace:
        trace_run_dir = create_run_dir(kind=args.trace_kind, out_root=Path(args.trace_out_root))
        trace_recorder = TraceRecorder(run_id=trace_run_dir.run_id, sample_limit=args.trace_sample_limit)

    if args.command == "pair":
        summary = run_pair(
            args.agent_a,
            args.agent_b,
            games=args.games,
            max_steps=args.max_steps,
            seed=args.seed,
            trace_recorder=trace_recorder,
        )
        report = format_pair_report(summary)
        trace_agents = [summary.agent_a, summary.agent_b]
    else:
        summary = run_matrix(
            args.agents,
            games_per_pair=args.games,
            max_steps=args.max_steps,
            seed=args.seed,
            include_self=args.include_self,
            trace_recorder=trace_recorder,
        )
        report = format_matrix_report(summary)
        trace_agents = list(summary.agents)

    print(report)
    if not args.no_write:
        json_path, report_path = write_summary(summary, args.out_dir)
        print("")
        print(f"wrote_json={json_path}")
        print(f"wrote_report={report_path}")

    if trace_recorder is not None and trace_run_dir is not None:
        outputs = trace_recorder.write(
            frames_path=trace_run_dir.output_path("frames.jsonl"),
            summary_path=trace_run_dir.output_path("summary.json"),
            report_path=trace_run_dir.output_path("report.md"),
        )
        frames_payload = [frame.to_jsonable() for frame in trace_recorder.frames]
        write_markdown(
            trace_run_dir.output_path("report.md"),
            build_trace_report(run_id=trace_run_dir.run_id, frames=frames_payload, agents=trace_agents),
        )
        manifest = build_manifest(
            run_id=trace_run_dir.run_id,
            kind=trace_run_dir.kind,
            command="python scripts\\arena.py " + " ".join(sys.argv[1:]),
            inputs={
                "command": args.command,
                "games": args.games,
                "seed": args.seed,
                "agents": trace_agents,
            },
            outputs=outputs,
            summary={
                **trace_recorder.summary(),
                "crashes": summary.crashes if hasattr(summary, "crashes") else sum(pair.crashes for pair in summary.pairs),
                "timeouts": summary.timeouts if hasattr(summary, "timeouts") else sum(pair.timeouts for pair in summary.pairs),
            },
        )
        write_manifest(trace_run_dir.manifest_path, manifest)
        print("")
        print(f"trace_run_dir={trace_run_dir.path}")
        print(f"trace_frames={outputs['frames_jsonl']}")
        print(f"trace_manifest={trace_run_dir.manifest_path}")


if __name__ == "__main__":
    main_cli()
