from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.artifacts.run_dir import create_run_dir  # noqa: E402
from deepmew.arena import parse_agent_spec  # noqa: E402
from deepmew.trace.attachment_timing import DEFAULT_MATCHUPS, run_trace_matchups  # noqa: E402
from deepmew.trace.evolve_sequencing import (  # noqa: E402
    RUN_KIND,
    load_source_run_frames,
    write_evolve_sequencing_outputs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit decoupled Mega-line EVOLVE sequencing for current-deck traces."
    )
    parser.add_argument("--source-run", help="Existing run directory containing frames.jsonl.")
    parser.add_argument("--agent", help="Agent spec to audit. Omit to preserve the legacy default-v2.4 matchup set.")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20240701)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--limit-frames", type=int)
    parser.add_argument("--out-root", default=".research/runs")
    parser.add_argument("--kind", default=RUN_KIND)
    parser.add_argument("--export-handoff-bundle", action="store_true")
    parser.add_argument("--handoff-root", default=".research/handoff_bundles")
    return parser.parse_args(argv)


def _command() -> str:
    return "python scripts\\audit_deck_v3_evolve_sequencing.py " + " ".join(sys.argv[1:])


def _pair_summary(pair: object) -> dict:
    return {
        "agent_a": getattr(pair, "agent_a", None),
        "agent_b": getattr(pair, "agent_b", None),
        "games": getattr(pair, "games", None),
        "completed": getattr(pair, "completed", None),
        "a_wins": getattr(pair, "a_wins", None),
        "b_wins": getattr(pair, "b_wins", None),
        "draws": getattr(pair, "draws", None),
        "crashes": getattr(pair, "crashes", None),
        "timeouts": getattr(pair, "timeouts", None),
        "avg_turns": getattr(pair, "avg_turns", None),
        "elapsed_seconds": getattr(pair, "elapsed_seconds", None),
    }


def _frames(args: argparse.Namespace, run_id: str, agent_spec: str, agent_label: str) -> tuple[list[dict], str | None, dict]:
    if args.source_run:
        return load_source_run_frames(args.source_run), args.source_run, {"source_run": args.source_run, "agent": agent_spec, "agent_label": agent_label}

    matchups = list(DEFAULT_MATCHUPS) if args.agent is None else [(agent_spec, "first-legal")]
    frames, pair_summaries, recorder_summary = run_trace_matchups(
        run_id=run_id,
        matchups=matchups,
        games=args.games,
        seed=args.seed,
        max_steps=args.max_steps,
        limit_frames=args.limit_frames,
    )
    return (
        frames,
        None,
        {
            "source_run": None,
            "agent": agent_spec,
            "agent_label": agent_label,
            "games": args.games,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "limit_frames": args.limit_frames,
            "matchups": [list(matchup) for matchup in matchups],
            "pair_summaries": [_pair_summary(pair) for pair in pair_summaries],
            "recorder_summary": recorder_summary,
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    agent_spec = args.agent or "v2.4"
    agent_label = parse_agent_spec(agent_spec).label
    run_dir = create_run_dir(kind=args.kind, out_root=Path(args.out_root))
    frames, source_run, inputs = _frames(args, run_dir.run_id, agent_spec, agent_label)
    result = write_evolve_sequencing_outputs(
        run_dir=run_dir.path,
        run_id=run_dir.run_id,
        command=_command(),
        source_run=source_run,
        frames=frames,
        agent_label=agent_label,
        inputs=inputs,
        export_handoff_bundle=args.export_handoff_bundle,
        handoff_root=args.handoff_root,
    )
    metrics = result.audit["metrics"]
    print(f"evolve_sequencing_run_dir={result.run_dir}")
    print(f"manifest={result.manifest_path}")
    print(f"evolve_sequencing={result.run_dir / 'evolve_sequencing.json'}")
    print(f"evolve_events={result.run_dir / 'evolve_events.jsonl'}")
    print(f"report={result.run_dir / 'report.md'}")
    if result.bundle_dir is not None:
        print(f"handoff_bundle_dir={result.bundle_dir}")
    print(f"total_important_evolve_legal_frames={metrics['total_important_evolve_legal_frames']}")
    print(f"same_frame_chosen_count={metrics['same_frame_chosen_count']}")
    print(f"delayed_count={metrics['delayed_count']}")
    print(f"never_observed_count={metrics['never_observed_count']}")
    print(f"most_frequent_action_that_beat_evolve={metrics['most_frequent_action_that_beat_evolve']}")
    print(f"median_score_gap_chosen_minus_evolve={metrics['median_score_gap_chosen_minus_evolve']}")
    print(f"conclusion={metrics['conclusion']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
