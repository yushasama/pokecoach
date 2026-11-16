from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.artifacts.run_dir import create_run_dir  # noqa: E402
from deepmew.trace.rulebook_audit import (  # noqa: E402
    DEFAULT_POLICIES,
    DEFAULT_SOURCE_AGENTS,
    RUN_KIND,
    collect_rulebook_frames,
    write_rulebook_outputs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BabyMew rulebook reset audit.")
    parser.add_argument("--source-agent", action="append", help="Source agent to collect frames from. Repeatable.")
    parser.add_argument("--policy", action="append", help="Policy to probe on each same observation. Repeatable.")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20240713)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--limit-frames", type=int)
    parser.add_argument("--out-root", default=".research/runs")
    parser.add_argument("--kind", default=RUN_KIND)
    parser.add_argument("--export-handoff-bundle", action="store_true")
    parser.add_argument("--handoff-root", default=".research/handoff_bundles")
    return parser.parse_args(argv)


def _command() -> str:
    return "python scripts\\audit_babymew_rulebook.py " + " ".join(sys.argv[1:])


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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_agents = tuple(args.source_agent or DEFAULT_SOURCE_AGENTS)
    policies = tuple(args.policy or DEFAULT_POLICIES)
    run_dir = create_run_dir(kind=args.kind, out_root=Path(args.out_root))
    frames, pair_summaries, recorder_summary = collect_rulebook_frames(
        run_id=run_dir.run_id,
        source_agents=source_agents,
        games=args.games,
        seed=args.seed,
        max_steps=args.max_steps,
        limit_frames=args.limit_frames,
    )
    result = write_rulebook_outputs(
        run_dir=run_dir.path,
        run_id=run_dir.run_id,
        command=_command(),
        frames=frames,
        source_agents=source_agents,
        policies=policies,
        inputs={
            "source_agents": list(source_agents),
            "policies": list(policies),
            "games": args.games,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "limit_frames": args.limit_frames,
            "pair_summaries": [_pair_summary(pair) for pair in pair_summaries],
            "recorder_summary": recorder_summary,
        },
        export_handoff_bundle=args.export_handoff_bundle,
        handoff_root=args.handoff_root,
    )
    metrics = result.summary["metrics"]
    print(f"rulebook_audit_run_dir={result.run_dir}")
    print(f"manifest={result.manifest_path}")
    print(f"report={result.run_dir / 'report.md'}")
    print(f"rule_inventory={result.run_dir / 'rule_inventory.json'}")
    print(f"decision_delta_summary={result.run_dir / 'decision_delta_summary.json'}")
    print(f"decision_delta_events={result.run_dir / 'decision_delta_events.jsonl'}")
    print(f"rule_impact_summary={result.run_dir / 'rule_impact_summary.json'}")
    if result.bundle_dir is not None:
        print(f"handoff_bundle_dir={result.bundle_dir}")
    print(f"total_frames_analyzed={metrics['total_frames_analyzed']}")
    print(f"total_delta_events={metrics['total_delta_events']}")
    print(f"classification_counts={metrics['classification_counts']}")
    print(f"likely_rule_family_counts={metrics['likely_rule_family_counts']}")
    print(f"recommendation={metrics['recommendation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
