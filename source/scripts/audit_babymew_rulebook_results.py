from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.artifacts.run_dir import create_run_dir  # noqa: E402
from deepmew.trace.rulebook_results_sweep import (  # noqa: E402
    DEFAULT_AGENTS,
    RUN_KIND,
    collect_results_sweep,
    _highest_value_next,
    write_results_outputs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BabyMew V3.10 rulebook results sweep.")
    parser.add_argument("--agent", action="append", help="Agent spec to include. Repeatable.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[20240720, 20240721, 20240722])
    parser.add_argument("--games-per-pair", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--limit-frames", type=int)
    parser.add_argument("--out-root", default=".research/runs")
    parser.add_argument("--kind", default=RUN_KIND)
    parser.add_argument("--export-handoff-bundle", action="store_true")
    parser.add_argument("--handoff-root", default=".research/handoff_bundles")
    return parser.parse_args(argv)


def _command() -> str:
    return "python scripts\\audit_babymew_rulebook_results.py " + " ".join(sys.argv[1:])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    agents = tuple(args.agent or DEFAULT_AGENTS)
    run_dir = create_run_dir(kind=args.kind, out_root=Path(args.out_root))
    frames, matrices, recorder_summary = collect_results_sweep(
        run_id=run_dir.run_id,
        agents=agents,
        seeds=args.seeds,
        games_per_pair=args.games_per_pair,
        max_steps=args.max_steps,
        limit_frames=args.limit_frames,
    )
    result = write_results_outputs(
        run_dir=run_dir.path,
        run_id=run_dir.run_id,
        command=_command(),
        frames=frames,
        matrices=matrices,
        inputs={
            "agents": list(agents),
            "seeds": args.seeds,
            "games_per_pair": args.games_per_pair,
            "max_steps": args.max_steps,
            "limit_frames": args.limit_frames,
            "recorder_summary": recorder_summary,
        },
        export_handoff_bundle=args.export_handoff_bundle,
        handoff_root=args.handoff_root,
    )
    multi = result.summary["multi_seed"]
    attack = result.summary["attack"]
    unknown = result.summary["unknown"]
    print(f"rulebook_results_sweep_run_dir={result.run_dir}")
    print(f"manifest={result.manifest_path}")
    print(f"report={result.run_dir / 'report.md'}")
    print(f"multi_seed_matrix_summary={result.run_dir / 'multi_seed_matrix_summary.json'}")
    print(f"pair_sweep_summary={result.run_dir / 'pair_sweep_summary.json'}")
    print(f"attack_pressure_collapse={result.run_dir / 'attack_pressure_collapse.json'}")
    print(f"unknown_collapse={result.run_dir / 'unknown_collapse.json'}")
    print(f"collapsed_delta_events={result.run_dir / 'collapsed_delta_events.jsonl'}")
    if result.bundle_dir is not None:
        print(f"handoff_bundle_dir={result.bundle_dir}")
    print(f"aggregate_rankings={multi['aggregate_rankings']}")
    print(f"crashes={multi['crashes']}")
    print(f"timeouts={multi['timeouts']}")
    print(f"seed_variance_too_high={multi['seed_variance_too_high']}")
    print(f"attack_pressure_collapse_counts={attack['collapse_counts']}")
    print(f"unknown_collapse_counts={unknown['collapse_counts']}")
    print(f"highest_value_next={_highest_value_next(attack, unknown, multi)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
