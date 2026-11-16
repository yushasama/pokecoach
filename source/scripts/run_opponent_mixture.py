from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.opponents.registry import list_mixtures  # noqa: E402
from deepmew.opponents.reporting import run_opponent_mixture  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one agent against a fixed opponent mixture.")
    parser.add_argument("--agent", required=True, help="Agent spec under test, e.g. v2.4 or v4.")
    parser.add_argument("--mixture", required=True, choices=list(list_mixtures()))
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out-root", default=".research/runs")
    parser.add_argument("--bundle-root", default=".research/handoff_bundles")
    parser.add_argument("--no-bundle", action="store_true", help="Skip compact review bundle export.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = "python scripts\\run_opponent_mixture.py " + " ".join(sys.argv[1:])
    result = run_opponent_mixture(
        agent=args.agent,
        mixture_name=args.mixture,
        games=args.games,
        max_steps=args.max_steps,
        seed=args.seed,
        out_root=args.out_root,
        bundle_root=args.bundle_root,
        write_bundle=not args.no_bundle,
        command=command,
    )
    summary = result.summary
    print(
        " ".join(
            [
                f"completed={summary['completed']}/{summary['games']}",
                f"agent_wins={summary['agent_wins']}",
                f"opponent_wins={summary['opponent_wins']}",
                f"draws={summary['draws']}",
                f"crashes={summary['crashes']}",
                f"timeouts={summary['timeouts']}",
            ]
        )
    )
    print(f"run_dir={result.run_dir}")
    print(f"manifest={result.manifest_path}")
    print(f"report={result.report_path}")
    if result.bundle_dir is not None:
        print(f"handoff_bundle={result.bundle_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
