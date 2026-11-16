from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.artifacts.run_dir import create_run_dir  # noqa: E402
from deepmew.trace.turn_plan_review import RUN_KIND, collect_turn_plan_frames, write_turn_plan_outputs  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the experimental V4 turn-plan policy.")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20240733)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--out-root", default=".research/runs")
    parser.add_argument("--kind", default=RUN_KIND)
    parser.add_argument("--export-handoff-bundle", action="store_true")
    parser.add_argument("--handoff-root", default=".research/handoff_bundles")
    return parser.parse_args(argv)


def _command() -> str:
    return "python scripts\\audit_v4_turn_plan.py " + " ".join(sys.argv[1:])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = create_run_dir(kind=args.kind, out_root=Path(args.out_root))
    frames, run_meta = collect_turn_plan_frames(
        run_id=run_dir.run_id,
        games=args.games,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    summary, bundle_dir = write_turn_plan_outputs(
        run_dir=run_dir.path,
        run_id=run_dir.run_id,
        command=_command(),
        frames=frames,
        run_meta={"games": args.games, "seed": args.seed, "max_steps": args.max_steps, **run_meta},
        export_handoff_bundle=args.export_handoff_bundle,
        handoff_root=args.handoff_root,
    )
    print(f"v4_turn_plan_run_dir={run_dir.path}")
    print(f"manifest={run_dir.manifest_path}")
    print(f"report={run_dir.path / 'report.md'}")
    print(f"turn_plan_summary={run_dir.path / 'turn_plan_summary.json'}")
    print(f"turn_plan_events={run_dir.path / 'turn_plan_events.jsonl'}")
    if bundle_dir is not None:
        print(f"handoff_bundle_dir={bundle_dir}")
    print(f"planned_count={summary['planned_count']}")
    print(f"fallback_count={summary['fallback_count']}")
    print(f"intent_counts={summary['intent_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
