#!/usr/bin/env python
"""Build a hash-verified statistical addendum from a completed postmortem run."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.submission_postmortem.addendum_runner import run_statistical_addendum  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a completed Submission Postmortem run and compute additive uncertainty, "
            "trend, matchup, initiative, timing, confidence, model, cluster, priority, and memo artifacts."
        )
    )
    parser.add_argument("--base-run", required=True, type=Path, help="completed base postmortem output directory")
    parser.add_argument("--out-dir", type=Path, help="addendum output; defaults to BASE_RUN/addendum-v0")
    parser.add_argument("--human-addendum", type=Path, help="optional second Markdown output path")
    parser.add_argument(
        "--authoritative-report",
        type=Path,
        help="optional authoritative human report to hash-bind in the addendum manifest",
    )
    parser.add_argument("--random-seed", type=int, default=54788723)
    parser.add_argument("--permutations", type=int, default=40_000)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.permutations <= 0:
        parser.error("--permutations must be positive")
    command_values = [sys.executable, *sys.argv] if argv is None else [sys.executable, __file__, *argv]
    command = " ".join(shlex.quote(value) for value in command_values)
    try:
        result = run_statistical_addendum(
            args.base_run,
            output_dir=args.out_dir,
            human_addendum_path=args.human_addendum,
            authoritative_report_path=args.authoritative_report,
            random_seed=args.random_seed,
            permutation_samples=args.permutations,
            command=command,
        )
    except Exception as exc:  # noqa: BLE001 - fail-closed CLI receipt
        print(f"submission postmortem addendum failed: {type(exc).__name__}: {exc}", file=sys.stderr)

        return 2
    print(json.dumps({
        "status": "complete",
        "output_dir": str(result.output_dir.resolve()),
        "manifest": str(result.manifest_path.resolve()),
        "addendum": str(result.addendum_path.resolve()),
        "payload": str(result.payload_path.resolve()),
    }, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
