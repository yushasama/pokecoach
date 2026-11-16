"""Run equal-budget nonlinear confirmation for F1-selected value features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepmew.value_lab.feature_confirmation import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_SEEDS,
    RUNG_TARGET_ROWS,
    run_feature_confirmation,
)


def _seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be unique nonnegative integers")

    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--representation", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rung", choices=sorted(RUNG_TARGET_ROWS), required=True)
    parser.add_argument("--rows", type=int, default=None, help="Targeted-test override only.")
    parser.add_argument("--seeds", type=_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-matched-controls", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = run_feature_confirmation(
        cache_dir=Path(args.cache),
        split_path=Path(args.split),
        selection_path=Path(args.selection),
        representation_paths=tuple(Path(path) for path in args.representation),
        out_dir=Path(args.out),
        rung=args.rung,
        presentations=args.rows,
        seeds=args.seeds,
        batch_size=args.batch_size,
        threads=args.threads,
        bootstrap_replicates=args.bootstrap_replicates,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        include_matched_controls=not args.no_matched_controls,
        command=[sys.executable, *sys.argv],
    )
    manifest = receipt["manifest"]
    print(json.dumps({
        "action": receipt["action"],
        "run_id": manifest["run_id"],
        "status": manifest["status"],
        "summary": manifest["summary"],
    }, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
