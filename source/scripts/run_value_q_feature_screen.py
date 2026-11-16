#!/usr/bin/env python3
"""Run the bounded F1 standardized logistic/PCA/semantic feature screen."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.value_lab.linear_screen import DEFAULT_SEEDS, run_linear_feature_screen  # noqa: E402


def _seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise argparse.ArgumentTypeError("exactly three distinct seeds are required")

    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--presentations-per-fold", type=int, default=1_048_576)
    parser.add_argument("--batch-rows", type=int, default=16_384)
    parser.add_argument("--seeds", type=_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--validation-limit", type=int, default=0, help="development smoke only; production uses all validation rows")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_linear_feature_screen(
        cache_dir=Path(args.cache),
        split_manifest_path=Path(args.split),
        inventory_path=Path(args.inventory),
        out_dir=Path(args.out),
        presentations_per_fold=args.presentations_per_fold,
        batch_rows=args.batch_rows,
        seeds=args.seeds,
        bootstrap_replicates=args.bootstrap_replicates,
        validation_limit=args.validation_limit,
    )
    print(manifest["run_id"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
