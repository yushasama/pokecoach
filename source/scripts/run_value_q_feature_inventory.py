#!/usr/bin/env python3
"""Create the exact F0 feature contract and training-only statistics receipt."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.value_lab.data import LabData  # noqa: E402
from deepmew.value_lab.feature_selection import (  # noqa: E402
    build_feature_inventory,
    write_feature_audit,
    write_feature_inventory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--inventory-out", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--correlation-sample-rows", type=int, default=131_072)
    parser.add_argument("--batch-rows", type=int, default=16_384)
    parser.add_argument("--seed", type=int, default=915001)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = LabData(args.cache, split_manifest_path=args.split)
    inventory = build_feature_inventory(
        data,
        cache_dir=Path(args.cache),
        split_manifest_path=Path(args.split),
        correlation_sample_rows=args.correlation_sample_rows,
        seed=args.seed,
        batch_rows=args.batch_rows,
    )
    write_feature_inventory(Path(args.inventory_out), inventory)
    write_feature_audit(Path(args.report_out), inventory)
    print(inventory["inventory_sha256"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
