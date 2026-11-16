"""Freeze label-blind test predictions and the one-shot scoring plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepmew.value_lab.sealed_test_prediction import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    run_sealed_test_prediction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--crossfit", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--control-pass", required=True)
    parser.add_argument("--screening", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = run_sealed_test_prediction(
        cache_dir=Path(args.cache),
        split_path=Path(args.split),
        crossfit_dir=Path(args.crossfit),
        calibration_dir=Path(args.calibration),
        control_pass_dir=Path(args.control_pass),
        screening_dir=Path(args.screening),
        out_dir=Path(args.out),
        batch_size=args.batch_size,
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
