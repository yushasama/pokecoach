"""Select and freeze grouped calibration for the completed value cross-fit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepmew.value_lab.calibration_selection import (  # noqa: E402
    CALIBRATION_BOOTSTRAP_REPLICATES,
    run_grouped_calibration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--crossfit", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=CALIBRATION_BOOTSTRAP_REPLICATES)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = run_grouped_calibration(
        cache_dir=Path(args.cache),
        split_path=Path(args.split),
        crossfit_dir=Path(args.crossfit),
        out_dir=Path(args.out),
        bootstrap_replicates=args.bootstrap_replicates,
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
