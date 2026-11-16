"""Preflight or execute the single-use aggregate-only sealed value test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepmew.value_lab.sealed_test_scoring import (  # noqa: E402
    preflight_sealed_scoring,
    run_sealed_test_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--preflight-only", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    common = {
        "cache_dir": Path(args.cache),
        "split_path": Path(args.split),
        "prediction_dir": Path(args.predictions),
        "out_dir": Path(args.out),
    }
    if args.preflight_only:
        print(json.dumps(preflight_sealed_scoring(**common), indent=2, sort_keys=True))
    else:
        receipt = run_sealed_test_score(
            **common,
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
