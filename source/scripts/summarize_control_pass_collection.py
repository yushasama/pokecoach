"""Verify and summarize a strict-public control-pass collection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.strategy_tensor.control_pass import summarize_control_pass_collection  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=".research/prospective_trace/control_pass_strategy_v1_20260711",
    )

    return parser.parse_args()


def main() -> int:
    result = summarize_control_pass_collection(collection_root=parse_args().root)
    print(json.dumps({
        "verdict": result["verdict"],
        "summary": result["summary"],
        "raw_coverage_gate": result["raw_coverage_gate"],
    }, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
