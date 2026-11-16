"""Build the exact nullable control-pass sidecar for the accepted value cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepmew.value_lab.control_pass_mask import run_control_pass_mask_build  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out", required=True)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = run_control_pass_mask_build(
        cache_dir=Path(args.cache),
        split_path=Path(args.split),
        out_dir=Path(args.out),
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
