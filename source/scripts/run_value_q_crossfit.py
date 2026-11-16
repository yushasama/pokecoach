"""Run the selected raw-S1 snapshot grouped cross-fit and final full-train model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepmew.value_lab.crossfit import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_THREADS,
    FINAL_SEED,
    run_grouped_crossfit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--representation", required=True)
    parser.add_argument("--feature-confirmation-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=FINAL_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--audit-only", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = run_grouped_crossfit(
        cache_dir=Path(args.cache),
        split_path=Path(args.split),
        representation_path=Path(args.representation),
        feature_confirmation_manifest=Path(args.feature_confirmation_manifest),
        out_dir=Path(args.out),
        seed=args.seed,
        batch_size=args.batch_size,
        threads=args.threads,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        stop_after_audit=args.audit_only,
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
