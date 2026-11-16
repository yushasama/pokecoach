"""Verify D1-successor checkpoint, leakage, and protected-runtime gates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from scripts import verify_h2_x3_posttraining_gates as gates


ROOT = Path(__file__).resolve().parents[1]
TRAINING_FLOOR_SECONDS = 11_900.0


def verify(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = gates.checkpoint_finiteness(Path(args.successor_checkpoint).resolve())
    training_path = Path(args.training_receipt).resolve()
    leakage_path = Path(args.leakage_receipt).resolve()
    training = gates._read_json(training_path)
    leakage = gates._read_json(leakage_path)
    summary = training.get("summary", {})
    progress = summary.get("progress", {})
    root_hashes = {name: gates._sha256(ROOT / name) for name in gates.EXPECTED_ROOT_HASHES}
    worktree_diff = gates._git_diff()
    index_diff = gates._git_diff("--cached")
    safety = summary.get("safety")
    training_clean = (
        training.get("status") == "PASS"
        and summary.get("status") == "minimum_training_floor_reached"
        and float(progress.get("cumulative_training_seconds", 0.0)) >= TRAINING_FLOOR_SECONDS
        and safety == {"crashes": 0, "illegal_actions": 0, "timeouts": 0}
        and summary.get("protected_surfaces_mutated") is False
    )
    leakage_clean = leakage.get("status") == "PASS" and leakage.get("marker_diffs") == 0
    runtime_clean = (
        root_hashes == gates.EXPECTED_ROOT_HASHES
        and not worktree_diff
        and not index_diff
        and training_clean
    )
    status = "PASS" if checkpoint["finite"] and leakage_clean and runtime_clean else "FAIL"

    return {
        "schema_version": "deepmew.d1_successor_posttraining_gate_receipt.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "leakage": not leakage_clean,
        "runtime_contamination": not runtime_clean,
        "successor_checkpoint_finite": checkpoint["finite"],
        "checkpoint": checkpoint,
        "training": {
            "path": str(training_path),
            "status": training.get("status"),
            "summary_status": summary.get("status"),
            "cumulative_training_seconds": progress.get("cumulative_training_seconds"),
            "required_floor_seconds": TRAINING_FLOOR_SECONDS,
            "safety": safety,
            "clean": training_clean,
        },
        "feature_leakage_audit": {
            "path": str(leakage_path),
            "status": leakage.get("status"),
            "rows_checked": leakage.get("rows_checked"),
            "marker_diffs": leakage.get("marker_diffs"),
        },
        "protected_runtime": {
            "paths": list(gates.PROTECTED_PATHS),
            "expected_root_hashes": gates.EXPECTED_ROOT_HASHES,
            "observed_root_hashes": root_hashes,
            "worktree_diff_empty": not worktree_diff,
            "index_diff_empty": not index_diff,
            "training_receipt_clean": training_clean,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--successor-checkpoint", required=True)
    parser.add_argument("--training-receipt", required=True)
    parser.add_argument("--leakage-receipt", required=True)
    parser.add_argument("--output", required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = verify(args)
    gates._atomic_json(Path(args.output).resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))

    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
