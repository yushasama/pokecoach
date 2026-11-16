"""Validate post-M artifacts and prepare, but never run, anchored evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.q_guided.acceptance_adapter import (  # noqa: E402
    ArtifactValidationError,
    Q_FEATURE_CLASSIFICATION,
    RECEIPT_SCHEMA_VERSION,
    validate_and_prepare,
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, help="synthetic/local adapter contract JSON")
    parser.add_argument("--out", required=True, help="durable validation/preparation receipt")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.out)
    try:
        receipt = validate_and_prepare(Path(args.contract))
    except ArtifactValidationError as exc:
        receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "status": "hard_stop",
            "classification": Q_FEATURE_CLASSIFICATION,
            "error_type": type(exc).__name__,
            "reason": str(exc),
            "anchored_evaluation": None,
        }
        _write_json(output, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))

        return 2
    _write_json(output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
