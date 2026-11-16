"""Validate, transition, and render DeepMew's capability-debt ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.capability_debt import (  # noqa: E402
    apply_change_file,
    load_ledger,
    render_outputs,
)


DEFAULT_ROOT = Path(".research/capability_debt")
DEFAULT_LEDGER = DEFAULT_ROOT / "capability_debt_ledger.json"
DEFAULT_MARKDOWN = DEFAULT_ROOT / "CAPABILITY_DEBT_LEDGER.md"
DEFAULT_CSV = DEFAULT_ROOT / "capability_debt_ledger.csv"
DEFAULT_MANIFEST = DEFAULT_ROOT / "manifest.json"


def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Maintain the research capability-debt ledger without rerunning a postmortem."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate the canonical JSON ledger")
    validate.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    render = subparsers.add_parser("render", help="regenerate Markdown, CSV, and manifest")
    _add_output_arguments(render)
    apply = subparsers.add_parser(
        "apply",
        help="apply one additive or history-preserving transition change, then render",
    )
    _add_output_arguments(apply)
    apply.add_argument("--change", type=Path, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        ledger = load_ledger(args.ledger)
        print(json.dumps({"status": "valid", "rows": len(ledger["rows"])}))

        return 0
    if args.command == "apply":
        apply_change_file(args.ledger, args.change)
    manifest = render_outputs(args.ledger, args.markdown, args.csv, args.manifest)
    print(json.dumps({
        "status": manifest["status"],
        "rows": manifest["row_count"],
        "manifest": str(args.manifest),
    }))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
