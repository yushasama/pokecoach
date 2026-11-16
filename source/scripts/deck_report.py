from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.deck import format_report, summarize_deck, validate_deck  # noqa: E402


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="Inspect and validate a DeepMew deck CSV.")
    parser.add_argument("deck", help="Path to deck.csv")
    parser.add_argument("--no-engine-check", action="store_true")
    parser.add_argument("--out", help="Optional path to write the Markdown report.")
    args = parser.parse_args()

    summary = summarize_deck(args.deck)
    validation = validate_deck(args.deck, check_engine=not args.no_engine_check)
    report = format_report(summary, validation)
    print(report)

    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report + "\n", encoding="utf-8")
        print("")
        print(f"wrote_report={output}")

    if not validation.valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main_cli()
