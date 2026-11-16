"""Build the evidence-backed compact strategy ontology and deck profiles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.strategy_tensor.ontology import build_strategy_ontology  # noqa: E402


DEFAULT_FINE_EXAMPLES = (
    ".research/runs/imitation_fine/20260707/"
    "20260707T054549Z-imitation_fine-r16/imitation_fine_examples.jsonl"
)
DEFAULT_FULL_LISTS = (
    ".research/runs/deck_zoo/20260707/"
    "20260707T054313Z-deck_zoo-r16/deck_zoo_decklists.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", default=DEFAULT_FINE_EXAMPLES)
    parser.add_argument("--full-lists", default=DEFAULT_FULL_LISTS)
    parser.add_argument("--out", default=".research/strategy_tensor/ontology_v1")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_strategy_ontology(
        examples_path=args.examples,
        full_lists_path=args.full_lists,
        output_dir=args.out,
    )
    print(json.dumps({
        "verdict": result["verdict"],
        "profile_support": result["profile_support"],
        "full_list_join": result["full_list_join"],
    }, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
