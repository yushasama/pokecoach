"""Train and independently gate the 20-slot full-list strategy classifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.strategy_tensor.classifier import run_full_list_classifier  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", default=".research/strategy_tensor/ontology_v1/deck_profiles.jsonl")
    parser.add_argument("--ontology", default=".research/strategy_tensor/ontology_v1/ontology.json")
    parser.add_argument("--deck-zoo", default=".research/runs/deck_zoo/20260707/20260707T054313Z-deck_zoo-r16/deck_zoo.csv")
    parser.add_argument("--complete-list-root", default=".research/opponent_decks")
    parser.add_argument("--out", default=".research/strategy_tensor/full_list_classifier_v1")
    parser.add_argument(
        "--feature-mode",
        choices=("card_counts", "semantic_profile"),
        default="card_counts",
    )
    parser.add_argument(
        "--v2-cards",
        default=".research/card_function_classifier_v2/all_cards_function_tags_v2.csv",
    )
    parser.add_argument("--fresh-split-salt", default=None)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_full_list_classifier(
        profiles_path=args.profiles,
        ontology_path=args.ontology,
        deck_zoo_path=args.deck_zoo,
        complete_list_root=args.complete_list_root,
        output_dir=args.out,
        feature_mode=args.feature_mode,
        v2_cards_path=args.v2_cards if args.feature_mode == "semantic_profile" else None,
        fresh_split_salt=args.fresh_split_salt,
    )
    print(json.dumps({
        "verdict": result["verdict"],
        "eval_gate": result["evaluation"]["eval_gate"],
        "holdout_gate": result["evaluation"]["holdout_gate"],
        "feature_contribution": result["feature_contribution"],
    }, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
