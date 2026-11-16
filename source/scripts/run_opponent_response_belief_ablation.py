"""Run the chronological opponent-response DSU contribution ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.opponent_response import run_response_ablation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-kg-dir", required=True)
    parser.add_argument("--deck-zoo-dir", required=True)
    parser.add_argument("--ingest-root", default=".research/kaggle_ingest")
    parser.add_argument("--opponent-decks-root", default=".research/opponent_decks")
    parser.add_argument("--canonical-split-manifest", required=True)
    parser.add_argument(
        "--dsu-correction-result",
        default=".research/dsu_belief_no_staple_correction_v0/result.json",
    )
    parser.add_argument("--out", default=".research/opponent_response_belief_v0")
    parser.add_argument("--max-games", type=int, default=None)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_response_ablation(
        meta_kg_dir=args.meta_kg_dir,
        deck_zoo_dir=args.deck_zoo_dir,
        ingest_root=args.ingest_root,
        opponent_decks_root=args.opponent_decks_root,
        canonical_split_manifest=args.canonical_split_manifest,
        dsu_correction_result=args.dsu_correction_result,
        output_dir=args.out,
        max_games=args.max_games,
    )
    evaluation = result["evaluation"]
    primary = evaluation["calibration"]
    print(json.dumps({
        "verdict": evaluation["dsu_response_feature_decision"],
        "marginal_contribution": primary["marginal_contribution"],
        "game_clustered_delta": primary["game_clustered_delta"],
        "dataset": {
            "train_examples": result["dataset"]["train_examples_used"],
            "validation_examples": result["dataset"]["examples_validation"],
            "calibration_examples": result["dataset"]["examples_calibration"],
            "sealed_test_examples_scored": result["dataset"]["sealed_test_examples_scored"],
        },
    }, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
