"""Run the train/eval-only correction for the strict-public DSU posterior."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.meta_kg.belief_correction import run_no_staple_correction  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-kg-dir", required=True)
    parser.add_argument("--deck-zoo-dir", required=True)
    parser.add_argument("--ingest-root", default=".research/kaggle_ingest")
    parser.add_argument("--opponent-decks-root", default=".research/opponent_decks")
    parser.add_argument("--out", default=".research/dsu_belief_no_staple_correction_v0")
    parser.add_argument("--max-games", type=int, default=None)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_no_staple_correction(
        meta_kg_dir=args.meta_kg_dir,
        deck_zoo_dir=args.deck_zoo_dir,
        ingest_root=args.ingest_root,
        opponent_decks_root=args.opponent_decks_root,
        output_dir=args.out,
        max_games=args.max_games,
    )
    evaluation = result["evaluation"]
    print(json.dumps({
        "feature_classification": evaluation["feature_classification"],
        "calibration_feature_decision": evaluation["calibration_feature_decision"],
        "selected_variant": evaluation["selected_variant"],
        "selected_evidence_scale": evaluation["selected_evidence_scale"],
        "audit_delta": evaluation["audit_delta_calibrated_vs_uncalibrated"],
    }, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
