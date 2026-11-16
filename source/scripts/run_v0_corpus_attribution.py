"""Audit frozen V0 corpus attribution without loading labels or predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.strategy_tensor import run_v0_corpus_attribution  # noqa: E402


DEFAULT_CANONICAL_ROOT = "D:/DeepMew_artifacts/datasets/value_q_canonical_v1"
DEFAULT_CACHE = (
    "D:/DeepMew_artifacts/datasets/value_q_canonical_v1/"
    "value_lab_cache_v2/7459a801-77afa18a"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--canonical-root", default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument(
        "--archetype-labels",
        default=(
            ".research/runs/meta_kg/20260707/"
            "20260707T054219Z-meta_kg-r16/deck_archetypes.csv"
        ),
    )
    parser.add_argument(
        "--deck-pools",
        default=(
            ".research/runs/bucket_pool/20260707/"
            "20260707T141406Z-bucket_pool-1c29c3dd/bucket_pools.json"
        ),
    )
    parser.add_argument(
        "--exact-starmie-deck",
        default=".research/submissions/20260708-starmie-awr/deck.csv",
    )
    parser.add_argument(
        "--additional-starmie-deck",
        action="append",
        default=[
            ".research/curriculum/decks/"
            "top_staryu_windecks_b15853cdfb83.deck.csv"
        ],
    )
    parser.add_argument("--out", default=".research/strategy_tensor/v0_corpus_attribution")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_v0_corpus_attribution(
        cache_dir=args.cache_dir,
        canonical_root=args.canonical_root,
        archetype_labels_path=args.archetype_labels,
        deck_pool_path=args.deck_pools,
        exact_starmie_deck_path=args.exact_starmie_deck,
        additional_starmie_deck_paths=args.additional_starmie_deck,
        output_dir=args.out,
    )
    audit = result["audit"]
    starmie_summary = {
        key: value
        for key, value in audit["starmie"].items()
        if key != "supported_hash_provenance"
    }
    print(json.dumps({
        "verdict": audit["classification"]["verdict"],
        "eligible_rows": audit["totals"]["eligible_rows"],
        "eligible_games": audit["totals"]["games_with_eligible_rows"],
        "starmie": starmie_summary,
    }, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
