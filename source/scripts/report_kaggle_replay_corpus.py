from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.kaggle_replays.corpus import CorpusOptions, report_kaggle_replay_corpus  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report on downloaded and ingested Kaggle replay corpus artifacts.")
    parser.add_argument("--replay-root", default=".research/kaggle_replays")
    parser.add_argument("--ingest-root", default=".research/kaggle_ingest")
    parser.add_argument("--opponent-decks-root", default=".research/opponent_decks")
    parser.add_argument("--out-root", default=".research")
    parser.add_argument("--section", default="corpus")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = report_kaggle_replay_corpus(
        CorpusOptions(
            replay_root=Path(args.replay_root),
            ingest_root=Path(args.ingest_root),
            opponent_decks_root=Path(args.opponent_decks_root),
            out_root=Path(args.out_root),
            section=args.section,
            command=" ".join(sys.argv),
            args=vars(args),
        )
    )
    print(f"manifest={result.manifest_path}")
    print(f"report={result.report_path}")
    print(f"deck_hash_summary={result.deck_hash_summary_path}")
    print(f"player_summary={result.player_summary_path}")
    print(f"archetype_hint_summary={result.archetype_hint_summary_path}")
    print(f"summary={result.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
