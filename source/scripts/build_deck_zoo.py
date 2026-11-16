"""Build the K3 Deck Zoo (curated real-meta opponent decks/sets) from a K2A run.

Reads a K2A meta-kg run directory and emits deck_zoo.csv, deck_zoo_summary.csv,
arena_opponent_sets.csv, arena_matchups.csv, deck_zoo_decklists.jsonl, and
deck_zoo_report.md under a run directory with a manifest. Offline/training-only.

Examples
--------
    python scripts/build_deck_zoo.py
    python scripts/build_deck_zoo.py --meta-kg .research/runs/meta_kg/<date>/<run_id> --suffix k3
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.deck_zoo.build import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
