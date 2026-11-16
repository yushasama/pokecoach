"""Build the K2B real-meta arena: validate K3 decks + evaluate DeepMew vs them.

Loads the latest K3 Deck Zoo decklists and K4 imitation examples, validates each
canonical decklist through the engine, builds a ReplayPolicy opponent from the K4
train tables, and runs fixed-policy matchups of `deck.csv` (v2.4 / first-legal)
against the top_meta / anti_overfit_holdout / unknown_mixed buckets (and a
ReplayPolicy opponent on top_meta). Eval-only; runtime/default v2.4 untouched.

Examples
--------
    python scripts/build_real_meta_arena.py
    python scripts/build_real_meta_arena.py --games 8 --max-decks-per-bucket 5 --suffix k2b
    python scripts/build_real_meta_arena.py --no-replay
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.real_meta_arena.build import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
