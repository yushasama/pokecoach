"""K2B eval of K5-lite pilots (filtered/reward-weighted BC) vs v2.4/first-legal.

Pilots the frozen `deck.csv` with each candidate policy against the K3 deck-zoo
buckets (first-legal piloted) plus the category ReplayPolicy on top_meta, and
reports our-deck win rate per pilot. Eval-only; runtime/default v2.4 untouched.

Examples
--------
    python scripts/eval_k5_lite_arena.py --games 6 --max-decks-per-bucket 3
    python scripts/eval_k5_lite_arena.py --no-replay --suffix smoke
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.imitation_fine.arena_eval import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
