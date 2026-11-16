"""Materialize scout leaderboard ranks onto mined deck artifacts (K1.7).

Joins the scout `selected_teams` rank table onto deck summaries (by team_name) and
K2A archetype labels (by deck_hash) to emit rank backfill / coverage / per-deck /
rank-weighted archetype / divergence / top-player tables + a report under
`.research/runs/rank_metadata/`. No redownload; original deck summaries untouched;
research/eval-only.

Examples
--------
    python scripts/build_rank_metadata.py
    python scripts/build_rank_metadata.py --suffix k1_7
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.rank_meta.backfill import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
