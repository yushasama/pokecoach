"""Thin CLI wrapper for MatchupThreatInference v0.

Derives per-deck matchup-threat surfaces (presents/exposes/coverage) + bounded
subject-vs-opponent pairings from CardFunctionClassifier v2 tags and
DeckProfileInference v0 decklists, writing CSV + JSONL + pairings + report under
``.research/matchup_threat_inference_v0/``.

Offline/training-only: no gameplay code, no engine import, no submission change, and
no matchup win rates.

Examples
--------
    python scripts/build_matchup_threat_inference_v0.py
    python scripts/build_matchup_threat_inference_v0.py --decklists path/to/deck_zoo_decklists.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.matchup_threat_inference.build import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
