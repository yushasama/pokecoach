"""Build the K2A meta knowledge graph from mined Kaggle ladder deck artifacts.

Reads mined `deck_summary.json` deck artifacts and emits a versioned
`deepmew.meta_kg.v2` artifact bundle: card_catalog.csv, deck_archetypes.csv,
archetype_summary.csv, support_packages.csv, staples.csv,
card_lift_by_archetype.csv, deck_similarity.csv, live_belief.json, and
deck_meta_report.md under a run directory with a manifest.

Offline/training-only: no gameplay code, no engine import, no submission change.

Examples
--------
    python scripts/build_meta_kg.py
    python scripts/build_meta_kg.py --input .research/opponent_decks --config configs/meta_kg/k2a.yaml
    python scripts/build_meta_kg.py --out-root .research/runs --suffix k2a_v2
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.meta_kg.build import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
