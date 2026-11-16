"""Thin CLI wrapper for DeckProfileInference v0.

Aggregates CardFunctionClassifier v2 tags over mined 60-card decklists into
count-weighted deck profiles, writing CSV + JSONL + report under
``.research/deck_profile_inference_v0/``.

Offline/training-only: no gameplay code, no engine import, no submission change.

Examples
--------
    python scripts/build_deck_profile_inference_v0.py
    python scripts/build_deck_profile_inference_v0.py --decklists path/to/deck_zoo_decklists.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.deck_profile_inference.build import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
