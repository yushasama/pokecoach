"""Thin CLI wrapper for StrategicActionScorer v0.

Scores candidate action records (each deck's own answer/attacker/engine cards)
against meta-representative opponent threat surfaces, emitting reason-tagged,
fully-explained action scores under ``.research/strategic_action_scoring_v0/``.

Offline/training-only: no gameplay code, no engine import, no submission change, no
trained labels, and no win rates.

Examples
--------
    python scripts/build_strategic_action_scoring_v0.py
    python scripts/build_strategic_action_scoring_v0.py --decklists path/to/deck_zoo_decklists.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.strategic_action_scoring.build import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
