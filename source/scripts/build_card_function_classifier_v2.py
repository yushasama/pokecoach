"""Thin CLI wrapper for CardFunctionClassifier v2.

Classifies every unique card in ``EN_Card_Data.csv`` into function tags and writes
the v2 CSV + report under ``.research/card_function_classifier_v2/``.

Offline/training-only: no gameplay code, no engine import, no submission change.

Examples
--------
    python scripts/build_card_function_classifier_v2.py
    python scripts/build_card_function_classifier_v2.py --out-dir .research/card_function_classifier_v2
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.card_functions.build import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
