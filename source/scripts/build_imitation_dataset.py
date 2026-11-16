"""Build the K4 imitation dataset + behavior-cloning baseline eval.

Joins real ladder action rows (.research/kaggle_ingest) to K2A archetype labels
and K3 deck-zoo splits, exports BC examples, bucket summaries, and a held-out
count-table action-category accuracy eval. Offline/training-only; the submission
runtime and default policy are untouched.

Examples
--------
    python scripts/build_imitation_dataset.py
    python scripts/build_imitation_dataset.py --meta-kg <k2a_run> --deck-zoo <k3_run> --suffix k4
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.imitation.build import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
