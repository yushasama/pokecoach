"""Build K4.1 fine-grained imitation labels + K5-lite filtered/reward-weighted BC.

Loads the latest K2A meta KG and K3 deck zoo, joins real ladder action rows to
archetype/split/bucket, recovers option/card/target identity, derives the K4.1
intent / punishment / risk / last-stand pseudo-labels, and compares two K5-lite
behavior-cloning tables against an unweighted context baseline. Eval-only;
runtime/default v2.4 untouched.

Examples
--------
    python scripts/build_imitation_fine.py
    python scripts/build_imitation_fine.py --suffix k4_1
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.imitation_fine.build import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
