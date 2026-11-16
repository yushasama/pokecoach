"""R18: measure the depth-0 R4-F prior piloting the bd27 Mega Lucario deck vs the
reference-agent suite. Search (GreedyV) was reconfirmed to not beat this prior, so
this is the honest 'can we beat ref agents >=75%' number for the ship candidate.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_r15_greedyv_on_r4f as r15  # noqa: E402
from deepmew.rl.metadata import normalized_deck_hash  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=24, help="games per reference opponent")
    ap.add_argument("--seed", type=int, default=915002)
    ap.add_argument("--out", default=str(ROOT / ".research/r18/bd27_prior_eval.json"))
    args = ap.parse_args(argv)

    # R4-F prior expects 304-dim second-order-augmented features; apply the same
    # runner-local patch build_greedyv_on_r4f uses, else the model crashes on dim.
    import run_unified_r4_ablation as r4  # noqa: E402
    r4.build_live_features = r15._r4f_build_live_features

    bd27 = [int(x) for x in tarfile.open(sorted(glob.glob(str(ROOT / ".research/submissions/*bd27*.tar.gz")))[0])
            .extractfile("deck.csv").read().decode().split()]
    prior = r15.R4FShortlistAdapter(Path(r15.DEFAULT_R4F), "reference_abomasnow")
    prior.deck = list(bd27)
    prior.deck_hash = normalized_deck_hash(bd27)
    prior.label = "R4F_prior_bd27"

    wr = r15.compare_winrates([("R4F_prior_bd27", prior)], games_per_opponent=int(args.games), seed=int(args.seed))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(wr, indent=2, default=str), encoding="utf-8")
    print(json.dumps(wr, indent=2, default=str)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
