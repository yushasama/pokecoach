"""Run the chronological actor-visible R0/R1 opponent-response ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepmew.opponent_response.prospective import run_prospective_response_ablation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--canonical-catalog", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run_prospective_response_ablation(
        source_root=args.source_root,
        canonical_catalog=args.canonical_catalog,
        output_dir=args.output_dir,
        entrypoint=Path(__file__),
    )
    print(json.dumps({
        "verdict": result["verdict"],
        "dataset": {
            "rows": result["dataset"]["rows"],
            "games": result["dataset"]["games"],
            "train_examples": result["dataset"]["train_examples"],
            "evaluation_examples": result["dataset"]["evaluation_examples"],
            "eligible_control_pass_rows": result["dataset"]["eligible_control_pass_rows"],
            "complete_control_pass_games": result["dataset"]["complete_control_pass_games"],
        },
        "marginal_contribution": (
            result["evaluation"]["chronological_T1"]["metrics"]["marginal_contribution"]
        ),
        "q_gate": result["q_gate"],
    }, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
