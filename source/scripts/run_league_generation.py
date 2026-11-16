"""Initialize, inspect, or advance a research-only league generation receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.curriculum.league_generation import (  # noqa: E402
    initialize_generation,
    load_generation,
    mark_candidate_ready,
    mark_data_ready,
    mark_decision_recorded,
    mark_evaluation_ready,
    save_generation_atomic,
)


def _write_advanced(path: str, prior, advanced) -> None:
    save_generation_atomic(path, advanced, expected_revision=prior.revision)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("init")
    init.add_argument("--state", required=True)
    init.add_argument("--generation-id", required=True)
    init.add_argument("--registry", required=True)
    init.add_argument("--snapshot-id", required=True)
    init.add_argument("--champion-member-id", required=True)
    init.add_argument("--feature-policy-id", required=True)
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--state", required=True)
    data = subparsers.add_parser("data-ready")
    data.add_argument("--state", required=True)
    data.add_argument("--manifest", required=True)
    candidate = subparsers.add_parser("candidate-ready")
    candidate.add_argument("--state", required=True)
    candidate.add_argument("--candidate-member-id", required=True)
    candidate.add_argument("--checkpoint", required=True)
    candidate.add_argument("--train-manifest", required=True)
    evaluation = subparsers.add_parser("evaluation-ready")
    evaluation.add_argument("--state", required=True)
    evaluation.add_argument("--shared-panel", required=True)
    evaluation.add_argument("--payoff", required=True)
    decision = subparsers.add_parser("decision-recorded")
    decision.add_argument("--state", required=True)
    decision.add_argument("--promotion-id", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        state = initialize_generation(
            generation_id=args.generation_id,
            registry_path=args.registry,
            snapshot_id=args.snapshot_id,
            champion_member_id=args.champion_member_id,
            feature_policy_id=args.feature_policy_id,
        )
        save_generation_atomic(args.state, state, expected_revision=None)
    elif args.command == "inspect":
        state = load_generation(args.state)
    else:
        state = load_generation(args.state)
        if args.command == "data-ready":
            advanced = mark_data_ready(state, args.manifest)
        elif args.command == "candidate-ready":
            advanced = mark_candidate_ready(
                state,
                candidate_member_id=args.candidate_member_id,
                checkpoint_path=args.checkpoint,
                train_manifest_path=args.train_manifest,
            )
        elif args.command == "evaluation-ready":
            advanced = mark_evaluation_ready(
                state,
                shared_panel_path=args.shared_panel,
                payoff_path=args.payoff,
            )
        else:
            advanced = mark_decision_recorded(state, promotion_id=args.promotion_id)
        _write_advanced(args.state, state, advanced)
        state = advanced
    print(json.dumps({
        "generation_id": state.generation_id,
        "phase": state.phase.value,
        "revision": state.revision,
        "state_fingerprint": state.state_fingerprint,
        "candidate_member_id": state.candidate_member_id,
        "evaluation_panel_id": state.evaluation_panel_id,
        "artifacts": sorted(state.artifacts),
    }, sort_keys=True, separators=(",", ":")))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
