"""Apply one explicit research promotion-gate decision to a v1 league registry.

This CLI never computes the decision. It verifies the named candidate and
incumbent against the durable registry, then atomically records an accepted or
rejected result supplied by an already completed evaluation gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.curriculum.league_promotion import (  # noqa: E402
    PromotionRequestV1,
    apply_promotion,
    apply_promotion_atomic,
)
from deepmew.curriculum.league_registry import LeagueRole, load_registry  # noqa: E402


RECEIPT_SCHEMA_VERSION = "deepmew.league_promotion_cli_receipt.v1"


def build_request(
    registry_path: str | Path,
    *,
    candidate_member_id: str,
    expected_champion_id: str,
    accepted: bool,
    reason: str,
    evaluation_panel_id: str,
    decided_at: str,
) -> PromotionRequestV1:
    registry = load_registry(registry_path)
    candidate = registry.member(candidate_member_id)
    if candidate is None:
        raise ValueError(f"candidate member is not registered: {candidate_member_id}")
    if candidate.role is not LeagueRole.CANDIDATE:
        matching_record = next(
            (
                record
                for record in registry.promotion_records
                if record.candidate_member_id == candidate_member_id
                and record.incumbent_member_id == expected_champion_id
                and record.accepted is accepted
                and record.reason == reason
                and record.evaluation_panel_id == evaluation_panel_id
            ),
            None,
        )
        if matching_record is None or not accepted or candidate.role is not LeagueRole.CHAMPION:
            raise ValueError(
                f"promotion input {candidate_member_id} has role {candidate.role.value!r}, "
                "and no identical accepted receipt permits an idempotent replay"
            )
        candidate = replace(candidate, role=LeagueRole.CANDIDATE, adopted_at=None)

    return PromotionRequestV1(
        candidate=candidate,
        accepted=accepted,
        reason=reason,
        evaluation_panel_id=evaluation_panel_id,
        expected_champion_id=expected_champion_id,
        decided_at=decided_at,
    )


def run_decision(
    registry_path: str | Path,
    *,
    candidate_member_id: str,
    expected_champion_id: str,
    accepted: bool,
    reason: str,
    evaluation_panel_id: str,
    decided_at: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    target = Path(registry_path).resolve()
    request = build_request(
        target,
        candidate_member_id=candidate_member_id,
        expected_champion_id=expected_champion_id,
        accepted=accepted,
        reason=reason,
        evaluation_panel_id=evaluation_panel_id,
        decided_at=decided_at,
    )
    if dry_run:
        outcome = apply_promotion(load_registry(target), request)
    else:
        outcome = apply_promotion_atomic(target, request)
    champion = outcome.registry.active_champion()

    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "registry_path": str(target),
        "dry_run": dry_run,
        "accepted": outcome.record.accepted,
        "changed": outcome.changed,
        "promotion_id": outcome.record.promotion_id,
        "candidate_member_id": outcome.record.candidate_member_id,
        "incumbent_member_id": outcome.record.incumbent_member_id,
        "active_champion_id": champion.league_member_id if champion is not None else None,
        "snapshot_id": outcome.record.snapshot_id,
        "evaluation_panel_id": outcome.record.evaluation_panel_id,
        "reason": outcome.record.reason,
        "registry_revision": outcome.registry.revision,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--candidate-member-id", required=True)
    parser.add_argument("--expected-champion-id", required=True)
    decision = parser.add_mutually_exclusive_group(required=True)
    decision.add_argument("--accept", action="store_true")
    decision.add_argument("--reject", action="store_true")
    parser.add_argument("--reason", required=True)
    parser.add_argument("--evaluation-panel-id", required=True)
    parser.add_argument("--decided-at", required=True)
    parser.add_argument("--dry-run", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        receipt = run_decision(
            args.registry,
            candidate_member_id=args.candidate_member_id,
            expected_champion_id=args.expected_champion_id,
            accepted=bool(args.accept),
            reason=args.reason,
            evaluation_panel_id=args.evaluation_panel_id,
            decided_at=args.decided_at,
            dry_run=bool(args.dry_run),
        )
    except Exception as exc:
        parser.exit(2, f"error: {type(exc).__name__}: {exc}\n")
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
