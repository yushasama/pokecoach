"""Seed or inspect a durable v1 league registry from an explicit JSON roster.

This CLI only registers the roles stated in the roster.  It never evaluates a
candidate, performs a promotion, adopts a checkpoint, or touches runtime files.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepmew.curriculum.league_registry import (  # noqa: E402
    LeagueMemberV1,
    LeagueRegistryV1,
    create_league_member,
    freeze_snapshot,
    load_registry,
    register_member,
    save_registry_atomic,
)


RECEIPT_SCHEMA_VERSION = "deepmew.league_registry_cli_receipt.v1"


def load_roster(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    """Load the roster envelope and return its member mappings."""
    roster_path = Path(path)
    try:
        data = json.loads(roster_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"missing league roster: {roster_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid league roster JSON {roster_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("league roster root must be a JSON object")
    raw_members = data.get("members")
    if not isinstance(raw_members, list):
        raise ValueError("league roster must contain a `members` list")
    members: list[Mapping[str, Any]] = []
    for index, raw_member in enumerate(raw_members):
        if not isinstance(raw_member, dict):
            raise ValueError(f"league roster member {index} must be a JSON object")
        members.append(raw_member)

    return tuple(members)


def create_roster_members(entries: tuple[Mapping[str, Any], ...]) -> tuple[LeagueMemberV1, ...]:
    """Verify every roster artifact before any registry write can occur."""
    members: list[LeagueMemberV1] = []
    for index, entry in enumerate(entries):
        try:
            member = create_league_member(**dict(entry))
        except Exception as exc:
            raise type(exc)(f"roster member {index}: {exc}") from exc
        members.append(member)

    return tuple(members)


def _load_starting_registry(path: Path) -> tuple[LeagueRegistryV1, bool]:
    if path.exists():
        return load_registry(path), True

    return LeagueRegistryV1(), False


def _merge_roster(
    registry: LeagueRegistryV1,
    members: tuple[LeagueMemberV1, ...],
) -> tuple[LeagueRegistryV1, tuple[str, ...], tuple[str, ...]]:
    updated = registry
    created_ids: list[str] = []
    existing_ids: list[str] = []
    for member in members:
        updated, registered, created = register_member(updated, member)
        target = created_ids if created else existing_ids
        target.append(registered.league_member_id)

    return updated, tuple(created_ids), tuple(existing_ids)


def _role_counts(registry: LeagueRegistryV1) -> dict[str, int]:
    counts = Counter(member.role.value for member in registry.members)

    return {role: counts[role] for role in sorted(counts)}


def run_registry_roster(
    *,
    roster_path: str | Path,
    registry_path: str | Path,
    snapshot_reason: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate and merge one roster, committing at most one atomic replacement."""
    roster = Path(roster_path).resolve()
    target = Path(registry_path).resolve()
    entries = load_roster(roster)
    members = create_roster_members(entries)
    current, existed = _load_starting_registry(target)
    updated, created_ids, existing_ids = _merge_roster(current, members)
    snapshot_id: str | None = None
    snapshot_created = False
    if snapshot_reason is not None:
        if not snapshot_reason.strip():
            raise ValueError("snapshot reason must not be empty")
        updated, snapshot, snapshot_created = freeze_snapshot(
            updated,
            reason=snapshot_reason,
        )
        snapshot_id = snapshot.snapshot_id
    if not dry_run:
        expected_revision = current.revision if existed else None
        save_registry_atomic(target, updated, expected_revision=expected_revision)
    champion = updated.active_champion()

    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "roster_path": str(roster),
        "registry_path": str(target),
        "dry_run": dry_run,
        "input_members": len(entries),
        "created_members": len(created_ids),
        "existing_members": len(existing_ids),
        "total_members": len(updated.members),
        "revision_before": current.revision if existed else None,
        "revision_after": updated.revision,
        "created_member_ids": list(created_ids),
        "existing_member_ids": list(existing_ids),
        "active_champion_id": champion.league_member_id if champion is not None else None,
        "role_counts": _role_counts(updated),
        "snapshot_id": snapshot_id,
        "snapshot_created": snapshot_created,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", required=True, help="JSON object containing a `members` list.")
    parser.add_argument("--registry", required=True, help="Target durable v1 registry JSON path.")
    parser.add_argument(
        "--snapshot-reason",
        help="Freeze the resulting active membership under this audit reason.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and inspect the merged registry without writing it.",
    )

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        receipt = run_registry_roster(
            roster_path=args.roster,
            registry_path=args.registry,
            snapshot_reason=args.snapshot_reason,
            dry_run=bool(args.dry_run),
        )
    except Exception as exc:
        parser.exit(2, f"error: {type(exc).__name__}: {exc}\n")
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
