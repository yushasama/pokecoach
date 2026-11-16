"""Explicit champion adoption transactions for the durable v1 league registry.

Evaluation remains outside this module.  Callers must provide the gate decision,
panel identity, reason, and expected incumbent; this code only applies the
result without silently enrolling rejected candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from deepmew.curriculum.league_registry import (
    LeagueMemberV1,
    LeagueRegistryConflictError,
    LeagueRegistryV1,
    LeagueRole,
    LeagueSnapshotV1,
    MirrorSemantics,
    PromotionRecordV1,
    add_promotion_record,
    create_promotion_record,
    freeze_snapshot,
    update_registry_atomic,
    validate_registry_artifacts,
)


@dataclass(frozen=True)
class PromotionRequestV1:
    """A gate result plus the exact incumbent state it was evaluated against."""

    candidate: LeagueMemberV1
    accepted: bool
    reason: str
    evaluation_panel_id: str
    expected_champion_id: str | None
    decided_at: str

    def __post_init__(self) -> None:
        if self.candidate.role is not LeagueRole.CANDIDATE:
            raise LeagueRegistryConflictError(
                f"promotion input must have role='candidate', got {self.candidate.role.value!r}"
            )
        if not self.reason:
            raise LeagueRegistryConflictError("promotion request requires a non-empty reason")
        if not self.evaluation_panel_id:
            raise LeagueRegistryConflictError(
                "promotion request requires a non-empty evaluation_panel_id"
            )


@dataclass(frozen=True)
class PromotionOutcomeV1:
    registry: LeagueRegistryV1
    record: PromotionRecordV1
    snapshot: LeagueSnapshotV1 | None
    changed: bool


def _validate_candidate_bytes(candidate: LeagueMemberV1) -> None:
    isolated = LeagueRegistryV1(members=(candidate,))
    validate_registry_artifacts(isolated)


def _find_snapshot(
    registry: LeagueRegistryV1,
    snapshot_id: str | None,
) -> LeagueSnapshotV1 | None:
    if snapshot_id is None:
        return None

    return next(
        (snapshot for snapshot in registry.snapshots if snapshot.snapshot_id == snapshot_id),
        None,
    )


def _existing_outcome(
    registry: LeagueRegistryV1,
    record: PromotionRecordV1,
) -> PromotionOutcomeV1 | None:
    existing = registry.promotion_record(record.promotion_id)
    if existing is None:
        return None
    if existing.identity() != record.identity():
        raise LeagueRegistryConflictError(
            f"promotion record {record.promotion_id!r} has conflicting immutable contents"
        )

    return PromotionOutcomeV1(
        registry=registry,
        record=existing,
        snapshot=_find_snapshot(registry, existing.snapshot_id),
        changed=False,
    )


def _validate_expected_champion(
    registry: LeagueRegistryV1,
    expected_champion_id: str | None,
) -> LeagueMemberV1 | None:
    champion = registry.active_champion()
    actual_id = champion.league_member_id if champion is not None else None
    if actual_id != expected_champion_id:
        raise LeagueRegistryConflictError(
            f"promotion incumbent changed: expected {expected_champion_id!r}, found {actual_id!r}"
        )

    return champion


def _historical_member(member: LeagueMemberV1) -> LeagueMemberV1:
    return replace(member, role=LeagueRole.HISTORICAL)


def _historical_mirror(member: LeagueMemberV1, incumbent_id: str) -> LeagueMemberV1:
    if (
        member.role is LeagueRole.MIRROR
        and member.mirror_semantics is MirrorSemantics.CURRENT_SNAPSHOT
        and member.mirror_of_member_id == incumbent_id
    ):
        return replace(member, mirror_semantics=MirrorSemantics.HISTORICAL_SNAPSHOT)

    return member


def _promoted_member(candidate: LeagueMemberV1, adopted_at: str) -> LeagueMemberV1:
    return replace(
        candidate,
        role=LeagueRole.CHAMPION,
        adopted_at=adopted_at,
        active=True,
        retirement_reason=None,
    )


def _adopt_candidate(
    registry: LeagueRegistryV1,
    request: PromotionRequestV1,
    incumbent: LeagueMemberV1 | None,
) -> tuple[LeagueRegistryV1, LeagueSnapshotV1]:
    existing_candidate = registry.member(request.candidate.league_member_id)
    if existing_candidate is not None:
        if existing_candidate.role is not LeagueRole.CANDIDATE:
            raise LeagueRegistryConflictError(
                f"promotion member {request.candidate.league_member_id!r} has role "
                f"{existing_candidate.role.value!r}, expected 'candidate'"
            )
        if existing_candidate.immutable_identity() != request.candidate.immutable_identity():
            raise LeagueRegistryConflictError(
                f"candidate {request.candidate.league_member_id!r} immutable identity changed"
            )
    incumbent_id = incumbent.league_member_id if incumbent is not None else None
    members: list[LeagueMemberV1] = []
    for member in registry.members:
        if member is incumbent:
            updated = _historical_member(member)
        elif existing_candidate is not None and member is existing_candidate:
            updated = _promoted_member(member, request.decided_at)
        else:
            updated = member
        if incumbent_id is not None:
            updated = _historical_mirror(updated, incumbent_id)
        members.append(updated)
    if existing_candidate is None:
        members.append(_promoted_member(request.candidate, request.decided_at))
    adopted = replace(
        registry,
        revision=registry.revision + 1,
        members=tuple(members),
    )
    snapshotted, snapshot, _ = freeze_snapshot(
        adopted,
        reason=f"promotion:{request.evaluation_panel_id}",
        created_at=request.decided_at,
    )

    return snapshotted, snapshot


def apply_promotion(
    registry: LeagueRegistryV1,
    request: PromotionRequestV1,
) -> PromotionOutcomeV1:
    """Apply one accepted/rejected gate result as an idempotent state transition."""
    _validate_candidate_bytes(request.candidate)
    provisional_record = create_promotion_record(
        candidate=request.candidate,
        incumbent_member_id=request.expected_champion_id,
        accepted=request.accepted,
        reason=request.reason,
        evaluation_panel_id=request.evaluation_panel_id,
        decided_at=request.decided_at,
    )
    existing = _existing_outcome(registry, provisional_record)
    if existing is not None:
        return existing
    incumbent = _validate_expected_champion(registry, request.expected_champion_id)
    if not request.accepted:
        existing_candidate = registry.member(request.candidate.league_member_id)
        if existing_candidate is not None and (
            existing_candidate.role is not LeagueRole.CANDIDATE
            or existing_candidate.immutable_identity() != request.candidate.immutable_identity()
        ):
            raise LeagueRegistryConflictError(
                f"rejected member {request.candidate.league_member_id!r} is not the same candidate"
            )
        updated, record, _ = add_promotion_record(registry, provisional_record)

        return PromotionOutcomeV1(
            registry=updated,
            record=record,
            snapshot=None,
            changed=True,
        )
    adopted, snapshot = _adopt_candidate(registry, request, incumbent)
    final_record = create_promotion_record(
        candidate=request.candidate,
        incumbent_member_id=request.expected_champion_id,
        accepted=True,
        reason=request.reason,
        evaluation_panel_id=request.evaluation_panel_id,
        decided_at=request.decided_at,
        snapshot_id=snapshot.snapshot_id,
    )
    updated, record, _ = add_promotion_record(adopted, final_record)

    return PromotionOutcomeV1(
        registry=updated,
        record=record,
        snapshot=snapshot,
        changed=True,
    )


def apply_promotion_atomic(
    registry_path: str | Path,
    request: PromotionRequestV1,
) -> PromotionOutcomeV1:
    """Persist all promotion effects in one atomic whole-registry replacement."""
    outcome: PromotionOutcomeV1 | None = None

    def update(current: LeagueRegistryV1) -> LeagueRegistryV1:
        nonlocal outcome
        outcome = apply_promotion(current, request)

        return outcome.registry

    persisted = update_registry_atomic(registry_path, update)
    assert outcome is not None

    return replace(outcome, registry=persisted)
