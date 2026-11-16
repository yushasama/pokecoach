"""Durable, content-addressed league membership for research workflows.

This module is deliberately separate from :mod:`deepmew.curriculum.league`.
The legacy ``League`` remains the active in-memory curriculum object; this v1
registry is an additive persistence primitive and has no runtime integration.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping


REGISTRY_SCHEMA_VERSION = "deepmew.league_registry.v1"
_MEMBER_ID_PREFIX = "lm1-"
_SNAPSHOT_ID_PREFIX = "ls1-"
_PROMOTION_ID_PREFIX = "lp1-"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class LeagueRegistryError(RuntimeError):
    """Base error for durable league-registry operations."""


class LeagueArtifactMissingError(LeagueRegistryError, FileNotFoundError):
    """A registered checkpoint, bundle, or deck does not exist."""


class LeagueHashMismatchError(LeagueRegistryError):
    """Artifact bytes no longer match their immutable registered hash."""


class LeagueRegistryConflictError(LeagueRegistryError):
    """A write or identity transition would silently replace existing state."""


class LeagueRegistryValidationError(LeagueRegistryError, ValueError):
    """Registry data violates the v1 schema or lifecycle invariants."""


class LeagueRole(str, Enum):
    """Closed v1 role vocabulary for durable league members."""

    CHAMPION = "champion"
    CANDIDATE = "candidate"
    HISTORICAL = "historical"
    MIRROR = "mirror"
    HEURISTIC = "heuristic"
    CALIBRATION = "calibration"
    EXPERIMENTAL = "experimental"


class LeagueArtifactKind(str, Enum):
    """Closed loader contract for a member's registered policy artifact."""

    CHECKPOINT = "checkpoint"
    BUNDLE = "bundle"
    NAMED_POLICY = "named_policy"


class MirrorSemantics(str, Enum):
    """Whether a mirror represents the current champion or an older one."""

    CURRENT_SNAPSHOT = "current_snapshot"
    HISTORICAL_SNAPSHOT = "historical_snapshot"


def utc_now_iso() -> str:
    """Return a canonical, timezone-aware UTC timestamp."""
    now = datetime.now(timezone.utc).replace(microsecond=0)

    return now.isoformat().replace("+00:00", "Z")


def canonical_timestamp(value: str) -> str:
    """Validate and normalize a timestamp to UTC with a trailing ``Z``."""
    if not value:
        raise LeagueRegistryValidationError("timestamp must not be empty")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise LeagueRegistryValidationError(f"invalid ISO-8601 timestamp {value!r}") from exc
    if parsed.tzinfo is None:
        raise LeagueRegistryValidationError(f"timestamp must include a timezone: {value!r}")
    normalized = parsed.astimezone(timezone.utc)

    return normalized.isoformat().replace("+00:00", "Z")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise LeagueRegistryValidationError(
        f"provenance values must be JSON-compatible, got {type(value).__name__}"
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]

    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _thaw_json(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _content_id(prefix: str, identity: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()

    return prefix + digest[:32]


def _validate_sha256(value: str, *, field_name: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise LeagueRegistryValidationError(
            f"{field_name} must be a lowercase 64-character SHA-256, got {value!r}"
        )


def _resolve_existing_path(path: str | Path, *, purpose: str) -> Path:
    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise LeagueArtifactMissingError(f"missing {purpose}: {candidate}") from exc
    if resolved.is_symlink():
        raise LeagueRegistryValidationError(f"{purpose} may not be a symbolic link: {resolved}")

    return resolved


def _update_hash_from_file(hasher: Any, path: Path) -> None:
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)


def sha256_path(path: str | Path, *, purpose: str = "artifact") -> str:
    """Hash a file or a directory tree deterministically.

    Directory hashes include each relative path and file length as well as file
    bytes, so renames and boundary changes cannot alias the same tree digest.
    """
    resolved = _resolve_existing_path(path, purpose=purpose)
    if resolved.is_file():
        hasher = hashlib.sha256()
        _update_hash_from_file(hasher, resolved)

        return hasher.hexdigest()
    if not resolved.is_dir():
        raise LeagueRegistryValidationError(f"{purpose} is neither a file nor directory: {resolved}")
    hasher = hashlib.sha256(b"deepmew.directory.v1\0")
    entries = sorted(resolved.rglob("*"), key=lambda item: item.relative_to(resolved).as_posix())
    for entry in entries:
        if entry.is_symlink():
            raise LeagueRegistryValidationError(f"{purpose} tree contains a symbolic link: {entry}")
        if not entry.is_file():
            continue
        relative_path = entry.relative_to(resolved)
        if "__pycache__" in relative_path.parts or entry.suffix.lower() in {".pyc", ".pyo"}:
            continue
        relative = relative_path.as_posix().encode("utf-8")
        hasher.update(len(relative).to_bytes(8, "big"))
        hasher.update(relative)
        hasher.update(entry.stat().st_size.to_bytes(8, "big"))
        _update_hash_from_file(hasher, entry)

    return hasher.hexdigest()


def _verified_hash(path: Path, expected: str | None, *, purpose: str) -> str:
    actual = sha256_path(path, purpose=purpose)
    if expected is not None and actual != expected.lower():
        raise LeagueHashMismatchError(
            f"{purpose} hash mismatch for {path}: expected {expected.lower()}, actual {actual}"
        )

    return actual


@dataclass(frozen=True)
class LeagueMemberV1:
    """One durable member whose identity is derived from bytes and provenance."""

    league_member_id: str
    role: LeagueRole
    artifact_kind: LeagueArtifactKind
    artifact_path: str
    artifact_sha256: str
    deck_path: str
    deck_sha256: str
    policy_family: str
    feature_policy_id: str
    created_at: str
    adopted_at: str | None = None
    parent_member_id: str | None = None
    training_run_id: str | None = None
    source_round: int | None = None
    source_generation: int | None = None
    deck_identity: str | None = None
    deck_version: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    active: bool = True
    retirement_reason: str | None = None
    behavior_fingerprint_version: str | None = None
    label: str = ""
    mirror_semantics: MirrorSemantics | None = None
    mirror_of_member_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", LeagueRole(self.role))
        object.__setattr__(self, "artifact_kind", LeagueArtifactKind(self.artifact_kind))
        if self.mirror_semantics is not None:
            object.__setattr__(self, "mirror_semantics", MirrorSemantics(self.mirror_semantics))
        object.__setattr__(self, "provenance", _freeze_json(self.provenance))
        object.__setattr__(self, "created_at", canonical_timestamp(self.created_at))
        if self.adopted_at is not None:
            object.__setattr__(self, "adopted_at", canonical_timestamp(self.adopted_at))
        _validate_sha256(self.artifact_sha256, field_name="artifact_sha256")
        _validate_sha256(self.deck_sha256, field_name="deck_sha256")
        if not self.policy_family:
            raise LeagueRegistryValidationError("policy_family must not be empty")
        if not self.feature_policy_id.strip():
            raise LeagueRegistryValidationError("feature_policy_id must not be empty")
        if self.source_round is not None and self.source_round < 0:
            raise LeagueRegistryValidationError("source_round must be non-negative")
        if self.source_generation is not None and self.source_generation < 0:
            raise LeagueRegistryValidationError("source_generation must be non-negative")
        if self.active and self.retirement_reason is not None:
            raise LeagueRegistryValidationError("an active member may not have a retirement_reason")
        if not self.active and not self.retirement_reason:
            raise LeagueRegistryValidationError("an inactive member must have a retirement_reason")
        if self.role is LeagueRole.MIRROR:
            if self.mirror_semantics is None or self.mirror_of_member_id is None:
                raise LeagueRegistryValidationError(
                    "a mirror member requires mirror_semantics and mirror_of_member_id"
                )
        elif self.mirror_semantics is not None or self.mirror_of_member_id is not None:
            raise LeagueRegistryValidationError(
                "mirror_semantics and mirror_of_member_id are valid only for role='mirror'"
            )
        expected_id = derive_league_member_id(self)
        if self.league_member_id != expected_id:
            raise LeagueRegistryValidationError(
                f"league_member_id {self.league_member_id!r} does not match immutable identity {expected_id!r}"
            )

    def immutable_identity(self) -> dict[str, Any]:
        """Return all fields that may never change for this member ID."""
        return {
            "artifact_kind": self.artifact_kind.value,
            "artifact_sha256": self.artifact_sha256,
            "deck_sha256": self.deck_sha256,
            "policy_family": self.policy_family,
            "feature_policy_id": self.feature_policy_id,
            "parent_member_id": self.parent_member_id,
            "training_run_id": self.training_run_id,
            "source_round": self.source_round,
            "source_generation": self.source_generation,
            "deck_identity": self.deck_identity,
            "deck_version": self.deck_version,
            "provenance": _thaw_json(self.provenance),
            "mirror_of_member_id": self.mirror_of_member_id,
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "league_member_id": self.league_member_id,
            "role": self.role.value,
            "artifact_kind": self.artifact_kind.value,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "deck_path": self.deck_path,
            "deck_sha256": self.deck_sha256,
            "policy_family": self.policy_family,
            "feature_policy_id": self.feature_policy_id,
            "created_at": self.created_at,
            "adopted_at": self.adopted_at,
            "parent_member_id": self.parent_member_id,
            "training_run_id": self.training_run_id,
            "source_round": self.source_round,
            "source_generation": self.source_generation,
            "deck_identity": self.deck_identity,
            "deck_version": self.deck_version,
            "provenance": _thaw_json(self.provenance),
            "active": self.active,
            "retirement_reason": self.retirement_reason,
            "behavior_fingerprint_version": self.behavior_fingerprint_version,
            "label": self.label,
            "mirror_semantics": self.mirror_semantics.value if self.mirror_semantics else None,
            "mirror_of_member_id": self.mirror_of_member_id,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "LeagueMemberV1":
        return cls(
            league_member_id=str(data["league_member_id"]),
            role=LeagueRole(str(data["role"])),
            artifact_kind=LeagueArtifactKind(str(data["artifact_kind"])),
            artifact_path=str(data["artifact_path"]),
            artifact_sha256=str(data["artifact_sha256"]),
            deck_path=str(data["deck_path"]),
            deck_sha256=str(data["deck_sha256"]),
            policy_family=str(data["policy_family"]),
            feature_policy_id=str(data["feature_policy_id"]),
            created_at=str(data["created_at"]),
            adopted_at=str(data["adopted_at"]) if data.get("adopted_at") is not None else None,
            parent_member_id=(
                str(data["parent_member_id"]) if data.get("parent_member_id") is not None else None
            ),
            training_run_id=(
                str(data["training_run_id"]) if data.get("training_run_id") is not None else None
            ),
            source_round=int(data["source_round"]) if data.get("source_round") is not None else None,
            source_generation=(
                int(data["source_generation"]) if data.get("source_generation") is not None else None
            ),
            deck_identity=str(data["deck_identity"]) if data.get("deck_identity") is not None else None,
            deck_version=str(data["deck_version"]) if data.get("deck_version") is not None else None,
            provenance=data.get("provenance") or {},
            active=bool(data.get("active", True)),
            retirement_reason=(
                str(data["retirement_reason"]) if data.get("retirement_reason") is not None else None
            ),
            behavior_fingerprint_version=(
                str(data["behavior_fingerprint_version"])
                if data.get("behavior_fingerprint_version") is not None
                else None
            ),
            label=str(data.get("label", "")),
            mirror_semantics=(
                MirrorSemantics(str(data["mirror_semantics"]))
                if data.get("mirror_semantics") is not None
                else None
            ),
            mirror_of_member_id=(
                str(data["mirror_of_member_id"])
                if data.get("mirror_of_member_id") is not None
                else None
            ),
        )


def derive_league_member_id(member: LeagueMemberV1) -> str:
    """Derive a stable ID without mutable role, label, status, path, or times."""
    return _content_id(_MEMBER_ID_PREFIX, member.immutable_identity())


def create_league_member(
    *,
    role: LeagueRole,
    artifact_kind: LeagueArtifactKind,
    artifact_path: str | Path,
    deck_path: str | Path,
    policy_family: str,
    feature_policy_id: str,
    created_at: str | None = None,
    adopted_at: str | None = None,
    parent_member_id: str | None = None,
    training_run_id: str | None = None,
    source_round: int | None = None,
    source_generation: int | None = None,
    deck_identity: str | None = None,
    deck_version: str | None = None,
    provenance: Mapping[str, Any] | None = None,
    behavior_fingerprint_version: str | None = None,
    label: str = "",
    mirror_semantics: MirrorSemantics | None = None,
    mirror_of_member_id: str | None = None,
    expected_artifact_sha256: str | None = None,
    expected_deck_sha256: str | None = None,
) -> LeagueMemberV1:
    """Create a member only after both artifact identities have been verified."""
    artifact = _resolve_existing_path(artifact_path, purpose="policy artifact")
    deck = _resolve_existing_path(deck_path, purpose="deck artifact")
    artifact_sha256 = _verified_hash(
        artifact,
        expected_artifact_sha256,
        purpose="policy artifact",
    )
    deck_sha256 = _verified_hash(deck, expected_deck_sha256, purpose="deck artifact")
    timestamp = canonical_timestamp(created_at or utc_now_iso())
    identity = {
        "artifact_kind": LeagueArtifactKind(artifact_kind).value,
        "artifact_sha256": artifact_sha256,
        "deck_sha256": deck_sha256,
        "policy_family": policy_family,
        "feature_policy_id": feature_policy_id,
        "parent_member_id": parent_member_id,
        "training_run_id": training_run_id,
        "source_round": source_round,
        "source_generation": source_generation,
        "deck_identity": deck_identity,
        "deck_version": deck_version,
        "provenance": _thaw_json(_freeze_json(provenance or {})),
        "mirror_of_member_id": mirror_of_member_id,
    }
    member_id = _content_id(_MEMBER_ID_PREFIX, identity)

    return LeagueMemberV1(
        league_member_id=member_id,
        role=role,
        artifact_kind=artifact_kind,
        artifact_path=str(artifact),
        artifact_sha256=artifact_sha256,
        deck_path=str(deck),
        deck_sha256=deck_sha256,
        policy_family=policy_family,
        feature_policy_id=feature_policy_id,
        created_at=timestamp,
        adopted_at=adopted_at,
        parent_member_id=parent_member_id,
        training_run_id=training_run_id,
        source_round=source_round,
        source_generation=source_generation,
        deck_identity=deck_identity,
        deck_version=deck_version,
        provenance=provenance or {},
        behavior_fingerprint_version=behavior_fingerprint_version,
        label=label,
        mirror_semantics=mirror_semantics,
        mirror_of_member_id=mirror_of_member_id,
    )


@dataclass(frozen=True)
class LeagueSnapshotMemberV1:
    """The frozen membership and byte identity recorded in one snapshot."""

    league_member_id: str
    role: LeagueRole
    artifact_kind: LeagueArtifactKind
    artifact_path: str
    artifact_sha256: str
    deck_path: str
    deck_sha256: str
    policy_family: str
    feature_policy_id: str
    active: bool
    mirror_semantics: MirrorSemantics | None = None
    mirror_of_member_id: str | None = None

    @classmethod
    def from_member(cls, member: LeagueMemberV1) -> "LeagueSnapshotMemberV1":
        return cls(
            league_member_id=member.league_member_id,
            role=member.role,
            artifact_kind=member.artifact_kind,
            artifact_path=member.artifact_path,
            artifact_sha256=member.artifact_sha256,
            deck_path=member.deck_path,
            deck_sha256=member.deck_sha256,
            policy_family=member.policy_family,
            feature_policy_id=member.feature_policy_id,
            active=member.active,
            mirror_semantics=member.mirror_semantics,
            mirror_of_member_id=member.mirror_of_member_id,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "league_member_id": self.league_member_id,
            "role": self.role.value,
            "artifact_kind": self.artifact_kind.value,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "deck_path": self.deck_path,
            "deck_sha256": self.deck_sha256,
            "policy_family": self.policy_family,
            "feature_policy_id": self.feature_policy_id,
            "active": self.active,
            "mirror_semantics": self.mirror_semantics.value if self.mirror_semantics else None,
            "mirror_of_member_id": self.mirror_of_member_id,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "LeagueSnapshotMemberV1":
        return cls(
            league_member_id=str(data["league_member_id"]),
            role=LeagueRole(str(data["role"])),
            artifact_kind=LeagueArtifactKind(str(data["artifact_kind"])),
            artifact_path=str(data["artifact_path"]),
            artifact_sha256=str(data["artifact_sha256"]),
            deck_path=str(data["deck_path"]),
            deck_sha256=str(data["deck_sha256"]),
            policy_family=str(data["policy_family"]),
            feature_policy_id=str(data["feature_policy_id"]),
            active=bool(data["active"]),
            mirror_semantics=(
                MirrorSemantics(str(data["mirror_semantics"]))
                if data.get("mirror_semantics") is not None
                else None
            ),
            mirror_of_member_id=(
                str(data["mirror_of_member_id"])
                if data.get("mirror_of_member_id") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class LeagueSnapshotV1:
    snapshot_id: str
    created_at: str
    reason: str
    members: tuple[LeagueSnapshotMemberV1, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "created_at", canonical_timestamp(self.created_at))
        expected = derive_snapshot_id(self.members)
        if self.snapshot_id != expected:
            raise LeagueRegistryValidationError(
                f"snapshot_id {self.snapshot_id!r} does not match frozen membership {expected!r}"
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at,
            "reason": self.reason,
            "members": [member.to_json() for member in self.members],
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "LeagueSnapshotV1":
        return cls(
            snapshot_id=str(data["snapshot_id"]),
            created_at=str(data["created_at"]),
            reason=str(data.get("reason", "")),
            members=tuple(LeagueSnapshotMemberV1.from_json(item) for item in data.get("members", [])),
        )


def derive_snapshot_id(members: tuple[LeagueSnapshotMemberV1, ...]) -> str:
    frozen = [member.to_json() for member in sorted(members, key=lambda item: item.league_member_id)]

    return _content_id(_SNAPSHOT_ID_PREFIX, {"members": frozen})


@dataclass(frozen=True)
class PromotionRecordV1:
    """Durable adoption or rejection receipt; rejection enrolls no member."""

    promotion_id: str
    candidate_member_id: str
    candidate_artifact_sha256: str
    candidate_deck_sha256: str
    incumbent_member_id: str | None
    accepted: bool
    reason: str
    evaluation_panel_id: str
    decided_at: str
    snapshot_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "decided_at", canonical_timestamp(self.decided_at))
        _validate_sha256(self.candidate_artifact_sha256, field_name="candidate_artifact_sha256")
        _validate_sha256(self.candidate_deck_sha256, field_name="candidate_deck_sha256")
        if not self.reason:
            raise LeagueRegistryValidationError("promotion reason must not be empty")
        if not self.evaluation_panel_id:
            raise LeagueRegistryValidationError("evaluation_panel_id must not be empty")
        expected = derive_promotion_id(self)
        if self.promotion_id != expected:
            raise LeagueRegistryValidationError(
                f"promotion_id {self.promotion_id!r} does not match decision identity {expected!r}"
            )

    def identity(self) -> dict[str, Any]:
        return {
            "candidate_member_id": self.candidate_member_id,
            "candidate_artifact_sha256": self.candidate_artifact_sha256,
            "candidate_deck_sha256": self.candidate_deck_sha256,
            "incumbent_member_id": self.incumbent_member_id,
            "accepted": self.accepted,
            "reason": self.reason,
            "evaluation_panel_id": self.evaluation_panel_id,
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "promotion_id": self.promotion_id,
            **self.identity(),
            "decided_at": self.decided_at,
            "snapshot_id": self.snapshot_id,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "PromotionRecordV1":
        return cls(
            promotion_id=str(data["promotion_id"]),
            candidate_member_id=str(data["candidate_member_id"]),
            candidate_artifact_sha256=str(data["candidate_artifact_sha256"]),
            candidate_deck_sha256=str(data["candidate_deck_sha256"]),
            incumbent_member_id=(
                str(data["incumbent_member_id"])
                if data.get("incumbent_member_id") is not None
                else None
            ),
            accepted=bool(data["accepted"]),
            reason=str(data["reason"]),
            evaluation_panel_id=str(data["evaluation_panel_id"]),
            decided_at=str(data["decided_at"]),
            snapshot_id=str(data["snapshot_id"]) if data.get("snapshot_id") is not None else None,
        )


def derive_promotion_id(record: PromotionRecordV1) -> str:
    return _content_id(_PROMOTION_ID_PREFIX, record.identity())


def create_promotion_record(
    *,
    candidate: LeagueMemberV1,
    incumbent_member_id: str | None,
    accepted: bool,
    reason: str,
    evaluation_panel_id: str,
    decided_at: str,
    snapshot_id: str | None = None,
) -> PromotionRecordV1:
    """Construct a content-addressed promotion receipt."""
    identity = {
        "candidate_member_id": candidate.league_member_id,
        "candidate_artifact_sha256": candidate.artifact_sha256,
        "candidate_deck_sha256": candidate.deck_sha256,
        "incumbent_member_id": incumbent_member_id,
        "accepted": accepted,
        "reason": reason,
        "evaluation_panel_id": evaluation_panel_id,
    }
    promotion_id = _content_id(_PROMOTION_ID_PREFIX, identity)

    return PromotionRecordV1(
        promotion_id=promotion_id,
        candidate_member_id=candidate.league_member_id,
        candidate_artifact_sha256=candidate.artifact_sha256,
        candidate_deck_sha256=candidate.deck_sha256,
        incumbent_member_id=incumbent_member_id,
        accepted=accepted,
        reason=reason,
        evaluation_panel_id=evaluation_panel_id,
        decided_at=decided_at,
        snapshot_id=snapshot_id,
    )


@dataclass(frozen=True)
class LeagueRegistryV1:
    """Immutable whole-file state; update functions return a new revision."""

    revision: int = 0
    members: tuple[LeagueMemberV1, ...] = ()
    snapshots: tuple[LeagueSnapshotV1, ...] = ()
    promotion_records: tuple[PromotionRecordV1, ...] = ()
    schema_version: str = REGISTRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REGISTRY_SCHEMA_VERSION:
            raise LeagueRegistryValidationError(
                f"unsupported registry schema {self.schema_version!r}; expected {REGISTRY_SCHEMA_VERSION!r}"
            )
        if self.revision < 0:
            raise LeagueRegistryValidationError("registry revision must be non-negative")
        validate_registry_structure(self)

    def member(self, member_id: str) -> LeagueMemberV1 | None:
        return next((member for member in self.members if member.league_member_id == member_id), None)

    def active_champion(self) -> LeagueMemberV1 | None:
        champions = [
            member for member in self.members if member.active and member.role is LeagueRole.CHAMPION
        ]

        return champions[0] if champions else None

    def promotion_record(self, promotion_id: str) -> PromotionRecordV1 | None:
        return next(
            (record for record in self.promotion_records if record.promotion_id == promotion_id),
            None,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "revision": self.revision,
            "members": [member.to_json() for member in self.members],
            "snapshots": [snapshot.to_json() for snapshot in self.snapshots],
            "promotion_records": [record.to_json() for record in self.promotion_records],
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "LeagueRegistryV1":
        return cls(
            schema_version=str(data.get("schema_version", "")),
            revision=int(data.get("revision", 0)),
            members=tuple(LeagueMemberV1.from_json(item) for item in data.get("members", [])),
            snapshots=tuple(LeagueSnapshotV1.from_json(item) for item in data.get("snapshots", [])),
            promotion_records=tuple(
                PromotionRecordV1.from_json(item) for item in data.get("promotion_records", [])
            ),
        )


def validate_registry_structure(registry: LeagueRegistryV1) -> None:
    member_ids = [member.league_member_id for member in registry.members]
    if len(member_ids) != len(set(member_ids)):
        raise LeagueRegistryValidationError("registry contains duplicate league_member_id values")
    snapshot_ids = [snapshot.snapshot_id for snapshot in registry.snapshots]
    if len(snapshot_ids) != len(set(snapshot_ids)):
        raise LeagueRegistryValidationError("registry contains duplicate snapshot_id values")
    promotion_ids = [record.promotion_id for record in registry.promotion_records]
    if len(promotion_ids) != len(set(promotion_ids)):
        raise LeagueRegistryValidationError("registry contains duplicate promotion_id values")
    champions = [
        member for member in registry.members if member.active and member.role is LeagueRole.CHAMPION
    ]
    if len(champions) > 1:
        raise LeagueRegistryValidationError("registry may contain at most one active champion")
    by_id = {member.league_member_id: member for member in registry.members}
    active_champion = champions[0] if champions else None
    for snapshot in registry.snapshots:
        frozen_ids = [member.league_member_id for member in snapshot.members]
        if len(frozen_ids) != len(set(frozen_ids)):
            raise LeagueRegistryValidationError(
                f"snapshot {snapshot.snapshot_id!r} contains duplicate member IDs"
            )
        for frozen in snapshot.members:
            live = by_id.get(frozen.league_member_id)
            if live is None:
                raise LeagueRegistryValidationError(
                    f"snapshot {snapshot.snapshot_id!r} references missing member "
                    f"{frozen.league_member_id!r}"
                )
            frozen_immutable = (
                frozen.artifact_kind,
                frozen.artifact_sha256,
                frozen.deck_sha256,
                frozen.policy_family,
                frozen.feature_policy_id,
            )
            live_immutable = (
                live.artifact_kind,
                live.artifact_sha256,
                live.deck_sha256,
                live.policy_family,
                live.feature_policy_id,
            )
            if frozen_immutable != live_immutable:
                raise LeagueRegistryValidationError(
                    f"snapshot {snapshot.snapshot_id!r} immutable identity does not match "
                    f"member {frozen.league_member_id!r}"
                )
    for member in registry.members:
        if member.role is not LeagueRole.MIRROR:
            continue
        source = by_id.get(member.mirror_of_member_id or "")
        if source is None:
            raise LeagueRegistryValidationError(
                f"mirror {member.league_member_id!r} references missing member {member.mirror_of_member_id!r}"
            )
        if member.artifact_sha256 != source.artifact_sha256 or member.deck_sha256 != source.deck_sha256:
            raise LeagueRegistryValidationError(
                f"mirror {member.league_member_id!r} bytes do not match source {source.league_member_id!r}"
            )
        if member.mirror_semantics is MirrorSemantics.CURRENT_SNAPSHOT:
            if active_champion is None or source.league_member_id != active_champion.league_member_id:
                raise LeagueRegistryValidationError(
                    f"current mirror {member.league_member_id!r} does not reference the active champion"
                )
        elif source.role is not LeagueRole.HISTORICAL:
            raise LeagueRegistryValidationError(
                f"historical mirror {member.league_member_id!r} must reference a historical member"
            )


def validate_registry_artifacts(registry: LeagueRegistryV1) -> None:
    """Re-hash every live registry member and fail on missing or changed bytes."""
    for member in registry.members:
        actual_artifact = sha256_path(member.artifact_path, purpose="policy artifact")
        if actual_artifact != member.artifact_sha256:
            raise LeagueHashMismatchError(
                f"policy artifact hash mismatch for {member.league_member_id}: "
                f"expected {member.artifact_sha256}, actual {actual_artifact}"
            )
        actual_deck = sha256_path(member.deck_path, purpose="deck artifact")
        if actual_deck != member.deck_sha256:
            raise LeagueHashMismatchError(
                f"deck artifact hash mismatch for {member.league_member_id}: "
                f"expected {member.deck_sha256}, actual {actual_deck}"
            )


def register_member(
    registry: LeagueRegistryV1,
    member: LeagueMemberV1,
) -> tuple[LeagueRegistryV1, LeagueMemberV1, bool]:
    """Register once; an identical resume returns the original record unchanged."""
    existing = registry.member(member.league_member_id)
    if existing is not None:
        if existing.immutable_identity() != member.immutable_identity():
            raise LeagueRegistryConflictError(
                f"member ID {member.league_member_id!r} cannot be repointed to new immutable identity"
            )
        if existing.role is not member.role or existing.active != member.active:
            raise LeagueRegistryConflictError(
                f"member {member.league_member_id!r} already exists with lifecycle "
                f"role={existing.role.value!r}, active={existing.active}; use an explicit lifecycle transition"
            )

        return registry, existing, False
    if member.role is LeagueRole.CHAMPION and registry.active_champion() is not None:
        raise LeagueRegistryConflictError("an active champion already exists; use the promotion transaction")
    updated = replace(registry, revision=registry.revision + 1, members=registry.members + (member,))

    return updated, member, True


def replace_member_lifecycle(
    registry: LeagueRegistryV1,
    updated_member: LeagueMemberV1,
) -> LeagueRegistryV1:
    """Replace mutable lifecycle fields while proving immutable identity stayed fixed."""
    current = registry.member(updated_member.league_member_id)
    if current is None:
        raise LeagueRegistryConflictError(f"unknown member {updated_member.league_member_id!r}")
    if current.immutable_identity() != updated_member.immutable_identity():
        raise LeagueRegistryConflictError(
            f"member {updated_member.league_member_id!r} immutable identity may not change"
        )
    if current.role is not LeagueRole.CHAMPION and updated_member.role is LeagueRole.CHAMPION:
        raise LeagueRegistryConflictError(
            "a member may become champion only through the promotion transaction"
        )
    if current.role is LeagueRole.CHAMPION and updated_member.role is not LeagueRole.CHAMPION:
        raise LeagueRegistryConflictError(
            "a champion role transition must preserve history through the promotion transaction"
        )
    members = tuple(
        updated_member if member.league_member_id == updated_member.league_member_id else member
        for member in registry.members
    )

    return replace(registry, revision=registry.revision + 1, members=members)


def freeze_snapshot(
    registry: LeagueRegistryV1,
    *,
    reason: str,
    created_at: str | None = None,
    active_only: bool = True,
) -> tuple[LeagueRegistryV1, LeagueSnapshotV1, bool]:
    """Freeze exact membership, roles, paths, and hashes into an immutable snapshot."""
    selected = [member for member in registry.members if member.active or not active_only]
    members = tuple(
        LeagueSnapshotMemberV1.from_member(member)
        for member in sorted(selected, key=lambda item: item.league_member_id)
    )
    snapshot = LeagueSnapshotV1(
        snapshot_id=derive_snapshot_id(members),
        created_at=created_at or utc_now_iso(),
        reason=reason,
        members=members,
    )
    existing = next(
        (item for item in registry.snapshots if item.snapshot_id == snapshot.snapshot_id),
        None,
    )
    if existing is not None:
        return registry, existing, False
    updated = replace(
        registry,
        revision=registry.revision + 1,
        snapshots=registry.snapshots + (snapshot,),
    )

    return updated, snapshot, True


def add_promotion_record(
    registry: LeagueRegistryV1,
    record: PromotionRecordV1,
) -> tuple[LeagueRegistryV1, PromotionRecordV1, bool]:
    existing = registry.promotion_record(record.promotion_id)
    if existing is not None:
        if existing.identity() != record.identity():
            raise LeagueRegistryConflictError(
                f"promotion record {record.promotion_id!r} has conflicting contents"
            )
        if record.snapshot_id is not None and existing.snapshot_id != record.snapshot_id:
            raise LeagueRegistryConflictError(
                f"promotion record {record.promotion_id!r} cannot point to a different snapshot"
            )

        return registry, existing, False
    updated = replace(
        registry,
        revision=registry.revision + 1,
        promotion_records=registry.promotion_records + (record,),
    )

    return updated, record, True


def load_registry(path: str | Path, *, verify_artifacts: bool = True) -> LeagueRegistryV1:
    registry_path = Path(path)
    if not registry_path.is_file():
        raise LeagueArtifactMissingError(f"missing league registry: {registry_path}")
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LeagueRegistryValidationError(f"cannot read league registry {registry_path}: {exc}") from exc
    registry = LeagueRegistryV1.from_json(data)
    if verify_artifacts:
        validate_registry_artifacts(registry)

    return registry


def _atomic_write_registry(path: Path, registry: LeagueRegistryV1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = json.dumps(registry.to_json(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_registry_atomic(
    path: str | Path,
    registry: LeagueRegistryV1,
    *,
    expected_revision: int | None = None,
) -> None:
    """Atomically replace a registry, with optional optimistic revision checking."""
    registry_path = Path(path)
    if registry_path.exists():
        current = load_registry(registry_path, verify_artifacts=False)
        if current.to_json() == registry.to_json():
            return
        if expected_revision is None:
            raise LeagueRegistryConflictError(
                f"refusing to overwrite existing registry {registry_path} without expected_revision"
            )
        if current.revision != expected_revision:
            raise LeagueRegistryConflictError(
                f"registry revision changed: expected {expected_revision}, found {current.revision}"
            )
    elif expected_revision not in (None, -1):
        raise LeagueRegistryConflictError(
            f"registry {registry_path} does not exist for expected revision {expected_revision}"
        )
    _atomic_write_registry(registry_path, registry)


def initialize_registry(path: str | Path) -> LeagueRegistryV1:
    """Create an empty v1 registry, or resume an existing valid one."""
    registry_path = Path(path)
    if registry_path.exists():
        return load_registry(registry_path)
    registry = LeagueRegistryV1()
    save_registry_atomic(registry_path, registry)

    return registry


def update_registry_atomic(
    path: str | Path,
    update: Callable[[LeagueRegistryV1], LeagueRegistryV1],
) -> LeagueRegistryV1:
    """Load, transform, validate, and atomically persist one registry revision."""
    registry_path = Path(path)
    current = load_registry(registry_path)
    updated = update(current)
    validate_registry_structure(updated)
    validate_registry_artifacts(updated)
    save_registry_atomic(registry_path, updated, expected_revision=current.revision)

    return updated


def register_member_atomic(
    path: str | Path,
    member: LeagueMemberV1,
) -> tuple[LeagueRegistryV1, LeagueMemberV1, bool]:
    """Persist a registration with resume-safe duplicate handling."""
    outcome: tuple[LeagueMemberV1, bool] | None = None

    def apply(current: LeagueRegistryV1) -> LeagueRegistryV1:
        nonlocal outcome
        updated, registered, created = register_member(current, member)
        outcome = (registered, created)

        return updated

    registry = update_registry_atomic(path, apply)
    assert outcome is not None

    return registry, outcome[0], outcome[1]
