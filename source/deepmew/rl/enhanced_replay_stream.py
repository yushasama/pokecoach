"""Deterministic bounded-memory cursor for the complete accepted replay corpus."""

from __future__ import annotations

import hashlib
import gzip
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from deepmew.rl.enhanced_replay_derived import (
    DerivedReplayReferenceIndex,
    REFERENCE_ROW_SCHEMA_VERSION,
)


STREAM_SCHEMA_VERSION = "deepmew.enhanced_replay_stream.v1"
CURSOR_SCHEMA_VERSION = "deepmew.enhanced_replay_cursor.v1"


class ReplayStreamHardStop(RuntimeError):
    """Replay identities, configuration, or cursor state are inconsistent."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    return hashlib.sha256(payload.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class ReplayStreamConfig:
    cache_manifest_sha256: str
    split_manifest_sha256: str
    action_schema_identity: str
    observation_schema_identity: str
    shuffle_seed: int
    epoch_count: int = 1

    def validate(self) -> None:
        for name in ("cache_manifest_sha256", "split_manifest_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ReplayStreamHardStop(f"{name} must be a SHA-256 digest")
        if not self.action_schema_identity or not self.observation_schema_identity:
            raise ReplayStreamHardStop("replay schema identities must be explicit")
        if isinstance(self.shuffle_seed, bool) or not isinstance(self.shuffle_seed, int):
            raise ReplayStreamHardStop("shuffle_seed must be an integer")
        if isinstance(self.epoch_count, bool) or self.epoch_count < 1:
            raise ReplayStreamHardStop("epoch_count must be positive")

    @property
    def identity_sha256(self) -> str:
        self.validate()

        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class ReplayCursor:
    schema_version: str
    stream_identity_sha256: str
    epoch: int
    shard_cursor: int
    row_cursor: int
    global_unique_row_count: int
    total_consumed_rows: int
    optimizer_step_count: int
    actor_weight_total: float
    forced_rows: int
    nonforced_rows: int
    all_forced_actor_skipped: int

    @classmethod
    def initial(cls, stream_identity_sha256: str) -> "ReplayCursor":
        return cls(
            schema_version=CURSOR_SCHEMA_VERSION,
            stream_identity_sha256=stream_identity_sha256,
            epoch=0,
            shard_cursor=0,
            row_cursor=0,
            global_unique_row_count=0,
            total_consumed_rows=0,
            optimizer_step_count=0,
            actor_weight_total=0.0,
            forced_rows=0,
            nonforced_rows=0,
            all_forced_actor_skipped=0,
        )

    def validate(self, stream_identity_sha256: str, *, epoch_count: int, shard_count: int) -> None:
        if self.schema_version != CURSOR_SCHEMA_VERSION:
            raise ReplayStreamHardStop("replay cursor schema mismatch")
        if self.stream_identity_sha256 != stream_identity_sha256:
            raise ReplayStreamHardStop("replay cursor configuration or corpus identity drifted")
        if not 0 <= self.epoch <= epoch_count:
            raise ReplayStreamHardStop("replay cursor epoch is invalid")
        if not 0 <= self.shard_cursor <= shard_count or self.row_cursor < 0:
            raise ReplayStreamHardStop("replay cursor position is invalid")
        counts = (
            self.global_unique_row_count,
            self.total_consumed_rows,
            self.optimizer_step_count,
            self.forced_rows,
            self.nonforced_rows,
            self.all_forced_actor_skipped,
        )
        if any(isinstance(value, bool) or value < 0 for value in counts):
            raise ReplayStreamHardStop("replay cursor counters must be non-negative")
        if self.forced_rows + self.nonforced_rows != self.total_consumed_rows:
            raise ReplayStreamHardStop("replay cursor forced accounting is inconsistent")
        if self.actor_weight_total != float(self.nonforced_rows):
            raise ReplayStreamHardStop("replay cursor actor-weight accounting is inconsistent")

    def with_batch(self, legal_action_counts: Sequence[int], *, optimizer_stepped: bool) -> "ReplayCursor":
        forced = sum(int(value) == 1 for value in legal_action_counts)
        nonforced = len(legal_action_counts) - forced

        return ReplayCursor(
            **{
                **asdict(self),
                "global_unique_row_count": self.global_unique_row_count + len(legal_action_counts),
                "total_consumed_rows": self.total_consumed_rows + len(legal_action_counts),
                "optimizer_step_count": self.optimizer_step_count + int(optimizer_stepped),
                "actor_weight_total": self.actor_weight_total + float(nonforced),
                "forced_rows": self.forced_rows + forced,
                "nonforced_rows": self.nonforced_rows + nonforced,
                "all_forced_actor_skipped": self.all_forced_actor_skipped + int(nonforced == 0),
            }
        )


@dataclass(frozen=True)
class ReplayRowIdentity:
    epoch: int
    shard_id: str
    global_row: int

    @property
    def identity(self) -> str:
        return f"e{self.epoch}:{self.shard_id}:{self.global_row}"


@dataclass(frozen=True)
class _Shard:
    shard_id: str
    row_start: int
    row_count: int


@dataclass(frozen=True)
class _ReferenceShard:
    shard_id: str
    row_count: int
    path: Path
    sha256: str


def _coprime_multiplier(length: int, seed: int) -> tuple[int, int]:
    if length <= 1:
        return 1, 0
    candidate = 1 + (seed % (length - 1))
    while math.gcd(candidate, length) != 1:
        candidate = 1 + (candidate % (length - 1))

    return candidate, (seed // max(length, 1)) % length


class CompleteReplayStream:
    """Stream every eligible train row with O(shards + metadata) memory."""

    def __init__(
        self,
        cache_dir: str | Path,
        split_manifest_path: str | Path,
        config: ReplayStreamConfig,
    ) -> None:
        config.validate()
        self.cache_dir = Path(cache_dir).resolve(strict=True)
        self.split_manifest_path = Path(split_manifest_path).resolve(strict=True)
        manifest_path = self.cache_dir / "cache_manifest.json"
        if _sha256_file(manifest_path) != config.cache_manifest_sha256:
            raise ReplayStreamHardStop("accepted replay cache manifest hash mismatch")
        if _sha256_file(self.split_manifest_path) != config.split_manifest_sha256:
            raise ReplayStreamHardStop("accepted replay split manifest hash mismatch")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("split_manifest_sha256") != config.split_manifest_sha256:
            raise ReplayStreamHardStop("cache does not bind the declared split manifest")
        inputs = self.manifest.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            raise ReplayStreamHardStop("cache input shard inventory is missing")
        self.shards = self._build_shards(inputs)
        metadata = np.load(self.cache_dir / "meta.npz", mmap_mode="r")
        self.split_idx = metadata["split_idx"]
        self.is_target_eligible = metadata["is_target_eligible"]
        self.legal_count = metadata["legal_count"]
        split_names = list(self.manifest.get("split_names") or [])
        if "train" not in split_names:
            raise ReplayStreamHardStop("accepted replay cache has no train split")
        self.train_index = split_names.index("train")
        self.config = config
        self.stream_identity_sha256 = _canonical_sha256({
            "schema_version": STREAM_SCHEMA_VERSION,
            "config": asdict(config),
            "shards": [asdict(shard) for shard in self.shards],
        })
        eligible = (self.split_idx == self.train_index) & self.is_target_eligible
        self.eligible_train_rows = int(eligible.sum())
        expected = int((self.manifest.get("eligible_target_rows_by_split") or {}).get("train", -1))
        if self.eligible_train_rows != expected:
            raise ReplayStreamHardStop("eligible train-row count differs from the accepted manifest")

    @staticmethod
    def _build_shards(inputs: Sequence[Mapping[str, Any]]) -> tuple[_Shard, ...]:
        shards = []
        row_start = 0
        seen = set()
        for item in inputs:
            rows = int(item.get("rows", -1))
            identity = f"{item.get('source_id')}:{item.get('unit_id')}:{item.get('part')}"
            if rows <= 0 or identity in seen:
                raise ReplayStreamHardStop("cache input shard inventory is invalid or duplicated")
            seen.add(identity)
            shards.append(_Shard(identity, row_start, rows))
            row_start += rows

        return tuple(shards)

    def shard_permutation(self, epoch: int) -> tuple[int, ...]:
        values = list(range(len(self.shards)))
        random.Random(self.config.shuffle_seed + epoch * 1_000_003).shuffle(values)

        return tuple(values)

    def _row_offset(self, shard: _Shard, epoch: int, position: int) -> int:
        seed = self.config.shuffle_seed ^ (epoch * 2_000_033) ^ int(
            hashlib.sha256(shard.shard_id.encode("utf-8")).hexdigest()[:16], 16
        )
        multiplier, offset = _coprime_multiplier(shard.row_count, seed)

        return (multiplier * position + offset) % shard.row_count

    def validate_cursor(self, cursor: ReplayCursor) -> None:
        cursor.validate(
            self.stream_identity_sha256,
            epoch_count=self.config.epoch_count,
            shard_count=len(self.shards),
        )
        if cursor.epoch < self.config.epoch_count and cursor.shard_cursor < len(self.shards):
            shard = self.shards[self.shard_permutation(cursor.epoch)[cursor.shard_cursor]]
            if cursor.row_cursor > shard.row_count:
                raise ReplayStreamHardStop("replay cursor row exceeds the current shard")

    def rows(self, cursor: ReplayCursor | None = None) -> Iterator[tuple[ReplayRowIdentity, ReplayCursor]]:
        current = cursor or ReplayCursor.initial(self.stream_identity_sha256)
        self.validate_cursor(current)
        for epoch in range(current.epoch, self.config.epoch_count):
            permutation = self.shard_permutation(epoch)
            first_shard = current.shard_cursor if epoch == current.epoch else 0
            for shard_position in range(first_shard, len(permutation)):
                shard = self.shards[permutation[shard_position]]
                first_row = current.row_cursor if epoch == current.epoch and shard_position == first_shard else 0
                for row_position in range(first_row, shard.row_count):
                    global_row = shard.row_start + self._row_offset(shard, epoch, row_position)
                    next_epoch = epoch
                    next_shard = shard_position
                    next_row = row_position + 1
                    if next_row == shard.row_count:
                        next_shard += 1
                        next_row = 0
                    if next_shard == len(permutation):
                        next_epoch += 1
                        next_shard = 0
                    next_cursor = ReplayCursor(
                        **{
                            **asdict(current),
                            "epoch": next_epoch,
                            "shard_cursor": next_shard,
                            "row_cursor": next_row,
                        }
                    )
                    current = next_cursor
                    if self.split_idx[global_row] != self.train_index or not self.is_target_eligible[global_row]:
                        continue
                    identity = ReplayRowIdentity(epoch, shard.shard_id, int(global_row))
                    yield identity, current


class CompleteReferenceReplayStream:
    """Shuffle and resume every row in one immutable reference replay.

    The reference manifest owns admission and observation identities.  This
    class owns only the deterministic training order and cursor; it never
    materializes a second replay dataset.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        config: ReplayStreamConfig,
    ) -> None:
        config.validate()
        self.index = DerivedReplayReferenceIndex(manifest_path)
        manifest = self.index.manifest
        identity = manifest["identity"]
        if identity.get("cache_manifest_sha256") != config.cache_manifest_sha256:
            raise ReplayStreamHardStop("reference replay cache identity drifted")
        if self.index.cache_manifest.get("split_manifest_sha256") != config.split_manifest_sha256:
            raise ReplayStreamHardStop("reference replay split identity drifted")
        self.config = config
        self.shards = self._build_reference_shards()
        if len(self.shards) != int(manifest["shard_count"]):
            raise ReplayStreamHardStop("reference replay shard coverage drifted")
        self.row_count = sum(shard.row_count for shard in self.shards)
        if self.row_count != int(manifest["row_count"]):
            raise ReplayStreamHardStop("reference replay row coverage drifted")
        self.stream_identity_sha256 = _canonical_sha256({
            "schema_version": STREAM_SCHEMA_VERSION,
            "reference_manifest_sha256": manifest["manifest_sha256"],
            "config": asdict(config),
            "shards": [
                {
                    "shard_id": shard.shard_id,
                    "row_count": shard.row_count,
                    "sha256": shard.sha256,
                }
                for shard in self.shards
            ],
        })

    def _build_reference_shards(self) -> tuple[_ReferenceShard, ...]:
        manifest = self.index.manifest
        receipts_root = Path(manifest["receipts_root"])
        shards = []
        for ordinal, expected_receipt_sha in enumerate(
            manifest["ordered_shard_receipt_sha256s"]
        ):
            receipt_path = receipts_root / f"shard_{ordinal:05d}.receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="ascii"))
            provided = receipt.pop("receipt_sha256", None)
            if (
                provided != expected_receipt_sha
                or _canonical_sha256(receipt) != expected_receipt_sha
            ):
                raise ReplayStreamHardStop("reference replay shard receipt drifted")
            path = (
                self.index.manifest_path.parent
                / "deepmew_derived_replay_v1"
                / receipt["shard_path"]
            )
            if not path.is_file():
                path = Path(manifest["shards_root"]) / Path(receipt["shard_path"]).name
            if _sha256_file(path) != receipt["shard_sha256"]:
                raise ReplayStreamHardStop("reference replay shard hash drifted")
            shards.append(_ReferenceShard(
                shard_id=expected_receipt_sha,
                row_count=int(receipt["row_count"]),
                path=path.resolve(strict=True),
                sha256=str(receipt["shard_sha256"]),
            ))

        return tuple(shards)

    def shard_permutation(self, epoch: int) -> tuple[int, ...]:
        values = list(range(len(self.shards)))
        random.Random(self.config.shuffle_seed + epoch * 1_000_003).shuffle(values)

        return tuple(values)

    def _row_offset(self, shard: _ReferenceShard, epoch: int, position: int) -> int:
        seed = self.config.shuffle_seed ^ (epoch * 2_000_033) ^ int(
            hashlib.sha256(shard.shard_id.encode("utf-8")).hexdigest()[:16],
            16,
        )
        multiplier, offset = _coprime_multiplier(shard.row_count, seed)

        return (multiplier * position + offset) % shard.row_count

    def validate_cursor(self, cursor: ReplayCursor) -> None:
        cursor.validate(
            self.stream_identity_sha256,
            epoch_count=self.config.epoch_count,
            shard_count=len(self.shards),
        )
        if cursor.epoch < self.config.epoch_count and cursor.shard_cursor < len(self.shards):
            shard = self.shards[self.shard_permutation(cursor.epoch)[cursor.shard_cursor]]
            if cursor.row_cursor > shard.row_count:
                raise ReplayStreamHardStop("reference replay cursor exceeds its shard")

    @staticmethod
    def _load_rows(shard: _ReferenceShard) -> tuple[dict[str, Any], ...]:
        rows = []
        with gzip.open(shard.path, "rt", encoding="ascii") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("schema_version") != REFERENCE_ROW_SCHEMA_VERSION:
                    raise ReplayStreamHardStop("reference replay row schema drifted")
                rows.append(row)
        if len(rows) != shard.row_count:
            raise ReplayStreamHardStop("reference replay shard row count drifted")
        identities = [str(row.get("stable_row_id") or "") for row in rows]
        if any(not identity for identity in identities) or len(set(identities)) != len(identities):
            raise ReplayStreamHardStop("reference replay shard row identities are invalid")

        return tuple(rows)

    def rows(
        self,
        cursor: ReplayCursor | None = None,
    ) -> Iterator[tuple[dict[str, Any], ReplayCursor]]:
        current = cursor or ReplayCursor.initial(self.stream_identity_sha256)
        self.validate_cursor(current)
        for epoch in range(current.epoch, self.config.epoch_count):
            permutation = self.shard_permutation(epoch)
            first_shard = current.shard_cursor if epoch == current.epoch else 0
            for shard_position in range(first_shard, len(permutation)):
                shard = self.shards[permutation[shard_position]]
                shard_rows = self._load_rows(shard)
                first_row = (
                    current.row_cursor
                    if epoch == current.epoch and shard_position == first_shard
                    else 0
                )
                for row_position in range(first_row, shard.row_count):
                    next_epoch = epoch
                    next_shard = shard_position
                    next_row = row_position + 1
                    if next_row == shard.row_count:
                        next_shard += 1
                        next_row = 0
                    if next_shard == len(permutation):
                        next_epoch += 1
                        next_shard = 0
                    current = ReplayCursor(**{
                        **asdict(current),
                        "epoch": next_epoch,
                        "shard_cursor": next_shard,
                        "row_cursor": next_row,
                    })
                    offset = self._row_offset(shard, epoch, row_position)
                    yield shard_rows[offset], current


def advance_replay_cursor(
    current: ReplayCursor,
    position: ReplayCursor,
    legal_action_counts: Sequence[int],
    *,
    optimizer_stepped: bool,
) -> ReplayCursor:
    """Merge a yielded position with the accounting for its committed batch."""

    if current.stream_identity_sha256 != position.stream_identity_sha256:
        raise ReplayStreamHardStop("cannot merge cursors from different replay streams")
    accounted = current.with_batch(
        legal_action_counts,
        optimizer_stepped=optimizer_stepped,
    )

    return ReplayCursor(**{
        **asdict(accounted),
        "epoch": position.epoch,
        "shard_cursor": position.shard_cursor,
        "row_cursor": position.row_cursor,
    })


__all__ = [
    "CompleteReferenceReplayStream",
    "CompleteReplayStream",
    "ReplayCursor",
    "ReplayRowIdentity",
    "ReplayStreamConfig",
    "ReplayStreamHardStop",
    "advance_replay_cursor",
]
