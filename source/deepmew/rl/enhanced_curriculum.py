"""Fail-closed local curriculum scheduling for enhanced research trainers.

This module is intentionally independent of the existing terminal-league
control runner. Each sampling cell binds an exact policy identity, deck hash,
and anchored strength bucket before an explicit bucket multiplier is applied.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


CONFIG_SCHEMA_VERSION = "deepmew.enhanced_curriculum_config.v1"
STATE_SCHEMA_VERSION = "deepmew.enhanced_curriculum_state.v1"
RECEIPT_SCHEMA_VERSION = "deepmew.enhanced_curriculum_receipt.v1"

_CELL_FIELDS = frozenset({
    "policy_identity",
    "deck_hash",
    "strength_bucket",
    "base_weight",
})
_STATE_FIELDS = frozenset({
    "schema_version",
    "config_sha256",
    "seed",
    "samples_drawn",
    "counts_by_cell_id",
    "rng_state",
})
_UNANCHORED_BUCKET_PREFIXES = ("unknown", "unassigned", "unverified")


class EnhancedCurriculumError(ValueError):
    """A curriculum identity, weight, receipt, or resume state failed closed."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise EnhancedCurriculumError("curriculum identity must be finite JSON data") from exc

    return rendered.encode("ascii")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _exact_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise EnhancedCurriculumError(f"{name} must be a non-empty string")
    if value != value.strip():
        raise EnhancedCurriculumError(f"{name} must not contain surrounding whitespace")

    return value


def _deck_sha256(value: Any) -> str:
    text = _exact_text(value, name="deck_hash").lower().removeprefix("sha256:")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise EnhancedCurriculumError("deck_hash must be a SHA-256 identity")

    return text


def _anchored_strength_bucket(value: Any) -> str:
    bucket = _exact_text(value, name="strength_bucket")
    normalized = bucket.casefold()
    if any(
        normalized == prefix
        or normalized.startswith(f"{prefix}_")
        or normalized.startswith(f"{prefix}-")
        or normalized.startswith(f"{prefix}:")
        for prefix in _UNANCHORED_BUCKET_PREFIXES
    ):
        raise EnhancedCurriculumError(
            f"strength_bucket {bucket!r} is not an anchored curriculum bucket"
        )

    return bucket


def _positive_finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise EnhancedCurriculumError(f"{name} must be finite and positive")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise EnhancedCurriculumError(f"{name} must be finite and positive") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise EnhancedCurriculumError(f"{name} must be finite and positive")

    return number


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EnhancedCurriculumError(f"{name} must be a non-negative integer")

    return value


@dataclass(frozen=True, order=True)
class CurriculumCellKey:
    """Exact scheduler key for one policy/deck/strength cell."""

    policy_identity: str
    deck_hash: str
    strength_bucket: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "policy_identity",
            _exact_text(self.policy_identity, name="policy_identity"),
        )
        object.__setattr__(self, "deck_hash", _deck_sha256(self.deck_hash))
        object.__setattr__(
            self,
            "strength_bucket",
            _anchored_strength_bucket(self.strength_bucket),
        )

    @property
    def cell_id(self) -> str:
        return _sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, str]:
        return {
            "policy_identity": self.policy_identity,
            "deck_hash": self.deck_hash,
            "strength_bucket": self.strength_bucket,
        }


@dataclass(frozen=True)
class CurriculumCell:
    """One immutable declared base-mixture entry."""

    key: CurriculumCellKey
    base_weight: float

    def __post_init__(self) -> None:
        if not isinstance(self.key, CurriculumCellKey):
            raise EnhancedCurriculumError("curriculum cell key must be CurriculumCellKey")
        object.__setattr__(
            self,
            "base_weight",
            _positive_finite(self.base_weight, name="base_weight"),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CurriculumCell":
        if not isinstance(value, Mapping) or set(value) != _CELL_FIELDS:
            raise EnhancedCurriculumError("curriculum cell fields are incomplete or unexpected")

        return cls(
            key=CurriculumCellKey(
                policy_identity=value["policy_identity"],
                deck_hash=value["deck_hash"],
                strength_bucket=value["strength_bucket"],
            ),
            base_weight=value["base_weight"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.key.to_dict(),
            "base_weight": self.base_weight,
        }


@dataclass(frozen=True)
class CurriculumSample:
    """One deterministic weighted draw plus its exact scheduler evidence."""

    sample_number: int
    cell_index: int
    cell: CurriculumCell
    strength_multiplier: float
    normalized_weight: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_number": self.sample_number,
            "cell_index": self.cell_index,
            "cell_id": self.cell.key.cell_id,
            "weighting_key": self.cell.key.to_dict(),
            "base_weight": self.cell.base_weight,
            "strength_multiplier": self.strength_multiplier,
            "normalized_weight": self.normalized_weight,
        }


class EnhancedCurriculumScheduler:
    """Deterministic weighted sampler over exact, anchored curriculum cells."""

    def __init__(
        self,
        cells: Sequence[CurriculumCell | Mapping[str, Any]],
        *,
        strength_bucket_multipliers: Mapping[str, float],
        seed: int,
    ) -> None:
        if isinstance(cells, (str, bytes)) or not isinstance(cells, Sequence) or not cells:
            raise EnhancedCurriculumError("curriculum requires a non-empty cell sequence")
        normalized_cells = tuple(
            value if isinstance(value, CurriculumCell) else CurriculumCell.from_mapping(value)
            for value in cells
        )
        keys = [cell.key for cell in normalized_cells]
        if len(set(keys)) != len(keys):
            raise EnhancedCurriculumError("curriculum contains a duplicate exact cell key")
        normalized_seed = _nonnegative_int(seed, name="seed")
        multipliers = self._validated_multipliers(
            strength_bucket_multipliers,
            expected_buckets={cell.key.strength_bucket for cell in normalized_cells},
        )
        base_total = sum(cell.base_weight for cell in normalized_cells)
        base_normalized = tuple(cell.base_weight / base_total for cell in normalized_cells)
        weighted = tuple(
            cell.base_weight * multipliers[cell.key.strength_bucket]
            for cell in normalized_cells
        )
        weighted_total = sum(weighted)
        normalized = tuple(value / weighted_total for value in weighted)

        self.cells = normalized_cells
        self.strength_bucket_multipliers = dict(sorted(multipliers.items()))
        self.seed = normalized_seed
        self.base_normalized_weights = base_normalized
        self.weighted_unnormalized = weighted
        self.normalized_weights = normalized
        self._index_by_key = {cell.key: index for index, cell in enumerate(self.cells)}
        self._rng = random.Random(self.seed)
        self._samples_drawn = 0
        self._counts_by_cell_id = {cell.key.cell_id: 0 for cell in self.cells}
        config = {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "cells": [cell.to_dict() for cell in self.cells],
            "strength_bucket_multipliers": self.strength_bucket_multipliers,
            "seed": self.seed,
        }
        self.config_sha256 = _sha256_json(config)

    @staticmethod
    def _validated_multipliers(
        value: Mapping[str, float],
        *,
        expected_buckets: set[str],
    ) -> dict[str, float]:
        if not isinstance(value, Mapping) or not value:
            raise EnhancedCurriculumError("strength bucket multipliers must be a mapping")
        normalized: dict[str, float] = {}
        for raw_bucket, raw_multiplier in value.items():
            bucket = _anchored_strength_bucket(raw_bucket)
            if bucket in normalized:
                raise EnhancedCurriculumError(f"duplicate strength multiplier for {bucket!r}")
            normalized[bucket] = _positive_finite(
                raw_multiplier,
                name=f"strength multiplier for {bucket!r}",
            )
        observed_buckets = set(normalized)
        if observed_buckets != expected_buckets:
            missing = sorted(expected_buckets - observed_buckets)
            unexpected = sorted(observed_buckets - expected_buckets)
            raise EnhancedCurriculumError(
                "strength multiplier buckets disagree with curriculum cells: "
                f"missing={missing}, unexpected={unexpected}"
            )

        return normalized

    @property
    def samples_drawn(self) -> int:
        return self._samples_drawn

    def weight_for_key(self, key: CurriculumCellKey) -> float:
        if not isinstance(key, CurriculumCellKey):
            raise TypeError("key must be CurriculumCellKey")
        index = self._index_by_key.get(key)
        if index is None:
            raise EnhancedCurriculumError("exact curriculum cell key is not scheduled")

        return self.normalized_weights[index]

    def sample(self) -> CurriculumSample:
        threshold = self._rng.random()
        cumulative = 0.0
        selected = len(self.cells) - 1
        for index, weight in enumerate(self.normalized_weights):
            cumulative += weight
            if threshold <= cumulative:
                selected = index
                break
        self._samples_drawn += 1
        cell = self.cells[selected]
        self._counts_by_cell_id[cell.key.cell_id] += 1

        return CurriculumSample(
            sample_number=self._samples_drawn,
            cell_index=selected,
            cell=cell,
            strength_multiplier=self.strength_bucket_multipliers[cell.key.strength_bucket],
            normalized_weight=self.normalized_weights[selected],
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "config_sha256": self.config_sha256,
            "seed": self.seed,
            "samples_drawn": self._samples_drawn,
            "counts_by_cell_id": dict(sorted(self._counts_by_cell_id.items())),
            "rng_state": copy.deepcopy(self._rng.getstate()),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or set(state) != _STATE_FIELDS:
            raise EnhancedCurriculumError("curriculum state fields are incomplete or unexpected")
        if state["schema_version"] != STATE_SCHEMA_VERSION:
            raise EnhancedCurriculumError("curriculum state schema is incompatible")
        if state["config_sha256"] != self.config_sha256 or state["seed"] != self.seed:
            raise EnhancedCurriculumError("curriculum state identity is incompatible")
        samples_drawn = _nonnegative_int(state["samples_drawn"], name="samples_drawn")
        raw_counts = state["counts_by_cell_id"]
        if not isinstance(raw_counts, Mapping) or set(raw_counts) != set(self._counts_by_cell_id):
            raise EnhancedCurriculumError("curriculum sample-count identities are incompatible")
        counts = {
            cell_id: _nonnegative_int(count, name=f"sample count for {cell_id}")
            for cell_id, count in raw_counts.items()
        }
        if sum(counts.values()) != samples_drawn:
            raise EnhancedCurriculumError("curriculum sample counts disagree with samples_drawn")
        candidate_rng = random.Random()
        try:
            candidate_rng.setstate(copy.deepcopy(state["rng_state"]))
        except (TypeError, ValueError) as exc:
            raise EnhancedCurriculumError("curriculum RNG state is invalid") from exc

        self._rng = candidate_rng
        self._samples_drawn = samples_drawn
        self._counts_by_cell_id = counts

    def receipt(self) -> dict[str, Any]:
        base_weight_sum = sum(cell.base_weight for cell in self.cells)
        weighted_sum = sum(self.weighted_unnormalized)
        l1_delta = sum(
            abs(base - effective)
            for base, effective in zip(
                self.base_normalized_weights,
                self.normalized_weights,
                strict=True,
            )
        )
        cell_rows = []
        for index, cell in enumerate(self.cells):
            multiplier = self.strength_bucket_multipliers[cell.key.strength_bucket]
            weighting_record = {
                "weighting_key": cell.key.to_dict(),
                "base_weight": cell.base_weight,
                "base_normalized_weight": self.base_normalized_weights[index],
                "strength_multiplier": multiplier,
                "weighted_unnormalized": self.weighted_unnormalized[index],
                "effective_normalized_weight": self.normalized_weights[index],
            }
            cell_rows.append({
                "cell_index": index,
                "cell_id": cell.key.cell_id,
                **weighting_record,
                "weighting_record_sha256": _sha256_json(weighting_record),
                "samples_drawn": self._counts_by_cell_id[cell.key.cell_id],
            })

        return {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "config_sha256": self.config_sha256,
            "seed": self.seed,
            "base_weight_sum": base_weight_sum,
            "weighted_unnormalized_sum": weighted_sum,
            "effective_distribution_l1_delta_from_base": l1_delta,
            "samples_drawn": self._samples_drawn,
            "cells": cell_rows,
            "deck_identity_participated_in_scheduler_weighting": True,
            "strength_bucket_participated_in_scheduler_weighting": True,
            "strength_bucket_multipliers_changed_effective_distribution": l1_delta > 1.0e-15,
            "proofs": {
                "base_mixture_weights_preserved": True,
                "exact_policy_identity_in_every_weighting_key": True,
                "exact_deck_hash_in_every_weighting_key": True,
                "anchored_strength_bucket_in_every_weighting_key": True,
                "explicit_strength_multiplier_applied_to_every_cell": True,
                "distinct_policy_identity_count": len({
                    cell.key.policy_identity for cell in self.cells
                }),
                "distinct_deck_hash_count": len({cell.key.deck_hash for cell in self.cells}),
                "distinct_strength_bucket_count": len({
                    cell.key.strength_bucket for cell in self.cells
                }),
            },
        }
