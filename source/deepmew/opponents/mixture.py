from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from deepmew.agents import AgentSpec, canonical_agent_label, is_experimental, parse_agent_spec


class MixtureValidationError(ValueError):
    pass


@dataclass(frozen=True)
class MixtureEntry:
    agent_spec: AgentSpec
    weight: float
    deck_id: str | None = None
    deck_hash: str | None = None

    @classmethod
    def from_text(
        cls,
        agent_spec: str,
        weight: float,
        *,
        deck_id: str | None = None,
        deck_hash: str | None = None,
    ) -> "MixtureEntry":
        return cls(parse_agent_spec(agent_spec), weight, deck_id=deck_id, deck_hash=deck_hash)

    @property
    def canonical_label(self) -> str:
        return canonical_agent_label(self.agent_spec)

    def to_manifest(self, *, normalized_weight: float | None = None) -> dict[str, Any]:
        return {
            "agent_spec": self.agent_spec.label,
            "agent_raw": self.agent_spec.raw,
            "canonical_label": self.canonical_label,
            "weight": self.weight,
            "normalized_weight": normalized_weight,
            "experimental": is_experimental(self.agent_spec),
            "deck_id": self.deck_id,
            "deck_hash": self.deck_hash,
        }


@dataclass(frozen=True)
class OpponentMixture:
    name: str
    version: str
    entries: tuple[MixtureEntry, ...]
    description: str
    intended_use: str
    diagnostic_only: bool = True
    warning: str | None = None

    @property
    def mixture_id(self) -> str:
        return f"{self.name}_{self.version}"

    def validate(self) -> "OpponentMixture":
        if not self.name.strip():
            raise MixtureValidationError("mixture name is required")
        if not self.version.strip():
            raise MixtureValidationError("mixture version is required")
        if not self.entries:
            raise MixtureValidationError("mixture must contain at least one entry")
        for entry in self.entries:
            if entry.weight is None:
                raise MixtureValidationError(f"missing weight for {entry.agent_spec.label}")
            if not isinstance(entry.weight, (int, float)) or not math.isfinite(float(entry.weight)):
                raise MixtureValidationError(f"invalid weight for {entry.agent_spec.label}: {entry.weight!r}")
            if float(entry.weight) <= 0:
                raise MixtureValidationError(f"full-support mixture requires positive weight for {entry.agent_spec.label}")
        return self

    def normalized_weights(self) -> tuple[float, ...]:
        self.validate()
        total = sum(float(entry.weight) for entry in self.entries)
        if total <= 0:
            raise MixtureValidationError("mixture weights must sum to a positive value")
        return tuple(float(entry.weight) / total for entry in self.entries)

    def to_manifest(self) -> dict[str, Any]:
        weights = self.normalized_weights()
        return {
            "mixture_id": self.mixture_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "intended_use": self.intended_use,
            "diagnostic_only": self.diagnostic_only,
            "warning": self.warning,
            "entries": [
                entry.to_manifest(normalized_weight=weight)
                for entry, weight in zip(self.entries, weights)
            ],
        }
