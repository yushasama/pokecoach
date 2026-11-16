"""Frozen local league identities and resource limits; no implicit asset download."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import orjson

from deepmew_ex.experiments.config import PROJECT_ROOT, OutputPaths
from deepmew_ex.registry.store import file_hash


@dataclass(frozen=True)
class LeagueConfig:
    runs: tuple[Path, ...]
    paths: OutputPaths
    run_id: str
    registry: Path = PROJECT_ROOT / 'league/registry.json'
    original_root: Path = Path('D:/DeepMew')
    artifact_root: Path = Path('D:/DeepMew_artifacts')
    seeds: tuple[int, ...] = (40009, 50021, 60013)
    repetitions: int = 2
    max_selections: int = 4000
    max_wall_seconds: int = 2400
    max_decision_seconds: float = 5.0
    canary: bool = False
    opponent_ids: tuple[str, ...] = ()
    temporal_checkpoint: Path | None = None
    temporal_checkpoint_sha256: str | None = None
    temporal_device: str = 'cpu'

    def validate(self) -> None:
        self.paths.validate()
        if (not self.runs and not self.canary and self.temporal_checkpoint is None) or not self.seeds or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in self.seeds):
            raise ValueError('explicit trained runs and evaluation seeds required')
        if (self.temporal_checkpoint is None) != (self.temporal_checkpoint_sha256 is None) or self.temporal_device not in ('cpu', 'cuda'):
            raise ValueError('temporal checkpoint, hash and device must be explicit')
        if not 1 <= self.repetitions <= 2 or not 1 <= self.max_selections <= 4000 or not 1 <= self.max_wall_seconds <= 2400 or not 0 < self.max_decision_seconds <= 5:
            raise ValueError('invalid frozen league resource budget')


@dataclass(frozen=True)
class OpponentBinding:
    identity: str
    label: str
    checkpoint: Path | None
    checkpoint_sha256: str | None
    deck: Path
    deck_sha256: str
    feature_policy_id: str
    reference_teacher: str | None


def population(config: LeagueConfig) -> list[OpponentBinding]:
    registry = orjson.loads(config.registry.read_bytes())
    result = []
    for member in registry['latest_training_population']:
        if config.opponent_ids and member['id'] not in config.opponent_ids:
            continue
        checkpoint = None
        if member.get('checkpoint_path'):
            portable = PROJECT_ROOT / member['portable_checkpoint']
            original = Path(member['checkpoint_path'])
            original = original if original.is_absolute() else config.original_root / original
            checkpoint = portable if portable.exists() else original
            if not any(checkpoint.resolve().is_relative_to(root.resolve()) for root in (PROJECT_ROOT / 'models', config.original_root, config.artifact_root)):
                raise ValueError('historical checkpoint outside declared archive roots')
            if file_hash(checkpoint) != member['checkpoint_sha256']:
                raise ValueError('historical checkpoint hash mismatch')
        deck = PROJECT_ROOT / member['portable_deck']
        if not deck.resolve().is_relative_to(PROJECT_ROOT / 'source') or file_hash(deck) != member['deck_sha256']:
            raise ValueError('historical deck hash mismatch')
        result.append(OpponentBinding(member['id'], member['label'], checkpoint, member.get('checkpoint_sha256'), deck, member['deck_sha256'], member['feature_policy_id'], member['spec'].removeprefix('spec:') if checkpoint is None else None))
    if len({row.identity for row in result}) != len(result):
        raise ValueError('duplicate league identity')
    if config.opponent_ids and set(config.opponent_ids) != {row.identity for row in result}:
        raise ValueError('unknown frozen opponent IDs')
    return result
