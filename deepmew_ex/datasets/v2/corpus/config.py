"""Explicit input, worker and output bounds for the full local corpus port."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from deepmew_ex.experiments.config import OutputPaths, component


@dataclass(frozen=True)
class CorpusConfig:
    inventory: Path
    run_id: str
    paths: OutputPaths
    workers: int = 4
    max_episode_bytes: int = 64 * 1024**2
    max_steps: int = 10000
    min_free_bytes: int = 20 * 1024**3

    @property
    def output(self) -> Path:
        return self.paths.run('deepmew_ex', 'v2_corpus', self.run_id)

    def validate(self) -> None:
        component(self.run_id)
        self.paths.validate()
        if not 1 <= self.workers <= 4 or not 1 <= self.max_episode_bytes <= 64 * 1024**2 or not 3 <= self.max_steps <= 10000:
            raise ValueError('invalid corpus worker bounds')
