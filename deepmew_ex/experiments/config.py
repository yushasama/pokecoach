"""Resolve and admit the single supported S0 composition before doing I/O."""

from __future__ import annotations

import json
import math
import os
import re
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepmew_ex.experiments.v2.config import V2ExperimentConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def current_output_bytes(root: Path) -> int:
    """Count current files; another worker may already have removed a temp file."""
    if not stat.S_ISDIR(root.stat().st_mode):
        raise ValueError('output root must be a directory')
    def scan(directory: Path | str) -> int:
        total = 0
        # Fresh directory entries include file metadata on Windows, avoiding a
        # separate filesystem call per file at every resource boundary.
        with os.scandir(directory) as entries:
            for entry in entries:
                try:
                    metadata = entry.stat()
                    if stat.S_ISREG(metadata.st_mode):
                        total += metadata.st_size
                    elif stat.S_ISDIR(metadata.st_mode) and not entry.is_symlink():
                        total += scan(entry.path)
                except FileNotFoundError:
                    continue
        return total

    total = scan(root)
    root.stat()  # A vanished owner is not an ordinary completed temp-file write.
    return total


def component(value: str) -> str:
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,99}', value):
        raise ValueError(f'unsafe path component: {value!r}')
    return value


@dataclass(frozen=True)
class OutputPaths:
    root: Path

    @property
    def test_results(self) -> Path:
        return self.root / 'test-results'

    def validate(self, project: Path = PROJECT_ROOT) -> None:
        root = self.root.resolve()
        if root.is_relative_to(project.resolve()) and not root.is_relative_to((project / 'outputs').resolve()):
            raise ValueError('repository runs must use outputs/')

    def run(self, family: str, variant: str, run_id: str) -> Path:
        root = self.root.resolve()
        path = root.joinpath(*(component(value) for value in (family, variant, run_id))).resolve()
        if not path.is_relative_to(root):
            raise ValueError('run path escapes output root')
        return path


@dataclass(frozen=True)
class ExperimentConfig:
    family: str = 'deepmew_ex'
    variant: str = 'snapshot_bc_s0_v1'
    stage: str = 'S0'
    dataset: str = 'synthetic_v1'
    encoder: str = 'fixed_304_144_v1'
    architecture: str = 'DeepMewEX-option-bc-v0'
    trainer: str = 'bc_adamw'
    objective: str = 'single_selection_snapshot_bc'
    evaluator: str = 'offline_unweighted_v1'
    device: str = 'cpu'
    seed: int = 42
    train_decisions: int = 64
    validation_decisions: int = 16
    updates: int = 32
    hidden_dim: int = 64
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    evaluate_every: int = 8
    checkpoint_selection: str = 'maximum_validation_unweighted_top1_earliest_update'
    output_root: str = 'outputs'

    def validate(self) -> None:
        expected = {'stage': 'S0', 'dataset': 'synthetic_v1', 'encoder': 'fixed_304_144_v1', 'architecture': 'DeepMewEX-option-bc-v0', 'trainer': 'bc_adamw', 'objective': 'single_selection_snapshot_bc', 'evaluator': 'offline_unweighted_v1', 'device': 'cpu', 'checkpoint_selection': 'maximum_validation_unweighted_top1_earliest_update'}
        for key, value in expected.items():
            if getattr(self, key) != value:
                raise ValueError(f'unsupported {key}: {getattr(self, key)!r}; S0 requires {value}')
        for key, cap in {'train_decisions': 64, 'validation_decisions': 16, 'updates': 32, 'hidden_dim': 64, 'evaluate_every': 32}.items():
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= cap:
                raise ValueError(f'{key} must be an integer in [1, {cap}]')
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('invalid seed')
        for key in ('learning_rate', 'weight_decay', 'gradient_clip'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (key != 'weight_decay' and value == 0):
                raise ValueError(f'invalid {key}')
        component(self.family)
        component(self.variant)
        if not isinstance(self.output_root, str) or not self.output_root:
            raise ValueError('output_root is required')

    def paths(self, project: Path = PROJECT_ROOT) -> OutputPaths:
        output = Path(self.output_root)
        root = (output if output.is_absolute() else project / output).resolve()
        paths = OutputPaths(root)
        paths.validate(project)
        return paths

    def resolved(self, project: Path = PROJECT_ROOT) -> dict[str, object]:
        return {**asdict(self), 'output_root': str(self.paths(project).root)}


def load_config(path: Path) -> ExperimentConfig | V2ExperimentConfig:
    raw = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise ValueError('configuration must be an object')
    if raw.get('stage') == 'S2':
        from deepmew_ex.experiments.v2.config import V2ExperimentConfig
        config_v2 = V2ExperimentConfig(**raw)
        config_v2.validate()
        config_v2.paths()
        return config_v2
    if raw.get('stage', 'S0') != 'S0':
        raise ValueError('unsupported stage; proposed S1 configurations cannot execute in the S0 trainer')
    config = ExperimentConfig(**raw)
    config.validate()
    config.paths()
    return config
