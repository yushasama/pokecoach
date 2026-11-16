"""Fail-closed S2 configuration, validated before datasets or models load."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

from deepmew_ex.datasets.v2.learning.semantic import HISTORY_TREATMENTS
from deepmew_ex.experiments.config import PROJECT_ROOT, OutputPaths, component


@dataclass(frozen=True)
class V2ExperimentConfig:
    cache: str
    history: str
    variant: str
    stage: str = 'S2'
    family: str = 'deepmew_ex'
    dataset: str = 'admitted_causal_v2'
    encoder: str = 'semantic_path_value_numeric_v1'
    architecture: str = 'semantic_shared_option_v1'
    objective: str = 'single_selection_causal_v2_bc'
    trainer: str = 'bc_adamw'
    evaluator: str = 'offline_unweighted_v2'
    device: str = 'cpu'
    seed: int = 42
    hidden_dim: int = 64
    embedding_dim: int = 24
    batch_size: int = 32
    presentations: int = 32768
    evaluate_every: int = 256
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    max_wall_seconds: int = 900
    checkpoint_selection: str = 'minimum_trained_validation_ce_earliest_update'
    output_root: str = 'outputs'

    def validate(self) -> None:
        expected = {'stage': 'S2', 'dataset': 'admitted_causal_v2', 'encoder': 'semantic_path_value_numeric_v1', 'architecture': 'semantic_shared_option_v1', 'objective': 'single_selection_causal_v2_bc', 'trainer': 'bc_adamw', 'evaluator': 'offline_unweighted_v2', 'device': 'cpu', 'checkpoint_selection': 'minimum_trained_validation_ce_earliest_update'}
        for key, value in expected.items():
            if getattr(self, key) != value:
                raise ValueError(f'unsupported V2 {key} before data loading')
        if self.history not in HISTORY_TREATMENTS:
            raise ValueError('unsupported V2 history treatment before data loading')
        for key, cap in {'hidden_dim': 256, 'embedding_dim': 64, 'batch_size': 128, 'presentations': 32768, 'evaluate_every': 4096, 'max_wall_seconds': 900}.items():
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= cap:
                raise ValueError(f'invalid bounded V2 {key}')
        if self.presentations % self.batch_size:
            raise ValueError('presentations must be a whole number of batches')
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('invalid V2 seed')
        for key in ('learning_rate', 'weight_decay', 'gradient_clip'):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or key != 'weight_decay' and value == 0:
                raise ValueError(f'invalid V2 {key}')
        component(self.family)
        component(self.variant)
        if not self.cache or not self.output_root:
            raise ValueError('explicit cache/output root required')

    def paths(self, project: Path = PROJECT_ROOT) -> OutputPaths:
        root = Path(self.output_root)
        paths = OutputPaths(root if root.is_absolute() else project / root)
        paths.validate(project)
        return paths

    def resolved(self, project: Path = PROJECT_ROOT) -> dict[str, object]:
        cache = Path(self.cache)
        return {**asdict(self), 'cache': str((cache if cache.is_absolute() else project / cache).resolve()), 'output_root': str(self.paths(project).root.resolve())}
