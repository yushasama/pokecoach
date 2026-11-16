"""Validate a declared synthetic contrast without performing extra training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from deepmew_ex.experiments.config import ExperimentConfig
from deepmew_ex.registry.records import validate_hash


@dataclass(frozen=True)
class Comparison:
    comparison_id: str
    control_run_id: str
    treatment_run_id: str
    kind: Literal['one_factor_ablation', 'practical_system_comparison']
    declared_factors: tuple[str, ...]
    control: dict[str, object]
    treatment: dict[str, object]
    frozen_identities: dict[str, object]
    treatment_identities: dict[str, object]

    def validate(self) -> dict[str, object]:
        if self.kind not in ('one_factor_ablation', 'practical_system_comparison'):
            raise ValueError('unknown comparison kind')
        if self.control_run_id == self.treatment_run_id:
            raise ValueError('comparison needs distinct run identities')
        # Both compositions must be admissible even though this is a no-training fixture.
        ExperimentConfig(**self.control).validate()
        ExperimentConfig(**self.treatment).validate()
        if self.frozen_identities != self.treatment_identities:
            raise ValueError('data/split/seed/budget/selection/evaluator identities changed')
        required = {'dataset', 'train_split', 'validation_split', 'seed', 'budget', 'checkpoint_selection', 'evaluator', 'source_population', 'legal_action_handling', 'device'}
        if not required <= self.frozen_identities.keys():
            raise ValueError('missing frozen comparison identities')
        for key in ('dataset', 'train_split', 'validation_split'):
            validate_hash(self.frozen_identities[key])
        for config in (ExperimentConfig(**self.control), ExperimentConfig(**self.treatment)):
            bound = {'seed': config.seed, 'budget': [config.train_decisions, config.validation_decisions, config.updates], 'checkpoint_selection': config.checkpoint_selection, 'evaluator': config.evaluator, 'source_population': 'synthetic_only', 'legal_action_handling': config.objective, 'device': config.device}
            if any(self.frozen_identities[key] != value for key, value in bound.items()):
                raise ValueError('comparison identity does not bind its configuration')
        changed = {key for key in self.control.keys() | self.treatment.keys() if self.control.get(key) != self.treatment.get(key)}
        if 'hidden_dim' in changed and self.control.get('variant') == self.treatment.get('variant'):
            raise ValueError('model specification change requires a new variant/version')
        changed.discard('variant')  # Organizational identity; the declared causal factor is width.
        allowed = {'hidden_dim', 'learning_rate', 'weight_decay', 'gradient_clip'}
        if not changed or changed != set(self.declared_factors) or not changed <= allowed:
            raise ValueError(f'undeclared or forbidden changes: {sorted(changed)}')
        if self.kind == 'one_factor_ablation' and len(changed) != 1:
            raise ValueError('a pure one-factor ablation must change exactly one factor')
        return {**asdict(self), 'status': 'passed', 'scope': 'configuration fixture only; treatment not executed', 'gameplay_evaluation': 'not_run'}
