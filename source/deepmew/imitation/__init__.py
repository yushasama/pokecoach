"""K4 first offline imitation layer: BC dataset export + count-table baseline.

Offline/training-only. Joins real ladder action rows to K2A archetype labels and
K3 deck-zoo splits, exports behavior-cloning examples + bucket summaries, and
fits a count-table action-category baseline evaluated on a held-out split. Never
imported by the submission/eval path.
"""

from __future__ import annotations

from deepmew.imitation.baseline import CountTablePolicy, evaluate_baselines
from deepmew.imitation.build import ImitationResult, build_imitation_dataset
from deepmew.imitation.schemas import ImitationConfig, ImitationExample

__all__ = [
    "CountTablePolicy",
    "ImitationConfig",
    "ImitationExample",
    "ImitationResult",
    "build_imitation_dataset",
    "evaluate_baselines",
]
