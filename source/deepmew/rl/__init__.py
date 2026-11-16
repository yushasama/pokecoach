"""Ablation-driven RL scaffold for DeepMew's Competition RL lane (DEC-040).

Modular boundaries (every ablation is config + a registered component, never a
trainer edit):

- `env_adapter`   EnvAdapter: single-agent view over the engine vs a fixed
                  opponent. Knows nothing about algorithm/reward/features.
- `observation`   ObservationEncoder: public current-state features (RL0 base).
- `action_mask`   ActionMasker: legal-only mask + safe projection (INVARIANT #1).
- `rewards`       RewardFn: terminal win/loss (RL0) + ablated shaping modules.
- `opponents`     OpponentSampler: fixed mixes drawn from arena agent specs.
- `features`      FeatureModule: optional support/DSU-belief/posterior add-ons,
                  ALL disabled in RL0, each with a declared leakage boundary.
- `trainer`       AlgorithmBackend (PPO/A2C/offline-RL/BC/... later) + SmokeTrainer.
- `eval`          Evaluator over a fixed opponent suite (no training).
- `metrics`/`comparator`  measurement + ablation comparison.
- `manifests`/`logging`   reproducible run headers (config/schema hashes, git).
- `config`/`registry`/`safety`  the experiment ladder and hard rules.

This package is offline training/research infrastructure. It does NOT change
`main.py`, `deck.csv`, `cg/`, or the active policy, and RL0 imports none of the
offline label/belief/prior packages.
"""

from __future__ import annotations

__all__ = [
    "EXPERIMENTS",
    "EXPERIMENT_ALIASES",
    "RLExperimentConfig",
    "build_encoder",
    "build_masker",
    "build_opponent_sampler",
    "build_reward",
    "resolve_experiment",
]


def __getattr__(name: str):
    if name == "RLExperimentConfig":
        from deepmew.rl.config import RLExperimentConfig

        return RLExperimentConfig
    if name in {
        "EXPERIMENTS",
        "EXPERIMENT_ALIASES",
        "build_encoder",
        "build_masker",
        "build_opponent_sampler",
        "build_reward",
        "resolve_experiment",
    }:
        from deepmew.rl import registry

        return getattr(registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
