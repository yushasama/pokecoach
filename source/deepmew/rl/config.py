"""Config-driven RL experiment definitions.

An `RLExperimentConfig` fully specifies one rung of the ablation ladder: which
observation/mask/reward/opponent-sampler/feature-modules are enabled. Validation
enforces the hard rules in `deepmew.rl.safety` (RL0 stays the pure masked-PPO
control). Adding an ablation is a config change, never a trainer edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from deepmew.rl import safety


@dataclass(frozen=True)
class RLExperimentConfig:
    experiment_id: str
    observation: str = "base"
    action_mask: str = "legal_only"
    reward: str = "terminal_win_loss"
    opponent_sampler: str = "simple_baseline_mix"
    feature_modules: tuple[str, ...] = ()
    bc_init: bool = False
    dsu: bool = False
    belief: bool = False
    support_features: bool = False
    static_priors: bool = False
    # Run knobs (kept tiny by default; long training is out of scope here).
    seed: int = 0
    max_steps_per_game: int = 2000
    trainer_backend: str = "smoke"
    agent_deck_id: str = "agent_root_deck"
    agent_deck_path: str = "deck.csv"
    opponent_deck_id: str = "opponent_policy_default"
    opponent_deck_path: str = "deck.csv"
    notes: str = ""

    @property
    def is_rl0(self) -> bool:
        return self.experiment_id.startswith("RL0")

    def validate(self) -> "RLExperimentConfig":
        if self.observation not in {"base"}:
            raise safety.RLSafetyError(f"unknown observation: {self.observation!r}")
        if self.action_mask != "legal_only":
            raise safety.RLSafetyError(f"unknown action_mask: {self.action_mask!r}")
        # Feature flags must agree with the enabled feature-module set.
        enabled = set(self.feature_modules)
        if self.support_features != ("support_chain" in enabled):
            raise safety.RLSafetyError("support_features flag disagrees with feature_modules")
        if self.dsu != ("dsu_belief" in enabled):
            raise safety.RLSafetyError("dsu flag disagrees with feature_modules")
        if self.belief != ("dsu_belief" in enabled):
            raise safety.RLSafetyError("belief flag disagrees with feature_modules")
        if self.static_priors != ("robust_posterior" in enabled):
            raise safety.RLSafetyError("static_priors flag disagrees with feature_modules")
        if self.is_rl0:
            safety.assert_rl0_is_pure(
                feature_modules=self.feature_modules,
                dsu=self.dsu,
                belief=self.belief,
                support_features=self.support_features,
                static_priors=self.static_priors,
                reward=self.reward,
                bc_init=self.bc_init,
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "observation": self.observation,
            "action_mask": self.action_mask,
            "reward": self.reward,
            "opponent_sampler": self.opponent_sampler,
            "feature_modules": list(self.feature_modules),
            "bc_init": self.bc_init,
            "dsu": self.dsu,
            "belief": self.belief,
            "support_features": self.support_features,
            "static_priors": self.static_priors,
            "seed": self.seed,
            "max_steps_per_game": self.max_steps_per_game,
            "trainer_backend": self.trainer_backend,
            "agent_deck_id": self.agent_deck_id,
            "agent_deck_path": self.agent_deck_path,
            "opponent_deck_id": self.opponent_deck_id,
            "opponent_deck_path": self.opponent_deck_path,
            "notes": self.notes,
        }


def extend(base: RLExperimentConfig, experiment_id: str, **overrides: Any) -> RLExperimentConfig:
    """Build a child experiment from a parent ('extends' in the YAML ladder)."""
    if "notes" not in overrides:
        overrides["notes"] = ""
    return replace(base, experiment_id=experiment_id, **overrides)
