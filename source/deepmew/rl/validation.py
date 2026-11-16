"""Reusable critical-path validators for RL experiments."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import torch

from deepmew.rl.action_mask import ActionMasker
from deepmew.rl.algorithms.masked_ppo import (
    ActorCritic,
    MaskedPPOConfig,
    PPOTrainingState,
    build_model_from_checkpoint,
    load_checkpoint,
)
from deepmew.rl.config import RLExperimentConfig
from deepmew.rl.safety import RLSafetyError


class RLValidationError(AssertionError):
    """Raised when a critical-path RL validation fails."""


def _finite_number(name: str, value: Any) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RLValidationError(f"{name} is not finite: {value!r}")


def validate_experiment_config(config: RLExperimentConfig) -> None:
    try:
        config.validate()
    except RLSafetyError as exc:
        raise RLValidationError(str(exc)) from exc
    if config.is_rl0 and config.feature_modules:
        raise RLValidationError("RL0 must not enable feature modules")
    if "label_candidates" in " ".join(config.feature_modules):
        raise RLValidationError("raw Phase 5 label candidates are not valid feature modules")


def validate_environment_contract(env: Any, encoder: Any, masker: ActionMasker) -> None:
    step = env.reset()
    if step.done:
        raise RLValidationError(f"environment reset ended immediately: {step}")
    obs = step.obs or {}
    features = encoder.encode(obs)
    if len(features) != encoder.schema.dim:
        raise RLValidationError("encoder output dim does not match schema")
    for index, value in enumerate(features):
        _finite_number(f"obs[{index}]", value)
    select = env.select or {}
    mask = masker.build_mask(select)
    if mask.n_options and not mask.legal_indices():
        raise RLValidationError("mask has options but no legal indices")
    selected = list(range(mask.min_count))
    if not masker.is_legal(selected, select):
        raise RLValidationError("first-minCount fallback is not legal")
    next_step = env.step(selected)
    if next_step.crashed or next_step.timed_out:
        raise RLValidationError(f"environment step failed: {next_step.info}")
    env.close()


def validate_numerics(metrics: dict[str, Any]) -> None:
    for key, value in metrics.items():
        if isinstance(value, dict):
            validate_numerics(value)
        elif isinstance(value, (int, float)):
            _finite_number(key, value)
    if metrics.get("parameter_delta_norm", 1.0) == 0:
        raise RLValidationError("parameters did not change")


def validate_checkpoint_roundtrip(
    checkpoint_path: Path | str,
    *,
    fixed_obs: Sequence[float],
    fixed_masker_dim: int = 64,
) -> None:
    path = Path(checkpoint_path)
    if not path.exists():
        raise RLValidationError(f"checkpoint missing: {path}")
    payload = load_checkpoint(path)
    model, config, _ = build_model_from_checkpoint(path)
    obs = torch.as_tensor(fixed_obs, dtype=torch.float32)
    with torch.no_grad():
        logits_a, value_a = model(obs)
    model_b = ActorCritic(config.obs_dim, config.action_dim, config.hidden_dim)
    model_b.load_state_dict(payload["model_state"])
    model_b.eval()
    with torch.no_grad():
        logits_b, value_b = model_b(obs)
    if not torch.allclose(logits_a, logits_b, atol=1e-6):
        raise RLValidationError("checkpoint logits changed after reload")
    if not torch.allclose(value_a, value_b, atol=1e-6):
        raise RLValidationError("checkpoint value changed after reload")
    if int(payload["algo_config"]["action_dim"]) != fixed_masker_dim:
        raise RLValidationError("checkpoint action dimension does not match masker")
    state = payload.get("state", {})
    if int(state.get("global_step", 0)) < 0 or int(state.get("update_count", 0)) < 0:
        raise RLValidationError("checkpoint counters are invalid")


def validate_training_run(result: Any, state: PPOTrainingState | None = None) -> None:
    if result.decision_steps <= 0:
        raise RLValidationError("training collected no decisions")
    if result.illegal_attempts != 0:
        raise RLValidationError(f"illegal attempts occurred: {result.illegal_attempts}")
    if result.crashes or result.timeouts:
        raise RLValidationError(f"crashes/timeouts occurred: {result.crashes}/{result.timeouts}")
    if state is not None:
        if state.update_count <= 0:
            raise RLValidationError("no PPO updates occurred")
        validate_numerics(state.last_update.to_dict())


def validate_ablation_pair(left: dict[str, Any], right: dict[str, Any]) -> None:
    for key in ("backend", "seed", "train_steps", "eval_games"):
        if left.get(key) != right.get(key):
            raise RLValidationError(f"ablation pair has unequal {key}: {left.get(key)} != {right.get(key)}")
    allowed = {"experiment", "feature_modules", "reward", "opponent_sampler"}
    observed = set(left.get("differences", ())) | set(right.get("differences", ()))
    unsupported = observed - allowed
    if unsupported:
        raise RLValidationError(f"unsupported ablation differences: {sorted(unsupported)}")


def validate_runtime_boundary(diff_text: str) -> None:
    forbidden = (
        " main.py",
        " deck.csv",
        " cg/",
        " deepmew/policy/",
        " deepmew/state_machine/",
        " deepmew/turn_plan/",
        " deepmew/submissions/",
    )
    hits = [marker for marker in forbidden if marker in diff_text.replace("\\", "/")]
    if hits:
        raise RLValidationError(f"runtime/submission files changed: {hits}")
