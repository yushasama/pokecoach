"""DeepMew-side helpers for the imported PufferLib Windows port."""

from __future__ import annotations

import importlib
from typing import Any

import torch

from deepmew.rl.puffer_env import check_puffer_compatibility


class PufferSupportError(RuntimeError):
    """Raised when the imported PufferLib runtime cannot support this backend."""


def require_puffer_cpu_advantage() -> tuple[Any, dict[str, Any]]:
    """Return the imported native module after verifying the CPU advantage API."""
    compatibility = check_puffer_compatibility()
    if not compatibility.available or not compatibility.native_available:
        raise PufferSupportError(str(compatibility.blocker))
    native = importlib.import_module("pufferlib._C")
    if not hasattr(native, "puff_advantage_cpu"):
        raise PufferSupportError("pufferlib._C is missing puff_advantage_cpu")
    return native, compatibility.to_dict()


def _require_cpu_float32_1d(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "cpu":
        raise PufferSupportError(f"{name} must be a CPU tensor for Puffer CPU advantage")
    if tensor.dtype != torch.float32:
        raise PufferSupportError(f"{name} must be float32 for Puffer CPU advantage")
    if tensor.ndim != 1:
        raise PufferSupportError(f"{name} must be 1-D for Puffer CPU advantage")
    return tensor.contiguous()


def compute_puffer_cpu_gae(
    *,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute DeepMew GAE through PufferLib's imported CPU advantage kernel.

    Puffer stores rollout rewards/terminals shifted by one slot. DeepMew stores a
    reward on the transition that just happened, so this wrapper builds the
    shifted buffer and includes `next_value` as the final bootstrap slot.
    """
    native, _compatibility = require_puffer_cpu_advantage()
    rewards = _require_cpu_float32_1d("rewards", rewards)
    dones = _require_cpu_float32_1d("dones", dones)
    values = _require_cpu_float32_1d("values", values)
    if rewards.shape != dones.shape or rewards.shape != values.shape:
        raise PufferSupportError(
            f"rewards/dones/values shapes differ: {tuple(rewards.shape)} "
            f"{tuple(dones.shape)} {tuple(values.shape)}"
        )

    horizon = int(rewards.shape[0]) + 1
    if horizon <= 1:
        empty = torch.zeros_like(rewards)
        return empty, empty

    bootstrap = torch.as_tensor(next_value, dtype=torch.float32, device="cpu").reshape(())
    shifted_values = torch.empty(horizon, dtype=torch.float32)
    shifted_rewards = torch.zeros(horizon, dtype=torch.float32)
    shifted_dones = torch.zeros(horizon, dtype=torch.float32)
    ratios = torch.ones(horizon, dtype=torch.float32)
    shifted_advantages = torch.zeros(horizon, dtype=torch.float32)

    shifted_values[:-1] = values
    shifted_values[-1] = bootstrap
    shifted_rewards[1:] = rewards
    shifted_dones[1:] = dones

    native.puff_advantage_cpu(
        shifted_values.data_ptr(),
        shifted_rewards.data_ptr(),
        shifted_dones.data_ptr(),
        ratios.data_ptr(),
        shifted_advantages.data_ptr(),
        1,
        horizon,
        float(gamma),
        float(gae_lambda),
        1.0,
        1.0,
    )
    advantages = shifted_advantages[:-1].clone()
    returns = advantages + values
    return advantages, returns
