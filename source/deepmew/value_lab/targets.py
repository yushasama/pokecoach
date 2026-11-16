"""Value-target construction over actor-step sequences (gamma = 1 episodic).

All functions operate on right-padded sequence batches:

- ``z``: [B] actor-relative terminal labels (win 1 / draw 0.5 / loss 0)
- ``sequence_mask``: [B, T] bool, a nonempty contiguous True prefix per row
- ``target_probs``: [B, T] target-network win probabilities

Target-network outputs and terminal labels are detached internally. Excluded
training targets must not be passed as ``sequence_mask``: target eligibility is
applied only to the downstream loss and metrics, while every real sequence row
remains available as temporal context and as a bootstrap state.

There are no intermediate rewards in this contract (none exist in the data;
none are fabricated), so with gamma = 1 every target is a probability on the
same actor-relative win scale. The horizon unit is one actor step: row t+1 is
the next decision by the same actor (deepmew.actor_transition.v1 adjacency).
Padding targets are finite zeros and must be ignored by the downstream target
mask.
"""

from __future__ import annotations

import math

import torch


def _validate_sequence_inputs(
    z: torch.Tensor,
    sequence_mask: torch.Tensor,
) -> tuple[int, int, torch.Tensor, torch.Tensor]:
    if not isinstance(z, torch.Tensor) or not isinstance(sequence_mask, torch.Tensor):
        raise TypeError("z and sequence_mask must be torch tensors")
    if sequence_mask.ndim != 2:
        raise ValueError("sequence_mask must have shape [B, T]")
    if sequence_mask.dtype != torch.bool:
        raise ValueError("sequence_mask must have boolean dtype")
    batch, horizon = sequence_mask.shape
    if batch < 1 or horizon < 1:
        raise ValueError("sequence batches must have nonzero batch and horizon dimensions")
    if z.ndim != 1 or z.shape[0] != batch:
        raise ValueError("z must have shape [B] aligned with sequence_mask")
    if z.device != sequence_mask.device:
        raise ValueError("z and sequence_mask must be on the same device")
    if not torch.is_floating_point(z):
        raise ValueError("z must have a floating-point dtype")
    if not bool(torch.isfinite(z).all().item()):
        raise ValueError("z must contain only finite terminal outcomes")
    allowed_z = (z == 0.0) | (z == 0.5) | (z == 1.0)
    if not bool(allowed_z.all().item()):
        raise ValueError("z must use the terminal outcome scale {0, 0.5, 1}")
    lengths = sequence_mask.sum(dim=1)
    if bool(torch.any(lengths == 0).item()):
        raise ValueError("every sequence must contain at least one real step")
    if horizon > 1:
        false_to_true = (~sequence_mask[:, :-1]) & sequence_mask[:, 1:]
        if bool(torch.any(false_to_true).item()):
            raise ValueError("sequence_mask must be a prefix-contiguous right-padding mask")

    return batch, horizon, lengths, z.detach().to(dtype=torch.float32)


def _validate_target_probs(
    target_probs: torch.Tensor,
    sequence_mask: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(target_probs, torch.Tensor):
        raise TypeError("target_probs must be a torch tensor")
    if target_probs.shape != sequence_mask.shape:
        raise ValueError("target_probs must have shape [B, T] aligned with sequence_mask")
    if target_probs.device != sequence_mask.device:
        raise ValueError("target_probs and sequence_mask must be on the same device")
    if not torch.is_floating_point(target_probs):
        raise ValueError("target_probs must have a floating-point dtype")
    real_probs = target_probs[sequence_mask]
    if not bool(torch.isfinite(real_probs).all().item()):
        raise ValueError("target_probs must be finite on every real sequence step")
    if bool(torch.any((real_probs < 0.0) | (real_probs > 1.0)).item()):
        raise ValueError("target_probs must be probabilities in [0, 1], not logits")

    return target_probs.detach().to(dtype=torch.float32)


def mc_targets(z: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
    """Repeat the actor-relative terminal result over every real sequence step."""
    _, horizon, _, detached_z = _validate_sequence_inputs(z, sequence_mask)
    targets = detached_z[:, None].expand(-1, horizon).clone()
    targets.masked_fill_(~sequence_mask, 0.0)

    return targets


def td0_targets(
    z: torch.Tensor,
    sequence_mask: torch.Tensor,
    target_probs: torch.Tensor,
) -> torch.Tensor:
    """Use V_target(t+1) on nonterminal actor steps and z on the final step."""
    batch, _, lengths, detached_z = _validate_sequence_inputs(z, sequence_mask)
    detached_probs = _validate_target_probs(target_probs, sequence_mask)
    targets = torch.zeros_like(detached_probs)
    targets[:, :-1] = detached_probs[:, 1:]
    batch_index = torch.arange(batch, device=sequence_mask.device)
    last_index = lengths - 1
    targets[batch_index, last_index] = detached_z.to(dtype=targets.dtype)
    targets.masked_fill_(~sequence_mask, 0.0)

    return targets


def nstep_targets(
    z: torch.Tensor,
    sequence_mask: torch.Tensor,
    target_probs: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """Use V_target(t+n) when real, otherwise the terminal actor result z."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    batch, horizon, lengths, detached_z = _validate_sequence_inputs(z, sequence_mask)
    detached_probs = _validate_target_probs(target_probs, sequence_mask)
    step_index = torch.arange(horizon, device=sequence_mask.device)[None, :]
    bootstrap_index = step_index + n
    has_bootstrap = bootstrap_index < lengths[:, None]
    gather_index = bootstrap_index.clamp(max=horizon - 1).expand(batch, -1)
    bootstrap = torch.gather(detached_probs, 1, gather_index)
    terminal = detached_z.to(dtype=detached_probs.dtype)[:, None].expand(-1, horizon)
    targets = torch.where(has_bootstrap, bootstrap, terminal)
    targets.masked_fill_(~sequence_mask, 0.0)

    return targets


def lambda_targets(
    z: torch.Tensor,
    sequence_mask: torch.Tensor,
    target_probs: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Build forward-view lambda returns by backward recursion (gamma = 1).

    G_(T-1) = z; G_t = (1 - lambda) * V_target(t+1) + lambda * G_(t+1).
    """
    if isinstance(lam, bool):
        raise ValueError("lam must be a finite float in [0, 1]")
    try:
        lambda_value = float(lam)
    except (TypeError, ValueError) as error:
        raise ValueError("lam must be a finite float in [0, 1]") from error
    if not math.isfinite(lambda_value) or not 0.0 <= lambda_value <= 1.0:
        raise ValueError("lam must be a finite float in [0, 1]")
    _, horizon, lengths, detached_z = _validate_sequence_inputs(z, sequence_mask)
    detached_probs = _validate_target_probs(target_probs, sequence_mask)
    terminal = detached_z.to(dtype=detached_probs.dtype)
    targets = torch.zeros_like(detached_probs)
    running = terminal.clone()
    last_index = lengths - 1
    for step in range(horizon - 1, -1, -1):
        is_real = sequence_mask[:, step]
        is_last = last_index == step
        bootstrap = detached_probs[:, min(step + 1, horizon - 1)]
        continued = (1.0 - lambda_value) * bootstrap + lambda_value * running
        value = torch.where(is_last, terminal, continued)
        running = torch.where(is_real, value, running)
        targets[:, step] = torch.where(is_real, value, torch.zeros_like(value))

    return targets


TARGET_METHODS = ("mc", "td0", "nstep4", "nstep16", "lam70", "lam90")


def build_targets(
    method: str,
    z: torch.Tensor,
    sequence_mask: torch.Tensor,
    target_probs: torch.Tensor | None,
) -> torch.Tensor:
    if method not in TARGET_METHODS:
        raise ValueError(f"unknown target method {method}")
    if method == "mc":
        return mc_targets(z, sequence_mask)
    if target_probs is None:
        raise ValueError(f"target method {method} requires target-network probabilities")
    if method == "td0":
        return td0_targets(z, sequence_mask, target_probs)
    if method.startswith("nstep"):
        return nstep_targets(z, sequence_mask, target_probs, int(method.removeprefix("nstep")))
    if method.startswith("lam"):
        return lambda_targets(
            z,
            sequence_mask,
            target_probs,
            float(method.removeprefix("lam")) / 100.0,
        )
    raise AssertionError(f"unhandled accepted target method {method}")
