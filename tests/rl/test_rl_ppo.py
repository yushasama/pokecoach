from __future__ import annotations

import numpy as np
import pytest
import torch

from deepmew_ex.rl.ppo import gae, ppo_policy_loss, upgo


def test_gae_limits_and_brute_force() -> None:
    rewards = np.array([0.0, 0.0, 0.0, 1.0])
    values = np.array([0.2, -0.1, 0.4, 0.6])
    advantages, targets = gae(rewards, values, lam=1.0)
    np.testing.assert_allclose(targets, np.ones(4))  # lambda=1, gamma=1: Monte Carlo return
    advantages, targets = gae(rewards, values, lam=0.0)
    np.testing.assert_allclose(advantages, [values[1] - values[0], values[2] - values[1], values[3] - values[2], 1.0 - values[3]])
    lam, gamma = 0.95, 0.99
    advantages, _ = gae(rewards, values, lam=lam, gamma=gamma)
    following = np.append(values[1:], 0.0)
    deltas = rewards + gamma * following - values
    brute = [sum((gamma * lam) ** k * deltas[t + k] for k in range(len(deltas) - t)) for t in range(len(deltas))]
    np.testing.assert_allclose(advantages, brute)


def test_upgo_follows_good_actions_and_bootstraps_bad_ones() -> None:
    rewards = np.array([0.0, 0.0, 0.0, -1.0])
    values = np.array([0.0, 0.3, 0.1, 0.5])
    # t=3 terminal: -1. t=2: Q(s3,a3) = -1 < V(s3) = 0.5 -> bootstrap 0.5.
    # t=1: Q(s2,a2) = 0 + V(s3) = 0.5 >= V(s2) = 0.1 -> follow G(2) = 0.5.
    # t=0: Q(s1,a1) = 0 + V(s2) = 0.1 < V(s1) = 0.3 -> bootstrap 0.3.
    np.testing.assert_allclose(upgo(rewards, values), [0.3, 0.5, 0.5, -1.0])
    # All actions better than expected: UPGO equals the Monte Carlo return.
    np.testing.assert_allclose(upgo(np.array([0.0, 0.0, 1.0]), np.array([0.0, -0.5, -0.5])), [1.0, 1.0, 1.0])


def test_ppo_clip_stops_gradient_outside_trust_region() -> None:
    old = torch.zeros(4)
    logp = torch.log(torch.tensor([1.5, 0.5, 1.05, 0.95])).requires_grad_(True)
    advantages = torch.tensor([1.0, -1.0, 1.0, -1.0])
    loss, stats = ppo_policy_loss(logp, old, advantages, clip=0.2)
    loss.backward()
    # Ratio 1.5 with A>0 and 0.5 with A<0 are clipped (no gradient); inside the region gradient is -ratio*A/N.
    np.testing.assert_allclose(logp.grad.numpy(), [0.0, 0.0, -1.05 / 4, 0.95 / 4], atol=1e-6)
    assert stats['clip_fraction'] == 0.5
    assert loss.item() == pytest.approx(-(1.2 - 0.8 + 1.05 - 0.95) / 4)
