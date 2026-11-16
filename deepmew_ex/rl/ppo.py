"""Value head, advantage estimators and the clipped PPO objective for the temporal actor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from deepmew_ex.datasets.v2.learning.leaf_tables import TABLES_ATTRIBUTE, LeafTables
from deepmew_ex.datasets.v2.learning.trajectory import ActorSequence, Prediction, StageTarget
from deepmew_ex.models.temporal import TemporalPolicy, option_matrix
from deepmew_ex.rl.records import Decision, Trajectory  # re-exported


class ValueHead(nn.Module):
    """Win/loss value of the actor-visible causal state (the same state the policy reads)."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


def gae(rewards: np.ndarray, values: np.ndarray, lam: float, gamma: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Generalized advantage estimation for one terminal episode; returns (advantages, value targets)."""
    advantages = np.zeros(len(values), dtype=np.float64)
    running = 0.0
    for step in range(len(values) - 1, -1, -1):
        following = values[step + 1] if step + 1 < len(values) else 0.0
        delta = rewards[step] + gamma * following - values[step]
        running = delta + gamma * lam * running
        advantages[step] = running
    return advantages, advantages + values


def upgo(rewards: np.ndarray, values: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """UPGO returns: follow the sampled return while the next action looked at least as good as V, else bootstrap."""
    returns = np.zeros(len(values), dtype=np.float64)
    following = 0.0
    for step in range(len(values) - 1, -1, -1):
        if step + 1 == len(values):
            returns[step] = rewards[step]
        else:
            action_value = rewards[step + 1] + gamma * (values[step + 2] if step + 2 < len(values) else 0.0)
            tail = following if action_value >= values[step + 1] else values[step + 1]
            returns[step] = rewards[step] + gamma * tail
        following = returns[step]
    return returns


def ppo_policy_loss(logp: torch.Tensor, old_logp: torch.Tensor, advantages: torch.Tensor, clip: float) -> tuple[torch.Tensor, dict[str, float]]:
    ratio = torch.exp(logp - old_logp)
    loss = -torch.minimum(ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages).mean()
    with torch.no_grad():
        stats = {'clip_fraction': float(((ratio - 1).abs() > clip).float().mean()), 'approx_kl': float(((ratio - 1) - torch.log(ratio)).mean())}
    return loss, stats


def policy_terms(model: TemporalPolicy, value_head: ValueHead, sequences: list[ActorSequence], detach_value: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per decision (prediction order): log-probability of the taken action, exact entropy (NaN if structured), value.

    ``detach_value`` keeps value regression from moving the shared behaviour-cloned trunk.
    """
    flat, padded_context, lengths = model.encode_padded(sequences)
    offsets = np.concatenate(([0], np.cumsum([len(sequence.records) for sequence in sequences])))
    rows, simple, complex_rows = [], [], []
    for game, sequence in enumerate(sequences):
        for prediction in sequence.predictions:
            if not 0 <= prediction.event_index < lengths[game]:
                raise IndexError('decision event outside its game history')
            identity = len(rows)
            rows.append((game, prediction.event_index))
            stage = prediction.stages[0]
            if len(prediction.stages) == 1 and stage.minimum == stage.maximum == 1:
                simple.append((identity, game, stage.options, stage.selected[0]))
            else:
                complex_rows.append((identity, game, prediction.stages))
    device, pad = flat.device, flat.shape[0]
    table = torch.cat((flat, flat.new_zeros((1, flat.shape[1]))))
    states = padded_context[torch.tensor([row[0] for row in rows], device=device), torch.tensor([row[1] for row in rows], device=device)]
    values = value_head(states.detach() if detach_value else states)
    logp = states.new_zeros(len(rows))
    entropy = states.new_full((len(rows),), float('nan'))
    if simple:
        index = torch.tensor([row[0] for row in simple], device=device)
        gather = torch.from_numpy(option_matrix([row[2] for row in simple], [int(offsets[row[1]]) for row in simple], [int(offsets[row[1] + 1] - offsets[row[1]]) for row in simple], pad)).to(device)
        mask = gather != pad
        logits = model.scorer(table[gather], states[index]).masked_fill(~mask, -torch.inf)
        logs = F.log_softmax(logits, dim=1)
        taken = logs.gather(1, torch.tensor([[row[3]] for row in simple], device=device)).squeeze(1)
        spread = -(logs.exp() * logs.masked_fill(~mask, 0.0)).sum(dim=1)
        logp = logp.index_copy(0, index, taken)
        entropy = entropy.index_copy(0, index, spread)
    if complex_rows:
        index = torch.tensor([row[0] for row in complex_rows], device=device)
        nll = model.batched_stage_losses(table, states[index], [int(offsets[row[1]]) for row in complex_rows], [int(offsets[row[1] + 1] - offsets[row[1]]) for row in complex_rows], [row[2] for row in complex_rows])
        logp = logp.index_copy(0, index, -nll)
    return logp, entropy, values
