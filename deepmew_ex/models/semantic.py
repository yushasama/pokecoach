"""Semantic leaf encoder and finite-context GRU around the existing option scorer."""

from __future__ import annotations

import torch
from torch import nn

from deepmew_ex.models.mlp import DeepMewEX
from deepmew_ex.datasets.v2.learning.semantic import HISTORY_TREATMENTS


class SemanticEncoder(nn.Module):
    def __init__(self, paths: int, values: int, hidden: int, embedding: int) -> None:
        super().__init__()
        self.paths = nn.Embedding(paths, embedding, padding_idx=0)
        self.values = nn.Embedding(values, embedding, padding_idx=0)
        self.leaf = nn.Sequential(nn.Linear(embedding * 2 + 1, hidden), nn.GELU())
        self.pool = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.GELU())

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = tokens[..., :1] != 0
        embedded = self.leaf(torch.cat((self.paths(tokens[..., 0].long()), self.values(tokens[..., 1].long()), tokens[..., 2:3]), dim=-1))
        count = mask.sum(dim=-2).clamp_min(1)
        mean = (embedded * mask).sum(dim=-2) / count
        maximum = embedded.masked_fill(~mask, -1e9).amax(dim=-2)
        maximum = torch.where(mask.any(dim=-2), maximum, torch.zeros_like(maximum))
        return self.pool(torch.cat((mean, maximum), dim=-1))


class SemanticBC(nn.Module):
    def __init__(self, paths: int, values: int, hidden: int = 64, embedding: int = 24, history: str = 'ordered') -> None:
        super().__init__()
        if history not in HISTORY_TREATMENTS:
            raise ValueError('unsupported semantic history treatment')
        self.history = history
        self.encoder = SemanticEncoder(paths, values, hidden, embedding)
        self.memory = None if history == 'snapshot' else nn.GRU(hidden, hidden, batch_first=True)
        self.scorer = DeepMewEX(option_dim=hidden, value_dim=hidden if self.memory is None else hidden * 2, hidden_dim=hidden)

    def forward(self, current: torch.Tensor, options: torch.Tensor, history: torch.Tensor, option_mask: torch.Tensor) -> torch.Tensor:
        state = self.encoder(current)
        candidates = self.encoder(options)
        if self.memory is not None:
            # Every prediction reconstructs an eight-request window from h0=0.
            # A null-history control retains exactly the same recurrent parameters.
            past = state.new_zeros((state.shape[0], history.shape[1], state.shape[-1])) if self.history == 'null' else self.encoder(history)
            _, hidden = self.memory(past)
            state = torch.cat((state, hidden[-1]), dim=-1)
        return self.scorer(candidates, state).masked_fill(~option_mask, -torch.finfo(state.dtype).max)
