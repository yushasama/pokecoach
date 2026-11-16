"""PR #1 shared state-option scorer; architecture and forward path unchanged."""

import torch
from torch import nn


class DeepMewEX(nn.Module):
    """Score each legal option with a shared state-conditioned network."""

    def __init__(self, option_dim: int = 304, value_dim: int = 144, hidden_dim: int = 256) -> None:
        super().__init__()
        self.state_encoder = nn.Sequential(nn.Linear(value_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.option_encoder = nn.Sequential(nn.Linear(option_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.option_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, options: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        state = self.state_encoder(values)
        encoded = self.option_encoder(options)
        # Identical per-option scorer, also accepting a padded decision batch.
        expanded = state.unsqueeze(-2).expand(*encoded.shape[:-1], state.shape[-1])
        return self.option_head(torch.cat((encoded, expanded), dim=-1)).squeeze(-1)
