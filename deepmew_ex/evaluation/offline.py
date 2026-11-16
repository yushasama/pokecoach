"""Independent checkpoint re-evaluation with the frozen offline protocol."""

from __future__ import annotations

from pathlib import Path

import torch

from deepmew_ex.data import Decision
from deepmew_ex.models.mlp import DeepMewEX
from deepmew_ex.training.bc import Metrics, train_epoch


def evaluate(model: DeepMewEX, decisions: list[Decision]) -> Metrics:
    return train_epoch(model, decisions)


def load_checkpoint(path: Path) -> tuple[DeepMewEX, dict[str, object]]:
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['architecture'] != 'DeepMewEX-option-bc-v0':
        raise ValueError('unsupported checkpoint architecture')
    model = DeepMewEX(hidden_dim=checkpoint['hidden_dim'])
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    return model, checkpoint
