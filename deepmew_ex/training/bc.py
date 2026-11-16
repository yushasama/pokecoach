"""PR #1's weighted per-decision CE/AdamW loop with explicit admission."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from deepmew_ex.data import Decision, Split
from deepmew_ex.admission.decisions import exclusion_reason, validate_decision
from deepmew_ex.models.mlp import DeepMewEX


@dataclass(frozen=True)
class Support:
    raw: int
    eligible: int
    excluded: dict[str, int]
    weight_sum: float


@dataclass(frozen=True)
class Metrics:
    rows: int
    correct: int
    loss_sum: float
    weight_sum: float
    weighted_loss_sum: float
    weighted_correct: float

    def report(self) -> dict[str, object]:
        return {**asdict(self), 'loss': self.loss_sum / self.rows, 'top1': self.correct / self.rows, 'weighted_loss': self.weighted_loss_sum / self.weight_sum, 'weighted_top1': self.weighted_correct / self.weight_sum}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def encode(decision: Decision) -> tuple[torch.Tensor, torch.Tensor]:
    # This is the entire actor-input allowlist. Outcome, identity, continuity and labels stay outside.
    return torch.from_numpy(decision.option_features), torch.from_numpy(decision.value_features)


def admit(decisions: Iterable[Decision], split: Split) -> tuple[list[Decision], Support]:
    rows: list[Decision] = []
    excluded: Counter[str] = Counter()
    identities: set[str] = set()
    raw = 0
    for decision in decisions:
        raw += 1
        if decision.split != split or decision.row_id in identities:
            raise ValueError('wrong split or duplicate row identity')
        identities.add(decision.row_id)
        validate_decision(decision)
        reason = exclusion_reason(decision)
        if reason:
            excluded[reason] += 1
            continue
        rows.append(decision)
    if not rows:
        raise ValueError(f'empty eligible {split} set; excluded={dict(excluded)}')
    return rows, Support(raw, len(rows), dict(excluded), sum(row.actor_loss_weight for row in rows))


def check_isolation(train: list[Decision], validation: list[Decision]) -> None:
    for field in ('row_id', 'game_id', 'trajectory_id'):
        if {getattr(row, field) for row in train} & {getattr(row, field) for row in validation}:
            raise ValueError(f'{field} crosses train/validation split')


def apply_gradients(loss: torch.Tensor, model: nn.Module, optimizer: torch.optim.Optimizer, gradient_clip: float) -> float:
    if not torch.isfinite(loss):
        raise ValueError('non-finite loss')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    if any(gradient is None or not torch.isfinite(gradient).all() for gradient in gradients):
        raise ValueError('missing or non-finite gradient')
    norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError('non-finite updated parameters')
    return float(norm)


def train_epoch(model: DeepMewEX, decisions: list[Decision], optimizer: torch.optim.Optimizer | None = None, gradient_clip: float = 1.0) -> Metrics:
    """One presentation per supplied row, preserving pilot loss/update semantics."""
    model.train(optimizer is not None)
    loss_sum = weighted_loss_sum = weighted_correct = weight_sum = 0.0
    correct = 0
    if not decisions:
        raise ValueError('cannot score an empty eligible set')
    for decision in decisions:
        options, values = encode(decision)
        target = int(decision.selection[0])
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(options, values)
            if logits.shape != (len(options),) or not torch.isfinite(logits).all():
                raise ValueError('invalid/non-finite logits')
            loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target]))
            weighted_loss = loss * float(decision.actor_loss_weight)
            if not torch.isfinite(weighted_loss):
                raise ValueError('non-finite loss')
            if optimizer is not None:
                apply_gradients(weighted_loss, model, optimizer, gradient_clip)
        hit = int(torch.argmax(logits).item() == target)
        correct += hit
        loss_sum += float(loss.detach())
        weighted_loss_sum += float(weighted_loss.detach())
        weighted_correct += hit * decision.actor_loss_weight
        weight_sum += decision.actor_loss_weight
    return Metrics(len(decisions), correct, loss_sum, weight_sum, weighted_loss_sum, weighted_correct)
