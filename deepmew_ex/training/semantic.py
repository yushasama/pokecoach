"""Ragged semantic batches and BC metrics; optimizer invariants use the shared trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch.nn import functional as F

from deepmew_ex.datasets.v2.learning.cache import CachedDataset
from deepmew_ex.models.semantic import SemanticBC
from deepmew_ex.training.bc import apply_gradients


@dataclass(frozen=True)
class SemanticBatch:
    current: torch.Tensor
    options: torch.Tensor
    history: torch.Tensor
    option_mask: torch.Tensor
    targets: torch.Tensor

    def logits(self, model: SemanticBC) -> torch.Tensor:
        return model(self.current, self.options, self.history, self.option_mask)


def make_batch(dataset: CachedDataset, rows: list[dict[str, object]], treatment: str, seed: int) -> SemanticBatch:
    if not rows:
        raise ValueError('empty semantic batch')
    current, options, histories = [], [], []
    for row in rows:
        current.append(dataset.record(row, row['state_record']))
        options.append([dataset.record(row, index) for index in row['option_records']])
        indices = list(row['history_records'])
        if treatment == 'shuffled':
            rng = np.random.default_rng(seed + int(row['actor_order']))
            rng.shuffle(indices)
        histories.append([dataset.record(row, index) for index in indices] if treatment in ('ordered', 'shuffled') else [])
    state_leaves = max(len(record) for record in current)
    option_count = max(len(records) for records in options)
    option_leaves = max(len(record) for records in options for record in records)
    history_leaves = max((len(record) for records in histories for record in records), default=1)
    curr = np.zeros((len(rows), state_leaves, 3), dtype=np.float32)
    opts = np.zeros((len(rows), option_count, option_leaves, 3), dtype=np.float32)
    hist = np.zeros((len(rows), dataset.window, history_leaves, 3), dtype=np.float32)
    mask = np.zeros((len(rows), option_count), dtype=bool)
    for index, (state, candidates, previous) in enumerate(zip(current, options, histories, strict=True)):
        curr[index, :len(state)] = state
        mask[index, :len(candidates)] = True
        for option, tokens in enumerate(candidates):
            opts[index, option, :len(tokens)] = tokens
        for position, tokens in enumerate(previous):
            hist[index, dataset.window - len(previous) + position, :len(tokens)] = tokens
    return SemanticBatch(torch.from_numpy(curr), torch.from_numpy(opts), torch.from_numpy(hist), torch.from_numpy(mask), torch.tensor([row['target'] for row in rows], dtype=torch.long))


def evaluate_semantic(model: SemanticBC, dataset: CachedDataset, rows: list[dict[str, object]], seed: int, batch_size: int = 32, check_time: Callable[[], None] | None = None) -> dict[str, object]:
    if not rows:
        raise ValueError('empty eligible semantic validation')
    model.eval()
    loss, correct, entropy = 0.0, 0, 0.0
    predictions = []
    with torch.no_grad():
        for offset in range(0, len(rows), batch_size):
            if check_time:
                check_time()
            selected = rows[offset:offset + batch_size]
            batch = make_batch(dataset, selected, model.history, seed)
            logits = batch.logits(model)
            if not torch.isfinite(logits).all():
                raise ValueError('nonfinite semantic evaluation logits')
            loss += F.cross_entropy(logits, batch.targets, reduction='sum').item()
            predicted = logits.argmax(dim=-1)
            correct += int((predicted == batch.targets).sum())
            probabilities = logits.softmax(dim=-1)
            entropy += float((-(probabilities * probabilities.clamp_min(1e-30).log()).sum(dim=-1)).sum())
            predictions.extend({'row_id': row['row_id'], 'game_id': row['game_id'], 'target': int(target), 'prediction': int(choice)} for row, target, choice in zip(selected, batch.targets, predicted, strict=True))
    return {'rows': len(rows), 'games': len({row['game_id'] for row in rows}), 'loss_sum': loss, 'loss': loss / len(rows), 'correct': correct, 'top1': correct / len(rows), 'mean_entropy': entropy / len(rows), 'predictions': predictions}


def train_batch(model: SemanticBC, batch: SemanticBatch, optimizer: torch.optim.Optimizer, gradient_clip: float) -> tuple[float, float]:
    model.train()
    logits = batch.logits(model)
    if not torch.isfinite(logits).all():
        raise ValueError('nonfinite semantic training logits')
    loss = F.cross_entropy(logits, batch.targets)
    norm = apply_gradients(loss, model, optimizer, gradient_clip)
    return float(loss.detach()), norm
