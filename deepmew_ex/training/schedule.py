"""Learning-rate clock measured in consumed decisions, independent of batch shape."""

from __future__ import annotations

import math

import torch


class DecisionSchedule:
    def __init__(self, optimizer: torch.optim.Optimizer, total_presentations: int, minimum_ratio: float = 0.1, base_lrs: list[float] | None = None) -> None:
        if total_presentations < 1 or not 0 < minimum_ratio <= 1:
            raise ValueError('invalid decision schedule')
        self.optimizer = optimizer
        self.total_presentations = total_presentations
        self.minimum_ratio = minimum_ratio
        self.base_lrs = base_lrs if base_lrs is not None else [float(group['lr']) for group in optimizer.param_groups]
        if len(self.base_lrs) != len(optimizer.param_groups) or any(not math.isfinite(rate) or rate <= 0 for rate in self.base_lrs):
            raise ValueError('invalid schedule base rates')
        self.presentations = 0

    def step(self, presentations: int) -> None:
        if not self.presentations <= presentations <= self.total_presentations:
            raise ValueError('decision clock moved backwards or beyond membership')
        self.presentations = presentations
        scale = self.minimum_ratio + (1 - self.minimum_ratio) * (1 + math.cos(math.pi * presentations / self.total_presentations)) / 2
        for group, rate in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group['lr'] = rate * scale

    def state_dict(self) -> dict[str, object]:
        return {'total_presentations': self.total_presentations, 'minimum_ratio': self.minimum_ratio, 'base_lrs': self.base_lrs, 'presentations': self.presentations}

    def load_state_dict(self, value: dict[str, object]) -> None:
        expected = self.state_dict()
        if any(value.get(key) != expected[key] for key in ('total_presentations', 'minimum_ratio', 'base_lrs')):
            raise ValueError('decision schedule identity changed')
        count = value.get('presentations')
        if type(count) is not int:
            raise ValueError('invalid decision clock')
        self.step(count)
