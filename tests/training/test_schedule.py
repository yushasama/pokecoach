from __future__ import annotations

import pytest
import torch

from deepmew_ex.training.schedule import DecisionSchedule


def test_schedule_uses_decisions_and_restores_exact_rate() -> None:
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=.001)
    schedule = DecisionSchedule(optimizer, 100)
    schedule.step(60)
    expected = optimizer.param_groups[0]['lr']
    other = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=.001)
    restored = DecisionSchedule(other, 100)
    restored.load_state_dict(schedule.state_dict())
    assert other.param_groups[0]['lr'] == expected
    with pytest.raises(ValueError, match='backwards'):
        restored.step(59)
    with pytest.raises(ValueError, match='identity changed'):
        DecisionSchedule(other, 200).load_state_dict(schedule.state_dict())
    restored.step(100)
    assert other.param_groups[0]['lr'] == pytest.approx(.0001)
