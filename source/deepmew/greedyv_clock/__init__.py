"""Whole-game clock budgeting for the GreedyV search agent (BUG-003 fix).

INVARIANT #4: the clock is a single whole-game bank (600 s overage), NOT a
per-move budget. Every search-cost parameter must be a function of the remaining
bank. This package wraps a GreedyV-style agent and, before each decision, sizes
the per-decision search budget (and clamps world count) from the observation's
``remainingOverageTime`` so a game can never overrun (an overrun is an instant
loss, INVARIANT #1).
"""

from .wrapper import ClockBudgetConfig, ClockBudgetedAgent

__all__ = ["ClockBudgetConfig", "ClockBudgetedAgent"]
