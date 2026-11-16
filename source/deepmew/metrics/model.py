from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PolicyMetrics:
    decisions: int = 0
    end_with_useful_options: int = 0
    attack_available_but_not_picked: int = 0
    unknown_action: int = 0
    risky_over_safer: int = 0

    def add(self, other: "PolicyMetrics") -> None:
        self.decisions += other.decisions
        self.end_with_useful_options += other.end_with_useful_options
        self.attack_available_but_not_picked += other.attack_available_but_not_picked
        self.unknown_action += other.unknown_action
        self.risky_over_safer += other.risky_over_safer
