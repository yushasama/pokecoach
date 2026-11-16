from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deepmew.agents.specs import AblationConfig, AgentSpec


@dataclass
class GameRecord:
    game_index: int
    agent_a: str
    agent_b: str
    seat_a: int
    result: int | None
    winner: str | None
    draws: int
    turns: int
    crashed: bool
    timed_out: bool
    elapsed_seconds: float
    error: str | None = None


@dataclass
class PairSummary:
    agent_a: str
    agent_b: str
    games: int
    completed: int
    a_wins: int
    b_wins: int
    draws: int
    crashes: int
    timeouts: int
    avg_turns: float
    elapsed_seconds: float
    records: list[GameRecord]
    metrics: dict[str, dict[str, int]]


@dataclass
class MatrixSummary:
    agents: list[str]
    games_per_pair: int
    seed: int | None
    include_self: bool
    pairs: list[PairSummary]
    matrix: dict[str, dict[str, Any]]
    rankings: list[dict[str, Any]]
    output_json: str | None = None
    output_report: str | None = None
