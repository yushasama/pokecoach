from __future__ import annotations

from typing import Any, Iterable

from deepmew.arena.models import MatrixSummary, PairSummary
from deepmew.arena.runner import run_pair
from deepmew.arena.specs import parse_agent_spec
from deepmew.trace.record import TraceRecorder


def _empty_stats() -> dict[str, Any]:
    return {
        "games": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "crashes": 0,
        "timeouts": 0,
        "turns": 0.0,
        "points": 0.0,
    }


def build_matrix(pairs: Iterable[PairSummary], agents: list[str]) -> dict[str, dict[str, Any]]:
    matrix = {agent: {opponent: _empty_stats() for opponent in agents} for agent in agents}
    for pair in pairs:
        cell_ab = matrix[pair.agent_a][pair.agent_b]
        cell_ba = matrix[pair.agent_b][pair.agent_a]
        for cell in (cell_ab, cell_ba):
            cell["games"] += pair.completed
            cell["draws"] += pair.draws
            cell["crashes"] += pair.crashes
            cell["timeouts"] += pair.timeouts
            cell["turns"] += pair.avg_turns * pair.completed
        cell_ab["wins"] += pair.a_wins
        cell_ab["losses"] += pair.b_wins
        cell_ab["points"] += pair.a_wins + pair.draws * 0.5
        cell_ba["wins"] += pair.b_wins
        cell_ba["losses"] += pair.a_wins
        cell_ba["points"] += pair.b_wins + pair.draws * 0.5

    for row in matrix.values():
        for cell in row.values():
            games = cell["games"]
            cell["win_rate"] = cell["wins"] / games if games else None
            cell["point_rate"] = cell["points"] / games if games else None
            cell["avg_turns"] = cell["turns"] / games if games else None
            del cell["turns"]
    return matrix


def rank_agents(matrix: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rankings = []
    for agent, row in matrix.items():
        games = sum(cell["games"] for cell in row.values())
        wins = sum(cell["wins"] for cell in row.values())
        losses = sum(cell["losses"] for cell in row.values())
        draws = sum(cell["draws"] for cell in row.values())
        crashes = sum(cell["crashes"] for cell in row.values())
        timeouts = sum(cell["timeouts"] for cell in row.values())
        points = sum(cell["points"] for cell in row.values())
        rankings.append(
            {
                "agent": agent,
                "games": games,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "point_rate": points / games if games else 0.0,
                "crashes": crashes,
                "timeouts": timeouts,
            }
        )
    return sorted(rankings, key=lambda item: (-item["point_rate"], item["crashes"] + item["timeouts"], item["agent"]))


def run_matrix(
    agents: list[str],
    *,
    games_per_pair: int,
    max_steps: int = 5000,
    seed: int | None = None,
    include_self: bool = False,
    trace_recorder: TraceRecorder | None = None,
) -> MatrixSummary:
    specs = [parse_agent_spec(agent) for agent in agents]
    labels = [spec.label for spec in specs]
    pairs: list[PairSummary] = []
    for left_index, left in enumerate(specs):
        for right_index, right in enumerate(specs):
            if right_index < left_index:
                continue
            if left_index == right_index and not include_self:
                continue
            pair_seed = None if seed is None else seed + left_index * 1009 + right_index * 9173
            pairs.append(
                run_pair(
                    left,
                    right,
                    games=games_per_pair,
                    max_steps=max_steps,
                    seed=pair_seed,
                    trace_recorder=trace_recorder,
                    game_id_prefix=f"matrix-{left_index:02d}-{right_index:02d}",
                )
            )
    matrix = build_matrix(pairs, labels)
    rankings = rank_agents(matrix)
    return MatrixSummary(labels, games_per_pair, seed, include_self, pairs, matrix, rankings)
