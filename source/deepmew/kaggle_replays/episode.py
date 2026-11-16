"""Parse downloaded Kaggle simulation-episode JSON for the PTCG environment.

Kaggle-environments episode JSONs are tolerant in shape across versions, so this
reads metadata defensively (multiple candidate locations) and never fabricates
missing values - absent fields come back as None and are recorded as such.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class KaggleEpisodeError(ValueError):
    """Raised when an episode JSON cannot be parsed."""


@dataclass(frozen=True)
class EpisodeMetadata:
    episode_id: str
    team_names: list[str]
    rewards: list[Any] | None
    statuses: list[Any] | None
    seed: int | None
    step_count: int
    agent_count: int


def load_episode(path: str | Path) -> dict[str, Any]:
    episode_path = Path(path)
    if not episode_path.is_file():
        raise KaggleEpisodeError(f"episode JSON not found: {episode_path}")
    try:
        data = json.loads(episode_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise KaggleEpisodeError(f"{episode_path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise KaggleEpisodeError(f"{episode_path}: episode root must be a JSON object")
    return data


def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, dict) else default


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def episode_id(data: dict[str, Any], *, path: str | Path) -> str:
    for key in ("id", "name", "episodeId", "EpisodeId"):
        value = data.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return Path(path).stem


def _seed(data: dict[str, Any]) -> int | None:
    config = _get(data, "configuration", {})
    for source in (config, _get(data, "info", {}), data):
        seed = _int_or_none(_get(source, "seed"))
        if seed is not None:
            return seed
    return None


def team_names(data: dict[str, Any], *, fallback: list[str] | None = None) -> list[str]:
    """Best-effort team/agent names from the usual kaggle-environments locations."""
    info = _get(data, "info", {})
    config = _get(data, "configuration", {})
    for candidate in (
        _get(info, "TeamNames"),
        _get(info, "teamNames"),
        _get(info, "team_names"),
        _get(config, "agents"),
        _get(data, "teamNames"),
    ):
        if isinstance(candidate, list) and candidate and all(isinstance(x, str) for x in candidate):
            return list(candidate)
    if fallback:
        return list(fallback)
    agent_count = _agent_count(data)
    return [f"player_{i}" for i in range(agent_count)]


def _agent_count(data: dict[str, Any]) -> int:
    rewards = data.get("rewards")
    if isinstance(rewards, list) and rewards:
        return len(rewards)
    steps = data.get("steps")
    if isinstance(steps, list) and steps and isinstance(steps[0], list):
        return len(steps[0])
    return 2


def statuses(data: dict[str, Any]) -> list[Any] | None:
    top = data.get("statuses")
    if isinstance(top, list):
        return top
    steps = data.get("steps")
    if isinstance(steps, list) and steps and isinstance(steps[-1], list):
        last = [_get(agent, "status") for agent in steps[-1]]
        if any(status is not None for status in last):
            return last
    return None


def episode_metadata(data: dict[str, Any], *, path: str | Path, name_fallback: list[str] | None = None) -> EpisodeMetadata:
    steps = data.get("steps")
    step_count = len(steps) if isinstance(steps, list) else 0
    rewards = data.get("rewards") if isinstance(data.get("rewards"), list) else None
    return EpisodeMetadata(
        episode_id=episode_id(data, path=path),
        team_names=team_names(data, fallback=name_fallback),
        rewards=rewards,
        statuses=statuses(data),
        seed=_seed(data),
        step_count=step_count,
        agent_count=_agent_count(data),
    )


def result_from_rewards(rewards: list[Any] | None) -> int | None:
    """Winning seat index (0/1), 2 for a draw, or None if rewards are unavailable."""
    if not isinstance(rewards, list) or len(rewards) < 2:
        return None
    numeric = [r for r in rewards if isinstance(r, (int, float)) and not isinstance(r, bool)]
    if len(numeric) != len(rewards):
        return None
    best = max(rewards)
    winners = [i for i, r in enumerate(rewards) if r == best]
    if len(winners) == 1:
        return winners[0]
    return 2
