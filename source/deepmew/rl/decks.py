"""Deck resolution helpers for RL experiment configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from deepmew.deck.io import read_deck


ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path: str | Path) -> Path:
    deck_path = Path(path)
    if deck_path.is_absolute():
        return deck_path
    return ROOT / deck_path


def read_config_deck(config: Any, *, role: str = "agent") -> list[int]:
    attr = "agent_deck_path" if role == "agent" else "opponent_deck_path"
    return read_deck(resolve_repo_path(getattr(config, attr, "deck.csv")))
