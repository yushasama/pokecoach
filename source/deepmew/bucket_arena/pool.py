"""Per-ELO-bucket opponent deck pools mined from R16 bucket replay sections.

Builds, from a directory of `episode-*-replay.json` files (one R16 bucket
section), the deduplicated deck pool actually played in that band: exact
60-card lists with game counts, win counts, and the teams that piloted them.
Offline/eval-only; never imported by the submission path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from deepmew.kaggle_replays.decks import DeckExtractionError, extract_decklists
from deepmew.rl.metadata import normalized_deck_hash


@dataclass
class DeckPoolEntry:
    deck_hash: str
    card_ids: list[int]
    games: int = 0
    wins: int = 0
    teams: set[str] = field(default_factory=set)

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "deck_hash": self.deck_hash,
            "card_ids": self.card_ids,
            "games": self.games,
            "wins": self.wins,
            "win_rate": round(self.win_rate, 4),
            "teams": sorted(self.teams),
        }


def entry_from_json(row: dict[str, Any]) -> DeckPoolEntry:
    return DeckPoolEntry(
        deck_hash=str(row["deck_hash"]),
        card_ids=[int(c) for c in row["card_ids"]],
        games=int(row.get("games", 0)),
        wins=int(row.get("wins", 0)),
        teams=set(row.get("teams") or []),
    )


def build_bucket_pool(replay_dir: Path, *, exclude_teams: set[str] | None = None) -> list[DeckPoolEntry]:
    """Scan a bucket replay section and aggregate the decks played in it."""
    exclude = exclude_teams or {"DeepMew"}
    pool: dict[str, DeckPoolEntry] = {}
    for path in sorted(Path(replay_dir).glob("episode-*-replay.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        info = data.get("info") or {}
        team_names = [str(name) for name in (info.get("TeamNames") or [])]
        rewards = data.get("rewards") or []
        try:
            decks = extract_decklists(data)
        except DeckExtractionError:
            continue
        if len(team_names) != 2 or len(rewards) != 2 or len(decks) != 2:
            continue
        for seat, team in enumerate(team_names):
            if team in exclude:
                continue
            card_ids = sorted(int(c) for c in decks[seat].card_ids)
            if len(card_ids) != 60:
                continue
            deck_hash = normalized_deck_hash(card_ids)
            entry = pool.get(deck_hash)
            if entry is None:
                entry = pool[deck_hash] = DeckPoolEntry(deck_hash=deck_hash, card_ids=card_ids)
            entry.games += 1
            reward = rewards[seat]
            if isinstance(reward, (int, float)) and reward > 0:
                entry.wins += 1
            entry.teams.add(team)
    return sorted(pool.values(), key=lambda e: (-e.games, e.deck_hash))
