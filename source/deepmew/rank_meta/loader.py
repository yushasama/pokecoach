"""Load scout leaderboard ranks and join them to mined deck artifacts (K1.7).

The scout layer stored a ``selected_teams`` table (rank / team_id / team_name /
score / source_range) in each ``.research/kaggle_scout/**/manifest.json``. Deck
summaries carry ``team_name`` but no rank, so this module materializes the
``team_name -> rank`` join that was never persisted.

Conflict policy (deterministic, safe): merge all manifests by team_name. If every
entry for a name agrees on rank, assign it; if entries disagree on rank, mark the
name AMBIGUOUS and assign NO rank (even if the team_id matches -- a changed rank is
not disambiguated by identity). team_id/score/source_range are taken from the entry
with the lowest (best) rank for stability. Offline/research-only.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from deepmew.rank_meta.schemas import RankInfo, TeamRank

DEFAULT_SCOUT_ROOT = ".research/kaggle_scout"
DEFAULT_OPPONENT_DECKS = ".research/opponent_decks"
SCOUT_MANIFEST_GLOB = "**/manifest.json"
DECK_SUMMARY_GLOB = "**/deck_summary.json"


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _iter_scout_team_entries(scout_root: str | Path) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (manifest_path, team_entry) for every scout-selected team."""
    root = Path(scout_root)
    if not root.exists():
        return
    for manifest_path in sorted(root.glob(SCOUT_MANIFEST_GLOB)):
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        for entry in data.get("selected_teams") or []:
            if isinstance(entry, dict) and entry.get("team_name") is not None:
                yield str(manifest_path), entry


def load_team_rank_map(
    scout_root: str | Path = DEFAULT_SCOUT_ROOT,
) -> tuple[dict[str, TeamRank], list[str]]:
    """Return (team_name -> TeamRank, warnings). Ambiguous names get rank=None."""
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for manifest_path, entry in _iter_scout_team_entries(scout_root):
        grouped[str(entry["team_name"])].append((manifest_path, entry))

    team_rank: dict[str, TeamRank] = {}
    warnings: list[str] = []
    for team_name in sorted(grouped):
        entries = grouped[team_name]
        ranks = {_as_int(e.get("rank")) for _, e in entries}
        ranks.discard(None)
        manifests = sorted({mp for mp, _ in entries})
        if len(ranks) > 1:
            warnings.append(f"ambiguous rank for team_name {team_name!r}: ranks={sorted(ranks)}")
            team_rank[team_name] = TeamRank(
                team_name=team_name, rank=None, team_id=None, score=None,
                source_range=None, scout_source_manifest=";".join(manifests), ambiguous=True,
            )
            continue
        # Deterministic: pick the entry with the best (lowest) rank for metadata.
        best_mp, best = min(entries, key=lambda me: (_as_int(me[1].get("rank")) if _as_int(me[1].get("rank")) is not None else 1 << 30))
        team_rank[team_name] = TeamRank(
            team_name=team_name,
            rank=_as_int(best.get("rank")),
            team_id=str(best["team_id"]) if best.get("team_id") is not None else None,
            score=str(best["score"]) if best.get("score") is not None else None,
            source_range=str(best["source_range"]) if best.get("source_range") is not None else None,
            scout_source_manifest=";".join(manifests),
        )
    return team_rank, warnings


def rank_info_for(team_rank_map: dict[str, TeamRank], team_name: str | None) -> RankInfo:
    """RankInfo for a team name, defaulting to missing when unknown."""
    if not team_name:
        return RankInfo()
    return RankInfo.from_team_rank(team_rank_map.get(team_name))


def iter_deck_summaries(opponent_decks_root: str | Path = DEFAULT_OPPONENT_DECKS) -> Iterator[dict[str, Any]]:
    """Stream minimal deck-summary records (episode/seat/team/hash/path)."""
    root = Path(opponent_decks_root)
    if not root.exists():
        return
    for path in sorted(root.glob(DECK_SUMMARY_GLOB)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        yield {
            "episode_id": str(data.get("episode_id") or ""),
            "deck_hash": str(data.get("deck_hash") or ""),
            "team_name": data.get("team_name"),
            "player_index": _as_int(data.get("player_index")),
            "source_path": str(path),
        }


def load_episode_seat_rank(
    team_rank_map: dict[str, TeamRank],
    opponent_decks_root: str | Path = DEFAULT_OPPONENT_DECKS,
) -> dict[tuple[str, int], RankInfo]:
    """(episode_id, player_index) -> RankInfo, for the K4.1 pilot/opponent join."""
    mapping: dict[tuple[str, int], RankInfo] = {}
    for record in iter_deck_summaries(opponent_decks_root):
        seat = record["player_index"]
        episode_id = record["episode_id"]
        if seat is None or not episode_id:
            continue
        mapping[(episode_id, seat)] = rank_info_for(team_rank_map, record["team_name"])
    return mapping
