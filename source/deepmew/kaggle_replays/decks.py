"""Extract public opponent decklists from a Kaggle episode's visualize payload.

The full ordered decklists are exposed in the initial `visualize` payload of a
downloaded replay (post-hoc the renderer can see both decks). This searches the
documented candidate locations, never fabricates a deck, and writes
DeepMew-compatible deck artifacts (a 60-line `deck.csv`, a human-readable
`deck_summary.json`, and a `deck_manifest.json`).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepmew.artifacts.manifest import build_manifest, write_manifest
from deepmew.artifacts.run_dir import safe_slug
from deepmew.artifacts.writers import write_json, write_text
from deepmew.kaggle_replays.episode import team_names as episode_team_names
from deepmew.options.identity import card_name_for_id
from deepmew.replay.decision_ingest import deck_metadata_from_path

OPPONENT_DECK_KIND = "opponent_deck"
STANDARD_DECK_SIZE = 60
CHECKED_VISUALIZE_PATHS = (
    "steps[0][*].visualize[0].action",
    "steps[0][*].action.visualize",
    "steps[0][*].observation.visualize",
    "steps[0][*].observation.action.visualize",
    "top-level visualize",
)


class DeckExtractionError(ValueError):
    """Raised when no decklists can be located in an episode."""


@dataclass(frozen=True)
class PlayerDeck:
    player_index: int
    team_name: str
    card_ids: list[int]
    card_count: int
    card_counts: dict[int, int]
    name_counts: dict[str, int]
    composition: list[dict[str, Any]]
    unknown_card_ids: list[int]

    @property
    def is_standard_size(self) -> bool:
        return self.card_count == STANDARD_DECK_SIZE


def _as_int_list(value: Any) -> list[int] | None:
    if not isinstance(value, list) or not value:
        return None
    out: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            return None
        out.append(item)
    return out


def _looks_like_player_entry(entry: Any) -> bool:
    return isinstance(entry, dict) and _as_int_list(entry.get("deck")) is not None


def _normalize_action_decklists(payload: Any, *, names: list[str]) -> list[dict[str, Any]] | None:
    """Normalize Kaggle's step-level visualize action into player deck entries."""
    if not isinstance(payload, list) or not payload:
        return None

    players: list[dict[str, Any]] = []
    for index, decklist in enumerate(payload):
        card_ids = _as_int_list(decklist)
        if card_ids is None or len(card_ids) != STANDARD_DECK_SIZE:
            return None
        team_name = names[index] if index < len(names) else f"player_{index}"
        players.append({"index": index, "name": team_name, "deck": card_ids})
    return players


def _normalize_players(payload: Any, *, names: list[str]) -> list[dict[str, Any]] | None:
    """Coerce a visualize payload into an ordered list of player dicts with decks."""
    action_players = _normalize_action_decklists(payload, names=names)
    if action_players is not None:
        return action_players

    if isinstance(payload, dict):
        players = payload.get("players")
        if isinstance(players, list) and any(_looks_like_player_entry(p) for p in players):
            return players
        # dict keyed by player index, e.g. {"0": {...}, "1": {...}}
        if all(_looks_like_player_entry(v) for v in payload.values()) and payload:
            return [payload[key] for key in sorted(payload, key=lambda k: str(k))]
    if isinstance(payload, list) and any(_looks_like_player_entry(p) for p in payload):
        return payload
    return None


def _candidate_payloads(data: dict[str, Any]) -> list[Any]:
    candidates: list[Any] = []
    steps = data.get("steps")
    if isinstance(steps, list) and steps and isinstance(steps[0], list):
        for agent in steps[0]:
            if isinstance(agent, dict):
                visualize = agent.get("visualize")
                if isinstance(visualize, list) and visualize and isinstance(visualize[0], dict):
                    candidates.append(visualize[0].get("action"))
                action = agent.get("action")
                observation = agent.get("observation")
                if isinstance(action, dict):
                    candidates.append(action.get("visualize"))
                if isinstance(observation, dict):
                    candidates.append(observation.get("visualize"))
                    candidates.append(_get_nested(observation, "action", "visualize"))
    candidates.append(data.get("visualize"))
    return candidates


def _get_nested(value: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def find_visualize_players(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the ordered per-player visualize entries, or raise if none are found."""
    names = episode_team_names(data)
    for payload in _candidate_payloads(data):
        players = _normalize_players(payload, names=names)
        if players is not None:
            return players
    raise DeckExtractionError(
        "no visualize decklists found in episode (checked " + ", ".join(CHECKED_VISUALIZE_PATHS) + ")"
    )


def _player_deck(entry: dict[str, Any], *, index: int) -> PlayerDeck:
    card_ids = _as_int_list(entry.get("deck")) or []
    team = entry.get("name") or entry.get("team") or entry.get("teamName") or f"player_{index}"
    player_index = entry.get("index")
    player_index = int(player_index) if isinstance(player_index, int) and not isinstance(player_index, bool) else index

    counts = Counter(card_ids)
    name_counts: Counter = Counter()
    composition: list[dict[str, Any]] = []
    unknown: list[int] = []
    for card_id, count in sorted(counts.items()):
        name = card_name_for_id(card_id)
        if name is None:
            unknown.append(card_id)
            name = f"UNKNOWN_{card_id}"
        name_counts[name] += count
        composition.append({"card_id": card_id, "card_name": name, "count": count})

    return PlayerDeck(
        player_index=player_index,
        team_name=str(team),
        card_ids=card_ids,
        card_count=len(card_ids),
        card_counts=dict(counts),
        name_counts=dict(name_counts),
        composition=composition,
        unknown_card_ids=unknown,
    )


def extract_decklists(data: dict[str, Any]) -> list[PlayerDeck]:
    players = find_visualize_players(data)
    decks = [_player_deck(entry, index=i) for i, entry in enumerate(players)]
    return sorted(decks, key=lambda d: d.player_index)


def deck_dir_name(deck: PlayerDeck) -> str:
    return f"{deck.player_index}-{safe_slug(deck.team_name, fallback='player')}"


def write_deck_artifacts(
    deck: PlayerDeck,
    *,
    episode_id: str,
    source_episode_path: str,
    out_root: str | Path = ".research/opponent_decks",
    command: str | None = None,
) -> dict[str, Any]:
    """Write deck.csv + deck_summary.json + deck_manifest.json for one player deck."""
    out_dir = Path(out_root) / episode_id / deck_dir_name(deck)
    out_dir.mkdir(parents=True, exist_ok=True)

    deck_csv_path = out_dir / "deck.csv"
    write_text(deck_csv_path, "\n".join(str(card_id) for card_id in deck.card_ids) + "\n" if deck.card_ids else "")

    _stem_id, deck_hash = deck_metadata_from_path(str(deck_csv_path))
    deck_id = f"opp-{episode_id}-p{deck.player_index}"
    warnings = [] if deck.is_standard_size else [f"deck has {deck.card_count} cards, expected {STANDARD_DECK_SIZE}"]
    if deck.unknown_card_ids:
        warnings.append(f"unknown card ids: {deck.unknown_card_ids}")

    summary = {
        "episode_id": episode_id,
        "player_index": deck.player_index,
        "team_name": deck.team_name,
        "deck_id": deck_id,
        "deck_hash": deck_hash,
        "card_count": deck.card_count,
        "is_standard_size": deck.is_standard_size,
        "card_ids": deck.card_ids,
        "card_counts": {str(card_id): count for card_id, count in sorted(deck.card_counts.items())},
        "name_counts": dict(sorted(deck.name_counts.items())),
        "composition": deck.composition,
        "unknown_card_ids": deck.unknown_card_ids,
        "warnings": warnings,
        "source_episode_path": source_episode_path,
    }
    summary_path = write_json(out_dir / "deck_summary.json", summary)

    manifest = build_manifest(
        run_id=f"{episode_id}-{deck_dir_name(deck)}",
        kind=OPPONENT_DECK_KIND,
        command=command or "",
        inputs={
            "episode_id": episode_id,
            "player_index": deck.player_index,
            "team_name": deck.team_name,
            "source_episode_path": source_episode_path,
        },
        outputs={"deck_csv": str(deck_csv_path), "deck_summary": str(summary_path)},
        summary={
            "deck_hash": deck_hash,
            "deck_id": deck_id,
            "card_count": deck.card_count,
            "is_standard_size": deck.is_standard_size,
            "name_counts": dict(sorted(deck.name_counts.items())),
            "warnings": warnings,
        },
    )
    manifest_path = write_manifest(out_dir / "deck_manifest.json", manifest)

    return {
        "player_index": deck.player_index,
        "team_name": deck.team_name,
        "deck_id": deck_id,
        "deck_hash": deck_hash,
        "card_count": deck.card_count,
        "is_standard_size": deck.is_standard_size,
        "out_dir": str(out_dir),
        "deck_csv": str(deck_csv_path),
        "deck_summary": str(summary_path),
        "deck_manifest": str(manifest_path),
        "warnings": warnings,
    }
