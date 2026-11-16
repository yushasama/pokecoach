"""Loading mined deck artifacts and writing K2A outputs.

Input: ``deck_summary.json`` files written by ``deepmew.kaggle_replays.decks``
(one per side per episode under ``.research/opponent_decks/...``). Names are read
straight from the baked ``composition`` field so this module never needs the
battle engine. Cards whose name is ``UNKNOWN_<id>`` (or empty) are treated as
name-missing and recovered downstream as ``Unknown Card <card_id>``.
"""

from __future__ import annotations

import csv
import io as _io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml

from deepmew.artifacts.writers import write_text
from deepmew.meta_kg.schemas import (
    CardId,
    DeckArtifact,
    DeckHash,
    K2ABuildConfig,
    UniqueDeck,
)

DEFAULT_INPUT_DIR = ".research/opponent_decks"
DECK_SUMMARY_GLOB = "deck_summary.json"


def load_config(config_path: str | Path | None) -> tuple[K2ABuildConfig, list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (config, raw_anchor_defs, raw_engine_package_defs).

    A missing/None path yields defaults with no anchors, so the pipeline always
    runs. Anchors/engine packages are returned raw (name-based) and resolved to
    card IDs later against the catalog.
    """
    if config_path is None:
        return K2ABuildConfig(), [], []
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"meta_kg config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"meta_kg config root must be a mapping: {path}")
    config = K2ABuildConfig.from_mapping(data)
    anchors = [a for a in (data.get("anchors") or []) if isinstance(a, dict)]
    engine_packages = [e for e in (data.get("engine_packages") or []) if isinstance(e, dict)]
    return config, anchors, engine_packages


def _name_is_missing(name: Any) -> bool:
    if name is None:
        return True
    text = str(name).strip()
    return not text or text.startswith("UNKNOWN_")


def _artifact_from_summary(data: dict[str, Any], source_path: str) -> DeckArtifact | None:
    composition = data.get("composition")
    cards: dict[CardId, int] = {}
    names: dict[CardId, str] = {}
    if isinstance(composition, list) and composition:
        for entry in composition:
            if not isinstance(entry, dict):
                continue
            card_id = entry.get("card_id")
            count = entry.get("count")
            if card_id is None or not isinstance(count, int) or isinstance(count, bool):
                continue
            cid = str(card_id)
            cards[cid] = cards.get(cid, 0) + int(count)
            if not _name_is_missing(entry.get("card_name")):
                names[cid] = str(entry.get("card_name"))
    else:
        card_counts = data.get("card_counts")
        if isinstance(card_counts, dict):
            for card_id, count in card_counts.items():
                if isinstance(count, int) and not isinstance(count, bool):
                    cards[str(card_id)] = int(count)
    if not cards:
        return None

    episode_id = str(data.get("episode_id") or "")
    team_name = data.get("team_name")
    team = str(team_name) if team_name not in (None, "") else None
    player_index = data.get("player_index")
    return DeckArtifact(
        episode_id=episode_id,
        cards=cards,
        names=names,
        deck_hash=str(data["deck_hash"]) if data.get("deck_hash") else None,
        player_id=team,
        team_id=team,
        side=str(player_index) if player_index is not None else None,
        rank=data.get("rank") if isinstance(data.get("rank"), int) else None,
        rank_bucket=str(data["rank_bucket"]) if data.get("rank_bucket") else None,
        source_path=source_path,
    )


def load_deck_artifacts(input_dir: str | Path) -> list[DeckArtifact]:
    """Load all mined deck artifacts from ``input_dir`` (recursive)."""
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"meta_kg input dir not found: {root}")
    artifacts: list[DeckArtifact] = []
    for path in sorted(root.rglob(DECK_SUMMARY_GLOB)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        artifact = _artifact_from_summary(data, str(path))
        if artifact is not None:
            artifacts.append(artifact)
    return artifacts


def stable_deck_hash(cards: dict[CardId, int]) -> DeckHash:
    """Deterministic fallback hash for decks lacking an upstream hash."""
    import hashlib

    parts = [f"{card_id}:{cards[card_id]}" for card_id in sorted(cards)]
    payload = "|".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def aggregate_unique_decks(
    artifacts: list[DeckArtifact],
) -> tuple[dict[DeckHash, UniqueDeck], list[tuple[str, str]]]:
    """Group artifacts by deck hash into representative unique decks.

    Returns (unique_decks, conflicts) where conflicts is a list of
    (deck_hash, detail) when one hash shows more than one card multiset.
    """
    grouped: dict[DeckHash, list[DeckArtifact]] = defaultdict(list)
    for artifact in artifacts:
        deck_hash = artifact.deck_hash or stable_deck_hash(artifact.cards)
        grouped[deck_hash].append(artifact)

    unique: dict[DeckHash, UniqueDeck] = {}
    conflicts: list[tuple[str, str]] = []
    for deck_hash, members in grouped.items():
        multisets = Counter()
        for member in members:
            key = tuple(sorted(member.cards.items()))
            multisets[key] += 1
        if len(multisets) > 1:
            conflicts.append((deck_hash, f"{len(multisets)} distinct card multisets"))
        representative_key = multisets.most_common(1)[0][0]
        cards = {cid: count for cid, count in representative_key}

        names: dict[CardId, str] = {}
        name_votes: dict[CardId, Counter] = defaultdict(Counter)
        episode_ids: list[str] = []
        player_ids: list[str] = []
        team_ids: list[str] = []
        rank_buckets: Counter = Counter()
        for member in members:
            for cid, name in member.names.items():
                name_votes[cid][name] += 1
            if member.episode_id:
                episode_ids.append(member.episode_id)
            if member.player_id:
                player_ids.append(member.player_id)
            if member.team_id:
                team_ids.append(member.team_id)
            if member.rank_bucket:
                rank_buckets[member.rank_bucket] += 1
        for cid in cards:
            if name_votes.get(cid):
                names[cid] = name_votes[cid].most_common(1)[0][0]

        unique[deck_hash] = UniqueDeck(
            deck_hash=deck_hash,
            cards=cards,
            names=names,
            artifact_count=len(members),
            episode_ids=sorted(set(episode_ids)),
            player_ids=sorted(set(player_ids)),
            team_ids=sorted(set(team_ids)),
            rank_buckets=dict(sorted(rank_buckets.items())),
        )
    return unique, conflicts


def json_cell(value: Any) -> str:
    """Compact, stable JSON serialization for list/dict CSV cells."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> Path:
    """Write rows to a CSV file with a stable header and atomic write."""
    handle = _io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fieldnames})
    return write_text(path, handle.getvalue())
