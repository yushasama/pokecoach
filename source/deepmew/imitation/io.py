"""Loaders for the K4 imitation dataset: action rows + K2A/K3 label joins."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

DEFAULT_INGEST_ROOT = ".research/kaggle_ingest"
DEFAULT_OPPONENT_DECKS = ".research/opponent_decks"
ACTION_TABLE_FILENAME = "replay_action_table.jsonl"


def load_label_map(meta_kg_dir: str | Path) -> dict[str, dict[str, Any]]:
    """deck_hash -> {archetype, variant_label, confidence, classification_status}."""
    path = Path(meta_kg_dir) / "deck_archetypes.csv"
    labels: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            labels[row["deck_hash"]] = {
                "archetype": row["primary_archetype"],
                "variant_label": row["variant_label"],
                "confidence": float(row.get("confidence") or 0.0),
                "classification_status": row["classification_status"],
            }
    return labels


def load_split_map(deck_zoo_dir: str | Path) -> dict[str, str]:
    """deck_hash -> train/eval/holdout split from the K3 deck zoo."""
    path = Path(deck_zoo_dir) / "deck_zoo.csv"
    splits: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            splits[row["deck_hash"]] = row.get("split", "train")
    return splits


def load_episode_seat_decks(opponent_decks_root: str | Path) -> dict[str, dict[int, str]]:
    """episode_id -> {seat(player_index): deck_hash} from mined deck summaries."""
    root = Path(opponent_decks_root)
    mapping: dict[str, dict[int, str]] = defaultdict(dict)
    for path in root.rglob("deck_summary.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        episode_id = str(data.get("episode_id") or "")
        deck_hash = data.get("deck_hash")
        player_index = data.get("player_index")
        if episode_id and deck_hash is not None and isinstance(player_index, int):
            mapping[episode_id][player_index] = str(deck_hash)
    return dict(mapping)


def iter_action_rows(ingest_root: str | Path) -> Iterator[dict[str, Any]]:
    """Stream every replay action row across ingested episodes."""
    root = Path(ingest_root)
    for path in sorted(root.rglob(ACTION_TABLE_FILENAME)):
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict):
                        yield row
        except OSError:
            continue
