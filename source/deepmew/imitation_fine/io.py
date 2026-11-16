"""Loaders for the K4.1 fine-grained imitation build.

Reuses the K4 action-row / label / split / opponent-deck loaders and adds a
deck_hash -> deck-zoo bucket join (from the K3 opponent sets) plus latest-run
discovery for K2A / K3 / K4 outputs. Offline/training-only.
"""

from __future__ import annotations

import csv
from pathlib import Path

# Reuse the proven K4 loaders verbatim (single source of truth).
from deepmew.imitation.io import (  # noqa: F401
    DEFAULT_INGEST_ROOT,
    DEFAULT_OPPONENT_DECKS,
    iter_action_rows,
    load_episode_seat_decks,
    load_label_map,
    load_split_map,
)

OPPONENT_SETS_FILENAME = "arena_opponent_sets.csv"


def load_bucket_map(deck_zoo_dir: str | Path) -> dict[str, str]:
    """deck_hash -> deck-zoo bucket/set name (first set wins; '' if unlisted)."""
    path = Path(deck_zoo_dir) / OPPONENT_SETS_FILENAME
    buckets: dict[str, str] = {}
    if not path.exists():
        return buckets
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            deck_hash = row.get("deck_hash")
            set_name = row.get("set_name")
            if deck_hash and set_name and deck_hash not in buckets:
                buckets[deck_hash] = set_name
    return buckets


def find_latest_run(runs_root: str | Path, marker: str) -> Path:
    """Latest run dir under runs_root that contains marker."""
    root = Path(runs_root)
    runs = [c for c in sorted(root.glob("*/*")) if (c / marker).exists()]
    if not runs:
        raise FileNotFoundError(f"no run with {marker} under {root}")
    return runs[-1]


def find_latest_meta_kg(runs_root: str | Path = ".research/runs/meta_kg") -> Path:
    return find_latest_run(runs_root, "deck_archetypes.csv")


def find_latest_deck_zoo(runs_root: str | Path = ".research/runs/deck_zoo") -> Path:
    return find_latest_run(runs_root, "deck_zoo.csv")


def find_latest_imitation(runs_root: str | Path = ".research/runs/imitation_dataset") -> Path:
    return find_latest_run(runs_root, "imitation_examples.jsonl")
