"""Load mined 60-card decklists (the deck-count source of truth).

Primary source: the K3 Deck Zoo's ``deck_zoo_decklists.jsonl`` — 186 canonical
decks, each exactly 60 cards, as ``card_counts: {card_id: copies}`` plus archetype
label and split. These are full decklists (re-aggregated from mined artifacts),
not partial/representative skeletons, so copy counts are trustworthy.
"""

from __future__ import annotations

import json
from pathlib import Path

from deepmew.deck_profile_inference.schemas import DeckList

DECK_ZOO_GLOB = ".research/runs/deck_zoo/*/*/deck_zoo_decklists.jsonl"


def find_latest_decklists(glob_pattern: str = DECK_ZOO_GLOB) -> Path | None:
    """Return the most recent ``deck_zoo_decklists.jsonl`` (by run timestamp)."""
    matches = sorted(Path().glob(glob_pattern))
    return matches[-1] if matches else None


def load_decklists(path: str | Path | None = None) -> list[DeckList]:
    """Load decklists from ``path`` (or the latest deck_zoo run if ``None``)."""
    resolved = Path(path) if path is not None else find_latest_decklists()
    if resolved is None or not Path(resolved).exists():
        raise FileNotFoundError(
            "deck_zoo_decklists.jsonl not found; pass an explicit path or build the K3 Deck Zoo first."
        )
    decks: list[DeckList] = []
    for line in Path(resolved).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        raw_counts = entry.get("card_counts") or {}
        card_counts = {str(cid): int(count) for cid, count in raw_counts.items()}
        decks.append(
            DeckList(
                deck_hash=str(entry.get("deck_hash", "")),
                archetype=str(entry.get("archetype", "Unknown")),
                split=str(entry.get("split", "")),
                card_counts=card_counts,
                is_canonical=bool(entry.get("is_canonical", True)),
            )
        )
    # Deterministic order by deck_hash.
    decks.sort(key=lambda d: d.deck_hash)
    return decks
