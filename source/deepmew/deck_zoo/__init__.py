"""K3 Deck Zoo: curated real-meta opponent decks/sets seeded from K2A outputs.

Offline/training-only. Reads a K2A meta-kg run and emits machine-readable
representative decks, near-duplicate clusters, train/eval/holdout splits, and
named arena opponent sets for K2B (Real-meta Arena) and K4 (imitation).
"""

from __future__ import annotations

from deepmew.deck_zoo.build import DeckZooResult, build_deck_zoo
from deepmew.deck_zoo.schemas import ARTIFACT_VERSION, DeckZooConfig, DeckZooEntry

__all__ = ["ARTIFACT_VERSION", "DeckZooConfig", "DeckZooEntry", "DeckZooResult", "build_deck_zoo"]
