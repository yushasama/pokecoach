from __future__ import annotations

from collections import Counter
from pathlib import Path

from deepmew.deck import read_deck


_DECK_PRIOR: Counter[int] | None = None


def reset_deck_prior() -> None:
    global _DECK_PRIOR
    _DECK_PRIOR = None


def load_deck_prior() -> Counter[int]:
    global _DECK_PRIOR
    if _DECK_PRIOR is None:
        for path in (Path("deck.csv"), Path("/kaggle_simulations/agent/deck.csv")):
            if path.exists():
                _DECK_PRIOR = Counter(read_deck(path))
                break
        else:
            _DECK_PRIOR = Counter()
    return _DECK_PRIOR
