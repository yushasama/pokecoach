"""Load handoff-derived hints used ONLY to prioritise manual audit order.

These hints are *not* training labels and are *not* truth about a card's
function. They answer one question: "which cards should a human look at first?"
Top-meta / load-bearing / high-frequency cards and known counter pieces are
audited before obscure cards. Everything degrades gracefully: if the handoff
bundle is absent, hints are empty and priority still falls back to the
text-derived semantic risk classes.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from deepmew.card_functions.schemas import PriorityHints, _norm_name

DEFAULT_HANDOFF_DIR = ".research/handoff_bundles/deepmew_card_meta_handoff_20260623"
_TOP_META_CSV = "data/top_meta_decklists_card_functions.csv"
_COUNTER_MATRIX_CSV = "references/deepmew_meta_counter_matrix.csv"


def _split_names(blob: str) -> list[str]:
    """Split a comma/slash-separated descriptive blob into candidate names."""
    parts: list[str] = []
    for chunk in str(blob).replace("/", ",").split(","):
        text = chunk.strip()
        if text:
            parts.append(text)
    return parts


def load_priority_hints(handoff_dir: str | Path | None = DEFAULT_HANDOFF_DIR) -> PriorityHints:
    """Return :class:`PriorityHints` from the handoff bundle (empty if missing)."""
    if handoff_dir is None:
        return PriorityHints()
    base = Path(handoff_dir)
    top_meta: set[str] = set()
    deck_appearances: dict[str, set[str]] = defaultdict(set)
    load_bearing: set[str] = set()

    top_path = base / _TOP_META_CSV
    if top_path.exists():
        with top_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                name = _norm_name(row.get("card_name", ""))
                if not name:
                    continue
                top_meta.add(name)
                deck_hash = row.get("deck_hash") or ""
                deck_appearances[name].add(deck_hash)

    matrix_path = base / _COUNTER_MATRIX_CSV
    if matrix_path.exists():
        with matrix_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                for column in ("load_bearing_cards", "counter_primitives"):
                    for candidate in _split_names(row.get(column, "")):
                        load_bearing.add(_norm_name(candidate))

    deck_frequency = {name: len(hashes) for name, hashes in deck_appearances.items()}
    return PriorityHints(
        load_bearing=frozenset(load_bearing),
        top_meta=frozenset(top_meta),
        deck_frequency=deck_frequency,
    )
