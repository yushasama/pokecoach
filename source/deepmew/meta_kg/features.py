"""Deck card weights for weighted similarity.

``weight(card) = idf(card) * non_staple_multiplier(card) * role_multiplier(card)``

IDF rewards rarer cards; the non-staple multiplier collapses generic staples;
role multipliers boost terminal/core/signature cards so deck identity follows
real cores, not shared consistency pieces.
"""

from __future__ import annotations

import math
from typing import Any

from deepmew.meta_kg.schemas import (
    AnchorDefinition,
    CardCatalogRow,
    CardId,
    K2ABuildConfig,
)

ROLE_MULTIPLIERS = {
    "core": 2.0,
    "support": 1.4,
    "tech": 1.1,
    "line": 1.3,
    "staple": 0.25,
    "unknown": 1.0,
}


def build_card_weights(
    catalog: list[CardCatalogRow],
    staple_scores: dict[CardId, float],
    anchors: list[AnchorDefinition],
    lift_rows: list[dict[str, Any]] | None,
    config: K2ABuildConfig,
) -> dict[CardId, float]:
    unique_deck_count = max(1, max((row.deck_count for row in catalog), default=1))
    n = sum(1 for _ in catalog)
    total_decks = max(1, _max_deck_universe(catalog))

    terminal_ids = {cid for anchor in anchors for cid in anchor.terminal_card_ids}
    line_ids = {cid for anchor in anchors for cid in anchor.line_card_ids}

    best_role: dict[CardId, str] = {}
    if lift_rows:
        priority = {"core": 4, "support": 3, "tech": 2, "unknown": 1, "staple": 0}
        for row in lift_rows:
            cid = row["card_id"]
            role = row["role_guess"]
            if priority.get(role, 0) > priority.get(best_role.get(cid, "staple"), 0):
                best_role[cid] = role

    weights: dict[CardId, float] = {}
    for row in catalog:
        cid = row.card_id
        idf = math.log((total_decks + 1) / (row.deck_count + 1)) + 1.0
        staple = staple_scores.get(cid, 0.0)
        non_staple = max(0.0, 1.0 - min(1.0, staple))
        if cid in terminal_ids:
            role_mult = ROLE_MULTIPLIERS["core"]
        elif cid in line_ids:
            role_mult = ROLE_MULTIPLIERS["line"]
        else:
            role_mult = ROLE_MULTIPLIERS.get(best_role.get(cid, "unknown"), 1.0)
        # Staples keep a small floor so they are not fully ignored, just downweighted.
        weight = idf * max(non_staple, 0.1) * role_mult
        weights[cid] = round(weight, 6)
    return weights


def _max_deck_universe(catalog: list[CardCatalogRow]) -> int:
    # global_deck_freq = deck_count / unique_deck_count -> recover universe size.
    for row in catalog:
        if row.global_deck_freq > 0:
            return round(row.deck_count / row.global_deck_freq)
    return len(catalog)
