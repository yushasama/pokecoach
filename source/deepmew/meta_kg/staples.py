"""Staple detection: broad, low-lift cards that must not bridge the meta.

Two passes: a preliminary pass from global frequency alone (needed before any
lift exists), then a final pass that also uses max archetype lift and spread.
Staples stay visible in outputs but are downweighted in classification,
similarity, and live belief.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from deepmew.meta_kg.schemas import (
    AnchorDefinition,
    CardCatalogRow,
    CardId,
    K2ABuildConfig,
)


def normalized_entropy(weights: list[float]) -> float:
    positive = [w for w in weights if w > 0]
    if len(positive) <= 1:
        return 0.0
    total = sum(positive)
    probs = [w / total for w in positive]
    h = -sum(p * math.log(p) for p in probs)
    return h / math.log(len(positive))


def staple_score(global_freq: float, max_lift: float, entropy_norm: float, config: K2ABuildConfig) -> float:
    freq_part = min(1.0, global_freq / max(config.staple_min_global_deck_freq, 1e-9))
    lift_part = max(0.0, 1.0 - max(0.0, max_lift - 1.0) / max(config.staple_max_lift - 1.0, 1e-9))
    entropy_part = min(1.0, entropy_norm / max(config.staple_min_entropy_norm, 1e-9))
    return max(0.0, min(1.0, 0.45 * freq_part + 0.35 * lift_part + 0.20 * entropy_part))


def downweight_for_score(score: float, config: K2ABuildConfig) -> float:
    """Map a staple score in [0,1] to a similarity/classification multiplier.

    Higher staple score -> smaller multiplier (downweight_min). Non-staples keep
    weight 1.0.
    """
    if score <= 0:
        return 1.0
    span = config.staple_downweight_max - config.staple_downweight_min
    # multiplier shrinks from downweight_max (mild staple) to downweight_min (strong staple)
    return max(config.staple_downweight_min, config.staple_downweight_max - span * score)


def detect_preliminary_staples(
    catalog: list[CardCatalogRow],
    config: K2ABuildConfig,
    *,
    terminal_card_ids: set[CardId] | None = None,
) -> dict[CardId, float]:
    """Frequency-only preliminary staple scores (lift not yet available)."""
    terminal_card_ids = terminal_card_ids or set()
    scores: dict[CardId, float] = {}
    for row in catalog:
        if config.staple_terminal_exclusion and row.card_id in terminal_card_ids:
            continue
        if row.global_deck_freq < config.staple_min_global_deck_freq:
            continue
        # Without lift, assume worst-case (broad) and use frequency strength.
        scores[row.card_id] = staple_score(row.global_deck_freq, 1.0, 1.0, config)
    return scores


def detect_final_staples(
    catalog: list[CardCatalogRow],
    lift_rows: list[dict[str, Any]],
    anchors: list[AnchorDefinition],
    config: K2ABuildConfig,
) -> list[dict[str, Any]]:
    """Final staple rows using global frequency, max archetype lift, and spread."""
    terminal_ids = {cid for anchor in anchors for cid in anchor.terminal_card_ids}
    freq_by_id = {row.card_id: row.global_deck_freq for row in catalog}
    artifact_freq_by_id = {row.card_id: row.global_artifact_freq for row in catalog}
    name_by_id = {row.card_id: row.card_name for row in catalog}

    lift_by_card: dict[CardId, list[tuple[str, float]]] = defaultdict(list)
    for row in lift_rows:
        lift_by_card[row["card_id"]].append((row["archetype"], float(row["lift"])))

    rows: list[dict[str, Any]] = []
    for card_id, global_freq in freq_by_id.items():
        if config.staple_terminal_exclusion and card_id in terminal_ids:
            continue
        if global_freq < config.staple_min_global_deck_freq:
            continue
        entries = lift_by_card.get(card_id, [])
        max_lift = max((lift for _, lift in entries), default=1.0)
        entropy_norm = normalized_entropy([lift for _, lift in entries]) if len(entries) > 1 else 0.0
        if max_lift > config.staple_max_lift:
            continue  # archetype-specific enough to not be a staple
        score = staple_score(global_freq, max_lift, entropy_norm, config)
        rows.append(
            {
                "card_id": card_id,
                "card_name": name_by_id.get(card_id, f"Unknown Card {card_id}"),
                "global_deck_freq": round(global_freq, 6),
                "global_artifact_freq": round(artifact_freq_by_id.get(card_id, 0.0), 6),
                "max_archetype_lift": round(max_lift, 4),
                "archetype_entropy": round(entropy_norm, 4),
                "staple_score": round(score, 4),
                "downweight": round(downweight_for_score(score, config), 4),
                "present_in_archetypes": sorted({archetype for archetype, _ in entries}),
                "notes": "broad low-lift card; downweighted for identity",
            }
        )
    rows.sort(key=lambda r: (-r["staple_score"], -r["global_deck_freq"], r["card_id"]))
    return rows
